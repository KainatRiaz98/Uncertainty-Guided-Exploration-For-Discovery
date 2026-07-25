#!/bin/bash
# Build the denoising venv on a node that is already running other domains.
#
# WHY A SEPARATE VENV AT ALL
# The denoising verifier is imported IN-PROCESS by the trainer
# (tinker_cookbook/rl/mlora_train.py:2426), so its deps must live in a venv the
# training process actually uses — a standalone `uv venv .venv` is never seen.
# But installing them moves numpy/scipy, and the training venv is shared with
# the node's other running domain. Instances reboot on a ~24h cycle and runs
# auto-resume, so an in-place install can break a co-resident domain a day
# later, silently. Hence: clone, install into the clone, launch denoising with
#   PYTHON=<clone>/bin/python.
#
# THE CONSTRAINT THAT ACTUALLY MATTERS: RAY
# Every run attaches to ONE shared Ray head per node (ray.init("auto"), see
# tasks/*/task.py and RAY_ADDRESS=auto in the launchers). Ray refuses to attach
# when the driver's version differs from the cluster's. numpy and scipy may
# differ freely between venvs — they never cross a process boundary — but ray
# MUST be identical in every venv on the node. The pins disagree across
# requirement files (math 2.51.1, ahc 2.53.0, denoising 2.53.0), so this script
# checks the running head and refuses to produce a venv that cannot attach.
#
# NOT A NUMPY-VS-TORCH CONFLICT. torch==2.9.1 in all three requirement files.
# The documented breakage (np.asarray(..., copy=False)) is a NumPy 2.0 API
# removal inside the bio code and happens under every pin here; the fix is the
# source patch below, not version juggling.
#
#   bash scripts/aws/setup_denoising_venv.sh [<clone-path>] [<source-venv>]
#
# Defaults: ~/venv-denoise, cloned from the venv this script is run under.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CLONE="${1:-$HOME/venv-denoise}"
SRC_VENV="${2:-$(python3 -c 'import sys; print(sys.prefix)')}"
OPENPROBLEMS_DIR="${OPENPROBLEMS_DIR:-$HOME/openproblems}"
REQ="requirements/denoising/requirements-denoising.txt"

say() { printf '\n=== %s\n' "$*"; }

[[ -d "$SRC_VENV" ]] || { echo "error: source venv not found: $SRC_VENV" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "error: $REQ not found (are you on aws-multi-gpu-launch?)" >&2; exit 1; }

# --- 0. What ray version must we end up with? --------------------------------
# If a head is already up, its version wins — we cannot change it without
# killing runs that are attached to it.
# NOTE: this reads the SOURCE venv's ray version, not the cluster's — Ray has no
# command that prints a running head's version. That is the right proxy here
# because the head is started by whichever venv launched first on this node, and
# that is the training venv we are cloning. If you started the head from some
# other environment, pass the correct version by setting HEAD_RAY yourself.
HEAD_RAY="${HEAD_RAY:-}"
if ray status >/dev/null 2>&1; then
  HEAD_RAY="${HEAD_RAY:-$("$SRC_VENV/bin/python" -c 'import ray; print(ray.__version__)')}"
  say "A Ray head is already up. Assuming it runs ray ${HEAD_RAY} (the source venv's)."
else
  HEAD_RAY="${HEAD_RAY:-$(grep -E '^ray==' "$REQ" | cut -d= -f3)}"
  say "No Ray head up yet. Will target ray==${HEAD_RAY} from ${REQ}."
  echo "IMPORTANT: start the head AFTER this script, so it runs this ray version."
fi

# --- 1. Clone the training venv ----------------------------------------------
if [[ -d "$CLONE" ]]; then
  say "Reusing existing clone at $CLONE"
else
  say "Cloning $SRC_VENV -> $CLONE"
  cp -a "$SRC_VENV" "$CLONE"
  # A copied venv keeps absolute paths in its activate scripts and shebangs.
  # --upgrade rewrites them to the new location without touching packages.
  python3 -m venv --upgrade "$CLONE" 2>/dev/null || true
fi
PY="$CLONE/bin/python"
[[ -x "$PY" ]] || { echo "error: no interpreter at $PY" >&2; exit 1; }

# --- 2. Install the bio stack, holding ray at the cluster's version ----------
# ray is pinned on the command line AFTER -r so it wins over the file's pin if
# an already-running head forces a different version.
say "Installing bio deps into the clone (ray held at ${HEAD_RAY})"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install -r "$REQ" "ray==${HEAD_RAY}"

# --- 3. Git dependencies ------------------------------------------------------
say "Installing git dependencies"
"$PY" -m pip install --quiet git+https://github.com/czbiohub/simscity.git
"$PY" -m pip install --quiet --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git

# --- 4. openproblems + the cellxgene API patch --------------------------------
if [[ ! -d "$OPENPROBLEMS_DIR" ]]; then
  say "Cloning openproblems -> $OPENPROBLEMS_DIR"
  git clone --quiet https://github.com/openproblems-bio/openproblems.git "$OPENPROBLEMS_DIR"
fi
say "Applying the cellxgene API fix"
PATCH="$REPO_ROOT/requirements/denoising/openproblems_api_fix.patch"
if git -C "$OPENPROBLEMS_DIR" apply --check "$PATCH" 2>/dev/null; then
  git -C "$OPENPROBLEMS_DIR" apply "$PATCH"
  echo "patch applied"
elif git -C "$OPENPROBLEMS_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied — skipping"
else
  echo "WARNING: patch does not apply cleanly to $OPENPROBLEMS_DIR." >&2
  echo "  The Tabula Muris loader will fail until the equivalent edit is made" >&2
  echo "  by hand — see requirements/denoising/README.md Known Issues #1." >&2
fi
"$PY" -m pip install --quiet -e "$OPENPROBLEMS_DIR"

# --- 5. Verify ----------------------------------------------------------------
# The only check that matters is the one the trainer itself does at
# mlora_train.py:2426. Everything else is diagnostics.
say "Verifying"
CLONE_RAY="$("$PY" -c 'import ray; print(ray.__version__)')"
echo "ray in clone: ${CLONE_RAY}   (must equal cluster: ${HEAD_RAY})"
if [[ "$CLONE_RAY" != "$HEAD_RAY" ]]; then
  echo "FAIL: ray version mismatch — denoising runs will not attach to the head." >&2
  echo "  Fix: $PY -m pip install ray==${HEAD_RAY}" >&2
  exit 1
fi

"$PY" - <<'PYEOF'
import importlib, sys
mods = ["numpy", "scipy", "scanpy", "anndata", "scprep", "graphtools", "numba", "magic"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m:<12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        bad.append(m); print(f"  FAIL {m:<12} {type(e).__name__}: {e}")
if bad:
    sys.exit(f"bio imports failed: {', '.join(bad)}")
PYEOF

say "Importing the verifier exactly as the trainer does"
PYTHONPATH="mLoRA:${PYTHONPATH:-}" "$PY" -c \
  'from tinker_cookbook.recipes.ttt.env_denoising import build_denoising_prompt, verify_denoising; print("verifier import OK")'

cat <<EOF

=== READY ===
Launch denoising on GPUs 4-7 with:

  DOMAIN=denoising GPU_START=4 PYTHON=$PY \\
    bash scripts/aws/launch_heavy_domain_wave.sh smoke

Pancreas data downloads to ~/.cache/denoising_datasets on the first run, so the
first epoch is slow. If the smoke run is healthy, same command with 'run'.
EOF
