#!/bin/bash
# ============================================================================
# Rebuild the two venvs the ICLR reruns need, from scratch.
#
#   bash scripts/iclr/rebuild_venvs.sh cp        # gamma + alpha0 arms
#   bash scripts/iclr/rebuild_venvs.sh denoise   # denoising pair
#   bash scripts/iclr/rebuild_venvs.sh both
#
# WHY THIS EXISTS
#   scripts/aws/setup_denoising_venv.sh builds the bio venv by CLONING an
#   existing training venv. On the RTX PRO 6000 box that venv lived on
#   /opt/dlami/nvme, which is AWS instance store: it is wiped on every instance
#   stop/start. After a wipe there is nothing to clone and that script cannot
#   run (with no venv active its SRC_VENV defaults to sys.prefix = /usr, so its
#   clone step would `cp -a /usr`). This script rebuilds both venvs from the
#   requirements files instead. Derived and verified 2026-08-22.
#
# FIDELITY NOTES — each of these was a real failure, not a precaution.
#   1. requirements/requirements-math.txt matches the cp26 runs' own
#      wandb-recorded manifest with ZERO version conflicts, so it is the correct
#      spec for the cp arms. But it OMITS 7 packages the real env had, and one
#      of them, tensorboard, is imported by the trainer via
#      torch.utils.tensorboard -- without it the trainer dies on import.
#   2. requirements/denoising/requirements-denoising.txt is NOT self-sufficient:
#      no chz, no tensorboard. It was only ever an overlay on top of a cloned
#      training venv. So the denoising venv = denoising reqs + the packages that
#      requirements-math.txt has and it does not, installed under
#      --constraint requirements-denoising.txt so the denoising pins cannot move.
#   3. openproblems must be the v0.8.0 TAG (its version.py says 0.7.0 -- ignore
#      that) and must be installed --no-deps --no-build-isolation: its setup.py
#      pins numpy<1.24 / pandas 1.3.5 / scipy<1.10, which contradict the
#      denoising stack and have no cp312 wheels. It needs setuptools<81 for
#      pkg_resources. BOTH patches must be applied -- the cellxgene API fix and
#      the np.int -> int fix (np.int was removed in numpy 1.24; the denoising
#      verifier hits it on the reward path).
#   4. pip's downloader repeatedly dies at 179/900 MB on the torch wheel on this
#      host; curl fetches the same URL first try. torch is therefore fetched with
#      curl, sha256-checked against PyPI, and installed from the local file.
#
# ALSO REMEMBER (not this script's job, but it will bite you):
#   Launch the denoising pair with RAY_TMP_ROOT=/nvme/raytmp. Ray's AF_UNIX
#   socket path cannot exceed 107 bytes and the default /opt/dlami/nvme/raytmp
#   overruns it by 6. Create /nvme -> /opt/dlami/nvme once: sudo ln -sfn /opt/dlami/nvme /nvme
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO"

VENV_CP="${VENV_CP:-/opt/dlami/nvme/venvs/venv-cp}"
VENV_DEN="${VENV_DEN:-/opt/dlami/nvme/venv-denoise}"
OPENPROBLEMS_DIR="${OPENPROBLEMS_DIR:-/opt/dlami/nvme/openproblems}"
WHEELS="${WHEELS:-/opt/dlami/nvme/wheels}"
TORCH_VER="2.9.1"
TORCH_WHL="torch-${TORCH_VER}-cp312-cp312-manylinux_2_28_x86_64.whl"
# present in the recorded cp26 env but absent from requirements-math.txt
EXTRAS=(accelerate==1.14.0 colorama==0.4.6 tensorboard==2.21.0
        tensorboard-data-server==0.7.2 werkzeug==3.1.8)

say() { printf '\n=== %s\n' "$*"; }

fetch_torch() {   # pip stalls on this wheel; curl does not
  mkdir -p "$WHEELS"
  local out="$WHEELS/$TORCH_WHL"
  local url want got
  url=$(curl -s "https://pypi.org/pypi/torch/${TORCH_VER}/json" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for f in d['urls']:
    if f['filename']=='$TORCH_WHL': print(f['url']); break")
  want=$(curl -s "https://pypi.org/pypi/torch/${TORCH_VER}/json" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for f in d['urls']:
    if f['filename']=='$TORCH_WHL': print(f['digests']['sha256']); break")
  [ -n "$url" ] || { echo "cannot resolve torch wheel URL" >&2; return 1; }
  for _ in $(seq 1 30); do
    got=$(sha256sum "$out" 2>/dev/null | cut -d' ' -f1)
    [ "$got" = "$want" ] && { echo "torch wheel present and sha256-verified"; return 0; }
    curl -sS -L -C - --retry 5 --retry-delay 2 --retry-all-errors \
         --connect-timeout 20 -o "$out" "$url"
  done
  echo "torch wheel did not verify (want $want)" >&2; return 1
}

build_cp() {
  say "cp venv -> $VENV_CP"
  [ -x "$VENV_CP/bin/python" ] || python3 -m venv "$VENV_CP"
  "$VENV_CP/bin/python" -m pip install --quiet --upgrade pip wheel
  fetch_torch || return 1
  "$VENV_CP/bin/python" -m pip install "$WHEELS/$TORCH_WHL"
  "$VENV_CP/bin/python" -m pip install -r requirements/requirements-math.txt
  "$VENV_CP/bin/python" -m pip install "${EXTRAS[@]}" setuptools==83.0.0
  say "verify cp"
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" "$VENV_CP/bin/python" -c \
    'import tinker_cookbook.rl.mlora_train; print("cp trainer import OK")' || return 1
}

build_denoise() {
  say "denoising venv -> $VENV_DEN"
  [ -x "$VENV_DEN/bin/python" ] || python3 -m venv "$VENV_DEN"
  "$VENV_DEN/bin/python" -m pip install --quiet --upgrade pip wheel
  fetch_torch || return 1
  "$VENV_DEN/bin/python" -m pip install "$WHEELS/$TORCH_WHL"
  "$VENV_DEN/bin/python" -m pip install -r requirements/denoising/requirements-denoising.txt

  say "trainer packages missing from the denoising reqs (see note 2)"
  python3 - > /tmp/denoise-delta.txt <<'PY'
import re
def parse(p):
    d={}
    for line in open(p):
        line=line.strip()
        if not line or line.startswith('#'): continue
        m=re.match(r'^([A-Za-z0-9._-]+)==(.+)$', line)
        if m: d[m.group(1).lower().replace('_','-')]=m.group(2)
    return d
math=parse('requirements/requirements-math.txt')
den =parse('requirements/denoising/requirements-denoising.txt')
for k,v in sorted(math.items()):
    if k not in den: print(f"{k}=={v}")
PY
  "$VENV_DEN/bin/python" -m pip install \
      --constraint requirements/denoising/requirements-denoising.txt \
      -r /tmp/denoise-delta.txt
  "$VENV_DEN/bin/python" -m pip install \
      --constraint requirements/denoising/requirements-denoising.txt "${EXTRAS[@]}"
  "$VENV_DEN/bin/python" -m pip install --quiet 'setuptools<81'   # pkg_resources

  say "git deps"
  "$VENV_DEN/bin/python" -m pip install --quiet git+https://github.com/czbiohub/simscity.git
  "$VENV_DEN/bin/python" -m pip install --quiet --no-deps \
      git+https://github.com/czbiohub/molecular-cross-validation.git

  say "openproblems v0.8.0 + both patches"
  [ -d "$OPENPROBLEMS_DIR" ] || git clone --quiet --branch v0.8.0 --depth 1 \
      https://github.com/openproblems-bio/openproblems.git "$OPENPROBLEMS_DIR"
  for p in requirements/denoising/openproblems_api_fix.patch \
           requirements/denoising/openproblems_npint_fix.patch; do
    if git -C "$OPENPROBLEMS_DIR" apply --check "$REPO/$p" 2>/dev/null; then
      git -C "$OPENPROBLEMS_DIR" apply "$REPO/$p"; echo "  applied $(basename $p)"
    else
      echo "  $(basename $p) already applied or does not apply — check manually"
    fi
  done
  "$VENV_DEN/bin/python" -m pip install --no-deps --no-build-isolation -e "$OPENPROBLEMS_DIR"

  say "verify denoising (the check that matters: verifier is imported IN-PROCESS)"
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" "$VENV_DEN/bin/python" -c \
    'from tinker_cookbook.recipes.ttt.env_denoising import build_denoising_prompt, verify_denoising; print("verifier import OK")' || return 1
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" "$VENV_DEN/bin/python" -c \
    'import tinker_cookbook.rl.mlora_train; print("denoise trainer import OK")' || return 1
}

case "${1:-}" in
  cp)      build_cp ;;
  denoise) build_denoise ;;
  both)    build_cp && build_denoise ;;
  *) echo "usage: $0 <cp|denoise|both>"; exit 1 ;;
esac
