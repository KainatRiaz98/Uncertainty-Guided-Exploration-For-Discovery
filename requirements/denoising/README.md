# OpenProblems Denoising Benchmark

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

**Do not create a fresh empty venv.** The denoising verifier is imported
in-process by the trainer (`tinker_cookbook/rl/mlora_train.py:2426`), so these
deps have to live in a venv that already has torch and the training stack. And
because the install churns NumPy (see Known Issues #2), do it in a **clone** of
the training venv rather than the training venv itself — otherwise it can break
co-resident runs of other domains when they auto-resume after a reboot. Launch
the denoising runs against the clone with
`PYTHON=~/venv-denoise/bin/python`.

```bash
cp -a ~/venv ~/venv-denoise
source ~/venv-denoise/bin/activate

# Install requirements
uv pip install -r requirements-denoising.txt

# Git dependencies
uv pip install git+https://github.com/czbiohub/simscity.git
uv pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
uv pip install -e ./openproblems

# Minro API Change (if not already applied)
cd openproblems && git apply ../openproblems_api_fix.patch && cd .. 
```

## Known Issues

### 1. CZI cellxgene API changed
Tabula Muris loader fails. The API now uses:
- `dataset["dataset_id"]` instead of `dataset["id"]`
- Assets embedded in dataset: `dataset["assets"]`
- `asset["url"]` instead of `asset["presigned_url"]`

**Fix**: `openproblems_api_fix.patch`

### 2. NumPy 2.0 API removal (NOT a torch conflict)
This was previously described as a "NumPy 2.x vs torch" conflict. It isn't —
`torch==2.9.1` is pinned identically in `requirements-math.txt`,
`requirements-ahc.txt` and this file. The break is a NumPy 2.0 API removal in
the bio code itself, and it happens under every pin here:
```python
# Old (breaks):
np.asarray(Y, dtype=np.float64, copy=False)

# New (works):
np.asarray(Y, dtype=np.float64)
```

### 3. ray must match the node's Ray head
The pins disagree: `requirements-math.txt` has `ray==2.51.1`, while this file
and `requirements-ahc.txt` have `ray==2.53.0`. Every run attaches to one shared
head per node (`ray.init("auto")`), and Ray refuses to attach across a version
mismatch. numpy and scipy may differ freely between venvs — they never cross a
process boundary — but ray may not. `scripts/aws/setup_denoising_venv.sh`
detects the node's version and pins to it; if you install by hand, match it
yourself and start the head *after* installing.

