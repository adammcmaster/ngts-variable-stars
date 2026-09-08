# NGTS variable stars

Search the NGTS photometry archives for variable stars, initially using
[UPSILoN-T](https://github.com/dwkim78/UPSILoN-T).

## Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
uv run jupyter lab
```

Python 3.11 is selected in `.python-version`. Commit `pyproject.toml` and `uv.lock`
to keep the environment reproducible. Notebook and development dependencies are
installed by default; `uv sync --locked --no-default-groups` installs only the
analysis dependencies.

On this workstation, `.venv` is a symlink to
`~/scratch/ngts-variable-stars/.venv`. To keep downloads on scratch too, use:

```bash
UV_CACHE_DIR="$HOME/scratch/ngts-variable-stars/uv-cache" uv sync --locked
```

The environment includes:

- Astropy for FITS, coordinates, time handling and period searches; Astroquery for
  archive and catalogue queries.
- NumPy, SciPy, pandas and scikit-learn for numerical analysis and tabular data.
- h5py and PyArrow for HDF5 and Parquet storage.
- Matplotlib, JupyterLab and ipykernel for exploration and plots.
- UPSILoN-T, PyTorch and pyFFTW for classification and FFT feature extraction.
- tqdm for progress reporting; pytest and Ruff for development.

UPSILoN-T is pinned to an upstream Git commit. Its package requires scikit-learn
1.5.0; NumPy is kept below 2 for its legacy feature-extraction code. PyTorch is
declared explicitly because upstream does not include it in its package metadata.

## Check UPSILoN-T

Upstream requires a CUDA-capable NVIDIA GPU for classification. Run these commands
with access to the host GPU (outside a sandbox that hides CUDA):

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

The upstream wheel omits the EROS light curve required by `test_predict()`.
Download that sample from the same pinned revision to reproduce the prediction:

```bash
uv run python - <<'PY'
from urllib.request import urlopen

import numpy as np
import pandas as pd
from upsilont import UPSILoNT
from upsilont.features import VariabilityFeatures

url = (
    "https://raw.githubusercontent.com/dwkim78/UPSILoN-T/"
    "2d95da6b52104e3461958d6c7c6dfecf3bb9bba3/"
    "upsilont/datasets/lightcurves/lm0134l19756.time"
)
with urlopen(url, timeout=30) as sample:
    time, magnitude, error = np.loadtxt(sample, usecols=(0, 1, 2), unpack=True)
valid = magnitude < 99.999  # Missing-value sentinel in the bundled example.
features = VariabilityFeatures(
    time[valid], magnitude[valid], error[valid]
).get_features()
label, probabilities = UPSILoNT().predict(
    pd.DataFrame([features]), return_prob=True
)
print(label, np.max(probabilities))
PY
```

Verified on this workstation's NVIDIA A100: 16 features, prediction `RRL_ab`,
maximum probability approximately 0.962086. Feature extraction alone runs on CPU.
Upstream's saved label encoder emits a scikit-learn version warning (saved with
0.22, upstream now requires 1.5.0); this smoke test reproduces the documented
prediction, but does not validate classification accuracy on NGTS data.

NGTS archive access and the survey search pipeline are not configured yet. The
initial pipeline will need to prepare time, magnitude and magnitude-uncertainty
arrays from NGTS photometry before extracting features and classifying candidates.
