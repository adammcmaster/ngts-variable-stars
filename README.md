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

## Initialise the NGTS DR2 data

```bash
uv run python init_manifest.py
```

This queries ESO's ObsCore service for DR2, checks that all 72 fields and 1,800
photometry tiles are present, and downloads the 72 field source catalogues. It
does not download the photometry tiles. All output goes under `data/` beside the
script, following the workstation's symlink to scratch. Use `--data-dir PATH` to
choose another location.

| Output | Contents |
| --- | --- |
| `data/catalogues/` | Original source catalogue FITS files, with FITS integrity checks |
| `data/source_catalogue.parquet` | Combined source catalogue, retaining all original columns and adding `FIELD` |
| `data/source_catalogue_manifest.csv` | Catalogue URLs, local paths, row counts, byte sizes and SHA-256 hashes |
| `data/tile_manifest.csv` | Tile fields/letters, ESO dataset IDs, download and DataLink URLs, estimated sizes, positions, observing bounds and cache paths |
| `data/archive_products.ecsv` | Original ESO product metadata, including units |
| `data/manifest_metadata.json` | Release provenance, query, timestamp and counts |
| `data/tiles/` | Cache directory for subsequent tile processing |

`cache_path` values are relative to `data/`; downstream code should use
`data_dir / row.cache_path`. Tile sizes are ESO estimates in **decimal kilobytes**,
not exact byte counts. The manifest contains inventory rather than mutable
processing status, so rerunning initialisation will not reset processing state.
Existing valid catalogue files are reused; interrupted downloads never replace
complete files. Generated files are replaced atomically, with the metadata JSON
written last after a successful run. Run only one initialiser at a time.

The combined catalogue preserves field membership without deduplicating sources
across fields. Source-to-tile membership should be taken from the actual
photometry files when processing them. The source catalogue's pixel positions
alone are not used to guess tile membership.

## Classify the photometry

Run with access to the host NVIDIA GPU:

```bash
uv run python classify_upsilont.py
```

The script works through `data/tile_manifest.csv` in order. It downloads one tile
into `data/tiles/`, validates its FITS identity and checksums, groups measurements
by source, extracts UPSILoN-T features in eight CPU worker processes, and predicts
classes in GPU batches in the main process. A separate download process prefetches
the next unfinished tile while the current tile is classified. At most one tile
is prefetched; `--max-tiles` also limits prefetching. FFT extraction defaults to
one thread per worker (`--fft-threads` controls this per-worker setting). tqdm displays overall tile progress, download bytes, and source progress.

Preprocessing defaults to **5-minute inverse-variance weighted flux bins**. All
finite, positive fluxes with finite, positive uncertainties are eligible. The
default rejection mask is 23 (bits 0, 1, 2 and 4: saturation, cosmic rays,
crossings and blooming spikes). Outlier-only flags are retained to avoid removing
real variability. No additional sigma clipping is performed. Times are sorted,
duplicate times are combined, and fluxes are converted to relative magnitudes
with propagated magnitude errors. At least 80 usable points after binning are
required. Times remain in days, with a constant HJD offset subtracted for numerical
conditioning. Binning smooths short signals and changes the classifier's features;
these first-pass classifications need scientific validation on NGTS.

All results and checkpoints are under `data/classifications/upsilon-t/`:

| File | Purpose |
| --- | --- |
| `checkpoints.sqlite` | Authoritative per-source results, feature checkpoints and completed-tile records |
| `tiles/{FIELD}{LETTER}.parquet` | Export for each completed tile, including source IDs, statuses, counts, features, predicted labels and all class probabilities |
| `run_config.json` | Preprocessing settings, package versions, feature/class order, model hashes and manifest hash |
| `summary.json` | Counts as of the last normally completed invocation (the SQLite checkpoint remains current after a crash) |

Each source's features are committed before inference; predictions are committed
individually. Rerunning the **same command automatically resumes**: completed
sources are skipped, extracted features awaiting prediction are reused, and
partial downloads use HTTP range requests. A tile is marked complete only after
all its sources have a recorded outcome and its Parquet export is saved. Completed
tile files are then deleted from the cache; use `--keep-tiles` to retain them.
The SQLite database includes results for the current incomplete tile, even before
its Parquet export exists. A file lock prevents concurrent writers.

Every source gets an explicit outcome: `classified`, `skipped`, or `error`.
Skipped sources include insufficient/constant light curves, nonfinite features,
and features outside the pretrained model's log-transform domain. In the latter
case the raw features and violated bounds are retained; they are not silently
clipped to force a prediction. Numerical extraction errors are recorded with a
reason; `--retry-errors` retries those sources. GPU, I/O and unexpected programming
failures stop the run, retaining the checkpoints. Completed-tile counts include
skipped/error outcomes, so inspect the status and reason columns when analysing
results.

Useful options:

```bash
# Stop after one additional tile, or a few additional unfinished sources.
uv run python classify_upsilont.py --max-tiles 1
uv run python classify_upsilont.py --max-sources 5

# Adjust CPU FFT threads and GPU batch size without resetting checkpoints.
uv run python classify_upsilont.py --fft-threads 4 --batch-size 32
```

`--bin-minutes 0` selects native cadence, which can require very large FFTs and
substantial RAM. `--reject-flags 31` also excludes outlier-flagged measurements.
Changing preprocessing, model/package versions, or the manifest is rejected when
existing checkpoints would mix incompatible results. Move the output directory
aside before starting a different configuration. `--min-period` passes through
UPSILoN-T's period-search parameter (default 0.03 days); upstream's internal
frequency-grid construction does not enforce it as a strict period cutoff.

## Norton CLEAN and phase-folding search

```bash
uv run python classify_norton.py
```

This is a Python port of Andrew Norton's SuperWASP Variable Stars period detector
([DOI: 10.3847/2515-5172/aaf291](https://doi.org/10.3847/2515-5172/aaf291)), using
both CLEAN and the local phase-folding search from his supplied `runfindper6.f`.
It runs on CPU, with Numba accelerating the numerical loops. The first invocation
may spend a few seconds compiling those loops.

Norton uses eight source-classification processes and one download process to
prefetch the next tile. Each classification process writes durable candidate
checkpoints through its own SQLite connection.

Like the UPSILoN-T script, it reads the tile manifest, defaults to 5-minute flux
bins and quality mask 23, displays tqdm progress, and resumes automatically. It
requires at least 1,000 usable points after NGTS binning, before Norton's own spike
removal. Use `--bin-minutes 0` for native cadence. It stores its tile cache under
`data/tiles/norton/` so independent Norton and UPSILoN-T runs do not delete or
partially overwrite each other's files. Completed Norton tiles are removed unless
`--keep-tiles` is selected.

Outputs are under `data/classifications/norton/`:

| File | Contents |
| --- | --- |
| `checkpoints.sqlite` | Per-source outcomes, CLEAN candidates and progress through candidate refinements |
| `tiles/{FIELD}{LETTER}.parquet` | One source-summary row, with the same identifiers, status, counts, label and probability columns as UPSILoN-T, plus the best period and detection statistics |
| `periods/{FIELD}{LETTER}.parquet` | Every accepted period, its CLEAN significance, folding ratio, warning flag and JSON-encoded 100-bin folded profile |
| `run_config.json` | DOI, algorithm settings, dependencies and manifest provenance |
| `summary.json` | Counts from the last normally completed invocation; SQLite is authoritative after a crash |

Norton's algorithm detects periodicity rather than physical stellar classes.
The labels are `periodic_candidate` (at least one accepted unflagged period),
`alias_only` (all accepted periods have Norton warning flags), and `no_period`.
`no_period` is not evidence that a star is non-variable. `probability` is null;
this algorithm supplies no calibrated class probability. A source summary picks
the highest folding ratio among unflagged periods, or among all accepted periods
if none is unflagged. The complete period table preserves Norton's descending
period order and numbered candidates. Periods are stored in both days and seconds.

CLEAN candidates are checkpointed before folding, and each candidate refinement
is committed independently. A failure during CLEAN repeats only the current
source's CLEAN stage; a failure during refinement resumes at that candidate.
Completed sources are reused. Both Parquet exports must be written before a tile
is marked complete or its cache is removed. `--max-sources`, `--max-tiles`,
`--retry-errors`, `--keep-tiles`, and `--data-dir` work as in the UPSILoN-T script.
Only one Norton invocation (with its worker pool) may use a given output directory. Changing numerical
settings, relevant dependencies or the manifest requires a separate
output run (move the existing Norton output directory aside). Source-code changes
do not block resuming; legacy source fingerprints are removed automatically when
opening compatible checkpoints.

See [the port notes](docs/norton_port.md) for the original thresholds, numerical
repairs, retained legacy behaviours and validation against the supplied Fortran.

## Preview classifications

Open `preview_upsilon.ipynb` or `preview_norton.ipynb` in Jupyter from the
repository root and run all cells. Each reads a read-only SQLite snapshot,
including results from unfinished tiles, shows category counts and processing
statuses, and plots up to three examples per category from a single tile.
Automatic selection prefers cached tiles, then category coverage. Set
`PREVIEW_TILE` to a field+tile identifier to choose one explicitly (use the same
value in both notebooks to share a tile). Counts cover the full run; categories
missing from the selected tile have empty panels. Rerun to refresh.

Missing photometry is downloaded into the separate `preview_cache/` directory;
set `DOWNLOAD_MISSING = False` for offline use. Norton detections use their saved
folded profiles. Its `no_period` examples use an explicitly labelled one-day
reference fold, which does not imply a detected period.
