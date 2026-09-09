"""Exercise the actual spawn boundary with small, local FITS tiles."""

from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import classify_norton
import classify_upsilont
from classification_workers import TilePrefetch


@pytest.mark.parametrize("module", [classify_norton, classify_upsilont])
def test_spawned_classification_budget_resume_and_cleanup(tmp_path, module):
    record = {
        "dp_id": "ADP.1",
        "field": "NG0000-0000",
        "tile": "A",
        "filename": "FLUX_NG0000-0000A.fits",
    }
    prefix = "tiles/norton" if module is classify_norton else "tiles"
    record["cache_path"] = f"{prefix}/{record['filename']}"
    path = tmp_path / record["cache_path"]
    path.parent.mkdir(parents=True)
    rows = Table(
        {
            "SOURCE_ID": [f"source{i}" for i in range(12)],
            "HJD": np.arange(12, dtype=float),
            "SYSFLUX": np.ones(12),
            "FLUX_ERR": np.ones(12),
            "FLAG": np.zeros(12, dtype=int),
        }
    )
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = record["filename"]
    fits.HDUList([primary, fits.BinTableHDU(rows)]).writeto(path, checksum=True)
    output = tmp_path / "results"
    for directory in ("tiles", "periods"):
        (output / directory).mkdir(parents=True)
    config = {
        "bin_minutes": 0,
        "reject_flags": 23,
        "min_points": 80,
        "feature_names": ["period"],
        "classes": ["A"],
    }
    args = SimpleNamespace(keep_tiles=False, batch_size=2, fft_threads=1)
    db = module.checkpoint_db(output / "checkpoints.sqlite", config)
    positional = [db, record, tmp_path, output]
    if module is classify_upsilont:
        positional.append(None)  # All short sources skip GPU inference.
    try:
        with get_context("spawn").Pool(8) as pool:
            assert module.process_tile(
                *positional, config, args, None, 3, pool=pool, tile_path=path
            ) == (3, False)
            assert path.exists()
            assert db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3
            assert module.process_tile(
                *positional, config, args, None, None, pool=pool, tile_path=path
            ) == (9, True)
        assert not path.exists()
        assert (
            db.execute(
                "SELECT COUNT(*) FROM results WHERE status='skipped'"
            ).fetchone()[0]
            == 12
        )
    finally:
        db.close()


def test_prefetch_only_submits_current_and_next_and_skips_done():
    records = [{"dp_id": str(i)} for i in range(5)]
    submitted = []

    class Result:
        def get(self):
            return "cached"

    class Pool:
        def apply_async(self, fn, args):
            submitted.append(args[0][0]["dp_id"])
            return Result()

    prefetch = TilePrefetch(records, {"1"}, None, max_tiles=3)
    prefetch.pool = Pool()
    prefetch._advance()
    assert submitted == ["0"]
    assert prefetch.take(records[0]) == "cached"
    assert submitted == ["0", "2"]
    prefetch.take(records[2])
    assert submitted == ["0", "2", "3"]
    prefetch.take(records[3])
    assert prefetch.pending is None


def test_spawned_prefetch_validates_cache_and_propagates_failure(tmp_path):
    record = {
        "dp_id": "one",
        "field": "NG0000-0000",
        "tile": "A",
        "filename": "FLUX_NG0000-0000A.fits",
        "cache_path": "tile.fits",
    }
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = record["filename"]
    rows = Table(
        {
            "SOURCE_ID": ["source"],
            "HJD": [1.0],
            "SYSFLUX": [1.0],
            "FLUX_ERR": [1.0],
            "FLAG": [0],
        }
    )
    path = tmp_path / record["cache_path"]
    fits.HDUList([primary, fits.BinTableHDU(rows)]).writeto(path, checksum=True)
    missing = {**record, "dp_id": "two", "cache_path": "missing.fits"}
    with TilePrefetch([record, missing], set(), tmp_path) as prefetch:
        assert prefetch.take(record) == path
        # No download URL: the next worker's failure must reach the main process.
        with pytest.raises(KeyError, match="download_url"):
            prefetch.take(missing)
