"""Exercise the actual spawn boundary with small, local FITS tiles."""

from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import classification_workers
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
        def get(self, timeout=None):
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


def test_norton_run_with_prefetch_and_parent_progress(tmp_path):
    """Run the CLI through download, real numerics, checkpointing and shutdown."""
    import subprocess
    import sys
    from pathlib import Path

    import pandas as pd

    records = []
    rng = np.random.default_rng(124)
    t = np.sort(rng.uniform(0, 30, 2200))
    rows = Table(
        {
            "SOURCE_ID": ["source"] * len(t),
            "HJD": 2450000 + t,
            "SYSFLUX": 1000
            + 100 * np.sin(2 * np.pi * t / 0.73)
            + rng.normal(0, 8, len(t)),
            "FLUX_ERR": np.full(len(t), 8.0),
            "FLAG": np.zeros(len(t), dtype=int),
        }
    )
    for tile in "AB":
        filename = f"FLUX_NG0000-0000{tile}.fits"
        primary = fits.PrimaryHDU()
        primary.header["ORIGFILE"] = filename
        payload = tmp_path / filename
        fits.HDUList([primary, fits.BinTableHDU(rows)]).writeto(payload, checksum=True)
        records.append(
            {
                "dp_id": f"ADP.{tile}",
                "field": "NG0000-0000",
                "tile": tile,
                "filename": filename,
                "cache_path": f"tiles/{filename}",
                "download_url": str(payload),
            }
        )
    pd.DataFrame(records).to_csv(tmp_path / "tile_manifest.csv", index=False)
    # Install a delayed local transport in every spawned process, without
    # depending on ESO or opening a listening network socket in the test.
    root = Path(classify_norton.__file__).parent
    harness = tmp_path / "run_local.py"
    harness.write_text(
        f"import sys\nsys.path.insert(0, {str(root)!r})\n"
        + """
from pathlib import Path
from time import sleep
import init_manifest
class Response:
    status_code = 200
    def __init__(self, content):
        self.content = content
        self.headers = {"Content-Length": str(len(content))}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, size):
        for offset in range(0, len(self.content), 16384):
            sleep(0.02)
            yield self.content[offset:offset+16384]
class Session:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def get(self, url, **kwargs):
        sleep(0.3)
        return Response(Path(url).read_bytes())
init_manifest.ArchiveSession = Session
if __name__ == "__main__":
    import classify_norton
    classify_norton.main()
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            str(harness),
            "--data-dir",
            str(tmp_path),
            "--bin-minutes",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"completed_tiles": 2' in result.stdout
    for message in ("Tiles", "Sources", "Connecting", "Checking downloaded"):
        assert message in result.stderr
    for tile in "AB":
        output = tmp_path / "classifications" / "norton"
        frame = pd.read_parquet(output / "tiles" / f"NG0000-0000{tile}.parquet")
        assert frame.status.tolist() == ["classified"]
        assert frame.period_days.iloc[0] == pytest.approx(0.73, abs=0.0002)
    assert not list((tmp_path / "tiles" / "norton").glob("*.fits"))


def test_worker_crash_raises_instead_of_waiting_forever():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from multiprocessing import get_context
from classification_workers import source_results
if __name__ == "__main__":
    with get_context("spawn").Pool(1) as pool:
        list(source_results(pool, os._exit, [17]))
""",
        ],
        cwd=Path(classify_norton.__file__).parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "exited unexpectedly (exit code 17)" in result.stderr


def test_download_worker_retries_from_beginning(tmp_path, monkeypatch):
    import queue

    import requests

    record = {
        "field": "NG0000-0000",
        "tile": "A",
        "cache_path": "tiles/tile.fits",
    }
    partial = tmp_path / "tiles" / "tile.fits.part"
    partial.parent.mkdir()
    attempts = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def cache_tile(session, record, data_dir, **kwargs):
        attempts.append(partial.stat().st_size if partial.exists() else 0)
        if len(attempts) < 3:
            with partial.open("ab") as output:
                output.write(b"next")
            partial.unlink()
            raise requests.ConnectionError("timed out")
        return partial

    events = queue.Queue()
    classification_workers.init_downloader(events)
    monkeypatch.setattr(classification_workers, "cache_tile", cache_tile, raising=False)
    monkeypatch.setattr("init_manifest.ArchiveSession", Session)
    # download_tile imports cache_tile inside the function, so patch its source module.
    monkeypatch.setattr(classify_upsilont, "cache_tile", cache_tile)
    assert classification_workers.download_tile((record, tmp_path)) == partial
    assert attempts == [0, 0, 0]
    reported = []
    while not events.empty():
        reported.append(events.get_nowait())
    statuses = [value for kind, value in reported if kind == "status"]
    assert any("retry 2/5 from the beginning" in status for status in statuses)
    assert any("retry 3/5 from the beginning" in status for status in statuses)
