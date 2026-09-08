"""Test Norton tile/checkpoint behaviour independently of expensive numerics."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table

import classify_norton as classifier
import norton_algorithm as norton


@pytest.fixture
def config():
    return {
        "algorithm": norton.DEFAULTS.copy(),
        "bin_minutes": 0,
        "min_points": 1000,
        "reject_flags": 23,
    }


@pytest.fixture
def rows():
    t = np.arange(1200, dtype=float)
    return Table(
        {
            "SOURCE_ID": ["source"] * len(t),
            "HJD": 2450000 + t / 1440,
            "SYSFLUX": 1000 + 10 * np.sin(t / 80),
            "FLUX_ERR": np.full(len(t), 10.0),
            "FLAG": np.zeros(len(t), dtype=int),
        }
    ).as_array()


def candidate(period):
    return {
        "clean_period_seconds": period,
        "period_flag": norton.period_flag(period),
        "sigma": 5.0,
    }


def accepted(item):
    period = item["clean_period_seconds"]
    return {
        **item,
        "period_seconds": period,
        "period_days": period / 86400,
        "chi_squared_ratio": 25.0,
        "period_number": 1,
        "folded_profile": {
            "phase": [0.5],
            "flux": [1.0],
            "flux_error": [0.1],
            "count": [100],
        },
    }


def test_preparation_preserves_flux_and_converts_hjd_days_to_seconds(rows, config):
    arrays, counts, reason = classifier.prepare_flux(rows, config)
    assert reason is None and counts["n_used"] == 1200
    assert arrays[0][1] == pytest.approx(60, abs=1e-4)
    np.testing.assert_array_equal(arrays[1], rows["SYSFLUX"])
    np.testing.assert_array_equal(arrays[2], rows["FLUX_ERR"])


def test_flags_bins_and_short_curves(rows, config):
    config["bin_minutes"] = 5
    rows["FLAG"][:5] = [1, 2, 4, 8, 16]
    rows["FLUX_ERR"][5] = -1
    rows["SYSFLUX"][6] = np.nan
    _, counts, reason = classifier.prepare_flux(rows, config)
    assert counts["n_valid"] == 1194
    assert reason == "too_few_points" and counts["n_used"] <= 240


def test_resume_after_candidate_refinement_failure(tmp_path, monkeypatch, rows, config):
    path = tmp_path / "checkpoints.sqlite"
    db = classifier.checkpoint_db(path, config)
    record = {"dp_id": "ADP.1"}
    calls = []

    def clean(*args):
        calls.append("clean")
        return np.array([1.0]), np.array([1.0]), {}

    monkeypatch.setattr(norton, "clean_spectrum", clean)
    monkeypatch.setattr(
        norton,
        "spectral_candidates",
        lambda *args: ([candidate(63000), candidate(86400)], {}),
    )

    def refine(t, x, e, items, algorithm, prepared):
        calls.append(items[0]["clean_period_seconds"])
        if items[0]["clean_period_seconds"] == 86400:
            raise RuntimeError("interrupted")
        return [accepted(items[0])]

    monkeypatch.setattr(norton, "refine_candidates", refine)
    with pytest.raises(RuntimeError, match="interrupted"):
        classifier.analyse_source(db, record, "source", rows, config)
    saved = json.loads(db.execute("SELECT payload FROM results").fetchone()[0])
    assert saved["next_candidate"] == 1 and len(saved["periods"]) == 1
    db.close()
    db = classifier.checkpoint_db(path, config)
    monkeypatch.setattr(
        norton,
        "refine_candidates",
        lambda t, x, e, items, algorithm, prepared: [accepted(items[0])],
    )
    payload = classifier.analyse_source(db, record, "source", rows, config, saved)
    assert calls == ["clean", 63000, 86400]
    assert payload["status"] == "classified"
    assert payload["label"] == "periodic_candidate"
    assert payload["n_periods"] == 2 and payload["n_good_periods"] == 1
    assert payload["best_period"]["period_flag"] == 0
    assert [p["period_number"] for p in payload["periods"]] == [1, 2]
    db.close()


def test_tile_completion_exports_and_cache_isolation(
    tmp_path, rows, config, monkeypatch
):
    record = {
        "dp_id": "ADP.1",
        "field": "NG0000-0000",
        "tile": "A",
        "filename": "FLUX_NG0000-0000A.fits",
        "cache_path": "tiles/FLUX_NG0000-0000A.fits",
    }
    # The shared UPSILoN-T cache must not be touched by this script.
    shared = tmp_path / record["cache_path"]
    shared.parent.mkdir()
    shared.write_bytes(b"owned by another classifier")
    cache = shared.parent / "norton" / shared.name
    cache.parent.mkdir()
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = record["filename"]
    fits.HDUList([primary, fits.BinTableHDU(rows)]).writeto(cache, checksum=True)
    output = tmp_path / "classifications" / "norton"
    for directory in ("tiles", "periods"):
        (output / directory).mkdir(parents=True)
    db = classifier.checkpoint_db(output / "checkpoints.sqlite", config)
    calls = []

    def analyse(db, record, source_id, rows, config, previous):
        calls.append(source_id)
        payload = {
            "status": "classified",
            "label": "no_period",
            "n_raw": len(rows),
            "n_valid": len(rows),
            "n_used": len(rows),
            "periods": [],
        }
        classifier.save_result(db, record["dp_id"], source_id, payload)
        return payload

    monkeypatch.setattr(classifier, "analyse_source", analyse)
    args = SimpleNamespace(keep_tiles=False)
    assert classifier.process_tile(
        db, record, tmp_path, output, config, args, None, None
    ) == (1, True)
    assert not cache.exists() and shared.read_bytes() == b"owned by another classifier"
    stem = "NG0000-0000A.parquet"
    frame = pd.read_parquet(output / "tiles" / stem)
    assert frame.label.tolist() == ["no_period"]
    assert frame.probability.isna().all()
    assert pd.read_parquet(output / "periods" / stem).empty
    (output / "tiles" / stem).unlink()
    assert classifier.process_tile(
        db, record, tmp_path, output, config, args, None, None
    ) == (0, True)
    assert calls == ["source"]
    assert (output / "tiles" / stem).exists()
    db.close()
