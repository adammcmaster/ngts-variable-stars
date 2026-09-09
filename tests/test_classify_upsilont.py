"""Exercise restart boundaries, preprocessing, grouping and tile cache lifecycle."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table

import classify_upsilont as classifier


@pytest.fixture
def config():
    return {
        "bin_minutes": 5,
        "min_points": 80,
        "reject_flags": 23,
        "min_period": 0.03,
        "feature_names": ["period"],
        "classes": ["A", "B"],
    }


def lightcurve(n=100):
    rows = np.zeros(
        n, dtype=[("HJD", "f8"), ("SYSFLUX", "f8"), ("FLUX_ERR", "f8"), ("FLAG", "i4")]
    )
    rows["HJD"] = 2450000 + np.arange(n) / 100
    rows["SYSFLUX"] = 100 + np.sin(np.arange(n))
    rows["FLUX_ERR"] = 1
    return rows


def test_flags_and_flux_cleaning(config):
    rows = lightcurve()
    rows["FLAG"][:6] = [0, 1, 2, 4, 8, 16]
    rows["SYSFLUX"][6] = 0
    rows["FLUX_ERR"][7] = -1
    rows["HJD"][8] = np.nan
    prepared, counts, reason = classifier.prepare_lightcurve(rows, config)
    assert reason is None
    assert counts == {"n_raw": 100, "n_valid": 93, "n_used": 93}
    assert np.isfinite(prepared).all()
    assert np.all(np.diff(prepared[0]) > 0)


def test_weighted_flux_binning(config):
    config["min_points"] = 1
    rows = lightcurve(4)
    rows["HJD"] = [100, 100 + 1 / 1440, 100 + 6 / 1440, 100 + 7 / 1440]
    rows["SYSFLUX"] = [100, 200, 200, 300]
    rows["FLUX_ERR"] = [1, 2, 1, 2]
    prepared, counts, reason = classifier.prepare_lightcurve(rows, config)
    assert reason is None and counts["n_used"] == 2
    _, mag, err = prepared
    flux = np.array([120, 220])
    np.testing.assert_allclose(mag, -2.5 * np.log10(flux / np.median(flux)))
    np.testing.assert_allclose(err, 2.5 / np.log(10) / np.sqrt(1.25) / flux)


def test_native_mode_merges_duplicate_times(config):
    config.update(bin_minutes=0, min_points=1)
    rows = lightcurve(4)
    rows["HJD"] = [3, 1, 1, 2]
    prepared, counts, reason = classifier.prepare_lightcurve(rows, config)
    assert reason is None and counts["n_used"] == 3
    np.testing.assert_array_equal(prepared[0], [0, 1, 2])


@pytest.mark.parametrize("ids", [[b"B", b"B", b"A"], [b"B", b"A", b"B", b"A"]])
def test_grouping_handles_contiguous_and_interleaved_rows(ids):
    ids = np.array(ids)
    groups = classifier.source_groups(ids)
    assert {name for name, _ in groups} == {"A", "B"}
    assert sum(len(ids[selector]) for _, selector in groups) == len(ids)
    for name, selector in groups:
        assert np.all(ids[selector] == name.encode())


def test_checkpoint_reopens_and_rejects_mixed_configuration(tmp_path, config):
    path = tmp_path / "checkpoints.sqlite"
    db = classifier.checkpoint_db(path, config)
    classifier.save_result(
        db, "tile", "source", {"status": "features", "features": {"period": 1.0}}
    )
    db.close()
    db = classifier.checkpoint_db(path, config)
    assert db.execute("SELECT status FROM results").fetchone()[0] == "features"
    db.close()
    with pytest.raises(ValueError, match="configuration differs"):
        classifier.checkpoint_db(path, {**config, "bin_minutes": 10})


class FakeModel:
    def predict(self, features, return_prob):
        assert return_prob and list(features.columns) == ["period"]
        return ["B"] * len(features), [[0.25, 0.75]] * len(features)


def test_resume_after_gpu_failure_reuses_features_and_finishes_tile(
    tmp_path, config, monkeypatch
):
    record = {
        "dp_id": "ADP.1",
        "field": "NG0000-0000",
        "tile": "A",
        "filename": "FLUX_NG0000-0000A.fits",
        "cache_path": "tiles/FLUX_NG0000-0000A.fits",
    }
    cache = tmp_path / record["cache_path"]
    cache.parent.mkdir()
    rows = Table(lightcurve(4))
    rows["SOURCE_ID"] = ["one", "one", "two", "two"]
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = record["filename"]
    fits.HDUList([primary, fits.BinTableHDU(rows)]).writeto(cache, checksum=True)
    output = tmp_path / "results"
    (output / "tiles").mkdir(parents=True)
    db = classifier.checkpoint_db(output / "checkpoints.sqlite", config)
    calls = []

    def extract(rows, config, fft_threads):
        calls.append(1)
        return {
            "status": "features",
            "features": {"period": 1.0},
            "n_raw": 2,
            "n_valid": 2,
            "n_used": 2,
        }

    monkeypatch.setattr(classifier, "extract_source", extract)
    args = SimpleNamespace(keep_tiles=False, batch_size=2, fft_threads=1)

    class FailingModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("GPU interrupted")

    with pytest.raises(RuntimeError, match="GPU interrupted"):
        classifier.process_tile(
            db, record, tmp_path, output, FailingModel(), config, args, None, None
        )
    assert len(calls) == 2 and cache.exists()
    assert (
        db.execute("SELECT COUNT(*) FROM results WHERE status='features'").fetchone()[0]
        == 2
    )
    assert not db.execute("SELECT * FROM tiles").fetchall()
    db.close()
    db = classifier.checkpoint_db(output / "checkpoints.sqlite", config)
    count, complete = classifier.process_tile(
        db, record, tmp_path, output, FakeModel(), config, args, None, None
    )
    assert complete and count == 2 and len(calls) == 2
    assert not cache.exists()
    parquet = output / "tiles" / "NG0000-0000A.parquet"
    frame = pd.read_parquet(parquet)
    assert frame.status.tolist() == ["classified", "classified"]
    assert frame.prob_B.tolist() == [0.75, 0.75]
    # Completed tile can regenerate a missing export without downloading again.
    parquet.unlink()
    assert classifier.process_tile(
        db, record, tmp_path, output, FakeModel(), config, args, None, None
    ) == (0, True)
    assert parquet.exists()
    db.close()


def test_bad_predictions_do_not_erase_features(tmp_path, config):
    db = classifier.checkpoint_db(tmp_path / "db.sqlite", config)
    payload = {"status": "features", "features": {"period": 1.0}}
    classifier.save_result(db, "tile", "source", payload)

    class BadModel:
        def predict(self, *args, **kwargs):
            return ["A"], [[np.nan, 1.0]]

    with pytest.raises(ValueError, match="invalid probabilities"):
        classifier.predict_pending(
            db, {"dp_id": "tile"}, [("source", payload)], BadModel(), config
        )
    assert (
        json.loads(db.execute("SELECT payload FROM results").fetchone()[0])["status"]
        == "features"
    )
    db.close()


def test_download_discards_partial_bytes_and_starts_over(tmp_path):
    from io import BytesIO

    import requests

    filename = "FLUX_NG0000-0000A.fits"
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = filename
    table = Table(lightcurve(2))
    table["SOURCE_ID"] = ["one", "one"]
    buffer = BytesIO()
    fits.HDUList([primary, fits.BinTableHDU(table)]).writeto(buffer, checksum=True)
    content = buffer.getvalue()
    record = {
        "filename": filename,
        "cache_path": f"tiles/{filename}",
        "download_url": "https://example.test/tile",
        "field": "NG0000-0000",
        "tile": "A",
    }

    class Response:
        def __init__(self, fail):
            self.status_code = 200
            self.headers = {"Content-Length": str(len(content))}
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            if self.fail:
                yield content[:1000]
                raise requests.ConnectionError("connection lost")
            yield content

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, headers, stream):
            self.calls.append(headers.copy())
            return Response(len(self.calls) == 1)

    session = Session()
    path = tmp_path / record["cache_path"]
    path.parent.mkdir(parents=True)
    path.with_name(path.name + ".part").write_bytes(b"stale partial")
    with pytest.raises(requests.ConnectionError):
        classifier.cache_tile(session, record, tmp_path)
    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()
    assert classifier.cache_tile(session, record, tmp_path) == path
    assert all("Range" not in headers for headers in session.calls)
    assert path.read_bytes() == content
    assert not path.with_name(path.name + ".part").exists()


def test_exclusive_run_prevents_two_writers(tmp_path):
    with (
        classifier.exclusive_run(tmp_path),
        pytest.raises(RuntimeError, match="Another classifier"),
        classifier.exclusive_run(tmp_path),
    ):
        pass


def test_outside_log_domain_does_not_poison_batch(tmp_path, config):
    db = classifier.checkpoint_db(tmp_path / "db.sqlite", config)
    model = FakeModel()
    model.min_values = {"min_period": 0.03}
    pending = [
        ("bad", {"status": "features", "features": {"period": 0.02}}),
        ("good", {"status": "features", "features": {"period": 0.5}}),
    ]
    classifier.predict_pending(db, {"dp_id": "tile"}, pending, model, config)
    assert not pending
    statuses = dict(db.execute("SELECT source_id, status FROM results"))
    assert statuses == {"bad": "skipped", "good": "classified"}
    db.close()
