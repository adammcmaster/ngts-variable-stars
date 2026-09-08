#!/usr/bin/env python3
"""Classify NGTS DR2 tile by tile, with durable per-source checkpoints."""

import argparse
import fcntl
import hashlib
import json
import re
import sqlite3
import warnings
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from init_manifest import ArchiveSession, write_atomic

PIPELINE_VERSION = 1
REQUIRED_COLUMNS = {"SOURCE_ID", "HJD", "SYSFLUX", "FLUX_ERR", "FLAG"}


@contextmanager
def exclusive_run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another classifier is using this output directory"
            ) from exc
        yield


def checkpoint_db(path, config):
    db = sqlite3.connect(path)
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS results (
            dp_id TEXT NOT NULL, source_id TEXT NOT NULL, status TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY (dp_id, source_id));
        CREATE TABLE IF NOT EXISTS tiles (
            dp_id TEXT PRIMARY KEY, source_count INTEGER NOT NULL);
    """)
    encoded = json.dumps(config, sort_keys=True)
    saved = db.execute("SELECT value FROM settings WHERE key='config'").fetchone()
    if saved and saved[0] != encoded:
        db.close()
        raise ValueError(
            "Checkpoint configuration differs (data, model, software or preprocessing). "
            "Restore the previous settings, or move the existing upsilon-t output "
            "directory aside before starting a different run."
        )
    with db:
        db.execute("INSERT OR IGNORE INTO settings VALUES ('config', ?)", (encoded,))
    return db


def save_result(db, dp_id, source_id, payload):
    with db:
        db.execute(
            "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
            (dp_id, source_id, payload["status"], json.dumps(payload, allow_nan=False)),
        )


def validate_tile(path, filename):
    with fits.open(path, memmap=True, character_as_bytes=True) as hdus:
        hdus.verify("exception")
        if hdus[0].header.get("ORIGFILE") != filename:
            raise ValueError(f"Wrong tile identity: {path}")
        if not REQUIRED_COLUMNS <= set(hdus[1].columns.names):
            raise ValueError(f"Missing photometry columns: {path}")
        expected_end = 0
        for hdu in hdus:
            info = hdu.fileinfo()
            expected_end = max(expected_end, info["datLoc"] + info["datSpan"])
            if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
                raise ValueError(f"Invalid FITS checksum: {path}")
            if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
                raise ValueError(f"Invalid FITS data checksum: {path}")
        if path.stat().st_size != expected_end:
            raise ValueError(f"Truncated or oversized tile: {path}")


def cache_tile(session, record, data_dir):
    path = data_dir / record["cache_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            validate_tile(path, record["filename"])
            return path
        except (OSError, ValueError) as exc:
            tqdm.write(f"Replacing invalid cached tile {path.name}: {exc}")
            path.unlink()
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with session.get(record["download_url"], headers=headers, stream=True) as response:
        if response.status_code == 416 and offset:
            # A previous download may have finished just before interruption.
            try:
                validate_tile(partial, record["filename"])
            except (OSError, ValueError):
                partial.unlink()
                raise ValueError(
                    f"Invalid partial tile removed; rerun to download {path.name}"
                )
            partial.replace(path)
            return path
        response.raise_for_status()
        if response.status_code == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match or int(match[1]) != offset:
                raise ValueError("Server returned an inconsistent download range")
            total = int(match[3])
        else:
            # If Range is ignored, start over rather than appending a full file.
            offset = 0
            total = int(response.headers.get("Content-Length", 0)) or None
        with (
            partial.open("ab" if offset else "wb") as output,
            tqdm(
                total=total,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=f"Download {record['field']}{record['tile']}",
                leave=False,
            ) as progress,
        ):
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
        if total is not None and partial.stat().st_size != total:
            raise ValueError(f"Incomplete download retained for resumption: {partial}")
    try:
        validate_tile(partial, record["filename"])
    except (OSError, ValueError):
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return path


def source_groups(ids):
    """Return source row selectors without expanding FITS byte IDs to Unicode."""
    if not len(ids):
        return []
    starts = [0]
    for begin in range(1, len(ids), 1_000_000):
        end = min(begin + 1_000_000, len(ids))
        starts.extend(
            (
                np.flatnonzero(ids[begin:end] != ids[begin - 1 : end - 1]) + begin
            ).tolist()
        )
    starts = np.asarray(starts)
    names = ids[starts]
    if len(set(names.tolist())) != len(names):
        # Some products may interleave sources; sort a row index, never the FITS file.
        order = np.argsort(ids, kind="stable")
        return [
            (name, order[selection]) for name, selection in source_groups(ids[order])
        ]
    ends = np.r_[starts[1:], len(ids)]
    return [
        (
            name.decode("ascii").strip()
            if isinstance(name, bytes)
            else str(name).strip(),
            slice(int(a), int(b)),
        )
        for name, a, b in zip(names, starts, ends, strict=True)
    ]


class UnusableLightCurve(ValueError):
    """A documented data-quality exclusion rather than an execution failure."""


def prepare_lightcurve(rows, config):
    t = np.asarray(rows["HJD"], dtype=float)
    flux = np.asarray(rows["SYSFLUX"], dtype=float)
    error = np.asarray(rows["FLUX_ERR"], dtype=float)
    flags = np.asarray(rows["FLAG"], dtype=np.int64)
    good = (
        np.isfinite(t)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (flux > 0)
        & (error > 0)
        & ((flags & config["reject_flags"]) == 0)
    )
    counts = {"n_raw": len(t), "n_valid": int(good.sum()), "n_used": 0}
    t, flux, error = t[good], flux[good], error[good]
    if not len(t):
        return None, counts, "no_valid_photometry"
    order = np.argsort(t, kind="stable")
    t, flux, error = t[order], flux[order], error[order]
    # Subtract the first HJD to improve numerical conditioning; retain day units.
    t = t - t[0]
    width = config["bin_minutes"] / 1440
    keys = np.floor(t / width).astype(np.int64) if width else t
    # Native mode still merges duplicate timestamps to avoid zero time differences.
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    # Relative weights avoid overflow from squaring very small uncertainties.
    weights = (error.min() / error) ** 2
    sums = np.add.reduceat(weights, starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.add.reduceat(weights * t, starts) / sums
        flux = np.add.reduceat(weights * flux, starts) / sums
        error = error.min() / np.sqrt(sums)
        mag = -2.5 * np.log10(flux / np.median(flux))
        mag_error = (2.5 / np.log(10)) * error / flux
    good = np.isfinite(t) & np.isfinite(mag) & np.isfinite(mag_error) & (mag_error > 0)
    t, mag, mag_error = t[good], mag[good], mag_error[good]
    counts["n_used"] = len(t)
    if len(t) < config["min_points"]:
        return None, counts, "too_few_points"
    if np.ptp(t) <= 0 or np.ptp(mag) <= 1e-12:
        return None, counts, "constant_or_zero_baseline"
    return (t, mag, mag_error), counts, None


def extract_source(rows, config, fft_threads):
    from upsilont.features import VariabilityFeatures

    prepared, counts, reason = prepare_lightcurve(rows, config)
    payload = {
        **counts,
        "status": "skipped" if reason else "features",
        "reason": reason,
    }
    if reason:
        return payload
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            extracted = VariabilityFeatures(
                *prepared, n_threads=fft_threads, min_period=config["min_period"]
            ).get_features()
        features = {name: float(value) for name, value in extracted.items()}
        if not all(np.isfinite(list(features.values()))):
            raise UnusableLightCurve("nonfinite_features")
        payload["features"] = features
        payload["warnings"] = sorted({str(w.message) for w in caught})
    except UnusableLightCurve as exc:
        payload.update(status="skipped", reason=str(exc))
    except (ValueError, ArithmeticError) as exc:
        # Record numerical/data errors explicitly; --retry-errors retries these.
        payload.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    return payload


def predict_pending(db, record, pending, model, config):
    if not pending:
        return
    supported = []
    for source_id, payload in pending:
        # Upstream applies log10(feature - training_min), allowing equality via
        # an epsilon but producing NaN below the bound. Do not invent a class
        # by silently clipping these out-of-domain observations.
        below = {
            key.removeprefix("min_"): float(bound)
            for key, bound in getattr(model, "min_values", {}).items()
            if payload["features"][key.removeprefix("min_")] < bound
        }
        if below:
            payload.update(
                status="skipped",
                reason="outside_model_log_domain",
                domain_bounds=below,
            )
            save_result(db, record["dp_id"], source_id, payload)
        else:
            supported.append((source_id, payload))
    pending[:] = supported
    if not pending:
        return
    features = pd.DataFrame([payload["features"] for _, payload in pending])
    features = features[config["feature_names"]]
    labels, probabilities = model.predict(features, return_prob=True)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != (len(pending), len(config["classes"]))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError(
            "Model returned invalid probabilities; feature checkpoints retained"
        )
    if len(labels) != len(pending):
        raise ValueError("Model returned wrong number of labels")
    for (source_id, payload), label, probability in zip(
        pending, labels, probabilities, strict=True
    ):
        if str(label) not in config["classes"]:
            raise ValueError(f"Unknown classifier label: {label}")
        payload.update(
            status="classified",
            label=str(label),
            probability=float(probability.max()),
            probabilities=dict(
                zip(config["classes"], probability.tolist(), strict=True)
            ),
        )
        save_result(db, record["dp_id"], source_id, payload)
    pending.clear()


def export_tile(db, record, output, config):
    rows = []
    for source_id, payload in db.execute(
        "SELECT source_id, payload FROM results WHERE dp_id=? ORDER BY source_id",
        (record["dp_id"],),
    ):
        payload = json.loads(payload)
        features = payload.pop("features", {})
        probabilities = payload.pop("probabilities", {})
        payload["warnings"] = json.dumps(payload.get("warnings", []))
        payload["domain_bounds"] = json.dumps(payload.get("domain_bounds", {}))
        rows.append(
            {
                "source_id": source_id,
                "field": record["field"],
                "tile": record["tile"],
                "dp_id": record["dp_id"],
                **payload,
                **{
                    f"feature_{name}": features.get(name, np.nan)
                    for name in config["feature_names"]
                },
                **{
                    f"prob_{name}": probabilities.get(name, np.nan)
                    for name in config["classes"]
                },
            }
        )
    columns = [
        "source_id",
        "field",
        "tile",
        "dp_id",
        "status",
        "reason",
        "n_raw",
        "n_valid",
        "n_used",
        "label",
        "probability",
        "warnings",
        "domain_bounds",
        *[f"feature_{name}" for name in config["feature_names"]],
        *[f"prob_{name}" for name in config["classes"]],
    ]
    frame = pd.DataFrame(rows).reindex(columns=columns)
    # Fixed nullable dtypes also allow reading a directory containing all-skipped tiles.
    text_columns = [
        "source_id",
        "field",
        "tile",
        "dp_id",
        "status",
        "reason",
        "label",
        "warnings",
        "domain_bounds",
    ]
    frame = frame.astype({name: "string" for name in text_columns})
    frame = frame.astype({name: "int64" for name in ("n_raw", "n_valid", "n_used")})
    numeric = [
        "probability",
        *[name for name in columns if name.startswith(("feature_", "prob_"))],
    ]
    frame = frame.astype({name: "float64" for name in numeric})
    target = output / "tiles" / f"{record['field']}{record['tile']}.parquet"
    write_atomic(target, lambda path: frame.to_parquet(path, index=False))
    return len(frame)


def process_tile(db, record, data_dir, output, model, config, args, session, budget):
    if db.execute("SELECT 1 FROM tiles WHERE dp_id=?", (record["dp_id"],)).fetchone():
        export_tile(db, record, output, config)
        if not args.keep_tiles:
            (data_dir / record["cache_path"]).unlink(missing_ok=True)
        return 0, True
    path = cache_tile(session, record, data_dir)
    saved = {
        source_id: json.loads(payload)
        for source_id, payload in db.execute(
            "SELECT source_id, payload FROM results WHERE dp_id=?", (record["dp_id"],)
        )
    }
    processed = 0
    pending = []
    with fits.open(path, memmap=True, character_as_bytes=True) as hdus:
        data = hdus[1].data
        groups = source_groups(data["SOURCE_ID"])
        ids = {source_id for source_id, _ in groups}
        if not set(saved) <= ids:
            raise ValueError("Checkpoint sources do not match the cached tile")
        done = sum(item["status"] != "features" for item in saved.values())
        with tqdm(
            total=len(groups),
            initial=done,
            desc=f"Sources {record['field']}{record['tile']}",
            leave=False,
        ) as progress:
            for source_id, selection in groups:
                if source_id in saved and saved[source_id]["status"] != "features":
                    continue
                if budget is not None and processed >= budget:
                    break
                payload = saved.get(source_id)
                if payload is None:
                    payload = extract_source(data[selection], config, args.fft_threads)
                    save_result(db, record["dp_id"], source_id, payload)
                if payload["status"] == "features":
                    pending.append((source_id, payload))
                    if len(pending) >= args.batch_size:
                        predict_pending(db, record, pending, model, config)
                processed += 1
                progress.update(1)
            predict_pending(db, record, pending, model, config)
        # Release all memory-mapped arrays before the cache file is unlinked.
        del data
    total, waiting = db.execute(
        "SELECT COUNT(*), SUM(status='features') FROM results WHERE dp_id=?",
        (record["dp_id"],),
    ).fetchone()
    complete = total == len(groups) and not waiting
    if complete:
        export_tile(db, record, output, config)
        with db:
            db.execute(
                "INSERT OR REPLACE INTO tiles VALUES (?, ?)", (record["dp_id"], total)
            )
        if not args.keep_tiles:
            path.unlink()
    return processed, complete


def run(args):
    import torch
    import upsilont
    from upsilont.features import get_train_feature_name

    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "tile_manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"dp_id", "field", "tile", "filename", "cache_path", "download_url"}
    if not required <= set(manifest.columns) or manifest.dp_id.duplicated().any():
        raise ValueError("Invalid tile manifest; rerun init_manifest.py")
    if manifest.empty or manifest[["field", "tile"]].duplicated().any():
        raise ValueError("Empty manifest or duplicate field/tile")
    for row in manifest.to_dict("records"):
        if not re.fullmatch(r"NG\d{4}[+-]\d{4}", row["field"]) or row[
            "tile"
        ] not in list("ABCDEFGHIJKLMNOPQRSTUVWXY"):
            raise ValueError("Invalid field/tile identifier")
        expected = f"FLUX_{row['field']}{row['tile']}.fits"
        if row["filename"] != expected or row["cache_path"] != f"tiles/{expected}":
            raise ValueError("Unsafe or inconsistent tile cache path")
    if not torch.cuda.is_available():
        raise RuntimeError("UPSILoN-T requires CUDA; run with access to the host GPU")
    torch.cuda.set_device(args.device)
    model = upsilont.UPSILoNT(device=args.device)
    model.load()
    model_dir = Path(upsilont.__file__).parent / "model"
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_dir.iterdir())
            if p.suffix in (".pt", ".pkl")
        },
        "versions": {
            name: version(name)
            for name in (
                "upsilont",
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "torch",
                "pyfftw",
            )
        },
        "bin_minutes": args.bin_minutes,
        "reject_flags": args.reject_flags,
        "min_points": args.min_points,
        "min_period": args.min_period,
        "feature_names": get_train_feature_name(),
        "classes": model.label_encoder.classes_.tolist(),
    }
    output = data_dir / "classifications" / "upsilon-t"
    with exclusive_run(output):
        (output / "tiles").mkdir(exist_ok=True)
        db = checkpoint_db(output / "checkpoints.sqlite", config)
        try:
            if args.retry_errors:
                with db:
                    affected = [
                        row[0]
                        for row in db.execute(
                            "SELECT DISTINCT dp_id FROM results WHERE status='error'"
                        )
                    ]
                    db.executemany(
                        "DELETE FROM tiles WHERE dp_id=?", [(x,) for x in affected]
                    )
                    db.execute("DELETE FROM results WHERE status='error'")
                for row in manifest.to_dict("records"):
                    if row["dp_id"] in affected:
                        (
                            output / "tiles" / f"{row['field']}{row['tile']}.parquet"
                        ).unlink(missing_ok=True)
            write_atomic(
                output / "run_config.json",
                lambda p: p.write_text(json.dumps(config, indent=2) + "\n"),
            )
            consumed = 0
            completed_this_run = 0
            done = {row[0] for row in db.execute("SELECT dp_id FROM tiles")}
            with (
                ArchiveSession() as session,
                tqdm(total=len(manifest), initial=len(done), desc="Tiles") as progress,
            ):
                for record in manifest.to_dict("records"):
                    was_done = record["dp_id"] in done
                    if (
                        not was_done
                        and args.max_tiles
                        and completed_this_run >= args.max_tiles
                    ):
                        break
                    budget = (
                        None
                        if args.max_sources is None
                        else args.max_sources - consumed
                    )
                    if not was_done and budget is not None and budget <= 0:
                        break
                    count, complete = process_tile(
                        db,
                        record,
                        data_dir,
                        output,
                        model,
                        config,
                        args,
                        session,
                        budget,
                    )
                    consumed += count
                    if complete and not was_done:
                        progress.update(1)
                        completed_this_run += 1
                    if not complete:
                        break
            summary = dict(
                db.execute("SELECT status, COUNT(*) FROM results GROUP BY status")
            )
            summary["completed_tiles"] = db.execute(
                "SELECT COUNT(*) FROM tiles"
            ).fetchone()[0]
            summary["total_tiles"] = len(manifest)
            summary["updated_utc"] = datetime.now(UTC).isoformat()
            write_atomic(
                output / "summary.json",
                lambda p: p.write_text(json.dumps(summary, indent=2) + "\n"),
            )
            tqdm.write(json.dumps(summary))
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--bin-minutes",
        type=float,
        default=5,
        help="Inverse-variance flux bins; 0 keeps native cadence",
    )
    parser.add_argument(
        "--reject-flags",
        type=lambda x: int(x, 0),
        default=23,
        help="Reject bitmask (default 23: bits 0,1,2,4; keep outlier-only points)",
    )
    parser.add_argument(
        "--min-points", type=int, default=80, help="Minimum usable points after binning"
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=0.03,
        help="UPSILoN-T min_period parameter, in days",
    )
    parser.add_argument("--fft-threads", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="GPU prediction batch size; features checkpointed individually",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--keep-tiles",
        action="store_true",
        help="Keep completed tile FITS files instead of deleting them",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry recorded per-source numerical errors",
    )
    parser.add_argument(
        "--max-tiles", type=int, help="Stop after this many newly completed tiles"
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        help="Stop after this many unfinished sources (for smoke tests)",
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.bin_minutes)
        or args.bin_minutes < 0
        or not np.isfinite(args.min_period)
        or args.min_period <= 0
    ):
        parser.error(
            "bin-minutes must be finite and nonnegative; min-period must be positive"
        )
    if args.min_points < 80 or not 0 <= args.reject_flags <= 255:
        parser.error("min-points must be >=80 and reject-flags must be in 0..255")
    for name in ("fft_threads", "batch_size", "max_tiles", "max_sources"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    run(args)


if __name__ == "__main__":
    main()
