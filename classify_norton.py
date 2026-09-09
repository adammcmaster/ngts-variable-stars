# Based on Andrew Norton's work developed for SuperWASP Variable Stars.
# Cite Norton (2018), DOI: 10.3847/2515-5172/aaf291.
"""Run the Python port of Norton's CLEAN + phase-folding detector on NGTS tiles."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

import norton_algorithm as norton
from classification_workers import (
    WORKERS,
    TilePrefetch,
    norton_source,
    source_results,
)
from classify_upsilont import (
    cache_tile,
    checkpoint_db,
    exclusive_run,
    save_result,
    source_groups,
)
from init_manifest import write_atomic

DOI = "10.3847/2515-5172/aaf291"


def prepare_flux(rows, config):
    """NGTS quality filtering and optional flux bins; Norton operates on flux."""
    t, flux, error = (
        np.asarray(rows[name], dtype=float) for name in ("HJD", "SYSFLUX", "FLUX_ERR")
    )
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
    # Subtract the HJD offset in float64 BEFORE converting days to seconds.
    t = (t - t[0]) * 86400
    width = config["bin_minutes"] * 60
    keys = np.floor(t / width).astype(np.int64) if width else t
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    weights = (error.min() / error) ** 2
    sums = np.add.reduceat(weights, starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.add.reduceat(t * weights, starts) / sums
        flux = np.add.reduceat(flux * weights, starts) / sums
        error = error.min() / np.sqrt(sums)
    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    t, flux, error = t[good], flux[good], error[good]
    counts["n_used"] = len(t)
    if len(t) < config["min_points"]:
        return None, counts, "too_few_points"
    if np.ptp(t) <= 0 or np.ptp(flux) <= 0:
        return None, counts, "constant_or_zero_baseline"
    return (t, flux, error), counts, None


def analyse_source(db, record, source_id, rows, config, saved=None, show_progress=True):
    prepared, counts, reason = prepare_flux(rows, config)
    if reason:
        payload = {**counts, "status": "skipped", "reason": reason, "periods": []}
        save_result(db, record["dp_id"], source_id, payload)
        return payload
    t, flux, error = norton.clip_spikes(*prepared, config["algorithm"])
    if len(t) < 3 or np.ptp(t) <= 0 or np.ptp(flux) <= 0:
        payload = {
            **counts,
            "n_clean": len(t),
            "status": "skipped",
            "reason": "insufficient_data_after_spike_removal",
            "periods": [],
        }
        save_result(db, record["dp_id"], source_id, payload)
        return payload
    try:
        if saved is None:
            frequency, power, diagnostics = norton.clean_spectrum(
                t, flux, config["algorithm"]
            )
            candidates, background = norton.spectral_candidates(
                frequency, power, config["algorithm"]
            )
            payload = {
                **counts,
                "n_clean": len(t),
                "status": "candidates",
                "reason": None,
                "candidates": candidates,
                "next_candidate": 0,
                "periods": [],
                "diagnostics": {
                    **diagnostics,
                    **background,
                    "n_clean_peaks": len(candidates),
                },
            }
            save_result(db, record["dp_id"], source_id, payload)
        else:
            payload = saved
        with tqdm(
            total=len(payload["candidates"]),
            initial=payload["next_candidate"],
            desc=f"Refine {source_id}",
            disable=not show_progress,
            leave=False,
        ) as progress:
            for i in range(payload["next_candidate"], len(payload["candidates"])):
                periods = norton.refine_candidates(
                    t,
                    flux,
                    error,
                    [payload["candidates"][i]],
                    config["algorithm"],
                    prepared,
                )
                payload["periods"].extend(periods)
                payload["next_candidate"] = i + 1
                save_result(db, record["dp_id"], source_id, payload)
                progress.update(1)
        payload["periods"].sort(key=lambda item: item["period_seconds"], reverse=True)
        payload["diagnostics"]["accepted_periods_before_limit"] = len(
            payload["periods"]
        )
        payload["periods"] = payload["periods"][: config["algorithm"]["max_periods"]]
        for rank, candidate in enumerate(payload["periods"], start=1):
            candidate["period_number"] = rank
        good = [p for p in payload["periods"] if p["period_flag"] == 0]
        label = (
            "periodic_candidate"
            if good
            else "alias_only"
            if payload["periods"]
            else "no_period"
        )
        best = max(
            good or payload["periods"],
            key=lambda p: p["chi_squared_ratio"],
            default=None,
        )
        payload.update(
            status="classified",
            label=label,
            best_period=best,
            n_periods=len(payload["periods"]),
            n_good_periods=len(good),
        )
        payload.pop("candidates", None)
        payload.pop("next_candidate", None)
    except (ArithmeticError, ValueError) as exc:
        payload = {
            **counts,
            "n_clean": len(t),
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "periods": [],
        }
    save_result(db, record["dp_id"], source_id, payload)
    return payload


def export_tile(db, record, output):
    sources, periods = [], []
    for source_id, encoded in db.execute(
        "SELECT source_id, payload FROM results WHERE dp_id=? ORDER BY source_id",
        (record["dp_id"],),
    ):
        payload = json.loads(encoded)
        common = {
            "source_id": source_id,
            "field": record["field"],
            "tile": record["tile"],
            "dp_id": record["dp_id"],
        }
        best = payload.get("best_period") or {}
        sources.append(
            {
                **common,
                "status": payload["status"],
                "reason": payload.get("reason"),
                **{
                    key: payload.get(key, 0)
                    for key in (
                        "n_raw",
                        "n_valid",
                        "n_used",
                        "n_clean",
                        "n_periods",
                        "n_good_periods",
                    )
                },
                "label": payload.get("label"),
                "probability": np.nan,
                **{
                    key: best.get(key)
                    for key in (
                        "period_days",
                        "period_seconds",
                        "sigma",
                        "chi_squared_ratio",
                        "period_flag",
                    )
                },
                "diagnostics": json.dumps(payload.get("diagnostics", {})),
            }
        )
        for candidate in payload.get("periods", []):
            periods.append(
                {
                    **common,
                    **{k: v for k, v in candidate.items() if k != "folded_profile"},
                    "folded_profile": json.dumps(candidate["folded_profile"]),
                }
            )
    common_columns = ["source_id", "field", "tile", "dp_id"]
    candidate_columns = [
        "period_number",
        "clean_period_seconds",
        "period_seconds",
        "period_days",
        "sigma",
        "chi_squared_ratio",
        "period_flag",
    ]
    source_columns = [
        *common_columns,
        "status",
        "reason",
        "n_raw",
        "n_valid",
        "n_used",
        "n_clean",
        "n_periods",
        "n_good_periods",
        "label",
        "probability",
        "period_days",
        "period_seconds",
        "sigma",
        "chi_squared_ratio",
        "period_flag",
        "diagnostics",
    ]
    source_frame = pd.DataFrame(sources).reindex(columns=source_columns)
    period_frame = pd.DataFrame(periods).reindex(
        columns=[*common_columns, *candidate_columns, "folded_profile"]
    )
    for frame, strings, integers in [
        (
            source_frame,
            [*common_columns, "status", "reason", "label", "diagnostics"],
            [
                "n_raw",
                "n_valid",
                "n_used",
                "n_clean",
                "n_periods",
                "n_good_periods",
                "period_flag",
            ],
        ),
        (
            period_frame,
            [*common_columns, "folded_profile"],
            ["period_number", "period_flag"],
        ),
    ]:
        for column in frame.columns:
            frame[column] = frame[column].astype(
                "string"
                if column in strings
                else "Int64"
                if column in integers
                else "float64"
            )
    stem = f"{record['field']}{record['tile']}.parquet"
    write_atomic(
        output / "periods" / stem, lambda p: period_frame.to_parquet(p, index=False)
    )
    write_atomic(
        output / "tiles" / stem, lambda p: source_frame.to_parquet(p, index=False)
    )


def process_tile(
    db,
    record,
    data_dir,
    output,
    config,
    args,
    session,
    budget,
    pool=None,
    tile_path=None,
):
    # Separate cache ownership permits simultaneous UPSILoN-T/Norton runs.
    record = {**record, "cache_path": f"tiles/norton/{record['filename']}"}
    if db.execute("SELECT 1 FROM tiles WHERE dp_id=?", (record["dp_id"],)).fetchone():
        if not all(
            (output / directory / f"{record['field']}{record['tile']}.parquet").exists()
            for directory in ("tiles", "periods")
        ):
            export_tile(db, record, output)
        if not args.keep_tiles:
            (data_dir / record["cache_path"]).unlink(missing_ok=True)
        return 0, True
    path = tile_path if tile_path is not None else cache_tile(session, record, data_dir)
    saved = {
        source_id: json.loads(payload)
        for source_id, payload in db.execute(
            "SELECT source_id, payload FROM results WHERE dp_id=?", (record["dp_id"],)
        )
    }
    processed = 0
    with fits.open(path, memmap=True, character_as_bytes=True) as hdus:
        data = hdus[1].data
        groups = source_groups(data["SOURCE_ID"])
        if not set(saved) <= {source_id for source_id, _ in groups}:
            raise ValueError("Checkpoint sources do not match the cached tile")
        done = sum(item["status"] != "candidates" for item in saved.values())
        with tqdm(
            total=len(groups),
            initial=done,
            desc=f"Sources {record['field']}{record['tile']}",
            leave=False,
        ) as progress:
            unfinished = [
                (source_id, selection)
                for source_id, selection in groups
                if source_id not in saved or saved[source_id]["status"] == "candidates"
            ][:budget]
            if pool is None:
                for source_id, selection in unfinished:
                    analyse_source(
                        db,
                        record,
                        source_id,
                        data[selection],
                        config,
                        saved.get(source_id),
                    )
                    processed += 1
                    progress.update(1)
            else:
                db_path = db.execute("PRAGMA database_list").fetchone()[2]
                tasks = (
                    (
                        path,
                        source_id,
                        selection,
                        config,
                        record,
                        db_path,
                        saved.get(source_id),
                    )
                    for source_id, selection in unfinished
                )
                for _ in source_results(pool, norton_source, tasks):
                    processed += 1
                    progress.update(1)
        del data
    total, waiting = db.execute(
        "SELECT COUNT(*), SUM(status='candidates') FROM results WHERE dp_id=?",
        (record["dp_id"],),
    ).fetchone()
    complete = total == len(groups) and not waiting
    if complete:
        export_tile(db, record, output)
        with db:
            db.execute(
                "INSERT OR REPLACE INTO tiles VALUES (?, ?)", (record["dp_id"], total)
            )
        if not args.keep_tiles:
            path.unlink()
    return processed, complete


def run(args):
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "tile_manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"dp_id", "field", "tile", "filename", "cache_path", "download_url"}
    if (
        not required <= set(manifest)
        or manifest.empty
        or manifest.dp_id.duplicated().any()
        or manifest[["field", "tile"]].duplicated().any()
    ):
        raise ValueError("Invalid tile manifest; rerun init_manifest.py")

    for record in manifest.to_dict("records"):
        expected = f"FLUX_{record['field']}{record['tile']}.fits"
        if (
            not re.fullmatch(r"NG\d{4}[+-]\d{4}", record["field"])
            or record["tile"] not in list("ABCDEFGHIJKLMNOPQRSTUVWXY")
            or record["filename"] != expected
            or record["cache_path"] != f"tiles/{expected}"
        ):
            raise ValueError("Unsafe or inconsistent manifest tile path")
    config = {
        "pipeline_version": 1,
        "algorithm_name": "Norton CLEAN + phase folding",
        "doi": DOI,
        "algorithm": norton.DEFAULTS,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "astropy", "numba")
        },
        "bin_minutes": args.bin_minutes,
        "reject_flags": args.reject_flags,
        "min_points": args.min_points,
    }
    output = data_dir / "classifications" / "norton"
    with exclusive_run(output):
        for directory in ("tiles", "periods"):
            (output / directory).mkdir(exist_ok=True)
        db = checkpoint_db(output / "checkpoints.sqlite", config)
        try:
            if args.retry_errors:
                with db:
                    affected = {
                        row[0]
                        for row in db.execute(
                            "SELECT DISTINCT dp_id FROM results WHERE status='error'"
                        )
                    }
                    db.executemany(
                        "DELETE FROM tiles WHERE dp_id=?", [(x,) for x in affected]
                    )
                    db.execute("DELETE FROM results WHERE status='error'")
                for row in manifest.to_dict("records"):
                    if row["dp_id"] in affected:
                        for directory in ("tiles", "periods"):
                            (
                                output
                                / directory
                                / f"{row['field']}{row['tile']}.parquet"
                            ).unlink(missing_ok=True)
            write_atomic(
                output / "run_config.json",
                lambda p: p.write_text(json.dumps(config, indent=2) + "\n"),
            )
            done = {row[0] for row in db.execute("SELECT dp_id FROM tiles")}
            consumed, newly_complete = 0, 0
            records = manifest.to_dict("records")
            records = [
                {**r, "cache_path": f"tiles/norton/{r['filename']}"} for r in records
            ]
            with (
                get_context("spawn").Pool(WORKERS) as pool,
                TilePrefetch(records, done, data_dir, args.max_tiles) as prefetch,
                tqdm(total=len(manifest), initial=len(done), desc="Tiles") as progress,
            ):
                for record in records:
                    was_done = record["dp_id"] in done
                    if (
                        not was_done
                        and args.max_tiles
                        and newly_complete >= args.max_tiles
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
                        config,
                        args,
                        None,
                        budget,
                        pool=pool,
                        tile_path=None if was_done else prefetch.take(record),
                    )
                    consumed += count
                    if complete and not was_done:
                        progress.update(1)
                        newly_complete += 1
                    if not complete:
                        break
            summary = dict(
                db.execute("SELECT status, COUNT(*) FROM results GROUP BY status")
            )
            summary.update(
                completed_tiles=db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0],
                total_tiles=len(manifest),
                updated_utc=datetime.now(UTC).isoformat(),
            )
            summary["labels"] = dict(
                db.execute(
                    "SELECT json_extract(payload, '$.label'), COUNT(*) FROM results WHERE status='classified' GROUP BY json_extract(payload, '$.label')"
                )
            )
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
        help="NGTS flux bin width, or 0 for native cadence",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1000,
        help="Minimum points after NGTS binning, before Norton clipping",
    )
    parser.add_argument("--reject-flags", type=lambda x: int(x, 0), default=23)
    parser.add_argument("--keep-tiles", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()
    if (
        not np.isfinite(args.bin_minutes)
        or args.bin_minutes < 0
        or args.min_points < 1000
        or not 0 <= args.reject_flags <= 255
    ):
        parser.error(
            "Require finite bin-minutes >=0, min-points >=1000, and reject-flags in 0..255"
        )
    for name in ("max_tiles", "max_sources"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    run(args)


if __name__ == "__main__":
    main()
