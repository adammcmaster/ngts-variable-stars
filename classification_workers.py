"""Spawned source workers and a single-tile download lookahead."""

from itertools import islice
from multiprocessing import get_context
from queue import Queue

from astropy.io import fits

WORKERS = 8


def source_results(pool, worker, tasks):
    """Bound outstanding work and return completed sources without head blocking."""
    tasks = iter(tasks)
    completed = Queue()

    def submit(task):
        pool.apply_async(
            worker,
            (task,),
            callback=lambda result: completed.put((True, result)),
            error_callback=lambda error: completed.put((False, error)),
        )

    outstanding = 0
    for task in islice(tasks, 2 * WORKERS):
        submit(task)
        outstanding += 1
    while outstanding:
        success, result = completed.get()
        outstanding -= 1
        if not success:
            raise result
        for task in islice(tasks, 1):
            submit(task)
            outstanding += 1
        yield result


def upsilon_source(task):
    from classify_upsilont import extract_source

    path, source_id, selection, config, fft_threads = task
    with fits.open(path, memmap=True, character_as_bytes=True) as hdus:
        payload = extract_source(hdus[1].data[selection], config, fft_threads)
    return source_id, payload


def norton_source(task):
    import sqlite3

    from classify_norton import analyse_source

    path, source_id, selection, config, record, db_path, previous = task
    # Each process owns its connection; short transactions serialize checkpoints.
    db = sqlite3.connect(db_path, timeout=60)
    db.execute("PRAGMA synchronous=FULL")
    try:
        with fits.open(path, memmap=True, character_as_bytes=True) as hdus:
            analyse_source(
                db,
                record,
                source_id,
                hdus[1].data[selection],
                config,
                previous,
                show_progress=False,
            )
    finally:
        db.close()
    return source_id


def download_tile(task):
    from classify_upsilont import cache_tile
    from init_manifest import ArchiveSession

    record, data_dir = task
    with ArchiveSession() as session:
        return cache_tile(session, record, data_dir)


class TilePrefetch:
    """Only current and next unfinished tiles are submitted to the downloader."""

    def __init__(self, records, done, data_dir, max_tiles=None):
        self.records = iter(
            islice(
                (record for record in records if record["dp_id"] not in done), max_tiles
            )
        )
        self.data_dir = data_dir
        self.pool = None
        self.pending = None

    def __enter__(self):
        self.pool = get_context("spawn").Pool(1)
        self._advance()
        return self

    def _advance(self):
        record = next(self.records, None)
        self.pending = (
            (
                record["dp_id"],
                self.pool.apply_async(download_tile, ((record, self.data_dir),)),
            )
            if record is not None
            else None
        )

    def take(self, record):
        if self.pending is None or self.pending[0] != record["dp_id"]:
            raise RuntimeError("Tile prefetch order differs from classification order")
        path = self.pending[1].get()
        self._advance()
        return path

    def __exit__(self, *exc):
        # A speculative download may be unfinished; retain its resumable .part file.
        self.pool.terminate()
        self.pool.join()
