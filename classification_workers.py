"""Spawned source workers and a single-tile download lookahead."""

from itertools import islice
from multiprocessing import get_context
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic

from astropy.io import fits
from requests import RequestException
from tqdm import tqdm

WORKERS = 8


def check_workers(workers):
    """Pool replaces dead processes silently; fail instead of awaiting lost jobs."""
    for worker in workers:
        if worker.exitcode is not None:
            raise RuntimeError(
                f"Worker {worker.pid} exited unexpectedly (exit code {worker.exitcode})"
            )


def source_results(pool, worker, tasks):
    """Bound outstanding work and return completed sources without head blocking."""
    tasks = iter(tasks)
    completed = Queue()
    workers = tuple(pool._pool)

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
        try:
            success, result = completed.get(timeout=0.2)
        except Empty:
            check_workers(workers)
            continue
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


_download_events = None


def init_downloader(events):
    global _download_events
    _download_events = events


def report_download(kind, value):
    _download_events.put((kind, value))


class DownloadReporter:
    """Send byte counts to the parent; the download process never draws bars."""

    def __init__(self, total, initial, desc, **kwargs):
        self.n = initial
        self.last_update = monotonic()
        report_download("start", (total, initial, desc))

    def __enter__(self):
        return self

    def update(self, amount):
        self.n += amount
        if monotonic() - self.last_update >= 0.2:
            report_download("bytes", self.n)
            self.last_update = monotonic()

    def __exit__(self, *exc):
        report_download("bytes", self.n)


def download_tile(task):
    from classify_upsilont import cache_tile
    from init_manifest import ArchiveSession

    record, data_dir = task
    with ArchiveSession() as session:
        for attempt in range(1, 6):
            try:
                path = cache_tile(
                    session,
                    record,
                    data_dir,
                    progress_factory=DownloadReporter,
                    report=report_download,
                )
                break
            except RequestException:
                if attempt == 5:
                    raise
                report_download(
                    "status",
                    f"Connection timed out; retry {attempt + 1}/5 from the beginning",
                )
    report_download("status", f"Ready {record['field']}{record['tile']}")
    return path


class TilePrefetch:
    """One download process, with progress rendered only in the parent process."""

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
        self.events = get_context("spawn").Queue()
        self.stop_display = Event()
        self.pool = get_context("spawn").Pool(
            1,
            initializer=init_downloader,
            initargs=(self.events,),
        )
        self.workers = tuple(self.pool._pool)
        self.display = Thread(target=self._display, daemon=True)
        self.display.start()
        self._advance()
        return self

    def _display(self):
        with tqdm(
            total=None,
            desc="Starting download worker",
            unit="B",
            unit_scale=True,
            position=2,
            leave=False,
        ) as bar:
            while not self.stop_display.is_set():
                try:
                    kind, value = self.events.get(timeout=0.2)
                except Empty:
                    bar.refresh()
                    continue
                if kind == "start":
                    total, initial, desc = value
                    bar.reset(total=total)
                    bar.update(initial)
                    bar.set_description_str(desc)
                elif kind == "bytes":
                    bar.update(value - bar.n)
                elif kind == "status":
                    bar.set_description_str(value)
                elif kind == "message":
                    tqdm.write(value)

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
        from multiprocessing import TimeoutError

        if self.pending is None or self.pending[0] != record["dp_id"]:
            raise RuntimeError("Tile prefetch order differs from classification order")
        while True:
            try:
                path = self.pending[1].get(timeout=0.2)
                break
            except TimeoutError:
                check_workers(self.workers)
        self._advance()
        return path

    def __exit__(self, *exc):
        # Stop reading before terminating a queue writer: termination may leave a
        # partial queue message. No child can hold the parent's terminal lock.
        self.stop_display.set()
        self.display.join()
        self.pool.terminate()
        self.pool.join()
        self.events.close()
        self.events.join_thread()
