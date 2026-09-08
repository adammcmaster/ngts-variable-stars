#!/usr/bin/env python3
"""Download NGTS DR2 source catalogues and inventory its photometry tiles."""

import argparse
import hashlib
import json
import re
import string
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import pyvo
import requests
from astropy.io import fits
from astropy.table import Table
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

TAP_URL = "https://archive.eso.org/tap_obs"
RELEASE_URL = "https://www.eso.org/rm/api/v1/public/releaseDescriptions/154"
QUERY = """
SELECT dp_id, obs_creator_did, access_url, access_estsize,
       s_ra, s_dec, t_min, t_max, release_description
FROM ivoa.ObsCore
WHERE obs_collection = 'NGTS'
  AND release_description LIKE '%/releaseDescriptions/154'
""".strip()
FIELD_PATTERN = r"NG\d{4}[+-]\d{4}"


class ArchiveSession(requests.Session):
    """Apply timeouts and retries to TAP queries as well as file downloads."""

    def __init__(self):
        super().__init__()
        retries = Retry(
            total=4,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        self.mount("https://", HTTPAdapter(max_retries=retries))
        self.headers["User-Agent"] = "ngts-variable-stars/0.1 (DR2 catalogue setup)"

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (30, 180))
        return super().request(method, url, **kwargs)


def build_manifests(products):
    """Validate the complete DR2 inventory before downloading any catalogues."""
    tiles, catalogues = [], []
    for product in products:
        if urlsplit(str(product["release_description"])).path != (
            "/rm/api/v1/public/releaseDescriptions/154"
        ):
            raise ValueError("Inventory contains a product outside DR2")
        filename = unquote(urlsplit(str(product["obs_creator_did"])).query)
        tile = re.fullmatch(rf"FLUX_({FIELD_PATTERN})([A-Y])\.fits", filename)
        catalogue = re.fullmatch(rf"SOURCE_CATALOGUE_({FIELD_PATTERN})\.fits", filename)
        if not tile and not catalogue:
            if re.fullmatch(rf"DITHERED_STACK_({FIELD_PATTERN})\.fits", filename):
                continue
            raise ValueError(f"Unexpected DR2 product: {filename}")
        dp_id = str(product["dp_id"])
        if not re.fullmatch(r"ADP\.[\dT:.\-]+", dp_id):
            raise ValueError(f"Unexpected dataset identifier: {dp_id}")
        field = (tile or catalogue).group(1)
        record = {
            "field": field,
            "filename": filename,
            "dp_id": dp_id,
            # ESO's file endpoint, as advertised by DataLink's #this record.
            "download_url": f"https://dataportal.eso.org/dataPortal/file/{dp_id}",
            "datalink_url": str(product["access_url"]).replace("http:", "https:", 1),
            "size_estimate_kb": int(product["access_estsize"]),
            "ra_deg": float(product["s_ra"]),
            "dec_deg": float(product["s_dec"]),
            "mjd_start": float(product["t_min"]),
            "mjd_end": float(product["t_max"]),
        }
        if tile:
            record.update(tile=tile.group(2), cache_path=f"tiles/{filename}")
            tiles.append(record)
        else:
            record["cache_path"] = f"catalogues/{filename}"
            catalogues.append(record)

    tiles = pd.DataFrame(tiles)
    catalogues = pd.DataFrame(catalogues)
    if len(tiles) != 1800 or len(catalogues) != 72:
        raise ValueError(
            f"Incomplete DR2 inventory: {len(tiles)} tiles, {len(catalogues)} catalogues; "
            "expected 1800 and 72"
        )
    all_products = pd.concat([tiles, catalogues])
    for column in ("filename", "dp_id"):
        if all_products[column].duplicated().any():
            raise ValueError(f"Duplicate {column} in DR2 inventory")
    if catalogues["field"].duplicated().any():
        raise ValueError("Duplicate source catalogue field")
    if set(tiles["field"]) != set(catalogues["field"]):
        raise ValueError("Source catalogue and tile fields do not match")
    for field, group in tiles.groupby("field"):
        if len(group) != 25 or set(group["tile"]) != set(string.ascii_uppercase[:25]):
            raise ValueError(f"Missing or duplicate tile letters for {field}")
    return (
        tiles.sort_values(["field", "tile"]).reset_index(drop=True),
        catalogues.sort_values("field").reset_index(drop=True),
    )


def read_catalogue(path, filename):
    """Check identity, FITS integrity and required columns, then read the table."""
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        if hdus[0].header.get("ORIGFILE") != filename:
            raise ValueError(f"Wrong catalogue identity in {path}")
        for hdu in hdus:
            if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
                raise ValueError(f"Invalid FITS checksum in {path}")
            if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
                raise ValueError(f"Invalid FITS data checksum in {path}")
        table = Table.read(hdus[1])
        if not {"SOURCE_ID", "RA_NGTS", "DEC_NGTS", "NGTS_MAG"} <= set(table.colnames):
            raise ValueError(f"Missing source catalogue columns in {path}")
        if not len(table):
            raise ValueError(f"Empty source catalogue: {path}")
    table.convert_bytestring_to_unicode()
    return table.to_pandas()


def download_catalogue(session, record, data_dir):
    path = data_dir / record["cache_path"]
    if path.exists():
        try:
            return read_catalogue(path, record["filename"])
        except (OSError, ValueError) as exc:
            tqdm.write(f"Replacing invalid cached catalogue {path.name}: {exc}")
    partial = path.with_suffix(".fits.part")
    for attempt in range(3):
        try:
            with session.get(record["download_url"], stream=True) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
                length = response.headers.get("Content-Length")
                if length and partial.stat().st_size != int(length):
                    raise ValueError(f"Truncated download: {path.name}")
            table = read_catalogue(partial, record["filename"])
            partial.replace(path)
            return table
        except (requests.RequestException, OSError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def write_atomic(path, writer):
    partial = path.with_name(path.name + ".part")
    try:
        writer(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def initialise(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("catalogues", "tiles"):
        (data_dir / directory).mkdir(exist_ok=True)

    with ArchiveSession() as session:
        print("Querying ESO for NGTS DR2 products...", flush=True)
        result = pyvo.dal.TAPService(TAP_URL, session=session).search(
            QUERY, maxrec=10000
        )
        if result.query_status != "OK":
            raise ValueError(f"Incomplete TAP response: {result.query_status}")
        products = result.to_table()
        tiles, catalogues = build_manifests(products)
        print(f"Found {len(tiles)} tiles and {len(catalogues)} source catalogues.")
        frames = []
        for index, record in tqdm(
            catalogues.iterrows(), total=len(catalogues), desc="Source catalogues"
        ):
            frame = download_catalogue(session, record, data_dir)
            # Keep field membership, including sources appearing in multiple fields.
            frame.insert(0, "FIELD", record["field"])
            frames.append(frame)
            path = data_dir / record["cache_path"]
            catalogues.loc[index, "source_count"] = len(frame)
            catalogues.loc[index, "size_bytes"] = path.stat().st_size
            catalogues.loc[index, "sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    sources = pd.concat(frames, ignore_index=True)
    catalogues = catalogues.astype({"source_count": "int64", "size_bytes": "int64"})
    write_atomic(data_dir / "source_catalogue.parquet", sources.to_parquet)
    write_atomic(
        data_dir / "source_catalogue_manifest.csv",
        lambda path: catalogues.to_csv(path, index=False),
    )
    write_atomic(
        data_dir / "tile_manifest.csv", lambda path: tiles.to_csv(path, index=False)
    )
    write_atomic(
        data_dir / "archive_products.ecsv",
        lambda path: products.write(path, format="ascii.ecsv", overwrite=True),
    )
    metadata = {
        "schema_version": 1,
        "release": "NGTS DR2",
        "release_description": RELEASE_URL,
        "created_utc": datetime.now(UTC).isoformat(),
        "tap_service": TAP_URL,
        "query": QUERY,
        "field_count": len(catalogues),
        "tile_count": len(tiles),
        "source_rows": len(sources),
        "unique_source_ids": int(sources["SOURCE_ID"].nunique()),
        "tile_size_estimate_kb": int(tiles["size_estimate_kb"].sum()),
        "cache_path_base": "Paths in manifests are relative to the data directory.",
    }
    write_atomic(
        data_dir / "manifest_metadata.json",
        lambda path: path.write_text(json.dumps(metadata, indent=2) + "\n"),
    )
    print(
        f"Saved {len(sources):,} source rows and {len(tiles):,} tiles under {data_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory (default: data/ beside this script; symlinks supported)",
    )
    args = parser.parse_args()
    initialise(args.data_dir)


if __name__ == "__main__":
    main()
