"""Guard against incomplete inventories and corrupt/repeated catalogue downloads."""

from pathlib import Path

import pytest
from astropy.io import fits
from astropy.table import Table

from init_manifest import (
    RELEASE_URL,
    build_manifests,
    download_catalogue,
    read_catalogue,
    write_atomic,
)


@pytest.fixture
def products():
    rows = []
    for index in range(72):
        field = f"NG{index:04d}-0000"
        names = [f"SOURCE_CATALOGUE_{field}.fits"] + [
            f"FLUX_{field}{letter}.fits" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXY"
        ]
        for name in names:
            rows.append(
                {
                    "release_description": RELEASE_URL,
                    "obs_creator_did": f"ivo://eso.org/origfile?{name}",
                    "dp_id": f"ADP.{len(rows)}",
                    "access_url": "http://archive.eso.org/datalink/links",
                    "access_estsize": 100,
                    "s_ra": 0,
                    "s_dec": 0,
                    "t_min": 57000,
                    "t_max": 58000,
                }
            )
    return rows


def test_complete_manifest_has_relative_cache_paths(products):
    tiles, catalogues = build_manifests(products)
    assert len(tiles) == 1800
    assert len(catalogues) == 72
    assert tiles.iloc[0].cache_path == "tiles/FLUX_NG0000-0000A.fits"
    assert not any(Path(path).is_absolute() for path in tiles.cache_path)


def test_rejects_truncated_inventory(products):
    with pytest.raises(ValueError, match="Incomplete"):
        build_manifests(products[:-1])


def test_rejects_duplicate_replacing_missing_tile(products):
    products[-1] = products[-2].copy()
    with pytest.raises(ValueError, match="Duplicate"):
        build_manifests(products)


def test_rejects_wrong_release(products):
    products[0]["release_description"] = RELEASE_URL.replace("154", "122")
    with pytest.raises(ValueError, match="outside DR2"):
        build_manifests(products)


def test_catalogue_reuse_and_identity(tmp_path):
    filename = "SOURCE_CATALOGUE_NG0000-0000.fits"
    path = tmp_path / filename
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = filename
    table = Table(
        {
            "SOURCE_ID": ["NGTSJ000000.0-000000"],
            "RA_NGTS": [0.0],
            "DEC_NGTS": [0.0],
            "NGTS_MAG": [12.0],
        }
    )
    fits.HDUList([primary, fits.BinTableHDU(table)]).writeto(path, checksum=True)
    record = {"cache_path": filename, "filename": filename}
    # None has no get() method: a validated cache hit must not use the network.
    frame = download_catalogue(None, record, tmp_path)
    assert frame.SOURCE_ID.iloc[0] == "NGTSJ000000.0-000000"
    with pytest.raises(ValueError, match="identity"):
        read_catalogue(path, "wrong.fits")
    # Corrupt bytes directly; Astropy's update mode repairs checksums on close.
    content = path.read_bytes()
    path.write_bytes(content.replace(b"NGTSJ000000.0-000000", b"NGTSJ100000.0-000000"))
    with pytest.raises(ValueError, match="checksum"):
        read_catalogue(path, filename)


def test_failed_atomic_write_preserves_previous_file(tmp_path):
    path = tmp_path / "tile_manifest.csv"
    path.write_text("previous")

    def fail(partial):
        partial.write_text("incomplete")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_atomic(path, fail)
    assert path.read_text() == "previous"
    assert not path.with_name(path.name + ".part").exists()
