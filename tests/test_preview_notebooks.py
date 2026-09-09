"""Exercise notebook downloads without requiring network access or survey tiles."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from classify_norton import prepare_flux
from classify_upsilont import cache_tile, prepare_lightcurve, source_groups


@pytest.mark.parametrize("name", ["upsilon", "norton"])
def test_missing_tile_download(name, tmp_path):
    filename = "FLUX_NG0445-3056A.fits"
    columns = [
        fits.Column(name="SOURCE_ID", format="8A", array=["source"] * 100),
        fits.Column(name="HJD", format="D", array=2450000 + np.arange(100) / 100),
        fits.Column(name="SYSFLUX", format="D", array=100 + np.sin(np.arange(100))),
        fits.Column(name="FLUX_ERR", format="D", array=np.ones(100)),
        fits.Column(name="FLAG", format="B", array=np.zeros(100, dtype=np.uint8)),
    ]
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = filename
    buffer = io.BytesIO()
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(
        buffer, checksum=True
    )
    content = buffer.getvalue()

    class Response:
        status_code = 200

        def __init__(self):
            self.headers = {"Content-Length": str(len(content))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield content

    class Session(Response):
        def get(self, *args, **kwargs):
            return Response()

    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / f"preview_{name}.ipynb").read_text()
    )
    source = "".join(notebook["cells"][4]["source"])
    namespace = {
        "PREVIEW_CACHE": tmp_path / "preview",
        "DATA_DIR": tmp_path / "data",
        "DOWNLOAD_MISSING": True,
        "PIPELINE": "upsilon-t" if name == "upsilon" else "norton",
        "ArchiveSession": Session,
        "cache_tile": cache_tile,
        "fits": fits,
        "source_groups": source_groups,
        "prepare_lightcurve": prepare_lightcurve,
        "prepare_flux": prepare_flux,
        "config": {"bin_minutes": 5, "min_points": 80, "reject_flags": 23},
    }
    # Execute the repository notebook helper to cover its actual download path.
    exec(source.split("# Prefer locally available photometry")[0], namespace)  # noqa: S102
    selected = pd.DataFrame(
        [
            {
                "dp_id": "test",
                "source_id": "source",
                "field": "NG0445-3056",
                "tile": "A",
                "filename": filename,
                "cache_path": f"tiles/{filename}",
                "download_url": "https://example.invalid/tile",
            }
        ]
    )
    curves, failures = namespace["load_curves"](selected)
    assert failures == {}
    assert len(curves[("test", "source")][0]) == 100
    assert (tmp_path / "preview" / "tiles" / filename).is_file()


@pytest.mark.parametrize("name", ["upsilon", "norton"])
def test_examples_stay_in_one_tile(name):
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / f"preview_{name}.ipynb").read_text()
    )
    source = "".join(notebook["cells"][4]["source"])
    selection = source[source.index("def select_examples(") :].split(
        "selected, selected_tile ="
    )[0]
    namespace = {
        "pd": pd,
        "EXAMPLES_PER_CATEGORY": 3,
        "SEED": 42,
        "categories": ["A", "B", "C"],
        "local_tile": lambda row: Path("cached.fits") if row.tile == "X" else None,
    }
    exec(selection, namespace)  # noqa: S102
    frame = pd.DataFrame(
        [
            {"dp_id": tile, "field": "field", "tile": tile, "label": label}
            for tile, labels in [("X", ["A"] * 5), ("Y", ["A", "B", "C"])]
            for label in labels
        ]
    )
    select = namespace["select_examples"]
    selected, tile = select(frame)
    assert tile == "fieldX"
    assert selected.dp_id.unique().tolist() == ["X"]
    assert len(selected) == 3
    selected, tile = select(frame, "fieldY")
    assert tile == "fieldY"
    assert selected.dp_id.unique().tolist() == ["Y"]
    namespace["local_tile"] = lambda row: None
    assert select(frame)[1] == "fieldY"
    empty, tile = select(frame.iloc[:0])
    assert empty.empty and tile is None
    with pytest.raises(ValueError, match="No classified results"):
        select(frame, "missing")
