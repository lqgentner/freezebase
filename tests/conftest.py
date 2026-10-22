"""Fixtures shared by the test modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyproj.datadir
import pytest
import rasterio
from rasterio.env import PROJDataFinder
import rasterio.shutil

from freezebase import raster

if TYPE_CHECKING:
    from upath import UPath

_S3_PREFIXES = ("/vsis3/", "s3://")
GRIDS = Path(pyproj.datadir.get_user_data_dir())
_SWISS_GRIDS = (
    "ch_swisstopo_chgeo2004_ETRS89_LN02.tif",
    "ch_swisstopo_chgeo2004_ETRS89_LHN95.tif",
)


@pytest.fixture(scope="session")
def proj_grids(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a PROJ data directory joining rasterio's PROJ data and pyproj's synced grids."""
    missing = [name for name in _SWISS_GRIDS if not (GRIDS / name).is_file()]
    if missing:
        reason = (
            f"PROJ grids missing ({', '.join(missing)}); run pyproj sync --source-id ch_swisstopo"
        )
        # A bad CI grid cache must not silently skip the vertical tests.
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    proj_data = PROJDataFinder().search()
    if proj_data is None:
        pytest.skip("rasterio bundles no PROJ data")
    joined = tmp_path_factory.mktemp("proj_data")
    # rasterio's files come first, so its proj.db wins any name clash.
    for directory in (Path(proj_data), GRIDS):
        for file in directory.iterdir():
            link = joined / file.name
            if not link.exists():
                link.symlink_to(file)
    return joined


@pytest.fixture
def s3_uploads(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Record every S3 upload, and fail any write GDAL makes to S3 itself.

    GDAL's own S3 writes are what lost data silently: a gateway refused the
    unsigned Content-Type header, and the failure never reached Python. So the
    route is asserted, not just the result.
    """
    uploads: dict[str, bytes] = {}

    def record(src: bytes | Path, dst: UPath) -> None:
        uploads[str(dst)] = src if isinstance(src, bytes) else Path(src).read_bytes()

    real_copy = rasterio.shutil.copy
    real_open = rasterio.open

    def guarded_copy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        assert not str(dst).startswith(_S3_PREFIXES), f"GDAL wrote to {dst}"
        return real_copy(src, dst, *args, **kwargs)

    def guarded_open(fp: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        assert mode == "r" or not str(fp).startswith(_S3_PREFIXES), f"GDAL opened {fp} as {mode}"
        return real_open(fp, mode, *args, **kwargs)

    monkeypatch.setattr(raster, "_upload", record)
    monkeypatch.setattr(rasterio.shutil, "copy", guarded_copy)
    monkeypatch.setattr(rasterio, "open", guarded_open)
    return uploads
