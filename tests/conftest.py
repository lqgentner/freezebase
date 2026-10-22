"""Fixtures shared by the test modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import rasterio
import rasterio.shutil

from freezebase import raster

if TYPE_CHECKING:
    from upath import UPath

_S3_PREFIXES = ("/vsis3/", "s3://")


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
