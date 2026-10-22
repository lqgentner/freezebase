"""Raster fixtures shared by the test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def make_tiff(
    path: Path,
    *,
    origin: tuple[float, float] = (500000.0, 5200000.0),
    res: tuple[float, float] = (10.0, 10.0),
    size: tuple[int, int] = (6, 4),
    fill: float | None = None,
    dtype: str = "float32",
    count: int = 1,
    nodata: float | None = np.nan,
    crs: str = "EPSG:32632",
    transform: Affine | None = None,
    descriptions: Sequence[str] | None = None,
) -> Path:
    """Write a GeoTIFF of ``size`` (width, height); without ``fill`` every pixel is distinct."""
    width, height = size
    if fill is not None:
        data = np.full((count, height, width), fill, dtype=dtype)
    else:
        data = np.arange(width * height * count).reshape(count, height, width).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform if transform is not None else from_origin(*origin, *res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
        for band, description in enumerate(descriptions or (), start=1):
            dst.set_band_description(band, description)
    return path
