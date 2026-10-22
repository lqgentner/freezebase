"""GDAL Virtual Dataset creation."""

from __future__ import annotations

from freezebase.vrt._builders import derive, from_raster, mosaic, stack
from freezebase.vrt._io import read
from freezebase.vrt._legacy import (
    build_vrt_mosaic,
    create_decibel_vrt,
    create_rgb_vrt,
    create_rgba_vrt,
    create_warped_vrt,
)
from freezebase.vrt._model import VRT, Band, PixelFunction, Source, SourceProperties, Warp
from freezebase.vrt._warp import warp

__all__ = [
    "VRT",
    "Band",
    "PixelFunction",
    "Source",
    "SourceProperties",
    "Warp",
    "build_vrt_mosaic",
    "create_decibel_vrt",
    "create_rgb_vrt",
    "create_rgba_vrt",
    "create_warped_vrt",
    "derive",
    "from_raster",
    "mosaic",
    "read",
    "stack",
    "warp",
]
