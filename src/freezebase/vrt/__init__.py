"""GDAL Virtual Dataset creation."""

from __future__ import annotations

from freezebase.vrt._legacy import (
    build_vrt_mosaic,
    create_decibel_vrt,
    create_rgb_vrt,
    create_rgba_vrt,
    create_warped_vrt,
)
from freezebase.vrt._model import VRT, Band, PixelFunction, Source, SourceProperties, Warp

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
]
