"""GDAL Virtual Dataset creation."""

from __future__ import annotations

from freezebase.vrt._legacy import (
    build_vrt_mosaic,
    create_decibel_vrt,
    create_rgb_vrt,
    create_rgba_vrt,
    create_warped_vrt,
)

__all__ = [
    "build_vrt_mosaic",
    "create_decibel_vrt",
    "create_rgb_vrt",
    "create_rgba_vrt",
    "create_warped_vrt",
]
