# freezebase

[![PyPI](https://img.shields.io/pypi/v/freezebase.svg)](https://pypi.org/project/freezebase/)
[![Python versions](https://img.shields.io/pypi/pyversions/freezebase.svg)](https://pypi.org/project/freezebase/)
[![License](https://img.shields.io/pypi/l/freezebase.svg)](https://github.com/lqgentner/freezebase/blob/main/LICENSE)
[![CI](https://github.com/lqgentner/freezebase/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lqgentner/freezebase/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lqgentner/freezebase/graph/badge.svg?token=XASWV60VNS)](https://codecov.io/gh/lqgentner/freezebase)
[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Lightweight geospatial helpers for reproducible MGRS grids and S3-backed raster workflows.

freezebase grew out of large-scale glacier mapping, where scattered areas of
interest require several local projections and large raster collections live
in object storage.

The centrepiece of freezebase is **`MGRSGrid`**: a 10 km grid for the UTM-covered
world that gives every MGRS code a dataset-independent target backed by an
[`odc.geo.GeoBox`](https://odc-geo.readthedocs.io/en/latest/intro-geobox.html).
Rasters resampled to the same code and resolution share their CRS, transform,
and shape. See
[The MGRS grid](https://lqgentner.github.io/freezebase/user-guide/mgrs-grid.html)
for the reasoning.

## Installation

Requires Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/) or pip.

freezebase is on [PyPI](https://pypi.org/project/freezebase/):

```bash
# uv
uv add freezebase

# pip
pip install freezebase
```

S3 support is optional; see [Raster and S3 helpers](#raster-and-s3-helpers)
below.

## MGRS quick start

Build the 10 km grid covering an area of interest:

```python
from shapely import box

from freezebase.mgrs import MGRSGrid

grid = MGRSGrid(box(8.4, 47.3, 8.6, 47.5))  # Zürich, WGS84
gdf = grid.to_geodataframe()  # mgrs_code, zone, epsg, geometry, ...
square = grid[0]  # an odc.geo GeoBox per grid square
```

Each square is an `MGRSGeoBox` — a subclass of
[`odc.geo.GeoBox`](https://odc-geo.readthedocs.io/en/latest/intro-geobox.html)
carrying the CRS, affine transform, and shape needed to resample any raster onto
it. Reconstruct one from its code alone, without building a grid:

```python
from freezebase.mgrs import MGRSGeoBox

square = MGRSGeoBox.from_mgrs("32TMS35", resolution=10.0)  # 1000 x 1000 px, EPSG:32632
```

## Raster and S3 helpers

The raster helpers use [Rasterio](https://rasterio.readthedocs.io/) for local
I/O. For object storage, freezebase bridges Rasterio/GDAL with
[Universal Pathlib](https://universal-pathlib.readthedocs.io/), so filesystem
operations and raster I/O use the same per-path credentials and endpoint
configuration.

Rewrite a local GeoTIFF as a Cloud-Optimized GeoTIFF:

```python
from freezebase.raster import COG_PROFILE, rewrite_tiff

rewrite_tiff("in.tif", "out.tif", profile=COG_PROFILE)  # copy
rewrite_tiff("in.tif", "out.tif", profile=COG_PROFILE, move=True)  # move
```

### The `s3` extra

S3 support needs the optional `s3fs`, `fsspec`, and `boto3` dependencies:

```bash
uv add "freezebase[s3]"
```

Credentials and endpoint configuration travel with each path. Named profiles
make it possible to select different stores without attaching raw credential
values:

```python
from freezebase.s3 import make_s3_upath

aws_path = make_s3_upath("s3://aws-bucket/data", profile="research")
ceph_path = make_s3_upath(
    "s3://ceph-bucket/data",
    profile="ceph-research",
    endpoint_url="https://objects.example.org",
)
```

An explicitly selected profile also prevents boto from falling back to ambient
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` values from the shell.

The same configured path works for pathlib-style discovery and Rasterio reads:

```python
from freezebase.raster import rasterio_open

scene = aws_path / "scene.tif"
if scene.exists():
    with rasterio_open(scene) as src:
        data = src.read(1)
```

Public buckets use `anon=True`. See [Rasters on
S3](https://lqgentner.github.io/freezebase/user-guide/rasters-on-s3.html) for
authentication patterns, custom endpoints, and S3-to-S3 copies.

## Other helpers

Download a file with retries and a progress bar:

```python
from freezebase.download import HTTPDownloader

with HTTPDownloader(auth=("user", "pass")) as download:
    path = download("https://example.org/data.zip", "cache/")  # -> Path
```

## Documentation

Full documentation, including the MGRS grid guide and the S3 raster guide, is at
**<https://lqgentner.github.io/freezebase/>**.

## Support

This is a small internal core library maintained on a best-effort basis. Please
file bugs and questions on the
[issue tracker](https://github.com/lqgentner/freezebase/issues).

## Contributing

```bash
git clone https://github.com/lqgentner/freezebase.git
cd freezebase
uv sync --all-extras
uv run pytest
```

See the [contributing guide](https://lqgentner.github.io/freezebase/user-guide/contributing.html)
for the full checks, the S3 integration tests, and how to build these docs.
