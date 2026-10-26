# freezebase

[![PyPI](https://img.shields.io/pypi/v/freezebase.svg)](https://pypi.org/project/freezebase/)
[![Python versions](https://img.shields.io/pypi/pyversions/freezebase.svg)](https://pypi.org/project/freezebase/)
[![License](https://img.shields.io/pypi/l/freezebase.svg)](https://github.com/lqgentner/freezebase/blob/main/LICENSE)
[![CI](https://github.com/lqgentner/freezebase/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lqgentner/freezebase/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lqgentner/freezebase/graph/badge.svg?token=XASWV60VNS)](https://codecov.io/gh/lqgentner/freezebase)
[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Raster I/O on the MGRS grid, locally or on S3.

freezebase grew out of large-scale glacier mapping, where study areas are
scattered across several UTM zones and large raster collections live in object
storage.

Its centrepiece is **`MGRSGrid`**, a 10 km grid covering the world between 80°S
and 84°N. Each MGRS square is an
[`odc.geo.GeoBox`](https://odc-geo.readthedocs.io/en/latest/intro-geobox.html)
with a fixed CRS, transform and shape, so rasters resampled onto the same square
at the same resolution line up pixel for pixel. See
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

S3 support needs the `s3` extra, which adds `s3fs`, `fsspec`, `boto3` and
`pydantic-settings`:

```bash
uv add "freezebase[s3]"
```

## MGRS quick start

Build the 10 km grid covering an area of interest:

```python
from shapely import box

from freezebase.mgrs import MGRSGrid

grid = MGRSGrid(box(8.4, 47.3, 8.6, 47.5))  # Zurich, WGS84
gdf = grid.to_geodataframe()  # mgrs_code, zone, epsg, geometry, ...
square = grid[0]  # an odc.geo GeoBox per grid square
```

Each square is an `MGRSGeoBox`, a subclass of
[`odc.geo.GeoBox`](https://odc-geo.readthedocs.io/en/latest/intro-geobox.html)
with the CRS, transform and shape needed to resample any raster onto it. You can
also rebuild a square from its code alone, without building a grid:

```python
from freezebase.mgrs import MGRSGeoBox

square = MGRSGeoBox.from_mgrs("32TMT55", resolution=10.0)  # 1000 x 1000 px, EPSG:32632
```

## Rasters on S3

freezebase connects [Rasterio](https://rasterio.readthedocs.io/) with
[Universal Pathlib](https://universal-pathlib.readthedocs.io/): an S3 path
carries its own credentials and endpoint, and both file operations and raster
reads and writes use them.

Any S3-compatible bucket works. A named AWS profile selects the credentials
without putting keys in your code:

```python
from freezebase.s3 import make_s3_upath

s3_path = make_s3_upath(
    "s3://my-bucket/data",
    profile="example",
    endpoint_url="https://objects.example.org",
)
# No AWS credentials required
s3_anon_path = make_s3_upath("s3://copernicus-dem-30m/", anon=True)
```

Selecting a profile also stops boto from falling back to `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` in the environment.

The same path works with pathlib methods such as `exists()` and with Rasterio:

```python
from freezebase.raster import rasterio_open

scene = s3_path / "scene.tif"
if scene.exists():
    with rasterio_open(scene) as src:
        data = src.read(1)
```

See [Accessing S3](https://lqgentner.github.io/freezebase/user-guide/accessing-s3.html)
for authentication and custom endpoints, and
[Reading and writing rasters](https://lqgentner.github.io/freezebase/user-guide/rasters.html)
for COG writes and copies between buckets.

## VRTs

A [GDAL VRT](https://gdal.org/en/stable/drivers/raster/vrt.html) describes a
raster built from other rasters without copying their pixels. `freezebase.vrt`
builds VRTs in Python with Rasterio alone, without the GDAL command-line tools.
Mosaic one COG per square of the Zurich grid above and reproject the mosaic to
the Swiss grid, EPSG:2056:

```python
from freezebase import vrt
from freezebase.raster import write_cog

tiles = [f"{code}.tif" for code in grid.to_geodataframe()["mgrs_code"]]
mosaic = vrt.mosaic(tiles)
lv95 = vrt.warp(mosaic, crs="EPSG:2056", resolution=10.0)
write_cog(lv95, "zurich_lv95.tif")
```

`vrt.mosaic` needs tiles in one CRS, so the squares must lie in the same UTM
zone. No pixels are read until `write_cog` streams them into a Cloud Optimized
GeoTIFF. `write_cog` also takes a raster file or a NumPy array, and the tiles and
the COG can be local or on S3. See
[VRTs](https://lqgentner.github.io/freezebase/user-guide/vrts.html) for stacking
bands, converting values and editing a VRT before writing it.

## Other helpers

Download a file with retries and a progress bar:

```python
from freezebase.download import HTTPDownloader

with HTTPDownloader(auth=("user", "pass")) as download:
    path = download("https://example.org/data.zip", "cache/")  # -> Path
```

## Documentation

Full documentation, including guides to the MGRS grid, rasters on S3 and VRTs, is at
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
for the full checks, the S3 integration tests and how to build the docs.
