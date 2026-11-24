# freezebase

[![PyPI](https://img.shields.io/pypi/v/freezebase.svg)](https://pypi.org/project/freezebase/)
[![Python versions](https://img.shields.io/pypi/pyversions/freezebase.svg)](https://pypi.org/project/freezebase/)
[![License](https://img.shields.io/pypi/l/freezebase.svg)](https://github.com/lqgentner/freezebase/blob/main/LICENSE)
[![CI](https://github.com/lqgentner/freezebase/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lqgentner/freezebase/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lqgentner/freezebase/graph/badge.svg?token=XASWV60VNS)](https://codecov.io/gh/lqgentner/freezebase)
[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

MGRS grids, S3 paths and GDAL VRTs for raster I/O in Python.

- [**MGRS grid**](#mgrs-quick-start): Build the 10 km MGRS squares over any area between
  80°S and 84°N. Rasters resampled onto the same square line up pixel for pixel.
- [**S3 paths**](#rasters-on-s3): Read and write rasters on S3 as if they were local
  files, with credentials and endpoint set per path.
- [**VRTs**](#vrts): Mosaic and reproject rasters as one lazy chain of GDAL virtual
  rasters, then write a Cloud Optimized GeoTIFF. No GDAL bindings or command-line tools
  needed.

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
with a fixed CRS, transform and shape, so rasters resampled onto the same square at
the same resolution line up pixel for pixel. You can also rebuild a square from its
code alone, without building a grid:

```python
from freezebase.mgrs import MGRSGeoBox

square = MGRSGeoBox.from_mgrs("32TMT55", resolution=10.0)  # 1000 x 1000 px, EPSG:32632
```

See [The MGRS grid](https://lqgentner.github.io/freezebase/user-guide/mgrs-grid.html)
for why freezebase uses a fixed grid.

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

`freezebase.vrt` builds [GDAL VRTs](https://gdal.org/en/stable/drivers/raster/vrt.html)
in Python, without the GDAL command-line tools. A VRT describes a raster built from
other rasters without copying their pixels, which makes it a cheap way to chain steps
such as mosaicking, reprojecting and writing a Cloud Optimized GeoTIFF. This mosaics two
Copernicus DEM tiles and reprojects them onto an MGRS square:

```python
from rasterio.enums import Resampling

from freezebase import vrt
from freezebase.mgrs import MGRSGeoBox

square = MGRSGeoBox.from_mgrs("32TMS25", resolution=20.0)
names = [f"Copernicus_DSM_COG_10_N46_00_E{lon:03d}_00_DEM" for lon in (7, 8)]
tiles = [f"https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif" for n in names]

dem = vrt.mosaic(tiles).reproject(
    crs=f"EPSG:{square.crs.epsg}",
    resolution=20.0,
    bounds=square.boundingbox,
    resampling=Resampling.bilinear,
)
dem.to_cog("32TMS25.tif")
```

GDAL reads pixels only when `to_cog` runs, and then only the blocks under the square:
about 4 MB of the two 42 MB tiles. The tiles can be local, on S3 or over HTTPS, and the
COG local or on S3. See [VRTs](https://lqgentner.github.io/freezebase/user-guide/vrts.html) for the
other steps and [VRT recipes](https://lqgentner.github.io/freezebase/user-guide/vrt-recipes.html)
for worked examples.

## Documentation

Full documentation, including guides to the MGRS grid, rasters on S3 and VRTs, is at
**<https://lqgentner.github.io/freezebase/>**.

## Support

Please file bugs and questions on the
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
