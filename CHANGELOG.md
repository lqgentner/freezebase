# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor releases may contain breaking changes.

## Unreleased

### Added

- `write_cog` takes a `checksum` keyword. When `True`, it returns the SHA-256
  of the written bytes as a multihash (`"1220"` + hex) instead of `None`.
- `subprocess_s3_env` builds the S3 environment variables a child process needs
  to open `/vsis3/` paths.
- `create_warped_vrt` reprojects a raster onto a fixed target grid, and
  `create_rgba_vrt` colour-maps a raster into a 4-band VRT via per-band lookup
  tables.
- `build_vrt_mosaic` takes a `bounds` keyword to fix the mosaic extent, and no
  longer requires tiles to live beside the output VRT.

## [0.6.1] - 2026-08-31

### Fixed

- `write_cog` now applies caller-supplied COG creation options to the finished
  COG instead of silently discarding them. Its intermediate GTiff remains
  losslessly compressed, avoiding double quantization for lossy codecs such as
  LERC.

## [0.6.0] - 2026-08-25

### Added

- `write_cog` and `rewrite_tiff` take `tags`, `band_tags` and `units`, writing
  dataset-level tags, per-band tags and band unit strings. All three are
  keyword-only and default to `None`, so existing calls are unchanged.

### Fixed

- `rewrite_tiff` no longer loses `LAYOUT: COG` when a non-GTiff destination
  carries band descriptions, tags, band tags or units. All of those live in the
  GDAL_METADATA TIFF tag, which grows when written into a finished file and is
  relocated behind the image data; they now go into an intermediate GTiff so
  the driver copy writes the final layout around them.
- Every per-band sequence (`band_names`, `color_interp`, `band_tags`, `units`)
  is now rejected if it does not match the band count.
- The `s3` extra now requires `s3fs>=2026.7.0`. Earlier versions cache a
  prefix-filtered listing under the unfiltered directory key
  ([fsspec/s3fs#1034](https://github.com/fsspec/s3fs/pull/1034)), so a single
  `glob("prefix*")` makes every later listing miss files.

## [0.5.0] - 2026-08-24

### Added

- Python 3.14 support, verified in CI.
- `freezebase.__version__`, read from the installed distribution metadata. The
  version is still derived from git tags at build time by hatch-vcs; it is now
  also reachable at runtime.

### Fixed

- `HTTPDownloader` now closes the streamed response on every path out of a
  download, not only on the successful one.

## [0.4.0] - 2026-08-17

### Added

- `HTTPDownloader.close()` and context-manager support, which close the
  downloader's session and release its pooled connections. Creating one
  downloader per download previously held a connection open until the garbage
  collector ran, which made downstream test suites fail unpredictably. The
  session is also closed on garbage collection as a fallback.

### Changed

- **Breaking:** `HTTPDownloader` no longer forwards arbitrary keyword arguments
  to `requests`. `method` and `timeout` are now named parameters and are spelled
  exactly as before, so calls that only used those keep working; `method` is
  restricted to `"GET"` and `"POST"`. Any other `requests` keyword, such as
  `params` or `headers`, now raises `TypeError` instead of reaching the request.
  To configure the transport, for example a connection pool size, mount an
  adapter on `HTTPDownloader.session`.
  
## [0.3.0] - 2026-08-05

### Added

- Per-path named AWS profile support in `make_s3_upath`. Combinable with a custom
  `endpoint_url`, mutually exclusive with `anon=True` and explicit
  `key`/`secret`/`token`. Setting a profile takes environment credentials out of
  boto's resolution chain, so the path signs with the profile's keys only.
- `aws_session`, returning the cached Rasterio `AWSSession` for an S3 path, and
  `clear_aws_session_cache` to discard those sessions. Prefer `aws_session` over
  constructing an `AWSSession` directly, so every layer signs identically.

### Changed

- Renamed the package to `freezebase`, to allow publication on
  PyPI under the same name. The import path, PyPI distribution name,
  `FREEZEBASE_CACHE`/`FREEZEBASE_DATA` environment variables, and the GitHub
  repository all changed accordingly; there is no compatibility shim for the old names.
- `s3_env` reuses a cached `AWSSession` per distinct S3 configuration instead of
  building one per call. Constructing a session resolves boto's whole credential
  chain eagerly, and `s3_env` is entered on every raster operation, so a
  profile-authenticated path previously re-read the shared credentials file
  for every raster read and write. Refreshable STS/SSO credentials still rotate, because
  Rasterio re-freezes them on each `Env` entry; static credentials rewritten
  mid-process now need `clear_aws_session_cache`.
- `make_s3_upath` rejects credentials in `client_kwargs`, which reach s3fs but
  not `s3_env`.

## [0.2.0] - 2026-08-03

### Added

- Explicit `anon` argument on `make_s3_upath`, matching the `anon` parameter of
  `s3fs.S3FileSystem`: `anon=True` uses an anonymous connection (public buckets
  only); `anon=False` (the default) uses the `key`/`secret` given, or boto's
  credential resolver. Combining `anon=True` with `key`, `secret`, or `token`
  raises `ValueError`.

### Changed

- **Breaking:** anonymous S3 access is now opt-in.

  ```python
  # before
  path = make_s3_upath("s3://copernicus-dem-30m/x.tif", region="eu-central-1")
  # after
  path = make_s3_upath("s3://copernicus-dem-30m/x.tif", region="eu-central-1", anon=True)

### Fixed

- `make_s3_upath` (for s3fs) and `s3_env` (for GDAL) now agree on the
anonymous/signed decision. Previously, without passing credentials, GDAL read
unsigned while s3fs signed via boto's resolver.

## [0.1.0] - 2026-07-27

### Added

- `py.typed` marker so downstream type checkers use the inline annotations.
- Optional `DatasetMetadata.sha256` to verify a raw download against a known
  hash (`freezebase.vectordata`).
- `freezebase.utils.file_sha256` helper.
- Packaging metadata: project URLs, keywords, and classifiers.
- GitHub Actions CI (lint, type-check, test matrix on 3.12/3.13, a
  lowest-bound dependency job, a MinIO-backed S3 integration job, and a build +
  wheel smoke-test) and a Trusted-Publishing release workflow.
- S3 integration tests (`tests/test_s3_integration.py`, `integration` marker),
  skipped unless `FREEZEBASE_TEST_S3_*` is set.
- `freezebase.s3.s3_env`, a rasterio context manager that configures
  credentials and endpoint for S3-compatible object storage from a `UPath`'s
  storage options. Usable with `rasterio` or `rioxarray`.
- `rewrite_tiff` can now copy between two different S3 backends (e.g. an
  unsigned public bucket to a private one); the source read and destination
  write each apply their own credentials, so the previous same-backend
  restriction is gone.

### Changed

- **Breaking:** `make_s3_upath` renamed its first parameter `root` → `path` and
  gained optional `token` and `region`; `key`/`secret` are now optional so it
  can build paths for anonymous (unsigned) access to public buckets.
- S3 rasterio setup consolidated into `freezebase.s3` and is now rasterio-only:
  the optional GDAL helper were removed
- Dependency lower bounds corrected to the oldest versions that actually
  install and work on Python 3.12: notably `numpy>=2.0` (code uses `np.concat`),
  `shapely>=2.1.0` (`transform(interleaved=...)`), `pyarrow>=17.0` (NumPy 2 ABI),
  `pyproj>=3.6.1`, `boto3>=1.36`, and `s3fs>=2026.2.0`/`fsspec>=2026.2.0`
  (`set_custom_error_handler`).
- Licensing metadata modernized to PEP 639 (`license = "MIT"` +
  `license-files`), and the sdist no longer ships `.python-version`/`uv.lock`.
- **Breaking:** `GeoVectorData.remove()` is renamed to `cleanup()` and now
  defaults to removing only the raw download (`raw=True, processed=False`),
  keeping the processed data.
- **Breaking:** `rewrite_tiff()` no longer deletes the source by default; pass
  `move=True` for the previous move semantics.
- `HTTPDownloader.__call__` gained an `overwrite` keyword (default `False`) and
  now rejects unsafe/inferred filenames that would escape the destination.
- S3 transient-retry codes narrowed so a permanent `AccessDenied`/403 is no
  longer retried.
- `COG_PROFILE` now sets `OVERVIEW_RESAMPLING=AVERAGE` (GDAL defaults to
  `CUBIC`, which propagates NODATA into overview pyramids) and
  `PREDICTOR=YES`/`PREDICTOR_OVERVIEW=YES`, which shrinks float32 rasters by
  ~10-15% at unchanged write cost.

### Fixed

- Filename confinement and redirect credential handling in the downloader.
- Destination/source preservation on failure in `rewrite_tiff`.
- Numerous MGRS/UTM/CRS validation and parsing defects.
- VRT/merge input validation and XML-safe VRT generation.
- Z-coordinate detection across polygon interior rings and mixed collections.
- Atomic cache writes and verification-state reset on removal.
