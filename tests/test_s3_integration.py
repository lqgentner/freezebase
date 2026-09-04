"""S3 integration tests against a live S3-compatible service (e.g. MinIO).

These exercise the credentialed :func:`freezebase.s3.s3_env` read/write paths
used by ``freezebase.raster`` that unit tests can't reach. They are skipped unless the
``FREEZEBASE_TEST_S3_*`` environment variables point at a reachable endpoint;
CI provides them via a MinIO service container.

Configure with:

- ``FREEZEBASE_TEST_S3_ENDPOINT``  (e.g. ``http://localhost:9000``)
- ``FREEZEBASE_TEST_S3_KEY``
- ``FREEZEBASE_TEST_S3_SECRET``
- ``FREEZEBASE_TEST_S3_BUCKET``    (optional, default ``freezebase-test``)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

import numpy as np
import pytest
from rasterio.transform import from_origin

pytest.importorskip("s3fs")

if TYPE_CHECKING:
    from collections.abc import Iterator

    from upath import UPath

from freezebase.raster import (
    COG_PROFILE,
    rasterio_open,
    rewrite_tiff,
    write_cog,
)
from freezebase.s3 import make_s3_upath
from freezebase.vrt import create_warped_vrt

_ENDPOINT = os.getenv("FREEZEBASE_TEST_S3_ENDPOINT")
_KEY = os.getenv("FREEZEBASE_TEST_S3_KEY")
_SECRET = os.getenv("FREEZEBASE_TEST_S3_SECRET")
_BUCKET = os.getenv("FREEZEBASE_TEST_S3_BUCKET", "freezebase-test")
# Name of the throwaway shared-credentials profile written by ``s3_profile_key``.
_AWS_PROFILE = "freezebase-integration"

_skip_reason = "FREEZEBASE_TEST_S3_ENDPOINT/KEY/SECRET not set"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (_ENDPOINT and _KEY and _SECRET), reason=_skip_reason),
]

WIDTH, HEIGHT = 8, 6
_RASTER_PROFILE = {
    "dtype": "float32",
    "count": 1,
    "width": WIDTH,
    "height": HEIGHT,
    "crs": "EPSG:32632",
    "transform": from_origin(500000, 5200000, 10, 10),
    "nodata": np.nan,
}


@pytest.fixture(scope="module")
def bucket_root() -> UPath:
    """Return a UPath rooted at the test bucket, creating it if needed."""
    root = make_s3_upath(
        f"s3://{_BUCKET}",
        key=_KEY,
        secret=_SECRET,
        endpoint_url=_ENDPOINT,
    )
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def s3_key(bucket_root: UPath) -> Iterator[UPath]:
    """Yield a unique object path inside the bucket, cleaned up afterwards."""
    path = bucket_root / f"{uuid.uuid4().hex}.tif"
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def s3_key2(bucket_root: UPath) -> Iterator[UPath]:
    """Yield a second unique object path inside the bucket, cleaned up afterwards."""
    path = bucket_root / f"{uuid.uuid4().hex}.tif"
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def s3_profile_key(
    bucket_root: UPath,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[UPath]:
    """Yield an object path authenticated solely by a named AWS profile.

    ``bucket_root`` (key/secret) is depended on only to create the bucket; the
    yielded path carries no credentials of its own, so the round trip fails
    unless both s3fs and GDAL resolve the profile.
    """
    credentials = tmp_path / "credentials"
    credentials.write_text(
        f"[{_AWS_PROFILE}]\naws_access_key_id = {_KEY}\naws_secret_access_key = {_SECRET}\n",
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    # Isolate from a developer's ~/.aws/config, which may define a same-named
    # profile with its own region or endpoint.
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))

    # Same bucket, but reached through a path that carries only the profile.
    root = make_s3_upath(
        str(bucket_root),
        profile=_AWS_PROFILE,
        endpoint_url=_ENDPOINT,
    )
    path = root / f"{uuid.uuid4().hex}.tif"
    fs = path.fs
    yield path
    path.unlink(missing_ok=True)
    # Drop the cached filesystem bound to the now-vanishing credentials file.
    fs.clear_instance_cache()


class TestWriteCogS3:
    def test_round_trip(self, s3_key: UPath) -> None:
        data = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)

        write_cog(data, s3_key, _RASTER_PROFILE, band_names=["VH"])

        assert s3_key.exists()
        with rasterio_open(s3_key) as ds:
            assert ds.descriptions == ("VH",)
            assert np.array_equal(ds.read(1), data)

    def test_round_trip_with_named_profile(self, s3_profile_key: UPath) -> None:
        data = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)

        write_cog(data, s3_profile_key, _RASTER_PROFILE, band_names=["VH"])

        assert s3_profile_key.exists()
        with rasterio_open(s3_profile_key) as ds:
            assert np.array_equal(ds.read(1), data)


class TestRewriteTiffS3:
    def test_local_to_s3_copy(self, tmp_path: Path, s3_key: UPath) -> None:
        src = tmp_path / "src.tif"
        data = np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32)
        write_cog(data, src, _RASTER_PROFILE, band_names=["VV"])

        rewrite_tiff(src, s3_key, profile=COG_PROFILE)

        assert src.exists()  # copy by default -- source preserved
        assert s3_key.exists()
        with rasterio_open(s3_key) as ds:
            assert ds.descriptions == ("VV",)
            assert np.array_equal(ds.read(1), data)

    def test_local_to_s3_move_deletes_source(self, tmp_path: Path, s3_key: UPath) -> None:
        src = tmp_path / "src.tif"
        data = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
        write_cog(data, src, _RASTER_PROFILE)

        rewrite_tiff(src, s3_key, profile=COG_PROFILE, move=True)

        assert not src.exists()
        assert s3_key.exists()

    def test_s3_to_s3_copy(self, s3_key: UPath, s3_key2: UPath) -> None:
        # Exercises the in-memory stage reading *from* S3 and writing back to
        # S3, with the source-read and destination-write each entering their own
        # credentialed env.
        data = np.full((HEIGHT, WIDTH), 4.0, dtype=np.float32)
        write_cog(data, s3_key, _RASTER_PROFILE, band_names=["VH"])

        rewrite_tiff(s3_key, s3_key2, profile=COG_PROFILE)

        assert s3_key.exists()  # copy by default -- source preserved
        assert s3_key2.exists()
        with rasterio_open(s3_key2) as ds:
            assert ds.descriptions == ("VH",)
            assert np.array_equal(ds.read(1), data)


class TestCreateWarpedVrtS3:
    def test_local_source_to_s3_destination(self, tmp_path: Path, s3_key: UPath) -> None:
        # The write needs s3_key's own credentialed env; entering only the
        # local source's (uncredentialed) env would leave the /vsis3/ write
        # unauthenticated.
        src = tmp_path / "src.tif"
        write_cog(np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32), src, _RASTER_PROFILE)

        create_warped_vrt(src, s3_key, crs="EPSG:3857", resolution=20.0)

        assert s3_key.exists()
        with rasterio_open(s3_key) as ds:
            assert ds.crs.to_epsg() == 3857


class TestListingsCache:
    """Guards the `s3fs>=2026.7.0` floor that keeps prefix globs from poisoning.

    Before fsspec/s3fs#1034, `_lsdir` wrote a prefix-filtered listing into
    `dircache` under the unfiltered directory key. These fail on a downgrade.
    """

    @staticmethod
    def _seed(root: UPath) -> str:
        """Write one object per prefix and return the shared parent."""
        parent = f"{uuid.uuid4().hex}"
        for name in ("AAA_one.tif", "BBB_two.tif"):
            (root / parent / name).write_bytes(b"x")
        return parent

    def test_glob_of_one_prefix_does_not_hide_another(self, bucket_root: UPath) -> None:
        # Globbing "AAA*" must not cache a partial listing under the directory
        # key, or the next glob for "BBB*" is answered from it and finds nothing.
        parent = self._seed(bucket_root)
        try:
            directory = bucket_root / parent
            assert len(list(directory.glob("AAA*.tif"))) == 1
            assert len(list(directory.glob("BBB*.tif"))) == 1
        finally:
            for name in ("AAA_one.tif", "BBB_two.tif"):
                (bucket_root / parent / name).unlink(missing_ok=True)

    def test_glob_does_not_make_an_existing_object_look_absent(
        self,
        bucket_root: UPath,
    ) -> None:
        # The same poisoned listing also answers exists() and info().
        parent = self._seed(bucket_root)
        try:
            directory = bucket_root / parent
            list(directory.glob("AAA*.tif"))
            assert (directory / "BBB_two.tif").exists()
        finally:
            for name in ("AAA_one.tif", "BBB_two.tif"):
                (bucket_root / parent / name).unlink(missing_ok=True)
