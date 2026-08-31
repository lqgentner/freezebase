"""Tests for freezebase.vectordata cache lifecycle."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import TYPE_CHECKING, Self

import geopandas as gpd
import pytest
import shapely

from freezebase import vectordata
from freezebase.utils import file_sha256
from freezebase.vectordata import (
    DatasetMetadata,
    GeoVectorData,
    _get_path_lock,
    _path_locks,
)

if TYPE_CHECKING:
    from pathlib import Path

# Wide enough that both threads reach the path lock during the first `_prepare`.
_PREPARE_DELAY_S = 0.2


def _sample_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [shapely.Point(0, 0)]}, crs="EPSG:4326")


class _FakeDataset(GeoVectorData):
    """Minimal concrete dataset backed by local files under the cache dir."""

    #: Optional expected raw hash, surfaced through metadata for checksum tests.
    sha256: str | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="fake",
            source_url="https://example.invalid/fake.geojson",
            attribution="test",
            sha256=self.sha256,
        )

    @property
    def raw_path(self) -> Path:
        return self.cache_dir / "raw.geojson"

    @property
    def processed_path(self) -> Path:
        return self.cache_dir / "processed.parquet"


class _RecordingDataset(_FakeDataset):
    """Record `_download`/`_prepare` calls and stall inside `_prepare`."""

    def __init__(self, cache_dir: Path, *, delay: float = 0.0) -> None:
        super().__init__(cache_dir)
        self.downloads: list[str] = []
        self.prepares: list[str] = []
        self._delay = delay

    def _download(self) -> None:
        self.downloads.append(threading.current_thread().name)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        _sample_gdf().to_file(self.raw_path, driver="GeoJSON")

    def _prepare(self) -> None:
        self.prepares.append(threading.current_thread().name)
        time.sleep(self._delay)
        super()._prepare()


@pytest.fixture
def dataset(tmp_path: Path) -> _FakeDataset:
    ds = _FakeDataset(cache_dir=tmp_path)
    _sample_gdf().to_file(ds.raw_path, driver="GeoJSON")
    return ds


def test_metadata_display_dict_flattens_additional_fields_and_omits_none() -> None:
    metadata = DatasetMetadata(
        name="Example",
        source_url="https://example.test/data",
        attribution="Example authors",
        additional_fields={"resolution": "10 m"},
    )

    assert metadata.to_display_dict() == {
        "name": "Example",
        "source_url": "https://example.test/data",
        "attribution": "Example authors",
        "resolution": "10 m",
    }


class TestCleanupResetsVerification:
    def test_get_data_after_full_wipe_reverifies(self, dataset: _FakeDataset) -> None:
        # Prime the cache: processed file is created and _verified becomes True.
        dataset.get_data()
        assert dataset.processed_path.exists()
        assert dataset._verified is True

        # Removing the processed file must invalidate the cached verification,
        # otherwise a later get_data() would try to read a deleted file.
        dataset.cleanup(raw=True, processed=True)
        assert dataset._verified is False

        with pytest.raises(FileNotFoundError):
            dataset.get_data(download=False)

    def test_remove_processed_only_resets_flag(self, dataset: _FakeDataset) -> None:
        dataset.get_data()
        assert dataset._verified is True

        dataset.cleanup(raw=False, processed=True)
        assert dataset._verified is False
        # The raw file survives, so a re-verify can re-prepare without download.
        assert dataset.raw_path.exists()
        regated = dataset.get_data(download=False)
        assert len(regated) == 1


class TestChecksumVerification:
    def test_matching_checksum_passes(self, dataset: _FakeDataset) -> None:
        dataset.sha256 = file_sha256(dataset.raw_path)
        # Should prepare without complaint.
        assert len(dataset.get_data(download=False)) == 1

    def test_mismatched_checksum_raises(self, dataset: _FakeDataset) -> None:
        dataset.sha256 = "0" * 64  # deliberately wrong
        with pytest.raises(ValueError, match="Checksum mismatch"):
            dataset.get_data(download=False)
        # A failed verification must not leave a processed file behind.
        assert not dataset.processed_path.exists()

    def test_no_checksum_is_noop(self, dataset: _FakeDataset) -> None:
        assert dataset.sha256 is None
        assert len(dataset.get_data(download=False)) == 1


class TestCleanupDefaults:
    def test_default_removes_raw_keeps_processed(self, dataset: _FakeDataset) -> None:
        dataset.get_data()
        assert dataset.raw_path.exists()

        dataset.cleanup()  # defaults: raw=True, processed=False

        assert not dataset.raw_path.exists()
        assert dataset.processed_path.exists()
        # Processed data is still valid, so verification stays primed.
        assert dataset._verified is True
        assert len(dataset.get_data(download=False)) == 1


class TestVerifyDownloadBranch:
    def test_download_true_fetches_and_prepares_when_raw_is_absent(self, tmp_path: Path) -> None:
        ds = _RecordingDataset(tmp_path)
        assert not ds.raw_path.exists()

        gdf = ds.get_data(download=True)

        assert len(gdf) == 1
        assert len(ds.downloads) == 1
        assert len(ds.prepares) == 1
        assert ds.processed_path.exists()

    def test_download_false_without_raw_raises(self, tmp_path: Path) -> None:
        ds = _RecordingDataset(tmp_path)

        with pytest.raises(FileNotFoundError, match="download=True"):
            ds.get_data(download=False)

        assert ds.downloads == []
        assert ds.prepares == []

    def test_existing_raw_is_prepared_without_downloading(self, tmp_path: Path) -> None:
        ds = _RecordingDataset(tmp_path)
        _sample_gdf().to_file(ds.raw_path, driver="GeoJSON")

        ds.get_data(download=True)

        # A raw file already on disk must short-circuit the download.
        assert ds.downloads == []
        assert len(ds.prepares) == 1

    def test_checksum_is_verified_after_download(self, tmp_path: Path) -> None:
        ds = _RecordingDataset(tmp_path)
        ds.sha256 = "0" * 64  # deliberately wrong

        # A corrupt download must never reach the processed cache.
        with pytest.raises(ValueError, match="Checksum mismatch"):
            ds.get_data(download=True)

        assert ds.downloads
        assert ds.prepares == []
        assert not ds.processed_path.exists()


class TestDefaultDownload:
    """The base `_download`, which subclasses inherit unless they override it."""

    def test_wires_the_downloader_to_the_raw_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, object] = {}

        class _FakeDownloader:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc_info: object) -> None:
                seen["closed"] = True

            def __call__(self, *, url: str, save_dir: Path, filename: str) -> Path:
                seen.update(url=url, save_dir=save_dir, filename=filename)
                save_dir.mkdir(parents=True, exist_ok=True)
                target = save_dir / filename
                _sample_gdf().to_file(target, driver="GeoJSON")
                return target

        monkeypatch.setattr(vectordata, "HTTPDownloader", _FakeDownloader)
        ds = _FakeDataset(cache_dir=tmp_path / "cache")

        gdf = ds.get_data(download=True)

        assert seen["url"] == ds.metadata.source_url
        assert seen["save_dir"] == ds.raw_path.parent
        assert seen["filename"] == ds.raw_path.name
        # The downloader is used as a context manager, so its session is closed.
        assert seen["closed"] is True
        assert len(gdf) == 1


class TestConcurrentPreparation:
    """Two threads racing for the same cache entry must do the work once."""

    def _run_in_two_threads(
        self,
        datasets: tuple[_RecordingDataset, _RecordingDataset],
        *,
        download: bool,
    ) -> list[gpd.GeoDataFrame]:
        barrier = threading.Barrier(len(datasets))

        def worker(ds: _RecordingDataset) -> gpd.GeoDataFrame:
            barrier.wait()  # release both threads as simultaneously as possible
            return ds.get_data(download=download)

        with ThreadPoolExecutor(max_workers=len(datasets)) as pool:
            return [f.result() for f in [pool.submit(worker, ds) for ds in datasets]]

    def test_prepare_runs_once_across_threads(self, tmp_path: Path) -> None:
        # Separate instances, so only the shared path lock can deduplicate.
        first = _RecordingDataset(tmp_path, delay=_PREPARE_DELAY_S)
        second = _RecordingDataset(tmp_path, delay=_PREPARE_DELAY_S)
        _sample_gdf().to_file(first.raw_path, driver="GeoJSON")

        results = self._run_in_two_threads((first, second), download=False)

        assert len(first.prepares) + len(second.prepares) == 1
        # Both callers still get usable data, whichever thread did the work.
        assert [len(gdf) for gdf in results] == [1, 1]
        assert all(gdf.crs == "EPSG:4326" for gdf in results)

    def test_download_runs_once_across_threads(self, tmp_path: Path) -> None:
        first = _RecordingDataset(tmp_path, delay=_PREPARE_DELAY_S)
        second = _RecordingDataset(tmp_path, delay=_PREPARE_DELAY_S)

        results = self._run_in_two_threads((first, second), download=True)

        assert len(first.downloads) + len(second.downloads) == 1
        assert len(first.prepares) + len(second.prepares) == 1
        assert [len(gdf) for gdf in results] == [1, 1]

    def test_no_staging_files_survive_the_race(self, tmp_path: Path) -> None:
        first = _RecordingDataset(tmp_path, delay=_PREPARE_DELAY_S)
        second = _RecordingDataset(tmp_path, delay=_PREPARE_DELAY_S)

        self._run_in_two_threads((first, second), download=True)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "processed.parquet",
            "raw.geojson",
        ]


class TestPathLock:
    def test_aliased_paths_share_one_lock(self, tmp_path: Path) -> None:
        direct = tmp_path / "sub" / "data.parquet"
        direct.parent.mkdir()
        aliased = tmp_path / "sub" / ".." / "sub" / "data.parquet"
        # Different Path spellings of the same file must map to the same lock.
        assert _get_path_lock(direct) is _get_path_lock(aliased)

    def test_registry_self_cleans(self, tmp_path: Path) -> None:
        path = tmp_path / "ephemeral.parquet"
        assert _get_path_lock(path) is not None
        # With no strong reference held, the weak registry drops the entry.
        assert path.resolve() not in _path_locks
