"""Download-and-cache base classes for geospatial vector datasets."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import threading
import weakref

import geopandas as gpd
import pandas as pd

from freezebase.download import HTTPDownloader
from freezebase.utils import file_sha256, get_cache_dir
from freezebase.vectools import save_and_read_parquet

logger = logging.getLogger(__name__)

# Weak locks coordinate aliases within one process; atomic writes cover processes.
_path_locks: weakref.WeakValueDictionary[Path, threading.Lock] = weakref.WeakValueDictionary()
_path_locks_mutex = threading.Lock()


def _get_path_lock(path: Path) -> threading.Lock:
    key = path.resolve()
    with _path_locks_mutex:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata container for dataset attribution and licensing."""

    name: str
    source_url: str
    attribution: str
    license: str | None = None
    license_url: str | None = None
    version: str | None = None
    description: str | None = None
    doi: str | None = None
    # Optional checksum for the raw download.
    sha256: str | None = None
    additional_fields: dict[str, str] = field(default_factory=dict)

    def to_display_dict(self) -> dict[str, str]:
        """Return populated metadata fields for display."""
        standard = asdict(self)
        standard.pop("additional_fields")

        for k, v in self.additional_fields.items():
            standard[k] = v

        return {k: v for k, v in standard.items() if v is not None}


class GeoVectorData(ABC):
    """Base class for cached geospatial vector datasets."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            Cache root. See :func:`freezebase.utils.get_cache_dir`.
        """
        self.cache_dir = get_cache_dir(cache_dir)
        self._verified: bool = False

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""

    @property
    @abstractmethod
    def raw_path(self) -> Path:
        """Return the raw data path."""

    @property
    @abstractmethod
    def processed_path(self) -> Path:
        """Return the processed data path."""

    def _prepare(self) -> None:
        """Convert the raw data to GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        data = gpd.read_file(self.raw_path)
        save_and_read_parquet(data, self.processed_path)

    def _download(self) -> None:
        """Download the raw dataset."""
        url = self.metadata.source_url
        save_dir = self.raw_path.parent
        filename = self.raw_path.name

        with HTTPDownloader() as downloader:
            downloader(url=url, save_dir=save_dir, filename=filename)

    def get_data(self, *, download: bool = False) -> gpd.GeoDataFrame:
        """Return the processed data, preparing it if necessary."""
        if not self._verified:
            self._verify(download=download)
        return self._load_data()

    def cleanup(self, *, raw: bool = True, processed: bool = False) -> None:
        """Remove selected raw and processed cache files."""

        def _remove_file_and_cleanup_dir(path: Path) -> None:
            if path.exists():
                path.unlink()
                if not any(path.parent.iterdir()):
                    path.parent.rmdir()

        if raw:
            _remove_file_and_cleanup_dir(self.raw_path)
        if processed:
            _remove_file_and_cleanup_dir(self.processed_path)
            self._verified = False

    def _load_data(self) -> gpd.GeoDataFrame:
        """Load the processed data."""
        if self.processed_path.suffix == ".parquet":
            gdf = gpd.read_parquet(self.processed_path)
        else:
            gdf = gpd.read_file(self.processed_path)
        return gdf

    def _verify(self, *, download: bool) -> None:
        """Ensure processed data exists and the raw checksum is valid."""
        if self.processed_path.exists():
            self._verified = True
            return
        with _get_path_lock(self.processed_path):
            # Another thread may have completed while this one waited.
            if self.processed_path.exists():
                self._verified = True
                return
            if self.raw_path.exists():
                self._verify_raw_checksum()
                self._prepare()
            elif download:
                self._download()
                self._verify_raw_checksum()
                self._prepare()
            else:
                msg = "Dataset not found. Set `download=True` to automatically download."
                raise FileNotFoundError(msg)
            self._verified = True

    def _verify_raw_checksum(self) -> None:
        """Verify the configured raw-file checksum."""
        expected = self.metadata.sha256
        if not expected:
            return
        actual = file_sha256(self.raw_path)
        if actual.lower() != expected.lower():
            msg = (
                f"Checksum mismatch for '{self.raw_path.name}': "
                f"expected {expected}, got {actual}. The download may be corrupt; "
                "remove it (`remove(processed=False)`) and retry with `download=True`."
            )
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            self.__class__.__name__
            + "("
            + ", ".join(f"{k}={v}" for k, v in self.metadata.additional_fields.items())
            + ")"
        )

    def _repr_html_(self) -> str:
        """Return a Jupyter HTML representation."""
        meta_dict = self.metadata.to_display_dict()
        df_metadata = pd.DataFrame.from_dict(meta_dict, orient="index", columns=["Value"])
        table_html = df_metadata.to_html(header=False, justify="left", render_links=True)

        return f"""
        <div class='data-container'>
            <div class='data-header'>{type(self).__module__}.{type(self).__name__}</div>
            {table_html}
        </div>
        <style>
        .data-container {{
            font-family: sans-serif;
            width: fit-content;

        }}
        .data-header {{
            padding: 6px 0 6px 3px;
            color: #888;
            margin-bottom: 2px;
            border-bottom: 1px solid #555;
        }}
            .data-container table td,
            .data-container table th {{
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 540px;
        }}
        </style>
        """
