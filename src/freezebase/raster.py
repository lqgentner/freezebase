"""Rasterio helpers with transparent S3 support.

This module provides utilities for opening, profiling, grouping, merging,
and rewriting GeoTIFF files on local disk or S3-compatible object storage.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
import logging
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING, Any, Literal, overload

from pyproj import CRS
from pyproj.exceptions import CRSError
import rasterio
import rasterio.env
from rasterio.io import MemoryFile
from rasterio.merge import merge
import rasterio.shutil
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import numpy as np
    from rasterio.enums import ColorInterp
    from rasterio.io import DatasetReader, DatasetWriter

logger = logging.getLogger(__name__)

RASTERIO_PROFILE_DEFAULTS = {
    "driver": "GTiff",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "interleave": "pixel",
    "compress": "deflate",
}
"""Suitable defaults for a Cloud-optimized GeoTIFF (COG)."""

type AnyPath = str | Path | UPath

COG_PROFILE: dict[str, Any] = {
    "driver": "COG",
    "compress": "deflate",
    "blocksize": 512,
    "overviews": "IGNORE_EXISTING",
    # AVERAGE skips NODATA pixels, unlike GDAL's default CUBIC
    "overview_resampling": "AVERAGE",
    # YES resolves to standard predictor (predictor=2) for integer data type
    # and floating-point predictor (predictor=3) for floating point data type
    "predictor": "YES",
    "overview_predictor": "YES",
}
"""Creation profile for the GDAL ``COG`` driver, used by :func:`write_cog` and by
callers rewriting existing files to COG via :func:`rewrite_tiff`."""

_COG_CREATION_OPTIONS = (
    "blocksize",
    "overviews",
    "overview_resampling",
    "overview_predictor",
    "overview_compress",
    # GTiff rejects the COG driver's YES/NO/STANDARD
    "predictor",
)
"""COG-driver creation options the GTiff driver rejects or does not know."""


@contextmanager
def _env_for_path(path: AnyPath) -> Generator[None]:
    """Enter the appropriate rasterio environment for a path's storage backend.

    Local paths get a plain ``Env``; S3 ``UPath`` instances get a credentialed
    env via :func:`freezebase.s3.s3_env`.

    Raises
    ------
    ValueError
        If ``path`` uses a protocol other than local (``""``) or ``"s3"``.
    """
    path = UPath(path)
    match path.protocol:
        case "":
            with rasterio.env.Env():
                yield
        case "s3":
            # Lazy import: keeps botocore/s3fs (the `[s3]` extra) out of the base import.
            from freezebase.s3 import s3_env  # noqa: PLC0415

            with s3_env(path):
                yield
        case p:
            msg = f"Unsupported protocol '{p}'."
            raise ValueError(msg)


def _to_vsi_uri(path: AnyPath) -> str:
    """Convert a path to a GDAL VSI URI, using ``/vsis3/`` for S3.

    ``rasterio.open`` and ``rasterio.merge.merge`` parse ``s3://`` URIs
    themselves, but ``rasterio.shutil.copy`` passes its arguments to GDAL
    verbatim and only understands the ``/vsis3/`` form, so its S3 operands must
    go through this helper.

    Raises
    ------
    ValueError
        If ``path`` uses a protocol other than local (``""``) or ``"s3"``.
    """
    path = UPath(path)
    match path.protocol:
        case "":
            return str(path)
        case "s3":
            return f"/vsis3/{path.path}"
        case p:
            msg = f"Unsupported protocol '{p}'."
            raise ValueError(msg)


@contextmanager
def _rasterio_open(
    path: AnyPath,
    mode: Literal["r", "r+", "w", "w+"] = "r",
    **kwargs,
) -> Generator[DatasetReader | DatasetWriter]:
    """Yield a rasterio dataset, configuring a GDAL env for S3 if needed.

    Internal generator backing the public :func:`rasterio_open` dispatcher.
    """
    path = UPath(path)
    with _env_for_path(path), rasterio.open(str(path), mode, **kwargs) as dataset:
        yield dataset


@overload
def rasterio_open(
    path: AnyPath,
    mode: Literal["r"] = ...,
    **kwargs,
) -> AbstractContextManager[DatasetReader]: ...
@overload
def rasterio_open(
    path: AnyPath,
    mode: Literal["r+", "w", "w+"],
    **kwargs,
) -> AbstractContextManager[DatasetWriter]: ...
def rasterio_open(
    path: AnyPath,
    mode: Literal["r", "r+", "w", "w+"] = "r",
    **kwargs,
) -> AbstractContextManager[DatasetReader | DatasetWriter]:
    """Open a rasterio dataset from a local path or an S3-backed UPath.

    A drop-in replacement for ``rasterio.open`` that automatically injects
    GDAL environment variables derived from a UPath's fsspec storage options,
    enabling reads and writes against S3-compatible object stores without
    manually managing credentials.

    Parameters
    ----------
    path : str | Path | UPath
        Path to the raster file. A ``UPath`` with protocol ``s3`` triggers
        credential extraction from its underlying fsspec filesystem. A plain
        ``Path`` or ``str`` is treated as a local file.
    mode : {"r", "r+", "w", "w+"}, optional
        File mode passed directly to ``rasterio.open``. Defaults to ``"r"``.
    **kwargs : Any
        Additional keyword arguments forwarded to ``rasterio.open``, such as
        ``driver``, ``width``, ``height``, ``count``, ``dtype``, or ``crs``.

    Returns
    -------
    AbstractContextManager[rasterio.DatasetReader]
        When ``mode="r"``.
    AbstractContextManager[rasterio.DatasetWriter]
        When ``mode`` is ``"r+"``, ``"w"``, or ``"w+"``.

    Raises
    ------
    ValueError
        If ``path`` uses a protocol other than local (``""``) or ``"s3"``.
    """
    return _rasterio_open(path, mode, **kwargs)


def build_rasterio_profile(*profiles: dict[str, Any] | None) -> dict[str, Any]:
    """Build a rasterio profile by merging the provided profiles, starting from defaults.

    Each argument is applied in order, so later profiles override earlier ones.

    The built-in defaults are::

        driver     = "GTiff"
        tiled      = True
        blockxsize = 512
        blockysize = 512
        interleave = "pixel"
        compress   = "deflate"

    The caller must supply ``dtype``, ``nodata``, ``count``,
    ``width``, and ``height`` before passing the profile to ``rasterio.open``.

    Parameters
    ----------
    *profiles : dict[str, Any] or None
        Profile dicts to merge, in order of increasing precedence.

    Returns
    -------
    dict[str, Any]
        Merged profile starting from the defaults above.
    """
    dst_profile = RASTERIO_PROFILE_DEFAULTS.copy()
    for profile in profiles:
        if profile is not None:
            dst_profile.update(profile)
    return dst_profile


def _inject_band_metadata(
    dst_file: AnyPath,
    *,
    band_names: list[str] | None = None,
    color_interp: list[ColorInterp] | None = None,
) -> None:
    """Inject band descriptions and color interpretation into a raster file.

    Supports both VRT files (used as COG source) and GeoTIFF COGs. For COGs,
    GDAL requires the IGNORE_COG_LAYOUT_BREAK open option to allow in-place
    updates without losing the COG layout.

    Parameters
    ----------
    dst_file : str | Path | UPath
        Target file to update (VRT or GeoTIFF).
    band_names : list[str] | None
        Band descriptions to set, one per band.
    color_interp : list[ColorInterp] | None
        Color interpretation per band.
    """
    if band_names is None and color_interp is None:
        return
    with rasterio_open(dst_file, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as ds:
        if band_names is not None:
            for i, name in enumerate(band_names, 1):
                ds.set_band_description(i, name)
        if color_interp is not None:
            ds.colorinterp = color_interp


def get_utm_zone_string(projparams: Any) -> str:
    """Extract zero-padded UTM zone identifier.

    Parameters
    ----------
    projparams : Any
        An object that can initialize a pyproj.CRS class instance,
        e.g., a EPSG code or any object that implements `to_wkt()`.

    Returns
    -------
    str
        UTM zone identifier (e.g., '32N', '01S')

    Raises
    ------
    ValueError
        If UTM zone cannot be extracted.
    """
    try:
        crs = CRS(projparams)
    except CRSError as err:
        msg = f"Invalid `projparams` {projparams!r}, could not initialize `pyproj.CRS`."
        raise ValueError(msg) from err

    utm_zone = crs.utm_zone
    if utm_zone is None:
        msg = f"Could not extract CRS identifier from: {crs.name}"
        raise ValueError(msg)
    zone_number = utm_zone[:-1]
    hemisphere_letter = utm_zone[-1]

    return f"{int(zone_number):02d}{hemisphere_letter}"


def get_epsg_string(projparams: Any) -> str:
    """Extract CRS identifier string from CRS object.

    Parameters
    ----------
    projparams : Any
        An object that can initialize a pyproj.CRS class instance,
        e.g., a EPSG code or any object that implements `to_wkt()`.

    Returns
    -------
    str
        'EPSG:' + EPSG code identifier (e.g., 'EPSG:4326')

    Raises
    ------
    ValueError
        If EPSG code cannot be extracted.
    """
    try:
        crs = CRS(projparams)
    except CRSError as err:
        msg = f"Invalid `projparams` {projparams!r}, could not initialize `pyproj.CRS`."
        raise ValueError(msg) from err

    epsg_code = crs.to_epsg()
    if epsg_code is None:
        msg = f"Could not extract EPSG code from: {crs.name}"
        raise ValueError(msg)

    return f"EPSG:{epsg_code}"


def utm_zone_to_crs(utm_zone: str) -> CRS:
    """Create a CRS from an UTM zone string.

    Parameters
    ----------
    utm_zone: str
        UTM zone identifier (e.g., '32N', '01S')

    Returns
    -------
    pyproj.CRS
        The coordinate reference system.

    Raises
    ------
    ValueError
        If ``utm_zone`` is malformed, the zone is outside 1-60, or the
        hemisphere is not ``'N'``/``'S'``.
    """
    hemisphere = utm_zone[-1:]
    if hemisphere not in ("N", "S"):
        msg = f"UTM zone hemisphere must be 'N' or 'S', got {utm_zone!r}."
        raise ValueError(msg)
    try:
        zone_number = int(utm_zone[:-1])
    except ValueError as err:
        msg = f"Invalid UTM zone string {utm_zone!r}: zone number is not an integer."
        raise ValueError(msg) from err
    if not 1 <= zone_number <= 60:  # noqa: PLR2004
        msg = f"UTM zone must be in [1, 60], got {zone_number} from {utm_zone!r}."
        raise ValueError(msg)
    # Create UTM CRS: northern hemisphere uses EPSG:326xx, southern uses EPSG:327xx
    epsg_code = 32600 + zone_number if hemisphere == "N" else 32700 + zone_number
    return CRS.from_epsg(epsg_code)


def group_tiffs_by_crs(src_tiffs: Sequence[AnyPath]) -> dict[str, list[UPath]]:
    """Group GeoTIFF files by their coordinate reference system.

    Parameters
    ----------
    src_tiffs : Sequence[str | Path | UPath]
        Sequence of paths to GeoTIFF files to group.

    Returns
    -------
    dict[str, list[UPath]]
        Dictionary mapping CRS identifiers to lists of ``UPath`` instances.
        UTM zones use format 'UTM<XX>[N|S]' (e.g., 'UTM32N', 'UTM01S').
        Non-UTM projections use format 'EPSG<XXXX>' (e.g., 'EPSG4326').

    Raises
    ------
    ValueError
        If src_tiffs is empty.
    TypeError
        If CRS cannot be read from any file.
    FileNotFoundError
        If any source file does not exist.
    """
    if not src_tiffs:
        msg = "src_tiffs list cannot be empty"
        raise ValueError(msg)

    groups: dict[str, list[UPath]] = {}

    for src_tiff in src_tiffs:
        src_path = UPath(src_tiff)

        # Validate file exists
        if not src_path.exists():
            msg = f"Source file not found: {src_path}"
            raise FileNotFoundError(msg)

        # Read CRS
        with rasterio_open(src_path) as src:
            if src.crs is None:
                msg = f"File has no CRS: {src_path}"
                raise TypeError(msg)

            crs = CRS.from_user_input(src.crs)
            try:
                utm_str = get_utm_zone_string(crs)
                crs_str = "UTM" + utm_str
            except ValueError:
                epsg_str = get_epsg_string(crs)
                crs_str = epsg_str.replace(":", "")

        # Add to group
        if crs_str not in groups:
            groups[crs_str] = []
        groups[crs_str].append(src_path)

        logger.debug("Grouped %s into CRS group '%s'", src_path.name, crs_str)

    logger.info("Grouped %d files into %d CRS groups", len(src_tiffs), len(groups))
    return groups


def merge_tiffs(
    src_files: Sequence[AnyPath],
    dst_file: AnyPath,
    *,
    method: str = "first",
    mem_limit_mb: int = 10_000,
    profile: dict[str, Any] | None = None,
) -> None:
    """Merge multiple GeoTIFF files into a single file.

    Uses rasterio.merge.merge() to combine overlapping rasters.
    Band names are copied from the first source file.

    Parameters
    ----------
    src_files : Sequence[str | Path | UPath]
        Sequence of source GeoTIFF files to merge. Must all have same CRS.
    dst_file : str | Path | UPath
        Path to output merged GeoTIFF.
    method : str, default "first"
        Method for handling overlapping pixels. Options:
        - "first": Use value from first raster in list
        - "last": Use value from last raster in list
        - "min": Use minimum value
        - "max": Use maximum value
    mem_limit_mb : int, default 10000
        Memory limit in megabytes for merge operation. Controls how much
        data is read into memory at once. Default is 10GB.
    profile : dict[str, Any] or None, optional
        Custom rasterio profile settings. If None, uses deflate compression
        with 512x512 tiling.

    Raises
    ------
    ValueError
        If src_files is empty.
        If files have different CRS (when validate_crs=True).
    FileNotFoundError
        If any source file does not exist.
    RuntimeError
        If merge operation fails.
    """
    if not src_files:
        msg = "src_files list cannot be empty"
        raise ValueError(msg)

    dst_file = UPath(dst_file)
    src_paths = [UPath(f) for f in src_files]

    # Validate source files exist
    for src_path in src_paths:
        if not src_path.exists():
            msg = f"Source file not found: {src_path}"
            raise FileNotFoundError(msg)

    logger.info("Merging %d GeoTIFFs into '%s'", len(src_files), dst_file.name)

    # Read the reference profile/descriptions and validate a common CRS. A
    # mismatched CRS would make rasterio.merge silently misplace pixels.
    ref_crs = None
    src_profile: dict[str, Any] = {}
    descriptions: tuple[str | None, ...] = ()
    for i, src_path in enumerate(src_paths):
        with rasterio_open(src_path) as src:
            if i == 0:
                src_profile = dict(src.profile)
                descriptions = src.descriptions
                ref_crs = src.crs
            elif src.crs != ref_crs:
                msg = (
                    f"All source files must share a CRS; '{src_path.name}' has {src.crs}, "
                    f"expected {ref_crs}."
                )
                raise ValueError(msg)

    # Rasterio returns a tuple that may contain None entries. Only carry the
    # descriptions forward when every band actually has one.
    if descriptions and all(d is not None for d in descriptions):
        band_names: list[str] | None = [d for d in descriptions if d is not None]
    else:
        band_names = None

    dst_profile = build_rasterio_profile(src_profile, profile)

    # Prefer an S3 path (source or dst) so credentials are installed for the
    # /vsis3/ read/write paths, honoring the module's transparent-S3 contract.
    env_path: AnyPath = dst_file if dst_file.protocol == "s3" else src_paths[0]

    try:
        with _env_for_path(env_path):
            merge(
                [str(p) for p in src_paths],
                method=method,
                mem_limit=mem_limit_mb,
                dst_path=str(dst_file),
                dst_kwds=dst_profile,
            )
            _inject_band_metadata(dst_file, band_names=band_names)
    except Exception as e:
        msg = f"Failed to merge GeoTIFFs: {e}"
        raise RuntimeError(msg) from e

    logger.info("Successfully merged into '%s'", dst_file.name)


def rewrite_tiff(
    src_file: AnyPath,
    dst_file: AnyPath,
    profile: dict[str, Any] | None = None,
    band_names: list[str] | None = None,
    color_interp: list[ColorInterp] | None = None,
    *,
    move: bool = False,
) -> None:
    """Rewrite a GeoTIFF, optionally to a different storage backend.

    Useful for applying compression/tiling, renaming bands, or copying files
    between local disk and S3, or between two different S3 backends (each side's
    credentials are applied independently). Source-side band descriptions, color
    interpretation, NODATA, and tags are preserved automatically.

    The rewrite is always staged (a unique local sibling temp file, or an
    in-memory image for S3 destinations) and only swapped into place after the
    full copy and metadata injection succeed. A pre-existing destination is
    therefore left untouched on any failure, and the source is only deleted
    when ``move=True`` (and never when it fails).

    Parameters
    ----------
    src_file : str | Path | UPath
        Source GeoTIFF (local or S3).
    dst_file : str | Path | UPath
        Destination GeoTIFF (local or S3). May be the same as ``src_file``,
        in which case the file is rewritten in place via the staging file.
    profile : dict[str, Any] or None, optional
        Custom rasterio profile settings. If None, uses deflate compression
        with 512x512 tiling.
    band_names : list[str] or None, optional
        New band descriptions, one per band. If None, the source's existing
        band descriptions are preserved.
    color_interp : list[ColorInterp] or None, optional
        New per-band color interpretation. If None, the source's existing
        color interpretation is preserved.
    move : bool, default False
        If ``True``, delete ``src_file`` after a successful rewrite to a
        different path (a move). If ``False`` (the default), the source is
        preserved (a copy). Ignored when ``src_file == dst_file``.

    Raises
    ------
    RuntimeError
        If the rewrite fails.
    """
    src_file = UPath(src_file)
    dst_file = UPath(dst_file)
    in_place = src_file == dst_file

    dst_profile = build_rasterio_profile(profile)
    driver = dst_profile.pop("driver", "GTiff")
    # Strip rasterio open()-only keys that are not GTiff creation options.
    # Otherwise, GDAL warns on unrecognised ones.
    for _k in ("dtype", "nodata", "crs", "transform", "count", "width", "height"):
        dst_profile.pop(_k, None)
    if driver != "GTiff":
        # RASTERIO_PROFILE_DEFAULTS carries GTiff-specific creation options
        # (e.g. blockxsize/blockysize instead of the COG driver's BLOCKSIZE)
        # that other drivers don't recognise; drop them so callers can target
        # e.g. driver="COG" via `profile` without GDAL warning on every key.
        for _k in ("tiled", "blockxsize", "blockysize", "interleave"):
            dst_profile.pop(_k, None)

    # Each stage installs the source-read and destination-write credentials
    # independently (see the stage helpers), so src and dst may live on
    # different S3 backends.
    stage = _rewrite_via_memory if dst_file.protocol == "s3" else _rewrite_via_tempfile
    try:
        stage(
            src_file,
            dst_file,
            driver=driver,
            dst_profile=dst_profile,
            band_names=band_names,
            color_interp=color_interp,
        )
    except Exception as e:
        msg = f"Failed to rewrite GeoTIFF: {e}"
        raise RuntimeError(msg) from e

    if move and not in_place:
        src_file.unlink()

    logger.debug("Rewrote GeoTIFF from '%s' to '%s'", src_file.name, dst_file.name)


def _rewrite_via_memory(
    src_file: UPath,
    dst_file: UPath,
    *,
    driver: str,
    dst_profile: dict[str, Any],
    band_names: list[str] | None,
    color_interp: list[ColorInterp] | None,
) -> None:
    """Stage a rewrite entirely in memory, then atomically PutObject to S3.

    GDAL CreateCopy can't read and write the same file at once, and on S3 a
    sibling temp object plus an fsspec rename hits stale-listing errors because
    GDAL's own S3 writes never touch s3fs's directory cache. Staging in memory
    and writing the key in one shot sidesteps both: a failure never touches an
    existing destination object (PutObject is atomic per key).

    The read and write are separate copies bridged by the in-memory image, so
    each runs under its own credentialed env; ``src_file`` and ``dst_file`` may
    therefore sit on different S3 backends.
    """
    with MemoryFile() as memfile:
        with _env_for_path(src_file):
            rasterio.shutil.copy(_to_vsi_uri(src_file), memfile.name, driver="GTiff")
        _inject_band_metadata(memfile.name, band_names=band_names, color_interp=color_interp)
        with _env_for_path(dst_file):
            rasterio.shutil.copy(memfile.name, _to_vsi_uri(dst_file), driver=driver, **dst_profile)


def _rewrite_via_tempfile(
    src_file: UPath,
    dst_file: UPath,
    *,
    driver: str,
    dst_profile: dict[str, Any],
    band_names: list[str] | None,
    color_interp: list[ColorInterp] | None,
) -> None:
    """Stage a local rewrite to a unique sibling temp, then atomically replace.

    The unique suffix avoids collisions between concurrent writers, and the
    destination is only replaced after the copy and metadata injection both
    succeed, so a pre-existing destination survives any failure. The
    destination is always local here, so only the source read needs credentials.
    """
    work_dst = dst_file.with_name(f".{dst_file.name}.{token_hex(8)}.tmp")
    try:
        with _env_for_path(src_file):
            rasterio.shutil.copy(
                _to_vsi_uri(src_file),
                _to_vsi_uri(work_dst),
                driver=driver,
                **dst_profile,
            )
        _inject_band_metadata(work_dst, band_names=band_names, color_interp=color_interp)
        work_dst.replace(dst_file)
    except BaseException:
        work_dst.unlink(missing_ok=True)
        raise


def write_cog(
    data: np.ndarray,
    dst_file: AnyPath,
    profile: dict[str, Any],
    *,
    band_names: list[str] | None = None,
    color_interp: list[ColorInterp] | None = None,
) -> None:
    """Write an in-memory array as a Cloud Optimized GeoTIFF.

    Stages ``data`` in a ``rasterio.io.MemoryFile``, then copies it straight
    to a COG at ``dst_file``.

    Parameters
    ----------
    data : numpy.ndarray
        Array to write, shaped ``(height, width)`` for a single band or
        ``(bands, height, width)`` for multiple.
    dst_file : str | Path | UPath
        Destination COG (local or S3).
    profile : dict[str, Any]
        Rasterio creation profile for the staging write, as for
        ``rasterio.open(..., "w")``: must include ``dtype``, ``count``,
        ``width``, ``height``, ``crs``, ``transform``, and ``nodata``. Merged
        onto :data:`RASTERIO_PROFILE_DEFAULTS` via :func:`build_rasterio_profile`.
    band_names : list[str] or None, optional
        Band descriptions, one per band.
    color_interp : list[ColorInterp] or None, optional
        Per-band color interpretation.

    Raises
    ------
    RuntimeError
        If the write fails.
    """
    dst_file = UPath(dst_file)
    # On S3, write the key directly: PutObject is atomic per key, so a failed
    # write leaves nothing behind, and there's no local rename needed (which
    # would otherwise hit stale-listing errors in s3fs, since GDAL's own S3
    # writes never touch its directory cache). Locally, stage via a sibling
    # temp file and swap in with an atomic rename, so a resumed run can't
    # mistake a partially-written file for a finished one.
    is_s3 = dst_file.protocol == "s3"
    work_dst = dst_file if is_s3 else dst_file.with_name(f".{dst_file.name}.{token_hex(8)}.tmp")

    mem_profile = build_rasterio_profile(profile)
    mem_profile.pop("driver", None)  # staging file is always GTiff
    # COG-driver-only options are rejected or ignored by GTiff, and the final
    # COG creation options come from COG_PROFILE below regardless.
    for key in _COG_CREATION_OPTIONS:
        mem_profile.pop(key, None)

    try:
        with MemoryFile() as memfile:
            with memfile.open(driver="GTiff", **mem_profile) as mem_ds:
                if data.ndim == 2:  # noqa: PLR2004
                    mem_ds.write(data, 1)
                else:
                    mem_ds.write(data)
                if band_names is not None:
                    for i, name in enumerate(band_names, 1):
                        mem_ds.set_band_description(i, name)
                if color_interp is not None:
                    mem_ds.colorinterp = color_interp

            with _env_for_path(dst_file):
                rasterio.shutil.copy(memfile.name, _to_vsi_uri(work_dst), **COG_PROFILE)

        if not is_s3:
            work_dst.replace(dst_file)
    except Exception as e:
        if not is_s3:
            work_dst.unlink(missing_ok=True)
        msg = f"Failed to write COG: {e}"
        raise RuntimeError(msg) from e

    logger.debug("Wrote COG to '%s'", dst_file.name)
