"""Rasterio helpers for local and S3-backed files."""

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
    from collections.abc import Generator, Mapping, Sequence

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
    # AVERAGE skips NODATA pixels.
    "overview_resampling": "AVERAGE",
    # Select the predictor from the data type.
    "predictor": "YES",
    "overview_predictor": "YES",
}
"""Default GDAL ``COG`` creation profile."""

_COG_CREATION_OPTIONS = (
    "blocksize",
    "compress",
    "interleave",
    "level",
    "max_z_error",
    "max_z_error_overview",
    "overviews",
    "overview_count",
    "overview_quality",
    "overview_resampling",
    "overview_predictor",
    "overview_compress",
    "predictor",
    "quality",
)


@contextmanager
def _env_for_path(path: AnyPath) -> Generator[None]:
    """Enter a local or credentialed S3 rasterio environment.

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
    """Return a GDAL path, converting S3 paths to ``/vsis3/`` URIs.

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
    """Yield a dataset in the appropriate rasterio environment."""
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
    """Open a local or S3-backed rasterio dataset.

    Parameters
    ----------
    path : str | Path | UPath
        Raster path. An S3 ``UPath`` supplies its storage options to GDAL.
    mode : {"r", "r+", "w", "w+"}, optional
        File mode passed directly to ``rasterio.open``. Defaults to ``"r"``.
    **kwargs : Any
        Additional arguments for ``rasterio.open``.

    Returns
    -------
    AbstractContextManager
        Context manager yielding a dataset reader or writer.

    Raises
    ------
    ValueError
        If ``path`` uses a protocol other than local (``""``) or ``"s3"``.
    """
    return _rasterio_open(path, mode, **kwargs)


def build_rasterio_profile(*profiles: dict[str, Any] | None) -> dict[str, Any]:
    """Merge profiles over :data:`RASTERIO_PROFILE_DEFAULTS` in order.

    Parameters
    ----------
    *profiles : dict[str, Any] or None
        Profile dicts to merge, in order of increasing precedence.

    Returns
    -------
    dict[str, Any]
        Merged rasterio profile.
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
    tags: Mapping[str, str] | None = None,
    band_tags: Sequence[Mapping[str, str]] | None = None,
    units: list[str] | None = None,
) -> None:
    """Update raster metadata in place."""
    if all(x is None for x in (band_names, color_interp, tags, band_tags, units)):
        return
    with rasterio_open(dst_file, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as ds:
        _apply_band_metadata(
            ds,
            band_names=band_names,
            color_interp=color_interp,
            tags=tags,
            band_tags=band_tags,
            units=units,
        )


def _check_band_lengths(count: int, **sequences: Sequence[Any] | None) -> None:
    """Reject a per-band sequence that does not hold exactly one entry per band."""
    for name, values in sequences.items():
        if values is not None and len(values) != count:
            msg = f"{name} must hold one entry per band; got {len(values)} for {count} bands."
            raise ValueError(msg)


def _apply_band_metadata(
    ds: DatasetWriter,
    *,
    band_names: list[str] | None = None,
    color_interp: list[ColorInterp] | None = None,
    tags: Mapping[str, str] | None = None,
    band_tags: Sequence[Mapping[str, str]] | None = None,
    units: list[str] | None = None,
) -> None:
    """Write metadata to an open dataset.

    Raises
    ------
    ValueError
        If a per-band sequence length does not match the band count.
    """
    _check_band_lengths(
        ds.count,
        band_names=band_names,
        color_interp=color_interp,
        band_tags=band_tags,
        units=units,
    )
    if band_names is not None:
        for i, name in enumerate(band_names, 1):
            ds.set_band_description(i, name)
    if color_interp is not None:
        ds.colorinterp = color_interp
    if tags is not None:
        ds.update_tags(**tags)
    if band_tags is not None:
        for i, band in enumerate(band_tags, 1):
            ds.update_tags(i, **band)
    if units is not None:
        for i, unit in enumerate(units, 1):
            ds.set_band_unit(i, unit)


def get_utm_zone_string(projparams: Any) -> str:
    """Return a zero-padded UTM zone identifier.

    Parameters
    ----------
    projparams : Any
        Value accepted by ``pyproj.CRS``.

    Returns
    -------
    str
        UTM zone such as ``"32N"`` or ``"01S"``.

    Raises
    ------
    ValueError
        If the CRS is invalid or has no UTM zone.
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
    """Return an ``EPSG:<code>`` identifier.

    Parameters
    ----------
    projparams : Any
        Value accepted by ``pyproj.CRS``.

    Returns
    -------
    str
        Identifier such as ``"EPSG:4326"``.

    Raises
    ------
    ValueError
        If the CRS is invalid or has no EPSG code.
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
    """Create a CRS from a UTM zone.

    Parameters
    ----------
    utm_zone : str
        UTM zone such as ``"32N"`` or ``"01S"``.

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
    epsg_code = 32600 + zone_number if hemisphere == "N" else 32700 + zone_number
    return CRS.from_epsg(epsg_code)


def group_tiffs_by_crs(src_tiffs: Sequence[AnyPath]) -> dict[str, list[UPath]]:
    """Group GeoTIFFs by CRS.

    Parameters
    ----------
    src_tiffs : Sequence[str | Path | UPath]
        GeoTIFF paths.

    Returns
    -------
    dict[str, list[UPath]]
        Paths keyed by ``UTM<zone>`` or ``EPSG<code>``.

    Raises
    ------
    ValueError
        If ``src_tiffs`` is empty.
    TypeError
        If a file has no CRS.
    FileNotFoundError
        If any source file does not exist.
    """
    if not src_tiffs:
        msg = "src_tiffs list cannot be empty"
        raise ValueError(msg)

    groups: dict[str, list[UPath]] = {}

    for src_tiff in src_tiffs:
        src_path = UPath(src_tiff)

        if not src_path.exists():
            msg = f"Source file not found: {src_path}"
            raise FileNotFoundError(msg)

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
    """Merge GeoTIFFs, preserving band names from the first source.

    Parameters
    ----------
    src_files : Sequence[str | Path | UPath]
        Source files, all in the same CRS.
    dst_file : str | Path | UPath
        Output file.
    method : str, default "first"
        Overlap method accepted by ``rasterio.merge.merge``.
    mem_limit_mb : int, default 10000
        Merge memory limit in megabytes.
    profile : dict[str, Any] or None, optional
        Output profile overrides.

    Raises
    ------
    ValueError
        If ``src_files`` is empty or contains different CRSs.
    FileNotFoundError
        If any source file does not exist.
    RuntimeError
        If merging fails.
    """
    if not src_files:
        msg = "src_files list cannot be empty"
        raise ValueError(msg)

    dst_file = UPath(dst_file)
    src_paths = [UPath(f) for f in src_files]

    for src_path in src_paths:
        if not src_path.exists():
            msg = f"Source file not found: {src_path}"
            raise FileNotFoundError(msg)

    logger.info("Merging %d GeoTIFFs into '%s'", len(src_files), dst_file.name)

    # rasterio.merge does not reject mismatched CRSs.
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

    # Preserve descriptions only when every band has one.
    if descriptions and all(d is not None for d in descriptions):
        band_names: list[str] | None = [d for d in descriptions if d is not None]
    else:
        band_names = None

    dst_profile = build_rasterio_profile(src_profile, profile)

    # Use an S3 operand to configure credentials when needed.
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
    tags: Mapping[str, str] | None = None,
    band_tags: Sequence[Mapping[str, str]] | None = None,
    units: list[str] | None = None,
    move: bool = False,
) -> None:
    """Rewrite a GeoTIFF locally or on S3 while preserving metadata.

    The staged write leaves an existing destination untouched on failure.

    Parameters
    ----------
    src_file : str | Path | UPath
        Source file.
    dst_file : str | Path | UPath
        Destination file. May equal ``src_file`` for an in-place rewrite.
    profile : dict[str, Any] or None, optional
        Output profile overrides.
    band_names : list[str] or None, optional
        Band descriptions. Defaults to the source descriptions.
    color_interp : list[ColorInterp] or None, optional
        Color interpretation. Defaults to the source values.
    tags : Mapping[str, str] or None, optional
        Dataset tags to merge.
    band_tags : Sequence[Mapping[str, str]] or None, optional
        Per-band tags to merge.
    units : list[str] or None, optional
        Band units. Defaults to the source values.
    move : bool, default False
        Delete the source after a successful rewrite to a different path.

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
    # Remove dataset metadata from the creation options.
    for _k in ("dtype", "nodata", "crs", "transform", "count", "width", "height"):
        dst_profile.pop(_k, None)
    if driver != "GTiff":
        # Remove GTiff-only defaults.
        for _k in ("tiled", "blockxsize", "blockysize", "interleave"):
            dst_profile.pop(_k, None)

    # The stage helpers configure each backend independently.
    stage = _rewrite_via_memory if dst_file.protocol == "s3" else _rewrite_via_tempfile
    try:
        stage(
            src_file,
            dst_file,
            driver=driver,
            dst_profile=dst_profile,
            band_names=band_names,
            color_interp=color_interp,
            tags=tags,
            band_tags=band_tags,
            units=units,
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
    tags: Mapping[str, str] | None,
    band_tags: Sequence[Mapping[str, str]] | None,
    units: list[str] | None,
) -> None:
    """Stage in memory, then atomically write the S3 destination."""
    with MemoryFile() as memfile:
        with _env_for_path(src_file):
            rasterio.shutil.copy(_to_vsi_uri(src_file), memfile.name, driver="GTiff")
        _inject_band_metadata(
            memfile.name,
            band_names=band_names,
            color_interp=color_interp,
            tags=tags,
            band_tags=band_tags,
            units=units,
        )
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
    tags: Mapping[str, str] | None,
    band_tags: Sequence[Mapping[str, str]] | None,
    units: list[str] | None,
) -> None:
    """Stage beside a local destination, then atomically replace it."""
    work_dst = dst_file.with_name(f".{dst_file.name}.{token_hex(8)}.tmp")
    needs_gtiff_stage = driver != "GTiff" and any(
        x is not None for x in (band_names, tags, band_tags, units)
    )
    work_stage = (
        dst_file.with_name(f".{dst_file.name}.{token_hex(8)}.stage.tif")
        if needs_gtiff_stage
        else None
    )
    inject_into = work_stage if work_stage is not None else work_dst
    try:
        with _env_for_path(src_file):
            rasterio.shutil.copy(
                _to_vsi_uri(src_file),
                _to_vsi_uri(inject_into),
                driver="GTiff" if work_stage is not None else driver,
                **({} if work_stage is not None else dst_profile),
            )
        _inject_band_metadata(
            inject_into,
            band_names=band_names,
            color_interp=color_interp,
            tags=tags,
            band_tags=band_tags,
            units=units,
        )
        if work_stage is not None:
            rasterio.shutil.copy(
                _to_vsi_uri(work_stage),
                _to_vsi_uri(work_dst),
                driver=driver,
                **dst_profile,
            )
        work_dst.replace(dst_file)
    except BaseException:
        work_dst.unlink(missing_ok=True)
        raise
    finally:
        if work_stage is not None:
            work_stage.unlink(missing_ok=True)


def write_cog(
    data: np.ndarray,
    dst_file: AnyPath,
    profile: dict[str, Any],
    *,
    band_names: list[str] | None = None,
    color_interp: list[ColorInterp] | None = None,
    tags: Mapping[str, str] | None = None,
    band_tags: Sequence[Mapping[str, str]] | None = None,
    units: list[str] | None = None,
) -> None:
    """Write an in-memory array as a Cloud Optimized GeoTIFF.

    Parameters
    ----------
    data : numpy.ndarray
        Single- or multi-band array.
    dst_file : str | Path | UPath
        Destination COG (local or S3).
    profile : dict[str, Any]
        Raster metadata and native GDAL ``COG`` creation options.
    band_names : list[str] or None, optional
        Band descriptions, one per band.
    color_interp : list[ColorInterp] or None, optional
        Per-band color interpretation.
    tags : Mapping[str, str] or None, optional
        Dataset-level tags.
    band_tags : Sequence[Mapping[str, str]] or None, optional
        Per-band tags, one mapping per band.
    units : list[str] or None, optional
        Band unit strings, one per band.

    Raises
    ------
    RuntimeError
        If the write fails.
    """
    dst_file = UPath(dst_file)
    # S3 PutObject is atomic; local writes use an atomic rename.
    is_s3 = dst_file.protocol == "s3"
    work_dst = dst_file if is_s3 else dst_file.with_name(f".{dst_file.name}.{token_hex(8)}.tmp")

    mem_profile = build_rasterio_profile(profile)
    mem_profile.pop("driver", None)
    # Keep the staging image lossless to avoid double quantisation.
    for key in _COG_CREATION_OPTIONS:
        mem_profile.pop(key, None)
    mem_profile["compress"] = "deflate"

    cog_profile = COG_PROFILE | {
        key: profile[key] for key in _COG_CREATION_OPTIONS if key in profile
    }
    # LERC does not use predictors.
    if str(cog_profile.get("compress", "")).lower().startswith("lerc"):
        cog_profile.pop("predictor", None)
        cog_profile.pop("overview_predictor", None)

    try:
        with MemoryFile() as memfile:
            with memfile.open(driver="GTiff", **mem_profile) as mem_ds:
                if data.ndim == 2:  # noqa: PLR2004
                    mem_ds.write(data, 1)
                else:
                    mem_ds.write(data)
                # Post-write metadata updates break COG layout.
                _apply_band_metadata(
                    mem_ds,
                    band_names=band_names,
                    color_interp=color_interp,
                    tags=tags,
                    band_tags=band_tags,
                    units=units,
                )

            with _env_for_path(dst_file):
                rasterio.shutil.copy(memfile.name, _to_vsi_uri(work_dst), **cog_profile)

        if not is_s3:
            work_dst.replace(dst_file)
    except Exception as e:
        if not is_s3:
            work_dst.unlink(missing_ok=True)
        msg = f"Failed to write COG: {e}"
        raise RuntimeError(msg) from e

    logger.debug("Wrote COG to '%s'", dst_file.name)
