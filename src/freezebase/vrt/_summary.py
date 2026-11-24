"""Summaries of a VRT for its text and HTML reprs, built from the model alone without I/O."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass, replace
import html
import json
import math
import numbers
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree as ET

from rasterio.enums import ColorInterp
import rasterio.windows
from rasterio.windows import Window
from upath import UPath

from freezebase.vrt import _map
from freezebase.vrt._crs import crs_label, parse_srs, wkt_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from rasterio.crs import CRS

    from freezebase.vrt._model import VRT, Band, Source, SourceRef

MAX_BANDS = 10
MAX_FILES = 5
MAX_CHILDREN = 5
# Bounds the recursion into nested VRTs; MAX_NODES bounds the number of steps.
MAX_DEPTH = 6
MAX_NODES = 40
MAX_CHARS = 60
# Deeper VRTs are compared by identity, which bounds the recursion of the comparison.
_KEY_DEPTH = 100
# Bounds the plain views unwrapped in a row.
_MAX_VIEWS = 10_000

_TIMES = "\N{MULTIPLICATION SIGN}"
_RGBA = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
_UNITS = {
    "metre": "m",
    "meter": "m",
    "kilometre": "km",
    "kilometer": "km",
    "degree": "°",
    "foot": "ft",
    "US survey foot": "US ft",
}
# Integers up to this size print in full; larger ones as floats, as they are stored.
_MAX_INTEGER = 2**64


@dataclass(frozen=True)
class File:
    """A file source: its short display name and its full path."""

    name: str
    path: str


@dataclass(frozen=True)
class Input:
    """One input of a step: a file, an earlier step, or inputs left out."""

    kind: Literal["file", "step", "more"]
    text: str
    title: str = ""
    number: int = 0


@dataclass(frozen=True)
class Step:
    """One step that builds the VRT, numbered in the order the steps run."""

    number: int
    verb: str
    detail: str
    inputs: tuple[Input, ...]
    grid: str
    # Whether the step reads only the step above it, so its inputs go unnamed.
    joined: bool = False


@dataclass(frozen=True)
class BandRow:
    """Display values of one band."""

    index: int
    description: str
    dtype: str
    nodata: str | None
    unit: str | None


@dataclass(frozen=True)
class Summary:
    """Everything both reprs show."""

    header: tuple[str, ...]
    size: str
    crs: str
    crs_name: str | None
    resolution: str
    bounds: tuple[str, str]
    dtypes: str
    bands: tuple[BandRow, ...]
    more_bands: int
    steps: tuple[Step, ...]
    footprint: _map.Footprint | None = None


def text_repr(vrt: VRT) -> str:
    """Return the text repr of ``vrt``, or a short form if the model is invalid."""
    try:
        return to_text(summarize(vrt))
    except Exception:  # noqa: BLE001  # a repr must not raise
        return safe_short(vrt)


def html_repr(vrt: VRT) -> str:
    """Return the HTML repr of ``vrt``, or a short form if the model is invalid."""
    try:
        summary = summarize(vrt)
        # A map that fails is left out, and the rest still shows.
        with suppress(Exception):
            summary = _with_footprint(summary, vrt)
        return to_html(summary, to_text(summary))
    except Exception:  # noqa: BLE001  # a repr must not raise
        return f"<pre>{html.escape(safe_short(vrt))}</pre>"


def safe_short(vrt: VRT) -> str:
    """Return ``short(vrt)``, or the default object repr if the model is invalid."""
    try:
        return short(vrt)
    except Exception:  # noqa: BLE001  # a repr must not raise
        return object.__repr__(vrt)


def short(vrt: VRT) -> str:
    """One-line description of a VRT, for reprs of objects that hold one."""
    crs = _crs_text(vrt.crs)
    return f"VRT({vrt.width}{_TIMES}{vrt.height}, {crs}, {_plural(len(vrt.bands), 'band')})"


def summarize(vrt: VRT) -> Summary:
    """Summarize ``vrt`` from the model alone."""
    crs = _crs_text(vrt.crs)
    size = f"{vrt.width} {_TIMES} {vrt.height} px"
    res = _resolution(vrt)
    dtypes = _clip(", ".join(dict.fromkeys(vrt.dtypes)))
    header = tuple(
        part for part in (size, _plural(len(vrt.bands), "band"), dtypes, crs, res) if part
    )
    name = None
    if vrt.crs is not None and (wkt := _clip(wkt_name(vrt.crs.to_wkt()))) != crs:
        name = wkt
    rows = tuple(_band_row(index, band) for index, band in enumerate(vrt.bands[:MAX_BANDS], 1))
    return Summary(
        header=header,
        size=size,
        crs=crs,
        crs_name=name,
        resolution=res,
        bounds=_bounds(vrt),
        dtypes=dtypes,
        bands=rows,
        more_bands=max(len(vrt.bands) - MAX_BANDS, 0),
        steps=_Steps().build(vrt),
    )


def _with_footprint(summary: Summary, vrt: VRT) -> Summary:
    """Add the footprints the HTML repr maps."""
    return replace(summary, footprint=_map.footprint(vrt, lambda path: _file(path).path))


def _crs_text(crs: CRS | None) -> str:
    """Label a CRS, or its absence."""
    return _clip(crs_label(crs)) if crs is not None else "no CRS"


def _plural(count: int, noun: str) -> str:
    """Format ``count noun``, adding an s unless the count is one."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _clip(text: str, limit: int = MAX_CHARS) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut, and blank control characters."""
    text = "".join(char if char.isprintable() else " " for char in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _nodata(value: float) -> str:
    """Format a nodata value exactly: integers in full, other floats as Python does."""
    if isinstance(value, numbers.Integral):
        return str(int(value))
    value = float(value)
    if value.is_integer() and abs(value) < _MAX_INTEGER:
        return str(int(value))
    return repr(value)


def _unit(vrt: VRT) -> str:
    """Return the short name of the CRS's unit, or an empty string."""
    if vrt.crs is None:
        return ""
    name = vrt.crs.units_factor[0]
    return _UNITS.get(name, name)


def _with_unit(value: str, unit: str) -> str:
    """Append a unit to a value, a degree sign without a space."""
    if unit == "°":
        return value + unit
    return f"{value} {unit}" if unit else value


def _resolution(vrt: VRT) -> str:
    """Format the pixel size with the CRS's unit."""
    x, y = (f"{value:.6g}" for value in vrt.res)
    return _with_unit(x if x == y else f"{x} {_TIMES} {y}", _unit(vrt))


def _bounds(vrt: VRT) -> tuple[str, str]:
    """Format the extent along x and along y, with the CRS's unit."""
    left, bottom, right, top = vrt.bounds
    unit = _unit(vrt)
    # Low to high, also for a south-up grid, whose bottom is above its top.
    xs, ys = sorted((left, right)), sorted((bottom, top))
    return (
        _with_unit(f"{xs[0]:.12g}{_map.TO}{xs[1]:.12g}", unit),
        _with_unit(f"{ys[0]:.12g}{_map.TO}{ys[1]:.12g}", unit),
    )


def _band_row(index: int, band: Band) -> BandRow:
    """Collect the display values of a band."""
    nodata = None if band.nodata is None else _nodata(band.nodata)
    unit = _clip(band.unit) if band.unit else None
    return BandRow(index, _clip(band.description), band.dtype, nodata, unit)


def _file(path: UPath) -> File:
    """Name a file by its base name, keeping the protocol of a remote one."""
    full, name = str(path), path.name
    if path.protocol in {"http", "https"}:
        # Query strings and user info may hold credentials, such as presigned signatures.
        parts = urlsplit(full)
        full = urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))
        name = PurePosixPath(parts.path).name
    # A bucket root: UPath versions disagree on whether its name is the bucket or empty.
    remote = path.protocol not in {"", "file", "http", "https"}
    if not name or (remote and "/" not in path.path.strip("/")):
        return File(_clip(full), full)
    if path.protocol not in {"", "file"}:
        return File(f"{path.protocol}://…/{_clip(name)}", full)
    return File(_clip(name), full)


def _srs_text(text: str, crs: CRS | None) -> str:
    """Label a CRS serialized in warp options, by its raw text if it did not parse."""
    return _clip(crs_label(crs)) if crs is not None else _clip(text)


def _warp_detail(vrt: VRT, options_xml: str) -> str:
    """Describe a warp by the CRSs it reprojects between, from its serialized options."""
    try:
        # The options come from GDAL or from a VRT the caller chose to read.
        options = ET.fromstring(options_xml)  # noqa: S314
    except ET.ParseError:
        return ""
    source = (options.findtext(".//SourceSRS") or "").strip()
    target = (options.findtext(".//TargetSRS") or "").strip()
    if not source:
        return _crs_text(vrt.crs)
    source_crs = parse_srs(source)
    target_crs = parse_srs(target) if target else vrt.crs
    target_text = _srs_text(target, target_crs) if target else _crs_text(vrt.crs)
    if source_crs is not None and target_crs is not None:
        same = source_crs == target_crs
    else:
        same = source == target
    if same:
        return target_text
    return f"{_srs_text(source, source_crs)} → {target_text}"


def _verb(vrt: VRT, refs: list[SourceRef]) -> tuple[str, str]:
    """Name the function that builds ``vrt`` from its sources, and describe what it does."""
    if vrt.warp is not None:
        return "reproject", _warp_detail(vrt, vrt.warp.options_xml)
    derived = [band.pixel_function.name for band in vrt.bands if band.pixel_function is not None]
    if derived:
        names = _clip(", ".join(dict.fromkeys(derived)))
        partly = len(derived) < len(vrt.bands)
        return (
            "derive",
            f"{names} in {len(derived)} of {len(vrt.bands)} bands" if partly else names,
        )
    if not refs:
        return "empty", ""
    if len(refs) == 1 and (named := _one_input_verb(vrt, refs[0])) is not None:
        return named
    sources = [source for band in vrt.bands for source in band.sources]
    # One input placed in windows is a mosaic of it, which crops or resamples it.
    placed = len(refs) == 1 and any(
        source.src_window is not None or source.dst_window is not None for source in sources
    )
    if placed or any(len(band.sources) > 1 for band in vrt.bands):
        return "mosaic", ""
    return "stack", ""


def _one_input_verb(vrt: VRT, ref: SourceRef) -> tuple[str, str] | None:
    """Name the one-input step that builds ``vrt`` from ``ref``, if it is one."""
    if _is_view(vrt, ref):
        return "from_raster", ""
    for verb, check in (("colorize", _colorized), ("select", _selected), ("clip_box", _clipped)):
        if (detail := check(vrt, ref)) is not None:
            return verb, detail
    return None


def _is_view(vrt: VRT, ref: SourceRef) -> bool:
    """Whether ``vrt`` shows ``ref`` unchanged: each band reads its namesake whole, as is."""
    whole = Window(0, 0, vrt.width, vrt.height)
    for index, band in enumerate(vrt.bands, 1):
        if len(band.sources) != 1 or band.scale is not None or band.offset is not None:
            return False
        source = band.sources[0]
        # A nested VRT on the same grid may be placed whole, as a full-extent clip does.
        placed = source.src_window is not None or source.dst_window is not None
        if placed and (
            isinstance(ref, UPath) or source.src_window != whole or source.dst_window != whole
        ):
            return False
        if source.band != index or source.nodata is not None or source.lut is not None:
            return False
    if isinstance(ref, UPath):
        return True
    same_bands = [band.dtype for band in ref.bands] == [band.dtype for band in vrt.bands]
    return same_bands and _same_grid(vrt, ref)


def _single(vrt: VRT) -> list[Source] | None:
    """Return the one source of each band, or None if a band has more or fewer."""
    if not all(len(band.sources) == 1 for band in vrt.bands):
        return None
    return [band.sources[0] for band in vrt.bands]


def _colorized(vrt: VRT, ref: SourceRef) -> str | None:
    """Describe the color ramp of a ``colorize`` step, or return None if it is none."""
    del ref
    sources = _single(vrt)
    interps = [band.color_interp for band in vrt.bands]
    if (
        sources is None
        or interps != list(_RGBA)
        or any(band.dtype != "uint8" for band in vrt.bands)
        or any(source.lut is None for source in sources)
        or len({source.band for source in sources}) != 1
    ):
        return None
    values = [
        float(value)
        for value, _ in sources[0].lut or ()
        if isinstance(value, numbers.Real) and math.isfinite(value)
    ]
    if not values:
        return ""
    low, high = min(values), max(values)
    return f"{low:g}" if low == high else f"{low:g}{_map.TO}{high:g}"


def _selected(vrt: VRT, ref: SourceRef) -> str | None:
    """Name the bands a ``select`` step reads whole, or return None if it is none."""
    sources = _single(vrt)
    if sources is None or any(_changes(source) for source in sources):
        return None
    if not isinstance(ref, UPath) and (
        not _same_grid(vrt, ref)
        or any(
            not 1 <= source.band <= len(ref.bands)
            or ref.bands[source.band - 1].dtype != band.dtype
            for source, band in zip(sources, vrt.bands, strict=True)
        )
    ):
        return None
    numbers = ", ".join(str(source.band) for source in sources)
    return _clip(f"band {numbers}" if len(sources) == 1 else f"bands {numbers}")


def _clipped(vrt: VRT, ref: SourceRef) -> str | None:
    """Locate the window a ``clip_box`` step cuts from a VRT, or return None if it is none."""
    sources = _single(vrt)
    if sources is None or isinstance(ref, UPath) or ref.crs != vrt.crs:
        return None
    window = sources[0].src_window
    whole = Window(0, 0, vrt.width, vrt.height)
    if (
        window is None
        or vrt.width <= 0
        or vrt.height <= 0
        or (window.width, window.height) != (vrt.width, vrt.height)
        or window == Window(0, 0, ref.width, ref.height)
        or not vrt.transform.almost_equals(rasterio.windows.transform(window, ref.transform))
    ):
        return None
    for index, source in enumerate(sources, 1):
        moved = replace(source, src_window=None, dst_window=None)
        if (
            source.band != index
            or source.src_window != window
            or source.dst_window != whole
            or _changes(moved)
        ):
            return None
    return (
        f"{vrt.width} {_TIMES} {vrt.height} px at col {window.col_off:g}, row {window.row_off:g}"
    )


def _changes(source: Source) -> bool:
    """Whether ``source`` places, masks or maps the pixels it reads."""
    return (
        source.src_window is not None
        or source.dst_window is not None
        or source.nodata is not None
        or source.lut is not None
    )


def _same_grid(vrt: VRT, other: VRT) -> bool:
    """Whether two VRTs share a CRS, transform and size."""
    return (other.crs, other.transform, other.width, other.height) == (
        vrt.crs,
        vrt.transform,
        vrt.width,
        vrt.height,
    )


def _rejoined(vrt: VRT, files: Callable[[VRT], frozenset[str]]) -> VRT:
    """Join the one-band copies ``vrt`` reads its bands through into one input VRT.

    ``derive``, ``clip_box`` and ``colorize`` read each band from a VRT holding only that
    band, so the copies of one input would otherwise show as one step per band. Copies
    that read different files, as a stack of separate inputs does, stay apart.
    """
    refs = vrt.sources
    sources = _single(vrt)
    if (
        vrt.warp is not None
        or len(refs) < 2  # noqa: PLR2004  # one copy needs no joining
        or sources is None
        or len(sources) != len(refs)
        or any(source.path is not ref for source, ref in zip(sources, refs, strict=True))
        or any(source.band != 1 for source in sources)
        # Read the same way: the fields after ``path`` agree.
        or len(
            {
                tuple(_value(getattr(source, f.name)) for f in fields(source)[1:])
                for source in sources
            }
        )
        != 1
    ):
        return vrt
    copies = [ref for ref in refs if not isinstance(ref, UPath)]
    if len(copies) != len(refs) or any(
        copy.warp is not None
        or len(copy.bands) != 1
        or not _same_grid(copies[0], copy)
        or _value((copy.metadata, copy.metadata_domains))
        != _value((copies[0].metadata, copies[0].metadata_domains))
        or files(copy) != files(copies[0])
        for copy in copies
    ):
        return vrt
    joined = replace(copies[0], bands=[copy.bands[0] for copy in copies])
    return replace(
        vrt,
        bands=[
            replace(band, sources=[replace(source, path=joined, band=index)])
            for index, (band, source) in enumerate(zip(vrt.bands, sources, strict=True), 1)
        ],
    )


def _grid_text(vrt: VRT) -> str:
    """Describe the grid of ``vrt`` on one line."""
    return f"{vrt.width} {_TIMES} {vrt.height} px · {_crs_text(vrt.crs)} · {_resolution(vrt)}"


def _value(value: object) -> object:
    """Make a field value hashable, NaN equal to NaN."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, dict):
        return tuple(sorted((str(key), _value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_value(item) for item in value)
    if isinstance(value, Window):
        return tuple(value.flatten())
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            *(_value(getattr(value, item.name)) for item in fields(value)),
        )
    return value


class _Interner:
    """Numbers VRTs so equal ones share a number, keying each node once.

    Comparing nested VRTs with ``==`` revisits shared inputs, so its time grows exponentially
    with depth; a key built from the numbers of the inputs does not.
    """

    def __init__(self) -> None:
        self.by_id: dict[int, int] = {}
        self.by_key: dict[object, int] = {}

    def number(self, vrt: VRT, depth: int = 0) -> int:
        """Return the number of ``vrt``, the same for every equal VRT."""
        if (known := self.by_id.get(id(vrt))) is not None:
            return known
        if depth >= _KEY_DEPTH:
            # Unequal to all others, which at worst lists an equal VRT twice.
            key: object = ("id", id(vrt))
        else:
            warp = vrt.warp
            key = (
                None if vrt.crs is None else vrt.crs.to_wkt(),
                tuple(vrt.transform[:6]),
                vrt.width,
                vrt.height,
                tuple(self.band(band, depth) for band in vrt.bands),
                _value(vrt.metadata),
                _value(vrt.metadata_domains),
                None
                if warp is None
                else (self.ref(warp.source, depth), warp.options_xml, warp.block_size),
            )
        number = self.by_key.setdefault(key, len(self.by_key))
        self.by_id[id(vrt)] = number
        return number

    def ref(self, ref: SourceRef, depth: int) -> object:
        """Key a source: a file by its path, a nested VRT by its number."""
        if isinstance(ref, UPath):
            return ("file", str(ref))
        return ("vrt", self.number(ref, depth + 1))

    def band(self, band: Band, depth: int) -> object:
        """Key a band and its sources."""
        sources = tuple(
            (
                self.ref(source.path, depth),
                *(_value(getattr(source, item.name)) for item in fields(source)[1:]),
            )
            for source in band.sources
        )
        rest = (
            _value(getattr(band, item.name)) for item in fields(band) if item.name != "sources"
        )
        return (sources, *rest)


class _Steps:
    """Lists the steps that build a VRT, its inputs first, each distinct VRT once."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.seen: dict[int, Input] = {}
        self.interner = _Interner()
        # Keeps the joined VRTs alive, so the interner's ids are not reused.
        self.joined: list[VRT] = []
        self.leaves: dict[int, frozenset[str]] = {}

    def build(self, vrt: VRT) -> tuple[Step, ...]:
        """Return the numbered steps; a VRT without sources has none."""
        if (
            vrt.warp is None
            and not any(band.sources for band in vrt.bands)
            and not any(band.pixel_function for band in vrt.bands)
        ):
            return ()
        self.visit(vrt, 0)
        steps = self.steps
        return tuple(
            Step(step.number, step.verb, step.detail, (), step.grid, joined=True)
            if index and step.inputs == (Input("step", f"#{index}", steps[index - 1].verb, index),)
            else step
            for index, step in enumerate(steps)
        )

    def rejoined(self, vrt: VRT) -> VRT:
        """Return ``vrt`` with its one-band copies joined, kept alive for the interner."""
        joined = _rejoined(vrt, self.files)
        if joined is not vrt:
            self.joined.append(joined)
        return joined

    def files(self, vrt: VRT) -> frozenset[str]:
        """Return the paths of the files ``vrt`` reads in the end, through nested VRTs."""
        stack = [vrt]
        while stack:
            node = stack[-1]
            if id(node) in self.leaves:
                stack.pop()
                continue
            refs = [*node.sources]
            pending = [
                ref for ref in refs if not isinstance(ref, UPath) and id(ref) not in self.leaves
            ]
            if pending:
                stack += pending
                continue
            stack.pop()
            self.leaves[id(node)] = frozenset().union(
                *({str(ref)} if isinstance(ref, UPath) else self.leaves[id(ref)] for ref in refs)
            )
        return self.leaves[id(vrt)]

    def unique(self, refs: list[SourceRef]) -> list[SourceRef]:
        """Drop sources equal to an earlier one, such as the copies a read-back VRT holds."""
        found: dict[object, SourceRef] = {}
        for ref in refs:
            found.setdefault(self.interner.ref(ref, 0), ref)
        return list(found.values())

    def visit(self, vrt: VRT, depth: int) -> Input:
        """Add the steps that build ``vrt`` and return the input that refers to it."""
        vrt = self.rejoined(vrt)
        refs = self.unique(vrt.sources)
        verb, detail = _verb(vrt, refs)
        # A plain view of one nested VRT adds nothing to show.
        for _ in range(_MAX_VIEWS):
            if verb != "from_raster" or isinstance(refs[0], UPath):
                break
            vrt = self.rejoined(refs[0])
            refs = self.unique(vrt.sources)
            verb, detail = _verb(vrt, refs)
        key = self.interner.number(vrt)
        if key in self.seen:
            return self.seen[key]
        if depth >= MAX_DEPTH or len(self.steps) >= MAX_NODES:
            return Input("more", f"{short(vrt)} …")
        inputs: list[Input] = []
        files = nested = hidden = 0
        for item in refs:
            if isinstance(item, UPath) and files < MAX_FILES:
                file = _file(item)
                inputs.append(Input("file", file.name, file.path))
                files += 1
            elif not isinstance(item, UPath) and nested < MAX_CHILDREN:
                inputs.append(self.visit(item, depth + 1))
                nested += 1
            else:
                hidden += 1
        if hidden:
            inputs.append(Input("more", f"… +{hidden}"))
        number = len(self.steps) + 1
        self.steps.append(Step(number, verb, detail, tuple(inputs), _grid_text(vrt)))
        ref = Input("step", f"#{number}", verb, number)
        self.seen[key] = ref
        return ref


def to_text(summary: Summary) -> str:
    """Render a summary as text."""
    lines = ["VRT  " + " · ".join(summary.header), *_band_lines(summary)]
    lines += _step_lines(summary.steps)
    return "\n".join(lines)


def _band_lines(summary: Summary) -> list[str]:
    """Render the bands as aligned columns, dropping empty ones."""
    if not summary.bands:
        return ["bands  none"]
    columns = [
        [str(row.index) for row in summary.bands],
        [
            json.dumps(row.description, ensure_ascii=False) if row.description else ""
            for row in summary.bands
        ],
        [row.dtype for row in summary.bands],
        [f"nodata={row.nodata}" if row.nodata is not None else "" for row in summary.bands],
        [f"unit={row.unit}" if row.unit else "" for row in summary.bands],
    ]
    columns = [column for column in columns if any(column)]
    widths = [max(len(cell) for cell in column) for column in columns]
    lines: list[str] = []
    for number, cells in enumerate(zip(*columns, strict=True)):
        padded = [cells[0].rjust(widths[0])]
        padded += [cell.ljust(width) for cell, width in zip(cells[1:], widths[1:], strict=True)]
        lines.append(("bands  " if number == 0 else " " * 7) + "  ".join(padded).rstrip())
    if summary.more_bands:
        lines.append(" " * 7 + f"… +{summary.more_bands} bands")
    return lines


def brackets(steps: tuple[Step, ...]) -> list[str]:
    """Mark each run of steps that read the step above: ┌ starts it, │ continues, └ ends."""
    marks: list[str] = []
    for index, step in enumerate(steps):
        follows = index + 1 < len(steps) and steps[index + 1].joined
        if step.joined:
            marks.append("│" if follows else "└")
        else:
            marks.append("┌" if follows else "")
    return marks


def _step_text(step: Step) -> str:
    """Render a step's inputs and detail."""
    inputs = ", ".join(item.text for item in step.inputs)
    return " · ".join(part for part in (inputs, step.detail) if part)


def _step_lines(steps: tuple[Step, ...]) -> list[str]:
    """Render the steps as numbered, aligned lines, bracketing each chain."""
    if not steps:
        return ["steps  none"]
    marks = brackets(steps)
    bracket = any(marks)
    number_width = len(str(len(steps)))
    verb_width = max(len(step.verb) for step in steps)
    lines: list[str] = []
    for index, (step, mark) in enumerate(zip(steps, marks, strict=True)):
        line = f"{mark or ' '} " if bracket else ""
        line += f"{str(step.number).rjust(number_width)}  {step.verb.ljust(verb_width)}"
        line += f"  {_step_text(step)}"
        lines.append(("steps  " if index == 0 else " " * 7) + line.rstrip())
    return lines


# Jupyter's and Bootstrap's table and list rules are more specific, hence !important.
# Text keeps the page's color; the accent marks graphics only. Its default reads on light and
# dark pages alike, and the dark themes of JupyterLab, VS Code, Quarto and pydata-sphinx get a
# lighter one.
_CSS = """
.fzb-vrt-wrap {
  display: block !important; color: inherit; line-height: 1.45; --fzb-vrt-accent: #2f7fb0;
}
body[data-jp-theme-light="false"] .fzb-vrt-wrap, body.vscode-dark .fzb-vrt-wrap,
body.quarto-dark .fzb-vrt-wrap, html[data-theme="dark"] .fzb-vrt-wrap,
html[theme="dark"] .fzb-vrt-wrap, body[data-theme="dark"] .fzb-vrt-wrap {
  --fzb-vrt-accent: #6cbad6;
}
.fzb-vrt-fallback { display: none; }
.fzb-vrt-title { font-weight: 600; margin-bottom: .35em; }
.fzb-vrt-muted { opacity: .65; }
.fzb-vrt-grid {
  display: flex; flex-wrap: wrap; align-items: stretch; gap: .8em 2em; margin-bottom: .6em;
}
.fzb-vrt-info { display: flex; flex-direction: column; gap: .6em; min-width: 0; }
.fzb-vrt-wrap dl {
  display: grid; grid-template-columns: max-content minmax(0, auto); gap: .1em 1.2em;
  margin: 0;
}
.fzb-vrt-wrap dt { opacity: .65; font-weight: normal; }
.fzb-vrt-wrap dd { margin: 0; overflow-wrap: anywhere; }
.fzb-vrt-wrap dd small { display: block; opacity: .65; font-size: .92em; }
.fzb-vrt-ax { display: inline-block; width: 2.2em; opacity: .65; }
.fzb-vrt-world, .fzb-vrt-zoom {
  display: block; box-sizing: border-box; width: 100%; max-width: 100%;
  border: 1px solid color-mix(in srgb, currentColor 18%, transparent); border-radius: 4px;
}
.fzb-vrt-world { height: auto; aspect-ratio: 2 / 1; }
.fzb-vrt-map {
  display: flex; flex-direction: column; flex: 1 1 16em; max-width: 24em; min-width: 0;
}
/* Stretched beside the table; at its own shape when wrapped below it. */
.fzb-vrt-zoom { flex: 1 1 auto; aspect-ratio: 1.26; min-height: 0; }
.fzb-vrt-land { fill: currentColor; fill-opacity: .22; }
.fzb-vrt-cross {
  stroke: var(--fzb-vrt-accent); stroke-width: 1px; stroke-opacity: .7; fill: none;
}
.fzb-vrt-spot { fill: var(--fzb-vrt-accent); }
.fzb-vrt-graticule {
  stroke: currentColor; stroke-opacity: .15; fill: none; vector-effect: non-scaling-stroke;
}
.fzb-vrt-src { opacity: .55; }
.fzb-vrt-src polygon {
  fill: none; stroke: currentColor; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke;
}
.fzb-vrt-out {
  fill: color-mix(in srgb, var(--fzb-vrt-accent) 16%, transparent);
  stroke: var(--fzb-vrt-accent); stroke-width: 2px; vector-effect: non-scaling-stroke;
}
.fzb-vrt-labels { fill: currentColor; fill-opacity: .65; }
.fzb-vrt-legend {
  display: flex; flex-wrap: wrap; gap: .2em 1em; font-size: .85em; opacity: .65;
  margin-top: .3em;
}
.fzb-vrt-legend i {
  display: inline-block; width: 1.1em; vertical-align: middle; margin-right: .35em;
  border-top: 2px solid var(--fzb-vrt-accent);
}
.fzb-vrt-legend i.fzb-vrt-dash { border-top: 1.5px dashed currentColor; }
.fzb-vrt-wrap details {
  border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent);
  padding: .15em 0;
}
.fzb-vrt-wrap summary { cursor: pointer; font-weight: 600; }
.fzb-vrt-wrap summary span { font-weight: normal; opacity: .65; }
.fzb-vrt-wrap table {
  border-collapse: collapse; border: none; margin: .25em 0 .35em 1.2em;
  font-size: inherit; color: inherit; table-layout: auto; width: auto;
}
.fzb-vrt-wrap tr, .fzb-vrt-wrap th, .fzb-vrt-wrap td {
  background: transparent !important; box-shadow: none !important; color: inherit;
  border: none; padding: .05em 1.2em .05em 0; text-align: left !important;
  vertical-align: top;
}
.fzb-vrt-wrap th { font-weight: normal; opacity: .65; }
.fzb-vrt-wrap thead {
  border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent);
}
.fzb-vrt-wrap ol {
  list-style: none !important; margin: .35em 0 .2em 1.2em !important; padding: 0 !important;
}
.fzb-vrt-step {
  list-style: none !important; margin: 0 !important; position: relative;
  display: grid; grid-template-columns: 1.6em minmax(0, 1fr); column-gap: .7em;
  padding-bottom: .5em;
}
.fzb-vrt-link::before {
  content: ""; position: absolute; left: calc(.8em - 1px); top: 1.6em; bottom: 0;
  border-left: 2px solid color-mix(in srgb, var(--fzb-vrt-accent) 45%, transparent);
}
.fzb-vrt-num {
  /* 1.6em of the step's font, so the rail runs through the center. */
  width: calc(1.6em / .85); height: calc(1.6em / .85); font-size: .85em; font-weight: 600;
  border-radius: 50%; display: grid; place-items: center; line-height: 1;
  /* Digits stand on the baseline, above the middle of the line box. */
  box-sizing: border-box; padding-top: .2em; font-variant-numeric: lining-nums tabular-nums;
  background: color-mix(in srgb, var(--fzb-vrt-accent) 22%, transparent);
}
.fzb-vrt-last .fzb-vrt-num { box-shadow: inset 0 0 0 1.5px var(--fzb-vrt-accent); }
.fzb-vrt-what {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: .15em .6em; min-width: 0;
}
.fzb-vrt-verb {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-weight: 600;
  min-width: 6.5em;
}
.fzb-vrt-chip {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .9em;
  padding: 0 .35em; border-radius: 3px;
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
}
.fzb-vrt-ref { border-color: var(--fzb-vrt-accent); }
.fzb-vrt-note { grid-column: 2; font-size: .9em; opacity: .65; }
""".strip()


def to_html(summary: Summary, text: str) -> str:
    """Render a summary as self-contained HTML, with ``text`` shown where styles are dropped."""
    esc = html.escape
    count = len(summary.bands) + summary.more_bands
    band_note = f"{count} · {summary.dtypes}" if summary.dtypes else str(count)
    sections = [
        _details("Bands", band_note, _bands_html(summary)),
        _details("Steps", str(len(summary.steps)), _steps_html(summary.steps)),
    ]
    return (
        f'<div class="fzb-vrt"><style>{_CSS}</style>'
        f'<pre class="fzb-vrt-fallback">{esc(text)}</pre>'
        # Hidden unless the style above applies, so the text shows where styles are dropped.
        f'<div class="fzb-vrt-wrap" style="display:none">'
        f'<div class="fzb-vrt-title">VRT</div>{_grid_html(summary)}'
        f"{''.join(sections)}</div></div>"
    )


def _lines(first: str, second: str, labels: tuple[str, str] = ("", "")) -> str:
    """Render two lines of a value, each after an optional muted axis label."""
    return "<br>".join(
        (f'<span class="fzb-vrt-ax">{label}</span>' if label else "") + html.escape(value)
        for label, value in zip(labels, (first, second), strict=True)
    )


def _grid_html(summary: Summary) -> str:
    """Render the grid table beside the maps."""
    esc = html.escape
    crs = esc(summary.crs)
    if summary.crs_name:
        crs += f"<small>{esc(summary.crs_name)}</small>"
    rows = [
        ("Size", esc(summary.size)),
        ("CRS", crs),
        ("Resolution", esc(summary.resolution)),
        ("Bounds", _lines(*summary.bounds, labels=("x", "y"))),
    ]
    fp = summary.footprint
    if fp is not None and not fp.plain_degrees:
        lons = [lon for lon, _ in fp.ring]
        lats = [lat for _, lat in fp.ring]
        lon_range = _map.degree_range(lons, "E", "W")
        lat_range = _map.degree_range(lats, "N", "S")
        rows.append(("Geographic", _lines(lon_range, lat_range, labels=("lon", "lat"))))
    table = "".join(f"<dt>{key}</dt><dd>{value}</dd>" for key, value in rows)
    if fp is None:
        return f'<div class="fzb-vrt-grid"><div class="fzb-vrt-info"><dl>{table}</dl></div></div>'
    legend = "<span><i></i>output</span>"
    if fp.sources:
        files = (
            "source files"
            if not fp.more_sources
            else (f"{len(fp.sources)} of {len(fp.sources) + fp.more_sources} source files")
        )
        legend += f'<span><i class="fzb-vrt-dash"></i>{files}</span>'
    return (
        f'<div class="fzb-vrt-grid"><div class="fzb-vrt-info"><dl>{table}</dl>'
        f"{_map.world_svg(fp)}</div>"
        f'<div class="fzb-vrt-map">{_map.zoom_svg(fp)}'
        f'<div class="fzb-vrt-legend">{legend}</div></div></div>'
    )


def _details(title: str, note: str, body: str) -> str:
    """Wrap ``body`` in a collapsed section."""
    esc = html.escape
    return f"<details><summary>{esc(title)} <span>({esc(note)})</span></summary>{body}</details>"


def _bands_html(summary: Summary) -> str:
    """Render the bands as a table."""
    if not summary.bands:
        return '<p class="fzb-vrt-muted">none</p>'
    esc = html.escape
    head = "".join(
        f'<th scope="col">{name}</th>' for name in ("#", "Description", "Dtype", "Nodata", "Unit")
    )
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(cell or '')}</td>"
            for cell in (str(row.index), row.description, row.dtype, row.nodata, row.unit)
        )
        + "</tr>"
        for row in summary.bands
    )
    if summary.more_bands:
        rows += (
            f'<tr><td colspan="5" class="fzb-vrt-muted">… +{summary.more_bands} bands</td></tr>'
        )
    # Quarto would otherwise restyle the table as a Bootstrap one.
    return (
        '<table data-quarto-disable-processing="true">'
        f"<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    )


def _input_html(item: Input) -> str:
    """Render one input of a step as a chip."""
    esc = html.escape
    title = f' title="{esc(item.title)}"' if item.title else ""
    if item.kind == "step":
        return f'<span class="fzb-vrt-chip fzb-vrt-ref"{title}>{esc(item.text)}</span>'
    if item.kind == "more":
        return f'<span class="fzb-vrt-muted">{esc(item.text)}</span>'
    return f'<span class="fzb-vrt-chip"{title}>{esc(item.text)}</span>'


def _steps_html(steps: tuple[Step, ...]) -> str:
    """Render the steps as a numbered rail; a line joins a step to the next it feeds alone."""
    if not steps:
        return '<p class="fzb-vrt-muted">none</p>'
    esc = html.escape
    items: list[str] = []
    for index, step in enumerate(steps):
        classes = ["fzb-vrt-step"]
        if index + 1 < len(steps) and steps[index + 1].joined:
            classes.append("fzb-vrt-link")
        last = index == len(steps) - 1
        if last:
            classes.append("fzb-vrt-last")
        parts = [f'<span class="fzb-vrt-verb">{esc(step.verb)}</span>']
        parts += [_input_html(item) for item in step.inputs]
        if step.detail:
            parts.append(f"<span>{esc(step.detail)}</span>")
        if last:
            parts.append('<span class="fzb-vrt-muted">this VRT</span>')
        note = ""
        if index == 0 or steps[index - 1].grid != step.grid:
            note = f'<div class="fzb-vrt-note">{esc(step.grid)}</div>'
        items.append(
            f'<li class="{" ".join(classes)}"><span class="fzb-vrt-num">{step.number}</span>'
            f'<div class="fzb-vrt-what">{"".join(parts)}</div>{note}</li>'
        )
    return f'<ol class="fzb-vrt-steps">{"".join(items)}</ol>'
