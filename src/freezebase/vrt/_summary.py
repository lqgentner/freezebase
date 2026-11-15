"""Summaries of a VRT for its text and HTML reprs, built from the model alone without I/O."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import re
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from rasterio.crs import CRS
from rasterio.errors import CRSError
from upath import UPath

if TYPE_CHECKING:
    from freezebase.vrt._model import VRT, Band

MAX_BANDS = 10
MAX_FILES = 5
MAX_CHILDREN = 5
MAX_DEPTH = 6
MAX_NODES = 40
MAX_CHARS = 60

_TIMES = "\N{MULTIPLICATION SIGN}"
_UNITS = {"metre": "m", "meter": "m", "degree": "°", "foot": "ft", "US survey foot": "US ft"}


@dataclass(frozen=True)
class File:
    """A file source: its short display name and its full path."""

    name: str
    path: str


@dataclass(frozen=True)
class Node:
    """One step of the pipeline, with the files it reads and its nested VRTs."""

    label: str
    files: tuple[File, ...] = ()
    more_files: int = 0
    children: tuple[Node, ...] = ()
    more_children: int = 0
    grid: str = ""


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
    grid: tuple[tuple[str, str], ...]
    bands: tuple[BandRow, ...]
    more_bands: int
    pipeline: Node


def crs_label(crs: CRS) -> str:
    """Name a CRS by its authority code, else by the name in its WKT."""
    if (authority := crs.to_authority()) is not None:
        return ":".join(authority)
    return _wkt_name(crs)


def short(vrt: VRT) -> str:
    """One-line description of a VRT, for reprs of objects that hold one."""
    crs = crs_label(vrt.crs) if vrt.crs is not None else "no CRS"
    return f"VRT({vrt.width}{_TIMES}{vrt.height}, {_clip(crs)}, {_plural(len(vrt.bands), 'band')})"


def summarize(vrt: VRT) -> Summary:
    """Summarize ``vrt`` from the model alone."""
    crs = _clip(crs_label(vrt.crs)) if vrt.crs is not None else "no CRS"
    size = f"{vrt.width} {_TIMES} {vrt.height} px"
    res = _resolution(vrt)
    dtypes = ", ".join(dict.fromkeys(vrt.dtypes))
    header = tuple(
        part for part in (size, _plural(len(vrt.bands), "band"), dtypes, crs, res) if part
    )
    grid = [("Size", size), ("CRS", crs)]
    if vrt.crs is not None and (name := _clip(_wkt_name(vrt.crs))) != crs:
        grid[-1] = ("CRS", f"{crs} ({name})")
    grid += [("Resolution", res), ("Bounds", ", ".join(_number(v) for v in vrt.bounds))]
    rows = tuple(_band_row(index, band) for index, band in enumerate(vrt.bands[:MAX_BANDS], 1))
    return Summary(
        header=header,
        grid=tuple(grid),
        bands=rows,
        more_bands=max(len(vrt.bands) - MAX_BANDS, 0),
        pipeline=_Tree().node(vrt, 0),
    )


def _plural(count: int, noun: str) -> str:
    """Format ``count noun``, adding an s unless the count is one."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _clip(text: str, limit: int = MAX_CHARS) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut, and blank control characters."""
    text = "".join(char if char.isprintable() else " " for char in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: float) -> str:
    """Format a number compactly."""
    return f"{value:.12g}"


def _wkt_name(crs: CRS) -> str:
    """Return the name in a CRS's WKT, else the WKT."""
    wkt = crs.to_wkt()
    match = re.search(r'"([^"]*)"', wkt)
    return match.group(1) if match else wkt


def _resolution(vrt: VRT) -> str:
    """Format the pixel size with the CRS's unit."""
    x, y = (f"{value:.6g}" for value in vrt.res)
    value = x if x == y else f"{x} {_TIMES} {y}"
    unit = ""
    if vrt.crs is not None:
        name = vrt.crs.units_factor[0]
        unit = _UNITS.get(name, name)
    if unit == "°":
        return value + unit
    return f"{value} {unit}" if unit else value


def _band_row(index: int, band: Band) -> BandRow:
    """Collect the display values of a band."""
    nodata = None if band.nodata is None else _number(band.nodata)
    unit = _clip(band.unit) if band.unit else None
    return BandRow(index, _clip(band.description), band.dtype, nodata, unit)


def _file(path: UPath) -> File:
    """Name a file by its base name, keeping the protocol of a remote one."""
    name = _clip(path.name)
    if path.protocol not in {"", "file"}:
        name = f"{path.protocol}://…/{name}"
    return File(name, str(path))


def _srs_label(text: str) -> str:
    """Label a CRS serialized in warp options."""
    try:
        return _clip(crs_label(CRS.from_user_input(text)))
    except CRSError:
        return _clip(text)


def _warp_label(vrt: VRT, options_xml: str) -> str:
    """Label a warp by the CRSs it reprojects between, from its serialized options."""
    target = _clip(crs_label(vrt.crs)) if vrt.crs is not None else "no CRS"
    try:
        # The options come from GDAL or from a VRT the caller chose to read.
        options = ET.fromstring(options_xml)  # noqa: S314
    except ET.ParseError:
        return "warped"
    source = options.findtext(".//SourceSRS")
    if not source:
        return f"resampled: {target}"
    if destination := options.findtext(".//TargetSRS"):
        target = _srs_label(destination)
    return f"reprojected: {_srs_label(source)} → {target}"


class _Tree:
    """Builds pipeline nodes, capping depth and the total node count."""

    def __init__(self) -> None:
        self.budget = MAX_NODES

    def node(self, vrt: VRT, depth: int) -> Node:
        """Describe ``vrt`` and its nested VRTs."""
        grid = short(vrt)
        self.budget -= 1
        if depth >= MAX_DEPTH or self.budget < 0:
            return Node(f"{grid} …", grid=grid)
        refs = vrt.sources
        files = [ref for ref in refs if isinstance(ref, UPath)]
        nested = [ref for ref in refs if not isinstance(ref, UPath)]
        label, arranges = _step(vrt, len(refs))
        if nested and not label:
            # A plain view of one nested VRT adds nothing to show.
            return self.node(nested[0], depth)
        if nested or not arranges:
            if arranges:
                label = f"{label} of {len(refs)} sources"
            return self.parent(label, files, nested, depth, grid)
        if not files:
            return Node("no sources", grid=grid)
        if label:
            label = f"{label} of {_plural(len(files), 'file')}"
        return _files_node(label or _files_label(files), files, grid)

    def parent(
        self, label: str, files: list[UPath], nested: list[VRT], depth: int, grid: str
    ) -> Node:
        """Return a node whose children are its nested VRTs, then its files."""
        children = [self.node(ref, depth + 1) for ref in nested[:MAX_CHILDREN]]
        if files:
            children.append(_files_node(_files_label(files), files))
        return Node(
            label,
            children=tuple(children),
            more_children=max(len(nested) - MAX_CHILDREN, 0),
            grid=grid,
        )


def _step(vrt: VRT, count: int) -> tuple[str, bool]:
    """Label what ``vrt`` does to its ``count`` sources, and whether it only arranges them."""
    if vrt.warp is not None:
        return _warp_label(vrt, vrt.warp.options_xml), False
    functions = dict.fromkeys(
        band.pixel_function.name for band in vrt.bands if band.pixel_function is not None
    )
    if functions:
        return f"derived: {', '.join(functions)}", False
    if any(len(band.sources) > 1 for band in vrt.bands):
        return "mosaic", True
    return ("stack" if count > 1 else ""), True


def _files_label(files: list[UPath]) -> str:
    """Label a list of files by their count, or nothing for a single file."""
    return "" if len(files) == 1 else f"{len(files)} files"


def _files_node(label: str, files: list[UPath], grid: str = "") -> Node:
    """Return a node listing the first few ``files``."""
    return Node(
        label,
        files=tuple(_file(path) for path in files[:MAX_FILES]),
        more_files=max(len(files) - MAX_FILES, 0),
        grid=grid,
    )


def to_text(summary: Summary) -> str:
    """Render a summary as text."""
    lines = ["VRT  " + " · ".join(summary.header), *_band_lines(summary)]
    lines += _tree_lines(summary.pipeline, "", last=True)
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


def _node_text(node: Node) -> str:
    """Render one node's label and files on one line."""
    names = [file.name for file in node.files]
    if node.more_files:
        names.append(f"… +{node.more_files}")
    if not names:
        return node.label
    return f"{node.label}: {', '.join(names)}" if node.label else ", ".join(names)


def _tree_lines(node: Node, prefix: str, *, last: bool) -> list[str]:
    """Render a node and its children as tree lines."""
    lines = [prefix + ("└─ " if last else "├─ ") + _node_text(node)]
    inner = prefix + ("   " if last else "│  ")
    for index, child in enumerate(node.children):
        is_last = index == len(node.children) - 1 and not node.more_children
        lines += _tree_lines(child, inner, last=is_last)
    if node.more_children:
        lines.append(f"{inner}└─ … +{node.more_children} more")
    return lines


_CSS = """
.fzb-vrt-wrap { display: block !important; color: inherit; line-height: 1.45; }
.fzb-vrt-fallback { display: none; }
.fzb-vrt-head { margin-bottom: .3em; }
.fzb-vrt-muted { opacity: .65; }
.fzb-vrt-wrap details {
  border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent);
  padding: .15em 0;
}
.fzb-vrt-wrap summary { cursor: pointer; font-weight: 600; }
.fzb-vrt-wrap dl {
  display: grid; grid-template-columns: max-content auto; gap: .1em 1.2em;
  margin: .25em 0 .35em 1.2em;
}
.fzb-vrt-wrap dt { opacity: .65; font-weight: normal; }
.fzb-vrt-wrap dd { margin: 0; }
.fzb-vrt-wrap table {
  border-collapse: collapse; border: none; margin: .25em 0 .35em 1.2em;
  font-size: inherit; color: inherit; table-layout: auto;
}
.fzb-vrt-wrap tr, .fzb-vrt-wrap th, .fzb-vrt-wrap td {
  background: transparent !important; color: inherit; border: none;
  padding: .05em 1.2em .05em 0; text-align: left !important; vertical-align: top;
}
.fzb-vrt-wrap th { font-weight: normal; opacity: .65; }
.fzb-vrt-wrap thead {
  border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent);
}
.fzb-vrt-wrap ul { list-style: none; margin: .25em 0 .35em 1.2em; padding: 0; }
.fzb-vrt-wrap ul ul {
  margin: .1em 0 .1em .35em; padding-left: .9em;
  border-left: 1px solid color-mix(in srgb, currentColor 25%, transparent);
}
.fzb-vrt-wrap li { margin: .1em 0; }
""".strip()


def to_html(summary: Summary, text: str) -> str:
    """Render a summary as self-contained HTML, with ``text`` shown where styles are dropped."""
    esc = html.escape
    header = esc(" · ".join(summary.header))
    grid = "".join(f"<dt>{esc(key)}</dt><dd>{esc(value)}</dd>" for key, value in summary.grid)
    sections = [
        _details("Grid", f"<dl>{grid}</dl>"),
        _details(f"Bands ({len(summary.bands) + summary.more_bands})", _bands_html(summary)),
        _details("Pipeline", f"<ul>{_node_html(summary.pipeline)}</ul>"),
    ]
    return (
        f'<div class="fzb-vrt"><style>{_CSS}</style>'
        f'<pre class="fzb-vrt-fallback">{esc(text)}</pre>'
        # Hidden unless the style above applies, so the text shows where styles are dropped.
        f'<div class="fzb-vrt-wrap" style="display:none">'
        f'<div class="fzb-vrt-head"><strong>VRT</strong> {header}</div>'
        f"{''.join(sections)}</div></div>"
    )


def _details(title: str, body: str) -> str:
    """Wrap ``body`` in an open, collapsible section."""
    return f"<details open><summary>{html.escape(title)}</summary>{body}</details>"


def _bands_html(summary: Summary) -> str:
    """Render the bands as a table."""
    if not summary.bands:
        return '<p class="fzb-vrt-muted">none</p>'
    esc = html.escape
    head = "".join(f"<th>{name}</th>" for name in ("#", "Description", "Dtype", "Nodata", "Unit"))
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
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _node_html(node: Node) -> str:
    """Render a node and its children as a list item."""
    esc = html.escape
    parts = [f'<span title="{esc(node.grid)}">{esc(node.label)}</span>'] if node.label else []
    files = [f'<span title="{esc(file.path)}">{esc(file.name)}</span>' for file in node.files]
    if node.more_files:
        files.append(f'<span class="fzb-vrt-muted">… +{node.more_files}</span>')
    if files:
        parts.append(", ".join(files))
    children = "".join(_node_html(child) for child in node.children)
    if node.more_children:
        children += f'<li class="fzb-vrt-muted">… +{node.more_children} more</li>'
    nested = f"<ul>{children}</ul>" if children else ""
    return f"<li>{': '.join(parts)}{nested}</li>"
