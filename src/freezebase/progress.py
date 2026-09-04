"""Shared rich progress-bar layout and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    ProgressType,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def create_progress(
    *,
    show_progress: bool = True,
    add_description: bool = True,
    columns: list[str | ProgressColumn] | None = None,
    **kwargs: Any,
) -> Progress:
    """Create a rich progress bar with a custom column layout.

    Parameters
    ----------
    show_progress : bool, default: True
        Whether to display the progress bar.
    add_description : bool
        Add a task-description column. Ignored with custom ``columns``.
    columns : list of str or ProgressColumn, optional
        Custom column layout.
    **kwargs : Any
        Additional ``Progress`` arguments.

    Returns
    -------
    Progress
        Configured progress manager.
    """
    if columns is None:
        columns = (
            [TextColumn("[progress.description]{task.description}")] if add_description else []
        )
        columns.extend(
            [
                BarColumn(),
                TaskProgressColumn(),
                "•",
                MofNCompleteColumn(),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
            ],
        )

    return Progress(*columns, disable=(not show_progress), **kwargs)


def track_progress(
    sequence: Iterable[ProgressType],
    description: str = "Working...",
    *,
    total: float | None = None,
    completed: int = 0,
    update_period: float = 0.1,
    show_progress: bool = True,
    **progress_kwargs: Any,
) -> Iterable[ProgressType]:
    """Yield items while tracking progress with the shared layout.

    Parameters
    ----------
    sequence : Iterable[ProgressType]
        Items to track.
    description : str, default: "Working..."
        Task label.
    total : float or None, default: None
        Total steps, inferred when possible.
    completed : int, default: 0
        Initially completed steps.
    update_period : float, default: 0.1
        Minimum update interval in seconds.
    show_progress : bool, default: True
        Whether to display the progress bar.
    **progress_kwargs : Any
        Additional ``Progress`` arguments.

    Yields
    ------
    ProgressType
        Items from `sequence`.
    """
    progress = create_progress(
        show_progress=show_progress,
        add_description=bool(description),
        **progress_kwargs,
    )

    with progress:
        yield from progress.track(
            sequence,
            total=total,
            completed=completed,
            description=description,
            update_period=update_period,
        )
