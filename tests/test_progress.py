"""Tests for freezebase.progress bar construction and iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import BarColumn, TextColumn

from freezebase.progress import create_progress, track_progress

if TYPE_CHECKING:
    from collections.abc import Iterator


class TestCreateProgress:
    def test_default_layout_starts_with_a_description_column(self) -> None:
        progress = create_progress()

        assert isinstance(progress.columns[0], TextColumn)
        assert any(isinstance(column, BarColumn) for column in progress.columns)

    def test_add_description_false_drops_the_description_column(self) -> None:
        with_description = create_progress()
        without_description = create_progress(add_description=False)

        assert isinstance(without_description.columns[0], BarColumn)
        assert len(without_description.columns) == len(with_description.columns) - 1

    def test_show_progress_false_disables_the_bar(self) -> None:
        assert create_progress(show_progress=False).disable is True
        assert create_progress(show_progress=True).disable is False

    def test_explicit_columns_are_used_verbatim(self) -> None:
        columns: list[str | TextColumn] = [TextColumn("{task.description}"), "•"]

        progress = create_progress(columns=columns, add_description=True)  # type: ignore[arg-type]

        # `add_description` is documented as ignored when `columns` is given.
        assert len(progress.columns) == len(columns)
        assert isinstance(progress.columns[0], TextColumn)


class TestTrackProgress:
    def test_yields_every_item_in_order(self) -> None:
        assert list(track_progress(range(4), show_progress=False)) == [0, 1, 2, 3]

    def test_works_for_a_sequence_without_a_length(self) -> None:
        # A bare generator has no len(); the total must come from the argument.
        items = (i * 2 for i in range(3))

        assert list(track_progress(items, total=3, show_progress=False)) == [0, 2, 4]

    def test_empty_sequence_completes(self) -> None:
        assert list(track_progress([], show_progress=False)) == []

    def test_lazy_iteration_does_not_consume_upfront(self) -> None:
        consumed: list[int] = []

        def source() -> Iterator[int]:
            for i in range(3):
                consumed.append(i)
                yield i

        tracked = track_progress(source(), total=3, show_progress=False)
        assert consumed == []  # nothing pulled before the first next()
        assert next(iter(tracked)) == 0
        assert consumed == [0]
