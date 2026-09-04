"""Pandas and GeoPandas helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

_VALID_CASE_TYPES = ("lower", "upper", "snake", "preserve")


@overload
def clean_names(df: gpd.GeoDataFrame, case_type: str = ...) -> gpd.GeoDataFrame: ...
@overload
def clean_names(df: pd.DataFrame, case_type: str = ...) -> pd.DataFrame: ...
def clean_names(
    df: pd.DataFrame | gpd.GeoDataFrame,
    case_type: str = "lower",
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Normalize DataFrame column names.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        DataFrame to rename.
    case_type : str, optional
        ``"lower"``, ``"upper"``, ``"snake"``, or ``"preserve"``.

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        Renamed copy.

    Raises
    ------
    ValueError
        If ``case_type`` is invalid or normalized names collide.
    """
    # Validate even when the DataFrame has no columns.
    if case_type not in _VALID_CASE_TYPES:
        msg = f"Unknown case_type: {case_type!r}. Valid options: {_VALID_CASE_TYPES}."
        raise ValueError(msg)

    def to_snake(name: str) -> str:
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
        return name.lower()

    # Column labels are not necessarily strings.
    def transform(name: object) -> str:
        text = re.sub(r"\s+", "_", str(name).strip())
        match case_type:
            case "lower":
                return text.lower()
            case "upper":
                return text.upper()
            case "snake":
                return to_snake(text)
            case _:  # "preserve"
                return text

    new_names = [transform(col) for col in df.columns]
    # pandas silently drops earlier duplicate names.
    seen: dict[str, object] = {}
    for original, new in zip(df.columns, new_names, strict=True):
        if new in seen:
            msg = (
                f"Cleaning columns with case_type={case_type!r} maps both "
                f"{seen[new]!r} and {original!r} to {new!r}."
            )
            raise ValueError(msg)
        seen[new] = original

    return df.rename(columns=dict(zip(df.columns, new_names, strict=True)))
