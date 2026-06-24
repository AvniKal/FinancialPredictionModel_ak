"""
src/data_loader.py
------------------
Loads time-series data from FRED (Federal Reserve Economic Data) as the primary
data source, with RSXFS (Advance Retail Sales: Clothing & Clothing Accessory
Stores) as the default series.

All functions are pure: they take plain arguments and return plain
pandas objects with no side-effects (no prints, no plots, no globals).
"""

from __future__ import annotations

import pandas as pd
import pandas_datareader.data as web
from src.preprocessing import clean_series


def load_fred_series(
    series_id: str,
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.Series:
    """
    Download a single FRED series as a ``pd.Series`` with a
    ``DatetimeIndex`` at the native FRED frequency.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. ``"RSXFS"`` (Advance Retail Sales:
        Clothing & Clothing Accessory Stores, monthly, in millions USD).
    start : str
        ISO-8601 start date (inclusive).  Default ``"2000-01-01"``.
    end : str or None
        ISO-8601 end date (inclusive).  ``None`` fetches through today.

    Returns
    -------
    pd.Series
        Raw series as returned by FRED, *before* any cleaning.
        Index is ``DatetimeIndex``; name is ``series_id``.

    Raises
    ------
    Exception
        Re-raises any network or parsing errors from ``pandas_datareader``
        so the caller can decide how to handle them.
    """
    df = web.DataReader(series_id, data_source="fred", start=start, end=end)
    series = df[series_id]
    series.name = series_id
    return series


# Alias to match exploration notebook import requirement
fetch_fred_series = load_fred_series

