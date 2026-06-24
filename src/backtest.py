"""
src/backtest.py
---------------
Model-agnostic walk-forward backtesting engine for univariate time series.

Design principle
----------------
This module knows nothing about specific models.  ``model_fn`` and
``forecast_fn`` are passed as plain callables, so ARIMA, Prophet, XGBoost,
or any future model can plug in without touching this file.

    model_fn   :: (train: pd.Series) -> fitted_model
    forecast_fn :: (fitted_model, steps: int) -> pd.Series
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: calculate_metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(actuals: pd.Series, forecasts: pd.Series) -> dict:
    """
    Calculate MAPE, RMSE, and MASE between *actuals* and *forecasts*.

    Why three metrics?
    ------------------
    MAPE (Mean Absolute Percentage Error)
        Intuitive percentage-based error.  Breaks when actuals are close
        to zero (division by zero or inflated percentages), so it should
        never be used alone for series with near-zero values.

    RMSE (Root Mean Squared Error)
        Penalises large errors more heavily than MAPE.  Expressed in the
        same units as the series, which makes it easy to interpret.

    MASE (Mean Absolute Scaled Error)
        Scales MAE by the in-sample naive lag-1 forecast MAE (i.e. the
        "just use yesterday's value" baseline).  MASE < 1 means the model
        beats naïve; MASE > 1 means it's worse than naïve.  This is the
        most robust of the three because it does not break on near-zero
        actuals and is scale-independent.

    Parameters
    ----------
    actuals : pd.Series
        Observed values.
    forecasts : pd.Series
        Model-predicted values.  Must be the same length as *actuals*.

    Returns
    -------
    dict with keys ``mape``, ``rmse``, ``mase`` — all ``float``, rounded
    to 4 decimal places.

    Raises
    ------
    ValueError
        If the series lengths do not match.
    """
    if len(actuals) != len(forecasts):
        raise ValueError(
            f"actuals and forecasts must be the same length; "
            f"got {len(actuals)} vs {len(forecasts)}."
        )

    actuals_arr   = np.asarray(actuals,   dtype=float)
    forecasts_arr = np.asarray(forecasts, dtype=float)
    errors        = actuals_arr - forecasts_arr
    abs_errors    = np.abs(errors)

    # ── MAPE ──────────────────────────────────────────────────────────────
    # Guard against division by zero: skip observations where actual == 0.
    nonzero_mask = actuals_arr != 0.0
    if nonzero_mask.any():
        mape = float(np.mean(abs_errors[nonzero_mask] /
                             np.abs(actuals_arr[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    # ── RMSE ──────────────────────────────────────────────────────────────
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # ── MASE ──────────────────────────────────────────────────────────────
    # Denominator: MAE of the naïve lag-1 forecast on the actuals themselves.
    # Using the actuals series as a proxy for the training series is standard
    # when the full training set is unavailable at aggregation time.
    naive_errors = np.abs(np.diff(actuals_arr))          # |y_t - y_{t-1}|
    naive_mae    = float(np.mean(naive_errors)) if len(naive_errors) > 0 else float("nan")

    if naive_mae != 0.0 and not np.isnan(naive_mae):
        mase = float(np.mean(abs_errors) / naive_mae)
    else:
        mase = float("nan")

    return {
        "mape": round(mape, 4),
        "rmse": round(rmse, 4),
        "mase": round(mase, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: walk_forward_backtest
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_backtest(
    series: pd.Series,
    model_fn: Callable,
    forecast_fn: Callable,
    initial_train_size: int,
    step_size: int = 1,
    forecast_horizon: int = 12,
) -> dict:
    """
    Roll a model forward through *series*, refitting at each origin.

    At each origin ``t``:
        1. Train on ``series[:t]``.
        2. Forecast ``forecast_horizon`` steps ahead.
        3. Record forecast vs. the corresponding actuals.
        4. Advance ``t`` by ``step_size``.

    The process stops when there are fewer than ``forecast_horizon``
    observations remaining after the current training window.

    Parameters
    ----------
    series : pd.Series
        The full historical series with a monotone ``DatetimeIndex``.
    model_fn : Callable
        ``(train: pd.Series) -> fitted_model``
        Any callable that accepts a ``pd.Series`` and returns a fitted
        model object (or any object that ``forecast_fn`` can consume).
    forecast_fn : Callable
        ``(fitted_model, steps: int) -> pd.Series``
        Any callable that accepts the object returned by ``model_fn`` and
        an integer step count, and returns a ``pd.Series`` of forecasts
        with a ``DatetimeIndex`` aligned to the future period.
    initial_train_size : int
        Number of observations to use for the first training window.
        Must satisfy ``initial_train_size + forecast_horizon <= len(series)``.
    step_size : int, optional
        Number of observations to advance the origin at each iteration.
        Default 1 (strict walk-forward).  Set to ``forecast_horizon`` for
        non-overlapping windows.
    forecast_horizon : int, optional
        Number of steps to forecast at each origin.  Default 12.

    Returns
    -------
    dict with keys:

        forecasts : list[pd.Series]
            One ``pd.Series`` per origin (successful fits only).
        actuals   : list[pd.Series]
            Corresponding actual observations, aligned to each forecast.
        metrics   : dict
            ``calculate_metrics()`` result aggregated over all origins by
            concatenating every forecast/actual pair first.
        n_origins : int
            Total number of rolling windows that were successfully evaluated.

    Notes
    -----
    Each ``model_fn`` call is wrapped in a ``try / except`` block.  If a
    single origin fails to fit, it is skipped with a ``WARNING`` log message
    and the backtest continues — the engine never crashes on a bad window.
    """
    n = len(series)
    if initial_train_size + forecast_horizon > n:
        raise ValueError(
            f"initial_train_size ({initial_train_size}) + forecast_horizon "
            f"({forecast_horizon}) exceeds series length ({n})."
        )

    all_forecasts: list[pd.Series] = []
    all_actuals:   list[pd.Series] = []
    n_origins = 0

    origins = range(initial_train_size, n - forecast_horizon + 1, step_size)

    for origin in origins:
        train  = series.iloc[:origin]
        actual = series.iloc[origin: origin + forecast_horizon]

        try:
            fitted_model = model_fn(train)
            forecast     = forecast_fn(fitted_model, forecast_horizon)

            # Align index: use actual's index so metrics comparisons are clean.
            forecast_aligned = pd.Series(
                np.asarray(forecast, dtype=float),
                index=actual.index,
                name="forecast",
            )

            all_forecasts.append(forecast_aligned)
            all_actuals.append(actual)
            n_origins += 1

        except Exception as exc:          # noqa: BLE001
            logger.warning(
                "Origin %d (train size %d) failed to fit: %s — skipping.",
                origin, len(train), exc,
            )

    if n_origins == 0:
        logger.warning("No origins succeeded.  Returning empty metrics.")
        return {
            "forecasts": [],
            "actuals":   [],
            "metrics":   {"mape": float("nan"), "rmse": float("nan"), "mase": float("nan")},
            "n_origins": 0,
        }

    # Aggregate metrics by concatenating all windows into a single pair.
    combined_actuals   = pd.concat(all_actuals)
    combined_forecasts = pd.concat(all_forecasts)
    aggregated_metrics = calculate_metrics(combined_actuals, combined_forecasts)

    return {
        "forecasts": all_forecasts,
        "actuals":   all_actuals,
        "metrics":   aggregated_metrics,
        "n_origins": n_origins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: plot_backtest_results
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest_results(
    series: pd.Series,
    backtest_result: dict,
    title: str = "Walk-Forward Backtest",
) -> plt.Figure:
    """
    Visualise the full historical series alongside all rolling forecast windows.

    Layout
    ------
    * The full *series* is drawn in blue (``#1f77b4``) as the baseline.
    * Each rolling forecast window is drawn in orange (``#ff7f0e``) at 35 %
      opacity.  Overlapping windows produce a darker band, making high-traffic
      regions of the backtest immediately visible.
    * A single orange entry appears in the legend (de-duplicated).

    Parameters
    ----------
    series : pd.Series
        The full historical series.  Should be the same series that was
        passed to :func:`walk_forward_backtest`.
    backtest_result : dict
        The dict returned by :func:`walk_forward_backtest`.
    title : str, optional
        Figure title.  Default ``"Walk-Forward Backtest"``.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object.  The caller is responsible for
        rendering or saving it — this function does **not** call
        ``plt.show()`` or ``fig.savefig()``.
    """
    metrics    = backtest_result.get("metrics", {})
    n_origins  = backtest_result.get("n_origins", 0)
    forecasts  = backtest_result.get("forecasts", [])

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Full historical series ────────────────────────────────────────────
    ax.plot(
        series.index,
        series.values,
        color="#1f77b4",
        linewidth=1.8,
        label="Actual series",
        zorder=3,
    )

    # ── Rolling forecast windows ──────────────────────────────────────────
    for i, fc in enumerate(forecasts):
        ax.plot(
            fc.index,
            fc.values,
            color="#ff7f0e",
            alpha=0.35,
            linewidth=1.2,
            # Only add one legend entry for all forecast windows.
            label="Rolling forecast" if i == 0 else "_nolegend_",
            zorder=2,
        )

    # ── Annotations ───────────────────────────────────────────────────────
    subtitle = (
        f"n_origins={n_origins}  |  "
        f"MAPE={metrics.get('mape', 'n/a')}%  |  "
        f"RMSE={metrics.get('rmse', 'n/a')}  |  "
        f"MASE={metrics.get('mase', 'n/a')}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(subtitle, fontsize=9, color="#555555")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig
