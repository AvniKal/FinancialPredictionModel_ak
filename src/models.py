"""
src/models.py
-------------
Model wrappers compatible with :func:`src.backtest.walk_forward_backtest`.

Each ``get_*_fns()`` factory returns a ``(model_fn, forecast_fn)`` tuple
where:

    model_fn    :: (train: pd.Series)            -> fitted_model
    forecast_fn :: (fitted_model, steps: int)     -> pd.Series

The backtest engine re-stamps the forecast index to match the actuals
(backtest.py L203-207), so forecast_fn only needs to return the correct
*values* with length == steps.  We still attach a proper DatetimeIndex
for standalone use outside the backtest loop.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER 1: ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def get_arima_fns(
    order: tuple = (1, 1, 1),
) -> tuple[Callable, Callable]:
    """
    Return ``(model_fn, forecast_fn)`` for a statsmodels ARIMA model.

    Parameters
    ----------
    order : tuple of (p, d, q)
        ARIMA order.  Default ``(1, 1, 1)`` — first differencing with
        one AR and one MA term, which is the baseline confirmed by
        ADF/KPSS analysis (d=1 is correct; structural break at COVID
        does not warrant d=2).

    Returns
    -------
    tuple[Callable, Callable]
        ``model_fn(train)`` fits an ARIMA and returns the
        ``ARIMAResultsWrapper``.
        ``forecast_fn(fitted, steps)`` returns a ``pd.Series`` of length
        *steps* with a ``DatetimeIndex`` continuing from the training end.
    """

    def model_fn(train: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train, order=order)
            fitted = model.fit()
        return fitted

    def forecast_fn(fitted_model, steps: int) -> pd.Series:
        forecast = fitted_model.forecast(steps=steps)
        # forecast is already a pd.Series with DatetimeIndex from statsmodels
        forecast.name = "forecast"
        return forecast

    return model_fn, forecast_fn


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER 2: SARIMAX
# ─────────────────────────────────────────────────────────────────────────────

def get_sarimax_fns(
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 12),
) -> tuple[Callable, Callable]:
    """
    Return ``(model_fn, forecast_fn)`` for a statsmodels SARIMAX model.

    Parameters
    ----------
    order : tuple of (p, d, q)
        Non-seasonal ARIMA order.  Default ``(1, 1, 1)``.
    seasonal_order : tuple of (P, D, Q, s)
        Seasonal order.  Default ``(1, 1, 0, 12)`` — one seasonal AR
        term with seasonal differencing at period 12 (monthly).
        This reflects the weak but present monthly seasonality observed
        in RSXFS decomposition (multiplicative factors 0.996–1.003).

    Returns
    -------
    tuple[Callable, Callable]
        Same contract as :func:`get_arima_fns`.
    """

    def model_fn(train: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
        return fitted

    def forecast_fn(fitted_model, steps: int) -> pd.Series:
        forecast = fitted_model.forecast(steps=steps)
        forecast.name = "forecast"
        return forecast

    return model_fn, forecast_fn


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER 3: Seasonal Naïve Baseline
# ─────────────────────────────────────────────────────────────────────────────

def get_naive_fns() -> tuple[Callable, Callable]:
    """
    Return ``(model_fn, forecast_fn)`` for a seasonal naïve baseline.

    Forecast rule: each future step ``h`` copies the value from the same
    calendar month one year ago (i.e. ``train[-(12 - (h % 12))]`` when
    indexing cyclically).  Falls back to simple lag-1 (last observed value)
    if the training series is shorter than 12 periods.

    This is a *stronger* baseline than a flat mean — it respects the
    seasonal pattern.  Any real model should beat this.  If it doesn't,
    the model is adding complexity without value.

    Returns
    -------
    tuple[Callable, Callable]
        ``model_fn`` stores the last 12 (or fewer) observations.
        ``forecast_fn`` tiles them forward for *steps* periods.
    """

    def model_fn(train: pd.Series):
        # "Fitting" a naïve model just means remembering the last season.
        season_length = min(12, len(train))
        last_season = train.iloc[-season_length:].values.copy()
        return {
            "last_season": last_season,
            "season_length": season_length,
            "last_index": train.index[-1],
            "freq": getattr(train.index, "freq", None) or pd.tseries.frequencies.to_offset("MS"),
        }

    def forecast_fn(fitted_model, steps: int) -> pd.Series:
        last_season = fitted_model["last_season"]
        season_len = fitted_model["season_length"]

        # Tile the last season forward to cover `steps` periods.
        repeats = (steps // season_len) + 1
        tiled = np.tile(last_season, repeats)[:steps]

        # Build a DatetimeIndex continuing from the end of training.
        start = fitted_model["last_index"] + fitted_model["freq"]
        idx = pd.date_range(start=start, periods=steps, freq=fitted_model["freq"])

        return pd.Series(tiled, index=idx, name="forecast")

    return model_fn, forecast_fn


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER 4: Facebook Prophet
# ─────────────────────────────────────────────────────────────────────────────

def get_prophet_fns(
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.3,
) -> tuple[Callable, Callable]:
    """
    Return ``(model_fn, forecast_fn)`` for Facebook Prophet.

    Parameters
    ----------
    yearly_seasonality : bool
        Enable yearly seasonality detection.  Default ``True``.
    changepoint_prior_scale : float
        Flexibility of the automatic changepoint detection.  Default
        ``0.3`` — significantly higher than Prophet's built-in default
        of ``0.05`` because RSXFS contains a genuine structural break
        around the COVID shock (2020-03 to 2020-06) that must be
        captured, not smoothed over.

    Returns
    -------
    tuple[Callable, Callable]
        ``model_fn`` converts the ``pd.Series`` to Prophet's required
        ``DataFrame(columns=['ds', 'y'])`` format, fits, and returns
        the fitted Prophet object.
        ``forecast_fn`` calls ``make_future_dataframe`` and returns
        only the ``yhat`` column as a ``pd.Series`` with
        ``DatetimeIndex``.

    Notes
    -----
    Prophet prints fitting progress to stdout by default.  We suppress
    this by setting the Prophet logger to ERROR level and redirecting
    ``cmdstanpy`` output.
    """

    def model_fn(train: pd.Series):
        # Lazy import — prophet is a heavy optional dependency.
        import logging as _logging
        from prophet import Prophet

        # Suppress Prophet's verbose stdout/stderr output.
        _logging.getLogger("prophet").setLevel(_logging.ERROR)
        _logging.getLogger("cmdstanpy").setLevel(_logging.ERROR)

        # Prophet expects a DataFrame with columns 'ds' and 'y'.
        df = pd.DataFrame({"ds": train.index, "y": train.values})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
            )
            m.fit(df)

        return {"prophet_model": m, "freq": "MS"}

    def forecast_fn(fitted_model, steps: int) -> pd.Series:
        m = fitted_model["prophet_model"]
        freq = fitted_model["freq"]

        future = m.make_future_dataframe(periods=steps, freq=freq)
        prediction = m.predict(future)

        # Extract only the forecast horizon (last `steps` rows).
        forecast_df = prediction.iloc[-steps:]
        idx = pd.DatetimeIndex(forecast_df["ds"])

        return pd.Series(
            forecast_df["yhat"].values, index=idx, name="forecast"
        )

    return model_fn, forecast_fn


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER 5: XGBoost with Lag Features
# ─────────────────────────────────────────────────────────────────────────────

def _build_lag_features(
    series: pd.Series, n_lags: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a univariate time series into a supervised-learning dataset.

    Features
    --------
    lag_1, lag_2, …, lag_{n_lags}
        Shifted values of the series.
    month
        Month of year (1–12) — captures calendar seasonality.
    year
        Calendar year — captures the long-term level/trend.

    Returns
    -------
    X : pd.DataFrame  (features)
    y : pd.Series      (target = next value)

    The first ``n_lags`` rows are dropped because their lag features
    contain NaN.
    """
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)

    df["month"] = series.index.month
    df["year"] = series.index.year

    df = df.dropna()
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y


def get_xgboost_fns(
    n_lags: int = 12,
    n_estimators: int = 100,
) -> tuple[Callable, Callable]:
    """
    Return ``(model_fn, forecast_fn)`` for XGBoost with lag features.

    Parameters
    ----------
    n_lags : int
        Number of lag features to create.  Default ``12`` (one full year
        of monthly history as features).
    n_estimators : int
        Number of boosting rounds.  Default ``100``.

    Returns
    -------
    tuple[Callable, Callable]
        ``model_fn`` builds a lag-feature matrix from the training
        series and fits an ``XGBRegressor``.  It returns the fitted
        model together with the last ``n_lags`` observations needed
        to seed recursive forecasting.
        ``forecast_fn`` forecasts *recursively* (one step at a time,
        appending each prediction to the history before predicting the
        next step).

    Notes
    -----
    **Why recursive (autoregressive) multi-step forecasting?**

    XGBoost is not a native time-series model — it has no concept of
    temporal ordering.  To forecast ``h`` steps ahead, we *cannot*
    simply build features from the original series and predict all ``h``
    steps at once, because steps 2..h would need lag values that have
    not been observed yet.

    Instead, we predict step 1, append that prediction to the history,
    rebuild features, predict step 2, and so on.  This is correct but
    has a known drawback: **errors accumulate** over the horizon because
    each prediction feeds into the features for the next.  Expect
    accuracy to degrade for longer horizons (h >> 1).
    """

    def model_fn(train: pd.Series):
        # Lazy import — xgboost is an optional dependency.
        from xgboost import XGBRegressor

        X_train, y_train = _build_lag_features(train, n_lags)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                verbosity=0,         # suppress XGBoost's own logging
            )
            reg.fit(X_train, y_train)

        return {
            "regressor": reg,
            "tail": train.iloc[-n_lags:].copy(),  # seed for recursive forecast
            "n_lags": n_lags,
            "freq": getattr(train.index, "freq", None)
                    or pd.tseries.frequencies.to_offset("MS"),
        }

    def forecast_fn(fitted_model, steps: int) -> pd.Series:
        reg = fitted_model["regressor"]
        tail = fitted_model["tail"].copy()
        n = fitted_model["n_lags"]
        freq = fitted_model["freq"]

        # Recursive multi-step forecasting:
        # Predict one step, append to history, repeat.
        # NOTE: Error accumulates over the horizon because each
        # predicted value feeds back as a lag feature for subsequent
        # steps.  This is inherent to autoregressive ML forecasting.
        predictions: list[float] = []
        history = list(tail.values)
        last_date = tail.index[-1]

        for h in range(steps):
            # Build feature vector for the next step.
            lags = history[-n:][::-1]            # [lag_1, lag_2, ..., lag_n]
            next_date = last_date + (h + 1) * freq
            features = lags + [next_date.month, next_date.year]

            feature_names = [f"lag_{i}" for i in range(1, n + 1)] + ["month", "year"]
            X_step = pd.DataFrame([features], columns=feature_names)

            pred = float(reg.predict(X_step)[0])
            predictions.append(pred)
            history.append(pred)

        start = last_date + freq
        idx = pd.date_range(start=start, periods=steps, freq=freq)
        return pd.Series(predictions, index=idx, name="forecast")

    return model_fn, forecast_fn


def run_model_race(
    series: pd.Series,
    initial_train_size: int = 120,
    step_size: int = 3,
    forecast_horizon: int = 12,
    models_to_run: list[str] = None
) -> list[dict]:
    """
    Run all model wrappers through walk_forward_backtest and return
    a leaderboard sorted by MASE ascending (lower is better).

    Models to include: naive, arima, sarimax, prophet, xgboost
    models_to_run allows caller to run a subset e.g. ['arima', 'prophet']

    Each entry in the returned list is a dict:
    {
        model_name: str,
        mape: float,
        rmse: float,
        mase: float,
        n_origins: int,
        fit_time_seconds: float,  # total wall time for that model's backtest
        status: str               # 'ok' or 'failed' if model crashed entirely
    }

    Requirements:
    - Catch exceptions per model — one model failing must not stop the race
    - If a model fails entirely, include it in leaderboard with status='failed'
      and metric values as None
    - Print a progress line as each model finishes:
      "[1/5] naive     — MASE: 1.2341  (2.3s)"
    - Return the full leaderboard sorted by MASE ascending,
      with failed models at the bottom regardless of MASE
    """
    import time
    import math
    from src.backtest import walk_forward_backtest

    all_models = {
        "naive": lambda: get_naive_fns(),
        "arima": lambda: get_arima_fns(order=(1, 1, 1)),
        "sarimax": lambda: get_sarimax_fns(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)),
        "prophet": lambda: get_prophet_fns(yearly_seasonality=True, changepoint_prior_scale=0.3),
        "xgboost": lambda: get_xgboost_fns(n_lags=12, n_estimators=100)
    }

    if models_to_run is None:
        run_list = list(all_models.keys())
    else:
        run_list = [m.lower() for m in models_to_run if m.lower() in all_models]

    leaderboard = []
    total_models = len(run_list)

    for idx, name in enumerate(run_list, 1):
        status = "ok"
        mape, rmse, mase, n_origins = None, None, None, 0
        start_time = time.time()
        
        try:
            model_fn, forecast_fn = all_models[name]()
            res = walk_forward_backtest(
                series=series,
                model_fn=model_fn,
                forecast_fn=forecast_fn,
                initial_train_size=initial_train_size,
                step_size=step_size,
                forecast_horizon=forecast_horizon,
            )
            
            n_origins = res.get("n_origins", 0)
            metrics = res.get("metrics", {})
            mape = metrics.get("mape")
            rmse = metrics.get("rmse")
            mase = metrics.get("mase")
            
            if n_origins == 0:
                status = "failed"
                mape, rmse, mase = None, None, None
                
        except Exception:
            status = "failed"
            mape, rmse, mase, n_origins = None, None, None, 0
            
        elapsed = time.time() - start_time
        display_name = f"{name} (failed)" if status == "failed" else name
        
        leaderboard.append({
            "model_name": display_name,
            "mape": mape,
            "rmse": rmse,
            "mase": mase,
            "n_origins": n_origins,
            "fit_time_seconds": round(elapsed, 2),
            "status": status
        })
        
        if status == "failed":
            print(f"[{idx}/{total_models}] {name:<9} — FAILED  ({elapsed:.1f}s)")
        else:
            mase_str = f"{mase:.4f}" if mase is not None else "None"
            print(f"[{idx}/{total_models}] {name:<9} — MASE: {mase_str}  ({elapsed:.1f}s)")

    def sort_key(item):
        is_fail = item["status"] == "failed"
        mase_val = item["mase"]
        if is_fail or mase_val is None or (isinstance(mase_val, float) and math.isnan(mase_val)):
            return (1, float('inf'))
        return (0, mase_val)

    leaderboard.sort(key=sort_key)
    return leaderboard

