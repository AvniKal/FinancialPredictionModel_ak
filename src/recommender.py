"""
src/recommender.py
------------------
Automated forecasting parameter recommender.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from src.preprocessing import check_stationarity


def analyze_series(series: pd.Series) -> dict:
    """
    Compute key characteristics of a time series.
    Returns a dict with these keys:

    length: int                  # number of observations
    frequency: str               # e.g. 'MS', 'D'
    trend_strength: float        # 0-1, from STL decomposition
                                 # formula: 1 - Var(residual)/Var(trend+residual)
    seasonal_strength: float     # 0-1, same formula for seasonal component
    has_structural_break: bool   # True if ADF and KPSS contradict after differencing
                                 # (reuse check_stationarity from preprocessing.py)
    recommended_period: int      # 12 for monthly, 7 for daily, 4 for quarterly
    cv_coefficient: float        # coefficient of variation = std/mean
                                 # measures overall volatility
    """
    length = len(series)
    
    # 1. Determine frequency and recommended period
    freq = series.index.freqstr if hasattr(series.index, 'freqstr') else getattr(series.index, 'freq', None)
    if freq is not None and not isinstance(freq, str):
        freq = getattr(freq, 'freqstr', None) or str(freq)

    if freq is None:
        freq = pd.infer_freq(series.index)

    if freq:
        if 'M' in freq:
            recommended_period = 12
        elif 'Q' in freq:
            recommended_period = 4
        elif 'D' in freq or 'B' in freq:
            recommended_period = 7
        elif 'W' in freq:
            recommended_period = 52
        else:
            recommended_period = 12
    else:
        # fallback to time delta check
        deltas = pd.Series(series.index).diff().dropna()
        if not deltas.empty:
            median_days = deltas.dt.total_seconds().median() / (24 * 3600)
            if 27 <= median_days <= 32:
                recommended_period = 12
                freq = 'MS'
            elif 88 <= median_days <= 93:
                recommended_period = 4
                freq = 'Q'
            elif 0.8 <= median_days <= 1.2:
                recommended_period = 7
                freq = 'D'
            else:
                recommended_period = 12
                freq = 'MS'
        else:
            recommended_period = 12
            freq = 'MS'
            
    # Clean up frequency string display if it is None
    freq_str = str(freq) if freq is not None else "MS"

    # 2. Run STL Decomposition to compute trend and seasonal strengths
    # STL requires at least 2 * period observations.
    if length >= 2 * recommended_period:
        res = STL(series, period=recommended_period).fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        var_resid = float(np.var(resid, ddof=0))
        var_trend_resid = float(np.var(trend + resid, ddof=0))
        if var_trend_resid > 0:
            trend_strength = max(0.0, min(1.0, 1.0 - var_resid / var_trend_resid))
        else:
            trend_strength = 0.0

        var_seasonal_resid = float(np.var(seasonal + resid, ddof=0))
        if var_seasonal_resid > 0:
            seasonal_strength = max(0.0, min(1.0, 1.0 - var_resid / var_seasonal_resid))
        else:
            seasonal_strength = 0.0
    else:
        # Fallback if too short for STL
        trend_strength = 0.0
        seasonal_strength = 0.0

    # 3. Check for structural break (ADF/KPSS contradiction after first diff)
    if length > 2:
        series_diff = series.diff().dropna()
        try:
            stat_res = check_stationarity(series_diff)
            has_structural_break = stat_res["adf_stationary"] != stat_res["kpss_stationary"]
        except Exception:
            has_structural_break = False
    else:
        has_structural_break = False

    # 4. Coefficient of variation
    mean_val = float(series.mean())
    cv_coefficient = (float(series.std()) / mean_val) if mean_val != 0 else 0.0

    return {
        "length": length,
        "frequency": freq_str,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "has_structural_break": has_structural_break,
        "recommended_period": recommended_period,
        "cv_coefficient": cv_coefficient
    }


def recommend_config(series: pd.Series) -> dict:
    """
    Takes a series, calls analyze_series() internally, then returns
    actionable recommendations:

    {
        "forecast_horizon": int,
        "initial_train_size": int,
        "recommended_models": list[str],
        "arima_order": tuple,
        "reasoning": dict  # key->str explaining each recommendation
    }
    """
    stats = analyze_series(series)
    length = stats["length"]
    period = stats["recommended_period"]
    seasonal_strength = stats["seasonal_strength"]
    trend_strength = stats["trend_strength"]

    # 1. forecast_horizon
    if length < 60:
        horizon = 3
        horizon_reason = f"Series length is short ({length} observations), suggesting a short horizon of 3."
    elif length < 120:
        horizon = 6
        horizon_reason = f"Series length is medium ({length} observations), suggesting a moderate horizon of 6."
    else:
        horizon = 12
        horizon_reason = f"Series length is long ({length} observations), suggesting a standard horizon of 12."

    if seasonal_strength > 0.4:
        old_horizon = horizon
        horizon = int(np.ceil(horizon / period) * period)
        horizon_reason += f" Seasonal strength is strong ({seasonal_strength:.4f}), rounding up horizon to {horizon} to align with seasonal period {period}."

    # 2. initial_train_size
    min_size_1 = 2 * horizon * period
    min_size_2 = int(np.ceil(length * 0.6))
    initial_train_size = max(min_size_1, min_size_2)
    initial_train_reason = (
        f"Selected size {initial_train_size} as the max of 60% of series length ({min_size_2}) "
        f"and 2 seasonal cycles of horizon ({min_size_1})."
    )

    # 3. recommended_models
    models = ["naive"]
    model_reasons = ["'naive' included as a baseline."]

    if trend_strength > 0.5:
        models.append("prophet")
        model_reasons.append(f"prophet added due to strong trend (strength {trend_strength:.4f} > 0.5).")
    if seasonal_strength > 0.3:
        models.append("sarimax")
        model_reasons.append(f"sarimax added due to strong seasonality (strength {seasonal_strength:.4f} > 0.3).")
    if length > 100:
        models.append("arima")
        models.append("xgboost")
        model_reasons.append(f"arima and xgboost added since series length ({length} > 100) provides sufficient data.")

    recommended_models = sorted(list(set(models)))
    models_reason = " ".join(model_reasons)

    # 4. arima_order
    # d is always 1 (confirmed by stationarity analysis)
    # p: 1 if length < 100, else 2
    # q: 1 always (safe default)
    p = 1 if length < 100 else 2
    arima_order = (p, 1, 1)
    arima_reason = f"ARIMA order {arima_order} recommended: p={p} based on length ({length}), d=1 for stationarity differencing, q=1 as a conservative default."

    return {
        "forecast_horizon": horizon,
        "initial_train_size": initial_train_size,
        "recommended_models": recommended_models,
        "arima_order": arima_order,
        "reasoning": {
            "forecast_horizon": horizon_reason,
            "initial_train_size": initial_train_reason,
            "recommended_models": models_reason,
            "arima_order": arima_reason
        }
    }
