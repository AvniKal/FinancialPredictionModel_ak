"""
src/preprocessing.py
--------------------
Stateless preprocessing utilities for financial time series.
All functions are pure (no side-effects, no prints, no plots) and return
plain Python objects so they can be unit-tested without a notebook runtime.
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Run the ADF and KPSS stationarity tests on *series* and return the
    raw statistics together with a plain-English interpretation.

    What each test measures
    -----------------------
    ADF (Augmented Dickey-Fuller)
        Null hypothesis: the series has a unit root (i.e. is *non-stationary*).
        A p-value <= alpha means we *reject* the null and conclude the series
        is stationary.  A large ADF statistic (more negative) is stronger
        evidence against the unit root.

    KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
        Null hypothesis: the series *is* stationary (trend or level stationary).
        A p-value <= alpha means we *reject* the null and conclude the series
        is non-stationary.  KPSS complements ADF: if ADF says stationary but
        KPSS also says stationary (p > alpha), you have strong evidence of
        stationarity; contradictory results suggest a more nuanced picture
        (e.g. fractional integration or structural breaks).

    Parameters
    ----------
    series : pd.Series
        The univariate time series to test.  Must contain no NaN values.
    alpha : float, optional
        Significance level used to interpret the p-values (default 0.05).

    Returns
    -------
    dict with keys:
        adf_statistic   – float   : ADF test statistic
        adf_pvalue      – float   : p-value for the ADF test
        adf_stationary  – bool    : True if series appears stationary by ADF
        adf_critical_values – dict: critical values at 1 %, 5 %, 10 % levels

        kpss_statistic  – float   : KPSS test statistic
        kpss_pvalue     – float   : p-value for the KPSS test
        kpss_stationary – bool    : True if series appears stationary by KPSS
        kpss_critical_values – dict: critical values at 10 %, 5 %, 2.5 %, 1 %

        is_stationary   – bool    : True only when *both* tests agree the
                                    series is stationary
        alpha           – float   : significance level used
    """
    # ------------------------------------------------------------------ ADF
    # Existing notebook logic preserved exactly; print calls removed and
    # values routed into the return dict instead.
    adf_result = adfuller(series)
    adf_stat   = adf_result[0]          # ADF test statistic  (result[0])
    adf_pvalue = adf_result[1]          # p-value             (result[1])
    adf_cv     = adf_result[4]          # critical values dict (result[4])

    # Original notebook decision rule: p-value <= 0.05  =>  stationary
    adf_stationary = bool(adf_pvalue <= alpha)

    # ------------------------------------------------------------------ KPSS
    # KPSS null is *stationarity*, so the decision is flipped relative to ADF.
    # InterpolationWarning is suppressed: it fires only when the test statistic
    # falls outside statsmodels' p-value look-up table (common for short series)
    # and would otherwise violate the no-side-effects contract of this function.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(series, regression="c", nlags="auto")
    kpss_stat  = kpss_result[0]     # KPSS test statistic
    kpss_pvalue = kpss_result[1]    # p-value
    kpss_cv    = kpss_result[3]     # critical values dict

    # p-value > alpha  =>  fail to reject null  =>  series IS stationary
    kpss_stationary = bool(kpss_pvalue > alpha)

    # --------------------------------------------------------- Combined verdict
    both_agree = adf_stationary and kpss_stationary

    if both_agree:
        conclusion = "Stationary: both ADF and KPSS agree the series has no unit root."
    elif adf_stationary and not kpss_stationary:
        conclusion = (
            "Contradictory: ADF rejects unit root, but KPSS rejects stationarity. "
            "Possible trend-stationarity or structural break — consider differencing."
        )
    elif not adf_stationary and kpss_stationary:
        conclusion = (
            "Contradictory: ADF cannot reject unit root, but KPSS does not reject "
            "stationarity. Series may be near-integrated — proceed with caution."
        )
    else:
        conclusion = "Non-stationary: both ADF and KPSS indicate a unit root. Differencing recommended."

    return {
        # ADF results
        "adf_statistic":       adf_stat,
        "adf_pvalue":          adf_pvalue,
        "adf_stationary":      adf_stationary,
        "adf_critical_values": adf_cv,

        # KPSS results
        "kpss_statistic":       kpss_stat,
        "kpss_pvalue":          kpss_pvalue,
        "kpss_stationary":      kpss_stationary,
        "kpss_critical_values": kpss_cv,

        # Overall verdict
        "is_stationary": both_agree,
        "conclusion":    conclusion,
        "alpha":         alpha,
    }


def decompose_series(
    series: pd.Series,
    model: str = "additive",
    period: int = 12,
) -> dict:
    """
    Decompose a time series into trend, seasonal, and residual components
    using classical (statsmodels) seasonal decomposition.

    Parameters
    ----------
    series : pd.Series
        A clean, evenly-spaced univariate series with a ``DatetimeIndex``.
        Must have at least ``2 * period`` observations.
    model : {"additive", "multiplicative"}
        Additive assumes ``observed = trend + seasonal + residual``.
        Multiplicative assumes ``observed = trend * seasonal * residual``.
        Use ``"multiplicative"`` only when the amplitude of seasonality
        grows proportionally with the level (always positive series).
        Default ``"additive"``.
    period : int
        Number of observations per seasonal cycle.  12 for monthly data,
        4 for quarterly, 52 for weekly.  Default 12.

    Returns
    -------
    dict with keys:
        observed  – pd.Series : the original series (NaN-trimmed to match)
        trend     – pd.Series : centred moving-average trend component
        seasonal  – pd.Series : repeating seasonal pattern
        residual  – pd.Series : what remains after removing trend + seasonal

    Raises
    ------
    ValueError
        If ``model`` is not ``"additive"`` or ``"multiplicative"``.
    """
    if model not in {"additive", "multiplicative"}:
        raise ValueError(
            f"model must be 'additive' or 'multiplicative', got {model!r}"
        )

    result: DecomposeResult = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend="freq",  # fills leading/trailing NaN in trend
    )

    return {
        "observed": result.observed,
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }


def clean_series(series: pd.Series, freq: str | None = None) -> pd.Series:
    """
    Minimal, non-destructive cleaning suitable for use before stationarity
    tests or decomposition.

    Steps applied (in order):
        1. Cast values to ``float64`` — guards against object-dtype artefacts
           that sometimes appear after FRED downloads.
        2. Drop ``NaN`` rows — removes leading/trailing gaps as well as
           any data-revision placeholders FRED encodes as ``NaN``.
        3. Sort the index ascending — guarantees monotone ordering required
           by ``seasonal_decompose`` and ``adfuller``.
        4. If ``freq`` is specified, explicitly sets the index frequency.

    Parameters
    ----------
    series : pd.Series
        Raw series, typically the output of :func:`load_fred_series`.
    freq : str or None, optional
        Pandas frequency string to set on the index (e.g. 'MS' for month start).

    Returns
    -------
    pd.Series
        A new ``pd.Series`` with the same name and a clean ``DatetimeIndex``.
        The original ``series`` is never mutated.
    """
    cleaned = (
        series
        .astype("float64")
        .dropna()
        .sort_index()
    )
    if freq is not None:
        cleaned = cleaned.asfreq(freq)
    return cleaned


