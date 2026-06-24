"""
src/agents.py
-------------
Anomaly detection and LLM explanation agent.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()


def detect_anomalies(
    series: pd.Series,
    residuals: pd.Series,
    threshold_std: float = 2.5
) -> pd.DataFrame:
    """
    Flag anomalies where residuals deviate beyond threshold_std
    standard deviations from the residual mean.

    Parameters
    ----------
    series : pd.Series
        The original observed series (for context values)
    residuals : pd.Series
        Residual component from decompose_series()
    threshold_std : float
        How many std deviations = anomalous. Default 2.5.
        (tighter than 3-sigma because our residuals are well-behaved)

    Returns
    -------
    pd.DataFrame with columns:
        date          : the anomaly timestamp (index)
        observed      : actual series value at that date
        residual      : residual value
        deviation_pct : how far residual deviates from mean, in %
        direction     : 'above' or 'below' expected
        severity      : 'moderate' (2.5-3.5 std) or 'severe' (>3.5 std)

    Sort by abs(deviation_pct) descending.
    """
    # Drop any leading/trailing NaNs from residuals and align with observed series
    clean_resid = residuals.dropna()
    aligned_series = series.loc[clean_resid.index]

    mean_resid = clean_resid.mean()
    std_resid = clean_resid.std()

    deviations = clean_resid - mean_resid
    abs_deviations = deviations.abs()
    anomaly_mask = abs_deviations > threshold_std * std_resid

    anomaly_indices = clean_resid.index[anomaly_mask]

    records = []
    for date in anomaly_indices:
        obs_val = aligned_series.loc[date]
        resid_val = clean_resid.loc[date]
        dev_val = deviations.loc[date]
        
        # deviation_pct: how far residual deviates from mean, in %
        # For multiplicative residuals (mean close to 1.0), it represents the percentage deviation
        dev_pct = (dev_val / mean_resid) * 100

        direction = "above" if dev_val > 0 else "below"
        
        # severity: 'moderate' (2.5-3.5 std) or 'severe' (>3.5 std)
        num_stds = abs_deviations.loc[date] / std_resid
        severity = "severe" if num_stds > 3.5 else "moderate"

        records.append({
            "date": date,
            "observed": float(obs_val),
            "residual": float(resid_val),
            "deviation_pct": float(dev_pct),
            "direction": direction,
            "severity": severity
        })

    if not records:
        df = pd.DataFrame(columns=["observed", "residual", "deviation_pct", "direction", "severity"])
        df.index.name = "date"
        return df

    df = pd.DataFrame(records).set_index("date")
    df = df.sort_values(by="deviation_pct", key=lambda x: x.abs(), ascending=False)
    return df


def get_anomaly_context(
    anomaly_date: pd.Timestamp,
    series: pd.Series,
    window_months: int = 3
) -> dict:
    """
    Extract context around an anomaly date for passing to the LLM.

    Returns dict with:
        date_str      : formatted date string e.g. "April 2020"
        observed      : value at anomaly date
        expected      : rolling mean of preceding 12 months
                        (proxy for what the model "expected")
        pct_deviation : (observed - expected) / expected * 100
        prior_trend   : 'increasing', 'decreasing', or 'stable'
                        based on slope of 6 months before anomaly
        window_values : dict of {date_str: value} for
                        window_months before and after the anomaly
                        (gives LLM local context)
    """
    # 1. Format date
    date_str = anomaly_date.strftime("%B %Y")
    observed_val = float(series.loc[anomaly_date])

    # Find integer position of the anomaly date
    idx = series.index.get_loc(anomaly_date)

    # 2. Expected value: rolling mean of preceding 12 months
    if idx >= 12:
        preceding = series.iloc[idx - 12:idx]
    else:
        preceding = series.iloc[:idx] if idx > 0 else series.iloc[:1]
    expected_val = float(preceding.mean())

    # 3. pct_deviation
    pct_dev = ((observed_val - expected_val) / expected_val * 100) if expected_val != 0 else 0.0

    # 4. prior_trend: based on slope of 6 months before anomaly
    if idx >= 6:
        trend_window = series.iloc[idx - 6:idx]
    else:
        trend_window = series.iloc[:idx] if idx > 0 else series.iloc[:1]

    if len(trend_window) >= 2:
        x = np.arange(len(trend_window))
        y = trend_window.values
        slope, _ = np.polyfit(x, y, 1)
        
        # pct change per month relative to window mean
        mean_trend = trend_window.mean()
        rel_slope = (slope / mean_trend * 100) if mean_trend != 0 else slope
        
        # If trend change is less than 0.2% per month, classify as stable
        if abs(rel_slope) < 0.2:
            prior_trend = "stable"
        else:
            prior_trend = "increasing" if slope > 0 else "decreasing"
    else:
        prior_trend = "stable"

    # 5. window_values: dict of {date_str: value} for window_months before and after
    start_win = max(0, idx - window_months)
    end_win = min(len(series), idx + window_months + 1)
    win_series = series.iloc[start_win:end_win]

    win_values = {
        d.strftime("%B %Y"): float(val)
        for d, val in win_series.items()
    }

    return {
        "date_str": date_str,
        "observed": observed_val,
        "expected": expected_val,
        "pct_deviation": pct_dev,
        "prior_trend": prior_trend,
        "window_values": win_values
    }


def explain_anomaly(
    anomaly_context: dict,
    model: str = "llama-3.1-8b-instant"
) -> dict:
    """
    Call Groq API to explain a detected anomaly.
    Same return structure as before:
    {
        explanation : str,
        confidence  : str  ('high', 'medium', or 'low'),
        factors     : list[str],
        raw_response: str
    }
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a financial analyst specializing in US retail
sales data. Analyze this anomaly and respond in EXACTLY this format:

EXPLANATION: [2-3 sentences on the most likely cause]
Confidence: [high/medium/low]
Factors:
1. [first contributing factor]
2. [second contributing factor]
3. [third contributing factor]

ANOMALY DETAILS:
- Date: {anomaly_context['date_str']}
- Observed value: {anomaly_context['observed']:,.0f}
- Expected value (12-month avg): {anomaly_context['expected']:,.0f}
- Deviation: {anomaly_context['pct_deviation']:+.2f}%
- Prior trend: {anomaly_context['prior_trend']}
- Local context window: {anomaly_context['window_values']}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        raw_text = response.choices[0].message.content

        # Parsing confidence
        confidence = "medium"
        conf_match = re.search(r"Confidence:\s*(high|medium|low)", raw_text, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).lower()

        # Parsing factors
        factors = []
        factors_match = re.search(r"Factors:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
        if factors_match:
            factors_text = factors_match.group(1)
            factors_list = re.findall(r"(?:\d+\.|\*|-)\s*(.*)", factors_text)
            factors = [f.strip() for f in factors_list if f.strip()][:3]

        # Parsing explanation paragraph
        explanation = raw_text
        explanation_match = re.search(r"(.*?)(?:Confidence:|$)", raw_text, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            # If EXPLANATION: prefix is present, strip it
            if explanation.upper().startswith("EXPLANATION:"):
                explanation = explanation[len("EXPLANATION:"):].strip()

        return {
            "explanation": explanation,
            "confidence": confidence,
            "factors": factors,
            "raw_response": raw_text
        }

    except Exception as e:
        return {
            "explanation": f"Explanation unavailable — API error: {str(e)}",
            "confidence": "low",
            "factors": [],
            "raw_response": ""
        }
