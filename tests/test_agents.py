"""
tests/test_agents.py
--------------------
Smoke tests for anomaly detection and LLM explanation agent.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.data_loader import load_fred_series
from src.preprocessing import clean_series, decompose_series
from src.agents import detect_anomalies, get_anomaly_context, explain_anomaly

SECTION = "\n" + "=" * 70 + "\n"
PASS_TAG = "\033[32m[PASS]\033[0m"
FAIL_TAG = "\033[31m[FAIL]\033[0m"

failures: list[str] = []


def _assert(condition: bool, label: str) -> None:
    if condition:
        print(f"  {PASS_TAG}  {label}")
    else:
        print(f"  {FAIL_TAG}  {label}")
        failures.append(label)


def main():
    print(SECTION + "STEP 1: Load, Clean, and Decompose RSXFS")
    raw = load_fred_series("RSXFS", start="2000-01-01")
    ts = clean_series(raw, freq="MS")
    decomp = decompose_series(ts, model="multiplicative", period=12)
    residuals = decomp["residual"]
    print(f"  Loaded RSXFS series: {len(ts)} observations, {ts.index[0].date()} to {ts.index[-1].date()}")

    print(SECTION + "STEP 2: Anomaly Detection")
    df_anomalies = detect_anomalies(ts, residuals, threshold_std=2.5)
    
    print("\nFlagged Anomalies:")
    print(df_anomalies.to_string())
    print()

    # April 2020 must be flagged
    april_2020 = pd.Timestamp("2020-04-01")
    _assert(april_2020 in df_anomalies.index, "April 2020 is flagged as an anomaly")

    # Assert severity is either 'moderate' or 'severe'
    all_severities_valid = all(s in ["moderate", "severe"] for s in df_anomalies["severity"])
    _assert(all_severities_valid, "All anomalies have severity of 'moderate' or 'severe'")

    print(SECTION + "STEP 3: Anomaly Context (April 2020)")
    context = get_anomaly_context(april_2020, ts, window_months=3)
    
    import pprint
    pprint.pprint(context)
    print()

    expected_keys = {"date_str", "observed", "expected", "pct_deviation", "prior_trend", "window_values"}
    _assert(set(context.keys()) == expected_keys, "Context dictionary has all required keys")
    _assert(context["date_str"] == "April 2020", f"date_str is 'April 2020' (got '{context['date_str']}')")

    print(SECTION + "STEP 4: Mocked LLM Explanation (broken key test)")
    original_key = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = "dummy-invalid-key"
    
    try:
        explanation_res = explain_anomaly(context)
    finally:
        if original_key is not None:
            os.environ["GROQ_API_KEY"] = original_key
        else:
            os.environ.pop("GROQ_API_KEY", None)

    print("Returned Fallback Dict:")
    pprint.pprint(explanation_res)
    print()

    _assert(explanation_res["confidence"] == "low", "crashed call returns confidence='low'")
    _assert(explanation_res["factors"] == [], "crashed call returns empty factors list")
    _assert("API error" in explanation_res["explanation"], "explanation contains API error message")

    print(SECTION)
    if not failures:
        print(f"{PASS_TAG}  All assertions passed.\n")
        sys.exit(0)
    else:
        print(f"{FAIL_TAG}  {len(failures)} assertion(s) failed:")
        for f in failures:
            print(f"        • {f}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
