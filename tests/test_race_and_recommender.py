"""
tests/test_race_and_recommender.py
----------------------------------
Smoke tests for model race execution and automated parameters recommendation.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.data_loader import load_fred_series
from src.preprocessing import clean_series
from src.recommender import recommend_config
from src.models import run_model_race

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
    print(SECTION + "STEP 1: Loading and Preprocessing RSXFS")
    raw = load_fred_series("RSXFS", start="2000-01-01")
    ts = clean_series(raw, freq="MS")
    print(f"  Loaded RSXFS series: {len(ts)} observations, {ts.index[0].date()} to {ts.index[-1].date()}")

    print(SECTION + "STEP 2: Automated Parameter Recommendation")
    rec = recommend_config(ts)
    
    # Print the full output cleanly
    import pprint
    pprint.pprint(rec)

    # Assertions on parameter output
    _assert(isinstance(rec["recommended_models"], list) and len(rec["recommended_models"]) > 0,
            "recommended_models is a non-empty list")
    _assert(rec["forecast_horizon"] in [3, 6, 12],
            f"forecast_horizon {rec['forecast_horizon']} is one of [3, 6, 12]")
    
    expected_keys = {"forecast_horizon", "initial_train_size", "recommended_models", "arima_order"}
    _assert(set(rec.keys()).issuperset(expected_keys),
            "recommend_config output has all expected keys")
    _assert(set(rec["reasoning"].keys()) == expected_keys,
            "reasoning dict has the identical keys as the main output")

    print(SECTION + "STEP 3: Running Model Race (naive + arima subset)")
    
    leaderboard = run_model_race(
        series=ts,
        initial_train_size=120,
        step_size=12,  # larger step size for fast test execution
        forecast_horizon=rec["forecast_horizon"],
        models_to_run=["naive", "arima"]
    )

    _assert(len(leaderboard) == 2, f"leaderboard has 2 entries (got {len(leaderboard)})")
    
    # Assert status of both is 'ok' or 'failed'
    all_status_valid = all(item["status"] in ["ok", "failed"] for item in leaderboard)
    _assert(all_status_valid, "all entries have valid status ('ok' or 'failed')")

    # Assert sorted by MASE ascending, with None/NaN at the bottom
    import math
    
    is_sorted = True
    prev_mase = -1.0
    for item in leaderboard:
        status = item["status"]
        mase = item["mase"]
        
        if status == "failed" or mase is None or math.isnan(mase):
            idx = leaderboard.index(item)
            for subsequent in leaderboard[idx + 1:]:
                sub_status = subsequent["status"]
                sub_mase = subsequent["mase"]
                if sub_status != "failed" and sub_mase is not None and not math.isnan(sub_mase):
                    is_sorted = False
                    break
        else:
            if mase < prev_mase:
                is_sorted = False
                break
            prev_mase = mase

    _assert(is_sorted, "leaderboard is sorted by MASE ascending with failures at the bottom")

    print(SECTION + "FINAL LEADERBOARD RESULTS:")
    print(f"{'Model':<20} | {'MAPE':<10} | {'RMSE':<12} | {'MASE':<10} | {'Origins':<8} | {'Fit Time (s)':<12} | {'Status':<8}")
    print("-" * 88)
    for row in leaderboard:
        m_name = row["model_name"]
        mape_val = f"{row['mape']:.4f}%" if row["mape"] is not None else "N/A"
        rmse_val = f"{row['rmse']:.4f}" if row["rmse"] is not None else "N/A"
        mase_val = f"{row['mase']:.4f}" if row["mase"] is not None else "N/A"
        origins_val = str(row["n_origins"])
        time_val = f"{row['fit_time_seconds']:.2f}s"
        status_val = row["status"]
        print(f"{m_name:<20} | {mape_val:<10} | {rmse_val:<12} | {mase_val:<10} | {origins_val:<8} | {time_val:<12} | {status_val:<8}")

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
