"""
tests/test_preprocessing.py
----------------------------
Smoke test — not a full unit-test suite.

Run with:
    python -m tests.test_preprocessing
    # or simply:
    python tests/test_preprocessing.py

What it does:
    1. Load FRED RSXFS (monthly retail sales, clothing) via data_loader.
    2. Clean the raw series via clean_series().
    3. Run check_stationarity() and assert the required keys are present.
    4. Run decompose_series() and assert the required keys are present.
    5. Print the output of every step so the results are visible at a glance.

No external test runner (pytest / unittest) is required.
"""

from __future__ import annotations

import sys
import traceback

# ---------------------------------------------------------------------------
# Path setup so the script runs from the project root without installation.
# ---------------------------------------------------------------------------
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import pandas as pd

from src.data_loader import load_fred_series
from src.preprocessing import clean_series, check_stationarity, decompose_series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SECTION = "\n" + "=" * 70 + "\n"
PASS_TAG = "\033[32m[PASS]\033[0m"
FAIL_TAG = "\033[31m[FAIL]\033[0m"

failures: list[str] = []


def _assert(condition: bool, label: str) -> None:
    """Record pass/fail without raising immediately so every check runs."""
    if condition:
        print(f"  {PASS_TAG}  {label}")
    else:
        print(f"  {FAIL_TAG}  {label}")
        failures.append(label)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
print(SECTION + "STEP 1 — load_fred_series('RSXFS')")

raw: pd.Series = load_fred_series("RSXFS", start="2000-01-01")

print(f"  Series name  : {raw.name}")
print(f"  Date range   : {raw.index[0].date()} → {raw.index[-1].date()}")
print(f"  Observations : {len(raw)}")
print(f"  NaN count    : {raw.isna().sum()}")
print(f"\n  Head:\n{raw.head().to_string()}")

# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
print(SECTION + "STEP 2 — clean_series()")

ts: pd.Series = clean_series(raw)

print(f"  Dtype after cleaning : {ts.dtype}")
print(f"  NaN after cleaning   : {ts.isna().sum()}")
print(f"  Index monotone asc.  : {ts.index.is_monotonic_increasing}")
print(f"  Observations kept    : {len(ts)}")
print(f"\n  Head:\n{ts.head().to_string()}")

_assert(ts.dtype == "float64",              "dtype is float64")
_assert(ts.isna().sum() == 0,              "no NaN values remain")
_assert(ts.index.is_monotonic_increasing,  "index is monotone ascending")

# ---------------------------------------------------------------------------
# 3. Stationarity check
# ---------------------------------------------------------------------------
print(SECTION + "STEP 3 — check_stationarity()")

stat_result: dict = check_stationarity(ts)

print(f"  ADF statistic   : {stat_result['adf_statistic']:.4f}")
print(f"  ADF p-value     : {stat_result['adf_pvalue']:.4f}")
print(f"  ADF stationary  : {stat_result['adf_stationary']}")
print(f"  KPSS statistic  : {stat_result['kpss_statistic']:.4f}")
print(f"  KPSS p-value    : {stat_result['kpss_pvalue']:.4f}")
print(f"  KPSS stationary : {stat_result['kpss_stationary']}")
print(f"  is_stationary   : {stat_result['is_stationary']}")
print(f"  conclusion      : {stat_result['conclusion']}")

REQUIRED_STAT_KEYS = {"adf_stationary", "kpss_stationary", "conclusion"}
for key in REQUIRED_STAT_KEYS:
    _assert(key in stat_result, f"check_stationarity() returns key '{key}'")

_assert(isinstance(stat_result["adf_stationary"], bool),  "adf_stationary is bool")
_assert(isinstance(stat_result["kpss_stationary"], bool), "kpss_stationary is bool")
_assert(isinstance(stat_result["conclusion"], str),       "conclusion is str")

# ---------------------------------------------------------------------------
# 4. Decomposition
# ---------------------------------------------------------------------------
print(SECTION + "STEP 4 — decompose_series()")

decomp: dict = decompose_series(ts, model="additive", period=12)

for component, values in decomp.items():
    print(f"  {component:<10}: {len(values)} observations, "
          f"dtype={values.dtype}, NaN={values.isna().sum()}")

print(f"\n  Trend head:\n{decomp['trend'].head().to_string()}")
print(f"\n  Seasonal head:\n{decomp['seasonal'].head().to_string()}")

REQUIRED_DECOMP_KEYS = {"trend", "seasonal", "residual", "observed"}
for key in REQUIRED_DECOMP_KEYS:
    _assert(key in decomp, f"decompose_series() returns key '{key}'")

for key in REQUIRED_DECOMP_KEYS:
    _assert(isinstance(decomp[key], pd.Series), f"decomp['{key}'] is pd.Series")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
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
