"""
tests/test_backtest.py
-----------------------
Smoke tests for src/backtest.py.

These tests use only synthetic data (sine wave + linear trend) and a
trivial model/forecast pair so no real model installation is required.
The goal is to verify the backtest *engine* — rolling mechanics, metric
shapes, error handling — not any specific forecasting model.

Run with:
    python tests/test_backtest.py
    # or:
    python -m tests.test_backtest
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import numpy as np
import pandas as pd

from src.backtest import calculate_metrics, walk_forward_backtest, plot_backtest_results

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

SECTION  = "\n" + "=" * 70 + "\n"
PASS_TAG = "\033[32m[PASS]\033[0m"
FAIL_TAG = "\033[31m[FAIL]\033[0m"

failures: list[str] = []


def _assert(condition: bool, label: str) -> None:
    if condition:
        print(f"  {PASS_TAG}  {label}")
    else:
        print(f"  {FAIL_TAG}  {label}")
        failures.append(label)


def _make_synthetic_series(n: int = 100) -> pd.Series:
    """
    Sine wave + linear trend — deterministic so tests are reproducible.

    Values are strictly positive (minimum ≈ 1.5), which keeps MAPE well-
    behaved and avoids the zero-division edge case in these smoke tests.
    """
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    t   = np.arange(n)
    values = 100 + 0.5 * t + 10 * np.sin(2 * math.pi * t / 12)
    return pd.Series(values, index=idx, name="synthetic")


# Trivial model: captures the training mean as the entire "model".
def _mean_model_fn(train: pd.Series) -> float:
    return float(train.mean())


# Trivial forecast: repeat the mean for however many steps requested.
def _mean_forecast_fn(fitted_model: float, steps: int) -> pd.Series:
    return pd.Series([fitted_model] * steps)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: calculate_metrics — shape and types
# ─────────────────────────────────────────────────────────────────────────────

print(SECTION + "TEST 1 — calculate_metrics()")

actuals   = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0])
forecasts = pd.Series([ 95.0, 112.0, 118.0, 133.0, 142.0])
metrics   = calculate_metrics(actuals, forecasts)

print(f"  metrics = {metrics}")

_assert(set(metrics.keys()) == {"mape", "rmse", "mase"},
        "calculate_metrics() returns exactly {'mape', 'rmse', 'mase'}")
_assert(all(isinstance(v, float) for v in metrics.values()),
        "all metric values are float")
_assert(all(not math.isnan(v) for v in metrics.values()),
        "no metric value is NaN")
_assert(metrics["mape"] >= 0.0, "MAPE is non-negative")
_assert(metrics["rmse"] >= 0.0, "RMSE is non-negative")
_assert(metrics["mase"] >= 0.0, "MASE is non-negative")

# Length mismatch must raise ValueError
raised = False
try:
    calculate_metrics(pd.Series([1, 2, 3]), pd.Series([1, 2]))
except ValueError:
    raised = True
_assert(raised, "calculate_metrics() raises ValueError on length mismatch")

# Perfect forecast → MAPE = 0, RMSE = 0
perfect = calculate_metrics(actuals, actuals)
_assert(perfect["mape"] == 0.0, "perfect forecast gives MAPE = 0.0")
_assert(perfect["rmse"] == 0.0, "perfect forecast gives RMSE = 0.0")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: walk_forward_backtest — mechanics and return shape
# ─────────────────────────────────────────────────────────────────────────────

print(SECTION + "TEST 2 — walk_forward_backtest()")

series = _make_synthetic_series(100)
result = walk_forward_backtest(
    series=series,
    model_fn=_mean_model_fn,
    forecast_fn=_mean_forecast_fn,
    initial_train_size=60,
    step_size=1,
    forecast_horizon=6,
)

print(f"  n_origins = {result['n_origins']}")
print(f"  metrics   = {result['metrics']}")
print(f"  len(forecasts) = {len(result['forecasts'])}")
print(f"  len(actuals)   = {len(result['actuals'])}")

_assert("forecasts" in result,   "result has key 'forecasts'")
_assert("actuals"   in result,   "result has key 'actuals'")
_assert("metrics"   in result,   "result has key 'metrics'")
_assert("n_origins" in result,   "result has key 'n_origins'")

_assert(result["n_origins"] > 0, "n_origins > 0")
_assert(len(result["forecasts"]) == result["n_origins"],
        "len(forecasts) == n_origins")
_assert(len(result["actuals"]) == result["n_origins"],
        "len(actuals) == n_origins")

_assert(set(result["metrics"].keys()) == {"mape", "rmse", "mase"},
        "aggregated metrics has exactly {'mape', 'rmse', 'mase'}")

# Each forecast window must have exactly forecast_horizon points.
_assert(all(len(f) == 6 for f in result["forecasts"]),
        "each forecast window has exactly forecast_horizon=6 points")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: walk_forward_backtest — step_size > 1 produces fewer windows
# ─────────────────────────────────────────────────────────────────────────────

print(SECTION + "TEST 3 — step_size > 1 produces fewer windows")

result_step6 = walk_forward_backtest(
    series=series,
    model_fn=_mean_model_fn,
    forecast_fn=_mean_forecast_fn,
    initial_train_size=60,
    step_size=6,
    forecast_horizon=6,
)
result_step1 = result   # reuse from Test 2

print(f"  n_origins (step=1): {result_step1['n_origins']}")
print(f"  n_origins (step=6): {result_step6['n_origins']}")

_assert(result_step6["n_origins"] < result_step1["n_origins"],
        "step_size=6 produces fewer origins than step_size=1")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: walk_forward_backtest — bad model_fn is tolerated (skip + warn)
# ─────────────────────────────────────────────────────────────────────────────

print(SECTION + "TEST 4 — failing model_fn is skipped, engine doesn't crash")


def _always_fails(train: pd.Series):
    raise RuntimeError("Deliberate failure for testing")


result_fail = walk_forward_backtest(
    series=series,
    model_fn=_always_fails,
    forecast_fn=_mean_forecast_fn,
    initial_train_size=60,
    step_size=1,
    forecast_horizon=6,
)

_assert(result_fail["n_origins"] == 0,
        "all-failing model_fn yields n_origins=0 (not an exception)")
_assert(isinstance(result_fail["metrics"]["mape"], float),
        "metrics still returned (as NaN floats) when no origins succeed")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: plot_backtest_results — returns Figure, no plt.show() side-effect
# ─────────────────────────────────────────────────────────────────────────────

print(SECTION + "TEST 5 — plot_backtest_results() returns plt.Figure")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in CI / scripts
import matplotlib.pyplot as plt

fig = plot_backtest_results(series, result, title="Synthetic Backtest Smoke Test")

_assert(isinstance(fig, plt.Figure), "plot_backtest_results() returns plt.Figure")
plt.close(fig)          # clean up so we don't leak figures

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

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
