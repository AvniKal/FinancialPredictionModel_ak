from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from api.models import (
    ForecastRequest, ForecastResponse,
    BacktestRequest, BacktestResponse,
    AnomalyRequest, AnomalyResponse,
    ExplainRequest, ExplainResponse,
    RecommendResponse
)
from src.data_loader import fetch_fred_series
from src.preprocessing import clean_series, decompose_series
from src.models import run_model_race, get_arima_fns, get_naive_fns
from src.backtest import walk_forward_backtest
from src.agents import detect_anomalies, get_anomaly_context, explain_anomaly
from src.recommender import recommend_config

app = FastAPI(
    title="AI Financial Forecasting API",
    description="REST API for time series forecasting, anomaly detection, and LLM explanation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── helper ──────────────────────────────────────────────
def load_and_clean(series_id: str, start: str) -> pd.Series:
    """Shared helper: fetch from FRED and clean."""
    try:
        series = fetch_fred_series(series_id, start=start)
        return clean_series(series, freq='MS')
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not load series '{series_id}': {str(e)}"
        )

# ── routes ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AI Financial Forecasting API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /health",
            "POST /recommend",
            "POST /forecast",
            "POST /backtest",
            "POST /anomalies",
            "POST /explain"
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(series_id: str = "RSXFS", start_date: str = "2000-01-01"):
    """
    Analyze a series and return recommended modeling configuration.
    """
    series = load_and_clean(series_id, start_date)
    config = recommend_config(series)
    return RecommendResponse(
        series_id=series_id,
        forecast_horizon=config['forecast_horizon'],
        initial_train_size=config['initial_train_size'],
        recommended_models=config['recommended_models'],
        arima_order=list(config['arima_order']),
        reasoning=config['reasoning']
    )

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Fit a model on the full series and return a future forecast.
    """
    series = load_and_clean(req.series_id, req.start_date)

    if req.model == "arima":
        model_fn, forecast_fn = get_arima_fns(order=(2, 1, 1))
    elif req.model == "naive":
        model_fn, forecast_fn = get_naive_fns()
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{req.model}'. Choose 'arima' or 'naive'."
        )

    try:
        fitted = model_fn(series)
        fc = forecast_fn(fitted, req.horizon)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model fitting failed: {str(e)}"
        )

    return ForecastResponse(
        series_id=req.series_id,
        horizon=req.horizon,
        model_used=req.model,
        forecast_dates=[str(d.date()) for d in fc.index],
        forecast_values=[round(float(v), 2) for v in fc.values],
        last_observed_date=str(series.index[-1].date()),
        last_observed_value=round(float(series.iloc[-1]), 2)
    )

@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    """
    Run walk-forward backtest model race and return leaderboard.
    This endpoint may take 30-60 seconds for full model race.
    """
    series = load_and_clean(req.series_id, req.start_date)

    leaderboard = run_model_race(
        series,
        initial_train_size=req.initial_train_size,
        step_size=req.step_size,
        forecast_horizon=req.forecast_horizon,
        models_to_run=req.models_to_run
    )

    ok_models = [m for m in leaderboard if m['status'] == 'ok']
    recommended = ok_models[0]['model_name'] if ok_models else 'none'

    total_origins = ok_models[0]['n_origins'] if ok_models else 0

    return BacktestResponse(
        series_id=req.series_id,
        leaderboard=leaderboard,
        n_origins=total_origins,
        recommended_model=recommended
    )

@app.post("/anomalies", response_model=AnomalyResponse)
def anomalies(req: AnomalyRequest):
    """
    Detect anomalies in a series using residual z-score method.
    """
    series = load_and_clean(req.series_id, req.start_date)
    decomp = decompose_series(series, model='multiplicative', period=12)
    anomaly_df = detect_anomalies(
        series,
        decomp['residual'],
        threshold_std=req.threshold_std
    )

    records = []
    for date, row in anomaly_df.iterrows():
        records.append({
            "date": str(date.date()),
            "observed": round(row['observed'], 2),
            "residual": round(row['residual'], 6),
            "deviation_pct": round(row['deviation_pct'], 4),
            "direction": row['direction'],
            "severity": row['severity']
        })

    return AnomalyResponse(
        series_id=req.series_id,
        n_anomalies=len(records),
        anomalies=records
    )

@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """
    Detect anomalies, find the requested date, and return LLM explanation.
    This endpoint calls the Groq API and may take 2-5 seconds.
    """
    series = load_and_clean(req.series_id, req.start_date)
    decomp = decompose_series(series, model='multiplicative', period=12)
    anomaly_df = detect_anomalies(
        series,
        decomp['residual'],
        threshold_std=req.threshold_std
    )

    try:
        anomaly_date = pd.Timestamp(req.anomaly_date)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format: '{req.anomaly_date}'. Use YYYY-MM-DD."
        )

    if anomaly_date not in anomaly_df.index:
        raise HTTPException(
            status_code=404,
            detail=f"Date {req.anomaly_date} is not flagged as an anomaly "
                   f"at threshold {req.threshold_std}. "
                   f"Try lowering the threshold or use /anomalies to see flagged dates."
        )

    context = get_anomaly_context(anomaly_date, series)
    result = explain_anomaly(context)

    return ExplainResponse(
        anomaly_date=req.anomaly_date,
        explanation=result['explanation'],
        confidence=result['confidence'],
        factors=result['factors']
    )
