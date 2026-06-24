from pydantic import BaseModel, Field
from typing import Optional

class ForecastRequest(BaseModel):
    series_id: str = Field(default="RSXFS",
        description="FRED series ID e.g. RSXFS, INDPRO, HOUST")
    start_date: str = Field(default="2000-01-01")
    horizon: int = Field(default=12, ge=1, le=24,
        description="Forecast horizon in months")
    model: str = Field(default="arima",
        description="Model to use: arima or naive")

class ForecastResponse(BaseModel):
    series_id: str
    horizon: int
    model_used: str
    forecast_dates: list[str]
    forecast_values: list[float]
    last_observed_date: str
    last_observed_value: float

class BacktestRequest(BaseModel):
    series_id: str = Field(default="RSXFS")
    start_date: str = Field(default="2000-01-01")
    initial_train_size: int = Field(default=120, ge=60)
    step_size: int = Field(default=3, ge=1)
    forecast_horizon: int = Field(default=12, ge=1, le=24)
    models_to_run: Optional[list[str]] = None

class BacktestResponse(BaseModel):
    series_id: str
    leaderboard: list[dict]
    n_origins: int
    recommended_model: str

class AnomalyRequest(BaseModel):
    series_id: str = Field(default="RSXFS")
    start_date: str = Field(default="2000-01-01")
    threshold_std: float = Field(default=2.5, ge=1.5, le=4.0)

class AnomalyResponse(BaseModel):
    series_id: str
    n_anomalies: int
    anomalies: list[dict]

class ExplainRequest(BaseModel):
    series_id: str = Field(default="RSXFS")
    start_date: str = Field(default="2000-01-01")
    anomaly_date: str = Field(
        description="Date string e.g. '2020-04-01'")
    threshold_std: float = Field(default=2.5)

class ExplainResponse(BaseModel):
    anomaly_date: str
    explanation: str
    confidence: str
    factors: list[str]

class RecommendResponse(BaseModel):
    series_id: str
    forecast_horizon: int
    initial_train_size: int
    recommended_models: list[str]
    arima_order: list[int]
    reasoning: dict
