from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    detail: str
    artifact_version: str | None = None


class MarketPoint(BaseModel):
    date: date
    series: str
    close: float


class MarketHistoryResponse(BaseModel):
    points: list[MarketPoint]
    data_as_of: date
    source_note: str
    data_quality: str


class ForecastResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    as_of_date: date
    target_date: date
    horizon: int
    current_price_cents_per_lb: float
    predicted_price_cents_per_lb: float
    predicted_return_pct: float
    direction: Literal["up", "down", "flat"]
    actual_price_cents_per_lb: float | None
    absolute_error: float | None
    direction_correct: bool | None
    origin_type: Literal["live", "backtest"]
    model_name: str
    model_version: str
    data_quality: str
    generated_at: datetime


class LatestForecastResponse(BaseModel):
    forecasts: list[ForecastResponse]
    data_as_of: date
    artifact_version: str
    data_quality: str
    stale: bool


class Contribution(BaseModel):
    feature: str
    display_name: str
    feature_value: float
    contribution_pct: float


class ExplanationResponse(BaseModel):
    forecast_id: str
    horizon: int
    model_name: str
    explainer: str
    base_value_pct: float
    predicted_return_pct: float
    contributions: list[Contribution]
    approximation_error_pct: float = 0.0
    equation: str = "ŷ = E[f(X)] + Σ φᵢ"


class SimulationAdjustments(BaseModel):
    dxy_pct_change: float = Field(default=0, ge=-5, le=5)
    wti_pct_change: float = Field(default=0, ge=-20, le=20)
    cftc_net_delta_contracts: int = Field(default=0, ge=-50_000, le=50_000)
    volatility_multiplier: float = Field(default=1, ge=0.5, le=2)


class SimulationRequest(BaseModel):
    as_of_date: date
    adjustments: SimulationAdjustments


class SimulationHorizonResult(BaseModel):
    horizon: int
    baseline_price_cents_per_lb: float
    scenario_price_cents_per_lb: float
    delta_cents_per_lb: float
    delta_pct: float
    model_name: str
    baseline_kind: Literal["production", "experimental"] = "production"


class SimulationResponse(BaseModel):
    id: str
    as_of_date: date
    results: list[SimulationHorizonResult]
    disclaimer: str = "Sensitivity simulation — not causal inference"
    model_version: str


class ModelMetric(BaseModel):
    model: str
    horizon: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    selected: bool
    data_quality: str
    walkforward: dict | None = None
    validation_metrics: dict | None = None
    parameters: dict | None = None
    training_history: list[dict] | None = None


class ModelEvaluationResponse(BaseModel):
    artifact_version: str | None = None
    walkforward_report: dict | None = None
    selection_audit: dict | None = None
    note: str


class ReplayResponse(BaseModel):
    as_of_date: date
    forecasts: list[ForecastResponse]


class ApiError(BaseModel):
    code: str
    message: str
    details: dict | list | None = None
