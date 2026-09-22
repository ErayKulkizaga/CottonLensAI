import math
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.database import get_db
from app.models import (
    ArtifactImport,
    FeatureSnapshot,
    Forecast,
    MarketObservation,
    ModelVersion,
    Simulation,
)
from app.runtime import ArtifactRuntime
from app.schemas import (
    ExplanationResponse,
    ForecastResponse,
    HealthResponse,
    LatestForecastResponse,
    MarketHistoryResponse,
    MarketPoint,
    ModelEvaluationResponse,
    ModelMetric,
    ReplayResponse,
    SimulationHorizonResult,
    SimulationRequest,
    SimulationResponse,
)

router = APIRouter(prefix="/api/v1")
runtime = ArtifactRuntime(get_settings().artifact_dir)
runtime_error: str | None = None


def load_runtime() -> None:
    global runtime_error
    try:
        runtime.load()
        runtime_error = None
    except Exception as exc:  # readiness reports the precise artifact issue
        runtime_error = str(exc)


def _active_artifact_version(db: Session) -> str | None:
    imported = db.scalar(select(ArtifactImport).order_by(desc(ArtifactImport.imported_at)))
    return imported.artifact_version if imported else None


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok", detail="API process is live")


@router.get("/health/ready", response_model=HealthResponse)
def ready(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    has_imported_model = bool(db.scalar(select(func.count()).select_from(ModelVersion)))
    runtime_version = runtime.manifest.get("artifact_version") if runtime.ready else None
    matching_import = (
        db.scalar(select(ArtifactImport).where(ArtifactImport.artifact_version == runtime_version))
        if runtime_version
        else None
    )
    if runtime.ready:
        if matching_import:
            return HealthResponse(
                status="ok",
                detail="database and artifact runtime are ready",
                artifact_version=runtime_version,
            )
        raise HTTPException(
            status_code=503,
            detail="artifact files are valid but their database import has not completed",
        )
    if get_settings().demo_mode and has_imported_model:
        return HealthResponse(
            status="ok",
            detail="development fixture mode; no validated Colab artifact loaded",
            artifact_version="demo-fixture-v1",
        )
    raise HTTPException(status_code=503, detail=runtime_error or "artifact runtime is not ready")


@router.get("/market/history", response_model=MarketHistoryResponse)
def market_history(
    start: date | None = None,
    end: date | None = None,
    series: list[str] = Query(default=["cotton"]),
    db: Session = Depends(get_db),
) -> MarketHistoryResponse:
    conditions = [MarketObservation.series.in_(series)]
    if start:
        conditions.append(MarketObservation.observed_on >= start)
    if end:
        conditions.append(MarketObservation.observed_on <= end)
    observations = db.scalars(
        select(MarketObservation)
        .where(and_(*conditions))
        .order_by(MarketObservation.observed_on, MarketObservation.series)
    ).all()
    if not observations:
        raise HTTPException(status_code=404, detail="no market observations match the request")
    model = db.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)))
    return MarketHistoryResponse(
        points=[MarketPoint(date=row.observed_on, series=row.series, close=row.close) for row in observations],
        data_as_of=max(row.observed_on for row in observations),
        source_note=(
            "Yahoo Finance continuous futures proxy; not an official ICE settlement feed."
        ),
        data_quality=model.data_quality if model else "unknown",
    )


@router.get("/forecasts/latest", response_model=LatestForecastResponse)
def latest_forecasts(db: Session = Depends(get_db)) -> LatestForecastResponse:
    active_version = _active_artifact_version(db)
    latest_query = select(func.max(Forecast.as_of_date)).join(ModelVersion).where(Forecast.origin_type == "live")
    if active_version:
        latest_query = latest_query.where(ModelVersion.artifact_version == active_version)
    latest_date = db.scalar(latest_query)
    if latest_date is None:
        raise HTTPException(status_code=404, detail="no live forecasts are available")
    query = (
        select(Forecast).join(ModelVersion)
        .options(joinedload(Forecast.model_version))
        .where(Forecast.origin_type == "live", Forecast.as_of_date == latest_date)
        .order_by(Forecast.horizon)
    )
    if active_version:
        query = query.where(ModelVersion.artifact_version == active_version)
    rows = db.scalars(query).all()
    mapped = [_forecast_response(row) for row in rows]
    return LatestForecastResponse(
        forecasts=mapped,
        data_as_of=latest_date,
        artifact_version=mapped[0].model_version,
        data_quality=mapped[0].data_quality,
        stale=(date.today() - latest_date).days > 5,
    )


@router.get("/forecasts/history", response_model=list[ForecastResponse])
def forecast_history(
    horizon: int = Query(default=1),
    origin_type: str = Query(default="backtest"),
    limit: int = Query(default=80, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[ForecastResponse]:
    if horizon not in (1, 5):
        raise HTTPException(status_code=422, detail="horizon must be 1 or 5")
    if origin_type not in ("backtest", "live"):
        raise HTTPException(status_code=422, detail="origin_type must be backtest or live")
    query = (
        select(Forecast).join(ModelVersion)
        .options(joinedload(Forecast.model_version))
        .where(Forecast.horizon == horizon, Forecast.origin_type == origin_type)
        .order_by(desc(Forecast.as_of_date))
        .limit(limit)
    )
    active_version = _active_artifact_version(db)
    if active_version:
        query = query.where(ModelVersion.artifact_version == active_version)
    rows = db.scalars(query).all()
    return [_forecast_response(row) for row in reversed(rows)]


@router.get("/forecasts/{forecast_id}/explanation", response_model=ExplanationResponse)
def forecast_explanation(forecast_id: str, db: Session = Depends(get_db)) -> ExplanationResponse:
    row = db.scalar(
        select(Forecast)
        .options(joinedload(Forecast.model_version), joinedload(Forecast.explanation))
        .where(Forecast.id == forecast_id)
    )
    if row is None or row.explanation is None:
        raise HTTPException(status_code=404, detail="forecast explanation is unavailable")
    return ExplanationResponse(
        forecast_id=row.id,
        horizon=row.horizon,
        model_name=row.model_version.model_name,
        explainer=row.explanation.explainer,
        base_value_pct=row.explanation.base_value,
        predicted_return_pct=row.predicted_return_pct,
        contributions=row.explanation.contributions,
        approximation_error_pct=(
            row.predicted_return_pct - row.explanation.base_value
            - sum(item["contribution_pct"] for item in row.explanation.contributions)
        ),
    )


@router.get("/models/metrics", response_model=list[ModelMetric])
def model_metrics(db: Session = Depends(get_db)) -> list[ModelMetric]:
    imported = db.scalar(select(ArtifactImport).order_by(desc(ArtifactImport.imported_at)))
    query = select(ModelVersion).order_by(ModelVersion.horizon, ModelVersion.model_name)
    if imported:
        query = query.where(ModelVersion.artifact_version == imported.artifact_version)
    rows = db.scalars(query).all()
    return [
        ModelMetric(
            model=row.model_name,
            horizon=row.horizon,
            selected=row.selected,
            data_quality=row.data_quality,
            **row.metrics,
        )
        for row in rows
    ]


@router.get("/models/evaluation", response_model=ModelEvaluationResponse)
def model_evaluation(db: Session = Depends(get_db)) -> ModelEvaluationResponse:
    imported = db.scalar(select(ArtifactImport).order_by(desc(ArtifactImport.imported_at)))
    if imported is None:
        return ModelEvaluationResponse(note="Development fixture; no measured evaluation available")
    report = imported.manifest.get("walkforward_report")
    return ModelEvaluationResponse(
        artifact_version=imported.artifact_version,
        walkforward_report=report,
        selection_audit=imported.manifest.get("selection_audit"),
        note=(
            "Four pre-2024-06-18 rolling-origin folds; 2024 onward is a previously "
            "observed historical audit, not an independent test."
            if report else "Legacy artifact: walk-forward evidence is unavailable."
        ),
    )


@router.get("/replay/{as_of_date}", response_model=ReplayResponse)
def replay(as_of_date: date, db: Session = Depends(get_db)) -> ReplayResponse:
    query = (
        select(Forecast).join(ModelVersion)
        .options(joinedload(Forecast.model_version))
        .where(Forecast.as_of_date == as_of_date, Forecast.origin_type == "backtest")
        .order_by(Forecast.horizon)
    )
    active_version = _active_artifact_version(db)
    if active_version:
        query = query.where(ModelVersion.artifact_version == active_version)
    rows = db.scalars(query).all()
    if not rows:
        raise HTTPException(status_code=404, detail="no backtest forecast exists for this date")
    return ReplayResponse(as_of_date=as_of_date, forecasts=[_forecast_response(row) for row in rows])


@router.post("/simulations", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
def create_simulation(payload: SimulationRequest, db: Session = Depends(get_db)) -> SimulationResponse:
    query = (
        select(Forecast).join(ModelVersion)
        .options(joinedload(Forecast.model_version))
        .where(Forecast.as_of_date == payload.as_of_date, Forecast.origin_type == "live")
        .order_by(Forecast.horizon)
    )
    active_version = _active_artifact_version(db)
    if active_version:
        query = query.where(ModelVersion.artifact_version == active_version)
    forecasts = db.scalars(query).all()
    if not forecasts:
        raise HTTPException(status_code=404, detail="no live baseline exists for the requested date")
    snapshot = db.scalar(select(FeatureSnapshot).where(FeatureSnapshot.as_of_date == payload.as_of_date))
    if snapshot is None:
        raise HTTPException(status_code=409, detail="baseline feature snapshot is unavailable")

    adjustments = payload.adjustments.model_dump()
    results: list[SimulationHorizonResult] = []
    for forecast in forecasts:
        experimental = False
        if runtime.ready:
            if runtime.model_format(forecast.horizon) == "onnx":
                snapshots = list(
                    db.scalars(
                        select(FeatureSnapshot)
                        .where(FeatureSnapshot.as_of_date <= payload.as_of_date)
                        .order_by(desc(FeatureSnapshot.as_of_date))
                        .limit(60)
                    ).all()
                )
                if len(snapshots) != 60:
                    raise HTTPException(status_code=409, detail="60 feature snapshots are required for LSTM inference")
                features: dict[str, float] | list[dict[str, float]] = [
                    dict(item.values) for item in reversed(snapshots)
                ]
                baseline_features = [dict(row) for row in features]
                _apply_adjustments(features[-1], adjustments)
            else:
                features = dict(snapshot.values)
                baseline_features = dict(features)
                _apply_adjustments(features, adjustments)
            experimental = runtime.model_name(forecast.horizon).startswith("XGBoost experimental")
            baseline_log_return = runtime.predict_return(forecast.horizon, baseline_features)
            baseline_price = runtime.price_from_log_return(forecast.current_price, baseline_log_return)
            predicted_log_return = runtime.predict_return(forecast.horizon, features)
            scenario_price = runtime.price_from_log_return(forecast.current_price, predicted_log_return)
        else:
            baseline_price = forecast.predicted_price
            scenario_price = _demo_scenario_price(forecast, adjustments)
        delta = scenario_price - baseline_price
        results.append(
            SimulationHorizonResult(
                horizon=forecast.horizon,
                baseline_price_cents_per_lb=round(baseline_price, 4),
                scenario_price_cents_per_lb=round(scenario_price, 4),
                delta_cents_per_lb=round(delta, 4),
                delta_pct=round(delta / baseline_price * 100, 4),
                model_name=(
                    runtime.model_name(forecast.horizon)
                    if runtime.ready
                    else forecast.model_version.model_name
                ),
                baseline_kind="experimental" if experimental else "production",
            )
        )
    simulation_id = str(uuid.uuid4())
    version = forecasts[0].model_version.artifact_version
    db.add(
        Simulation(
            id=simulation_id,
            as_of_date=payload.as_of_date,
            adjustments=adjustments,
            results=[result.model_dump() for result in results],
            model_version=version,
        )
    )
    db.commit()
    return SimulationResponse(
        id=simulation_id,
        as_of_date=payload.as_of_date,
        results=results,
        model_version=version,
    )


def _forecast_response(row: Forecast) -> ForecastResponse:
    predicted_direction = _direction(row.predicted_price - row.current_price)
    actual_direction = _direction(row.actual_price - row.current_price) if row.actual_price is not None else None
    return ForecastResponse(
        id=row.id,
        as_of_date=row.as_of_date,
        target_date=row.target_date,
        horizon=row.horizon,
        current_price_cents_per_lb=row.current_price,
        predicted_price_cents_per_lb=row.predicted_price,
        predicted_return_pct=row.predicted_return_pct,
        direction=predicted_direction,
        actual_price_cents_per_lb=row.actual_price,
        absolute_error=(abs(row.predicted_price - row.actual_price) if row.actual_price is not None else None),
        direction_correct=(predicted_direction == actual_direction if actual_direction is not None else None),
        origin_type=row.origin_type,
        model_name=row.model_version.model_name,
        model_version=row.model_version.artifact_version,
        data_quality=row.model_version.data_quality,
        generated_at=row.created_at,
    )


def _direction(delta: float) -> str:
    if abs(delta) < 1e-8:
        return "flat"
    return "up" if delta > 0 else "down"


def _apply_adjustments(features: dict[str, float], adjustments: dict) -> None:
    if "dxy_ret_1" in features:
        features["dxy_ret_1"] += adjustments["dxy_pct_change"] / 100
    if "wti_ret_1" in features:
        features["wti_ret_1"] += adjustments["wti_pct_change"] / 100
    if "cftc_managed_money_net" in features:
        features["cftc_managed_money_net"] += adjustments["cftc_net_delta_contracts"]
    if "cotton_volatility_20" in features:
        features["cotton_volatility_20"] *= adjustments["volatility_multiplier"]


def _demo_scenario_price(forecast: Forecast, adjustments: dict) -> float:
    horizon_scale = math.sqrt(forecast.horizon)
    return_delta_pct = horizon_scale * (
        -0.04 * adjustments["dxy_pct_change"]
        + 0.015 * adjustments["wti_pct_change"]
        + 0.000004 * adjustments["cftc_net_delta_contracts"]
        - 0.22 * (adjustments["volatility_multiplier"] - 1)
    )
    return forecast.predicted_price * math.exp(return_delta_pct / 100)
