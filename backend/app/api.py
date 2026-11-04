import math
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.database import get_db
from app.demo import DEMO_SCHEMA_HASH, DEMO_VERSION
from app.models import (
    ArtifactImport,
    FeatureSnapshot,
    Forecast,
    MarketObservation,
    ModelVersion,
    Simulation,
)
from app.runtime import ArtifactRuntime, ArtifactRuntimeError
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
        runtime.load(with_snapshots=True)
        runtime_error = None
    except Exception as exc:  # readiness reports the precise artifact issue
        runtime.manifest = {}
        runtime.models = {}
        runtime_error = str(exc)


def _active_artifact_import(db: Session) -> ArtifactImport | None:
    return db.scalar(select(ArtifactImport).order_by(desc(ArtifactImport.imported_at), ArtifactImport.id.desc()))


def _active_artifact_version(db: Session) -> str | None:
    imported = _active_artifact_import(db)
    return imported.artifact_version if imported else None


def _require_active_runtime(imported: ArtifactImport | None) -> None:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime_error or "artifact runtime is not ready")
    if imported is None:
        raise HTTPException(
            status_code=503,
            detail="artifact files are valid but their database import has not completed",
        )
    if (
        runtime.manifest.get("artifact_version") != imported.artifact_version
        or runtime.manifest != imported.manifest
        or runtime.payload_checksum is None
        or runtime.payload_checksum != imported.checksum
    ):
        raise HTTPException(status_code=503, detail="runtime does not match the active database artifact")


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok", detail="API process is live")


@router.get("/health/ready", response_model=HealthResponse)
def ready(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    imported = _active_artifact_import(db)
    if not runtime.ready and imported is None and get_settings().demo_mode:
        demo_exists = db.scalar(select(ModelVersion.id).where(
            ModelVersion.artifact_version == DEMO_VERSION,
            ModelVersion.data_quality == "illustrative",
        ).limit(1))
        if demo_exists:
            return HealthResponse(
                status="ok",
                detail="development fixture mode; no validated Colab artifact loaded",
                artifact_version=DEMO_VERSION,
            )
    _require_active_runtime(imported)
    return HealthResponse(
        status="ok",
        detail="database and artifact runtime are ready",
        artifact_version=runtime.manifest["artifact_version"],
    )


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
    imported = _active_artifact_import(db)
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
    imported = _active_artifact_import(db)
    if imported is None:
        return ModelEvaluationResponse(note="Development fixture; no measured evaluation available")
    report = imported.manifest.get("walkforward_report")
    return ModelEvaluationResponse(
        artifact_version=imported.artifact_version,
        walkforward_report=report,
        selection_audit=imported.manifest.get("selection_audit"),
        note=(
            f"{report.get('fold_count', len(report.get('folds', [])))} pre-2024 nested chronological research folds; "
            "reused historical evidence, not an independent holdout. 2024+ is excluded from selection."
            if report and imported.manifest.get('artifact_schema_version') == 3 else
            "Four pre-2024-06-18 rolling-origin folds; 2024 onward is a previously "
            "observed descriptive historical audit; it never selects or rejects a candidate."
            if report else "Legacy artifact: walk-forward evidence is unavailable. Model superiority cannot be inferred unless metrics use verified identical forecast origins and targets."
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
    imported = _active_artifact_import(db)
    use_runtime = runtime.ready or imported is not None or not get_settings().demo_mode
    if use_runtime:
        _require_active_runtime(imported)
        query = query.where(ModelVersion.artifact_version == runtime.manifest["artifact_version"])
    else:
        query = query.where(
            ModelVersion.artifact_version == DEMO_VERSION,
            ModelVersion.data_quality == "illustrative",
        )
    forecasts = db.scalars(query).all()
    if not forecasts:
        raise HTTPException(status_code=404, detail="no live baseline exists for the requested date")
    if not use_runtime and db.scalar(select(FeatureSnapshot.id).where(
        FeatureSnapshot.as_of_date == payload.as_of_date,
        FeatureSnapshot.schema_hash == DEMO_SCHEMA_HASH,
    )) is None:
        raise HTTPException(status_code=409, detail="baseline feature snapshot is unavailable")

    adjustments = payload.adjustments.model_dump()
    results: list[SimulationHorizonResult] = []
    for forecast in forecasts:
        experimental = False
        if use_runtime:
            window = runtime.required_window(forecast.horizon)
            try:
                baseline_window = runtime.feature_window(payload.as_of_date, window)
            except ArtifactRuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            scenario_features = dict(baseline_window[-1])
            _apply_adjustments(scenario_features, adjustments)
            baseline_features = baseline_window if window > 1 else baseline_window[-1]
            features = [*baseline_window[:-1], scenario_features] if window > 1 else scenario_features
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
                    if use_runtime
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
    identity = row.model_version.metrics.get(
        "deployment_identity" if row.origin_type == "live" else "evaluation_identity", {}
    )
    predicted_direction = _direction(row.predicted_price - row.current_price)
    actual_direction = _direction(row.actual_price - row.current_price) if row.actual_price is not None else None
    return ForecastResponse(
        id=row.id,
        as_of_date=row.as_of_date,
        target_date=row.target_date,
        target_calendar_policy=("next_valid_cotton_observation; future_session_date_unknown"
                                if identity else "legacy_calendar_unverified"),
        model_role=identity.get("model_role", "legacy"),
        model_identity=identity.get("model_identity"),
        fit_cutoff=identity.get("fit_cutoff"),
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
