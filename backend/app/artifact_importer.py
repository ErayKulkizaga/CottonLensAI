import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.artifacts import sha256_file, verify_directory
from app.models import (
    ArtifactImport,
    FeatureSnapshot,
    Forecast,
    ForecastExplanation,
    MarketObservation,
    ModelVersion,
)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def import_artifact_directory(db: Session, root: Path) -> str:
    import pyarrow.parquet as pq

    manifest = verify_directory(root)
    version = manifest["artifact_version"]
    if db.scalar(select(ArtifactImport).where(ArtifactImport.artifact_version == version)):
        return version
    _remove_development_fixture(db)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    model_versions: dict[tuple[int, str], ModelVersion] = {}
    for metric in metrics:
        model = ModelVersion(
            artifact_version=version,
            horizon=int(metric["horizon"]),
            model_name=metric["model"],
            model_format=_model_format(manifest, int(metric["horizon"]), metric["model"]),
            selected=bool(metric.get("selected")),
            data_quality=manifest.get("data_quality", "validated_holdout"),
            metrics={
                key: float(metric[key])
                for key in ("mae", "rmse", "mape", "directional_accuracy")
            },
        )
        db.add(model)
        model_versions[(model.horizon, model.model_name)] = model
    db.flush()

    for row in pq.read_table(root / "market_history.parquet").to_pylist():
        observed_on = _as_date(row.get("observed_on") or row.get("date"))
        series = str(row["series"])
        existing = db.scalar(
            select(MarketObservation).where(
                MarketObservation.observed_on == observed_on,
                MarketObservation.series == series,
            )
        )
        if existing is None:
            db.add(
                MarketObservation(
                    observed_on=observed_on,
                    series=series,
                    close=float(row["close"]),
                    source="colab-artifact",
                )
            )

    schema_hash = manifest["feature_schema_hash"]
    for row in pq.read_table(root / "feature_snapshots.parquet").to_pylist():
        as_of = _as_date(row.pop("date"))
        if db.scalar(select(FeatureSnapshot).where(FeatureSnapshot.as_of_date == as_of)) is None:
            db.add(
                FeatureSnapshot(
                    as_of_date=as_of,
                    schema_hash=schema_hash,
                    values={key: float(value) for key, value in row.items() if value is not None},
                )
            )

    forecast_lookup: dict[str, Forecast] = {}
    selected_by_horizon = {
        model.horizon: model for model in model_versions.values() if model.selected
    }
    for row in pq.read_table(root / "forecasts.parquet").to_pylist():
        horizon = int(row["horizon"])
        model = selected_by_horizon.get(horizon)
        if model is None:
            model = next(item for (item_horizon, _), item in model_versions.items() if item_horizon == horizon)
        forecast = Forecast(
            id=str(row["id"]),
            as_of_date=_as_date(row["as_of_date"]),
            target_date=_as_date(row["target_date"]),
            horizon=horizon,
            current_price=float(row["current_price"]),
            predicted_price=float(row["predicted_price"]),
            predicted_return_pct=float(row["predicted_return_pct"]),
            actual_price=float(row["actual_price"]) if row.get("actual_price") is not None else None,
            origin_type=str(row["origin_type"]),
            model_version_id=model.id,
        )
        db.add(forecast)
        forecast_lookup[forecast.id] = forecast
    db.flush()

    for row in pq.read_table(root / "explanations.parquet").to_pylist():
        forecast_id = str(row["forecast_id"])
        if forecast_id not in forecast_lookup:
            continue
        contributions = row["contributions"]
        if isinstance(contributions, str):
            contributions = json.loads(contributions)
        db.add(
            ForecastExplanation(
                forecast_id=forecast_id,
                base_value=float(row["base_value"]),
                explainer=str(row["explainer"]),
                contributions=contributions,
            )
        )
    db.add(
        ArtifactImport(
            artifact_version=version,
            manifest=manifest,
            checksum=sha256_file(root / "manifest.json"),
        )
    )
    db.commit()
    return version


def _remove_development_fixture(db: Session) -> None:
    demo_models = list(
        db.scalars(select(ModelVersion).where(ModelVersion.data_quality == "illustrative")).all()
    )
    if not demo_models:
        return
    model_ids = [model.id for model in demo_models]
    forecast_ids = list(
        db.scalars(select(Forecast.id).where(Forecast.model_version_id.in_(model_ids))).all()
    )
    if forecast_ids:
        db.execute(delete(ForecastExplanation).where(ForecastExplanation.forecast_id.in_(forecast_ids)))
        db.execute(delete(Forecast).where(Forecast.id.in_(forecast_ids)))
    db.execute(delete(ModelVersion).where(ModelVersion.id.in_(model_ids)))
    db.execute(delete(MarketObservation).where(MarketObservation.source == "development-fixture"))
    db.execute(delete(FeatureSnapshot).where(FeatureSnapshot.schema_hash == "demo-schema-not-for-production"))


def _model_format(manifest: dict, horizon: int, model_name: str) -> str:
    for entry in manifest.get("production_models", []):
        if int(entry["horizon"]) == horizon and (
            entry.get("name") == model_name or entry.get("primary_forecast") == model_name
        ):
            return entry["format"]
    return "evaluation_only"
