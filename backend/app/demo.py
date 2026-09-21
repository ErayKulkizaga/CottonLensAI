import math
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    FeatureSnapshot,
    Forecast,
    ForecastExplanation,
    MarketObservation,
    ModelVersion,
)

DEMO_VERSION = "demo-fixture-v1"
DEMO_SCHEMA_HASH = "demo-schema-not-for-production"


def _business_days(end: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def seed_demo_data(db: Session) -> None:
    existing = db.scalar(select(func.count()).select_from(ModelVersion))
    if existing:
        return

    dates = _business_days(date.today() - timedelta(days=1), 180)
    prices: list[float] = []
    for index, observed_on in enumerate(dates):
        cotton = 71.5 + 2.9 * math.sin(index / 15) + index * 0.018 + 0.35 * math.sin(index / 3)
        dxy = 102.0 + 1.4 * math.cos(index / 21)
        wti = 73.0 + 5.5 * math.sin(index / 19)
        prices.append(cotton)
        db.add_all(
            [
                MarketObservation(
                    observed_on=observed_on,
                    series="cotton",
                    close=round(cotton, 3),
                    source="development-fixture",
                ),
                MarketObservation(
                    observed_on=observed_on,
                    series="dxy",
                    close=round(dxy, 3),
                    source="development-fixture",
                ),
                MarketObservation(
                    observed_on=observed_on,
                    series="wti",
                    close=round(wti, 3),
                    source="development-fixture",
                ),
            ]
        )

    metric_sets = {
        1: {
            "Naive": (1.24, 1.58, 1.72, 49.1),
            "XGBoost": (0.82, 1.09, 1.13, 61.8),
            "LSTM": (0.91, 1.18, 1.26, 59.7),
        },
        5: {
            "Naive": (2.62, 3.31, 3.65, 48.6),
            "XGBoost": (1.78, 2.26, 2.47, 63.4),
            "LSTM": (1.83, 2.34, 2.53, 62.1),
        },
    }
    versions: dict[int, ModelVersion] = {}
    for horizon, model_metrics in metric_sets.items():
        for model_name, values in model_metrics.items():
            version = ModelVersion(
                artifact_version=DEMO_VERSION,
                horizon=horizon,
                model_name=model_name,
                model_format="development-fixture",
                selected=model_name == "XGBoost",
                data_quality="illustrative",
                metrics={
                    "mae": values[0],
                    "rmse": values[1],
                    "mape": values[2],
                    "directional_accuracy": values[3],
                },
            )
            db.add(version)
            if model_name == "XGBoost":
                versions[horizon] = version
    db.flush()

    for index in range(60, len(dates) - 5, 4):
        as_of = dates[index]
        for horizon in (1, 5):
            target_index = index + horizon
            current = prices[index]
            actual = prices[target_index]
            signal = 0.006 * math.sin(index / 7 + horizon) - 0.002 * math.cos(index / 13)
            predicted = current * math.exp(signal)
            forecast = Forecast(
                as_of_date=as_of,
                target_date=dates[target_index],
                horizon=horizon,
                current_price=round(current, 4),
                predicted_price=round(predicted, 4),
                predicted_return_pct=round(signal * 100, 4),
                actual_price=round(actual, 4),
                origin_type="backtest",
                model_version_id=versions[horizon].id,
            )
            db.add(forecast)
            db.flush()
            db.add(
                ForecastExplanation(
                    forecast_id=forecast.id,
                    base_value=0.08,
                    explainer="development fixture",
                    contributions=_contributions(index, signal * 100),
                )
            )

    latest_index = len(dates) - 1
    latest_date = dates[latest_index]
    latest_price = prices[latest_index]
    db.add(
        FeatureSnapshot(
            as_of_date=latest_date,
            schema_hash=DEMO_SCHEMA_HASH,
            values={
                "dxy_ret_1": -0.12,
                "wti_ret_1": 0.48,
                "cftc_managed_money_net": -42310,
                "cotton_volatility_20": 0.014,
            },
        )
    )
    for horizon, signal in ((1, 0.0068), (5, 0.0149)):
        forecast = Forecast(
            as_of_date=latest_date,
            target_date=latest_date + timedelta(days=1 if horizon == 1 else 7),
            horizon=horizon,
            current_price=round(latest_price, 4),
            predicted_price=round(latest_price * math.exp(signal), 4),
            predicted_return_pct=round(signal * 100, 4),
            actual_price=None,
            origin_type="live",
            model_version_id=versions[horizon].id,
        )
        db.add(forecast)
        db.flush()
        db.add(
            ForecastExplanation(
                forecast_id=forecast.id,
                base_value=0.08,
                explainer="development fixture",
                contributions=_contributions(latest_index, signal * 100),
            )
        )
    db.commit()


def _contributions(index: int, predicted_return_pct: float) -> list[dict]:
    raw = [
        ("wti_ret_5", "WTI momentum", 1.42, 0.31),
        ("cftc_net_z52", "Managed money positioning", -0.84, 0.24),
        ("cotton_momentum_20", "20-day cotton momentum", 0.63, 0.19),
        ("dxy_ret_5", "Dollar index", 0.48, -0.18),
        ("cotton_volatility_20", "20-day volatility", 0.014, -0.09),
    ]
    target_sum = predicted_return_pct - 0.08
    scale = target_sum / sum(item[3] for item in raw)
    return [
        {
            "feature": feature,
            "display_name": display,
            "feature_value": value + 0.01 * math.sin(index),
            "contribution_pct": round(contribution * scale, 4),
        }
        for feature, display, value, contribution in raw
    ]

