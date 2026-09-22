import os
import tempfile
from datetime import date
from pathlib import Path

os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.gettempdir()) / 'cottonlens-api-tests.db'}"
os.environ["DEMO_MODE"] = "true"
os.environ["ARTIFACT_DIR"] = str(Path(tempfile.gettempdir()) / "missing-cottonlens-artifact")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import api
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import ArtifactImport, FeatureSnapshot, Forecast, ModelVersion


@pytest.fixture(scope="module", autouse=True)
def database() -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_distinguishes_live_from_ready(client: TestClient) -> None:
    assert client.get("/api/v1/health/live").json()["status"] == "ok"
    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert "development fixture" in ready.json()["detail"]


def test_ready_rejects_runtime_without_matching_database_import(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api.runtime, "manifest", {"artifact_version": "unimported-v1"})
    monkeypatch.setattr(api.runtime, "models", {1: ("xgboost_json", object())})
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 503
    assert "database import has not completed" in response.json()["detail"]


def test_latest_is_explicitly_illustrative(client: TestClient) -> None:
    response = client.get("/api/v1/forecasts/latest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_quality"] == "illustrative"
    assert {item["horizon"] for item in payload["forecasts"]} == {1, 5}


def test_explanation_is_additive(client: TestClient) -> None:
    forecast = client.get("/api/v1/forecasts/latest").json()["forecasts"][0]
    explanation = client.get(f"/api/v1/forecasts/{forecast['id']}/explanation").json()
    reconstructed = explanation["base_value_pct"] + sum(
        item["contribution_pct"] for item in explanation["contributions"]
    )
    assert reconstructed == pytest.approx(explanation["predicted_return_pct"], abs=1e-3)


def test_simulation_validates_bounds_and_preserves_snapshot(client: TestClient) -> None:
    latest = client.get("/api/v1/forecasts/latest").json()
    as_of = latest["data_as_of"]
    with SessionLocal() as db:
        before = dict(db.scalar(select(FeatureSnapshot).where(FeatureSnapshot.as_of_date == as_of)).values)
    invalid = client.post(
        "/api/v1/simulations",
        json={"as_of_date": as_of, "adjustments": {"dxy_pct_change": 6}},
    )
    assert invalid.status_code == 422
    valid = client.post(
        "/api/v1/simulations",
        json={
            "as_of_date": as_of,
            "adjustments": {
                "dxy_pct_change": -2,
                "wti_pct_change": 10,
                "cftc_net_delta_contracts": 15000,
                "volatility_multiplier": 1.2,
            },
        },
    )
    assert valid.status_code == 201
    assert len(valid.json()["results"]) == 2
    with SessionLocal() as db:
        after = db.scalar(select(FeatureSnapshot).where(FeatureSnapshot.as_of_date == as_of)).values
    assert after == before


def test_experimental_sensitivity_uses_its_own_zero_change_baseline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest = client.get("/api/v1/forecasts/latest").json()
    monkeypatch.setattr(api.runtime, "manifest", {
        "production_models": [
            {"horizon": horizon, "name": "XGBoost experimental sensitivity model"}
            for horizon in (1, 5)
        ],
    })
    monkeypatch.setattr(api.runtime, "models", {horizon: ("xgboost_json", object()) for horizon in (1, 5)})
    monkeypatch.setattr(
        api.runtime, "predict_return",
        lambda horizon, features: float(features["dxy_ret_1"]),
    )
    response = client.post(
        "/api/v1/simulations",
        json={"as_of_date": latest["data_as_of"], "adjustments": {}},
    )
    assert response.status_code == 201
    for result in response.json()["results"]:
        assert result["baseline_kind"] == "experimental"
        assert result["delta_cents_per_lb"] == 0
        assert result["delta_pct"] == 0


def test_replay_only_returns_backtest_rows(client: TestClient) -> None:
    history = client.get("/api/v1/forecasts/history?horizon=1&origin_type=backtest").json()
    response = client.get(f"/api/v1/replay/{history[0]['as_of_date']}")
    assert response.status_code == 200
    assert all(item["origin_type"] == "backtest" for item in response.json()["forecasts"])


def test_current_artifact_endpoints_do_not_mix_older_forecasts(client: TestClient) -> None:
    initial = client.get("/api/v1/forecasts/latest").json()
    as_of = initial["data_as_of"]
    with SessionLocal() as db:
        imported = ArtifactImport(
            artifact_version="test-artifact-v2", manifest={"artifact_version": "test-artifact-v2"},
            checksum="test",
        )
        db.add(imported)
        created_models = []
        created_forecasts = []
        for original in initial["forecasts"]:
            model = ModelVersion(
                artifact_version="test-artifact-v2", horizon=original["horizon"],
                model_name="Naive", model_format="evaluation_only", selected=True,
                data_quality="historical_audit", metrics={"mae": 1.0},
            )
            db.add(model)
            db.flush()
            forecast = Forecast(
                as_of_date=date.fromisoformat(as_of), target_date=date.fromisoformat(original["target_date"]),
                horizon=original["horizon"], current_price=original["current_price_cents_per_lb"],
                predicted_price=original["current_price_cents_per_lb"], predicted_return_pct=0,
                actual_price=None, origin_type="live", model_version_id=model.id,
            )
            db.add(forecast)
            created_models.append(model)
            created_forecasts.append(forecast)
        db.commit()
        try:
            latest = client.get("/api/v1/forecasts/latest").json()
            assert len(latest["forecasts"]) == 2
            assert all(item["model_version"] == "test-artifact-v2" for item in latest["forecasts"])
            history = client.get("/api/v1/forecasts/history?horizon=1&origin_type=live").json()
            assert all(item["model_version"] == "test-artifact-v2" for item in history)
        finally:
            for forecast in created_forecasts:
                db.delete(forecast)
            db.flush()
            for model in created_models:
                db.delete(model)
            db.delete(imported)
            db.commit()
