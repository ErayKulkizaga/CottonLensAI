import os
import tempfile
from pathlib import Path

os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.gettempdir()) / 'cottonlens-api-tests.db'}"
os.environ["DEMO_MODE"] = "true"
os.environ["ARTIFACT_DIR"] = str(Path(tempfile.gettempdir()) / "missing-cottonlens-artifact")

import pytest
from app import api
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import FeatureSnapshot
from fastapi.testclient import TestClient
from sqlalchemy import select


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


def test_replay_only_returns_backtest_rows(client: TestClient) -> None:
    history = client.get("/api/v1/forecasts/history?horizon=1&origin_type=backtest").json()
    response = client.get(f"/api/v1/replay/{history[0]['as_of_date']}")
    assert response.status_code == 200
    assert all(item["origin_type"] == "backtest" for item in response.json()["forecasts"])
