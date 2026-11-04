"""Artifact roundtrip with untrained constant XGBoost payloads; never calls fit."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi import HTTPException
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from app import api
from app.api import _forecast_response
from app.artifact_importer import import_artifact_directory
from app.artifacts import ArtifactVerificationError, install_bundle, sha256_file, verify_directory
from app.database import Base
from app.models import ArtifactImport, ArtifactMarketObservation, Forecast, MarketObservation
from app.runtime import ArtifactRuntime
from app.schemas import SimulationRequest

REPO = Path(__file__).resolve().parents[2]


class ConstantModel:
    def __init__(self, value, names):
        self.booster = xgb.Booster(params={"num_feature": len(names), "base_score": value,
                                          "objective": "reg:squarederror"})
        self.booster.feature_names = names

    def predict(self, frame, pred_contribs=False):
        return self.booster.predict(xgb.DMatrix(frame, feature_names=self.booster.feature_names),
                                    pred_contribs=pred_contribs)

    def get_booster(self):
        return self.booster


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "ml/src"))
    exporter = importlib.import_module("cottonlens_ml.export")
    walkforward = importlib.import_module("cottonlens_ml.walkforward")
    Candidate = importlib.import_module("cottonlens_ml.training").Candidate
    dates = pd.bdate_range("2024-01-01", periods=80)
    frame = pd.DataFrame({name: np.full(80, 0.1) for name in exporter.FEATURE_NAMES})
    frame["date"] = dates
    frame["cotton_close"] = 80.0
    for horizon in (1, 5):
        frame[f"target_date_{horizon}"] = frame.date.shift(-horizon)
        frame[f"target_return_{horizon}"] = 0.01
        frame.loc[frame.index[-horizon:], f"target_return_{horizon}"] = np.nan
    test = frame.iloc[60:63].copy()
    monkeypatch.setattr(walkforward, "audit_split", lambda _: {
        "train": frame.iloc[:20], "validation": frame.iloc[25:45], "test": test,
    })
    metrics = {"mae": 1.0, "rmse": 1.1, "mape": 1.2, "directional_accuracy": 55.0}

    def candidate(horizon, role, value):
        cutoff = dates[60] if role == "evaluation_backtest" else dates[-1]
        return Candidate("XGBoost", horizon, ConstantModel(value, exporter.FEATURE_NAMES), np.full(len(test), value), metrics,
                         parameters={"model_role": role, "model_identity": f"{role}-{horizon}",
                                     "fit_cutoff": cutoff.isoformat(),
                                     "fit_origin_cutoff": (cutoff - pd.Timedelta(days=10)).isoformat(),
                                     "fit_label_cutoff": (cutoff - pd.Timedelta(days=1)).isoformat(),
                                     "recipe_identity": f"locked-recipe-{horizon}"})

    evaluated = {horizon: candidate(horizon, "evaluation_backtest", -0.01) for horizon in (1, 5)}
    deployed = {horizon: candidate(horizon, "deployment_live", 0.02) for horizon in (1, 5)}
    evidence = {key: {"id": key} for key in ("code_identity", "protocol_identity", "data_identity", "cohort_identity")}
    evidence.update({"data_quality": {"modeling_rows": 75}, "environment_smoke": {"status": "passed"}})
    root = tmp_path / "artifacts/releases"
    root.mkdir(parents=True)
    path = exporter.export_release(root, frame, test,
        pd.DataFrame({"date": dates, "series": "cotton", "close": 80.0}),
        list(evaluated.values()), evaluated, {},
        {"aggregate": {f"XGBoost-T+{h}": metrics for h in (1, 5)}},
        deployment_selected=deployed, deployment_candidates=list(deployed.values()), evidence_identity=evidence)
    return path, evaluated, deployed


def _rehash(root):
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
             for path in sorted(root.rglob("*")) if path.is_file() and path.name != "checksums.sha256"]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_export_import_runtime_preserves_evaluation_and_deployment(release, tmp_path):
    bundle, _, _ = release
    root = tmp_path / "current"
    manifest = install_bundle(bundle, root)
    assert manifest["artifact_schema_version"] == 2
    assert not (bundle.parent / "latest.txt").exists()
    rows = pd.read_parquet(root / "forecasts.parquet")
    assert set(rows.loc[rows.origin_type == "backtest", "predicted_return_pct"]) == {-1.0}
    assert rows.loc[rows.origin_type == "live", "predicted_return_pct"].to_numpy() == pytest.approx(2.0)
    assert rows.loc[rows.origin_type == "live", "target_date"].isna().all()
    assert rows.loc[rows.origin_type == "backtest", "target_date"].notna().all()
    assert rows.id.str.len().max() <= 36
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        assert import_artifact_directory(db, root) == manifest["artifact_version"]
        assert import_artifact_directory(db, root) == manifest["artifact_version"]
        assert db.scalar(select(func.count()).select_from(ArtifactImport)) == 1
        imported = db.scalars(select(Forecast)).all()
        assert len(imported) == 8
        for forecast in imported:
            response = _forecast_response(forecast)
            role = "deployment_live" if forecast.origin_type == "live" else "evaluation_backtest"
            assert response.model_role == role
            assert response.model_identity == f"{role}-{forecast.horizon}"
            assert response.data_quality == "historical_audit"
            assert (response.target_date is None) == (forecast.origin_type == "live")
    runtime = ArtifactRuntime(str(root))
    runtime.load()
    names = json.loads((root / "feature_schema.json").read_text())["features"]
    assert runtime.predict_return(1, dict.fromkeys(names, 0.1)) == pytest.approx(0.02)
    assert runtime.predict_return(5, dict.fromkeys(names, 0.1)) == pytest.approx(0.02)


def test_chart_switches_to_new_release_without_rewriting_previous_forecasts(release, tmp_path):
    original = tmp_path / "original"
    revised = tmp_path / "revised"
    install_bundle(release[0], original)
    manifest = install_bundle(release[0], revised)
    manifest["artifact_version"] = "revised-market-v2"
    (revised / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    market_path = revised / "market_history.parquet"
    market = pd.read_parquet(market_path)
    earliest = market.observed_on.min().date()
    market = market.iloc[1:].copy()
    market.loc[market.index[-1], "close"] = 84.0
    market.to_parquet(market_path, index=False)
    forecast_path = revised / "forecasts.parquet"
    forecasts = pd.read_parquet(forecast_path)
    forecasts["id"] = "revised-" + forecasts.index.astype(str)
    forecasts.loc[forecasts.origin_type == "live", "current_price"] = 84.0
    forecasts.loc[forecasts.origin_type == "live", "predicted_price"] = 84.0 * np.exp(.02)
    forecasts.to_parquet(forecast_path, index=False)
    _rehash(revised)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, original)
        first_forecast = api.latest_forecasts(db).forecasts[0]
        date = first_forecast.as_of_date
        assert api.market_history(date, date, ["cotton"], db).points[0].close == 80.0

        import_artifact_directory(db, revised)
        latest = api.latest_forecasts(db).forecasts[0]
        chart = api.market_history(date, date, ["cotton"], db)
        assert chart.points[0].close == latest.current_price_cents_per_lb == 84.0
        assert db.get(Forecast, first_forecast.id).current_price == 80.0
        rows = db.execute(
            select(ArtifactImport.artifact_version, ArtifactMarketObservation.close)
            .join(ArtifactMarketObservation)
            .where(ArtifactMarketObservation.observed_on == date, ArtifactMarketObservation.series == "cotton"),
        ).all()
        assert sorted(close for _, close in rows) == [80.0, 84.0]
        with pytest.raises(HTTPException) as missing_date:
            api.market_history(earliest, earliest, ["cotton"], db)
        assert missing_date.value.status_code == 404


def test_exact_existing_receipt_backfills_only_its_market_history(release, tmp_path):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        imported = db.scalar(select(ArtifactImport))
        original = db.scalar(select(Forecast).where(Forecast.origin_type == "live"))
        db.execute(delete(ArtifactMarketObservation))  # Shape of a pre-0003 database.
        db.add(MarketObservation(observed_on=original.as_of_date, series="cotton", close=76.0, source="colab-artifact"))
        db.commit()
        date = original.as_of_date
        legacy = api.market_history(date, date, ["cotton"], db)
        assert legacy.data_quality == "legacy"
        assert legacy.points[0].close == 76.0

        assert import_artifact_directory(db, root) == imported.artifact_version
        scoped = api.market_history(date, date, ["cotton"], db)
        assert scoped.data_quality == "historical_audit"
        assert scoped.points[0].close == original.current_price == 80.0
        assert db.scalar(select(MarketObservation.close)) == 76.0
        assert db.scalar(select(func.count()).select_from(ArtifactImport)) == 1
        assert db.scalar(select(func.count()).select_from(Forecast)) == 8


@pytest.mark.parametrize("corruption", ["duplicate", "wrong_live_close", "nonfinite"])
def test_import_rejects_invalid_market_payload_before_persisting_forecasts(release, tmp_path, corruption):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    market_path = root / "market_history.parquet"
    market = pd.read_parquet(market_path)
    if corruption == "duplicate":
        market = pd.concat([market, market.tail(1)], ignore_index=True)
        message = "duplicate release market observation"
    elif corruption == "nonfinite":
        market.loc[market.index[0], "close"] = np.inf
        message = "invalid release market observation"
    else:
        market.loc[market.index[-1], "close"] = 84.0
        message = "disagrees with the live forecast origin price"
    market.to_parquet(market_path, index=False)
    _rehash(root)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        with pytest.raises(ArtifactVerificationError, match=message):
            import_artifact_directory(db, root)
        for table in (ArtifactImport, Forecast, ArtifactMarketObservation):
            assert db.scalar(select(func.count()).select_from(table)) == 0


def test_chart_does_not_promote_claims_from_legacy_manifest(release, tmp_path):
    root = tmp_path / "current"
    manifest = install_bundle(release[0], root)
    manifest.update(artifact_schema_version=1, data_quality="validated_holdout")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _rehash(root)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        result = api.market_history(None, None, ["cotton"], db)
        assert result.data_quality == "legacy"
        assert len(result.points) == 80


def test_simulation_rejects_loaded_runtime_from_an_older_import(release, tmp_path, monkeypatch):
    root = tmp_path / "current"
    manifest = install_bundle(release[0], root)
    runtime = ArtifactRuntime(str(root))
    runtime.load(with_snapshots=True)
    monkeypatch.setattr(api, "runtime", runtime)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        live = db.scalar(select(Forecast).where(Forecast.origin_type == "live"))
        request = SimulationRequest(as_of_date=live.as_of_date, adjustments={})
        assert api.ready(db).artifact_version == manifest["artifact_version"]
        assert api.create_simulation(request, db).model_version == manifest["artifact_version"]

        manifest["artifact_version"] = "replacement-v2"
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        forecasts = pd.read_parquet(root / "forecasts.parquet")
        forecasts["id"] = "replacement-" + forecasts.index.astype(str)
        forecasts.to_parquet(root / "forecasts.parquet", index=False)
        snapshots = pd.read_parquet(root / "feature_snapshots.parquet")
        snapshots.tail(1).to_parquet(root / "feature_snapshots.parquet", index=False)
        _rehash(root)
        import_artifact_directory(db, root)

        with pytest.raises(HTTPException, match="active database artifact") as readiness:
            api.ready(db)
        assert readiness.value.status_code == 503
        with pytest.raises(HTTPException, match="active database artifact") as scenario:
            api.create_simulation(request, db)
        assert scenario.value.status_code == 503

        runtime.load(with_snapshots=True)
        assert api.ready(db).artifact_version == "replacement-v2"
        assert api.create_simulation(request, db).model_version == "replacement-v2"
        with monkeypatch.context() as sequence:
            sequence.setattr(runtime, "required_window", lambda _: 2)
            sequence.setattr(runtime, "predict_return", lambda *_: pytest.fail("incomplete release window was used"))
            with pytest.raises(HTTPException, match="2 feature snapshots") as incomplete:
                api.create_simulation(request, db)
            assert incomplete.value.status_code == 409

        monkeypatch.setattr(api, "runtime", ArtifactRuntime(str(tmp_path / "missing")))
        with pytest.raises(HTTPException) as missing_runtime:
            api.ready(db)
        assert missing_runtime.value.status_code == 503
        with pytest.raises(HTTPException) as missing_scenario:
            api.create_simulation(request, db)
        assert missing_scenario.value.status_code == 503


@pytest.mark.parametrize("surface", ["import", "readiness", "simulation"])
def test_same_manifest_cannot_reuse_receipt_after_payload_changes(release, tmp_path, monkeypatch, surface):
    root = tmp_path / "current"
    manifest = install_bundle(release[0], root)
    original_manifest_sha = sha256_file(root / "manifest.json")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        receipt = db.scalar(select(ArtifactImport))
        original_checksum = receipt.checksum
        live = db.scalar(select(Forecast).where(Forecast.origin_type == "live"))
        request = SimulationRequest(as_of_date=live.as_of_date, adjustments={})

        snapshots_path = root / "feature_snapshots.parquet"
        snapshots = pd.read_parquet(snapshots_path)
        snapshots.loc[snapshots.index[-1], "dxy_ret_1"] = 0.5
        snapshots.to_parquet(snapshots_path, index=False)
        _rehash(root)
        assert sha256_file(root / "manifest.json") == original_manifest_sha
        assert verify_directory(root) == manifest

        runtime = ArtifactRuntime(str(root))
        runtime.load(with_snapshots=True)
        monkeypatch.setattr(api, "runtime", runtime)
        if surface == "import":
            with pytest.raises(ArtifactVerificationError, match="different payload"):
                import_artifact_directory(db, root)
        else:
            with pytest.raises(HTTPException, match="active database artifact") as mismatch:
                if surface == "readiness":
                    api.ready(db)
                else:
                    api.create_simulation(request, db)
            assert mismatch.value.status_code == 503
        assert receipt.checksum == original_checksum


def test_manifest_only_receipt_cannot_be_promoted_from_current_files(release, tmp_path, monkeypatch):
    root = tmp_path / "current"
    manifest = install_bundle(release[0], root)
    legacy_checksum = sha256_file(root / "manifest.json")
    runtime = ArtifactRuntime(str(root))
    runtime.load(with_snapshots=True)
    monkeypatch.setattr(api, "runtime", runtime)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        receipt = db.scalar(select(ArtifactImport))
        receipt.checksum = legacy_checksum
        db.commit()
        with pytest.raises(ArtifactVerificationError, match="legacy manifest-only receipt"):
            import_artifact_directory(db, root, expected_payload_checksum=runtime.payload_checksum)
        with pytest.raises(HTTPException) as readiness:
            api.ready(db)
        assert readiness.value.status_code == 503
        assert receipt.checksum == legacy_checksum
        assert api.model_evaluation(db).artifact_version == manifest["artifact_version"]
        assert db.scalar(select(func.count()).select_from(Forecast)) == 8


def test_startup_rejects_install_swap_before_database_writes(release, tmp_path):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    runtime = ArtifactRuntime(str(root))
    runtime.load(with_snapshots=True)
    assert runtime.payload_checksum is not None
    metrics_path = root / "metrics.json"
    metrics_path.write_text(metrics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _rehash(root)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        with pytest.raises(ArtifactVerificationError, match="changed since the runtime was loaded"):
            import_artifact_directory(db, root, expected_payload_checksum=runtime.payload_checksum)
        assert db.scalar(select(func.count()).select_from(ArtifactImport)) == 0
        assert db.scalar(select(func.count()).select_from(Forecast)) == 0


def test_import_identity_is_independent_of_inventory_order(release, tmp_path):
    root = tmp_path / "current"
    manifest = install_bundle(release[0], root)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        inventory = root / "checksums.sha256"
        entries = inventory.read_text(encoding="utf-8").splitlines()
        inventory.write_text("\n".join(reversed(entries)) + "\n", encoding="utf-8")
        assert import_artifact_directory(db, root) == manifest["artifact_version"]
        assert db.scalar(select(func.count()).select_from(ArtifactImport)) == 1


def test_experimental_sensitivity_uses_its_own_zero_change_baseline(release, tmp_path, monkeypatch):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    runtime = ArtifactRuntime(str(root))
    runtime.load(with_snapshots=True)
    monkeypatch.setattr(api, "runtime", runtime)
    monkeypatch.setattr(runtime, "model_name", lambda _: "XGBoost experimental sensitivity model")
    monkeypatch.setattr(runtime, "predict_return", lambda _, features: float(features["dxy_ret_1"]))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        import_artifact_directory(db, root)
        live = db.scalar(select(Forecast).where(Forecast.origin_type == "live"))
        request = SimulationRequest(as_of_date=live.as_of_date, adjustments={})
        for result in api.create_simulation(request, db).results:
            assert result.baseline_kind == "experimental"
            assert result.delta_cents_per_lb == 0
            assert result.delta_pct == 0


def test_unchecked_payload_and_model_identity_mismatch_are_rejected(release, tmp_path):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    (root / "extra.json").write_text("unchecked", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="inventory"):
        verify_directory(root)
    (root / "extra.json").unlink()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["evaluation_models"][0]["model_role"] = "deployment_live"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _rehash(root)
    with pytest.raises(ArtifactVerificationError, match="evaluation_backtest"):
        verify_directory(root)


def test_import_rejects_fabricated_future_date_before_database_mutation(release, tmp_path):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    frame = pd.read_parquet(root / "forecasts.parquet")
    frame.loc[frame.origin_type == "live", "target_date"] = pd.Timestamp("2027-01-01")
    frame.to_parquet(root / "forecasts.parquet", index=False)
    _rehash(root)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        with pytest.raises(ArtifactVerificationError, match="must be unknown"):
            import_artifact_directory(db, root)
        assert db.scalar(select(func.count()).select_from(Forecast)) == 0


def test_import_rejects_incomplete_feature_snapshot_before_database_mutation(release, tmp_path):
    root = tmp_path / "current"
    install_bundle(release[0], root)
    frame = pd.read_parquet(root / "feature_snapshots.parquet")
    frame.loc[0, frame.columns[1]] = np.nan
    frame.to_parquet(root / "feature_snapshots.parquet", index=False)
    _rehash(root)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        with pytest.raises(ArtifactVerificationError, match="complete finite schema"):
            import_artifact_directory(db, root)
        assert db.scalar(select(func.count()).select_from(ArtifactImport)) == 0


def test_release_pointer_changes_only_after_successful_validation(release, tmp_path, monkeypatch):
    validator = importlib.import_module("cottonlens_ml.validate_release")
    monkeypatch.setattr(validator, "source_identity", lambda _: {"id": "code_identity"})
    bundle = release[0]
    pointer = bundle.parent / "latest.txt"
    pointer.write_text("previous.zip", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["validate_release", "--repo", str(REPO), "--drive-root", str(tmp_path),
                                    "--release", str(bundle), "--inference-python", sys.executable])
    def fail(*args, **kwargs):
        raise RuntimeError("parity failed")
    monkeypatch.setattr(validator.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="parity failed"):
        validator.main()
    assert pointer.read_text() == "previous.zip"
    assert not bundle.with_suffix(".validation.json").exists()
    def probe(command, **kwargs):
        cases = json.loads(Path(command[command.index("--cases") + 1]).read_text())
        assert [case["expected"] for case in cases.values()] == pytest.approx([0.02, 0.02])
    monkeypatch.setattr(validator.subprocess, "run", probe)
    validator.main()
    assert pointer.read_text() == bundle.name
    receipt = json.loads(bundle.with_suffix(".validation.json").read_text())
    assert receipt["zip_sha256"] == sha256_file(bundle)


def test_validator_rejects_changed_source_before_publishing(release, tmp_path, monkeypatch):
    validator = importlib.import_module("cottonlens_ml.validate_release")
    bundle = release[0]
    monkeypatch.setattr(validator, "source_identity", lambda _: {"id": "different-code"})
    monkeypatch.setattr(sys, "argv", ["validate_release", "--repo", str(REPO), "--drive-root", str(tmp_path),
                                    "--release", str(bundle), "--inference-python", sys.executable])
    with pytest.raises(ValueError, match="source identity differs"):
        validator.main()
    assert not (bundle.parent / "latest.txt").exists()


def test_schema3_adapter_roundtrip_and_reproduction_gate(release, tmp_path):
    root = tmp_path / 'research-runtime'
    manifest = install_bundle(release[0], root)
    manifest['artifact_schema_version'] = 3
    manifest['research_protocol'] = 'cotton-research-v1'
    manifest['reproduction'] = {str(h): {'status': 'passed', 'fresh_fits': True,
        'max_abs_log_return_difference': 0.} for h in (1, 5)}
    names = json.loads((root / 'feature_schema.json').read_text())['features']
    adapter = {'processor': {'names': names, 'median': [0.] * len(names),
        'mean': [0.] * len(names), 'scale': [1.] * len(names)},
        'target': {'kind': 'raw_log', 'mean': 0., 'scale': 1.}, 'window': 1}
    (root / 'adapter.json').write_text(json.dumps(adapter))
    (root / 'linear.json').write_text(json.dumps({'coef': [0.] * (2 * len(names)), 'intercept': .02}))
    for entry in manifest['production_models']:
        entry.update(format='research_adapter_v1', members=[{'path': 'linear.json',
            'adapter': 'adapter.json', 'format': 'linear_json', 'weight': 1.}])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    frame = pd.read_parquet(root / 'feature_snapshots.parquet')
    frame.loc[0, names[0]] = np.nan
    frame.to_parquet(root / 'feature_snapshots.parquet', index=False)
    _rehash(root)
    verify_directory(root)
    runtime = ArtifactRuntime(str(root))
    runtime.load()
    assert runtime.predict_return(1, dict.fromkeys(names, None)) == pytest.approx(.02)
    database = create_engine('sqlite://')
    Base.metadata.create_all(database)
    with Session(database) as db:
        import_artifact_directory(db, root)
        live = db.scalar(select(Forecast).where(Forecast.origin_type == 'live'))
        assert _forecast_response(live).predicted_return_pct == pytest.approx(2.)
    manifest['reproduction']['1']['fresh_fits'] = False
    (root / 'manifest.json').write_text(json.dumps(manifest))
    _rehash(root)
    with pytest.raises(ArtifactVerificationError, match='fresh reproduction'):
        verify_directory(root)
