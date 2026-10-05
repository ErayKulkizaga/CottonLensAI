"""Artifact roundtrip with untrained constant XGBoost payloads; never calls fit."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.api import _forecast_response
from app.artifact_importer import import_artifact_directory
from app.artifacts import ArtifactVerificationError, install_bundle, sha256_file, verify_directory
from app.database import Base
from app.models import ArtifactImport, Forecast
from app.runtime import ArtifactRuntime

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
