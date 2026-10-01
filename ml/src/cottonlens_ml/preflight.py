"""Freeze data and origin identities and run blocking no-fit protocol checks."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, source_identity
from cottonlens_ml.cohort import content_id, freeze_cohort, verify_cohort
from cottonlens_ml.config import EXTERNAL_MAX_AGE_SESSIONS, FEATURE_NAMES, PipelinePaths
from cottonlens_ml.features import require_latest_live_row, select_feature_rows
from cottonlens_ml.protocol import PROTOCOL_VERSION, refit_rows, validate_partition
from cottonlens_ml.sequences import sequence_diagnostics
from cottonlens_ml.walkforward import audit_split, build_folds


def experiment_directory(paths: PipelinePaths, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Experiment name must be a plain identifier, without directory components")
    return paths.root / "experiments" / name


def _snapshot(path: Path, expected_id: str, expected_files: dict) -> tuple[dict, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("complete") is not True or manifest.get("snapshot_id") != expected_id
            or content_id(manifest["identity"]) != expected_id or manifest["files"] != expected_files):
        raise ValueError("Frozen data snapshot identity mismatch")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or digest(path.parent / name) != checksum:
            raise ValueError("Frozen data snapshot file checksum mismatch")
    return manifest, path.parent


def load_frozen_data(identity: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed, folder = _snapshot(Path(identity["manifest_path"]), identity["data_id"], identity["checksums"])
    _, raw_folder = _snapshot(Path(identity["raw_manifest_path"]), identity["raw_id"], identity["source_checksums"])
    if processed["identity"]["metadata"]["raw_id"] != identity["raw_id"]:
        raise ValueError("Processed and raw snapshots are unrelated")
    history = pd.read_parquet(folder / "feature_history.parquet")
    history.attrs["data_identity"] = identity
    history.attrs["data_quality"] = processed["identity"]["metadata"]["data_quality"]
    market = pd.read_parquet(raw_folder / "market.parquet")
    return market.loc[market.date <= history.date.max()].copy(), history


def verify_target_contract(history: pd.DataFrame) -> None:
    """Check recorded Cotton observation horizons and conservative feature clock."""
    if history.empty or history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError("Cotton observation dates must be unique, present and ascending")
    prices = history.cotton_close.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Cotton prices must be positive and finite")
    if not np.array_equal(history.cotton_session_index.to_numpy(), np.arange(len(history))):
        raise ValueError("Cotton session ordinals must include every recorded source observation")
    date = history.date.reset_index(drop=True)
    close = history.cotton_close.reset_index(drop=True)
    for horizon in (1, 5):
        if not history[f"target_date_{horizon}"].reset_index(drop=True).equals(date.shift(-horizon)):
            raise ValueError(f"T+{horizon} target dates differ from recorded Cotton observations")
        expected = np.log(close.shift(-horizon) / close).to_numpy(dtype=float)
        actual = history[f"target_return_{horizon}"].to_numpy(dtype=float)
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"T+{horizon} target returns differ from recorded Cotton prices")
    expected_decision = pd.to_datetime(history.date, utc=True) + pd.Timedelta(days=1)
    if not pd.Series(pd.to_datetime(history.decision_time, utc=True)).reset_index(drop=True).equals(expected_decision.reset_index(drop=True)):
        raise ValueError("Decision timestamps must follow the recorded Cotton source date")
    previous = history.date.shift(1)
    for series in ("dxy", "wti"):
        source = history[f"{series}_source_date"]
        available = pd.to_datetime(history[f"{series}_available_at"], utc=True)
        age = history[f"{series}_age_sessions"]
        present = source.notna()
        if (source[present] > previous[present]).any() or age[present].isna().any():
            raise ValueError(f"{series} uses a same-origin/future source or lacks age")
        if not available.equals(pd.to_datetime(source, utc=True) + pd.Timedelta(days=1)):
            raise ValueError(f"{series} availability timestamp differs from the source policy")
        if (available[present] > expected_decision[present]).any():
            raise ValueError(f"{series} feature was unavailable at the decision clock")
        stale = age.gt(EXTERNAL_MAX_AGE_SESSIONS)
        if not history[f"{series}_stale"].equals(stale):
            raise ValueError(f"{series} staleness flag is inconsistent with the age policy")
        if history.loc[stale, f"{series}_ret_1"].notna().any():
            raise ValueError(f"{series} stale values entered model features")


def protocol_checks(history: pd.DataFrame, cohort: dict) -> dict:
    verify_target_contract(history)
    verify_cohort(cohort, history, cohort["data_id"])
    modeling = select_feature_rows(history, FEATURE_NAMES)
    folds = build_folds(modeling, cohort=cohort)
    diagnostics = []
    for fold in folds:
        for name in ("train", "validation", "test"):
            validate_partition(getattr(fold, name), name)
        refit = refit_rows(fold.train, fold.validation, fold.test, history)
        sequence = sequence_diagnostics(fold.test, history)
        if len(fold.validation) != 126 or sequence["sample_count"] != 126:
            raise ValueError("Every frozen fold requires 126 validation and 126 evaluable sequence origins")
        diagnostics.append({"fold": fold.number, "refit_rows": len(refit), "sequence": sequence})
    splits = audit_split(modeling)
    audit = sequence_diagnostics(splits["test"], history)
    if audit["sample_count"] != len(splits["test"]):
        raise ValueError("Audit sequence coverage incomplete")
    require_latest_live_row(history)
    return {"folds": diagnostics, "seen_historical_audit": audit,
            "evaluation_count_per_horizon": sum(len(fold.test) for fold in folds),
            "fit_policy": "one locked refit per 126-origin development fold; static locked audit model",
            "deployment_policy": "fresh current refit with locked recipe and available labels only"}


def freeze_readiness(repo: Path, paths: PipelinePaths, experiment: str, smoke_path: Path) -> dict:
    code = source_identity(repo)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (smoke.get("status") != "passed" or smoke.get("source_id") != code["source_id"]
            or smoke["environment"].get("lock_sha256") != digest(repo / "ml/uv.lock")):
        raise ValueError("Successful Colab environment smoke must identify this exact source and dependency lock")
    identity = json.loads((paths.processed / "data_identity.json").read_text(encoding="utf-8"))
    _, history = load_frozen_data(identity)
    verify_target_contract(history)
    directory = experiment_directory(paths, experiment)
    cohort = freeze_cohort(directory / "cohort.json", history, identity["data_id"])
    checks = protocol_checks(history, cohort)
    payload = {
        "status": "ready", "experiment": experiment, "code_identity": code,
        "protocol_identity": {"version": PROTOCOL_VERSION, "evaluation_refit_cadence": 126,
                              "cadence_21_status": "unvalidated_hypothesis_not_enabled",
                              "audit_role": "seen_historical_audit_descriptive_only_no_selection_or_veto"},
        "data_identity": identity, "cohort_identity": cohort,
        "data_quality": history.attrs["data_quality"], "environment_smoke": smoke,
        "smoke_path": str(smoke_path), "smoke_sha256": digest(smoke_path), "checks": checks,
    }
    ready = {**payload, "readiness_id": content_id(payload)}
    target = directory / "ready.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != ready:
            raise ValueError("Existing experiment identity changed; retain evidence and use a new experiment identifier")
    else:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(ready, handle, indent=2, allow_nan=False)
    return ready


def verify_readiness(repo: Path, paths: PipelinePaths, experiment: str):
    directory = experiment_directory(paths, experiment)
    ready = json.loads((directory / "ready.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in ready.items() if key != "readiness_id"}
    if (ready.get("status") != "ready" or ready.get("readiness_id") != content_id(payload)
            or ready["code_identity"] != source_identity(repo)
            or ready["protocol_identity"]["version"] != PROTOCOL_VERSION
            or ready["smoke_sha256"] != digest(Path(ready["smoke_path"]))):
        raise ValueError("Pre-training identity changed; training is blocked")
    cohort = json.loads((directory / "cohort.json").read_text(encoding="utf-8"))
    if cohort != ready["cohort_identity"]:
        raise ValueError("Frozen cohort was changed after preflight")
    market, history = load_frozen_data(ready["data_identity"])
    protocol_checks(history, cohort)
    return ready, market, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    ready = freeze_readiness(args.repo, PipelinePaths(args.drive_root), args.experiment, args.smoke_report)
    print(json.dumps({"status": ready["status"], "readiness_id": ready["readiness_id"],
                      "cohort_id": ready["cohort_identity"]["cohort_id"], "checks": ready["checks"]}, indent=2))


if __name__ == "__main__":
    main()
