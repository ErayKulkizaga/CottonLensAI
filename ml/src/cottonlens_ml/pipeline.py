"""Colab orchestration over an immutable, preflight-approved experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cottonlens_ml.config import FEATURE_NAMES, PipelinePaths
from cottonlens_ml.features import select_feature_rows
from cottonlens_ml.preflight import experiment_directory, verify_readiness
from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.selection import select_walkforward_name


def run(drive_root: Path, *, repo: Path, experiment: str) -> Path:
    require_colab_training()  # Before heavy imports, data work or fitting.
    paths = PipelinePaths(drive_root)
    ready, market, features = verify_readiness(repo, paths, experiment)
    import mlflow

    from cottonlens_ml.export import export_release
    from cottonlens_ml.tracking import tracked_run, tracking_session
    from cottonlens_ml.training import refit_deployment, refit_locked_evaluation
    from cottonlens_ml.walkforward import audit_split, run_walkforward

    directory = experiment_directory(paths, experiment)
    checkpoint_root = paths.checkpoints / ready["readiness_id"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    modeling = select_feature_rows(features, FEATURE_NAMES)
    splits = audit_split(modeling)
    with tracking_session(paths.root), tracked_run(run_name=f"corrected-v2-{experiment}"):
        mlflow.log_params({"readiness_id": ready["readiness_id"],
                           "source_id": ready["code_identity"]["source_id"],
                           "data_id": ready["data_identity"]["data_id"],
                           "cohort_id": ready["cohort_identity"]["cohort_id"]})
        mlflow.log_artifact(str(directory / "ready.json"))
        report, locked_recipes = run_walkforward(
            modeling, features, checkpoint_root, cohort=ready["cohort_identity"],
        )
        # Freeze both decisions BEFORE any seen historical audit metric.
        decisions = {h: select_walkforward_name(report, h, {}) for h in (1, 5)}
        decision_path = directory / "selection-before-audit.json"
        frozen = {str(h): audit for h, (_, audit) in decisions.items()}
        if decision_path.exists():
            if json.loads(decision_path.read_text(encoding="utf-8")) != frozen:
                raise ValueError("Resumed selection differs from the frozen pre-audit decision")
        else:
            with decision_path.open("x", encoding="utf-8") as handle:
                json.dump(frozen, handle, indent=2, allow_nan=False)
        # The last development fold supplies recipes; audit starts no new search.
        audit_families = {
            name: refit_locked_evaluation(values, features, splits["test"], checkpoint_root / "audit")
            for name, values in locked_recipes.items()
        }
        selected = {h: audit_families[decisions[h][0]][h] for h in (1, 5)}
        selection_audit = {
            h: {**decisions[h][1], "historical_audit": {name: values[h].metrics for name, values in audit_families.items()}}
            for h in (1, 5)
        }
        candidates = [candidate for family in audit_families.values() for candidate in family.values()]
        for candidate in candidates:
            mlflow.log_metrics({f"{candidate.name.lower()}_t{candidate.horizon}_{key}": value for key, value in candidate.metrics.items()})
        # Current weights are distinct from historical audit weights.
        cutoff = features.date.max()
        deployment_selected = refit_deployment(selected, features, cutoff, checkpoint_root / "deployment")
        deployment_trees = refit_deployment(audit_families["XGBoost"], features, cutoff, checkpoint_root / "deployment")
        deployment_candidates = list({(candidate.name, candidate.horizon): candidate for candidate in [
            *deployment_selected.values(), *deployment_trees.values(),
        ]}.values())
        output = export_release(
            paths.releases, features, splits["test"], market, candidates, selected, selection_audit, report,
            deployment_selected=deployment_selected, deployment_candidates=deployment_candidates,
            evidence_identity={key: ready[key] for key in (
                "code_identity", "protocol_identity", "data_identity", "cohort_identity", "data_quality", "environment_smoke",
            )},
        )
        # Validator alone publishes latest.txt after parity and checksums succeed.
        (directory / "pending-release.txt").write_text(output.name, encoding="utf-8")
        mlflow.log_artifact(str(decision_path))
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen corrected-V2 experiment in Google Colab")
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    output = run(args.drive_root, repo=args.repo, experiment=args.experiment)
    print(f"Exported bundle (run validate_release before use): {output}")


if __name__ == "__main__":
    main()
