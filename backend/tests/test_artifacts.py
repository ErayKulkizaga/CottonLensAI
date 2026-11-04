import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from app.artifacts import (
    REQUIRED_FILES,
    ArtifactVerificationError,
    install_bundle,
    verify_directory_with_identity,
)


def _build_bundle(root: Path, corrupt: bool = False, version: str = "vtest") -> Path:
    source = root / "bundle-source" / "cottonlens-model-vtest"
    source.mkdir(parents=True)
    manifest = {
        "artifact_version": version,
        "feature_schema_hash": "abc",
        "production_models": [
            {"horizon": 1, "name": "XGBoost", "format": "xgboost_json", "path": "model/t1.json"}
        ],
    }
    for name in REQUIRED_FILES - {"checksums.sha256"}:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest) if name == "manifest.json" else "fixture", encoding="utf-8")
    model = source / "model" / "t1.json"
    model.parent.mkdir()
    model.write_text("model", encoding="utf-8")
    lines = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if corrupt and path.name == "metrics.json":
            digest = "0" * 64
        lines.append(f"{digest}  {path.relative_to(source).as_posix()}")
    (source / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = root / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file():
                handle.write(path, arcname=f"cottonlens-model-vtest/{path.relative_to(source).as_posix()}")
    archive.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="utf-8",
    )
    return archive


def test_install_bundle_verifies_and_installs(tmp_path: Path) -> None:
    destination = tmp_path / "runtime" / "current"
    manifest = install_bundle(_build_bundle(tmp_path), destination)
    assert manifest["artifact_version"] == "vtest"
    assert (destination / "manifest.json").exists()


def test_install_bundle_rejects_bad_checksum(tmp_path: Path) -> None:
    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        install_bundle(_build_bundle(tmp_path, corrupt=True), tmp_path / "runtime" / "current")


def test_install_bundle_rejects_bad_outer_checksum(tmp_path: Path) -> None:
    archive = _build_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ArtifactVerificationError, match="bundle checksum mismatch"):
        install_bundle(archive, tmp_path / "runtime" / "current")


def test_interrupted_install_keeps_previous_release_when_retry_fails(tmp_path, monkeypatch):
    destination = tmp_path / "runtime" / "current"
    backup = destination.with_name(".current.previous")
    install_bundle(_build_bundle(tmp_path / "original"), destination)
    previous = verify_directory_with_identity(destination)
    destination.replace(backup)  # State left by termination between the two renames.
    bundle = _build_bundle(tmp_path / "candidate", version="vnew")
    replace = Path.replace

    def fail_activation(source, target):
        if source.name == ".current.staged" and Path(target) == destination:
            raise PermissionError("simulated activation failure")
        return replace(source, target)

    with monkeypatch.context() as activation:
        activation.setattr(Path, "replace", fail_activation)
        with pytest.raises(PermissionError, match="simulated activation failure"):
            install_bundle(bundle, destination)
    assert verify_directory_with_identity(destination) == previous
    assert install_bundle(bundle, destination)["artifact_version"] == "vnew"


def test_competing_installer_cannot_publish_into_the_same_destination(tmp_path, monkeypatch):
    destination = tmp_path / "runtime" / "current"
    primary = _build_bundle(tmp_path / "primary", version="vprimary")
    competing = _build_bundle(tmp_path / "competing", version="vcompeting")
    replace = Path.replace
    observed = []

    def competing_activation(source, target):
        if source.name == ".current.staged" and Path(target) == destination:
            child = subprocess.run(
                [sys.executable, "-m", "scripts.import_artifact", str(competing), "--destination", str(destination)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15, check=False,
            )
            observed.append((child.returncode, child.stderr))
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", competing_activation)
    assert install_bundle(primary, destination)["artifact_version"] == "vprimary"
    assert len(observed) == 1
    assert observed[0][0] != 0
    assert "another artifact installation" in observed[0][1]
    assert verify_directory_with_identity(destination)[0]["artifact_version"] == "vprimary"


def test_invalid_candidate_does_not_prevent_recovery_of_previous_release(tmp_path):
    destination = tmp_path / "runtime" / "current"
    backup = destination.with_name(".current.previous")
    install_bundle(_build_bundle(tmp_path / "original"), destination)
    previous = verify_directory_with_identity(destination)
    destination.replace(backup)
    candidate = _build_bundle(tmp_path / "candidate", corrupt=True, version="vnew")
    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        install_bundle(candidate, destination)
    assert verify_directory_with_identity(destination) == previous


def test_installer_verifies_staged_copy_before_replacing_active_release(tmp_path, monkeypatch):
    destination = tmp_path / "runtime" / "current"
    install_bundle(_build_bundle(tmp_path / "original"), destination)
    previous = verify_directory_with_identity(destination)
    candidate = _build_bundle(tmp_path / "candidate", version="vnew")
    copytree = shutil.copytree

    def corrupt_copy(source, target, *args, **kwargs):
        result = copytree(source, target, *args, **kwargs)
        if Path(target).name == ".current.staged":
            (Path(target) / "metrics.json").write_text("damaged copy", encoding="utf-8")
        return result

    monkeypatch.setattr(shutil, "copytree", corrupt_copy)
    with pytest.raises(ArtifactVerificationError, match="checksum mismatch: metrics.json"):
        install_bundle(candidate, destination)
    assert verify_directory_with_identity(destination) == previous
