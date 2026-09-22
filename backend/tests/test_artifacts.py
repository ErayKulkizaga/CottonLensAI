import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from app.artifacts import REQUIRED_FILES, ArtifactVerificationError, install_bundle


def _build_bundle(root: Path, corrupt: bool = False) -> Path:
    source = root / "bundle-source" / "cottonlens-model-vtest"
    source.mkdir(parents=True)
    manifest = {
        "artifact_version": "vtest",
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
