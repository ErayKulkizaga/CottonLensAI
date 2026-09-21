import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


class ArtifactVerificationError(ValueError):
    pass


REQUIRED_FILES = {
    "manifest.json",
    "feature_schema.json",
    "metrics.json",
    "market_history.parquet",
    "feature_snapshots.parquet",
    "forecasts.parquet",
    "explanations.parquet",
    "checksums.sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    safe: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ArtifactVerificationError(f"unsafe ZIP path: {member.filename}")
        safe.append(member)
    return safe


def verify_directory(root: Path) -> dict:
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ArtifactVerificationError(f"artifact is missing required files: {', '.join(missing)}")
    expected: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip().lstrip("*")] = digest
    if not expected:
        raise ArtifactVerificationError("checksums.sha256 is empty")
    for relative, digest in expected.items():
        target = root / relative
        if not target.is_file():
            raise ArtifactVerificationError(f"checksummed file is missing: {relative}")
        if sha256_file(target) != digest:
            raise ArtifactVerificationError(f"checksum mismatch: {relative}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("artifact_version") or not manifest.get("production_models"):
        raise ArtifactVerificationError("manifest lacks artifact_version or production_models")
    return manifest


def install_bundle(bundle_path: Path, destination: Path) -> dict:
    """Verify fully before replacing the active runtime artifact directory."""
    if not bundle_path.is_file():
        raise ArtifactVerificationError(f"bundle not found: {bundle_path}")
    sidecar = bundle_path.with_suffix(f"{bundle_path.suffix}.sha256")
    if not sidecar.is_file():
        raise ArtifactVerificationError(f"bundle checksum sidecar not found: {sidecar.name}")
    expected_bundle_digest = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(bundle_path) != expected_bundle_digest:
        raise ArtifactVerificationError("bundle checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cottonlens-artifact-") as temp_dir:
        extracted = Path(temp_dir) / "bundle"
        extracted.mkdir()
        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(extracted, members=_safe_members(archive))
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        root = roots[0] if len(roots) == 1 and not (extracted / "manifest.json").exists() else extracted
        manifest = verify_directory(root)
        staged = destination.parent / f".{destination.name}.staged"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(root, staged)
        backup = destination.parent / f".{destination.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.replace(backup)
        staged.replace(destination)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
