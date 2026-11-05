import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath


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
    names: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or ":" in member.filename or member.filename in names
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ArtifactVerificationError(f"unsafe ZIP path: {member.filename}")
        names.add(member.filename)
        safe.append(member)
    return safe


def _payload_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts or "\\" in relative
            or ":" in relative or path.as_posix() != relative):
        raise ArtifactVerificationError(f"unsafe artifact path: {relative}")
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ArtifactVerificationError(f"unsafe artifact path: {relative}")
    return target


def _checksum_inventory(root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[a-fA-F0-9]{64}", parts[0]):
            raise ArtifactVerificationError("invalid checksum entry")
        digest, relative = parts
        relative = relative.strip().lstrip("*")
        if relative in expected or relative == "checksums.sha256":
            raise ArtifactVerificationError(f"duplicate or invalid checksum path: {relative}")
        expected[relative] = digest.lower()
    if not expected:
        raise ArtifactVerificationError("checksums.sha256 is empty")
    return expected


def _inventory_identity(expected: dict[str, str]) -> str:
    inventory = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"cottonlens-artifact-payload-v1\0" + inventory).hexdigest()


def payload_identity(root: Path) -> str:
    """Read the declared identity, after the caller has verified the directory."""
    return _inventory_identity(_checksum_inventory(root))


def verify_directory(root: Path) -> dict:
    manifest, _ = verify_directory_with_identity(root)
    return manifest


def verify_directory_with_identity(root: Path) -> tuple[dict, str]:
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ArtifactVerificationError(f"artifact is missing required files: {', '.join(missing)}")
    expected = _checksum_inventory(root)
    for relative, digest in expected.items():
        target = _payload_path(root, relative)
        if not target.is_file():
            raise ArtifactVerificationError(f"checksummed file is missing: {relative}")
        if sha256_file(target) != digest:
            raise ArtifactVerificationError(f"checksum mismatch: {relative}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("artifact_version") or not manifest.get("production_models"):
        raise ArtifactVerificationError("manifest lacks artifact_version or production_models")
    payloads = {file.relative_to(root).as_posix() for file in root.rglob("*")
                if file.is_file() and file.name != "checksums.sha256"}
    if payloads != set(expected):
        raise ArtifactVerificationError("checksum inventory does not cover every artifact payload")
    for entry in manifest["production_models"]:
        if entry.get("path") not in expected or not _payload_path(root, entry["path"]).is_file():
            raise ArtifactVerificationError("production model payload is absent or unchecked")
    version = manifest.get("artifact_schema_version", 1)
    if version not in (1, 2, 3):
        raise ArtifactVerificationError("unsupported artifact schema version")
    if version in (2, 3):
        schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
        if (schema.get("schema_hash") != manifest.get("feature_schema_hash") or not schema.get("features")
                or len(set(schema["features"])) != len(schema["features"])):
            raise ArtifactVerificationError("invalid feature schema identity")
        required = ("code_identity", "protocol_identity", "data_identity", "cohort_identity")
        if any(not isinstance(manifest.get(key), dict) or not manifest[key] for key in required):
            raise ArtifactVerificationError("schema v2 requires immutable evidence identities")
        if manifest.get("data_quality") != "historical_audit":
            raise ArtifactVerificationError("schema v2 evidence must be historical_audit")
        evidence_path = manifest.get("evidence_path")
        if evidence_path not in expected:
            raise ArtifactVerificationError("frozen release evidence is missing")
        evidence = json.loads(_payload_path(root, evidence_path).read_text(encoding="utf-8"))
        if any(evidence.get(key) != manifest[key] for key in required):
            raise ArtifactVerificationError("release evidence identity mismatch")
        if not evidence.get("data_quality") or evidence.get("environment_smoke", {}).get("status") != "passed":
            raise ArtifactVerificationError("release requires frozen quality and passed environment evidence")
        for key, role in (("evaluation_models", "evaluation_backtest"), ("production_models", "deployment_live")):
            models = manifest.get(key, [])
            if len(models) != 2 or {model.get("horizon") for model in models} != {1, 5}:
                raise ArtifactVerificationError(f"{key} must define each horizon exactly once")
            for model in models:
                if model.get("model_role") != role or not model.get("model_identity"):
                    raise ArtifactVerificationError(f"invalid {role} model identity")
                if model.get("name") != "Naive" and not model.get("primary_forecast"):
                    if not all(model.get(field) for field in ("fit_cutoff", "fit_label_cutoff", "recipe_identity")):
                        raise ArtifactVerificationError(f"{role} learned model lacks fit identity")
                    if datetime.fromisoformat(model["fit_label_cutoff"]) >= datetime.fromisoformat(model["fit_cutoff"]):
                        raise ArtifactVerificationError("fitted labels must precede the prediction cutoff")
                if model.get("primary_forecast") == "Naive":
                    runtime = model.get("runtime_model_identity", {})
                    if (runtime.get("name") != "XGBoost" or runtime.get("model_role") != "deployment_live"
                            or runtime.get("horizon") != model["horizon"] or not runtime.get("model_identity")
                            or not runtime.get("fit_cutoff") or not runtime.get("fit_label_cutoff")
                            or datetime.fromisoformat(runtime["fit_label_cutoff"]) >= datetime.fromisoformat(runtime["fit_cutoff"])):
                        raise ArtifactVerificationError("experimental fallback lacks a valid deployment identity")
    if version == 3:
        if manifest.get('research_protocol') != 'cotton-research-v1':
            raise ArtifactVerificationError('Unknown research protocol')
        for model in manifest['production_models']:
            if model['format'] != 'research_adapter_v1':
                raise ArtifactVerificationError('Research release requires explicit adapters')
            members = model.get('members', [])
            if model['name'] != 'Naive' and not members:
                raise ArtifactVerificationError('Research model has no members')
            for member in [*members, *model.get('experimental_members', [])]:
                if member.get('path') not in expected or member.get('adapter') not in expected:
                    raise ArtifactVerificationError('Research adapter/model not checksummed')
            receipt = manifest.get('reproduction', {}).get(str(model['horizon']), {})
            difference = receipt.get('max_abs_log_return_difference')
            if (receipt.get('status') != 'passed' or receipt.get('fresh_fits') is not True
                    or not isinstance(difference, (float, int)) or not 0 <= difference <= 1e-6):
                raise ArtifactVerificationError('Research release requires fresh reproduction')
    return manifest, _inventory_identity(expected)


@contextmanager
def _installation_lock(lock_path: Path):
    # Keep the file in place so contenders always lock the same file. The OS
    # releases the lock when its descriptor closes, including on process exit.
    with lock_path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ArtifactVerificationError(
                f"cannot acquire artifact install lock; another artifact installation may be active: {lock_path}"
            ) from exc
        yield


def install_bundle(bundle_path: Path, destination: Path) -> dict:
    """Install one verified release at a time and recover interrupted activation."""
    destination = destination.parent.resolve() / destination.name
    staged = destination.with_name(f".{destination.name}.staged")
    backup = destination.with_name(f".{destination.name}.previous")
    lock_path = destination.with_name(f".{destination.name}.install.lock")
    for path in (destination, staged, backup, lock_path):
        if path.is_symlink() or path.is_junction():
            raise ArtifactVerificationError(f"linked artifact installation path: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _installation_lock(lock_path):
        if backup.exists() and not destination.exists():
            verify_directory(backup)
            backup.replace(destination)
        return _install_bundle_locked(bundle_path, destination, staged, backup)


def _install_bundle_locked(bundle_path: Path, destination: Path, staged: Path, backup: Path) -> dict:
    if not bundle_path.is_file():
        raise ArtifactVerificationError(f"bundle not found: {bundle_path}")
    sidecar = bundle_path.with_suffix(f"{bundle_path.suffix}.sha256")
    if not sidecar.is_file():
        raise ArtifactVerificationError(f"bundle checksum sidecar not found: {sidecar.name}")
    fields = sidecar.read_text(encoding="utf-8").split()
    if not fields or sha256_file(bundle_path) != fields[0]:
        raise ArtifactVerificationError("bundle checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="cottonlens-artifact-") as temp_dir:
        extracted = Path(temp_dir) / "bundle"
        extracted.mkdir()
        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(extracted, members=_safe_members(archive))
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        root = roots[0] if len(roots) == 1 and not (extracted / "manifest.json").exists() else extracted
        manifest = verify_directory(root)
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(root, staged)
        verify_directory(staged)
        if backup.exists():
            # A prior activation may have succeeded before its cleanup ran.
            # Only retire its recovery copy if the active release verifies.
            verify_directory(destination)
            shutil.rmtree(backup)
        if destination.exists():
            destination.replace(backup)
        try:
            staged.replace(destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
