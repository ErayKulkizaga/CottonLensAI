"""Identify the exact uncommitted source supplied to Colab, without a remote."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name)


def source_identity(repo: Path) -> dict:
    manifest = json.loads((repo / ".source-manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "source_id"}
    if manifest.get("source_id") != manifest_id(payload) or not manifest.get("files"):
        raise ValueError("Source manifest identity mismatch")
    for name, expected in manifest["files"].items():
        if not safe_member(name) or not (repo / name).is_file() or digest(repo / name) != expected:
            raise ValueError(f"Source file missing or changed: {name}")
    # An injected module must not be invisible to the recorded identity.
    for base in (repo / "ml/src", repo / "backend/app"):
        for path in base.rglob("*.py"):
            if path.relative_to(repo).as_posix() not in manifest["files"]:
                raise ValueError(f"Unmanifested source module: {path.name}")
    return manifest
