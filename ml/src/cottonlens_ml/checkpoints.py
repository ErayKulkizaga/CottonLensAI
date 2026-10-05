"""Checksummed, identity-bound completion records; model presence is not completion."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity_hash(identity: dict) -> str:
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completion_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".complete.json")


def history_path(path: Path) -> Path:
    return path.with_suffix(".history.json")


def completed_checkpoint(path: Path, identity: dict, *, require_history: bool = False) -> bool:
    """Reject old, interrupted, mismatched or corrupt caches without loading a model."""
    try:
        record = json.loads(completion_path(path).read_text(encoding="utf-8"))
        expected_files = [path, history_path(path)] if require_history else [path]
        if (not isinstance(record, dict)
                or record.get("state") != "completed" or record.get("schema") != 1
                or record.get("identity") != identity
                or record.get("identity_hash") != identity_hash(identity)
                or set(record.get("files", {})) != {item.name for item in expected_files}):
            return False
        return all(
            item.is_file() and item.stat().st_size > 0
            and record["files"][item.name] == file_hash(item)
            for item in expected_files
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def save_completed_checkpoint(path: Path, identity: dict, save_model, *, history: list | None = None) -> None:
    """Commit the manifest last, after all payloads have been staged and replaced.

    A crash before marker replacement leaves an unusable checkpoint. One writer
    per checkpoint root is required (enforced by the pipeline tracking lock).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = completion_path(path)
    marker.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    staged_model = path.with_name(f".{path.stem}.{token}.pending{path.suffix}")
    staged_history = path.with_name(f".{path.stem}.{token}.pending.history.json")
    staged_marker = marker.with_name(f".{marker.name}.{token}.pending")
    try:
        save_model(staged_model)
        if not staged_model.is_file() or staged_model.stat().st_size == 0:
            raise ValueError("Checkpoint writer did not produce a nonempty model")
        files = {path.name: file_hash(staged_model)}
        if history is not None:
            staged_history.write_text(json.dumps(history, allow_nan=False), encoding="utf-8")
            files[history_path(path).name] = file_hash(staged_history)
        staged_marker.write_text(canonical_json({
            "schema": 1, "state": "completed", "identity": identity,
            "identity_hash": identity_hash(identity), "files": files,
        }), encoding="utf-8")
        os.replace(staged_model, path)
        if history is not None:
            os.replace(staged_history, history_path(path))
        os.replace(staged_marker, marker)
    finally:
        for temporary in (staged_model, staged_history, staged_marker):
            temporary.unlink(missing_ok=True)
