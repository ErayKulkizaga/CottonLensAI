"""Content-addressed, checksum-verified data evidence; no model imports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def frame_digest(frame: pd.DataFrame) -> str:
    # Preserve numeric bits rather than rounding observations through JSON.
    # Index is not a source value; ordered columns, dtypes and rows are identity.
    digest = hashlib.sha256(json.dumps(
        [(name, str(frame[name].dtype)) for name in frame.columns], separators=(",", ":")
    ).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_snapshot(root: Path, frames: dict[str, pd.DataFrame], metadata: dict) -> dict:
    """Publish a complete directory atomically; never replace existing evidence."""
    identity = {
        "frames": {name: frame_digest(frame) for name, frame in sorted(frames.items())},
        "metadata": metadata,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    destination = root / snapshot_id
    root.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        try:
            for name, frame in frames.items():
                value = frame.copy()
                value.attrs = {}  # Avoid absolute paths or nested evidence in parquet metadata.
                value.to_parquet(staging / f"{name}.parquet", index=False)
            manifest = {
                "snapshot_id": snapshot_id,
                "identity": identity,
                "created_at": datetime.now(UTC).isoformat(),
                "files": {
                    f"{name}.parquet": file_digest(staging / f"{name}.parquet") for name in sorted(frames)
                },
                "complete": True,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
            )
            try:
                os.rename(staging, destination)
            except OSError:
                if not destination.exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    manifest_path = destination / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (
            manifest["complete"] is True
            and manifest["snapshot_id"] == snapshot_id
            and manifest["identity"] == identity
            and set(manifest["files"]) == {f"{name}.parquet" for name in frames}
            and all(file_digest(destination / name) == checksum for name, checksum in manifest["files"].items())
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete or corrupted data snapshot: {destination}") from exc
    if not valid:
        raise ValueError(f"Data snapshot identity/checksum mismatch: {destination}")
    return {"snapshot_id": snapshot_id, "manifest_path": str(manifest_path), "checksums": manifest["files"]}
