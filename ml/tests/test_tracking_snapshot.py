import sqlite3

import pytest
from cottonlens_ml.tracking import TrackingSession, copy_database


def test_sqlite_backup_includes_committed_wal_and_can_be_restored(tmp_path):
    source = tmp_path / "local.sqlite"
    snapshot = tmp_path / "drive.sqlite"
    restored = tmp_path / "restored.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE runs(id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO runs VALUES('child-run')")
        connection.commit()
        copy_database(source, snapshot)
    copy_database(snapshot, restored)
    with sqlite3.connect(restored) as connection:
        assert connection.execute("SELECT id FROM runs").fetchall() == [("child-run",)]


def test_corrupt_snapshot_is_rejected(tmp_path):
    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError):
        copy_database(corrupt, tmp_path / "restored.sqlite")


def test_missing_snapshot_is_not_silently_created(tmp_path):
    source = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        copy_database(source, tmp_path / "restored.sqlite")
    assert not source.exists()


def test_snapshot_failure_preserves_previous_backup(tmp_path, monkeypatch):
    session = TrackingSession(tmp_path)
    session.root.mkdir()
    session.snapshot = session.root / "mlflow.sqlite"
    session.snapshot.write_bytes(b"previous valid snapshot")
    session.database = tmp_path / "missing.sqlite"
    class Temporary:
        name = str(tmp_path)
    session.temp = Temporary()
    def fail(*args):
        raise OSError("Drive unavailable")
    monkeypatch.setattr("cottonlens_ml.tracking.copy_database", fail)
    with pytest.raises(OSError):
        session.backup()
    assert session.snapshot.read_bytes() == b"previous valid snapshot"
