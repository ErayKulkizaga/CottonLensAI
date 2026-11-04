"""Exercise schema changes on isolated SQLite databases."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_migration_makes_future_target_date_nullable(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    database = tmp_path / "migration-only.db"
    assert not database.exists()
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{database.as_posix()}"}
    for revision in ("0001", "0002"):
        subprocess.run([sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", revision],
                       cwd=backend, env=env, check=True, capture_output=True, text=True)
    with sqlite3.connect(database) as connection:
        target = next(row for row in connection.execute("PRAGMA table_info(forecasts)") if row[1] == "target_date")
        assert target[3] == 0  # notnull flag is off
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0002"


def test_market_history_migration_preserves_legacy_rows(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    database = tmp_path / "market-history-migration.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{database.as_posix()}"}

    def upgrade(revision):
        subprocess.run([sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", revision],
                       cwd=backend, env=env, check=True, capture_output=True, text=True)

    upgrade("0002")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO market_observations (observed_on, series, close, source, ingested_at) VALUES (?, ?, ?, ?, ?)",
            ("2024-04-19", "cotton", 80.0, "colab-artifact", "2024-04-20"),
        )
    upgrade("head")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == ("0003",)
        assert connection.execute("SELECT close FROM market_observations").fetchall() == [(80.0,)]
        assert connection.execute("SELECT count(*) FROM artifact_market_observations").fetchone() == (0,)
        foreign_key = connection.execute("PRAGMA foreign_key_list(artifact_market_observations)").fetchone()
        assert foreign_key[2:5] == ("artifact_imports", "artifact_import_id", "id")

        connection.execute(
            "INSERT INTO artifact_imports (id, artifact_version, manifest, checksum, imported_at) VALUES (?, ?, ?, ?, ?)",
            ("test-receipt", "test-release", "{}", "f" * 64, "2024-04-20"),
        )
        connection.execute(
            "INSERT INTO artifact_market_observations (artifact_import_id, observed_on, series, close) VALUES (?, ?, ?, ?)",
            ("test-receipt", "2024-04-19", "cotton", 84.0),
        )

    downgrade = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", "downgrade", "0002"],
        cwd=backend, env=env, capture_output=True, text=True, check=False,
    )
    assert downgrade.returncode != 0
    assert "Cannot downgrade while artifact-scoped market observations exist" in downgrade.stderr
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT close FROM artifact_market_observations").fetchall() == [(84.0,)]
