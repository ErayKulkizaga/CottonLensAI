"""Exercise the nullable live target date migration on an isolated temporary DB."""

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
    for revision in ("0001", "head"):
        subprocess.run([sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", revision],
                       cwd=backend, env=env, check=True, capture_output=True, text=True)
    with sqlite3.connect(database) as connection:
        target = next(row for row in connection.execute("PRAGMA table_info(forecasts)") if row[1] == "target_date")
        assert target[3] == 0  # notnull flag is off
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0002"
