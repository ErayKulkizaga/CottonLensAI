"""Isolate application configuration before any test imports app modules."""

import os
import tempfile
from pathlib import Path

_test_workspace = tempfile.TemporaryDirectory(prefix="cottonlens-backend-tests-")
_test_root = Path(_test_workspace.name)
os.environ["DATABASE_URL"] = f"sqlite:///{(_test_root / 'api.db').as_posix()}"
os.environ["DEMO_MODE"] = "true"
os.environ["ARTIFACT_DIR"] = str(_test_root / "missing-artifact")


def pytest_sessionstart(session):
    from app.database import engine

    if Path(engine.url.database).resolve() != (_test_root / "api.db").resolve():
        raise RuntimeError("Refusing backend tests: application database is not the isolated test database")


def pytest_sessionfinish(session, exitstatus):
    from app.database import engine

    engine.dispose()
    _test_workspace.cleanup()
