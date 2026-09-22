"""SQLite on Colab's local disk; consistent, resumable snapshots on Drive.

One writer per Drive root. A killed session can leave .writer-lock behind: only
remove that directory after confirming the previous Colab session has stopped.
Legacy mlruns files are left untouched; they are not silently migrated.
"""

import shutil
import sqlite3
import tempfile
import threading
from contextlib import closing, contextmanager
from pathlib import Path


def copy_database(source: Path, destination: Path) -> None:
    """SQLite's backup API includes committed WAL pages; plain file copy cannot."""
    if not source.is_file():
        raise FileNotFoundError(source)
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(destination)) as dst:
        if src.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"Invalid SQLite snapshot: {source}")
        src.backup(dst)


class TrackingSession:
    def __init__(self, drive_root: Path, experiment: str = "cottonlens-forecasting"):
        self.drive_root = drive_root
        self.root = drive_root / "tracking"
        self.experiment = experiment
        self.stop = threading.Event()
        self.mutex = threading.Lock()
        self.errors: list[Exception] = []

    def __enter__(self):
        import mlflow

        self.root.mkdir(parents=True, exist_ok=True)
        self.writer_lock = self.root / ".writer-lock"
        try:
            self.writer_lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"Tracking writer lock exists: {self.writer_lock}. Stop the old Colab "
                "session before explicitly removing this stale lock; do not run two writers."
            ) from exc
        self.temp = tempfile.TemporaryDirectory(prefix="cottonlens-tracking-", ignore_cleanup_errors=True)
        self.database = Path(self.temp.name) / "mlflow.sqlite"
        self.snapshot = self.root / "mlflow.sqlite"
        try:
            if self.snapshot.exists():
                copy_database(self.snapshot, self.database)
            mlflow.set_tracking_uri("sqlite:///" + self.database.as_posix())
            artifact_root = self.drive_root.resolve() / "mlruns" / "artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)
            experiment = mlflow.get_experiment_by_name(self.experiment)
            experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
                self.experiment, artifact_location=artifact_root.as_uri()
            )
            mlflow.set_experiment(experiment_id=experiment_id)
            self.backup()
            self.thread = threading.Thread(target=self._periodic_backup, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.temp.cleanup()
            self.writer_lock.rmdir()
            raise

    def backup(self):
        with self.mutex:
            local_snapshot = Path(self.temp.name) / "snapshot.sqlite"
            copy_database(self.database, local_snapshot)
            staged = self.root / "mlflow.sqlite.pending"
            shutil.copyfile(local_snapshot, staged)
            if self.snapshot.exists():
                shutil.copyfile(self.snapshot, self.root / "mlflow.previous.sqlite")
            staged.replace(self.snapshot)

    def _periodic_backup(self):
        while not self.stop.wait(30):
            try:
                self.backup()
            except Exception as exc:  # noqa: BLE001 - surface any failed Drive snapshot
                self.errors.append(exc)
                print(f"MLflow Drive backup failed: {exc}", flush=True)
                return

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        try:
            self.backup()
        except BaseException:
            recovery = Path(tempfile.mkdtemp(prefix="cottonlens-tracking-recovery-"))
            copy_database(self.database, recovery / "mlflow.sqlite")
            print(f"Final Drive backup failed. Local recovery database: {recovery}. "
                  "Keep this Colab session running; writer lock retained.", flush=True)
            raise
        self.temp.cleanup()
        self.writer_lock.rmdir()
        if self.errors and exc is None:
            raise RuntimeError("Periodic MLflow backup failed; final snapshot was saved") from self.errors[0]


_active_session: TrackingSession | None = None


@contextmanager
def tracking_session(drive_root: Path, experiment: str = "cottonlens-forecasting"):
    global _active_session
    with TrackingSession(drive_root, experiment) as session:
        _active_session = session
        try:
            yield session
        finally:
            _active_session = None


@contextmanager
def tracked_run(**kwargs):
    import mlflow

    try:
        with mlflow.start_run(**kwargs) as run:
            yield run
    finally:
        if _active_session is not None:
            _active_session.backup()
