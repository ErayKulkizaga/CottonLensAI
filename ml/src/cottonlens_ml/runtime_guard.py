"""Fail closed before importing training runtimes on a local machine or in CI."""

import os
import sys
from pathlib import Path


def require_colab_training() -> None:
    # The isolated pinned training venv does not install google.colab. Identify
    # the host runtime from inherited Colab metadata and its mounted workspace,
    # not by importing the notebook kernel's package into the training venv.
    if not (
        sys.platform == "linux" and Path("/content").is_dir()
        and Path("/content/drive/MyDrive").is_dir()
        and bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))
    ):
        raise RuntimeError("Real model training is permitted only in Google Colab; local/CI fits are blocked")
