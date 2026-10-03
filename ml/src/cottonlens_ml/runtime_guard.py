"""Fail closed before importing training runtimes on a local machine or in CI."""

import os
import sys
from pathlib import Path

CPU_FAMILIES = frozenset({'ridge', 'elasticnet', 'xgboost', 'ewma', 'har', 'garch'})
COLAB_CPU_FAMILIES = frozenset({'arima', 'drift', 'naive'})


def require_training(family, device='cuda', threads=2):
    """Explicit CPU permission is not a GPU fallback or a CI exception."""
    if device not in ('cpu', 'cuda'):
        raise RuntimeError('Explicit cpu or cuda device required')
    if family in COLAB_CPU_FAMILIES and device != 'cpu':
        raise RuntimeError('Statistical baselines require CPU, not GPU fallback')
    if device != 'cpu':
        require_colab_training()
        return
    if any(os.environ.get(k, '').lower() in ('1', 'true', 'yes') for k in ('CI', 'GITHUB_ACTIONS')):
        raise RuntimeError('CI cannot train real models')
    if family in COLAB_CPU_FAMILIES:
        if type(threads) is not int or not 1 <= threads <= 2:
            raise RuntimeError('Two-thread limit required')
        require_colab_training()
        return
    if family not in CPU_FAMILIES or type(threads) is not int or not 1 <= threads <= 2:
        raise RuntimeError('CPU allowlist and two-thread limit required')
    if os.environ.get('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR') != '1':
        raise RuntimeError('Explicit COTTONLENS_ALLOW_LOCAL_CPU_TABULAR=1 required')


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
