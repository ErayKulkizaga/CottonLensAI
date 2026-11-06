"""Explicit synthetic admissions for research tests that fit generated labels."""
import pytest
from cottonlens_ml.research import volatility

from ml.tests.conftest import admit_generated_backends, admit_generated_jobs

__all__ = ['admit_generated_backends', 'admit_generated_jobs', 'admit_generated_risk']


@pytest.fixture
def admit_generated_risk(admit_generated_backends, monkeypatch):
    from research import ct_shock_control
    # This module retains an imported adapter alias; use the explicitly admitted
    # generated-data wrapper for its whole call, including nested optimization.
    monkeypatch.setattr(ct_shock_control, 'fit_predict', volatility.fit_predict)
