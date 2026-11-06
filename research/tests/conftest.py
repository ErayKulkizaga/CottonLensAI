"""Explicit synthetic admissions for research tests that fit generated labels."""
import pytest
from cottonlens_ml.research import admission

from ml.tests.admission_fixture import synthetic_admitter
from ml.tests.conftest import admit_generated_jobs

__all__ = ['admit_generated_jobs', 'admit_generated_risk']


@pytest.fixture
def admit_generated_risk(tmp_path, monkeypatch):
    original = admission.require_backend_inputs

    def require(history, train, test, recipe):
        ledger_identity = {'profile': 'generated-research-risk-fixture'}
        specification = {'recipe': recipe}
        permit = synthetic_admitter(tmp_path/'risk-fixture-admission', ledger_identity,
                                   model_inputs=admission.model_inputs_id(history, train, None, test, recipe, None))
        with permit.training_scope(ledger_identity, specification):
            return original(history, train, test, recipe)

    monkeypatch.setattr(admission, 'require_backend_inputs', require)
