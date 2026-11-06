"""Opt-in generated-job admission for historical protocol/ledger unit tests."""
import pytest
from cottonlens_ml.research.admission import identity, model_inputs_id
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.ledger import Ledger

from ml.tests.admission_fixture import synthetic_admitter


@pytest.fixture
def admit_generated_jobs(tmp_path, monkeypatch):
    """Only request in tests whose operations are stubs or generated data.

    This does not bypass inspect/require: each job gets a synthetic witness and
    the real admission validator. Production entrypoints never import tests.
    """
    original = Ledger.run
    original_fit = Experiment.fit

    def fit(experiment, spec, train, validation, test, role, *, iterations=None, repeat=None):
        experiment.admission = synthetic_admitter(
            tmp_path/'generated-model-admission', experiment.identity,
            model_inputs=model_inputs_id(experiment.history, train, validation, test, spec, iterations))
        return original_fit(experiment, spec, train, validation, test, role, iterations=iterations, repeat=repeat)

    def run(ledger, specification, operation, **kwargs):
        if kwargs.get('admission') is None:
            kwargs['admission'] = synthetic_admitter(tmp_path/'generated-job-admission', ledger.identity)
        return original(ledger, specification, operation, **kwargs)

    monkeypatch.setattr(Ledger, 'run', run)
    monkeypatch.setattr(Experiment, 'fit', fit)


@pytest.fixture
def admit_generated_backends(tmp_path, monkeypatch):
    """Opt-in direct adapter checks for tests with explicitly generated labels."""
    from cottonlens_ml.research import admission as module
    original = module.require_backend_inputs

    def check(history, train, test, spec):
        ledger_identity = {'profile': 'generated-backend-test'}
        specification = {'recipe': spec}
        permit = synthetic_admitter(tmp_path/'generated-backend-admission', ledger_identity,
                                    model_inputs=model_inputs_id(history, train, None, test, spec, None))
        with permit.training_scope(ledger_identity, specification):
            return original(history, train, test, spec)

    monkeypatch.setattr(module, 'require_backend_inputs', check)


@pytest.fixture
def admit_generated_models(tmp_path, monkeypatch):
    from cottonlens_ml.research import models
    original = models.fit_predict

    def fit(history, train, validation, test, recipe, workspace, *, iterations=None):
        ledger_identity, specification = {'profile': 'generated-model-test'}, {'recipe': recipe}
        permit = synthetic_admitter(tmp_path/'generated-model-admission', ledger_identity,
                                    model_inputs=model_inputs_id(history, train, validation, test, recipe, iterations))
        with permit.training_scope(ledger_identity, specification):
            return original(history, train, validation, test, recipe, workspace, iterations=iterations)

    monkeypatch.setattr(models, 'fit_predict', fit)


@pytest.fixture
def admit_generated_legacy(tmp_path, monkeypatch):
    """For fake-estimator wiring tests only; real guard still checks exact IDs."""
    from cottonlens_ml import training
    original = training.require_legacy_fit

    def check(fit_identity):
        ledger_identity = {'profile': 'generated-legacy-test'}
        specification = {'legacy_fit': fit_identity}
        admission = synthetic_admitter(tmp_path/'generated-legacy-admission', ledger_identity,
                                       legacy_fit_ids=[identity(fit_identity)])
        with admission.training_scope(ledger_identity, specification):
            return original(fit_identity)

    monkeypatch.setattr(training, 'require_legacy_fit', check)
