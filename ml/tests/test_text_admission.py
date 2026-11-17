"""Generated text only: admission must precede vocabulary/SVD learning."""
from contextlib import contextmanager

import pytest
from cottonlens_ml.research.admission import (
    AdmissionError,
    model_inputs_id,
    require_model_inputs,
)
from cottonlens_ml.research.text_adapter import TextPreprocessor

from ml.tests.admission_fixture import synthetic_admitter
from ml.tests.test_text_adapter import fixture


@contextmanager
def admitted(tmp_path, train, names, config, *, dispatch=True):
    recipe = {'family': 'ridge', 'horizon': 5, 'features': names, 'text_adapter': config}
    ledger = {'profile': 'generated-text-admission-test'}
    permit = synthetic_admitter(
        tmp_path/'admission', ledger,
        model_inputs=model_inputs_id(None, train, None, None, recipe, None))
    with permit.training_scope(ledger, recipe):
        if dispatch:
            require_model_inputs(None, train, None, None, recipe, None)
        yield


@pytest.fixture
def forbid_learning(monkeypatch):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    def forbidden(*args, **kwargs):
        pytest.fail('Learning started before rejection')

    monkeypatch.setattr(TfidfVectorizer, 'fit_transform', forbidden)
    monkeypatch.setattr(TruncatedSVD, 'fit', forbidden)


def test_direct_text_fit_requires_admission(forbid_learning):
    train, names, config = fixture()
    with pytest.raises(AdmissionError, match='Text transform inputs are not admitted'):
        TextPreprocessor.fit(train, names, config)


@pytest.mark.parametrize('change', [
    'scope_only', 'train', 'names', 'config', 'exit', 'nested_scope',
])
def test_text_fit_checks_exact_dispatch_and_scope(tmp_path, forbid_learning, change):
    train, names, config = fixture()
    with admitted(tmp_path, train, names, config, dispatch=change != 'scope_only'):
        if change == 'train':
            train.loc[0, 'value'] += 1
        elif change == 'names':
            names = ['other', *names[1:]]
        elif change == 'config':
            config = {**config, 'column': 'other_text'}
        if change == 'exit':
            pass
        elif change == 'nested_scope':
            with (admitted(tmp_path/'nested', train, names, config, dispatch=False),
                  pytest.raises(AdmissionError, match='Text transform inputs are not admitted')):
                TextPreprocessor.fit(train, names, config)
        else:
            with pytest.raises(AdmissionError, match='Text transform inputs are not admitted'):
                TextPreprocessor.fit(train, names, config)
    if change == 'exit':
        with pytest.raises(AdmissionError, match='Text transform inputs are not admitted'):
            TextPreprocessor.fit(train, names, config)


@pytest.mark.parametrize('prefix', ['target_', 'variance_target_', 'gk_target_'])
def test_target_feature_rejected_before_text_learning(tmp_path, forbid_learning, prefix):
    train, names, config = fixture()
    names = [prefix+'future', *names[1:]]
    with (admitted(tmp_path, train, names, config),
          pytest.raises(ValueError, match='Future target columns')):
        TextPreprocessor.fit(train, names, config)


def test_valid_text_transform_reuses_and_restores_scope(tmp_path):
    train, names, config = fixture()
    with admitted(tmp_path, train, names, config):
        first = TextPreprocessor.fit(train, names, config)
        with (admitted(tmp_path/'nested', train, names, config, dispatch=False),
              pytest.raises(AdmissionError)):
            TextPreprocessor.fit(train, names, config)
        second = TextPreprocessor.fit(train, names, config)
    assert first.as_dict() == second.as_dict()
