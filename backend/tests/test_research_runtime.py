"""No fitting: synthetic linear adapter and missingness contract."""
import json

import numpy as np
import pytest

from app.research_runtime import ResearchModel


def payload(tmp_path):
    adapter = {'processor': {'names': ['a'], 'median': [2.], 'mean': [1.], 'scale': [2.]},
               'target': {'kind': 'scaled_log', 'mean': .01, 'scale': .02}, 'window': 1}
    (tmp_path / 'adapter.json').write_text(json.dumps(adapter))
    (tmp_path / 'linear.json').write_text(json.dumps({'coef': [2., 3.], 'intercept': 1.}))
    entry = {'name': 'ridge', 'members': [{'adapter': 'adapter.json', 'path': 'linear.json',
                                          'format': 'linear_json', 'weight': 1.}]}
    return adapter, entry


def test_linear_missingness_and_target_inverse(tmp_path):
    adapter, entry = payload(tmp_path)
    model = ResearchModel(tmp_path, entry)
    assert model.predict({'a': 3.}) == pytest.approx(.07)
    assert model.predict({'a': None}) == pytest.approx(.11)
    assert model.predict({'a': np.nan}) == pytest.approx(.11)
    adapter['target'] = {'kind': 'price_delta', 'mean': 0., 'scale': 1.}
    (tmp_path / 'adapter.json').write_text(json.dumps(adapter))
    assert ResearchModel(tmp_path, entry).predict({'a': 3., 'cotton_close': 80.}) == pytest.approx(np.log(83/80))


def test_invalid_adapter_and_weights_fail_closed(tmp_path):
    adapter, entry = payload(tmp_path)
    entry['members'][0]['weight'] = .5
    with pytest.raises(ValueError, match='sum to one'):
        ResearchModel(tmp_path, entry)
    entry['members'][0]['weight'] = 1.
    adapter['processor']['scale'] = [0.]
    (tmp_path / 'adapter.json').write_text(json.dumps(adapter))
    with pytest.raises(ValueError, match='scale'):
        ResearchModel(tmp_path, entry)
    assert ResearchModel(tmp_path, {'name': 'Naive', 'members': []}).predict({}) == 0.
