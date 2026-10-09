import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.legacy_loss_lineage import (
    generated_candidates,
    validate_recipe,
)


def test_legacy_generator_factorial_scope():
    recipes = generated_candidates(['cotton_ret_1'])
    assert len(recipes) == len({content_id(r) for r in recipes}) == 128
    counts = {(target, loss): sum(r['target'] == target and r['loss'] == loss for r in recipes)
              for target in ('scaled_log', 'price_delta') for loss in ('reg:squarederror', 'reg:absoluteerror')}
    assert set(counts.values()) == {32}
    assert recipes[0]['params']['max_depth'] == 7
    assert recipes[39]['params']['alpha'] == 0


def test_exact_recorded_float_is_not_rounded_or_replaced():
    generated = generated_candidates(['x'])[39]
    recorded = copy.deepcopy(generated)
    recorded['seed'] = 17
    recorded['params']['eta'] = np.nextafter(generated['params']['eta'], -np.inf)
    before = copy.deepcopy(recorded)
    differences = validate_recipe(recorded, generated)
    assert differences['eta']['recorded'] == before['params']['eta']
    assert recorded == before
    assert content_id(recorded) != content_id(generated)


@pytest.mark.parametrize('change', ['loss', 'target', 'features', 'seed', 'parameter_key'])
def test_recipe_structure_drift_rejected(change):
    generated = generated_candidates(['x'])[0]
    recorded = copy.deepcopy(generated)
    if change == 'loss':
        recorded['loss'] = 'reg:absoluteerror'
    elif change == 'target':
        recorded['target'] = 'price_delta'
    elif change == 'features':
        recorded['features'] = ['future_return']
    elif change == 'seed':
        recorded['seed'] = 999
    else:
        recorded['params']['unexpected'] = 1
    with pytest.raises(ValueError):
        validate_recipe(recorded, generated)


@pytest.mark.parametrize('value', [np.inf, np.nan, 1.0])
def test_meaningful_parameter_difference_rejected(value):
    generated = generated_candidates(['x'])[0]
    recorded = copy.deepcopy(generated)
    recorded['params']['eta'] = value
    with pytest.raises((ValueError, AssertionError)):
        validate_recipe(recorded, generated)


def test_optimization_cannot_disable_integrity_guards(tmp_path):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    code = 'from cottonlens_ml.research.legacy_loss_lineage import run; run("missing", "missing", "must-not-exist")'
    result = subprocess.run([sys.executable, '-O', '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and 'Scientific integrity assertions must remain enabled' in result.stderr
    assert not (tmp_path / 'must-not-exist').exists()
