"""Closed-form generated controls, no TensorFlow/native training."""
from copy import deepcopy

import numpy as np
import pytest

from research.session_math import independent_replay


def fixture():
    dense = {'units': 1, 'activation': 'linear', 'use_bias': True,
             'dtype': {'config': {'name': 'float32'}},
             'kernel_initializer': {'class_name': 'Constant', 'config': {'value': .1}},
             'bias_initializer': {'class_name': 'Zeros'},
             **dict.fromkeys(('kernel_regularizer', 'bias_regularizer',
                              'kernel_constraint', 'bias_constraint'))}
    optimizer = {'beta_1': .9, 'beta_2': .999, 'epsilon': 1e-7, 'learning_rate': .001,
                 'amsgrad': False, 'use_ema': False,
                 **dict.fromkeys(('weight_decay', 'clipnorm', 'global_clipnorm', 'clipvalue',
                                  'loss_scale_factor', 'gradient_accumulation_steps'))}
    config = {'config': {'layers': [
        {'class_name': 'InputLayer', 'config': {'batch_shape': [None, 5]}},
        {'class_name': 'Dense', 'config': dense}]},
        'compile_config': {'optimizer': {'class_name': 'Adam', 'config': optimizer},
                           'loss': 'mse', 'jit_compile': False}}
    # With zero input masks all kernel gradients are exactly zero.
    # Bias-only Adam closed-form constants, calculated with 55-digit Decimal:
    # step1 gradient=2.47; step2 gradient=2*(bias1+.435).
    constants = [(-.0009999987197272828, .247, .0061009),
                 (-.001878944987708802, .30910000025605455, .006848223104445107)]
    states, reference = {}, {}
    for step, (bias, m, v) in enumerate(constants, 1):
        arrays = [np.full((5, 1), .1), np.asarray([bias]), np.asarray(step), np.asarray(.001),
                  np.zeros((5, 1)), np.zeros((5, 1)), np.asarray([m]), np.asarray([v])]
        states['step1.keras' if step == 1 else 'reference2.keras'] = (deepcopy(config), arrays)
        reference[f'trace{step}'] = {'indexes': list(range((step-1)*16, step*16)),
                                    'mask': np.zeros((16, 5)).tolist()}
    # Closed-form Dense output: .1*sum(unmasked feature row)+bias2.
    reference['prediction'] = [[.1*((5*r-160)*5+10)/100+constants[1][0]] for r in range(64)]
    return states, reference


def test_closed_form_bias_updates_and_predictions_pass():
    result = independent_replay(*fixture())
    assert result['steps_checked'] == 2 and result['new_fits'] == 0
    assert result['adam_max_abs_error'] < 1e-6
    assert result['prediction_max_abs_error'] < 1e-6


@pytest.mark.parametrize('field,value', [
    ('beta_1', .8), ('learning_rate', .01), ('epsilon', .1),
    ('weight_decay', .1), ('use_ema', True), ('gradient_accumulation_steps', 2),
])
def test_changed_optimizer_recipe_rejected(field, value):
    states, reference = fixture()
    states['step1.keras'][0]['compile_config']['optimizer']['config'][field] = value
    with pytest.raises(ValueError, match='recipe'):
        independent_replay(states, reference)


@pytest.mark.parametrize('field', ['weights', 'moments', 'iteration', 'nan'])
def test_same_but_wrong_native_states_cannot_pass(field):
    states, reference = fixture()
    for _, arrays in states.values():
        if field == 'weights':
            arrays[0] += .1
        elif field == 'moments':
            arrays[6] += .1
        elif field == 'iteration':
            arrays[2] += 1
        else:
            arrays[0][0, 0] = np.nan
    with pytest.raises((AssertionError, ValueError)):
        independent_replay(states, reference)


@pytest.mark.parametrize('change', ['wrong_value', 'nan', 'shape'])
def test_reported_prediction_must_match_native_math(change):
    states, reference = fixture()
    if change == 'shape':
        reference['prediction'] = [[0.]]
    else:
        reference['prediction'][0][0] = np.nan if change == 'nan' else 99.
    with pytest.raises((AssertionError, ValueError)):
        independent_replay(states, reference)


def test_wrong_minibatch_shape_rejected():
    states, reference = fixture()
    reference['trace1']['indexes'][0] = reference['trace1']['indexes'][1]
    with pytest.raises(ValueError, match='minibatch'):
        independent_replay(states, reference)
