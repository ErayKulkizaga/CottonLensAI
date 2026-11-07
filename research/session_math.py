"""Check the fixed synthetic Adam states/predictions with NumPy, never fit."""
import numpy as np


def independent_replay(states, reference):
    config = states['step1.keras'][0]
    layers = config['config']['layers']
    compile_config = config['compile_config']
    optimizer = compile_config['optimizer']
    dense = layers[-1]['config']
    cfg = optimizer['config']
    if ([v['class_name'] for v in layers] != ['InputLayer', 'Dense']
            or layers[0]['config']['batch_shape'] != [None, 5]
            or dense['units'] != 1 or dense['activation'] != 'linear' or not dense['use_bias']
            or dense['kernel_initializer']['class_name'] != 'Constant'
            or dense['kernel_initializer']['config']['value'] != .1
            or dense['bias_initializer']['class_name'] != 'Zeros'
            or dense['dtype']['config']['name'] != 'float32'
            or any(dense[k] is not None for k in (
                'kernel_regularizer', 'bias_regularizer', 'kernel_constraint', 'bias_constraint'))
            or compile_config['loss'] != 'mse' or compile_config['jit_compile'] is not False
            or optimizer['class_name'] != 'Adam'
            or cfg['beta_1'] != .9 or cfg['beta_2'] != .999 or cfg['epsilon'] != 1e-7
            or cfg['amsgrad'] is not False or cfg['use_ema'] is not False
            or any(cfg[k] is not None for k in (
                'weight_decay', 'clipnorm', 'global_clipnorm', 'clipvalue',
                'loss_scale_factor', 'gradient_accumulation_steps'))
            or not np.isclose(cfg['learning_rate'], .001, rtol=0, atol=1e-10)):
        raise ValueError('Unsupported fixed session math recipe')
    x = (np.arange(320, dtype='float32').reshape(64, 5)-160)/100
    y = 2*x[:, :1]-x[:, 1:2]
    w, b = np.full((5, 1), .1, dtype='float32'), np.zeros(1, dtype='float32')
    moments = [np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b)]
    maximum = 0.
    for step, name in ((1, 'step1.keras'), (2, 'reference2.keras')):
        trace = reference[f'trace{step}']
        indexes = np.asarray(trace['indexes'])
        mask = np.asarray(trace['mask'], dtype='float32')
        if (indexes.shape != (16,) or indexes.dtype.kind not in 'iu'
                or len(set(indexes.tolist())) != 16 or (indexes < 0).any() or (indexes >= 64).any()
                or mask.shape != (16, 5) or not np.isfinite(mask).all()):
            raise ValueError('Invalid recorded minibatch')
        batch = x[indexes]*mask
        residual = batch@w+b-y[indexes]
        gradients = [np.float32(2/16)*(batch.T@residual), np.float32(2)*residual.mean(axis=0)]
        parameters = []
        alpha = float(np.float32(.001))*np.sqrt(1-.999**step)/(1-.9**step)
        for j, (p, g) in enumerate(zip((w, b), gradients, strict=True)):
            m, v = moments[2*j:2*j+2]
            m2, v2 = .9*m+.1*g, .999*v+.001*np.square(g)
            parameters.append(p-alpha*m2/(np.sqrt(v2)+1e-7))
            moments[2*j:2*j+2] = [m2, v2]
        w, b = parameters
        expected = [w, b, np.asarray(step), np.asarray(.001, dtype='float32'), *moments]
        for actual, wanted in zip(states[name][1], expected, strict=True):
            actual = np.asarray(actual)
            if actual.shape != wanted.shape or not np.isfinite(actual).all():
                raise ValueError('Invalid native state for independent replay')
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(actual-wanted))))
    prediction = np.asarray(reference['prediction'])
    expected_prediction = x@w+b
    if prediction.shape != (64, 1) or not np.isfinite(prediction).all():
        raise ValueError('Invalid session prediction vector')
    np.testing.assert_allclose(prediction, expected_prediction, rtol=0, atol=1e-6)
    return {'adam_max_abs_error': maximum,
            'prediction_max_abs_error': float(np.max(np.abs(prediction-expected_prediction))),
            'steps_checked': 2, 'new_fits': 0}
