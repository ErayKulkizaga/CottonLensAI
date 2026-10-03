"""Colab-only estimators. Every GPU family must fail closed on device fallback."""
import json
import os
import subprocess
import threading
import time

import numpy as np

from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research.protocol import Preprocessor, Target, inputs
from cottonlens_ml.runtime_guard import require_colab_training, require_training

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def gpu_sample():
    require_colab_training()
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], text=True)
    name, utilization, used, total = [s.strip() for s in output.splitlines()[0].split(',')]
    return {'name': name, 'utilization_pct': float(utilization), 'used_mb': float(used), 'total_mb': float(total)}


class Telemetry:
    def __enter__(self):
        self.samples, self.errors = [], []
        self.stopped = threading.Event()
        self.samples.append(gpu_sample())

        def sample():
            while not self.stopped.wait(2):
                try:
                    self.samples.append(gpu_sample())
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    self.errors.append(str(exc))
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=10)
        self.samples.append(gpu_sample())

    def report(self):
        if self.errors:
            raise RuntimeError('GPU telemetry failed: ' + self.errors[0])
        return {'samples': self.samples, 'peak_memory_fraction': max(s['used_mb'] / s['total_mb'] for s in self.samples),
                'mean_utilization_pct': float(np.mean([s['utilization_pct'] for s in self.samples]))}


class CPUTelemetry:
    def __enter__(self):
        from threadpoolctl import threadpool_limits
        self.limits = threadpool_limits(limits=2)
        self.limits.__enter__()
        return self

    def __exit__(self, *args):
        self.limits.__exit__(*args)

    def report(self):
        return {'device': 'cpu', 'thread_limit': 2, 'gpu_utilization_claimed': False}


def tensorflow_gpu(seed):
    require_colab_training()
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('TensorFlow GPU required; no CPU fallback')
    # Called before first TF fit in the process. Bound TF allocation to 80%.
    try:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(
            memory_limit=int(gpu_sample()['total_mb'] * .8))])
    except RuntimeError:
        if not tf.config.list_logical_devices('GPU'):
            raise
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.set_soft_device_placement(False)
    return tf


def require_gpu_variables(model):
    devices = [str(getattr(getattr(v, 'value', v), 'device', '')) for v in model.trainable_variables]
    if not devices or any('GPU:' not in device.upper() for device in devices):
        raise RuntimeError('TensorFlow trainable variables must reside on GPU')
    return sorted(set(devices))


def class_labels(rows, horizon):
    return (np.sign(rows[f'target_return_{horizon}'].to_numpy()) + 1).astype(int)


def class_metrics(labels, probability):
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(labels), 3) or not np.isfinite(probability).all():
        raise ValueError('Three finite direction probabilities required')
    probability = np.clip(probability, 1e-9, 1)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    recalls = [float(np.mean(predicted[labels == c] == c)) for c in (0, 1, 2) if np.any(labels == c)]
    return {'log_loss': float(-np.log(probability[np.arange(len(labels)), labels]).mean()),
            'brier': float(np.square(probability - np.eye(3)[labels]).sum(axis=1).mean()),
            'balanced_accuracy': 100 * float(np.mean(recalls)),
            'directional_accuracy': 100 * float(np.mean(predicted == labels)),
            'flat_targets': int(np.sum(labels == 1)), 'sample_count': len(labels)}


def temperature(probability, value):
    logits = np.log(np.clip(probability, 1e-9, 1)) / value
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate(probability, labels):
    return min((.5, .75, 1., 1.5, 2., 3.), key=lambda t: (class_metrics(labels, temperature(probability, t))['log_loss'], abs(t - 1)))


class PriceMetric:
    """CatBoost GPU custom metric; weights carry origin prices for log targets.

    Validation weights never enter fitting. Delta targets need no price weights.
    This is deliberately smoke-tested in the installed CatBoost GPU runtime.
    """
    def __init__(self, kind, mean, scale):
        self.kind, self.mean, self.scale = kind, mean, scale

    def is_max_optimal(self):
        return False

    def get_final_error(self, error, weight):
        return error / max(weight, 1.)

    def evaluate(self, approxes, target, weight):
        error = 0.
        for i in range(len(target)):
            actual = target[i] * self.scale + self.mean
            predicted = approxes[0][i] * self.scale + self.mean
            if self.kind == 'price_delta':
                error += abs(actual - predicted)
            else:
                error += (1. if weight is None else weight[i]) * abs(np.exp(actual) - np.exp(predicted))
        return error, float(len(target))


def fit_predict(history, train, validation, test, spec, workspace, *, iterations=None):
    device_requested = spec.get('device', 'cuda')
    require_training(spec.get('family'), device_requested)
    if spec['family'] in ('har', 'garch'):
        from cottonlens_ml.research.volatility import fit_predict as volatility_fit
        with CPUTelemetry():
            return volatility_fit(history, train, test, spec, workspace)
    started = time.monotonic()
    family, h = spec['family'], spec['horizon']
    classification = spec.get('task', 'price') == 'direction'
    transfer = spec.get('training_policy') is not None
    weights, transfer_details = None, {}
    if transfer:
        from cottonlens_ml.research.transfer import training_state
        processor, target, weights, transfer_details = training_state(history,train,test,spec)
    else:
        processor = Preprocessor.fit(train, spec['features'])
        target = Target.fit(train, h, spec.get('target', 'scaled_log'))
    window = spec.get('window', 1)
    x = processor.transform(train) if transfer else inputs(history, train, processor, window)
    xt = inputs(history, test, processor, window)
    xv = inputs(history, validation, processor, window) if validation is not None else None
    train_labels = class_labels(train, h) if classification else target.forward(train, h)
    classes = np.unique(train_labels).astype(int).tolist() if classification else []
    if classification and len(classes) < 2:
        raise ValueError('Direction training needs at least two historical classes')
    y = np.searchsorted(classes, train_labels) if classification else train_labels
    yv = class_labels(validation, h) if classification and validation is not None else (
        target.forward(validation, h) if validation is not None else None)
    if classification and validation is not None:
        # Tree classes always cover down/flat/up, including an unseen rare class.
        y = train_labels
    data_seconds = time.monotonic() - started
    curves, details = {}, {}
    params, seed = spec.get('params', {}), spec['seed']
    limit = iterations or spec.get('max_iterations', 300 if family in ('mlp', 'lstm', 'tcn') else 4000)
    model_path = workspace / 'model.json'

    def metric(actual, predicted):
        if validation is None:
            raise ValueError('Validation price metric without validation')
        log = target.inverse(predicted, validation.cotton_close.to_numpy())
        return evaluate(validation.cotton_close.to_numpy(), validation[f'target_return_{h}'].to_numpy(), log)['mae']

    with (CPUTelemetry() if device_requested == 'cpu' else Telemetry()) as telemetry:
        if family == 'xgboost':
            import xgboost as xgb
            if classification:
                # Low-level multi:softprob supports absent classes with num_class=3.
                config = {**params, 'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
                          'device': device_requested, 'tree_method': 'hist', 'seed': seed, 'nthread': 2}
                dtrain = xgb.DMatrix(x, label=train_labels)
                evals = [(dtrain, 'train')]
                if validation is not None:
                    evals.append((xgb.DMatrix(xv, label=yv), 'validation'))
                model = xgb.train(config, dtrain, num_boost_round=limit, evals=evals,
                                  early_stopping_rounds=150 if iterations is None else None,
                                  evals_result=curves, verbose_eval=False)
            else:
                config = {**params, 'objective': spec['loss'], 'device': device_requested, 'tree_method': 'hist',
                          'seed': seed, 'nthread': 2, 'disable_default_eval_metric': 1}
                xgb_weights = weights if transfer else (
                    train.cotton_close.to_numpy() / train.cotton_close.mean() if spec.get('price_weighted') else None)
                dtrain = xgb.DMatrix(x, label=y, weight=xgb_weights)
                evals = [(dtrain, 'train')]
                prices_by_id = {id(dtrain): train.cotton_close.to_numpy()}
                if validation is not None:
                    dval = xgb.DMatrix(xv, label=yv)
                    evals.append((dval, 'validation'))
                    prices_by_id[id(dval)] = validation.cotton_close.to_numpy()

                def price_eval(pred, data):
                    prices = prices_by_id[id(data)]
                    a = target.inverse(data.get_label(), prices)
                    p = target.inverse(pred, prices)
                    measured = train.asset.eq('cotton').to_numpy() if transfer and data is dtrain else np.ones(len(prices),dtype=bool)
                    value = evaluate(prices[measured], a[measured], p[measured])['mae']
                    if transfer:
                        loss_weights = xgb_weights if data is dtrain else np.ones(len(prices))
                        target_loss = float(np.average(np.square(data.get_label()-pred),weights=loss_weights))
                        # Last metric drives stopping: Cotton price-MAE, never pooled price levels.
                        return [('weighted_target_mse',target_loss),('price_mae',value)]
                    return 'price_mae', value
                model = xgb.train(config, dtrain, num_boost_round=limit, evals=evals, custom_metric=price_eval,
                                  early_stopping_rounds=spec.get('patience', 150) if iterations is None else None,
                                  evals_result=curves, verbose_eval=False)
            device = json.loads(model.save_config())['learner']['generic_param']['device']
            if device_requested == 'cuda' and not device.startswith('cuda'):
                raise RuntimeError(f'XGBoost silently fell back to {device}')
            if device_requested == 'cpu' and device != 'cpu':
                raise RuntimeError('CPU fit used an unexpected device')
            count = model.best_iteration + 1 if iterations is None and validation is not None else limit
            model = model[:count]
            prediction = model.predict(xgb.DMatrix(xt))
            training_prediction = model.predict(xgb.DMatrix(x))
            model.save_model(model_path)
            dump = model.get_dump(dump_format='json')
            details = {'effective_device': device, 'trees': len(dump),
                       'splits': sum(tree.count('"split":') for tree in dump),
                       'used_features': model.get_score(), 'iterations_run': len(next(iter(curves['train'].values()))),
                       **transfer_details}
        elif family == 'catboost':
            from catboost import CatBoostClassifier, CatBoostRegressor, Pool
            config = {**params, 'iterations': limit, 'random_seed': seed, 'task_type': 'GPU', 'devices': '0',
                      'gpu_ram_part': .8, 'allow_writing_files': False, 'verbose': False}
            if classification:
                config.update(loss_function='MultiClass', classes_count=3, eval_metric='MultiClass')
                model = CatBoostClassifier(**config)
                training = Pool(x, label=train_labels)
                valid = Pool(xv, label=yv) if validation is not None else None
            else:
                config.update(loss_function='MAE' if spec['loss'] == 'reg:absoluteerror' else 'RMSE',
                              eval_metric=PriceMetric(target.kind, target.mean, target.scale))
                model = CatBoostRegressor(**config)
                training = Pool(x, label=y)
                valid = Pool(xv, label=yv, weight=validation.cotton_close.to_numpy()) if validation is not None else None
            model.fit(training, eval_set=valid, early_stopping_rounds=150 if iterations is None else None,
                      use_best_model=False)
            if model.get_all_params().get('task_type') != 'GPU':
                raise RuntimeError('CatBoost GPU task required')
            count = model.get_best_iteration() + 1 if validation is not None and iterations is None else model.tree_count_
            curves = model.get_evals_result()
            ran = model.tree_count_
            if not classification:
                # Training Pool has uniform weights. Recompute its exact price curve
                # rather than labelling the unweighted log-domain callback as price MAE.
                curves['training_price_mae'] = [evaluate(train.cotton_close.to_numpy(), train[f'target_return_{h}'].to_numpy(),
                    target.inverse(p, train.cotton_close.to_numpy()))['mae'] for p in model.staged_predict(x)]
            if count < model.tree_count_:
                model.shrink(count)
            prediction = model.predict_proba(xt) if classification else model.predict(xt)
            training_prediction = model.predict_proba(x) if classification else model.predict(x)
            model_path = workspace / 'model.cbm'
            model.save_model(str(model_path))
            details = {'effective_device': 'GPU', 'trees': count,
                       'feature_importance': model.feature_importances_.tolist(),
                       'iterations_run': ran}
        elif family in ('mlp', 'lstm', 'tcn'):
            tf = tensorflow_gpu(seed)
            with tf.device('/GPU:0'):
                entry = tf.keras.Input(shape=x.shape[1:])
                hidden = entry
                if family == 'mlp':
                    hidden = tf.keras.layers.Flatten()(hidden)
                    hidden = tf.keras.layers.Dense(params['width'], activation='relu')(hidden)
                elif family == 'lstm':
                    hidden = tf.keras.layers.LSTM(params['width'])(hidden)
                else:
                    for dilation in (1, 2, 4, 8):
                        hidden = tf.keras.layers.Conv1D(params['width'], 3, padding='causal',
                                                       dilation_rate=dilation, activation='relu')(hidden)
                    hidden = tf.keras.layers.GlobalAveragePooling1D()(hidden)
                hidden = tf.keras.layers.Dropout(params['dropout'])(hidden)
                output = tf.keras.layers.Dense(1)(hidden)
                model = tf.keras.Model(entry, output)
                model.compile(optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                              loss='mae' if spec['loss'] == 'reg:absoluteerror' else 'mse')
            # Keras creates CPU-only tf.data operations in fit. Do not inherit
            # the model-construction GPU scope into its data adapter or saving.
            recorded, best_weights, best_score, best_epoch = [], [None], [float('inf')], [0]

            class PriceEarlyStop(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    train_pred = np.asarray(self.model(x, training=False)).reshape(-1)
                    train_log = target.inverse(train_pred, train.cotton_close.to_numpy())
                    row = {'epoch': epoch + 1, 'training_price_mae': evaluate(train.cotton_close.to_numpy(),
                           train[f'target_return_{h}'].to_numpy(), train_log)['mae']}
                    if validation is not None:
                        value = metric(yv, np.asarray(self.model(xv, training=False)).reshape(-1))
                        row['validation_price_mae'] = value
                        if value < best_score[0]:
                            best_score[0], best_epoch[0], best_weights[0] = value, epoch + 1, self.model.get_weights()
                        if iterations is None and epoch + 1 - best_epoch[0] >= 25:
                            self.model.stop_training = True
                    recorded.append(row)
            model.fit(x, y, batch_size=params['batch_size'], epochs=limit, shuffle=False, verbose=0,
                      callbacks=[PriceEarlyStop()])
            if iterations is None and best_weights[0] is not None:
                model.set_weights(best_weights[0])
            count = best_epoch[0] if validation is not None and iterations is None else len(recorded)
            variable_devices = require_gpu_variables(model)
            tensor = model(xt, training=False)
            if 'GPU' not in tensor.device.upper():
                raise RuntimeError('TensorFlow computation is not on GPU')
            prediction = np.asarray(tensor).reshape(-1)
            training_prediction = np.asarray(model(x, training=False)).reshape(-1)
            model_path = workspace / 'model.keras'
            model.save(model_path)
            curves = {'epochs': recorded}
            details = {'effective_device': tensor.device, 'variable_devices': variable_devices,
                       'iterations_run': len(recorded), 'parameters': model.count_params()}
        elif family in ('ridge', 'elasticnet', 'logistic'):
            from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
            if family == 'logistic':
                model = LogisticRegression(C=params['C'], max_iter=3000, random_state=seed).fit(x, train_labels)
                def probabilities(values):
                    result = np.full((len(values), 3), 1e-9)
                    result[:, model.classes_.astype(int)] = model.predict_proba(values)
                    return result / result.sum(axis=1, keepdims=True)
                prediction, training_prediction = probabilities(xt), probabilities(x)
                linear = {'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(), 'classes': model.classes_.tolist()}
            else:
                model = (Ridge(alpha=params['alpha']) if family == 'ridge' else ElasticNet(
                    alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=10000, random_state=seed))
                model.fit(x,y,**({'sample_weight':weights} if transfer else {}))
                prediction, training_prediction = model.predict(xt), model.predict(x)
                linear = {'coef': model.coef_.tolist(), 'intercept': float(model.intercept_)}
            model_path.write_text(json.dumps(linear), encoding='utf-8')
            count, details = 1, {'effective_device': 'CPU_reference_explicit', **transfer_details}
        else:
            raise ValueError(f'Unsupported family: {family}')
    if classification:
        metrics = class_metrics(class_labels(test, h), prediction)
        train_metrics = class_metrics(train_labels, training_prediction)
        predictions = np.asarray(prediction).tolist()
    else:
        predictions = target.inverse(prediction, test.cotton_close.to_numpy()).tolist()
        training_log = target.inverse(training_prediction, train.cotton_close.to_numpy())
        metrics = (evaluate(test.cotton_close.to_numpy(), test[f'target_return_{h}'].to_numpy(), predictions)
                   if np.isfinite(test[f'target_return_{h}']).all() else {'role': 'unlabeled_live_prediction'})
        measured = train.asset.eq('cotton').to_numpy() if transfer else np.ones(len(train),dtype=bool)
        train_metrics = evaluate(train.cotton_close.to_numpy()[measured],
            train[f'target_return_{h}'].to_numpy()[measured],training_log[measured])
    metadata = {'spec': spec, 'processor': processor.as_dict(), 'target': target.as_dict(),
                'model_file': model_path.name, 'iterations': count, 'fit_cutoff': test.date.min().isoformat(),
                'last_fit_label': train.target_date_5.max().isoformat(), 'window': window}
    (workspace / 'adapter.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (workspace / 'curves.json').write_text(json.dumps(curves, allow_nan=False), encoding='utf-8')
    return {'predictions': predictions, 'origins': test.date.dt.strftime('%Y-%m-%d').tolist(),
            'metrics': metrics, 'training_metrics': train_metrics, 'iterations': count, 'details': details,
            'data_seconds': data_seconds, 'fit_and_save_seconds': time.monotonic() - started - data_seconds,
            'telemetry': telemetry.report(), 'unique_predictions': len(np.unique(np.asarray(predictions), axis=0))}
