"""Small Colab-CPU statistical references with observed-only state updates."""
import json
import time
import warnings

import numpy as np

from cottonlens_ml.cohort import frame_identity
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.runtime_guard import require_training

ORDERS = ((1,1,0),(0,1,1))


class InvalidStatisticalFit(ValueError):
    """A recorded numerical failure, never an implicit Naive fallback."""


def fit_predict(history, train, test, spec, workspace):
    require_training(spec['family'],spec.get('device','cpu'))
    started = time.monotonic()
    if (spec['device']!='cpu' or spec['family'] not in ('naive','drift','arima')
            or spec['horizon'] not in (1,5) or spec['cadence']!=21 or spec['years'] is not None):
        raise ValueError('Frozen statistical recipe required')
    h = spec['horizon']
    cutoff = test.date.min()
    preceding = history.loc[history.date<cutoff]
    if len(preceding)<5:
        raise ValueError('Five-observation purge context required')
    if (history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or test.date.duplicated().any() or not test.date.is_monotonic_increasing
            or len(train)<126 or not train.date.lt(preceding.date.iloc[-5]).all()
            or not train.target_date_5.lt(cutoff).all()):
        raise ValueError('Statistical fit violates chronology/purge/label maturity')
    series = history.loc[history.date.between(train.date.min(),train.date.max()),['date','cotton_close']]
    if (not series.date.tolist()==train.date.tolist() or not np.isfinite(series.cotton_close).all()
            or (series.cotton_close<=0).any()):
        raise ValueError('Contiguous positive recorded observations required; no time compression')
    expected = history.set_index('date').loc[test.date,'cotton_close'].to_numpy()
    if not np.array_equal(expected,test.cotton_close):
        raise ValueError('Test prices differ from frozen source observations')
    logs = np.log(series.cotton_close.to_numpy())
    center = float(logs.mean())
    drift = float(np.diff(logs).mean()) if spec['family']=='drift' else 0.
    fitted = None
    optimizer = None
    data_seconds = time.monotonic()-started
    if spec['family']=='arima':
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from statsmodels.tsa.arima.model import ARIMA
        if spec['params']!={'order':spec['params'].get('order'),'trend':'n','maxiter':100} or tuple(spec['params']['order']) not in ORDERS:
            raise ValueError('Only two frozen ARIMA orders, no automatic search')
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                fitted = ARIMA(logs-center,order=tuple(spec['params']['order']),trend='n',
                    enforce_stationarity=True,enforce_invertibility=True).fit(method='statespace',
                        method_kwargs={'maxiter':100,'disp':0},cov_type='none')
            optimizer = {k:v.item() if isinstance(v,np.generic) else v for k,v in fitted.mle_retvals.items()
                if k in ('converged','iterations','fcalls','warnflag')}
            if (not fitted.mle_retvals.get('converged',False) or not np.isfinite(fitted.params).all()
                    or any(issubclass(w.category,ConvergenceWarning) for w in caught)):
                raise InvalidStatisticalFit('ARIMA did not converge; candidate is not complete')
        except (np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
            raise InvalidStatisticalFit('ARIMA numerical failure') from exc
    elif spec['params']:
        raise ValueError('Analytic reference parameters must be empty')
    model = {'family':spec['family'],'center_training_only':center,'daily_log_drift':drift,
        'params':None if fitted is None else np.asarray(fitted.params).tolist(),
        'training_price_identity':frame_identity(series,['date','cotton_close']),
        'training_observations':len(series),'first_training_date':series.date.min().isoformat(),
        'last_training_date':series.date.max().isoformat(),'optimizer':optimizer,
        'state_update_policy':'observed prices through origin only, no parameter refit'}
    if fitted is not None:
        model['last_filtered_state'] = fitted.filter_results.filtered_state[:,-1].tolist()
    predictions = []
    cursor = series.date.max()
    for origin in test.itertuples():
        if fitted is not None:
            updates = history.loc[(history.date>cursor)&(history.date<=origin.date),'cotton_close'].to_numpy()
            if not len(updates) or not np.isfinite(updates).all() or (updates<=0).any():
                raise ValueError('Observed-only state update prices required')
            try:
                fitted = fitted.extend(np.log(updates)-center)
                forecast = float(np.asarray(fitted.forecast(steps=h))[-1])+center
            except (np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
                raise InvalidStatisticalFit('ARIMA forecast numerical failure') from exc
            predicted = forecast-float(np.log(origin.cotton_close))
        else:
            predicted = h*drift
        if not np.isfinite(predicted) or not np.isfinite(np.exp(predicted)):
            raise InvalidStatisticalFit('Nonfinite statistical forecast')
        predictions.append(float(predicted)); cursor = origin.date
    (workspace/'statistical-model.json').write_text(json.dumps(model,allow_nan=False),encoding='utf-8')
    (workspace/'adapter.json').write_text(json.dumps({'spec':spec,'iterations':1,
        'fit_cutoff':cutoff.isoformat(),'last_fit_label':train.target_date_5.max().isoformat(),
        'model_file':'statistical-model.json','input':'recorded Cotton log prices',
        'target':'log return relative to observed origin close'},allow_nan=False),encoding='utf-8')
    return {'predictions':predictions,'origins':test.date.dt.strftime('%Y-%m-%d').tolist(),'iterations':1,
        'metrics':evaluate(test.cotton_close.to_numpy(),test[f'target_return_{h}'].to_numpy(),predictions),
        'data_seconds':data_seconds,'fit_and_save_seconds':time.monotonic()-started-data_seconds,
        'details':{'effective_device':'cpu','thread_limit':2,'optimizer':optimizer,
            'training_observations':len(series),'state_updates_refit_parameters':False},
        'telemetry':{'device':'cpu','thread_limit':2,'gpu_utilization_claimed':False}}
