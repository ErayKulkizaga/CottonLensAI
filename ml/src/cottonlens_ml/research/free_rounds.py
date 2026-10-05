"""Past-only sequence, direction and refinement rounds for the free-data profile."""
from copy import deepcopy

import numpy as np

from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.search import recipes, simplex_weights
from cottonlens_ml.runtime_guard import require_colab_training


def prior_choices(experiment, horizon, fold, namespaces):
    """Never rank with other folds' outcomes or full-history report aggregates."""
    choices = []
    for namespace in namespaces:
        for path in sorted((experiment.root / 'decisions' / namespace).glob(f'*-t{horizon}-fold{fold["fold"]}.json')):
            chosen = read_record(path)['chosen']
            if chosen['recipe'].get('task', 'price') == 'price':
                choices.append({'namespace': namespace, 'decision': chosen, 'family': chosen['recipe']['family']})
    if not choices:
        raise ValueError('Complete past-only tabular selection before this round')
    return sorted(choices, key=lambda r: (r['decision']['score'], len(r['decision']['recipe']['features']), r['family']))


def sequence_smoke(experiment):
    from cottonlens_ml.config import FEATURE_NAMES
    from cottonlens_ml.research.diagnostics import synthetic_control
    from cottonlens_ml.research.models import fit_predict
    marker = experiment.root / 'sequence-smoke.json'
    if marker.exists():
        if read_record(marker)['status'] != 'passed':
            raise ValueError('Sequence smoke failed; fix and create a new experiment')
        return
    receipts = []
    for family in ('mlp', 'lstm', 'tcn'):
        frame, train, validation, test = synthetic_control('known_signal')
        spec = recipes(family, 1, FEATURE_NAMES, 1)[0]
        spec.update(window=20, loss='reg:squarederror', target='scaled_log',
                    params={'width': 32, 'dropout': 0., 'batch_size': 64, 'learning_rate': .003})
        train = train.iloc[19:]
        record = experiment.ledger.run({'role': 'free_sequence_gpu_smoke', 'recipe': spec},
            lambda workspace, s=spec, f=frame, tr=train, va=validation, te=test:
                fit_predict(f, tr, va, te, s, workspace, iterations=40))
        baseline = evaluate(test.cotton_close.to_numpy(), test.target_return_1.to_numpy(), np.zeros(len(test)))
        gain = 1 - record['result']['metrics']['mae'] / baseline['mae']
        receipts.append({'family': family, 'synthetic_gain': gain, 'record': record['experiment_id']})
    status = 'passed' if all(r['synthetic_gain'] > 0 for r in receipts) else 'failed'
    freeze_record(marker, {'status': status, 'receipts': receipts, 'market_evidence': False})
    if status != 'passed':
        raise ValueError('Sequence GPU learning smoke failed')


def run(experiment, round_name):
    require_colab_training()
    if (experiment.root / 'locked.json').exists():
        raise ValueError('Experiment is locked')
    if read_record(experiment.root / 'diagnosis.json')['status'] != 'passed':
        raise ValueError('Diagnosis gate must pass')
    read_record(experiment.root / 'ablation-summary.json')
    if round_name not in ('B', 'C', 'E'):
        raise ValueError('Use ablate for data groups; available free rounds are A/B/C/E')
    if round_name == 'C':
        sequence_smoke(experiment)
    tabular_names = [p.name for p in sorted((experiment.root / 'decisions').glob('adaptive-A*')) if p.is_dir()]
    for horizon in (1, 5):
        for fold in experiment.identity['split']['folds']:
            prior = prior_choices(experiment, horizon, fold, tabular_names)
            features = prior[0]['decision']['recipe']['features']
            if round_name == 'B':
                for family in ('logistic', 'xgboost', 'catboost'):
                    specs = recipes(family, horizon, features, 32, task='direction')
                    selected = experiment.tune(family, horizon, fold, specs, 'free-B')
                    experiment.outer(selected['chosen'], fold, 'free-B')
            elif round_name == 'C':
                initial = []
                for family in ('mlp', 'lstm', 'tcn'):
                    specs = recipes(family, horizon, features, 24)
                    selected = experiment.tune(family, horizon, fold, specs, 'free-C')
                    experiment.outer(selected['chosen'], fold, 'free-C')
                    initial.append(selected['chosen'])
                # Allocation uses only this fold's inner scores, not future fold results.
                for chosen in sorted(initial, key=lambda r: r['score'])[:2]:
                    family = chosen['recipe']['family']
                    specs = recipes(family, horizon, features, 48, offset=24)
                    selected = experiment.tune(family, horizon, fold, specs, 'free-C-extension')
                    experiment.outer(selected['chosen'], fold, 'free-C-extension')
            else:
                prior = prior_choices(experiment, horizon, fold,
                    [*tabular_names, 'free-C', 'free-C-extension'])
                chosen = prior[0]['decision']
                if chosen['score'] < 1:
                    base = chosen['recipe']
                    specs = [{**base, 'years': years, 'cadence': cadence}
                             for years in (3, 5, None) for cadence in (5, 21, 63, 126)]
                    selected = experiment.tune(base['family'], horizon, fold, specs, 'free-E')
                    experiment.outer(selected['chosen'], fold, 'free-E')
                    prior.append({'family': base['family'], 'namespace': 'free-E', 'decision': selected['chosen']})
                    prior.sort(key=lambda r: r['decision']['score'])
                else:
                    freeze_record(experiment.root / 'skipped' / f'free-E-t{horizon}-fold{fold["fold"]}.json',
                                  {'reason': 'no_positive_inner_gain', 'outer_used': False})
                ensemble_fold(experiment, horizon, fold, prior)


def ensemble_fold(experiment, horizon, fold, ranked):
    """One member per family, selected by this fold's inner evidence only."""
    selected, families = [], set()
    for candidate in ranked:
        if candidate['family'] not in families:
            selected.append(candidate)
            families.add(candidate['family'])
        if len(selected) == 3:
            break
    if len(selected) < 2:
        return
    inner, outer = [], []
    origins = selected[0]['decision']['seeds'][0]['origins']
    for member in selected:
        chosen = member['decision']
        if any(s['origins'] != origins for s in chosen['seeds']):
            raise ValueError('Ensemble requires identical past OOF origins')
        inner.append(np.mean([s['predictions'] for s in chosen['seeds']], axis=0))
        observed = experiment.outer(chosen, fold, member['namespace'])
        if observed['origins'] != fold['origins']:
            raise ValueError('Ensemble outer origins differ')
        outer.append(observed['predictions'])
    past = rows_at(experiment.history, origins)
    weights = simplex_weights(np.asarray(inner).T, past.cotton_close.to_numpy(), past[f'target_return_{horizon}'].to_numpy())
    predicted_inner = np.asarray(inner).T @ weights
    baseline = evaluate(past.cotton_close.to_numpy(), past[f'target_return_{horizon}'].to_numpy(), np.zeros(len(past)))
    score = evaluate(past.cotton_close.to_numpy(), past[f'target_return_{horizon}'].to_numpy(), predicted_inner)['mae'] / baseline['mae']
    model = {'family': 'ensemble', 'horizon': horizon, 'task': 'price', 'features': list(dict.fromkeys(
        name for m in selected for name in m['decision']['recipe']['features'])), 'params': {}, 'seed': 42, 'cadence': 126}
    chosen = {'recipe': model, 'score': score, 'members': deepcopy(selected), 'weights': weights}
    marker = f'ensemble-t{horizon}-fold{fold["fold"]}.json'
    freeze_record(experiment.root / 'decisions/free-E-ensemble' / marker,
                  {'chosen': chosen, 'selection_used_outer': False, 'family_selection': 'this_fold_inner_only'})
    test = rows_at(experiment.history, fold['origins'])
    prediction = np.asarray(outer).T @ weights
    from cottonlens_ml.research.uncertainty import for_candidate
    freeze_record(experiment.root / 'outer/free-E-ensemble' / marker,
                  {'recipe': model, 'predictions': prediction.tolist(), 'origins': fold['origins'],
                   'fold': fold['fold'], 'inner_score': score,
                   'metrics': evaluate(test.cotton_close.to_numpy(), test[f'target_return_{horizon}'].to_numpy(), prediction),
                   'uncertainty': for_candidate(experiment.history, chosen, fold['origins'], prediction),
                   'weight_fit_role': 'past_inner_oof_only'})
