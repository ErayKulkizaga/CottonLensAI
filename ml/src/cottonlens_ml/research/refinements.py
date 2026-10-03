"""Controlled information, history and cadence rounds; no silent cohort changes."""
from copy import deepcopy

import numpy as np

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.search import recipes, simplex_weights


def run_refinement(experiment, round_name):
    from cottonlens_ml.research.engine import compare
    if round_name not in ('D', 'E'):
        raise ValueError('Expected round D or E')
    report = compare(experiment)
    for h in (1, 5):
        candidates = [r for r in report['candidates'] if r['task'] == 'price' and r['horizon'] == h
                      and r['namespace'] in ('A', 'C')]
        if not candidates:
            raise ValueError('Complete base model searches before feature/history refinement')
        if round_name == 'D':
            groups = [experiment.identity['groups']['expanded']]
            for members in ([r for r in candidates if r['family'] in ('xgboost', 'catboost')],
                            [r for r in candidates if r['family'] in ('mlp', 'lstm', 'tcn')]):
                if not members:
                    raise ValueError('Round D requires completed tabular and sequence searches')
                best = min(members, key=lambda r: r['inner_score'])
                for features in groups:
                    namespace = 'D-' + content_id({'features': features})[:8]
                    specs = recipes(best['family'], h, features, 32, offset=1000)
                    for fold in experiment.identity['split']['folds']:
                        chosen = experiment.tune(best['family'], h, fold, specs, namespace)
                        experiment.outer(chosen['chosen'], fold, namespace)
        else:
            candidates += [r for r in report['candidates'] if r['task'] == 'price' and r['horizon'] == h
                           and r['namespace'].startswith('D-')]
            best = min(candidates, key=lambda r: r['inner_score'])
            if best['inner_score'] >= 1:
                freeze_record(experiment.root / f'refit-skipped-t{h}.json', {'reason': 'no_positive_inner_price_gain'})
                continue
            for fold in experiment.identity['split']['folds']:
                previous = read_record(experiment.root / 'decisions' / best['namespace'] /
                                       f'{best["family"]}-t{h}-fold{fold["fold"]}.json')
                base = previous['chosen']['recipe']
                specs = [{**base, 'years': years, 'cadence': cadence} for years in (3, 5, None) for cadence in (5, 21, 63, 126)]
                chosen = experiment.tune(best['family'], h, fold, specs, 'E')
                experiment.outer(chosen['chosen'], fold, 'E')
            ensemble(experiment, h, candidates)


def ensemble(experiment, horizon, candidates):
    """Fit weights only on inner OOF predictions; never on the outer outcomes."""
    selected = sorted(candidates, key=lambda r: r['inner_score'])[:3]
    if len(selected) < 2:
        return
    for fold in experiment.identity['split']['folds']:
        inner, outer, identities = [], [], []
        for family in selected:
            path = f'{family["family"]}-t{horizon}-fold{fold["fold"]}.json'
            chosen = read_record(experiment.root / 'decisions' / family['namespace'] / path)['chosen']
            predictions = np.mean([s['predictions'] for s in chosen['seeds']], axis=0)
            inner.append(predictions)
            rows = read_record(experiment.root / 'outer' / family['namespace'] / path)
            outer.append(rows['predictions'])
            identities.append({'family': family['family'], 'namespace': family['namespace'], 'decision': chosen})
        origins = chosen['seeds'][0]['origins']
        if any(read_record(experiment.root / 'decisions' / f['namespace'] /
               f'{f["family"]}-t{horizon}-fold{fold["fold"]}.json')['chosen']['seeds'][0]['origins'] != origins for f in selected):
            raise ValueError('Ensemble requires identical inner OOF origins')
        past = rows_at(experiment.history, origins)
        weights = simplex_weights(np.asarray(inner).T, past.cotton_close.to_numpy(), past[f'target_return_{horizon}'].to_numpy())
        prediction = np.asarray(outer).T @ weights
        from cottonlens_ml.evaluation import evaluate
        test = rows_at(experiment.history, fold['origins'])
        model = {'family': 'ensemble', 'horizon': horizon, 'task': 'price', 'features': experiment.identity['groups']['base'],
                 'params': {}, 'members': identities, 'weights': weights, 'seed': 42, 'cadence': 126}
        metrics = evaluate(test.cotton_close.to_numpy(), test[f'target_return_{horizon}'].to_numpy(), prediction)
        inner_log = np.asarray(inner).T @ weights
        inner_metrics = evaluate(past.cotton_close.to_numpy(), past[f'target_return_{horizon}'].to_numpy(), inner_log)
        naive = evaluate(past.cotton_close.to_numpy(), past[f'target_return_{horizon}'].to_numpy(), np.zeros(len(past)))
        freeze_record(experiment.root / 'outer' / 'E-ensemble' / f'ensemble-t{horizon}-fold{fold["fold"]}.json',
                      {'recipe': model, 'predictions': prediction.tolist(), 'origins': fold['origins'],
                       'fold': fold['fold'], 'metrics': metrics, 'inner_score': inner_metrics['mae'] / naive['mae'],
                       'weight_fit_role': 'inner_oof_only'})
        freeze_record(experiment.root / 'decisions' / 'E-ensemble' / f'ensemble-t{horizon}-fold{fold["fold"]}.json',
                      {'chosen': {'recipe': deepcopy(model), 'score': inner_metrics['mae'] / naive['mae'],
                                  'members': identities, 'weights': weights}, 'selection_used_outer': False})
