"""Equal-budget, common-origin information tests before any wider search."""
import numpy as np

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.search import recipes
from cottonlens_ml.runtime_guard import require_colab_training

PROFILE = 'free-data-v1'


def groups(history, existing):
    result = dict(existing)
    result['cotton_no_volume'] = [c for c in FEATURE_NAMES
        if not c.startswith(('dxy_', 'wti_', 'cotton_dxy_', 'cotton_wti_', 'cotton_volume'))]
    expanded = list(existing['expanded'])
    result['expanded'] = [c for c in expanded if c != 'days_since_previous_observation']
    result['expanded_availability'] = expanded
    # Missing masks already belong to every train-fitted Preprocessor. Add explicit
    # causal time-since-observed channels without fitting on validation rows.
    for name in FEATURE_NAMES:
        age = f'age_{name}'
        observed = history[name].notna().to_numpy()
        last = np.maximum.accumulate(np.where(observed, np.arange(len(history)), -1))
        history[age] = np.where(last >= 0, np.arange(len(history)) - last, np.nan)
        result['expanded_availability'].append(age)
    return result


def admission(base, candidate):
    """Research priority only; never changes earlier fold selections."""
    if set(base) != set(candidate) or not base:
        raise ValueError('Matched nonempty folds required')
    b = np.asarray([base[k] for k in sorted(base)], dtype=float)
    c = np.asarray([candidate[k] for k in sorted(base)], dtype=float)
    if not np.isfinite([b, c]).all() or (b <= 0).any():
        raise ValueError('Finite positive baseline scores required')
    gain = float(1 - c.mean() / b.mean())
    wins = int((c < b).sum())
    return {'relative_inner_gain': gain, 'inner_fold_wins': wins,
            'admitted': bool(len(b) == 8 and gain >= .005 and wins >= 5),
            'evidence': 'reused_historical_research_priority_not_holdout'}


def run_ablation(experiment):
    require_colab_training()
    if experiment.identity.get('profile') != PROFILE:
        raise ValueError('Ablation requires a new free-data-v1 experiment')
    if (experiment.root / 'locked.json').exists():
        raise ValueError('Locked experiment is immutable')
    if read_record(experiment.root / 'diagnosis.json')['status'] != 'passed':
        raise ValueError('Diagnosis gate must pass first')
    names = list(experiment.identity['groups'])
    plan = {'groups': names, 'families': ['xgboost', 'catboost'], 'horizons': [1, 5],
            'candidates_per_unit': 16, 'folds': len(experiment.identity['split']['folds']),
            'decision_units': len(names) * 4 * len(experiment.identity['split']['folds']),
            'comparison': 'identical frozen origins; missingness masks in all groups'}
    freeze_record(experiment.root / 'ablation-plan.json', plan)
    scores = {}
    for group in names:
        namespace = f'ablate-{group}'
        for family in plan['families']:
            for horizon in plan['horizons']:
                key = f'{family}-t{horizon}'
                scores.setdefault(key, {})[group] = {}
                candidates = recipes(family, horizon, experiment.identity['groups'][group], 16)
                for fold in experiment.identity['split']['folds']:
                    decision = experiment.tune(family, horizon, fold, candidates, namespace)
                    scores[key][group][str(fold['fold'])] = decision['chosen']['score']
                    experiment.outer(decision['chosen'], fold, namespace)
    comparisons = {key: {group: admission(rows['base'], values)
                        for group, values in rows.items() if group != 'base'}
                   for key, rows in scores.items()}
    freeze_record(experiment.root / 'ablation-summary.json',
                  {'plan': plan, 'inner_scores': scores, 'comparisons': comparisons,
                   'outer_used_for_admission': False})
