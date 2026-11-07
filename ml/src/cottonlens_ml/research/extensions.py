"""Causal legacy search extensions with immutable per-fold stopping receipts."""
import numpy as np

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.decision_cache import (
    execution_binding,
    read_bound,
    read_selection,
)
from cottonlens_ml.research.free_rounds import prior_choices
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.protocol import validate_inner_fold
from cottonlens_ml.research.refinements import publish_choice
from cottonlens_ml.research.search import BUDGETS, next_round, recipes


def namespace(round_name, number):
    return f'{round_name}-extension-{number}-local-v2'


def choices(experiment, fold, name):
    return {h: prior_choices(experiment, h, fold, [name]) for h in (1, 5)}


def best_score(pool):
    scores = [c['decision']['score'] for rows in pool.values() for c in rows]
    if not np.isfinite(scores).all() or min(scores) < 0:
        raise ValueError('Finite nonnegative inner scores required')
    return float(np.mean([min(c['decision']['score'] for c in pool[h]) for h in (1, 5)]))


def family_order(pool):
    families = {c['family'] for rows in pool.values() for c in rows}
    scores = {f: np.mean([min(c['decision']['score'] for c in pool[h] if c['family'] == f)
                         for h in (1, 5)]) for f in families}
    return sorted(families, key=lambda f: (scores[f], f))[:2]


def candidate_contract(experiment, fold, name, pool, round_name, index=0):
    for h, rows in pool.items():
        for row in rows:
            family = row['family']
            expected = recipes(family, h, experiment.identity['groups']['base'],
                64 if index else BUDGETS[round_name][family],
                offset=BUDGETS[round_name][family]+(index-1)*64 if index else 0)
            record = read_selection(experiment, experiment.root/'decisions'/name/
                f'{family}-t{h}-fold{fold["fold"]}.json', fold, name, family, h)
            if record['execution_request'].get('candidates') != expected:
                raise ValueError('Extension candidate/feature contract changed')


def extend(experiment, round_name, number):
    if round_name not in ('A', 'C') or type(number) is not int or number < 1:
        raise ValueError('Positive A/C extension required')
    if (experiment.root/'locked.json').exists():
        raise ValueError('Locked experiment is immutable')
    for fold in experiment.identity['split']['folds']:
        validate_inner_fold(experiment.history, fold)
        pool = choices(experiment, fold, round_name)
        for h in (1, 5):
            if {c['family'] for c in pool[h]} != set(BUDGETS[round_name]):
                raise ValueError('Complete paired initial families required in this fold')
        candidate_contract(experiment, fold, round_name, pool, round_name)
        initial = best_score(pool)
        history = []
        for index in range(1, number+1):
            before = best_score(pool)
            gains = [1-initial, *[r['relative_improvement'] for r in history]]
            active = (before > 0 and not any(r['action'] == 'STOP' for r in history)
                      and next_round(gains) == 'extend_top_two_families_64_each')
            families = family_order(pool) if active else []
            request = {'operation': 'fold-local-extension-v2', 'round': round_name, 'index': index,
                       'sources': content_id(pool), 'previous': [content_id(r) for r in history],
                       'candidates_per_family': 64}
            binding = execution_binding(experiment, fold, request, selection=True)
            marker = experiment.root/'extensions/fold-local-v2'/f'{round_name}-{index}-fold{fold["fold"]}.json'
            saved = read_bound(marker, binding) if marker.exists() else None
            if saved is None and index != number:
                raise ValueError('Previous fold extension incomplete; resume its exact index first')
            name = namespace(round_name, index)
            if active:
                if saved is None:
                    for family in families:
                        for h in (1, 5):
                            candidates = recipes(family, h, experiment.identity['groups']['base'], 64,
                                offset=BUDGETS[round_name][family]+(index-1)*64)
                            selected = experiment.tune(family, h, fold, candidates, name)
                            experiment.outer(selected['chosen'], fold, name)
                added = choices(experiment, fold, name)
                candidate_contract(experiment, fold, name, added, round_name, index)
                for h in (1, 5):
                    if (len(added[h]) != len(families)
                            or {c['family'] for c in added[h]} != set(families)):
                        raise ValueError('Extension family/horizon coverage changed')
                    pool[h] += added[h]
            after = best_score(pool)
            result = {'execution_request': request, 'execution_binding': binding,
                      'action': 'CONTINUE' if active else 'STOP', 'families': families,
                      'baseline_inner_score': before, 'new_inner_score': after,
                      'relative_improvement': max(0., (before-after)/before) if before else 0.,
                      'selection_used_outer': False}
            if saved is not None and saved != result:
                raise ValueError('Extension outcome disagrees with bound selection records')
            freeze_record(marker, result)
            history.append(result)
            policy = f'{round_name}-extension-{index}-policy-v2'
            for h in (1, 5):
                selected = min(pool[h], key=lambda c: (c['decision']['score'], c['family'], c['namespace']))
                publish_choice(experiment, h, fold, selected, policy, cached_only=True,
                               selection_policy='fold-local-search-extension-v2')
