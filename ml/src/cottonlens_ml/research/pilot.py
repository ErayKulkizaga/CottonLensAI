"""Bounded new-information pilots using the existing Experiment fit/ledger path."""
import math
import time
from pathlib import Path

import pandas as pd

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.search import recipes
from cottonlens_ml.runtime_guard import require_colab_training

VERSION = 'information-pilot-v1'


def pilot_plan(identity, history=None):
    """No fit, file write, outer-result selection, or automatic seed expansion."""
    sources = []
    for source in identity.get('publication_sources', []):
        manifest = source['manifest']
        if manifest['kind'] not in ('ams', 'export_sales'):
            continue
        if (manifest.get('vintage_policy') != 'as_published'
                or manifest.get('usage', {}).get('cost_tl') != 0
                or manifest.get('usage', {}).get('research_allowed') is not True):
            raise ValueError('Verified zero-cost as-published packages required')
        sources.append(manifest['kind'])
    if len(sources) != len(set(sources)):
        raise ValueError('One frozen package per source required')
    plan = {'version': VERSION, 'experiment_identity': content_id(identity),
        'evidence': 'seen_historical_short_cohort_pilot_not_gate_evidence',
        'seed': 42, 'candidates_per_unit': 4, 'families': ['xgboost', 'catboost'],
        'horizons': [5], 'max_outer_blocks': 4, 'outer_used_for_selection': False,
        'automatic_seed_confirmation': False, 'sources': sorted(sources)}
    if not sources:
        return {**plan, 'status': 'blocked', 'jobs': [], 'core_fit_budget': 0,
            'outer_fit_budget': 0,
            'blocker': 'No admitted AMS/Export Sales package; do not rerun market-only ablation'}
    if identity.get('profile') != 'free-data-v1' or history is None:
        raise ValueError('Frozen free-data profile and history required for an exact fit budget')
    groups = ['base', *(f'source_{kind}' for kind in sorted(sources))]
    if any(group not in identity['groups'] for group in groups):
        raise ValueError('Frozen feature group missing')
    folds = identity['split']['folds'][:4]
    if not folds or identity['split'].get('audit_used'):
        raise ValueError('Pre-audit frozen historical folds required')
    if any(fold['year'] >= 2024 for fold in folds):
        raise ValueError('Seen audit cannot enter the pilot')
    ordinal = history.set_index('date').cotton_session_index
    jobs, core = [], 0
    for group in groups:
        for family in plan['families']:
            candidates = recipes(family, 5, identity['groups'][group], 4)
            for fold in folds:
                if len(fold['inner']) != 3:
                    raise ValueError('Three frozen inner validation blocks required')
                # One early stop plus actual ordinal-based refit chunks per inner block.
                fit_count = 0
                for block in fold['inner']:
                    values = ordinal.loc[pd.to_datetime(block['origins'])]
                    fit_count += 1 + ((values - values.iloc[0]) // 126).nunique()
                for index, recipe in enumerate(candidates):
                    job = {'group': group, 'family': family, 'horizon': 5,
                           'fold': fold['fold'], 'candidate': index + 1, 'recipe': recipe,
                           'fit_budget': int(fit_count)}
                    jobs.append({**job, 'job_id': content_id(job)})
                    core += fit_count
    return {**plan, 'status': 'ready', 'groups': groups,
        'folds': [fold['fold'] for fold in folds], 'fold_years': [fold['year'] for fold in folds],
        'jobs': jobs, 'decision_units': len(groups) * 2 * len(folds),
        'core_fit_budget': int(core), 'outer_fit_budget': len(groups) * 2 * len(folds),
        'budget_includes_baselines': False, 'bootstrap_and_release': 'not run in the pilot',
        'pause_policy': 'deadline checked before each new fit; active fit may exceed the limit'}


def run_pilot(experiment, *, max_minutes=60., clock=time.monotonic):
    require_colab_training()
    if not math.isfinite(max_minutes) or not 0 < max_minutes <= 120:
        raise ValueError('Pilot duration must be between 0 and 120 minutes')
    if (experiment.root / 'locked.json').exists():
        raise ValueError('Locked experiment is immutable')
    plan = pilot_plan(experiment.identity, experiment.history)
    if plan['status'] != 'ready':
        raise ValueError(plan['blocker'])
    if read_record(experiment.root / 'diagnosis.json')['status'] != 'passed':
        raise ValueError('Diagnostic controls must pass before the pilot')
    folder = experiment.root / 'pilot'
    freeze_record(folder / 'plan.json', plan)
    deadline = clock() + max_minutes * 60
    previous = getattr(experiment, 'before_compute', None)

    def before_compute():
        if clock() >= deadline:
            raise FitBudgetReached('Planned pause before the next fit')
        if previous is not None:
            previous()

    experiment.before_compute = before_compute
    folds = {f['fold']: f for f in experiment.identity['split']['folds']}
    completed, choices = 0, {}
    try:
        for job in plan['jobs']:
            path = folder / 'candidates' / (job['job_id'] + '.json')
            if path.exists():
                result = read_record(path)
                if result['job'] != job or result['plan_id'] != content_id(plan):
                    raise ValueError('Cached pilot identity mismatch')
                print(f'PILOT cached candidate {job["job_id"][:12]}', flush=True)
            else:
                result = {'plan_id': content_id(plan), 'job': job,
                          'inner': experiment.inner(job['recipe'], folds[job['fold']])}
                freeze_record(path, result)
            key = job['group'], job['family'], job['fold']
            choices.setdefault(key, []).append(result['inner'])
            completed += 1
            print(f'PILOT saved={completed}/{len(plan["jobs"])} {job["group"]} '
                  f'{job["family"]} T+5 fold={job["fold"]} candidate={job["candidate"]}/4', flush=True)
        outer_saved = 0
        for (group, family, fold), candidates in choices.items():
            best = min(candidates, key=lambda c: (c['score'],
                c['recipe']['params'].get('max_depth', c['recipe']['params'].get('depth', 0)),
                content_id(c['recipe'])))
            chosen = {'recipe': best['recipe'], 'score': best['score'], 'seeds': [best]}
            namespace = 'pilot-' + group
            freeze_record(experiment.root / 'decisions' / namespace / f'{family}-t5-fold{fold}.json',
                          {'chosen': chosen, 'fold': fold, 'selection_used_outer': False})
            experiment.outer(chosen, folds[fold], namespace)
            outer_saved += 1
        comparisons = {}
        for family in plan['families']:
            base = [min(c['score'] for c in choices[('base', family, f)]) for f in plan['folds']]
            for group in plan['groups'][1:]:
                values = [min(c['score'] for c in choices[(group, family, f)]) for f in plan['folds']]
                gain = 1 - sum(values) / sum(base)
                wins = sum(c < b for c, b in zip(values, base, strict=True))
                comparisons[f'{family}:{group}'] = {'relative_inner_gain': gain,
                    'inner_block_wins': wins, 'total_blocks': len(base),
                    'provisional_priority': gain >= .005 and wins > len(base) / 2,
                    'full_eight_fold_gate_evaluated': False}
        result = {'status': 'complete', 'plan_id': content_id(plan), 'saved_candidates': completed,
                  'outer_units': outer_saved, 'comparisons': comparisons, 'gate_pass_claimed': False}
        freeze_record(folder / 'summary.json', result)
        return result
    except FitBudgetReached:
        print('PILOT planned_pause: completed fits remain saved; rerun the same pilot to resume.', flush=True)
        return {'status': 'planned_pause', 'saved_candidates': completed,
                'total_candidates': len(plan['jobs']), 'gate_pass_claimed': False}
    finally:
        experiment.before_compute = previous


def saved_pilot_status(root):
    """Small markers only; does not verify fitted payloads or claim completion."""
    folder = Path(root) / 'pilot'
    plan = read_record(folder / 'plan.json') if (folder / 'plan.json').exists() else None
    return {'prepared': plan is not None,
            'saved_candidates': len(list((folder / 'candidates').glob('*.json'))),
            'total_candidates': len(plan['jobs']) if plan else 0,
            'summary_present': (folder / 'summary.json').exists(),
            'payloads_verified': False}
