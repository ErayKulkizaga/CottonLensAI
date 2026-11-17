"""Sequential Optuna search with immutable proposal/result journals and fit reuse."""
import numpy as np

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.decision_cache import (
    execution_binding,
    read_bound,
    read_selection,
)
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import validate_inner_fold
from cottonlens_ml.research.search import SEEDS
from cottonlens_ml.runtime_guard import require_colab_training


def specification(trial, family, horizon, features):
    if family == 'xgboost':
        parameters = {
            'max_depth': trial.suggest_int('depth', 1, 8),
            'eta': trial.suggest_float('learning_rate', .005, .15, log=True),
            'min_child_weight': trial.suggest_float('child_weight', 1, 100, log=True),
            'alpha': (0. if trial.suggest_categorical('zero_l1', [True, False]) else
                      trial.suggest_float('l1', 1e-6, 10, log=True)),
            'lambda': trial.suggest_float('l2', 1e-3, 100, log=True),
            'subsample': trial.suggest_float('subsample', .6, 1.),
            'colsample_bytree': trial.suggest_float('colsample', .6, 1.)}
    elif family == 'catboost':
        parameters = {'depth': trial.suggest_int('depth', 4, 8),
                      'learning_rate': trial.suggest_float('learning_rate', .01, .15, log=True),
                      'l2_leaf_reg': trial.suggest_float('l2', 1, 100, log=True)}
    else:
        raise ValueError('Adaptive initial search supports tabular GPU families')
    return {'family': family, 'horizon': horizon, 'task': 'price', 'params': parameters,
            'features': list(features), 'seed': 42, 'window': 1, 'years': None, 'cadence': 126,
            'target': trial.suggest_categorical('target', ['scaled_log', 'price_delta']),
            'loss': trial.suggest_categorical('loss', ['reg:squarederror', 'reg:absoluteerror']),
            'hypothesis': 'free information with adaptive past-only tabular search'}


def tune(experiment, family, horizon, fold, features, count=128, namespace='adaptive-A'):
    validate_inner_fold(experiment.history, fold)
    import optuna
    request = {'operation': 'adaptive', 'family': family, 'horizon': horizon,
               'features': list(features), 'count': count, 'namespace': namespace,
               'search_policy': 1, 'seeds': list(SEEDS), 'optuna_version': optuna.__version__}
    binding = execution_binding(experiment, fold, request, selection=True)

    from cottonlens_ml.research.engine import complexity
    filename = f'{family}-t{horizon}-fold{fold["fold"]}'
    marker = experiment.root / 'decisions' / namespace / (filename + '.json')
    if marker.exists():
        return read_bound(marker, binding)
    journal = experiment.root / 'adaptive-trials' / filename
    journal_request = {k: v for k, v in request.items() if k not in ('namespace', 'count')}
    journal_binding = execution_binding(experiment, fold, journal_request, selection=True)
    journal_marker = journal/'binding.json'
    if journal_marker.exists():
        read_bound(journal_marker, journal_binding)
    elif journal.exists() and any(journal.iterdir()):
        raise ValueError('Unbound adaptive journal preserved; no automatic replay')
    else:
        freeze_record(journal_marker, {'execution_binding': journal_binding})
    attempt = experiment.root/'adaptive-attempts'/namespace/(filename+'.json')
    if attempt.exists():
        read_bound(attempt, binding)
    else:
        freeze_record(attempt, {'execution_binding': binding})
    # Small in-memory study; no SQLite on Drive. Proposals/results, not sampler
    # pickles, are durable. Seed each ask by ordinal so replay is deterministic.
    study = optuna.create_study(direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=16, n_warmup_steps=1))
    completed = []
    for index in range(count):
        result_path = journal / f'{index:04d}-result.json'
        proposal_path = journal / f'{index:04d}-proposal.json'
        study.sampler = optuna.samplers.TPESampler(seed=42 + index)
        trial = study.ask()
        spec = specification(trial, family, horizon, features)
        proposal = {'recipe': spec, 'params': trial.params,
                    'distributions': {k: optuna.distributions.distribution_to_json(v)
                                      for k, v in trial.distributions.items()}}
        freeze_record(proposal_path, proposal)  # identity mismatch fails closed
        if result_path.exists():
            saved = read_record(result_path)
            if (saved.get('execution_binding') != journal_binding
                    or saved.get('proposal_id') != content_id(proposal)
                    or saved.get('state') not in ('COMPLETE', 'PRUNED')):
                raise ValueError('Adaptive trial binding or state changed')
            for step, value in saved['intermediate'].items():
                trial.report(value, int(step))
        else:
            intermediate = {}
            def progress(step, value, intermediate=intermediate, trial=trial):
                intermediate[str(step)] = value
                trial.report(value, step)
                if step >= 1 and trial.should_prune():
                    raise optuna.TrialPruned()
            try:
                result = experiment.inner(spec, fold, progress=progress)
                saved = {'state': 'COMPLETE', 'result': result, 'intermediate': intermediate}
            except optuna.TrialPruned:
                saved = {'state': 'PRUNED', 'intermediate': intermediate}
            saved.update(execution_binding=journal_binding, proposal_id=content_id(proposal))
            freeze_record(result_path, saved)
        if saved['state'] == 'COMPLETE':
            study.tell(trial, saved['result']['score'])
            completed.append(saved['result'])
        else:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        print(f'STAGE adaptive {family} T+{horizon} fold={fold["fold"]} '
              f'candidate={index + 1}/{count} state={saved["state"]}', flush=True)
    if not completed:
        raise ValueError('No fully completed adaptive candidates')
    confirmed = []
    for candidate in sorted(completed, key=lambda r: (r['score'], complexity(r['recipe'])))[:5]:
        seeds = [candidate if seed == 42 else experiment.inner(
            {**candidate['recipe'], 'seed': seed}, fold, repeat='new_seed') for seed in SEEDS]
        confirmed.append({'score': float(np.mean([r['score'] for r in seeds])),
                          'seeds': seeds, 'recipe': candidate['recipe']})
    chosen = min(confirmed, key=lambda r: (r['score'], complexity(r['recipe'])))
    decision = {'chosen': chosen, 'fold': fold['fold'], 'selection_used_outer': False,
                'trials': [{'recipe': r['recipe'], 'score': r['score'], 'records': r['records']}
                           for r in completed], 'pruned_count': count - len(completed)}
    decision.update(execution_request=request, execution_binding=binding)
    freeze_record(marker, decision)
    return decision


def search_tabular(experiment, extension=0):
    require_colab_training()
    if (experiment.root / 'locked.json').exists():
        raise ValueError('Experiment is locked')
    if not experiment.identity.get('publication_sources'):
        raise ValueError('Wider search requires reviewed new information; do not repeat base-data search')
    if read_record(experiment.root / 'diagnosis.json')['status'] != 'passed':
        raise ValueError('Diagnosis gate must pass')
    from cottonlens_ml.research.ablation import read_summary
    ablation = read_summary(experiment)
    if not any(row['admitted'] for groups in ablation['comparisons'].values() for row in groups.values()):
        raise ValueError('No information group passed inner-only research admission; diagnose or add a new source')
    if extension < 0:
        raise ValueError('Extension must be nonnegative')
    namespace = 'adaptive-A' if extension == 0 else f'adaptive-A-extension-{extension}'
    for family in ('xgboost', 'catboost'):
        for horizon in (1, 5):
            for fold in experiment.identity['split']['folds']:
                filename = f'{family}-t{horizon}-fold{fold["fold"]}.json'
                if extension:
                    prior_namespace = 'adaptive-A' if extension == 1 else f'adaptive-A-extension-{extension - 1}'
                    previous = read_selection(experiment,
                        experiment.root/'decisions'/prior_namespace/filename,
                        fold, prior_namespace, family, horizon)
                    gains = []
                    for number in range(max(1, extension - 2), extension):
                        older_namespace = f'adaptive-A-extension-{number}'
                        older = read_selection(experiment,
                            experiment.root/'decisions'/older_namespace/filename,
                            fold, older_namespace, family, horizon)
                        receipt = experiment.root / 'extensions' / f'adaptive-A-extension-{number}' / filename
                        if receipt.exists():
                            predecessor = 'adaptive-A' if number == 1 else f'adaptive-A-extension-{number-1}'
                            earlier = read_selection(experiment,
                                experiment.root/'decisions'/predecessor/filename,
                                fold, predecessor, family, horizon)
                            gain_request = {'operation': 'adaptive-gain',
                                            'previous': content_id(earlier), 'current': content_id(older)}
                            gains.append(read_bound(receipt, execution_binding(
                                experiment, fold, gain_request, selection=True))['relative_inner_gain'])
                        elif older.get('extension_skipped'):
                            gains.append(0.)
                        else:
                            raise ValueError('Previous extension is incomplete; resume it first')
                    stopped = previous['chosen']['score'] >= 1 or (len(gains) == 2 and all(g < .005 for g in gains))
                    if stopped:
                        request = {'operation': 'adaptive-skip', 'family': family, 'horizon': horizon,
                                   'namespace': namespace, 'previous': content_id(previous)}
                        freeze_record(experiment.root / 'decisions' / namespace / filename,
                            {**previous, 'extension_relative_gain': 0., 'extension_skipped': True,
                             'execution_request': request,
                             'execution_binding': execution_binding(experiment, fold, request, selection=True),
                             'reason': 'no_positive_inner_gain_or_two_round_plateau'})
                        outer_request = {'operation': 'outer', 'chosen': previous['chosen'],
                                         'namespace': prior_namespace, 'repeat': None}
                        prior_outer = read_bound(experiment.root/'outer'/prior_namespace/filename,
                            execution_binding(experiment, fold, outer_request, selection=False))
                        outer_request['namespace'] = namespace
                        prior_outer = {**prior_outer, 'execution_binding': execution_binding(
                            experiment, fold, outer_request, selection=False)}
                        freeze_record(experiment.root / 'outer' / namespace / filename, prior_outer)
                        continue
                    features = previous['chosen']['recipe']['features']
                    # Keep the original search space/features for deterministic replay
                    # of all earlier proposals; only the trial budget is extended.
                    chosen = tune(experiment, family, horizon, fold, features,
                                  count=128 * (extension + 1), namespace=namespace)
                    gain = 1 - chosen['chosen']['score'] / previous['chosen']['score']
                    gain_request = {'operation': 'adaptive-gain',
                                    'previous': content_id(previous), 'current': content_id(chosen)}
                    freeze_record(experiment.root / 'extensions' / namespace / filename,
                                  {'relative_inner_gain': gain, 'outer_used': False,
                                   'execution_binding': execution_binding(
                                       experiment, fold, gain_request, selection=True)})
                    experiment.outer(chosen['chosen'], fold, namespace)
                    continue
                # Selection is local to this fold's past inner predictions, never
                # the full-history admission aggregate or the outer results.
                decisions = [read_selection(experiment,
                    experiment.root/'decisions'/f'ablate-{group}'/filename,
                    fold, f'ablate-{group}', family, horizon)['chosen']
                    for group in experiment.identity['groups']]
                selected = min(decisions, key=lambda r: (r['score'], len(r['recipe']['features'])))
                chosen = tune(experiment, family, horizon, fold, selected['recipe']['features'])
                experiment.outer(chosen['chosen'], fold, namespace)
