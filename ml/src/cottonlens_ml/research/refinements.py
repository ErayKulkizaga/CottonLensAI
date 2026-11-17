"""Fold-local information/history/cadence selection; preserve legacy evidence."""
from copy import deepcopy

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.decision_cache import execution_binding, read_bound
from cottonlens_ml.research.free_rounds import ensemble_fold, prior_choices
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.protocol import validate_inner_fold, verified_outer
from cottonlens_ml.research.search import recipes


def publish_choice(experiment, horizon, fold, selected, namespace, *,
                   cached_only=False, selection_policy='fold-local-refinement-v2'):
    """One-member ensemble: fixed selection rule, actual native leaves retained."""
    decision = deepcopy(selected['decision'])
    chosen = {
        'recipe': {'family': 'ensemble', 'horizon': horizon, 'task': 'price',
                   'features': decision['recipe']['features'], 'params': {},
                   'seed': 42, 'cadence': 126},
        'score': decision['score'], 'members': [deepcopy(selected)], 'weights': [1.],
        'seeds': decision['seeds'], 'selection_policy': selection_policy}
    request = {'operation': 'fold-choice', 'namespace': namespace,
               'family': 'ensemble', 'horizon': horizon, 'selected': content_id(selected)}
    freeze_record(experiment.root/'decisions'/namespace/f'ensemble-t{horizon}-fold{fold["fold"]}.json',
                  {'chosen': chosen, 'selection_used_outer': False,
                   'execution_request': request,
                   'execution_binding': execution_binding(experiment, fold, request, selection=True)})
    if cached_only:
        source_namespace = selected['namespace']
        family = decision['recipe']['family']
        source_request = {'operation': 'outer', 'chosen': decision,
                          'namespace': source_namespace, 'repeat': None}
        source = read_bound(experiment.root/'outer'/source_namespace/
            f'{family}-t{horizon}-fold{fold["fold"]}.json',
            execution_binding(experiment, fold, source_request, selection=False))
        verified_outer(experiment.history, fold, source, family, horizon)
        request = {'operation': 'outer', 'chosen': chosen, 'namespace': namespace, 'repeat': None}
        freeze_record(experiment.root/'outer'/namespace/f'ensemble-t{horizon}-fold{fold["fold"]}.json',
                      {**source, 'recipe': chosen['recipe'], 'inner_score': chosen['score'],
                       'execution_binding': execution_binding(experiment, fold, request, selection=False)})
    else:
        experiment.outer(chosen, fold, namespace)


def leaf_recipe(chosen):
    while chosen['recipe']['family'] == 'ensemble':
        if chosen['weights'] != [1.] or len(chosen['members']) != 1:
            raise ValueError('History refinement requires one selected native recipe')
        chosen = chosen['members'][0]['decision']
    return chosen['recipe']


def run_refinement(experiment, round_name):
    if round_name not in ('D', 'E'):
        raise ValueError('Expected round D or E')
    if (experiment.root/'locked.json').exists():
        raise ValueError('Locked experiment is immutable')
    for h in (1, 5):
        for fold in experiment.identity['split']['folds']:
            validate_inner_fold(experiment.history, fold)
            namespaces = ['A', 'C']
            if round_name == 'E':
                namespaces += [p.name for p in sorted((experiment.root/'decisions').glob('D-*'))
                               if p.is_dir()]
            candidates = prior_choices(experiment, h, fold, namespaces)
            if round_name == 'D':
                features = experiment.identity['groups']['expanded']
                native_namespace = 'D-' + content_id({'features': features})[:8]
                for kind, families in (
                    ('tabular', ('xgboost', 'catboost')),
                    ('sequence', ('mlp', 'lstm', 'tcn'))):
                    eligible = [c for c in candidates if c['family'] in families]
                    if not eligible:
                        raise ValueError('Round D requires this fold tabular and sequence decisions')
                    best = eligible[0]
                    specs = recipes(best['family'], h, features, 32, offset=1000)
                    result = experiment.tune(best['family'], h, fold, specs, native_namespace)
                    publish_choice(experiment, h, fold,
                        {'namespace': native_namespace, 'family': best['family'],
                         'decision': result['chosen']},
                        native_namespace + f'-{kind}-policy-v2')
            else:
                best = candidates[0]
                if best['decision']['score'] >= 1:
                    freeze_record(experiment.root/'refinement-v2'/f'skipped-t{h}-fold{fold["fold"]}.json',
                                  {'reason': 'no_positive_this_fold_inner_gain',
                                   'decision_identity': content_id(best),
                                   'execution_binding': execution_binding(
                                       experiment, fold, {'selected': best}, selection=True)})
                    continue
                base = leaf_recipe(best['decision'])
                specs = [{**base, 'years': years, 'cadence': cadence}
                         for years in (3, 5, None) for cadence in (5, 21, 63, 126)]
                result = experiment.tune(base['family'], h, fold, specs, 'E-native-v2')
                publish_choice(experiment, h, fold,
                    {'namespace': 'E-native-v2', 'family': base['family'], 'decision': result['chosen']},
                    'E-policy-v2')
        if round_name == 'E':
            ensemble(experiment, h)


def ensemble(experiment, horizon, candidates=None):
    """Ignore legacy aggregate scores; select verified members within each fold."""
    for fold in experiment.identity['split']['folds']:
        namespaces = ['A', 'C'] + [
            p.name for p in sorted((experiment.root/'decisions').glob('D-*')) if p.is_dir()]
        ranked = prior_choices(experiment, horizon, fold, namespaces)
        if ranked[0]['decision']['score'] >= 1:
            continue
        ensemble_fold(experiment, horizon, fold, ranked, output_namespace='E-ensemble-v2')
