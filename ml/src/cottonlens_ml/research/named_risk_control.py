"""Prepare one conditional T+1 Log-HAR label control; preparation never fits."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.full_year import SHRINKAGE, chunks, volatility_recipe
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.volatility import FLOOR, qlike

PROFILE = 'named-risk-label-control-t1-v1'
ARMS = ('ct', 'named')
RECIPE = {**volatility_recipe('har', 1), 'label_control': PROFILE}


def decision(day):
    return pd.Timestamp(day, tz='UTC') + pd.Timedelta(days=1, minutes=15)


def named_ewma(dates, quotes, available):
    """Same named contract, exact consecutive Cotton dates, past events only.

    Missing pairs do not create returns or elapsed-time decay; lambda .94 is
    applied per observed return. A delayed pair enters only once available.
    Initialization uses the first observed square, matching the old EWMA.
    Contract selection uses the current table only, never the future front.
    """
    events = defaultdict(list)
    for previous, current in pairwise(dates):
        a, b = quotes.get(previous), quotes.get(current)
        if a is None or b is None:
            continue
        for contract in sorted(set(a['quotes']) & set(b['quotes'])):
            first, last = a['quotes'][contract], b['quotes'][contract]
            if any(not math.isfinite(p) or p <= 0 for p in (first, last)):
                raise ValueError('Positive finite named EWMA prices required')
            events[contract].append((current, max(available(a), available(b)), math.log(last / first) ** 2))
    results = []
    for day in dates:
        quote = quotes.get(day)
        row = {'named_ewma': np.nan, 'named_ewma_observations': 0,
               'named_ewma_last_date': None, 'named_ewma_eligible': False}
        if quote is not None and available(quote) <= decision(day):
            contract = quote['second_contract']
            past = [(end, square) for end, clock, square in events[contract]
                    if end <= day and clock <= decision(day)]
            if past:
                value = past[0][1]
                for _, square in past[1:]:
                    value = .94 * value + .06 * square
                row.update(named_ewma=max(value, FLOOR), named_ewma_observations=len(past),
                           named_ewma_last_date=past[-1][0], named_ewma_eligible=len(past) >= 22)
        results.append(row)
    return pd.DataFrame(results)


def attach(history, panel, helper):
    if (history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or history.date.max() >= pd.Timestamp('2024-01-01')):
        raise ValueError('Unique sorted pre2024 parent required')
    if any(c.startswith('named_') for c in history):
        raise ValueError('Metadata already present')
    if panel['model_eligible'] or panel['training_ready'] or panel['release_allowed']:
        raise ValueError('Candidate source must remain unadmitted')
    quotes = {q['report_date']: q for q in panel['rows']}
    if len(quotes) != len(panel['rows']) or any(q['model_eligible'] for q in quotes.values()):
        raise ValueError('Duplicate/admitted quotes')
    h = history.reset_index(drop=True).copy()
    dates = h.date.dt.strftime('%Y-%m-%d').tolist()
    labels = []
    for i, day in enumerate(dates):
        if i + 1 < len(dates):
            label = helper.named_label(quotes.get(day), quotes.get(dates[i+1]), day, dates[i+1], 'second')
        else:
            label = {'eligible': False, 'risk': None, 'contract': None,
                     'label_available_assumed': None, 'reason': 'missing_parent_target'}
        labels.append({'named_eligible': label['eligible'], 'named_variance': label['risk'],
                       'named_contract': label['contract'], 'named_reason': label['reason'],
                       'named_label_available_at': label['label_available_assumed']})
    h = pd.concat([h, pd.DataFrame(labels), named_ewma(dates, quotes, helper.assumed_available)], axis=1)
    h['named_label_available_at'] = pd.to_datetime(h.named_label_available_at, utc=True)
    # No feature, target, price or original row may change.
    pd.testing.assert_frame_equal(h[list(history)], history.reset_index(drop=True), check_exact=True)
    for name in RECIPE['features']:
        if not np.isfinite(h.loc[h.named_eligible, name]).all():
            raise ValueError('HAR features must be finite at common named origins')
    return h


def mature_dates(history, cutoff):
    cut = pd.Timestamp(cutoff)
    use = (history.named_eligible & history.date.lt(cut)
           & history.target_date_5.lt(cut) & history.date.ge(history.date.iloc[119])
           & history.named_label_available_at.le(decision(cut.strftime('%Y-%m-%d')))
           & np.isfinite(history.variance_target_1) & np.isfinite(history.named_variance))
    return history.loc[use, 'date'].dt.strftime('%Y-%m-%d').tolist()


def control_jobs(history, origins, role):
    # Preserve the established 21-Cotton-observation buckets/cutoffs; only
    # replace the common mature population used by both label arms.
    start = int(rows_at(history, origins).cotton_session_index.iloc[0])
    calendar = history.set_index('cotton_session_index').date
    result = []
    for chunk in chunks(history, origins):
        bucket = (int(chunk.cotton_session_index.iloc[0]) - start) // 21
        cutoff = calendar.loc[start + bucket * 21].strftime('%Y-%m-%d')
        result.append({'role': role, 'bucket': bucket, 'cutoff': cutoff,
                       'origins': chunk.date.dt.strftime('%Y-%m-%d').tolist(),
                       'train_dates': mature_dates(history, cutoff)})
    return result


def plan(history, parent_split):
    folds, readiness = [], []
    for parent in parent_split['folds']:
        past = mature_dates(history, parent['origins'][0])[-189:]
        inner = [past[i:i+63] for i in range(0, len(past), 63)]
        work = [j for i, block in enumerate(inner) for j in control_jobs(history, block, f'inner_{i}')]
        work += control_jobs(history, parent['origins'], 'outer')
        baseline_ready = bool(past) and rows_at(history, past).named_ewma_eligible.all()
        supported = (len(past) == 189 and len(inner) == 3
                     and all(len(b) == 63 for b in inner)
                     and all(len(j['train_dates']) >= 500 for j in work) and baseline_ready)
        readiness.append({'year': parent['year'], 'supported': bool(supported),
                          'validation_labels': len(past), 'named_baseline_ready': bool(baseline_ready),
                          'minimum_train_rows': min((len(j['train_dates']) for j in work), default=0)})
        if supported:
            outer = rows_at(history, parent['origins'])
            folds.append({'year': parent['year'], 'origins': parent['origins'], 'inner': inner,
                          'jobs': work, 'eligible_outer': int((outer.named_eligible & outer.named_ewma_eligible).sum())})
    return {'profile': PROFILE, 'arms': list(ARMS), 'recipe': RECIPE, 'folds': folds, 'readiness': readiness,
            'fit_budget': {'market': 2 * sum(len(f['jobs']) for f in folds), 'synthetic': 1,
                           'threads': 2, 'processes': 1, 'session_minutes': 30},
            'primary': 'Raw named-trained minus CT-trained QLIKE improvement on common named T+1 targets',
            'secondary': 'Both raw and past-shrunk forecasts versus causal same-contract EWMA',
            'baseline': {'lambda': .94, 'minimum_observed_pairs': 22, 'missing_pair': 'skip_no_return_imputation',
                         'initialization': 'first_observed_square', 'contract': 'origin_second_named'},
            'selection': {'weights': list(SHRINKAGE), 'criterion': 'equal mean inner-block named QLIKE',
                          'tie': 'smallest_weight', 'model_family': 'fixed_log_har_no_search'},
            'clock': 'date+1day00:15UTC; source calendar+1day00:00UTC assumed, archive final vintage assumed',
            'confidence': 'Research-only paired within-year full-ordinal block20/60,10000 replicates,seed42',
            'gate_evaluation_allowed': False, 'release_allowed': False, 'historical_source_admitted': False,
            'source_timestamp_verified': False, 'first_vintage_verified': False,
            'role': 'seen_history_mechanism_test_NOT_8year_gate_or_independent_skill',
            'deliberate_repeat_reason': 'Old risk transport did not train named labels or causal named EWMA'}


def projection(history, dates, arm):
    if arm not in ARMS:
        raise ValueError('Unknown label arm')
    rows = rows_at(history, dates).copy()
    if arm == 'named':
        rows['variance_target_1'] = rows.named_variance
    if not rows.named_eligible.all() or not np.isfinite(rows.variance_target_1).all():
        raise ValueError('Exact common finite training labels required')
    return rows


def load_inputs(folder):
    spec = importlib.util.spec_from_file_location('frozen_risk_label_helper', folder/'label-helper.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    panel = read_record(folder/'quotes.json')
    parent = read_record(folder/'parent-ready.json')
    quote_proof = read_record(folder/'quote-proof.json')
    measurement = read_record(folder/'measurement-proof.json')
    report = read_record(folder/'measurement-report.json')
    if (digest(folder/'history.parquet') != parent['history_sha256']
            or digest(folder/'quotes.json') != quote_proof['candidate_panel_sha256']
            or digest(folder/'label-helper.py') != measurement['analysis_code_sha256']
            or digest(folder/'measurement-report.json') != measurement['report_sha256']
            or digest(folder/'bootstrap.py') != report['input_hashes']['bootstrap.py']):
        raise ValueError('Parent source/measurement binding mismatch')
    return attach(pd.read_parquet(folder/'history.parquet'), panel, helper), parent


def prepare(repo, folder, evidence, parent_ready):
    repo, folder, evidence, parent_ready = map(Path, (repo, folder, evidence, parent_ready))
    if folder.exists():
        raise ValueError('Preserve existing registration; choose new namespace')
    # Source helper is the exact independently verified measurement code.
    paths = {'history.parquet': evidence/'inputs/history.parquet', 'quotes.json': evidence/'inputs/quotes.json',
             'quote-proof.json': evidence/'inputs/quote-proof.json', 'parent-ready.json': parent_ready,
             'label-helper.py': evidence/'risk_readiness.py',
             'bootstrap.py': evidence/'inputs/bootstrap.py', 'measurement-report.json': evidence/'result/report.json',
             'measurement-proof.json': repo/'research/evidence/risk-measurement-readiness-20261012.json'}
    for path in paths.values():
        if not path.is_file():
            raise ValueError('Required frozen input missing')
    with writer(folder):
        for name, path in paths.items():
            copy_immutable(path, folder/'inputs'/name)
        history, parent = load_inputs(folder/'inputs')
        design = plan(history, parent['identity']['split'])
        if not design['folds']:
            raise ValueError('No supported fixed-rule fold; preserve diagnostic inputs')
        validate(repo/'research')
        registry, trials = load_registry(repo/'research')
        related = check(registry, trials, family='har', horizon=1, query='named')
        identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
                    'design': design, 'design_id': content_id(design), 'split': parent['identity']['split'],
                    'inputs_sha256': {n: digest(p) for n, p in paths.items()},
                    'research_data_id': frame_identity(history, list(history)),
                    'python': sys.version.split()[0], 'lock_sha256': digest(repo/'ml/uv.lock'),
                    'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'pyarrow', 'scikit-learn')},
                    'historical_availability_verified': False, 'no_fit_on_prepare': True}
        history.to_parquet(folder/'history.parquet', index=False)
        freeze_record(folder/'history-check.json', {'status': related['status'],
                      'related': [x['id'] for x in related['experiments']], 'new_fits': 0})
        freeze_record(folder/'source-manifest.json', research_source_identity(repo))
        freeze_record(folder/'decision-contract.json', design)
        freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
    return design


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder/'ready.json')
    identity = ready['identity']
    if (identity['profile'] != PROFILE or identity['source_id'] != research_source_identity(repo)['source_id']
            or identity['python'] != sys.version.split()[0] or identity['lock_sha256'] != digest(repo/'ml/uv.lock')
            or identity['versions'] != {p: importlib.metadata.version(p) for p in identity['versions']}):
        raise ValueError('Source/profile/environment changed; no cache reuse')
    for name, expected in identity['inputs_sha256'].items():
        if digest(folder/'inputs'/name) != expected:
            raise ValueError('Frozen input changed')
    history, parent = load_inputs(folder/'inputs')
    design = plan(history, parent['identity']['split'])
    if (design != identity['design'] or content_id(design) != identity['design_id']
            or design != read_record(folder/'decision-contract.json')
            or digest(folder/'history.parquet') != ready['history_sha256']
            or frame_identity(history, list(history)) != identity['research_data_id']):
        raise ValueError('Frozen data/cohort/contract mismatch')
    pd.testing.assert_frame_equal(history, pd.read_parquet(folder/'history.parquet'), check_exact=True)
    return ready, history


def output(history, fold, arm, records):
    predictions, ids = {}, {}
    for job, record in zip(fold['jobs'], records, strict=True):
        result = record['result']
        values = result['variance_predictions']
        if (result['origins'] != job['origins'] or len(values) != len(job['origins'])
                or not np.isfinite(values).all() or any(v <= 0 for v in values)
                or set(predictions).intersection(job['origins'])):
            raise ValueError('Exact finite positive disjoint variance predictions required')
        predictions.update(zip(job['origins'], values, strict=True))
        ids[f'{job["role"]}-{job["bucket"]}'] = record['experiment_id']
    scores = []
    for weight in SHRINKAGE:
        losses = []
        for dates in fold['inner']:
            rows = rows_at(history, dates)
            if not rows.named_eligible.all() or not rows.named_ewma_eligible.all():
                raise ValueError('Complete named inner target/reference required')
            raw = np.array([predictions[d] for d in dates])
            losses.append(float(qlike(rows.named_variance.to_numpy(),
                                      weight * raw + (1-weight) * rows.named_ewma.to_numpy()).mean()))
        scores.append({'weight': weight, 'mean_inner_qlike': float(np.mean(losses))})
    selected = min(scores, key=lambda s: (s['mean_inner_qlike'], s['weight']))
    rows = rows_at(history, fold['origins'])
    entries = []
    for row, day in zip(rows.itertuples(), fold['origins'], strict=True):
        eligible = bool(row.named_eligible and row.named_ewma_eligible)
        raw = predictions[day]
        entries.append({'date': day, 'target_date': row.target_date_1.strftime('%Y-%m-%d'),
                        'contract': row.named_contract if row.named_eligible else None,
                        'eligible': eligible, 'named_reason': row.named_reason,
                        'ewma_observations': int(row.named_ewma_observations),
                        'named_actual': float(row.named_variance) if row.named_eligible else None,
                        'baseline': float(row.named_ewma) if row.named_ewma_eligible else None,
                        'raw': raw, 'selected': float(selected['weight'] * raw + (1-selected['weight']) * row.named_ewma)
                        if row.named_ewma_eligible else None})
    return {'year': fold['year'], 'arm': arm, 'fit_ids': ids, 'selection': selected,
            'all_inner_weights': scores, 'selection_used_outer': False, 'records': entries}


def learning_control(experiment):
    """One noiseless synthetic Log-HAR control, separate from market evidence."""
    from cottonlens_ml.research.models import fit_predict
    rng = np.random.default_rng(42)
    x = rng.uniform(-10, -4, size=(1200, 3))
    y = np.exp(-1 + x @ np.array([.2, .4, .1]))
    frame = pd.DataFrame(x, columns=RECIPE['features'])
    frame['date'] = pd.bdate_range('2000-01-03', periods=len(frame))
    frame['variance_target_1'] = y
    train, test = frame.iloc[:1000].copy(), frame.iloc[1000:].copy()
    ledger = Ledger(experiment.root/'learning-ledger', {**experiment.identity, 'role': 'synthetic_not_market'})
    from cottonlens_ml.research.contract_curve_execution import fit_consumption
    consumed, keys = fit_consumption(ledger.root)
    if consumed > 1 or consumed != len(keys):
        raise ValueError('Failed/excess learning fits; no silent retry')
    record = ledger.run({'recipe': RECIPE, 'role': 'synthetic_noiseless_log_har_control', 'kind': 'known_log_har_signal', 'seed': 42},
                        lambda w: fit_predict(frame, train, None, test, RECIPE, w, iterations=1))
    prediction = np.asarray(record['result']['variance_predictions'])
    relative_error = float(np.max(np.abs(prediction / y[1000:] - 1)))
    if relative_error > 1e-10 or record['result']['origins'] != test.date.dt.strftime('%Y-%m-%d').tolist():
        raise ValueError('Known Log-HAR signal failed; market fits prohibited')
    freeze_record(experiment.root/'learning-control.json', {'fit_id': record['experiment_id'],
                  'maximum_relative_error': relative_error, 'status': 'passed', 'market_skill': False})


def run(folder, repo, authorization, max_minutes=30):
    from cottonlens_ml.research.contract_curve_execution import fit_consumption
    from cottonlens_ml.research.engine import Experiment
    from cottonlens_ml.runtime_guard import require_training
    require_training('har', 'cpu')
    ready, history = verify(folder, repo)
    auth = read_record(Path(authorization))
    design = ready['identity']['design']
    if (auth.get('ready_sha256') != digest(Path(folder)/'ready.json')
            or auth.get('design_id') != ready['identity']['design_id']
            or auth.get('source_id') != ready['identity']['source_id']
            or auth.get('fresh_restore_verified') is not True
            or auth.get('fit_budget') != design['fit_budget'] or not 0 < max_minutes <= 30):
        raise ValueError('Frozen preregistration/fresh recovery/budget authorization required')
    deadline = time.monotonic() + max_minutes * 60
    with writer(Path(folder)):
        experiment = Experiment(folder, repo=repo)
        learning_control(experiment)
        used, keys = fit_consumption(experiment.ledger.root)
        if used > design['fit_budget']['market'] or used != len(keys):
            raise ValueError('Failed/excess market fits; preserve and inspect, no silent retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('30-minute checkpoint pause')
            if count[0] >= design['fit_budget']['market']:
                raise ValueError('Frozen market budget exhausted')
            count[0] += 1

        experiment.before_compute = before_compute
        try:
            for fold in design['folds']:
                for arm in ARMS:
                    records = []
                    for job in fold['jobs']:
                        train = projection(history, job['train_dates'], arm)
                        test = rows_at(history, job['origins'])
                        spec = {**RECIPE, 'label_arm': arm}
                        records.append(experiment.fit(spec, train, None, test,
                            f'risk-{fold["year"]}-{job["role"]}-{job["bucket"]}', iterations=1))
                    freeze_record(Path(folder)/'risk-outputs'/f'{arm}-{fold["year"]}.json',
                                  output(history, fold, arm, records))
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0], 'synthetic_fits': 1}
    return {'status': 'completed', 'market_fits': count[0], 'synthetic_fits': 1, 'market_skill_claimed': False}


def compare(folder, repo):
    """Rebuild from exact native receipts. Never call fit from reporting."""
    ready, history = verify(folder, repo)
    folder = Path(folder)
    ledger = Ledger(folder/'ledger', ready['identity'])
    spec = importlib.util.spec_from_file_location('frozen_risk_blocks', folder/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    paired, verified_fit_ids = [], set()
    for fold in ready['identity']['design']['folds']:
        arms = {}
        for arm in ARMS:
            records = []
            for job in fold['jobs']:
                train = projection(history, job['train_dates'], arm)
                test = rows_at(history, job['origins'])
                fit_spec = {'recipe': {**RECIPE, 'label_arm': arm},
                            'role': f'risk-{fold["year"]}-{job["role"]}-{job["bucket"]}', 'iterations': 1,
                            'train_dates': job['train_dates'], 'validation_dates': [], 'test_dates': job['origins'],
                            'train_identity': frame_identity(train, list(train))}
                key = ledger.key({**fit_spec, 'repeat_reason': None})
                record = read_record(ledger.root/'completed'/f'{key}.json')
                if (record['identity'] != ready['identity'] or record['experiment_id'] != key
                        or record['specification'] != {**fit_spec, 'repeat_reason': None} or not record['files']):
                    raise ValueError('Native fit identity mismatch/incomplete payload')
                for name, expected in record['files'].items():
                    if not safe_member(name) or digest(ledger.root/name) != expected:
                        raise ValueError('Native payload hash mismatch')
                native = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_bytes())
                if native['kind'] != 'log_har' or native['features'] != RECIPE['features']:
                    raise ValueError('Wrong native model family/features')
                prediction = np.maximum(np.exp(np.column_stack([np.ones(len(test)), test[RECIPE['features']]])
                                               @ np.array(native['coef'])) * native['smearing'], FLOOR)
                np.testing.assert_allclose(prediction, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
                records.append(record)
                verified_fit_ids.add(key)
            expected = output(history, fold, arm, records)
            if expected != read_record(folder/'risk-outputs'/f'{arm}-{fold["year"]}.json'):
                raise ValueError('Output or past-only weight differs from native fits')
            arms[arm] = expected
        for ct, named in zip(arms['ct']['records'], arms['named']['records'], strict=True):
            for name in ('date', 'target_date', 'contract', 'eligible', 'named_actual', 'baseline'):
                if ct[name] != named[name]:
                    raise ValueError('Paired identity/target/reference mismatch; no silent intersection')
            row = {'year': fold['year'], 'date': ct['date'], 'target_date': ct['target_date'],
                   'contract': ct['contract'], 'eligible': ct['eligible'],
                   'actual': ct['named_actual'], 'baseline_variance': ct['baseline'],
                   'ct_raw': ct['raw'], 'named_raw': named['raw'],
                   'ct_selected': ct['selected'], 'named_selected': named['selected']}
            if ct['eligible']:
                for key, forecast in [('baseline', ct['baseline']), ('ct_raw', ct['raw']), ('named_raw', named['raw']),
                                      ('ct_selected', ct['selected']), ('named_selected', named['selected'])]:
                    row[key+'_loss'] = float(qlike(np.array([ct['named_actual']]), np.array([forecast]))[0])
            paired.append(row)
    frame = pd.DataFrame(paired)
    valid = frame.loc[frame.eligible]
    comparisons = {}
    for name, base, candidate in [('label_mechanism', 'ct_raw', 'named_raw'), ('named_raw_vs_ewma', 'baseline', 'named_raw'),
                                   ('ct_raw_vs_ewma', 'baseline', 'ct_raw'), ('named_selected_vs_ewma', 'baseline', 'named_selected')]:
        a, b = base+'_loss', candidate+'_loss'
        offset = max(0., 1 - min(valid[a].min(), valid[b].min()))
        sample = frame[['year']].copy()
        sample['a'] = np.where(frame.eligible, frame[a].fillna(0)+offset, 0)
        sample['b'] = np.where(frame.eligible, frame[b].fillna(0)+offset, 0)
        sample['weight'] = frame.eligible.astype(int)
        parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in sample.groupby('year')]
        intervals = {}
        for block in (20, 60):
            measured = bootstrap.weighted_bootstrap(parts, block=block)
            intervals[str(block)] = {k: measured[k] for k in ('eligible', 'ordinal_slots', 'block', 'replicates',
                                                             'paired_difference', 'difference_ci_95')}
        trimmed = valid.drop(valid.nlargest(max(1, math.ceil(len(valid)*.01)), 'baseline_loss').index)
        comparisons[name] = {'delta': float((valid[a]-valid[b]).mean()), 'bootstrap': intervals,
                             'trim_top1pct_ewma_loss_delta': float((trimmed[a]-trimmed[b]).mean())}
    mechanism = comparisons['label_mechanism']
    better_labels = all(v['difference_ci_95'][0] > 0 for v in mechanism['bootstrap'].values())
    worse_labels = all(v['difference_ci_95'][1] <= 0 for v in mechanism['bootstrap'].values())
    report = {'profile': PROFILE, 'new_fits_during_report': 0, 'verified_native_market_fits': len(verified_fit_ids),
              'ordinal_origins': len(frame), 'eligible_origins': len(valid), 'comparisons': comparisons,
              'classification': 'DECISIVE_POSITIVE' if better_labels else 'DECISIVE_NEGATIVE' if worse_labels else 'INCONCLUSIVE',
              'classification_scope': 'Conditional one-year label mechanism ONLY; not market-wide source/model absence',
              'independent_holdout': False, 'eight_year_gate_allowed': False,
              'release_allowed': False, 'source_timestamp_verified': False, 'first_vintage_verified': False,
              'scope': 'One seen-year conditional mechanism evidence; not independent or overall risk success'}
    output_dir = folder/'reports'
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir/'paired-losses.csv'
    if csv_path.exists():
        raise ValueError('Immutable report exists; preserve it')
    frame.to_csv(csv_path, index=False, lineterminator='\n')
    report['paired_losses_sha256'] = digest(csv_path)
    freeze_record(output_dir/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'verify', 'pilot', 'compare'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--parent-ready', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--max-minutes', type=float, default=30)
    args = parser.parse_args()
    if args.stage == 'compare':
        print(compare(args.root, args.repo))
        return
    if args.stage == 'pilot':
        if args.authorization is None:
            parser.error('pilot requires frozen authorization')
        print(run(args.root, args.repo, args.authorization, args.max_minutes))
        return
    if args.stage == 'prepare':
        if args.evidence is None or args.parent_ready is None:
            parser.error('prepare requires evidence and parent-ready')
        design = prepare(args.repo, args.root, args.evidence, args.parent_ready)
    else:
        ready, _ = verify(args.root, args.repo)
        design = ready['identity']['design']
    print({'new_fits': 0, 'years': [f['year'] for f in design['folds']], 'budget': design['fit_budget']})


if __name__ == '__main__':
    main()
