"""Manual, immutable own-level risk research. No scheduler, grid or live deployment."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.volatility import FLOOR, qlike

from research import ct_shock_control as learner
from research.ct_manual_source import (
    CottonCalendar,
    classify_chart,
    utc,
    verified_response,
)
from research.ct_prospective_readiness import append_snapshot, candidate_history

PROFILE = 'ct-own-level-manual-forward-t1-v1'
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')
HELPERS = ('research/ct_manual_forward.py', 'research/tests/test_ct_manual_forward.py',
           'research/ct_shock_control.py', 'research/ct_manual_source.py',
           'research/ct_prospective_readiness.py', 'research/verify_ct_manual_forward.py',
           'research/verify_risk_weak_teacher.py')
NAMES = ['har_1', 'har_5', 'har_22', 'origin_state']
WEIGHTS = (0., .25, .5, .75, 1.)
SOURCE_PROOF = 'evidence/ct-manual-source-admission-20261030.json'


def now():
    return pd.Timestamp(datetime.now(UTC))


def decision(day):
    return pd.Timestamp(day, tz='UTC')+pd.Timedelta(days=1, minutes=15)


def history(seed):
    h = candidate_history(seed)
    for horizon in (1, 5):
        h[f'target_date_{horizon}'] = h.date.shift(-horizon)
    return h


def projection(h, dates):
    rows = h.set_index('date', drop=False).loc[pd.to_datetime(dates)]
    result = rows[['date', 'ct_ewma', *NAMES, 'variance_target_1']].reset_index(drop=True).copy()
    if not np.isfinite(result[['ct_ewma', *NAMES]].to_numpy()).all():
        raise ValueError('No missing feature origin may be dropped or imputed')
    return result


def mature(h, cutoff):
    cut = pd.Timestamp(cutoff)
    use = h.date.ge(h.date.iloc[119]) & h.date.lt(cut) & h.target_date_5.lt(cut)
    selected = h.loc[use]
    if len(selected) < 500 or not np.isfinite(selected.variance_target_1).all():
        raise ValueError('At least500 exact mature training rows after original warmup required')
    return selected.date.dt.strftime('%Y-%m-%d').tolist()


def annual_jobs(h, first_origin):
    past = mature(h, first_origin)[-189:]
    if len(past) != 189:
        raise ValueError('Exactly3x63 mature past validation origins required')
    jobs = []
    for block in range(3):
        for bucket in range(3):
            origins = past[block*63+bucket*21:block*63+(bucket+1)*21]
            cut = origins[0]
            jobs.append({'year': pd.Timestamp(first_origin).year, 'role': f'inner_{block}',
                         'bucket': bucket, 'cutoff': cut, 'origins': origins,
                         'train_dates': mature(h, cut)})
    return jobs


def source(repo):
    body = {'ml': research_source_identity(repo), 'helpers': {n: digest(repo/n) for n in HELPERS}}
    return {**body, 'source_id': content_id(body)}


def environment():
    return {'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in VERSIONS}}


def native_learning(parent, repo):
    """Reuse the existing known-mean payload only after exact learner/env/200prediction checks."""
    parent_ready = read_record(parent/'ready.json')
    previous = parent_ready['identity']
    if (previous['source_id'] != '5846f9cbb0f360731e84f251ed311f5f665510590594645204daa95c6438ab26'
            or previous['profile'] != learner.STATE_PROFILE
            or previous['python'] != environment()['python'] or previous['versions'] != environment()['versions']):
        raise ValueError('Native learning environment changed; no generic passed-cache reuse')
    for name in ('research/ct_shock_control.py', 'ml/src/cottonlens_ml/research/ct_ewma_har.py',
                 'ml/src/cottonlens_ml/research/volatility.py', 'ml/src/cottonlens_ml/runtime_guard.py'):
        if digest(repo/name) != digest(parent/'source-snapshot'/name):
            raise ValueError('Native learner bytes changed')
    paths = list((parent/'learning/ledger/completed').glob('*.json'))
    if len(paths) != 1 or list((parent/'learning/ledger/attempts').glob('*.json')):
        raise ValueError('Exact completed native learning control required')
    record = read_record(paths[0]); root = parent/'learning/ledger'
    if (record['identity'] != {**previous, 'role': 'synthetic_not_market'}
            or record['specification']['recipe'] != learner.recipe('dxy_state', learner.STATE_PROFILE)
            or record['experiment_id'] != content_id({'identity': record['identity'], 'specification': record['specification']})):
        raise ValueError('Native learning identity mismatch')
    for name, checksum in record['files'].items():
        if not safe_member(name) or digest(root/name) != checksum:
            raise ValueError('Native learning payload hash mismatch')
    model = json.loads((root/record['payload_root']/'variance-model.json').read_bytes())
    if model['features'] != NAMES or len(model['coef']) != 5 or model['effective_parameters'] != 5:
        raise ValueError('Native learning did not consume the five-parameter representation')
    rng = np.random.default_rng(42); x = rng.uniform(-2, 2, (1200, 4))
    baseline = np.exp(rng.uniform(-12, -4, 1200))
    replay = baseline[1000:]*np.exp(model['coef'][0]+x[1000:]@model['coef'][1:])
    expected = baseline[1000:]*np.exp(.1+x[1000:]@np.array([.2, .1, -.1, -.15]))
    np.testing.assert_allclose(replay, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(replay, expected, rtol=1e-5)
    return {'status': 'native_learning_reverified', 'new_synthetic_fits': 0,
            'fit_id': record['experiment_id'], 'held_predictions_verified': 200,
            'native_record_sha256': digest(paths[0]), 'market_skill': False}


def prepare(repo, root, admission, parent):
    if root.exists():
        raise ValueError('Preserve completed/partial protocol namespace')
    validate(repo/'research'); registry, trials = load_registry(repo/'research')
    prior = check(registry, trials, family='har', horizon=1, profile=learner.STATE_PROFILE)
    proof = read_record(repo/'research'/SOURCE_PROOF)
    registered = {p['path']: p['sha256'] for p in registry['local_evidence']}
    seed_path = admission/'proposed-seed.parquet'; result_path = admission/'result.json'
    if (registered.get(SOURCE_PROOF) != digest(repo/'research'/SOURCE_PROOF)
            or digest(seed_path) != proof['proposed_seed_sha256']
            or digest(result_path) != proof['result_sha256'] or proof['forward_authorized']):
        raise ValueError('Exact registered, freshly restored source proposal required')
    control = native_learning(parent, repo)
    design = {'profile': PROFILE, 'recipe': learner.recipe('own_level', learner.STATE_PROFILE),
              'target': 'next Cotton session squared log return', 'baseline': {'family': 'ewma', 'lambda': .94},
              'required_origins': 252, 'maximum_calendar_years': 2, 'last_allowed_year': 2027,
              'decision_utc': '00:15', 'publication_end_utc': '00:30',
              'warmup_index': 119, 'maturity': 'target_date_5 strictly before origin-date cutoff',
              'selection': 'Each new calendar year: last189 mature origins,3x63,21observation inner refits; weights0/.25/.5/.75/1, smallest tie',
              'refit': 'Every21 official Cotton origin slots within year; count missing slots, reset at new year',
              'fit_budget': {'market': 31, 'synthetic': 0, 'processes': 1, 'threads': 2, 'session_minutes': 30},
              'fit_budget_reason': 'At most18annual inner+13outer across252slots in two years. Earlier21fit sketch incorrectly omitted annual reselection.',
              'primary': 'Mean paired QLIKE(EWMA)-QLIKE(past-selected own-level) on252 fixed future slots; positive favors model',
              'forward_gate': 'All252 timely paired records and mature targets; positive mean EWMA-minus-model QLIKE and both20/60block95LCI>0',
              'historical_gate_retained': 'Parent secondary already satisfied positive mean,bothpositiveCI,7/8years; original6/8historical threshold unchanged',
              'bootstrap': {'replicates': 10000, 'seed': 42, 'blocks': [20, 60], 'cross_year_blocks': False},
              'no_partial_score': True, 'no_outcome_based_extension': True,
              'missing': 'Retain every official slot; no retrospective forecast or replacement. Any missing slot prevents confirmatory superiority; partial paired analysis remains descriptive at end only.',
              'power': 'Bounded evidence study, not calibrated80%power or promised success; prior126conditional single-CI probability~9-10%.',
              'calendar': 'Reviewed IFUS Cotton only; no inference beyond coverage. Future official extension requires additive dated evidence with exact known calendar overlap; otherwise stop.',
              'source': 'Yahoo CT=F provider proxy; original settlement/named-contract/historical first vintage unverified',
              'automatic_tasks': False, 'runtime_deployment': False, 'trading_claim': False,
              'deliberate_repeat_reason': 'First independent prospective evaluation of the already chosen own-level recipe; no historical refit repeated.'}
    manifest = source(repo)
    paths = {'seed.parquet': seed_path, 'source-proof.json': repo/'research'/SOURCE_PROOF,
             'admission-result.json': result_path, 'calendar.json': admission/'inputs/calendar.json',
             'initial-chart.json': admission/'inputs/chart.json', 'initial-receipt.json': admission/'inputs/chart-receipt.json'}
    with writer(root):
        for name, p in paths.items():
            dest = root/'inputs'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
        for name in {**manifest['ml']['files'], **manifest['helpers']}:
            dest = root/'source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(repo/name, dest)
        shutil.copytree(parent/'learning', root/'native-learning/learning')
        shutil.copyfile(parent/'ready.json', root/'native-learning/ready.json')
        for name in ('research/ct_shock_control.py', 'ml/src/cottonlens_ml/research/ct_ewma_har.py',
                     'ml/src/cottonlens_ml/research/volatility.py', 'ml/src/cottonlens_ml/runtime_guard.py'):
            dest = root/'native-learning/source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent/'source-snapshot'/name, dest)
        identity = {'profile': PROFILE, 'source_id': manifest['source_id'], 'design_id': content_id(design),
                    'seed_sha256': digest(seed_path), **environment()}
        frozen = {p.relative_to(root).as_posix(): digest(p) for directory in ('inputs', 'native-learning')
                  for p in (root/directory).rglob('*') if p.is_file()}
        freeze_record(root/'source-manifest.json', manifest)
        freeze_record(root/'decision-contract.json', design)
        freeze_record(root/'learning-control.json', control)
        freeze_record(root/'history-check.json', {'status': prior['status'], 'related_ids': [r['id'] for r in prior['experiments']]})
        freeze_record(root/'ready.json', {'identity': identity, 'frozen_hashes': frozen})
    return {'status': 'prepared_not_locked', **identity, 'new_fits': 0, 'fit_budget': design['fit_budget']}


def verify(root, repo):
    ready = read_record(root/'ready.json'); identity = ready['identity']; design = read_record(root/'decision-contract.json')
    if (source(repo) != read_record(root/'source-manifest.json') or source(repo)['source_id'] != identity['source_id']
            or any(identity[k] != v for k, v in environment().items()) or content_id(design) != identity['design_id']):
        raise ValueError('Protocol source/environment/design changed')
    for name, checksum in ready['frozen_hashes'].items():
        if not safe_member(name) or digest(root/name) != checksum:
            raise ValueError('Frozen protocol/native input changed')
    if native_learning(root/'native-learning', repo) != read_record(root/'learning-control.json'):
        raise ValueError('Native learning replay changed')
    return ready, design


def calendar_body(root):
    body = json.loads((root/'inputs/calendar.json').read_bytes())
    paths = sorted((root/'calendar-extensions').glob('*/extension.json'))
    if len(paths) != len([p for p in (root/'calendar-extensions').glob('*') if p.is_dir()]):
        raise ValueError('Incomplete calendar extension is not admitted evidence')
    for path in paths:
        extension = read_record(path)
        if extension['previous_calendar_id'] != content_id(body):
            raise ValueError('Calendar extension chain changed')
        for name, checksum in extension['files'].items():
            if not safe_member(name) or digest(path.parent/name) != checksum:
                raise ValueError('Calendar extension evidence changed')
        check_calendar_extension(body, extension['calendar'], utc(extension['reviewed_at']))
        body = extension['calendar']
    return body


def calendar(root):
    return CottonCalendar(calendar_body(root))


def check_calendar_extension(old, new, reviewed_at):
    a, b = CottonCalendar(old), CottonCalendar(new)
    if (a.start != b.start or b.end <= a.end or old['market'] != new['market']
            or old['timezone'] != new['timezone']
            or old['completion_time_local'] != new['completion_time_local']
            or a.closed != {d for d in b.closed if d <= a.end}
            or reviewed_at >= pd.Timestamp(a.end, tz='UTC')+pd.Timedelta(days=1)):
        raise ValueError('Calendar extension must preserve known sessions and precede the new coverage period')


def extend_calendar(root, repo, reviewed_path, document, receipt_path):
    verify(root, repo); old = calendar_body(root); reviewed = json.loads(reviewed_path.read_bytes())
    receipt = verified_response(document, receipt_path); timestamp = now()
    if (reviewed.get('reviewed_market_column') != 'Cotton'
            or reviewed.get('document_sha256') != digest(document)
            or utc(receipt['observed_available_at']) > timestamp):
        raise ValueError('Explicit reviewed Cotton column and actual document receipt required')
    from urllib.parse import urlparse
    host = urlparse(receipt['source_url']).hostname
    if host != 'ice.com' and not (host or '').endswith('.ice.com'):
        raise ValueError('Official ICE calendar evidence required')
    check_calendar_extension(old, reviewed['calendar'], timestamp)
    with writer(root):
        if content_id(calendar_body(root)) != content_id(old):
            raise ValueError('Calendar changed during extension review')
        directory = root/'calendar-extensions'/reviewed['calendar']['coverage_end']
        directory.mkdir(parents=True, exist_ok=False)
        for src, name in ((reviewed_path, 'review.json'), (document, 'document.pdf'), (receipt_path, 'http-receipt.json')):
            shutil.copyfile(src, directory/name)
        if (json.loads((directory/'review.json').read_bytes()) != reviewed
                or verified_response(directory/'document.pdf', directory/'http-receipt.json') != receipt):
            raise ValueError('Calendar source changed during admission')
        freeze_record(directory/'extension.json', {'calendar': reviewed['calendar'],
            'previous_calendar_id': content_id(old), 'reviewed_at': timestamp.isoformat(),
            'files': {p.name: digest(p) for p in directory.iterdir() if p.is_file()}, 'new_fits': 0})
    return {'status': 'calendar_extended_with_preserved_overlap', 'coverage_end': reviewed['calendar']['coverage_end']}


def slots(cal, first):
    return [d.strftime('%Y-%m-%d') for d in pd.date_range(first, min(cal.end, pd.Timestamp('2027-12-31').date()))
            if cal.session(d.date())][:252]


def first_origin(cal, timestamp):
    # Target is still future at decision; never use an origin whose publication window started before lock.
    for d in pd.date_range(timestamp.normalize().tz_localize(None)-pd.Timedelta(days=1), cal.end):
        if cal.session(d.date()) and decision(str(d.date())) > timestamp:
            return str(d.date())
    raise ValueError('No future decision in reviewed calendar coverage')


def authorize(root, ready, design, auth):
    a = read_record(auth)
    if (a.get('ready_sha256') != digest(root/'ready.json') or a.get('source_id') != ready['identity']['source_id']
            or a.get('design_id') != ready['identity']['design_id'] or a.get('fresh_restore_verified') is not True
            or a.get('fit_budget') != design['fit_budget']):
        raise ValueError('Exact private restoration/preregistration/budget authorization required')


def lock(root, repo, auth):
    ready, design = verify(root, repo); authorize(root, ready, design, auth)
    if (root/'cohort-lock.json').exists():
        raise ValueError('Cohort is immutable; cannot restart after missed origins')
    timestamp = now(); first = first_origin(calendar(root), timestamp)
    if pd.Timestamp(first).year != 2026:
        raise ValueError('This bounded two-year protocol must start in2026; no silent postponement')
    body = {'profile': PROFILE, 'identity': ready['identity'], 'locked_at': timestamp.isoformat(),
            'first_origin': first, 'required_origins': 252, 'old_baseline_cohort_modified': False,
            'authorization_sha256': digest(auth), 'automatic_tasks': False}
    with writer(root):
        freeze_record(root/'cohort-lock.json', body)
    return body


def current_history(root):
    records = sorted((root/'observations').glob('*/receipt.json'))
    directories = [p for p in (root/'observations').glob('*') if p.is_dir()]
    if len(records) != len(directories):
        raise ValueError('Incomplete observation is not an admitted source')
    if not records:
        return pd.read_parquet(root/'inputs/seed.parquet'), None
    items = [read_record(p) for p in records]; items.sort(key=lambda x: utc(x['observed_available_at']))
    if len({utc(item['observed_available_at']) for item in items}) != len(items):
        raise ValueError('Ambiguous equal source observation times')
    previous = None
    for item in items:
        base = root/'observations'/item['observation_id']
        if item['previous_observation_id'] != previous:
            raise ValueError('Observation chain changed')
        for name, checksum in item['files'].items():
            if not safe_member(name) or digest(base/name) != checksum:
                raise ValueError('Observation payload changed')
        previous = item['observation_id']
    return pd.read_parquet(root/'observations'/previous/'history.parquet'), items[-1]


def admit(root, repo, raw, receipt_path):
    verify(root, repo); read_record(root/'cohort-lock.json')
    receipt = verified_response(raw, receipt_path)
    raw_bytes, receipt_bytes = raw.read_bytes(), receipt_path.read_bytes()
    if (hashlib.sha256(raw_bytes).hexdigest() != receipt['response_sha256']
            or json.loads(receipt_bytes) != receipt):
        raise ValueError('Raw source changed during admission')
    if utc(receipt['observed_available_at']) > now():
        raise ValueError('Future source receipt')
    oid = content_id({'source_sha256': hashlib.sha256(raw_bytes).hexdigest(), 'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest()})
    directory = root/'observations'/oid
    with writer(root):
        _, completed = classify_chart(json.loads(raw_bytes), receipt, calendar(root))
        old, previous = current_history(root); combined = append_snapshot(old, completed)
        if previous and utc(receipt['observed_available_at']) <= utc(previous['observed_available_at']):
            raise ValueError('Source observations must advance chronologically')
        if directory.exists():
            raise ValueError('Preserve existing source observation')
        directory.mkdir(parents=True)
        (directory/'chart.json').write_bytes(raw_bytes); (directory/'http-receipt.json').write_bytes(receipt_bytes)
        combined.to_parquet(directory/'history.parquet', index=False)
        body = {'observation_id': oid, 'previous_observation_id': previous['observation_id'] if previous else None,
                'observed_available_at': receipt['observed_available_at'], 'latest_completed_date': str(combined.date.iloc[-1].date()),
                'calendar_id': content_id(calendar_body(root)),
                'files': {p.name: digest(p) for p in directory.iterdir() if p.is_file()}, 'new_fits': 0}
        freeze_record(directory/'receipt.json', body)
    return body


def capture(root, repo):
    """One explicit manual HTTP capture; never schedule, retry or loop."""
    verify(root, repo); read_record(root/'cohort-lock.json'); started = now()
    if not started.normalize() <= started <= started.normalize()+pd.Timedelta(minutes=15):
        raise ValueError('Manual capture only within00:00–00:15UTC; no background polling')
    import requests
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/CT%3DF?range=2y&interval=1d&includePrePost=false'
    with writer(root):
        directory = root/'captures'/started.strftime('%Y%m%dT%H%M%S%fZ'); directory.mkdir(parents=True, exist_ok=False)
        response = requests.get(url, timeout=20, headers={'User-Agent': 'CottonLensAI source verification research'})
        observed = now(); raw = directory/'chart.json'; raw.write_bytes(response.content)
        receipt = {'source_url': url, 'request_started_at': started.isoformat(), 'observed_available_at': observed.isoformat(),
                   'http_status': response.status_code, 'response_sha256': digest(raw), 'response_bytes': raw.stat().st_size,
                   'publication_timestamp_verified': False, 'first_historical_vintage_verified': False,
                   'client_requests_version': requests.__version__, 'new_fits': 0, 'new_forecasts': 0,
                   'headers': {k: response.headers[k] for k in ('Date', 'Content-Type') if k in response.headers}}
        (directory/'http-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return admit(root, repo, raw, directory/'http-receipt.json')


def native_record(ledger, specification):
    key = ledger.key({**specification, 'repeat_reason': None})
    record = read_record(ledger.root/'completed'/(key+'.json'))
    if record['identity'] != ledger.identity or record['experiment_id'] != key:
        raise ValueError('Fit identity changed')
    for name, checksum in record['files'].items():
        if not safe_member(name) or digest(ledger.root/name) != checksum:
            raise ValueError('Native checkpoint changed')
    return record


def predict_model(ledger, record, test):
    m = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_bytes())
    if m['features'] != NAMES or m['recipe'] != learner.recipe('own_level', learner.STATE_PROFILE):
        raise ValueError('Different learner or representation')
    relative = learner.relative(test, False, learner.STATE_PROFILE)
    values = test.ct_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), relative[NAMES]])@m['coef'])
    if not np.isfinite(values).all() or (values < FLOOR).any():
        raise ValueError('Invalid variance prediction; no fallback or clipping')
    return values


def specification(job, train, observation_id):
    return {'recipe': learner.recipe('own_level', learner.STATE_PROFILE), 'role': f'{job["year"]}-{job["role"]}-{job["bucket"]}',
            'cutoff': job['cutoff'], 'iterations': 1, 'train_dates': job['train_dates'],
            'train_identity': frame_identity(train, list(train)), 'test_dates': job['origins'],
            'validation_dates': [], 'observation_id': observation_id,
            'prediction_scope': 'IN_SAMPLE_STATE_CHECK_ONLY' if job['role'] == 'outer' else 'PAST_SELECTION_ONLY'}


def verify_selection(root, ledger, h, year, first):
    selection = read_record(root/'selections'/f'{year}.json')
    jobs = annual_jobs(h, first); origins = [d for j in jobs for d in j['origins']]
    if (selection['cutoff'] != first or selection['validation_origins'] != origins
            or len(selection['fit_ids']) != 9 or selection['future_results_used']):
        raise ValueError('Annual past-selection population changed')
    predictions = []
    for job, key in zip(jobs, selection['fit_ids'], strict=True):
        record = read_record(ledger.root/'completed'/(key+'.json'))
        record = native_record(ledger, {k: v for k, v in record['specification'].items() if k != 'repeat_reason'})
        expected = specification(job, projection(h, job['train_dates']), record['specification']['observation_id'])
        if record['specification'] != {**expected, 'repeat_reason': None}:
            raise ValueError('Annual validation training population or labels changed')
        predicted = predict_model(ledger, record, projection(h, job['origins']))
        np.testing.assert_allclose(predicted, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
        predictions.extend(predicted)
    rows = projection(h, origins); baseline = rows.ct_ewma.to_numpy()
    scores = [(float(qlike(rows.variance_target_1, w*np.array(predictions)+(1-w)*baseline).mean()), w) for w in WEIGHTS]
    np.testing.assert_allclose(selection['scores'], scores, rtol=1e-12, atol=1e-14)
    if selection['weight'] != min(scores)[1]:
        raise ValueError('Past-only shrinkage weight changed')
    return selection


def fit_for_origin(root, repo, auth, origin, minutes=30):
    ready, design = verify(root, repo); authorize(root, ready, design, auth)
    locked = read_record(root/'cohort-lock.json'); cal = calendar(root); dates = slots(cal, locked['first_origin'])
    if origin not in dates or decision(origin) <= now() or not 0 < minutes <= 30:
        raise ValueError('Only a future locked origin within30minute session is eligible')
    seed, observation = current_history(root); h = history(seed)
    previous = [d for d in pd.date_range(cal.start, pd.Timestamp(origin)-pd.Timedelta(days=1)) if cal.session(d.date())][-1]
    if h.date.iloc[-1] not in (previous, pd.Timestamp(origin)):
        raise ValueError('Do not fit from stale history or skip unobserved Cotton sessions')
    year = pd.Timestamp(origin).year; annual = [d for d in dates if pd.Timestamp(d).year == year]
    index = annual.index(origin); bucket = index//21; cutoff = annual[bucket*21]
    ledger = Ledger(root/'ledger', {**ready['identity'], 'cohort_lock_id': content_id(locked)})
    used, keys = fit_consumption(ledger.root)
    if used != len(keys) or used > 31:
        raise ValueError('Failed/excess attempt; no silent retry or new recipe')
    deadline = time.monotonic()+60*minutes
    selection_path = root/'selections'/f'{year}.json'
    with writer(root):
        def compute(job):
            train = projection(h, job['train_dates'])
            if job['role'] == 'outer':
                test = train.tail(1).copy(); test['variance_target_1'] = np.nan
            else:
                test = projection(h, job['origins']); test['variance_target_1'] = np.nan
            spec = specification(job, train, observation['observation_id'] if observation else 'initial_seed')
            plan_path = root/'fit-plans'/f'{job["year"]}-{job["role"]}-{job["bucket"]}.json'
            if plan_path.exists():
                planned = read_record(plan_path)
                if {**spec, 'observation_id': planned['observation_id']} != planned:
                    raise ValueError('Frozen fit role changed; do not train another snapshot under the same role')
                spec = planned
            else:
                freeze_record(plan_path, spec)
            # Model address is frozen by role; never silently re-fit it on a later snapshot.
            address = root/'fits'/f'{job["year"]}-{job["role"]}-{job["bucket"]}.json'
            if address.exists():
                pointer = read_record(address); saved = read_record(ledger.root/'completed'/(pointer['fit_id']+'.json'))
                expected = {**spec, 'observation_id': saved['specification']['observation_id'], 'repeat_reason': None}
                if (saved['specification'] != expected
                        or pointer['specification_id'] != content_id({k: v for k, v in expected.items() if k != 'repeat_reason'})):
                    raise ValueError('Mature training population changed for an existing role')
                return native_record(ledger, {k: v for k, v in saved['specification'].items() if k != 'repeat_reason'})
            def before():
                count, _ = fit_consumption(ledger.root)
                if count >= 31:
                    raise ValueError('Frozen31fit budget exhausted')
                if time.monotonic() >= deadline:
                    raise FitBudgetReached('Checkpoint session boundary')
            record = ledger.run(spec, lambda p: {**learner.fit(train, test, p, 'own_level', learner.STATE_PROFILE),
                'prediction_scope': spec['prediction_scope']}, before_compute=before)
            freeze_record(address, {'fit_id': record['experiment_id'], 'specification_id': content_id(spec)})
            return record
        if not selection_path.exists():
            validation = []; predictions = []; ids = []
            for job in annual_jobs(h, annual[0]):
                record = compute(job); validation += job['origins']; ids.append(record['experiment_id'])
                predictions += record['result']['variance_predictions']
            rows = projection(h, validation); baseline = rows.ct_ewma.to_numpy(); predicted = np.array(predictions)
            scores = [(float(qlike(rows.variance_target_1, w*predicted+(1-w)*baseline).mean()), w) for w in WEIGHTS]
            freeze_record(selection_path, {'year': year, 'cutoff': annual[0], 'validation_origins': validation,
                'fit_ids': ids, 'scores': scores, 'weight': min(scores)[1], 'created_at': now().isoformat(),
                'future_results_used': False})
        outer = {'year': year, 'role': 'outer', 'bucket': bucket, 'cutoff': cutoff,
                 'origins': [], 'train_dates': mature(h, cutoff)}
        record = compute(outer)
        verify_selection(root, ledger, h, year, annual[0])
    return {'status': 'state_ready_no_forward_forecast', 'fit_id': record['experiment_id'],
            'market_fits': fit_consumption(ledger.root)[0], 'new_forward_forecasts': 0,
            'weight': read_record(selection_path)['weight']}


def publish(root, repo):
    ready, _ = verify(root, repo); locked = read_record(root/'cohort-lock.json')
    seed, observation = current_history(root)
    if observation is None:
        raise ValueError('A fresh actual source receipt is required; seed is not a timely origin capture')
    origin = observation['latest_completed_date']; stamp = now(); cutoff = decision(origin)
    if not cutoff <= stamp < cutoff+pd.Timedelta(minutes=15) or utc(observation['observed_available_at']) > cutoff:
        raise ValueError('No late source, early publication or retrospective forecast')
    dates = slots(calendar(root), locked['first_origin'])
    if origin not in dates or stamp <= utc(locked['locked_at']):
        raise ValueError('Origin outside locked prospective cohort')
    year = pd.Timestamp(origin).year; annual = [d for d in dates if pd.Timestamp(d).year == year]
    bucket = annual.index(origin)//21
    pointer = read_record(root/'fits'/f'{year}-outer-{bucket}.json')
    ledger = Ledger(root/'ledger', {**ready['identity'], 'cohort_lock_id': content_id(locked)})
    record = read_record(ledger.root/'completed'/(pointer['fit_id']+'.json'))
    record = native_record(ledger, {k: v for k, v in record['specification'].items() if k != 'repeat_reason'})
    h = history(seed)
    expected_job = {'year': year, 'role': 'outer', 'bucket': bucket, 'cutoff': annual[bucket*21],
                    'origins': [], 'train_dates': mature(h, annual[bucket*21])}
    expected_spec = specification(expected_job, projection(h, expected_job['train_dates']), record['specification']['observation_id'])
    if (record['specification'] != {**expected_spec, 'repeat_reason': None}
            or pointer['specification_id'] != content_id(expected_spec)):
        raise ValueError('Wrong bucket or mature training state at forecast origin')
    selection = verify_selection(root, ledger, h, year, annual[0])
    if utc(selection['created_at']) > cutoff or utc(record['completed_at']) > cutoff:
        raise ValueError('Model or shrinkage was not available at decision time')
    rows = projection(h, [origin]); rows['variance_target_1'] = np.nan
    raw = float(predict_model(ledger, record, rows)[0]); baseline = float(rows.ct_ewma.iloc[0]); w = selection['weight']
    body = {'origin': origin, 'ordinal_slot': dates.index(origin), 'decision_at': cutoff.isoformat(),
            'generated_at': stamp.isoformat(), 'observation_id': observation['observation_id'],
            'history_sha256': frame_identity(seed, list(seed)), 'fit_id': record['experiment_id'],
            'selection_id': content_id(selection), 'cohort_lock_id': content_id(locked),
            'calendar_id': content_id(calendar_body(root)),
            'baseline_variance': baseline, 'raw_variance': raw, 'weight': w,
            'selected_variance': w*raw+(1-w)*baseline, 'market_skill': False}
    directory = root/'forecasts'/origin
    with writer(root):
        if directory.exists():
            raise ValueError('Preserve existing completed or partial forecast')
        freeze_record(directory/'forecast.json', body)
        completed = now()
        freeze_record(directory/'publication-receipt.json', {'forecast_sha256': digest(directory/'forecast.json'),
            'completed_at': completed.isoformat(), 'status': 'TIMELY' if completed < cutoff+pd.Timedelta(minutes=15) else 'LATE'})
        previous = None
        for index, day in enumerate(dates[:dates.index(origin)+1]):
            event_path = root/'events'/f'{index:03d}.json'
            if event_path.exists():
                event = read_record(event_path)
                if event['previous_event_id'] != previous or event['origin'] != day:
                    raise ValueError('Forward event chain changed')
            else:
                if day != origin and decision(day)+pd.Timedelta(minutes=15) > stamp:
                    raise ValueError('Cannot mark a future slot missing')
                event = {'ordinal_slot': index, 'origin': day, 'previous_event_id': previous,
                         'created_at': now().isoformat(), 'kind': 'FORECAST' if day == origin else 'MISSING',
                         'forecast_sha256': digest(directory/'forecast.json') if day == origin else None,
                         'publication_receipt_sha256': digest(directory/'publication-receipt.json') if day == origin else None}
                freeze_record(event_path, event)
            previous = content_id(event)
    return {'origin': origin, 'status': read_record(directory/'publication-receipt.json')['status'], 'new_fits': 0}


def status(root, repo):
    if (root/'.writer-lock').exists():
        raise RuntimeError('A writer is active or interrupted; do not read partial publication as complete')
    _, design = verify(root, repo)
    if not (root/'cohort-lock.json').exists():
        return {'status': 'not_locked', 'performance_scored': False, 'automatic_tasks': False}
    locked = read_record(root/'cohort-lock.json'); dates = slots(calendar(root), locked['first_origin']); stamp = now()
    elapsed = [d for d in dates if decision(d)+pd.Timedelta(minutes=15) <= stamp]
    timely = []
    previous = None; event_forecasts = set()
    for index, path in enumerate(sorted((root/'events').glob('*.json'))):
        event = read_record(path)
        if (path.stem != f'{index:03d}' or index >= len(dates) or event['ordinal_slot'] != index or event['origin'] != dates[index]
                or event['previous_event_id'] != previous):
            raise ValueError('Missing, reordered or altered forward event')
        if event['kind'] == 'FORECAST':
            event_forecasts.add(event['origin'])
            directory = root/'forecasts'/event['origin']
            if (event['forecast_sha256'] != digest(directory/'forecast.json')
                    or event['publication_receipt_sha256'] != digest(directory/'publication-receipt.json')):
                raise ValueError('Forward event payload changed')
        elif event['kind'] != 'MISSING':
            raise ValueError('Unsupported forward event')
        previous = content_id(event)
    for day in dates:
        p = root/'forecasts'/day/'publication-receipt.json'
        if p.exists():
            if day not in event_forecasts:
                raise ValueError('Partial publication lacks its completed forward event')
            receipt = read_record(p); forecast = read_record(p.parent/'forecast.json')
            if receipt['forecast_sha256'] != digest(p.parent/'forecast.json') or forecast['origin'] != day:
                raise ValueError('Published forecast chain changed')
            if receipt['status'] == 'TIMELY':
                cutoff = decision(day)
                if (not cutoff <= utc(forecast['generated_at']) <= utc(receipt['completed_at']) < cutoff+pd.Timedelta(minutes=15)
                        or forecast['cohort_lock_id'] != content_id(locked)):
                    raise ValueError('False timely publication receipt')
                timely.append(day)
    return {'status': 'collecting', 'known_calendar_slots': len(dates), 'required_origins': design['required_origins'],
            'timely_forecasts': len(timely), 'elapsed_missing_origins': [d for d in elapsed if d not in timely],
            'calendar_extension_required': len(dates) < 252, 'performance_scored': False, 'automatic_tasks': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'lock', 'fit', 'admit', 'capture', 'publish', 'status', 'extend-calendar'))
    parser.add_argument('--repo', type=Path, required=True); parser.add_argument('--root', type=Path, required=True)
    for name in ('admission', 'parent', 'authorization', 'raw', 'receipt', 'calendar-review'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--origin'); parser.add_argument('--minutes', type=float, default=30)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.repo, args.root, args.admission, args.parent)
    elif args.command == 'verify':
        result = {'status': 'verified', 'identity': verify(args.root, args.repo)[0]['identity']}
    elif args.command == 'lock':
        result = lock(args.root, args.repo, args.authorization)
    elif args.command == 'fit':
        result = fit_for_origin(args.root, args.repo, args.authorization, args.origin, args.minutes)
    elif args.command == 'admit':
        result = admit(args.root, args.repo, args.raw, args.receipt)
    elif args.command == 'capture':
        result = capture(args.root, args.repo)
    elif args.command == 'publish':
        result = publish(args.root, args.repo)
    elif args.command == 'extend-calendar':
        result = extend_calendar(args.root, args.repo, args.calendar_review, args.raw, args.receipt)
    else:
        result = status(args.root, args.repo)
    print(json.dumps(result, indent=2))
