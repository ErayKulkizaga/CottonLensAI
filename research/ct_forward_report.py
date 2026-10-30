"""End-only, zero-fit evaluation of the frozen v4 manual risk study."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.volatility import qlike

from research import ct_manual_forward as forward
from research.ct_manual_source import CottonCalendar, classify_chart, verified_response
from research.ct_prospective_readiness import append_snapshot
from research.verify_ct_prospective_readiness import decimal_difference
from research.verify_ct_signed_risk_score import independent_history
from research.verify_risk_weak_teacher import difference_interval

PROFILE = 'ct-own-level-forward-end-report-v1'
HELPERS = ('research/ct_forward_report.py', 'research/tests/test_ct_forward_report.py',
           'research/verify_ct_prospective_readiness.py', 'research/verify_ct_signed_risk_score.py',
           'research/verify_risk_weak_teacher.py')


def completeness_scenarios():
    """Hypothetical operational probabilities only; never estimate actual uptime."""
    rows = []
    for failure in (.001, .005, .01, .02):
        rows.append({'assumed_daily_failure_probability': failure,
                     'iid_probability_all252_present': (1-failure)**252,
                     'arbitrary_dependence_complete_probability_bounds': [max(0., 1-252*failure), 1-failure]})
    return {'assumptions_only_not_estimated_uptime': True, 'forecast_skill_probability': False,
            'all252_is_new_operational_gate_not_original_human_risk_gate': True, 'scenarios': rows}


def score_pairs(frame):
    """Keep all slots; incomplete coverage is inconclusive, never a market negative."""
    if (len(frame) != 252 or frame.origin.duplicated().any() or not frame.origin.is_monotonic_increasing
            or frame.horizon.ne(1).any() or frame.target_date.le(frame.origin).any()
            or not frame.published.map(lambda x: isinstance(x, (bool, np.bool_))).all()
            or not np.isfinite(frame.actual_variance).all() or frame.actual_variance.lt(0).any()):
        raise ValueError('Exact252 ordered unique T1 slots, valid targets and explicit publication flags required')
    paired = frame.loc[frame.published].copy()
    if len(paired):
        paired['baseline_loss'] = qlike(paired.actual_variance, paired.baseline_variance)
        paired['selected_loss'] = qlike(paired.actual_variance, paired.selected_variance)
        paired['delta'] = [decimal_difference(y, b, m) for y, b, m in zip(
            paired.actual_variance, paired.baseline_variance, paired.selected_variance, strict=True)]
        np.testing.assert_allclose(paired.delta, paired.baseline_loss-paired.selected_loss, rtol=1e-10, atol=1e-12)
        mean = float(paired.delta.mean())
        yearly = {str(y): float(g.delta.mean()) for y, g in paired.groupby(paired.origin.str[:4])}
        trimmed = paired.drop(paired.nlargest(math.ceil(len(paired)*.01), 'baseline_loss').index)
        trim = float(trimmed.delta.mean()) if len(trimmed) else None
    else:
        mean, yearly, trim = None, {}, None
        for name in ('baseline_loss', 'selected_loss', 'delta'):
            paired[name] = pd.Series(dtype=float)
    complete = len(paired) == 252
    intervals = None
    if complete:
        parts = [np.column_stack([g.delta, np.ones(len(g))]) for _, g in paired.groupby(paired.origin.str[:4], sort=True)]
        intervals = {str(block): difference_interval(parts, block) for block in (20, 60)}
    supported = bool(complete and mean > 0 and all(v[0] > 0 for v in intervals.values()))
    if not complete:
        verdict = 'INCONCLUSIVE_INCOMPLETE_PUBLICATION'
    elif supported:
        verdict = 'FORWARD_STATISTICAL_SUPERIORITY'
    elif all(v[1] <= 0 for v in intervals.values()):
        verdict = 'FIXED_RECIPE_SUPERIORITY_NOT_SUPPORTED'
    else:
        verdict = 'INCONCLUSIVE'
    return {'classification': verdict, 'required_slots': 252, 'paired_slots': len(paired),
            'missing_or_late_slots': 252-len(paired), 'all_slots_retained': True,
            'mean_paired_qlike_gain': mean, 'gain_ci': intervals, 'yearly_gain': yearly,
            'trim_top1pct_baseline_loss_gain': trim, 'trim_is_gate': False,
            'confirmatory_gate_passed': supported, 'incomplete_result_is_no_signal_evidence': False,
            'economic_or_price_direction_success': False,
            'claim_scope': 'Frozen Yahoo-proxy squared-return forecast recipe; operational completeness and skill are distinct.'}, paired


def require_end(root, repo):
    ready, design = forward.verify(root, repo)
    if not (root/'cohort-lock.json').exists():
        raise ValueError('No real cohort lock; no performance may be scored')
    lock = read_record(root/'cohort-lock.json'); cal = forward.calendar(root)
    if (lock['identity'] != ready['identity'] or lock['required_origins'] != 252
            or lock['first_origin'] != forward.first_origin(cal, forward.utc(lock['locked_at']))):
        raise ValueError('Locked cohort identity or first origin changed')
    dates = forward.slots(cal, lock['first_origin'])
    if len(dates) != 252:
        raise ValueError('Full252slot official calendar is unavailable; no partial score')
    last_target = forward.target_date(cal, dates[-1])
    if forward.now() < cal.completion(pd.Timestamp(last_target).date()):
        raise ValueError('Last target has not completed; no intermediate score')
    if design['no_partial_score'] is not True:
        raise ValueError('Unexpected scoring protocol')
    forward.status(root, repo)  # verifies publication/event hashes, timestamps and target dates
    return ready, lock, cal, dates


def study_hashes(root):
    files = {}
    for p in root.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or '.writer-lock' in p.parts:
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Study input cannot escape its frozen root')
        files[p.relative_to(root).as_posix()] = digest(p)
    return files


def observation_history(root):
    """Rebuild every admitted source transition from preserved raw bytes, without network."""
    final, latest = forward.current_history(root)
    if latest is None or forward.utc(latest['observed_available_at']) > forward.now():
        raise ValueError('A real final source observation is required')
    initial = pd.read_parquet(root/'inputs/seed.parquet'); combined = initial
    calendars = {content_id(json.loads((root/'inputs/calendar.json').read_bytes())):
                 json.loads((root/'inputs/calendar.json').read_bytes())}
    for p in (root/'calendar-extensions').glob('*/extension.json'):
        body = read_record(p)['calendar']; calendars[content_id(body)] = body
    records = [read_record(p) for p in (root/'observations').glob('*/receipt.json')]
    records.sort(key=lambda r: forward.utc(r['observed_available_at']))
    frames, receipts = {}, {}
    for r in records:
        base = root/'observations'/r['observation_id']
        http = verified_response(base/'chart.json', base/'http-receipt.json')
        if (http['observed_available_at'] != r['observed_available_at'] or r['calendar_id'] not in calendars
                or http.get('source_url') != 'https://query1.finance.yahoo.com/v8/finance/chart/CT%3DF?range=2y&interval=1d&includePrePost=false'
                or r['observation_id'] != content_id({'source_sha256': digest(base/'chart.json'),
                                                     'receipt_sha256': digest(base/'http-receipt.json')})):
            raise ValueError('Source receipt/calendar identity changed')
        _, quotes = classify_chart(json.loads((base/'chart.json').read_bytes()), http, CottonCalendar(calendars[r['calendar_id']]))
        combined = append_snapshot(combined, quotes)
        stored = pd.read_parquet(base/'history.parquet')
        pd.testing.assert_frame_equal(combined, stored, check_exact=True)
        pd.testing.assert_frame_equal(final.iloc[:len(stored)].reset_index(drop=True), stored, check_exact=True)
        if r['latest_completed_date'] != str(stored.date.iloc[-1].date()):
            raise ValueError('Incorrect source final-session label')
        frames[r['observation_id']], receipts[r['observation_id']] = stored, r
    pd.testing.assert_frame_equal(combined, final, check_exact=True)
    return final, latest, frames, receipts


def extract(root, repo):
    ready, lock, cal, dates = require_end(root, repo)
    final, latest, frames, receipts = observation_history(root)
    h = forward.history(final); checked = independent_history(h)
    ledger = Ledger(root/'ledger', {**ready['identity'], 'cohort_lock_id': content_id(lock)})
    consumed, keys = fit_consumption(ledger.root)
    completed = [read_record(p) for p in (ledger.root/'completed').glob('*.json')]
    if (consumed != len(keys) or len(keys) != len(completed) or consumed > 31
            or len({r['specification']['role'] for r in completed}) != len(completed)):
        raise ValueError('Excess, incomplete, failed or duplicate-role fits; no selection after evaluation')
    selection_cache = {}; certified = set(); rows = []
    for index, day in enumerate(dates):
        target = forward.target_date(cal, day)
        if day not in checked.index or target not in checked.index:
            raise ValueError('Missing realized endpoint; do not fill or intersect targets')
        position = h.index[h.date.eq(pd.Timestamp(day))][0]
        if str(h.date.iloc[position+1].date()) != target:
            raise ValueError('Observed target endpoint differs from official next session')
        actual = float(checked.loc[day, 'checked_actual'])
        row = {'ordinal_slot': index, 'origin': day, 'target_date': target, 'horizon': 1,
               'actual_variance': actual, 'published': False, 'publication_status': 'MISSING',
               'baseline_variance': None, 'raw_variance': None, 'selected_variance': None}
        directory = root/'forecasts'/day
        if directory.exists():
            if not all((directory/n).exists() for n in ('forecast.json', 'publication-receipt.json')):
                raise ValueError('Partial publication is not a completed experiment')
            forecast = read_record(directory/'forecast.json'); pub = read_record(directory/'publication-receipt.json')
            if pub['status'] not in ('TIMELY', 'LATE'):
                raise ValueError('Unsupported publication status')
            oid = forecast['observation_id']; stamp = forward.decision(day)
            if (oid not in frames or forward.utc(receipts[oid]['observed_available_at']) > stamp
                    or receipts[oid]['latest_completed_date'] != day
                    or forecast['history_sha256'] != frame_identity(frames[oid], list(frames[oid]))):
                raise ValueError('Forecast consumed an unproved or postcutoff source')
            record = read_record(ledger.root/'completed'/(forecast['fit_id']+'.json'))
            record = forward.native_record(ledger, {k: v for k, v in record['specification'].items() if k != 'repeat_reason'})
            annual = [d for d in dates if d[:4] == day[:4]]; bucket = annual.index(day)//21
            job = {'year': int(day[:4]), 'role': 'outer', 'bucket': bucket, 'cutoff': annual[bucket*21],
                   'origins': [], 'train_dates': forward.mature(h, annual[bucket*21])}
            spec = forward.specification(job, forward.projection(h, job['train_dates']), record['specification']['observation_id'])
            if record['specification'] != {**spec, 'repeat_reason': None} or forward.utc(record['completed_at']) > stamp:
                raise ValueError('Wrong model bucket, mature training population or completion time')
            training_observation = spec['observation_id']
            if training_observation == 'initial_seed':
                availability = json.loads((root/'inputs/initial-receipt.json').read_bytes())['observed_available_at']
            elif training_observation in receipts:
                availability = receipts[training_observation]['observed_available_at']
            else:
                raise ValueError('Unproved training source')
            if forward.utc(availability) > forward.utc(record['completed_at']):
                raise ValueError('Training source arrived after model completion')
            year = day[:4]
            if year not in selection_cache:
                selection_cache[year] = forward.verify_selection(root, ledger, h, int(year), annual[0])
            selection = selection_cache[year]
            if (content_id(selection) != forecast['selection_id'] or forward.utc(selection['created_at']) > stamp
                    or forecast['weight'] != selection['weight']):
                raise ValueError('Changed or future shrinkage selection')
            model = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_bytes())
            if (model['features'] != forward.NAMES or model['effective_parameters'] != 5
                    or model['recipe'] != spec['recipe'] or len(model['coef']) != 5):
                raise ValueError('Wrong native learner/feature payload')
            if record['experiment_id'] not in certified:
                train = forward.learner.relative(forward.projection(h, job['train_dates']), True, forward.learner.STATE_PROFILE)
                x_train = train[forward.NAMES].to_numpy()
                eta = np.column_stack([np.ones(len(train)), x_train])@model['coef']
                ratio = train.variance_target_1.to_numpy()*np.exp(-eta)
                opt = model['optimization']; z = (x_train-opt['training_feature_mean'])/opt['training_feature_scale']
                gradient = np.column_stack([np.ones(len(train)), z]).T@(1-ratio)/len(train)
                if (not opt['success'] or np.max(np.abs(gradient)) > 1.001e-6
                        or not math.isclose(float(np.mean(eta+ratio)), opt['final_training_qlike'], abs_tol=1e-10)):
                    raise ValueError('Uncertified native training loss/gradient')
                certified.add(record['experiment_id'])
            baseline = float(checked.loc[day, 'checked_ewma'])
            x = [float(checked.loc[day, 'checked_'+n])-math.log(baseline) for n in ('har_1', 'har_5', 'har_22')]
            raw = baseline*math.exp(np.dot([1., *x, math.log(baseline)], model['coef']))
            selected = selection['weight']*raw+(1-selection['weight'])*baseline
            np.testing.assert_allclose([baseline, raw, selected],
                [forecast['baseline_variance'], forecast['raw_variance'], forecast['selected_variance']], rtol=1e-10, atol=1e-14)
            row.update(published=pub['status'] == 'TIMELY', publication_status=pub['status'],
                       baseline_variance=baseline, raw_variance=raw, selected_variance=selected)
        rows.append(row)
    return pd.DataFrame(rows), latest


def audit(root, repo, output):
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Use a new output outside the preserved study')
    require_end(root, repo)  # must run before reading numeric predictions or computing any loss
    before = study_hashes(root)
    frame, final_source = extract(root, repo); summary, paired = score_pairs(frame)
    analysis = {'protocol_source': forward.source(repo), 'helpers': {n: digest(repo/n) for n in HELPERS}}
    with writer(output):
        freeze_record(output/'study-files.json', {'files': before})
        freeze_record(output/'analysis-manifest.json', analysis)
        for name in {**analysis['protocol_source']['ml']['files'], **analysis['protocol_source']['helpers'], **analysis['helpers']}:
            dest = output/'source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, dest)
        frame.to_csv(output/'all-slots.csv', index=False, lineterminator='\n')
        paired.to_csv(output/'paired-losses.csv', index=False, lineterminator='\n')
        if before != study_hashes(root):
            raise ValueError('Study changed during end-only audit; no performance verdict')
        freeze_record(output/'result.json', {'profile': PROFILE, 'analysis_source_id': content_id(analysis),
            'new_fits': 0, 'new_data_downloads': 0, 'original_study_unchanged': True,
            'final_source_observed_at': final_source['observed_available_at'], **summary})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('root', 'repo', 'output'):
        parser.add_argument('--'+n, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.repo, args.output), indent=2))
