"""Offline NASS Texas T+1 preregistration; no fitting or source admission."""
import argparse
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.diagnostics import VALIDATION_POLICY
from cottonlens_ml.research.full_year import SHRINKAGE, chunks
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import freeze_record, writer
from cottonlens_ml.research.protocol import full_year_manifest, rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import training_rows, verify_history
from cottonlens_ml.sources.nass_regional import CATEGORIES, STAGES

PROFILE = 'nass-regional-t1-pilot-v1'
DELAYS = (0, 1)
MAX_AGE = 10
NATIONAL = ['national_good_excellent', 'national_poor_very_poor', 'national_ge_change']
REGIONAL = ['texas_minus_national_ge', 'texas_minus_national_pvp', 'texas_minus_national_ge_change',
            *['texas_' + stage.lower().replace(' ', '_') + '_gap' for stage in STAGES]]
RECIPE = {'family': 'ridge', 'params': {'alpha': 1.}, 'horizon': 1, 'task': 'price',
          'device': 'cpu', 'seed': 42, 'target': 'scaled_log', 'years': None, 'cadence': 21, 'window': 1}
CLOCK = {'decision': 'Cotton source date + 1 day 00:15 UTC',
         'assumed_value_available_at': 'printed release date + 1 day 00:00 UTC',
         'basis': 'explicit_unverified_availability_and_vintage_assumption',
         'publication_verified': False, 'first_version_verified': False,
         'delays': list(DELAYS), 'delay_unit': 'additional recorded Cotton decision',
         'max_age_sessions': MAX_AGE, 'expiry_anchor': 'D0 first assumed eligible decision for both arms/delays',
         'winter_carry': False, 'absent_stage': 'unknown; never carry an older stage across a newer report'}
RULES = {'primary': 'selected numeric_D0 vs mask_D0 paired T+1 price-MAE; incremental Texas over national condition',
         'secondary': 'numeric_D1 vs mask_D1; do not select delay using OOS results',
         'baseline': 'same-origin zero return / Naive; no fit required',
         'bootstrap': {'blocks': [20, 60], 'repetitions': 10000, 'seed': 42, 'within_year': True},
         'practical_reference': {'price_mae_gain_pct': 5, 'direction_pct': 53, 'years_won': '6/8'},
         'diagnostics': ['raw and shrunk predictions', 'annual results and active/all-origin direction',
                         'largest ceil(1%*n) Naive errors removed with date tie break',
                         'source-age 0..4 and stage coverage; diagnostic only'],
         'decision_order': ['invalid identity/alignment/payload: no scientific conclusion',
                            'both delays: control gain lower bounds >0 at blocks20/60; D0 Naive point gain>=5%, direction>=53%, years>=6/8: assumption-conditional candidate, no release',
                            'both delays: Naive gain upper bounds<5% at blocks20/60: fixed recipe below practical goal; keep Naive, no automatic grid',
                            'both delays: control improvement supported but practical gates fail: record limited contribution',
                            'otherwise inconclusive/delay-sensitive; no OOS delay selection or automatic extra fitting'],
         'independent_holdout': False, 'automatic_release': False, 'source_admission': False,
         'interpretation': 'Reused history, partial stage coverage and unverified vintage/clocks; no universal no-signal or production claim.'}


def condition_values(values):
    if set(values) != set(CATEGORIES):
        raise ValueError('Five condition categories required')
    numbers = [v['value'] if isinstance(v, dict) else v for v in (values[c] for c in CATEGORIES)]
    if any(v is None or not np.isfinite(v) or not 0 <= v <= 100 for v in numbers) or sum(numbers) != 100:
        raise ValueError('Finite bounded condition percentages summing to100 required')
    return (numbers[3] + numbers[4]) / 100., (numbers[0] + numbers[1]) / 100.


def report_features(panel):
    if (panel.get('model_eligible') is not False or panel.get('release_allowed') is not False
            or panel.get('availability_policy') != 'UNSET' or not panel.get('panel')
            or len(panel['panel']) != panel.get('reports')):
        raise ValueError('Unadmitted complete regional quarantine panel required')
    result = []
    for item in panel['panel']:
        week, released = pd.Timestamp(item['week_ending']), pd.Timestamp(item['release_day'])
        if (pd.isna(week) or pd.isna(released) or week.tz is not None or released.tz is not None or week != week.normalize()
                or released != released.normalize() or week.dayofweek != 6
                or released < week or released >= pd.Timestamp('2024-01-01')
                or item.get('available_at') is not None or item.get('first_version_verified') is not False):
            raise ValueError('Date-only observation/release and unset vintage/clock required')
        ge, pvp = condition_values(item['national_condition'])
        txge, txpvp = condition_values(item['texas_condition'])
        record = {'week_ending': week, 'report_date': released, 'report_sha256': item['report_sha256'],
                  NATIONAL[0]: ge, NATIONAL[1]: pvp,
                  REGIONAL[0]: txge - ge, REGIONAL[1]: txpvp - pvp}
        if set(item['progress']) != set(STAGES):
            raise ValueError('Five stages required, including explicit absent sections')
        for stage, column in zip(STAGES, REGIONAL[3:], strict=True):
            progress = item['progress'][stage]
            if progress['section_present'] is False:
                if progress['cells'] is not None or progress['gap_to_average_pp'] is not None:
                    raise ValueError('Absent stage must remain unknown')
                record[column] = np.nan
                continue
            if progress['section_present'] is not True or progress['published_average_years'] != [week.year - 5, week.year - 1]:
                raise ValueError('Published prior-five-year average required')
            current, average = [progress['cells'][c]['value'] for c in ('current', 'published_average')]
            if any(v is not None and (not np.isfinite(v) or not 0 <= v <= 100) for v in (current, average)):
                raise ValueError('Invalid stage percentage')
            gap = None if current is None or average is None else current - average
            if progress['gap_to_average_pp'] != gap:
                raise ValueError('Stage gap differs from current/published average')
            record[column] = np.nan if gap is None else gap / 100.
        result.append(record)
    frame = pd.DataFrame(result)
    if (frame.week_ending.duplicated().any() or frame.report_date.duplicated().any()
            or not frame.week_ending.is_monotonic_increasing or not frame.report_date.is_monotonic_increasing):
        raise ValueError('Unique chronological reports required')
    consecutive = frame.week_ending.diff().dt.days.eq(7) & frame.week_ending.dt.year.eq(frame.week_ending.shift().dt.year)
    # Use the prior report's own current values, never a later revised previous-week column.
    frame[NATIONAL[2]] = frame[NATIONAL[0]].diff().where(consecutive)
    frame[REGIONAL[2]] = frame[REGIONAL[0]].diff().where(consecutive)
    return frame


def groups():
    return {f'{arm}_D{delay}': [*FEATURE_NAMES, f'nass_D{delay}_age_sessions', f'nass_D{delay}_unavailable',
                              *[f'nass_D{delay}_{f}' for f in NATIONAL],
                              *[f'{arm}_D{delay}_{f}' for f in REGIONAL]]
            for delay in DELAYS for arm in ('mask', 'numeric')}


def align_assumed(history, reports):
    if (history.empty or history.date.isna().any() or history.date.duplicated().any()
            or not history.date.is_monotonic_increasing or history.date.dt.tz is not None
            or not history.date.eq(history.date.dt.normalize()).all()
            or reports.empty or reports.report_date.isna().any() or reports.report_date.duplicated().any()
            or not reports.report_date.is_monotonic_increasing):
        raise ValueError('Unique sorted source and date-only Cotton history required')
    decision = pd.to_datetime(history.date, utc=True) + pd.Timedelta(days=1, minutes=15)
    assumed = pd.to_datetime(reports.report_date, utc=True) + pd.Timedelta(days=1)
    first = np.searchsorted(decision.to_numpy(), assumed.to_numpy(), side='left')
    added = {'nass_decision_time': decision.to_numpy()}
    values = reports[[*NATIONAL, *REGIONAL]].to_numpy(dtype=float)
    for delay in DELAYS:
        selected = np.searchsorted(first + delay, np.arange(len(history)), side='right') - 1
        found, safe = selected >= 0, np.maximum(selected, 0)
        age = np.where(found, np.arange(len(history)) - first[safe], np.nan)
        same_year = history.date.dt.year.to_numpy() == reports.week_ending.dt.year.to_numpy()[safe]
        usable = found & (age <= MAX_AGE) & same_year
        matrix = np.full((len(history), len(NATIONAL) + len(REGIONAL)), np.nan)
        matrix[usable] = values[selected[usable]]
        added[f'nass_D{delay}_age_sessions'] = age
        added[f'nass_D{delay}_unavailable'] = (~usable).astype(float)
        for i, name in enumerate(NATIONAL):
            added[f'nass_D{delay}_{name}'] = matrix[:, i]
        for i, name in enumerate(REGIONAL, len(NATIONAL)):
            column = matrix[:, i]
            added[f'numeric_D{delay}_{name}'] = column
            added[f'mask_D{delay}_{name}'] = np.where(np.isfinite(column), 0., np.nan)
        for field, source in [('report_date', reports.report_date), ('week_ending', reports.week_ending),
                              ('assumed_available_at', assumed), ('report_sha256', reports.report_sha256)]:
            series = source.reset_index(drop=True).iloc[safe].reset_index(drop=True)
            series.loc[~found] = None if field == 'report_sha256' else pd.NaT
            added[f'nass_D{delay}_{field}'] = series.to_numpy()
        times = pd.to_datetime(added[f'nass_D{delay}_assumed_available_at'], utc=True)
        if (times[found] > decision.to_numpy()[found]).any():
            raise ValueError('Source after decision detected')
    if set(added) & set(history.columns):
        raise ValueError('NASS features already present; never overwrite source history')
    result = pd.concat([history.copy(), pd.DataFrame(added, index=history.index)], axis=1)
    result.attrs = {}
    return result


def make_design(history, reports):
    expanded = align_assumed(history, reports)
    split = full_year_manifest(history)
    inner = len(groups()) * sum(len(chunks(history, b['origins'])) for f in split['folds'] for b in f['inner'])
    outer = len(groups()) * sum(len(chunks(history, f['origins'])) for f in split['folds'])
    coverage = []
    for fold in split['folds']:
        for role, dates in [('outer', fold['origins']), *[(f'inner_{i+1}', b['origins']) for i, b in enumerate(fold['inner'])]]:
            rows = rows_at(expanded, dates)
            train = training_rows(expanded, rows.date.min(), RECIPE, None)
            if len(train) < 500 or train.target_date_5.ge(rows.date.min()).any():
                raise ValueError('Insufficient mature/common-warmup history')
            for delay in DELAYS:
                coverage.append({'year': fold['year'], 'role': role, 'delay': delay, 'origins': len(rows),
                                 'mature_train_rows': len(train),
                                 'distinct_usable_train_reports': int(train.loc[train[f'nass_D{delay}_unavailable'].eq(0), f'nass_D{delay}_report_sha256'].nunique()),
                                 'distinct_usable_evaluation_reports': int(rows.loc[rows[f'nass_D{delay}_unavailable'].eq(0), f'nass_D{delay}_report_sha256'].nunique()),
                                 'unavailable_origins': int(rows[f'nass_D{delay}_unavailable'].sum()),
                                 **{f'known_{name}': int(rows[f'numeric_D{delay}_{name}'].notna().sum()) for name in REGIONAL}})
    origins = sum(len(f['origins']) for f in split['folds'])
    design = {'profile': PROFILE, 'experiment': 'research-nass-regional-t1-pilot-v1',
              'hypothesis': 'Texas-vs-national condition and published stage-average gaps add T+1 information beyond fixed Cotton core, national condition and identical missingness/timing.',
              'groups': groups(), 'recipe': RECIPE, 'split': split, 'clock': CLOCK, 'rules': RULES,
              'shrinkage_weights': list(SHRINKAGE),
              'selection': 'fixed Ridge; only shrinkage selected separately per arm by past 3x63 mean relative price-MAE; ties choose smaller weight',
              'preprocessing': 'existing train-only median/mean/std and automatic missing indicators; shared regional null patterns',
              'fit_budget': {'inner': inner, 'outer': outer, 'total': inner + outer,
                             'annual_outputs': len(groups()) * len(split['folds']), 'prediction_rows': origins * len(groups()),
                             'processes': 1, 'threads': 2, 'session_minutes': 30},
              'learning_control': {'family': 'ridge', 'policy': VALIDATION_POLICY, 'fits': 1,
                                   'minimum_known_signal_mae_gain': .5, 'market_evidence': False,
                                   'failure_action': 'stop before market fitting; never depend on unused TCN/GPU'},
              'target': 'T+1 standardized log return to next recorded Cotton observation; existing price reconstruction',
              'maturity': 'target_date_5 < cutoff; same 120-observation warmup and no feature-complete filtering',
              'feature_data_id': frame_identity(expanded, list(expanded.columns)),
              'deliberate_repeat_reason': 'National condition T+5 is related evidence; neither regional Texas/progress nor T+1 increment above national was tested. No model search.',
              'release_allowed': False, 'historical_source_admitted': False, 'fits_completed': 0,
              'execution_status': 'preregistered_not_runnable; no engine profile or admission receipt'}
    return expanded, design, coverage


def prepare(repo, output, reference, panel_path):
    repo, output, reference, panel_path = map(Path, (repo, output, reference, panel_path))
    validate(repo / 'research')
    registry, trials = load_registry(repo / 'research')
    evidence = repo / 'research/evidence'
    numeric = json.loads((evidence / 'nass-regional-audit-20261009.json').read_bytes())
    clock = json.loads((evidence / 'nass-clock-case-20261009.json').read_bytes())
    if (digest(panel_path) != numeric['panel_sha256'] or clock['historical_availability_verified'] is not False
            or clock['admitted_rows'] != 0 or clock['market_fits'] != 0):
        raise ValueError('Registered regional panel and unverified clock case required')
    parent, history = verify_history(reference)
    if parent['identity'].get('profile') != 'full-year-v1':
        raise ValueError('Frozen full-year reference required')
    reports = report_features(json.loads(panel_path.read_bytes()))
    expanded, design, coverage = make_design(history, reports)
    if design['fit_budget'] != {'inner': 288, 'outer': 388, 'total': 676, 'annual_outputs': 32,
                                'prediction_rows': 8024, 'processes': 1, 'threads': 2, 'session_minutes': 30}:
        raise ValueError('Approved eight-year common cohort/fit budget differs')
    inputs = {'reference/ready.json': reference / 'ready.json', 'reference/history.parquet': reference / 'history.parquet',
              'candidate/panel.json': panel_path, 'registry/registry.json': repo / 'research/registry.json',
              'registry/trials.json': repo / 'research/trials.json',
              'evidence/nass-regional-audit-20261009.json': evidence / 'nass-regional-audit-20261009.json',
              'evidence/nass-clock-case-20261009.json': evidence / 'nass-clock-case-20261009.json'}
    hashes = {name: digest(path) for name, path in inputs.items()}
    source = research_source_identity(repo)
    identity = {'source_id': source['source_id'], 'input_sha256': hashes, 'design': design,
                'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in
                    ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')},
                'lock_sha256': digest(repo / 'ml/uv.lock'), 'fits_completed': 0}
    related = check(registry, trials, query='nass')
    proposed = []
    for group, features in design['groups'].items():
        result = check(registry, trials, proposal={'scope_id': content_id(identity), 'recipe': {**RECIPE, 'features': features}})
        if result['exact_fit_recipes']:
            raise ValueError('Frozen recipe already fitted')
        proposed.append({'group': group, 'status': result['status'], 'exact_matches': 0,
                         'related_fit_recipes': len(result['fit_recipes'])})
    registration = {'profile': PROFILE, 'identity': identity, 'registration_id': content_id(identity),
                    'created_at': datetime.now(UTC).isoformat(), 'fits': 0,
                    'completed_preregistration': True, 'completed_experiment': False,
                    'availability_verified': False, 'model_eligible': False}
    with writer(output):
        if any(p.name != '.writer-lock' for p in output.iterdir()):
            raise ValueError('Preserve existing/partial registration; use new namespace')
        for name, path in inputs.items():
            copy_immutable(path, output / 'inputs' / name)
        for name, expected in source['files'].items():
            copy_immutable(repo / name, output / 'source-snapshot' / name)
            if digest(output / 'source-snapshot' / name) != expected:
                raise ValueError('Copied source changed')
        expanded.to_parquet(output / 'history.parquet', index=False)
        reports.to_parquet(output / 'report-features.parquet', index=False)
        pd.DataFrame(coverage).to_csv(output / 'source-coverage.csv', index=False)
        freeze_record(output / 'source-manifest.json', source)
        freeze_record(output / 'history-check.json', {'related_experiments': related['experiments'],
                      'related_fit_recipes': related['fit_recipes'], 'proposals': proposed, 'fits': 0})
        freeze_record(output / 'preregistered.json', registration)
        if research_source_identity(repo) != source or any(digest(path) != hashes[name] or digest(output / 'inputs' / name) != hashes[name] for name, path in inputs.items()):
            raise ValueError('Code/data/registry changed during registration')
        files = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*')
                 if p.is_file() and '.writer-lock' not in p.parts}
        freeze_record(output / 'complete.json', {'profile': PROFILE, 'completed': True, 'fits': 0,
                      'completed_experiment': False, 'availability_verified': False, 'model_eligible': False,
                      'registration_id': registration['registration_id'], 'files': files})
    return registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'output', 'reference', 'panel'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.repo, args.output, args.reference, args.panel)
    print({'registration_id': result['registration_id'], 'fits': 0,
           'fit_budget': result['identity']['design']['fit_budget']})


if __name__ == '__main__':
    main()
