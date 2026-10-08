"""Offline preregistration only. No fitting or verified source admission."""
import argparse
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.ams_exploration import short_split
from cottonlens_ml.research.full_year import SHRINKAGE, chunks
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import freeze_record, read_record, writer
from cottonlens_ml.research.protocol import mature, rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.sources.wasde_regional import LEVELS, VERSION, add_revisions
from cottonlens_ml.sources.wasde_regional_validation import verify_candidate_values

PROFILE = 'wasde-regional-t1-pilot-v1'
NUMERIC = [*LEVELS, *[name + '_revision' for name in LEVELS]]
DELAYS = (0, 1)
MAX_AGE = 40
RECIPE = {'family': 'ridge', 'params': {'alpha': 1.}, 'horizon': 1, 'task': 'price',
          'device': 'cpu', 'seed': 42, 'target': 'scaled_log', 'years': None, 'cadence': 21, 'window': 1}
CLOCK = {'decision': 'Cotton source date + 1 day 00:15 UTC',
         'assumed_value_available_at': 'WASDE report date + 1 day 00:00 UTC',
         'basis': 'explicit_unverified_assumption', 'publication_verified': False,
         'vintage_clock_verified': False, 'delay_stress': 'one additional recorded Cotton decision',
         'max_age_sessions': MAX_AGE, 'expiry_anchor': 'first decision allowed by assumed clock; identical across delays'}
RULES = {'primary': 'T+1 selected numeric_D0 versus mask_D0 paired price-MAE',
         'delay_stress': 'numeric_D1 versus mask_D1; never choose delay from OOS results',
         'baseline': 'zero log return / Naive on exactly the same origins and realized prices',
         'raw_predictions': 'mandatory separate diagnostic; never replace primary after observing results',
         'bootstrap': {'blocks': [20, 60], 'repetitions': 10000, 'seed': 42, 'within_year': True},
         'diagnostics': ['annual gains', 'active rate and active/all-origin direction',
                         'remove largest ceil(1%*n) Naive errors with date tie break',
                         'source age 0..4 subset, predefined diagnostic only'],
         'practical_reference': {'price_mae_gain_pct': 5, 'direction_pct': 53, 'full_year_wins': '6/8'},
         'gate_evaluation_allowed': False, 'independent_holdout': False, 'automatic_release': False,
         'interpretation': 'Five reused years and unverified clocks cannot establish release eligibility or universal lack of WASDE signal.'}


def completed_files(root):
    root = Path(root)
    complete = read_record(root / 'complete.json')
    if complete.get('completed') is not True or complete.get('fits') != 0 or complete.get('model_eligible') is not False:
        raise ValueError('Completed no-fit, unadmitted evidence required')
    files = complete.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Checksummed completed payload required')
    for name, expected in files.items():
        if not safe_member(name) or digest(root / name) != expected:
            raise ValueError('Completed evidence payload changed')
    return complete


def candidate_rows(candidate, numeric_audit, csv_audit):
    candidate, numeric_audit, csv_audit = map(Path, (candidate, numeric_audit, csv_audit))
    completed_files(candidate)
    manifest = read_record(candidate / 'candidate-manifest.json')
    if (manifest.get('version') != VERSION or manifest.get('features') != NUMERIC
            or manifest.get('model_eligible') is not False or manifest.get('release_allowed') is not False
            or digest(candidate / 'regional.csv') != manifest['files']['regional.csv']):
        raise ValueError('Pinned unadmitted regional candidate required')
    frame = pd.read_csv(candidate / 'regional.csv', parse_dates=['report_date'], float_precision='round_trip')
    validate_reports(frame)
    for root in (numeric_audit, csv_audit):
        completed_files(root)
    pdf = read_record(numeric_audit / 'report.json')
    official = read_record(csv_audit / 'report.json')
    if (pdf.get('model_eligible') is not False or pdf.get('historical_model_eligible_rows') != 0
            or pdf.get('pdf_xml_mismatch_cells') != 0 or pdf.get('numeric_verified_rows') != len(frame)
            or official.get('numeric_verified') is not True or official.get('mismatches') != 0
            or official.get('reports') != len(frame) or official.get('model_eligible') is not False
            or official['input_sha256']['candidate_table'] != digest(candidate / 'regional.csv')
            or official['input_sha256']['candidate_manifest'] != digest(candidate / 'candidate-manifest.json')):
        raise ValueError('Numeric evidence does not bind this unadmitted candidate')
    verify_candidate_values(frame, pdf['records'])
    return frame


def registered_audits(repo, numeric_audit, csv_audit):
    """A self-rechecksummed replacement must not replace registered numeric evidence."""
    evidence = Path(repo) / 'research/evidence'
    numeric = json.loads((evidence / 'wasde-regional-verification-20261008.json').read_text('utf-8'))
    official = json.loads((evidence / 'wasde-as-reported-20261008.json').read_text('utf-8'))
    if (digest(Path(numeric_audit) / 'report.json') != numeric['completion']['files']['report.json']
            or digest(Path(csv_audit) / 'report.json') != official['payload_sha256']['audit/report.json']):
        raise ValueError('Audit differs from registered numeric evidence')


def validate_reports(reports):
    if (not {'report_date', 'crop_year', *NUMERIC}.issubset(reports)
            or reports.empty or reports.report_date.isna().any() or reports.report_date.duplicated().any()
            or not reports.report_date.is_monotonic_increasing
            or reports.report_date.dt.tz is not None
            or not reports.report_date.eq(reports.report_date.dt.normalize()).all()
            or reports.report_date.max() >= pd.Timestamp('2024-01-01')):
        raise ValueError('Unique sorted pre-2024 date-only reports required')
    if (not np.isfinite(reports[LEVELS].to_numpy()).all()
            or (reports[LEVELS].to_numpy() < 0).any()
            or not np.isfinite(reports.crop_year).all()
            or not reports.crop_year.eq(reports.crop_year.astype(int)).all()):
        raise ValueError('Finite nonnegative verified levels and integral crop years required')
    expected = add_revisions(reports[['report_date', 'crop_year', *LEVELS]])
    revisions = [name + '_revision' for name in LEVELS]
    if not np.allclose(reports[revisions], expected[revisions], rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError('Unknown/new-crop revisions must remain missing; causal deltas required')


def groups():
    result = {}
    for delay in DELAYS:
        timing = [f'regional_D{delay}_age_sessions', f'regional_D{delay}_unavailable']
        for arm in ('mask', 'numeric'):
            result[f'{arm}_D{delay}'] = [*FEATURE_NAMES, *timing,
                                        *[f'{arm}_D{delay}_{name}' for name in NUMERIC]]
    return result


def align_assumed(history, reports):
    """Causal only under CLOCK's explicit assumption; no published_at is assigned."""
    validate_reports(reports)
    if (history.empty or history.date.isna().any() or history.date.duplicated().any()
            or not history.date.is_monotonic_increasing or history.date.dt.tz is not None
            or not history.date.eq(history.date.dt.normalize()).all()):
        raise ValueError('Unique sorted date-only Cotton observations required')
    added = {}
    decision = pd.to_datetime(history.date, utc=True) + pd.Timedelta(days=1, minutes=15)
    assumed = pd.to_datetime(reports.report_date, utc=True) + pd.Timedelta(days=1)
    first = np.searchsorted(decision.to_numpy(), assumed.to_numpy(), side='left')
    values = reports[NUMERIC].to_numpy(dtype=float)
    added['regional_decision_time'] = decision
    for delay in DELAYS:
        selected = np.searchsorted(first + delay, np.arange(len(history)), side='right') - 1
        found = selected >= 0
        safe = np.maximum(selected, 0)
        age = np.where(found, np.arange(len(history)) - first[safe], np.nan)
        usable = found & (age <= MAX_AGE)
        matrix = np.full((len(history), len(NUMERIC)), np.nan)
        matrix[usable] = values[selected[usable]]
        added[f'regional_D{delay}_age_sessions'] = age
        added[f'regional_D{delay}_unavailable'] = (~usable).astype(float)
        for index, name in enumerate(NUMERIC):
            column = matrix[:, index]
            added[f'numeric_D{delay}_{name}'] = column
            # Preserve identical automatic Preprocessor masks. Zero is a
            # counterfactual constant column, NEVER a changed source value.
            added[f'mask_D{delay}_{name}'] = np.where(np.isfinite(column), 0., np.nan)
        for field, source in [('report_date', reports.report_date), ('assumed_available_at', assumed)]:
            series = source.reset_index(drop=True).iloc[safe].reset_index(drop=True)
            series.loc[~found] = pd.NaT
            added[f'regional_D{delay}_{field}'] = series.to_numpy()
        if (series.loc[found].to_numpy() > decision.loc[found].to_numpy()).any():
            raise ValueError('Source after decision detected')
    if set(added) & set(history.columns):
        raise ValueError('Regional features already exist; never overwrite an input')
    result = pd.concat([history.copy(), pd.DataFrame(added, index=history.index)], axis=1)
    result.attrs = {}
    return result


def make_design(history, reports):
    expanded = align_assumed(history, reports)
    # Match prior source cohort, not the eight-year core's cold-source years.
    start = max(history.date.min(), reports.report_date.min() + pd.Timedelta(days=1))
    split = short_split(history, start, profile=PROFILE, first_year=2016)
    if ([fold['year'] for fold in split['folds']] != list(range(2019, 2024))
            or sum(len(fold['origins']) for fold in split['folds']) != 1254):
        raise ValueError('Approved five-year source cohort differs')
    inner = len(groups()) * sum(len(chunks(history, block['origins'])) for fold in split['folds'] for block in fold['inner'])
    outer = len(groups()) * sum(len(chunks(history, fold['origins'])) for fold in split['folds'])
    coverage = []
    for fold in split['folds']:
        for role, dates in [('outer', fold['origins']), *[(f'inner_{i+1}', block['origins']) for i, block in enumerate(fold['inner'])]]:
            rows = rows_at(expanded, dates)
            train = mature(expanded, rows.date.min()).loc[lambda f: f.date >= start]
            if len(train) < 500 or train.target_date_5.ge(rows.date.min()).any():
                raise ValueError('Five-observation label maturity or source training history failed')
            for delay in DELAYS:
                field = f'regional_D{delay}_report_date'
                coverage.append({'year': fold['year'], 'role': role, 'delay': delay,
                                 'origins': len(rows), 'mature_train_rows': len(train),
                                 'distinct_train_reports': int(train[field].nunique()),
                                 'distinct_evaluation_reports': int(rows[field].nunique()),
                                 'unavailable_origins': int(rows[f'regional_D{delay}_unavailable'].sum()),
                                 'first_five_age_origins': int(rows[f'regional_D{delay}_age_sessions'].between(0, 4).sum())})
    design = {'profile': PROFILE, 'experiment': 'research-wasde-regional-t1-pilot-v1',
              'hypothesis': '26 regional WASDE levels/revisions add T+1 information beyond fixed 24-feature core and identical age/missingness.',
              'groups': groups(), 'recipe': RECIPE, 'split': split, 'clock': CLOCK, 'rules': RULES,
              'shrinkage_weights': list(SHRINKAGE),
              'selection': 'one fixed recipe; each arm selects only shrinkage by mean past 3x63 relative price-MAE; argmin ties prefer smallest weight',
              'preprocessing': 'existing training-only median/mean/std plus automatic missing indicators; mask controls share exact numeric missingness',
              'fit_budget': {'inner': inner, 'outer': outer, 'total': inner + outer,
                             'annual_outputs': len(groups()) * len(split['folds']), 'prediction_rows': 1254 * len(groups()),
                             'processes': 1, 'threads': 2, 'session_minutes': 30},
              'target': 'T+1 log(close[next recorded Cotton observation]/close[origin]); existing scaled_log train-only transform',
              'maturity': 'existing common target_date_5 < refit cutoff; no feature-complete origin filtering',
              'feature_data_id': frame_identity(expanded, list(expanded.columns)),
              'deliberate_repeat_reason': 'Related World balance and text tests are T+5/three fields, not regional 26-field T+1; matching source cohort, new recipe and namespace.',
              'release_allowed': False, 'historical_source_admitted': False, 'fits_completed': 0,
              'execution_status': 'preregistered_not_runnable; no engine profile or source admission created'}
    return expanded, design, coverage


def prepare(repo, output, reference, candidate, numeric_audit, csv_audit):
    repo, output, reference, candidate, numeric_audit, csv_audit = map(Path, (repo, output, reference, candidate, numeric_audit, csv_audit))
    validate(repo / 'research')
    registry, trials = load_registry(repo / 'research')
    related = check(registry, trials, query='wasde')
    inputs = {}
    for prefix, root, names in [('reference', reference, ['ready.json', 'history.parquet']),
                                ('candidate', candidate, ['complete.json', 'candidate-manifest.json', 'regional.csv']),
                                ('numeric_audit', numeric_audit, ['complete.json', 'report.json', 'coverage.csv']),
                                ('csv_audit', csv_audit, ['complete.json', 'report.json', 'coverage.csv', 'comparison.csv']),
                                ('registry', repo / 'research', ['registry.json', 'trials.json']),
                                ('evidence', repo / 'research/evidence',
                                 ['wasde-regional-verification-20261008.json', 'wasde-as-reported-20261008.json'])]:
        for name in names:
            inputs[f'{prefix}/{name}'] = (root / name, digest(root / name))
    source = research_source_identity(repo)
    registered_audits(repo, numeric_audit, csv_audit)
    parent, history = verify_history(reference)
    if parent['identity'].get('profile') != 'full-year-v1':
        raise ValueError('Frozen full-year core reference required')
    reports = candidate_rows(candidate, numeric_audit, csv_audit)
    expanded, design, coverage = make_design(history, reports)
    identity = {'source_id': source['source_id'], 'input_sha256': {name: expected for name, (_, expected) in inputs.items()},
                'python': sys.version.split()[0], 'versions': {name: importlib.metadata.version(name) for name in
                    ['numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl']},
                'lock_sha256': digest(repo / 'ml/uv.lock'), 'design': design, 'fits_completed': 0}
    proposed = []
    for group, features in design['groups'].items():
        recipe = {**RECIPE, 'features': features}
        result = check(registry, trials, proposal={'scope_id': content_id(identity), 'recipe': recipe})
        if result['exact_fit_recipes']:
            raise ValueError('Exact frozen recipe already fitted; deliberate review required')
        proposed.append({'group': group, 'recipe': recipe, 'lookup_status': result['status'],
                         'related_fit_recipes': len(result['fit_recipes']), 'exact_matches': 0})
    registration = {'profile': PROFILE, 'created_at': datetime.now(UTC).isoformat(),
                    'identity': identity, 'registration_id': content_id(identity), 'fits': 0,
                    'completed_preregistration': True, 'completed_experiment': False,
                    'availability_verified': False, 'model_eligible': False}
    with writer(output):
        if any(path.name != '.writer-lock' for path in output.iterdir()):
            raise ValueError('Preserve existing/partial registration; choose new namespace')
        for name, (path, expected) in inputs.items():
            copy_immutable(path, output / 'inputs' / name)
            if digest(output / 'inputs' / name) != expected:
                raise ValueError('Input changed during copy')
        expanded.to_parquet(output / 'history.parquet', index=False)
        pd.DataFrame(coverage).to_csv(output / 'source-coverage.csv', index=False)
        freeze_record(output / 'source-manifest.json', source)
        for name, expected in source['files'].items():
            copy_immutable(repo / name, output / 'source-snapshot' / name)
            if digest(output / 'source-snapshot' / name) != expected:
                raise ValueError('Source changed during copy')
        freeze_record(output / 'history-check.json', {'related_experiments': related['experiments'],
                      'related_fit_recipes': related['fit_recipes'], 'proposals': proposed,
                      'limits': related['limits'], 'fits': 0})
        freeze_record(output / 'preregistered.json', registration)
        if source != research_source_identity(repo) or any(digest(path) != expected for path, expected in inputs.values()):
            raise ValueError('Code/data/evidence/registry changed during registration')
        files = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob('*')
                 if path.is_file() and '.writer-lock' not in path.relative_to(output).parts}
        freeze_record(output / 'complete.json', {'profile': PROFILE, 'completed': True, 'fits': 0,
                      'completed_experiment': False, 'availability_verified': False, 'model_eligible': False,
                      'registration_id': registration['registration_id'], 'files': files})
    return registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['repo', 'output', 'reference', 'candidate', 'numeric-audit', 'csv-audit']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.repo, args.output, args.reference, args.candidate, args.numeric_audit, args.csv_audit)
    print({'registration_id': result['registration_id'], 'fits': 0, 'fit_budget': result['identity']['design']['fit_budget']})


if __name__ == '__main__':
    main()
