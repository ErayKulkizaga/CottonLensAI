"""Form 304 pricing commitments, distinct from the completed COT position test.

Uses the existing information runner, ledger, full-year split and fixed recipes.
Source assumptions remain Tier A; no lock/export or historical success gate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.sources.oncall_archive import FIELDS

PROFILE = 'oncall-exploration-v1'
ASSUMPTION_PROFILE = 'oncall-exploration-v2'
LAGS = information.LAGS


def read_oncall(table):
    table = Path(table)
    source = read_record(table.with_suffix('.manifest.json'))
    if (digest(table) != source['table_sha256'] or source['schema'] not in ('oncall-tier-a-table-v1', 'oncall-tier-a-table-v2')
            or source['fields'] != list(FIELDS) or source['model_eligible'] or source['release_allowed']
            or source['publication_timestamp_verified'] or source['first_version_verified']):
        raise ValueError('Pinned Tier-A Form 304 table required')
    if source['schema'] == 'oncall-tier-a-table-v2' and source.get('footer_policy') != 'latest_date_assumption':
        raise ValueError('Explicit unverified latest-date assumption required')
    rows = pd.read_csv(table, parse_dates=['report_date', 'assumed_day'], keep_default_na=False)
    for field in FIELDS:
        rows[field] = pd.to_numeric(rows[field].replace('', np.nan), errors='raise')
    if (len(rows) != source['rows'] or rows.report_date.isna().any() or rows.assumed_day.isna().any()
            or rows.report_date.duplicated().any() or not rows.report_date.is_monotonic_increasing
            or rows.report_date.min() < pd.Timestamp('2010-01-01')
            or rows.report_date.max() >= pd.Timestamp('2024-01-01')
            or (rows.assumed_day < rows.report_date+pd.Timedelta(days=7)).any()
            or np.isinf(rows[list(FIELDS)].to_numpy(dtype=float)).any()
            or (rows.sales_share < 0).any()):
        raise ValueError('Chronological pre-audit rows and conservative assumptions required')
    valid = rows.report_date.diff().dt.days.between(5, 9) & rows.input_mask_reason.eq('') & rows.input_mask_reason.shift().eq('')
    expected = rows.net_share.diff().where(valid)
    if (not np.allclose(rows.net_share_change, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
            or rows.loc[rows.input_mask_reason.ne(''), list(FIELDS)].notna().any().any()):
        raise ValueError('Masked inputs/causal changes differ')
    if source['schema'] == 'oncall-tier-a-table-v2':
        versions = source['parsed_versions']
        if len(versions) != len(rows) or len({v['as_of'] for v in versions}) != len(rows):
            raise ValueError('Unique source versions required for delay verification')
        indexed = {v['as_of']: v for v in versions}
        for row in rows.itertuples():
            version = indexed[row.report_date.strftime('%Y-%m-%d')]
            assumed = row.report_date+pd.Timedelta(days=7)
            bound = version['printed_release_not_before_utc']
            if bound is not None:
                assumed = max(assumed, pd.Timestamp(bound).tz_convert('UTC').tz_localize(None).normalize()+pd.Timedelta(days=1))
            for day in version['footer_additional_dates']:
                assumed = max(assumed, pd.Timestamp(day)+pd.Timedelta(days=2))
            if (row.assumed_day != assumed or row.source_sha256 != version['source_sha256']
                    or row.source_url != version['source_url']):
                raise ValueError('Table delay/source differs from pinned footer evidence')
    return rows, source


def add_oncall(history, rows):
    if (history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or rows.report_date.isna().any() or rows.report_date.duplicated().any()
            or not rows.report_date.is_monotonic_increasing or rows.assumed_day.isna().any()):
        raise ValueError('Unique chronological observations required')
    result = history.copy()
    calendar = history.date.to_numpy(dtype='datetime64[ns]')
    # Late older reports cannot replace a more recent report already observed.
    starts = np.searchsorted(calendar, rows.assumed_day.to_numpy(dtype='datetime64[ns]'), side='right')
    dates = rows.report_date.to_numpy(dtype='datetime64[ns]')
    reported = np.searchsorted(calendar, dates, side='right')-1
    raw = rows[list(FIELDS)].to_numpy(dtype=float)
    for lag in LAGS:
        available = starts+lag-1
        events = np.lexsort((dates, available))
        values = np.full((len(history), len(FIELDS)), np.nan)
        age = np.full(len(history), np.nan)
        cursor, selected = 0, None
        for index in range(len(history)):
            while cursor < len(events) and available[events[cursor]] <= index:
                row = events[cursor]
                if selected is None or dates[row] >= dates[selected]:
                    selected = row
                cursor += 1
            if selected is not None:
                age[index] = index-reported[selected]
                if index-starts[selected] <= 10:
                    values[index] = raw[selected]
        for field, column in zip(FIELDS, values.T, strict=True):
            result[f'oncall_{field}_L{lag}'] = column
        result[f'oncall_age_L{lag}'] = age
        result[f'oncall_missing_L{lag}'] = (~np.isfinite(values)).any(axis=1).astype(float)
    return result


def group_names():
    groups = {'base': list(FEATURE_NAMES)}
    for lag in LAGS:
        controls = [f'oncall_age_L{lag}', f'oncall_missing_L{lag}']
        groups[f'missing_L{lag}'] = [*FEATURE_NAMES, *controls]
        groups[f'oncall_L{lag}'] = [*FEATURE_NAMES, *controls, *[f'oncall_{f}_L{lag}' for f in FIELDS]]
    return groups


def prepare(repo, folder, reference, table):
    rows, source = read_oncall(table)
    profile = ASSUMPTION_PROFILE if source['schema'] == 'oncall-tier-a-table-v2' else PROFILE
    information.prepare_information(repo, folder, Path(reference), profile=profile, source=source,
        add_features=lambda history: add_oncall(history, rows), first_source_day=rows.assumed_day.min(),
        groups=group_names(), controls={f'oncall_L{lag}': f'missing_L{lag}' for lag in LAGS}, prefix='oncall',
        hypothesis='Form 304 unfixed pricing commitments/OI adds T+5 information beyond age/missingness and existing Cotton features',
        policy={'assumed_day': source['calendar_policy'], 'cotton_lags': list(LAGS),
                'max_age_after_first_post_assumption_observation_including_lag': 10,
                'ambiguous_inputs_masked': True, 'late_old_report_cannot_replace_newer': True,
                'footer_conflicts': source.get('footer_policy', 'mask_conflicts'),
                'publication_clock_verified': False, 'first_version_verified': False})


def dispatch(args):
    if args.stage == 'pilot':
        # Acquisition/prepare success does not imply sufficient historical coverage.
        from cottonlens_ml.research.engine import root_path
        folder = root_path(args.drive_root, args.experiment)
        if getattr(args, 'mirror_root', None):
            from cottonlens_ml.research.mirror import Mirror
            Mirror(folder, root_path(args.mirror_root, args.experiment)).hydrate(metadata_only=True)
        receipt = folder/'source-readiness.json'
        if not receipt.exists():
            raise ValueError('On-Call pilot requires a reviewed source-readiness record; no fits started')
        readiness = read_record(receipt)
        ready = read_record(folder/'ready.json')
        if (readiness.get('status') != 'passed'
                or readiness.get('ready_sha256') != digest(folder/'ready.json')
                or readiness.get('table_sha256') != ready['identity']['source_evidence']['table_sha256']):
            raise ValueError('On-Call source coverage is blocked or readiness identity changed; no fits started')
    def prepare_source(folder):
        if not all((args.reference_root, args.oncall_table)):
            raise ValueError('Pinned parent and On-Call table required')
        _, source = read_oncall(args.oncall_table)
        expected = ASSUMPTION_PROFILE if source['schema'] == 'oncall-tier-a-table-v2' else PROFILE
        if args.profile != expected:
            raise ValueError('Table policy and experiment profile must match; old protocol preserved')
        prepare(args.repo, folder, args.reference_root, args.oncall_table)
    information.dispatch_information(args, prepare_source)
    if args.stage == 'status':
        from cottonlens_ml.research.engine import root_path
        receipt = root_path(args.drive_root, args.experiment)/'source-readiness.json'
        state = read_record(receipt) if receipt.exists() else {}
        print(json.dumps({'source_readiness': state.get('status', 'not_reviewed'),
                          'scope': 'Tier A exploratory pilot only; publication/vintage unverified',
                          'release_allowed': False}))
