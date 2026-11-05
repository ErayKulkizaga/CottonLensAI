import json
import subprocess
import sys
from datetime import date, timedelta
from decimal import ROUND_UP, Decimal, localcontext
from pathlib import Path

import pytest
from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.sources import fas_country_candidate as candidate
from test_fas_country_audit import fixture


def observation(day, code=5700, sales=-2, exports=10, outstanding=20):
    return (date.fromisoformat(day), code,
            {f: Decimal(v) for f, v in zip(candidate.FIELDS, (sales, exports, outstanding))})


def test_shares_use_all_reported_countries_and_preserve_negative_sales():
    rows = candidate._panel([observation('2020-06-04'),
                             observation('2020-06-04', 9999, 3, 90, 180)], (5700,))
    row = rows[0]
    assert row['values'] == dict(zip(candidate.FIELDS, ('-2', '10', '20')))
    assert row['reported_national_totals'] == dict(zip(candidate.FIELDS, ('1', '100', '200')))
    assert row['shares_of_reported_total'] == {'weeklyExports': '0.1', 'outstandingSales': '0.1'}
    assert 'currentMYNetSales' not in row['shares_of_reported_total']
    assert 'accumulatedExports' not in json.dumps(rows)


def test_missing_country_is_distinct_from_reported_zero_and_zero_denominator():
    rows = candidate._panel([observation('2020-06-04', exports=0, outstanding=0)], (5350, 5700))
    missing, present = rows
    assert not missing['reported_row_present']
    assert set(missing['values'].values()) == {None}
    assert present['reported_row_present'] and present['values']['weeklyExports'] == '0'
    assert set(present['shares_of_reported_total'].values()) == {None}
    assert set(missing['shares_of_reported_total'].values()) == {None}


def test_four_week_flow_sum_does_not_compress_missing_dates_or_country_rows():
    first = date(2020, 5, 7)
    observations = [observation((first + timedelta(weeks=i)).isoformat()) for i in range(5)]
    rows = candidate._panel(observations, (5700,))
    assert rows[2]['trailing_four_calendar_weeks']['currentMYNetSales'] is None
    assert rows[3]['trailing_four_calendar_weeks'] == {'currentMYNetSales': '-8', 'weeklyExports': '40'}
    observations.pop(1)
    rows = candidate._panel(observations, (5700,))
    assert rows[-1]['trailing_four_calendar_weeks']['weeklyExports'] is None
    observations.append(observation('2020-05-14', 9999))
    rows = candidate._panel(observations, (5700,))
    assert rows[1]['values']['weeklyExports'] is None
    assert rows[-1]['trailing_four_calendar_weeks']['weeklyExports'] is None


def test_four_week_sum_restarts_at_marketing_year_boundary():
    observations = [observation(d) for d in ('2020-07-16', '2020-07-23', '2020-07-30',
                                            '2020-08-06', '2020-08-13', '2020-08-20', '2020-08-27')]
    rows = candidate._panel(observations, (5700,))
    assert rows[2]['marketing_year'] == 2020 and rows[3]['marketing_year'] == 2021
    assert all(r['trailing_four_calendar_weeks']['weeklyExports'] is None for r in rows[3:6])
    assert rows[6]['trailing_four_calendar_weeks']['weeklyExports'] == '40'


def test_future_change_and_append_leave_past_rows_identical():
    history = [observation(d) for d in ('2020-05-07', '2020-05-14', '2020-05-21', '2020-05-28')]
    past = candidate._panel(history, (5700,))
    changed = candidate._panel(history + [observation('2020-06-04', sales=999999)], (5700,))
    assert changed[:4] == past


def test_decimal_ratio_does_not_inherit_caller_precision_or_rounding():
    observations = [observation('2020-06-04', exports=10001),
                    observation('2020-06-04', 9999, exports=99999)]
    normal = candidate._panel(observations, (5700,))
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_UP
        assert candidate._panel(observations, (5700,)) == normal


def test_negative_shipment_rejected():
    day, code, values = observation('2020-06-04')
    values['weeklyExports'] = Decimal(-1)
    with pytest.raises(ValueError, match='Negative shipment'):
        candidate._panel([(day, code, values)], (5700,))


def test_signed_outstanding_is_preserved_and_disables_shares_for_all_countries():
    rows = candidate._panel([observation('2020-06-04'),
                             observation('2020-06-04', 9999, outstanding=-1)], (5700, 9999))
    assert rows[1]['values']['outstandingSales'] == '-1'
    assert rows[0]['reported_national_totals']['outstandingSales'] == '19'
    assert all(r['shares_of_reported_total']['outstandingSales'] is None for r in rows)
    assert all(r['share_denominator_defined']['outstandingSales'] is False for r in rows)
    assert rows[0]['shares_of_reported_total']['weeklyExports'] == '0.5'


def test_duplicate_observation_rejected():
    row = observation('2020-06-04')
    with pytest.raises(ValueError, match='Duplicate'):
        candidate._panel([row, row], (5700,))


def test_prepare_preserves_sources_boundary_rule_and_ineligible_identity(tmp_path):
    args = fixture(tmp_path)
    before = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = candidate.prepare(*args, countries=(1002, 1001))
    assert result == candidate.prepare(*args, countries=(1001, 1002))
    assert result['row_count'] == 30 and result['weeks'] == 15
    assert result['rows'][0]['values']['weeklyExports'] == '10'  # old-year overlap excluded
    assert result['candidate_id'] == manifest_id({k: v for k, v in result.items() if k != 'candidate_id'})
    for flag in ('model_eligible', 'release_allowed', 'historical_publication_verified',
                 'first_version_verified', 'country_names_verified_for_full_history'):
        assert result[flag] is False
    assert result['market_fits'] == 0 and result['availability_policy'].startswith('UNSET')
    assert {p: digest(p) for p in before} == before


@pytest.mark.parametrize('codes', [(), (1001, 1001), (True,), ('1001',), (-1,)])
def test_invalid_country_scope_rejected_before_reading(tmp_path, codes):
    with pytest.raises(ValueError, match='Unique nonnegative integer'):
        candidate.prepare(tmp_path, tmp_path, tmp_path, countries=codes)


def test_changed_input_during_preparation_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = candidate._panel
    def mutate(*arguments):
        result = original(*arguments)
        args[2].write_text('changed', encoding='utf-8')
        return result
    monkeypatch.setattr(candidate, '_panel', mutate)
    with pytest.raises(ValueError, match='checksum mismatch'):
        candidate.prepare(*args)


@pytest.mark.parametrize('corrupt', [False, True])
def test_cli_is_dependency_free_and_failure_has_no_partial_payload(tmp_path, corrupt):
    raw, manifest, table = fixture(tmp_path)
    if corrupt:
        table.write_text('changed', encoding='utf-8')
    cli = Path(__file__).resolve().parents[1] / 'prepare_fas_countries.py'
    completed = subprocess.run([sys.executable, '-S', str(cli), '--raw-root', str(raw),
                               '--manifest', str(manifest), '--table', str(table)],
                              capture_output=True, text=True, check=False)
    if corrupt:
        assert completed.returncode == 2 and not completed.stdout
    else:
        assert completed.returncode == 0
        result = json.loads(completed.stdout)
        assert result['row_count'] == 60 and result['model_eligible'] is False
        assert all(r['values']['weeklyExports'] is None for r in result['rows'])
