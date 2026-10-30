import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from research import ct_manual_source as study
from research.verify_ct_manual_source import verify


def calendar():
    return study.CottonCalendar(json.loads(Path('research/calendars/ice-cotton-2026.json').read_bytes()))


def chart():
    dates = ['2026-10-28', '2026-10-29', '2026-10-30', '2026-11-01']
    stamps = [int(pd.Timestamp(d, tz='America/New_York').timestamp()) for d in dates]
    values = [78., 79., 80., None]
    return {'chart': {'error': None, 'result': [{'meta': {'symbol': 'CT=F', 'dataGranularity': '1d',
        'exchangeTimezoneName': 'America/New_York', 'regularMarketTime': stamps[-2]}, 'timestamp': stamps,
        'indicators': {'quote': [{k: values.copy() for k in ('open', 'high', 'low', 'close', 'volume')}]}}]}}


def receipt():
    return {'observed_available_at': '2026-10-30T09:58:30+00:00'}


def test_every_raw_row_preserved_partial_and_empty_rows_cannot_enter_history():
    with patch('cottonlens_ml.research.ledger.Ledger.run', side_effect=AssertionError('No fit')):
        rows, completed = study.classify_chart(chart(), receipt(), calendar())
    assert rows.classification.tolist() == ['completed_quote', 'completed_quote', 'incomplete_quote', 'empty_placeholder']
    assert len(rows) == 4 and len(completed) == 2
    assert completed.cotton_close.tolist() == [78., 79.]


def test_close_bound_includes_equality_but_not_one_second_before():
    body = chart()
    for observed, expected in [('2026-10-30T18:19:59Z', 2), ('2026-10-30T18:20:00Z', 3)]:
        _, completed = study.classify_chart(body, {'observed_available_at': observed}, calendar())
        assert len(completed) == expected
    assert str(calendar().completion(pd.Timestamp('2026-11-02').date())) == '2026-11-02 19:20:00+00:00'


@pytest.mark.parametrize('change', ['gap', 'array', 'duplicate', 'wrong_symbol', 'wrong_frequency',
                                  'wrong_timezone', 'future_market', 'future_row', 'negative', 'zero',
                                  'missing_close', 'nonfinite', 'priced_sunday'])
def test_unusable_quotes_fail_before_any_feature_or_fit(change):
    body = chart(); raw = body['chart']['result'][0]; quotes = raw['indicators']['quote'][0]
    if change == 'gap':
        for v in quotes.values():
            v[1] = None
    elif change == 'array':
        quotes['volume'].pop()
    elif change == 'duplicate':
        raw['timestamp'][1] = raw['timestamp'][0]
    elif change.startswith('wrong_'):
        field = {'wrong_symbol': 'symbol', 'wrong_frequency': 'dataGranularity', 'wrong_timezone': 'exchangeTimezoneName'}[change]
        raw['meta'][field] = 'OTHER'
    elif change == 'future_market':
        raw['meta']['regularMarketTime'] += 999999
    elif change == 'future_row':
        raw['timestamp'][2] += 200000
    elif change in ('negative', 'zero', 'missing_close', 'nonfinite'):
        quotes['close'][1] = {'negative': -1., 'zero': 0., 'missing_close': None, 'nonfinite': float('inf')}[change]
    else:
        for v in quotes.values():
            v[-1] = 81.
    with pytest.raises(ValueError):
        study.classify_chart(body, receipt(), calendar())


def test_future_partial_quote_changes_cannot_change_closed_history():
    raw = chart(); altered = copy.deepcopy(raw)
    for values in altered['chart']['result'][0]['indicators']['quote'][0].values():
        values[2] *= 5
    pd.testing.assert_frame_equal(study.classify_chart(raw, receipt(), calendar())[1],
                                  study.classify_chart(altered, receipt(), calendar())[1], check_exact=True)


def test_cotton_holidays_not_federal_holidays_and_coverage_is_finite():
    cal = calendar()
    for day in ('2026-10-12', '2026-11-11', '2026-12-28'):
        assert cal.session(pd.Timestamp(day).date())
    for day in ('2026-11-26', '2026-12-25', '2027-01-01'):
        assert not cal.session(pd.Timestamp(day).date())
    with pytest.raises(ValueError, match='coverage'):
        cal.completed(pd.Timestamp('2026-12-30').date(), study.utc('2027-01-04T00:15Z'))
    with pytest.raises(ValueError, match='timezone'):
        study.utc('2026-10-30')


def test_receipt_requires_exact_original_bytes_and_ordered_times(tmp_path):
    raw = tmp_path/'source.json'; raw.write_bytes(b'{}')
    path = tmp_path/'receipt.json'
    body = {**receipt(), 'request_started_at': '2026-10-30T09:58:29Z', 'http_status': 200,
            'response_bytes': 2, 'response_sha256': hashlib.sha256(b'{}').hexdigest()}
    path.write_text(json.dumps(body))
    assert study.verified_response(raw, path) == body
    raw.write_bytes(b'[]')
    with pytest.raises(ValueError, match='hash mismatch'):
        study.verified_response(raw, path)
    raw.write_bytes(b'{}'); body['request_started_at'] = '2026-10-31T00:00Z'; path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='availability order'):
        study.verified_response(raw, path)


def seed_and_decision():
    old = pd.DataFrame({'date': pd.to_datetime(['2026-10-28', '2026-10-29']), 'cotton_close': [78., 79.]})
    new = pd.DataFrame({'date': pd.to_datetime(['2026-10-28', '2026-10-29', '2026-10-30']), 'cotton_close': [78., 80., 81.]})
    decision = {'purpose': 'PROPOSED_PRE_COHORT_SEED_VERSION', 'original_seed_sha256': 'old',
                'snapshot_sha256': 'new', 'changed_date': '2026-10-29', 'old_close': 79., 'new_close': 80.,
                'cohort_locked': False, 'forward_forecasts': 0}
    return old, new, decision


def test_initial_fork_preserves_original_and_never_weakens_default_revision_guard():
    old, new, decision = seed_and_decision(); original = old.copy(deep=True)
    with pytest.raises(ValueError, match='revision'):
        study.append_snapshot(old, new)
    fork = study.initial_seed(old, new, decision, 'old', 'new')
    pd.testing.assert_frame_equal(old, original, check_exact=True)
    pd.testing.assert_frame_equal(fork, new, check_exact=True)


@pytest.mark.parametrize('change', ['old_hash', 'new_hash', 'locked', 'forecast', 'second_revision', 'calendar', 'wrong_close'])
def test_explicit_seed_version_cannot_accept_unrelated_or_postlock_revision(change):
    old, new, decision = seed_and_decision()
    if change == 'old_hash':
        decision['original_seed_sha256'] = 'different'
    elif change == 'new_hash':
        decision['snapshot_sha256'] = 'different'
    elif change == 'locked':
        decision['cohort_locked'] = True
    elif change == 'forecast':
        decision['forward_forecasts'] = 1
    elif change == 'second_revision':
        new.loc[0, 'cotton_close'] += 1
    elif change == 'calendar':
        new = new.drop(index=1)
    else:
        decision['new_close'] += 1
    with pytest.raises(ValueError):
        study.initial_seed(old, new, decision, 'old', 'new')


def test_offline_audit_keeps_original_bytes_rejects_revision_and_never_fits(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; (repo/'research/calendars').mkdir(parents=True)
    body = json.loads(Path('research/calendars/ice-cotton-2026.json').read_bytes()); body['evidence'] = []
    (repo/'research/calendars/ice-cotton-2026.json').write_text(json.dumps(body))
    source = tmp_path/'source'; source.mkdir()
    raw = source/'yahoo-ct-chart.json'; raw.write_text(json.dumps(chart()))
    source_receipt = {**receipt(), 'request_started_at': '2026-10-30T09:58:29Z', 'http_status': 200,
                      'response_bytes': raw.stat().st_size, 'response_sha256': study.digest(raw)}
    (source/'yahoo-ct-chart.json.receipt.json').write_text(json.dumps(source_receipt))
    old = pd.DataFrame({'date': pd.to_datetime(['2026-10-28', '2026-10-29']), 'cotton_close': [78., 80.]})
    seed_path = tmp_path/'seed.parquet'; old.to_parquet(seed_path, index=False)
    original = study.digest(seed_path)
    monkeypatch.setattr(study, 'OLD_SEED_SHA', original)
    monkeypatch.setattr(study, 'HELPERS', ())
    monkeypatch.setattr(study, 'research_source_identity', lambda _: {'files': {}})
    with (patch('cottonlens_ml.research.ledger.Ledger.run', side_effect=AssertionError('No fitting')),
          patch('socket.socket.connect', side_effect=AssertionError('No network'))):
        report = study.audit(repo, seed_path, source, None, tmp_path/'strict')
    assert report['status'] == 'REJECTED_REVISION' and report['proposed_seed_rows'] is None
    assert not (tmp_path/'strict/proposed-seed.parquet').exists()
    assert report['new_fits'] == report['new_forecasts'] == 0 and not report['forward_authorized']
    assert study.digest(seed_path) == original
    # Empty placeholder closes must retain a numeric CSV column in the independent reader.
    assert verify(tmp_path/'strict')['raw_rows_checked'] == 4
    with pytest.raises(ValueError, match='Preserve existing'):
        study.audit(repo, seed_path, source, None, tmp_path/'strict')
    (source/'yahoo-ct-chart.json.receipt.json').unlink()
    with pytest.raises(FileNotFoundError):
        study.audit(repo, seed_path, source, None, tmp_path/'missing-receipt')
    assert not (tmp_path/'missing-receipt').exists()
