import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.live import contract_symbols


def test_later_credentials_are_not_blocked_by_public_daily_receipt(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from cottonlens_ml.research import live
    from cottonlens_ml.research.ledger import freeze_record
    calls = []
    now = datetime.now(UTC)
    freeze_record(tmp_path/'daily'/(now.strftime('%Y-%m-%d')+'.json'), {'tasks': []})
    monkeypatch.setattr(live, 'market_snapshots', lambda root: [])
    monkeypatch.setattr(live, 'capture_market', lambda root: tmp_path/'synthetic-market.json')
    for name in live.KEYS.values():
        monkeypatch.delenv(name, raising=False)
    live.collect(tmp_path)
    class Client:
        def __init__(self, provider):
            self.provider = provider
        def fetch(self, *args):
            calls.append(self.provider)
            return tmp_path/'synthetic-receipt.json', {}
        report = exports = cotton_by_year = fetch
    monkeypatch.setattr(live, 'USDAClient', Client)
    for name in live.KEYS.values():
        monkeypatch.setenv(name, 'synthetic-test-only')
    live.collect(tmp_path)
    live.collect(tmp_path)
    assert calls == ['ams','fas','nass']
from cottonlens_ml.research.prospective import (
    baseline_lock,
    record_baselines,
    score_baselines,
    verify_baseline_chain,
)


def snapshot(last='2026-10-02', observed='2026-10-03T00:05:00Z'):
    dates = pd.bdate_range('2026-07-01', last)
    return (pd.DataFrame({'date': dates, 'cotton_close': 80*np.exp(np.sin(np.arange(len(dates)))*.01)}),
            {'observed_available_at': observed, 'synthetic_fixture': True})


def test_on_time_forward_record_and_duplicates(tmp_path):
    baseline_lock(tmp_path, now='2026-10-01T01:00Z')
    inputs = [snapshot()]
    first = record_baselines(tmp_path, inputs, now='2026-10-03T00:20Z')
    assert first['registered_origins'] == 2
    _, records = verify_baseline_chain(tmp_path)
    assert [r['state'] for r in records] == ['missing', 'published']
    assert records[-1]['predictions']['5']['price_model'] == 'naive'
    assert record_baselines(tmp_path, inputs, now='2026-10-03T00:25Z')['added_origins'] == 0
    assert score_baselines(tmp_path, inputs[0][0])['status'] == 'pending'


def test_late_snapshot_cannot_backfill_and_no_premature_missing(tmp_path):
    baseline_lock(tmp_path, now='2026-10-02T23:00Z')
    late = [snapshot(observed='2026-10-03T00:19Z')]
    assert record_baselines(tmp_path, late, now='2026-10-03T00:20Z')['registered_origins'] == 0
    assert record_baselines(tmp_path, late, now='2026-10-03T01:00Z')['registered_origins'] == 1
    _, records = verify_baseline_chain(tmp_path)
    assert records[0]['state'] == 'missing' and records[0]['predictions'] == {}
    assert record_baselines(tmp_path, [snapshot()], now='2026-10-03T02:00Z')['added_origins'] == 0


def test_chain_detects_input_corruption(tmp_path):
    baseline_lock(tmp_path, now='2026-10-02T23:00Z')
    record_baselines(tmp_path, [snapshot()], now='2026-10-03T00:20Z')
    _, records = verify_baseline_chain(tmp_path)
    (tmp_path / records[0]['input_file']).write_text('changed')
    with pytest.raises(ValueError, match='input changed'):
        verify_baseline_chain(tmp_path)


def test_contract_calendar_does_not_claim_rolls():
    assert contract_symbols(pd.Timestamp('2026-10-01')) == ['CTV26.NYB', 'CTZ26.NYB', 'CTH27.NYB']
