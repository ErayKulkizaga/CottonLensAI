"""Synthetic publication-window execution; no providers or real forecasts."""
from datetime import datetime

import pandas as pd
import pytest
from cottonlens_ml.research import live, prospective
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.prospective import baseline_lock, verify_baseline_chain
from test_forward_baselines import snapshot


@pytest.mark.parametrize('when', ['00:15', '00:20', '00:29'])
def test_publication_window_does_no_network_scoring_or_mirroring(tmp_path, monkeypatch, when):
    baseline_lock(tmp_path / 'forward', now='2026-10-02T23:00Z')
    class Clock:
        @staticmethod
        def now(tz):
            return datetime.fromisoformat('2026-10-03T' + when + ':00+00:00')
    monkeypatch.setattr(live, 'datetime', Clock)
    monkeypatch.setattr(prospective, 'datetime', Clock)
    monkeypatch.setattr(live, 'market_snapshots', lambda root: [snapshot()])
    def forbidden(*args, **kwargs):
        pytest.fail('Publication window must use existing inputs only')
    for name in ('capture_market', 'archive_observation', 'USDAClient', 'Mirror', 'score_baselines'):
        monkeypatch.setattr(live, name, forbidden)
    status = live.collect(tmp_path, mirror_root=tmp_path / 'mirror')
    assert status['phase'] == 'publication_only'
    assert status['forward']['published_origins'] == 1
    assert not status['training'] and len(status['execution_source_id']) == 64
    saved = read_record(next((tmp_path / 'runs').glob('*.json')))
    assert saved['execution_source_id'] == status['execution_source_id']
    assert saved['forward_record_ids']


def test_slow_input_write_crossing_deadline_becomes_missing(tmp_path, monkeypatch):
    baseline_lock(tmp_path, now='2026-10-02T23:00Z')
    current = [datetime.fromisoformat('2026-10-03T00:29:59+00:00')]
    class Clock:
        @staticmethod
        def now(tz):
            return current[0]
    monkeypatch.setattr(prospective, 'datetime', Clock)
    original = pd.DataFrame.to_parquet
    def slow(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        current[0] = datetime.fromisoformat('2026-10-03T00:30:00+00:00')
        return result
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', slow)
    result = prospective.record_baselines(tmp_path, [snapshot()])
    assert result['published_origins'] == 0
    _, records = verify_baseline_chain(tmp_path)
    assert records[0]['state'] == 'missing' and not records[0]['predictions']
    assert pd.Timestamp(records[0]['recorded_at']).minute == 30


@pytest.mark.parametrize('when', ['00:14', '00:30'])
def test_collection_outside_publication_window_remains_enabled(tmp_path, monkeypatch, when):
    from cottonlens_ml.research.ledger import freeze_record

    class Clock:
        @staticmethod
        def now(tz):
            return datetime.fromisoformat('2026-10-03T' + when + ':00+00:00')
    monkeypatch.setattr(live, 'datetime', Clock)
    freeze_record(tmp_path / 'daily/2026-10-03.json', {'tasks': []})
    monkeypatch.setattr(live, 'market_snapshots', lambda root: [])
    calls = []
    monkeypatch.setattr(live, 'capture_market', lambda root: calls.append('CT=F'))
    for name in live.KEYS.values():
        monkeypatch.delenv(name, raising=False)
    status = live.collect(tmp_path)
    assert calls == ['CT=F'] and status['phase'] == 'collection'
