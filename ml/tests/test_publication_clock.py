import sys

import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import publications
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.protocol import attach_releases


def releases(stamps):
    return pd.DataFrame({'published_at': stamps, 'vintage_id': [f'v{i}' for i in range(len(stamps))],
        'source_url': 'https://example.org/synthetic', 'source_sha256': 'a' * 64,
        'timestamp_verified': True, 'wasde_value': list(range(1, len(stamps) + 1))})


def test_new_clock_includes_equal_cutoff_excludes_later_and_keeps_legacy():
    history = pd.DataFrame({'date': pd.to_datetime(['2020-06-11', '2020-06-12'])})
    rows = releases(['2020-06-12T00:00:00Z', '2020-06-12T00:15:00Z', '2020-06-12T00:15:01Z'])
    old = attach_releases(history, rows, ['wasde_value'])
    new = attach_releases(history, rows, ['wasde_value'], decision_clock='cotton-next-day-0015-v1')
    assert pd.isna(old.wasde_value.iloc[0]) and old.wasde_value.iloc[1] == 3
    assert new.wasde_value.tolist() == [2, 3]
    assert new.decision_at.iloc[0] == pd.Timestamp('2020-06-12T00:15:00Z')
    changed = rows.copy()
    changed.loc[2, 'wasde_value'] = 999
    again = attach_releases(history, changed, ['wasde_value'], decision_clock='cotton-next-day-0015-v1')
    assert again.wasde_value.iloc[0] == new.wasde_value.iloc[0]
    with pytest.raises(ValueError, match='Unknown decision clock'):
        attach_releases(history, rows, ['wasde_value'], decision_clock='arbitrary-lag')


def test_weekend_missing_day_and_freshness_do_not_remove_origins():
    history = pd.DataFrame({'date': pd.to_datetime(['2020-06-12', '2020-06-15', '2020-06-17'])})
    manifest = {'kind': 'wasde', 'max_age_days': 2, 'decision_clock': 'cotton-next-day-0015-v1'}
    result = publications.attach_package(history, manifest, releases(['2020-06-13T00:15:00Z']), ['wasde_value'])
    assert len(result) == 3 and result.wasde_value.iloc[0] == 1
    assert result.wasde_value.iloc[1:].isna().all()
    assert result.wasde_unavailable.tolist() == [0., 1., 1.]
    for bad in [history.iloc[::-1], pd.concat([history, history.head(1)])]:
        with pytest.raises(ValueError, match='chronological'):
            publications.attach_package(bad, manifest, releases(['2020-06-13T00:15:00Z']), ['wasde_value'])


def test_cli_uses_same_clock_age_and_missingness_as_engine(tmp_path, monkeypatch):
    history = pd.DataFrame({'date': pd.to_datetime(['2020-06-11', '2020-06-15']), 'cotton_close': [60., 61.]})
    rows = releases(['2020-06-12T00:15:00Z'])
    manifest = {'kind': 'wasde', 'max_age_days': 2, 'decision_clock': 'cotton-next-day-0015-v1'}
    features = ['wasde_value']
    monkeypatch.setattr(publications, 'load_package', lambda _: (manifest, rows, features))
    input_file = tmp_path / 'history.parquet'
    history.to_parquet(input_file, index=False)
    before = digest(input_file)
    out = tmp_path / 'new'
    monkeypatch.setattr(sys, 'argv', ['publications', '--package', str(tmp_path),
        '--history', str(input_file), '--output', str(out)])
    publications.main()
    pd.testing.assert_frame_equal(pd.read_parquet(out / 'history.parquet'),
        publications.attach_package(history, manifest, rows, features))
    assert read_record(out / 'availability.json')['coverage_rows'] == 1
    assert digest(input_file) == before
    with pytest.raises(ValueError, match='overwrite'):
        publications.main()
