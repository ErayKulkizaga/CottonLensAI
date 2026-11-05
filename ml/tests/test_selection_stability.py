import numpy as np
import pytest
from cottonlens_ml.research.selection_stability import analyze, bootstrap, scores


def test_stable_winner_and_paired_ties():
    parts = [np.tile([1., .8, .6, .4, .2], (63, 1)) for _ in range(3)]
    result = bootstrap(parts, 4, block=20, repetitions=100)
    assert result['selection_frequencies'] == [0., 0., 0., 0., 1.]
    np.testing.assert_allclose(result['score_gain_ci_pp']['versus_zero'], [80., 80.])
    result = bootstrap([np.ones((63, 5))] * 3, 0, block=60, repetitions=100)
    assert result['selection_frequencies'] == [1., 0., 0., 0., 0.]
    assert result['score_gain_ci_pp']['versus_raw'] == [0., 0.]


def test_ratio_of_losses_not_average_row_ratios():
    part = np.array([[1., 1., 1., 1., 1.], [9., 4., 4., 4., 4.]])
    np.testing.assert_array_equal(scores([part] * 3), [1., .5, .5, .5, .5])


@pytest.mark.parametrize('change', ['nan', 'negative', 'zero_baseline', 'wrong_choice', 'missing_block'])
def test_invalid_loss_or_choice_is_rejected(change):
    parts = [np.ones((63, 5)) for _ in range(3)]
    choice = 0
    if change == 'nan':
        parts[0][0, 2] = np.nan
    elif change == 'negative':
        parts[0][0, 2] = -1
    elif change == 'zero_baseline':
        parts[0][:, 0] = 0
    elif change == 'wrong_choice':
        choice = 4
    else:
        parts.pop()
    with pytest.raises(ValueError):
        bootstrap(parts, choice, block=20, repetitions=100)


def test_deterministic_temporally_paired_resampling():
    rng = np.random.default_rng(4)
    parts = [rng.uniform(.1, 2., (63, 5)) for _ in range(3)]
    choice = int(np.argmin(scores(parts)))
    a = bootstrap(parts, choice, block=20, repetitions=100)
    assert a == bootstrap(parts, choice, block=20, repetitions=100)
    assert sum(a['selection_frequencies']) == pytest.approx(1.)


def test_original_study_cannot_be_output(tmp_path):
    with pytest.raises(ValueError, match='outside immutable study'):
        analyze(tmp_path, tmp_path / 'absent.json', tmp_path / 'reports')


def test_inventory_tampering_fails_before_output(tmp_path):
    import json

    study = tmp_path / 'study'
    study.mkdir()
    (tmp_path / 'checksums.json').write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'inventory_member': 'checksums.json', 'inventory_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='inventory changed'):
        analyze(study, manifest, tmp_path / 'analysis')
    assert not (tmp_path / 'analysis').exists()
