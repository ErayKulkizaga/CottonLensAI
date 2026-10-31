import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from research import ct_inference_calibration as analysis
from research import verify_ct_inference_calibration as verifier


def test_independent_endpoint_counts_equal_native_resampler_counts():
    for sizes, block in [((5,), 3), ((43, 209), 60)]:
        np.testing.assert_array_equal(verifier.independent_weights(sizes, block),
                                      analysis.bootstrap_weights(sizes, block))
        assert verifier.exact_iid_ratio(sizes, block) == pytest.approx(
            analysis.iid_variance_ratio(sizes, block)['ratio'])


def test_contract_changes_fail_before_analysis_or_output(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path/'repo'
    for name in analysis.FILES:
        dest = fake_repo/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, dest)
    path = fake_repo/analysis.FILES[-1]
    contract = json.loads(path.read_bytes())
    contract['blocks'] = [20, 50]
    path.write_text(json.dumps(contract), encoding='utf-8')
    with pytest.raises(ValueError, match='contract differs'):
        analysis.audit(fake_repo, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_changed_native_helper_cannot_reuse_the_frozen_identity(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path/'repo'
    for name in analysis.FILES:
        dest = fake_repo/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, dest)
    (fake_repo/analysis.FILES[2]).write_text('changed helper', encoding='utf-8')
    with pytest.raises(ValueError, match='helper differs'):
        analysis.audit(fake_repo, tmp_path/'out')
    assert not (tmp_path/'out').exists()
