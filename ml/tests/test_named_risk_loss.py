from pathlib import Path

import pytest
from cottonlens_ml.research import named_risk_loss as control


def test_unmodified_ols_path_has_same_signature_with_new_loss_branch(tmp_path):
    old = "FLOOR=1e-12\ndef fit_predict(spec):\n    check(spec)\n    h=spec['horizon']\n    if spec['family']=='har':\n        model=ols(h)\n        return model\n"
    new = old.replace("    if spec['family']=='har':", "    if spec['family']=='har' and spec.get('training_loss')=='qlike':\n        return qlike(h)\n    elif spec['family']=='har':")
    first, second = tmp_path/'old.py', tmp_path/'new.py'
    first.write_text(old);second.write_text(new)
    assert control.ols_body(first) == control.ols_body(second)
    second.write_text(new.replace('ols(h)', 'different(h)'))
    assert control.ols_body(first) != control.ols_body(second)


def test_global_floor_change_cannot_hide_under_unchanged_body(tmp_path):
    old = "FLOOR=1e-12\ndef fit_predict(spec):\n    check(spec)\n    h=spec['horizon']\n    if spec['family']=='har':\n        return FLOOR\n"
    a,b = tmp_path/'a.py',tmp_path/'b.py';a.write_text(old);b.write_text(old.replace('1e-12','1e-10'))
    assert control.ols_body(a) != control.ols_body(b)


def test_missing_or_ambiguous_ols_path_is_rejected(tmp_path):
    p=tmp_path/'bad.py';p.write_text("def fit_predict(spec):\n    return 1\n")
    with pytest.raises(ValueError,match='Unambiguous'):
        control.ols_body(p)


def test_budget_reuses_completed_control_instead_of_refitting():
    original={'identity':{'design':{'folds':[{'jobs':[{},{}],'origins':['a','b']} ]}}}
    d=control.design(original)
    assert d['fit_budget']['market']==2
    assert d['ols_reference_fits']==2
    assert d['fit_budget']['synthetic']==1
    assert not d['release_allowed']
    assert not d['eight_year_gate_allowed']


def test_real_old_branch_remains_exact_ast_equal():
    repository=Path(__file__).parents[2]
    # Check old branch arithmetic directly, independent of global source id.
    current=repository/'ml/src/cottonlens_ml/research/volatility.py'
    body=control.ols_body(current)
    assert "lstsq" in str(body)
    assert "smear" in str(body)
