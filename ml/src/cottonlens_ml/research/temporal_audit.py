"""Functional causality sentry for feature builders; no fitting or source admission."""
import json

import numpy as np
import pandas as pd

from cottonlens_ml.cohort import frame_identity
from cottonlens_ml.research.admission import identity


class TemporalAuditFailure(ValueError):
    pass


def audit_features(history, builder, columns, cutoffs):
    """Compare full-history outputs against independently rebuilt prefixes.

    Cutoffs are row counts. This verifies recorded-row causality only, not
    historical vendor availability, target correctness or unseen model skill.
    The caller must freeze cutoffs/columns before interpreting results.
    """
    dates = pd.DatetimeIndex(history.date)
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise TemporalAuditFailure('Unique chronological source observations required')
    if not columns or len(set(columns)) != len(columns):
        raise TemporalAuditFailure('Explicit unique feature columns required')
    if not cutoffs or len(set(cutoffs)) != len(cutoffs):
        raise TemporalAuditFailure('Explicit unique cutoffs required')
    if any(type(n) is not int or not 1 <= n < len(history) for n in cutoffs):
        raise TemporalAuditFailure('Each cutoff must retain both a prefix and future rows')
    original = history.copy(deep=True)

    def build(frame):
        before = frame.copy(deep=True)
        result = builder(frame)
        try:
            pd.testing.assert_frame_equal(frame, before)
        except AssertionError as exc:
            raise TemporalAuditFailure('Feature builder mutated its source') from exc
        if (not isinstance(result, pd.DataFrame) or len(result) != len(before)
                or 'date' not in result or not result.date.reset_index(drop=True).equals(before.date.reset_index(drop=True))):
            raise TemporalAuditFailure('Feature builder dropped/reordered source origins')
        if any(c not in result for c in columns):
            raise TemporalAuditFailure('Declared feature missing')
        return result[columns].reset_index(drop=True)

    full = build(original.copy(deep=True))
    for n in cutoffs:
        prefix = build(original.iloc[:n].copy(deep=True))
        try:
            pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True), prefix,
                                          check_exact=True, check_dtype=True)
        except AssertionError as exc:
            raise TemporalAuditFailure(f'Future rows changed earlier features at cutoff {n}') from exc
        # A second check perturbs future numeric observations while leaving the
        # date/session axis intact. Target data are not inferred to be features.
        changed = original.copy(deep=True)
        for c in changed.select_dtypes(include=[np.number]).columns:
            if c != 'cotton_session_index':
                changed.iloc[n:, changed.columns.get_loc(c)] = changed[c].iloc[n:]*7+123
        future_changed = build(changed)
        try:
            pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True),
                                          future_changed.iloc[:n].reset_index(drop=True), check_exact=True)
        except AssertionError as exc:
            raise TemporalAuditFailure(f'Future perturbation changed earlier features at cutoff {n}') from exc
    return {'status': 'functional_prefix_and_future_perturbation_passed',
            'source_frame_id': frame_identity(original, list(original.columns)),
            'feature_frame_id': frame_identity(full, list(full.columns)),
            'columns': list(columns), 'cutoffs': list(cutoffs), 'new_fits': 0,
            'historical_source_availability_proven': False, 'market_skill': False}


def audit_predictions(history, predictor, state, columns, cutoffs):
    """Audit non-fitting native inference with frozen recipe/selection/model state.

    The caller supplies a read-only native predictor, not a training callback.
    Mutation of passed state is rejected; this is not a universal detector for
    hidden process-global state or a proof that model selection was valid.
    """
    frozen = json.loads(json.dumps(state, allow_nan=False))
    if not all(k in frozen for k in ('recipe', 'selection', 'native')):
        raise TemporalAuditFailure('Explicit frozen recipe, selection and native state required')
    state_id = identity(frozen)

    def build(frame):
        context = json.loads(json.dumps(frozen))
        result = predictor(frame, context)
        if identity(context) != state_id:
            raise TemporalAuditFailure('Predictor mutated frozen model or selection state')
        return result

    result = audit_features(history, build, columns, cutoffs)
    result['status'] = 'fixed_state_prediction_prefix_and_future_perturbation_passed'
    result['prediction_frame_id'] = result.pop('feature_frame_id')
    result['fixed_state_id'] = state_id
    result['selection_procedure_validity_proven'] = False
    return result
