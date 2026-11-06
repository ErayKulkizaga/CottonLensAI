"""Fail-closed, role-aware dependency admission. No model fitting or data repair.

Hashes establish identity, not historical truth. Evidence must be admitted by
source-specific verifiers before building this graph; a self-asserted timestamp
is not proof of availability. Unknowns are deliberately not defaulted.
"""
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from pathlib import Path

_TRAINING_JOB = ContextVar('cottonlens_admitted_training_job', default=None)
_BACKEND_INPUTS = ContextVar('cottonlens_verified_backend_inputs', default=None)


class AdmissionError(ValueError):
    pass


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def utc(value):
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('timezone required')
        return stamp
    except (TypeError, AttributeError, ValueError) as exc:
        raise AdmissionError('Explicit timezone timestamp required') from exc


def file_hash(root, name):
    base = Path(root).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise AdmissionError('Missing or unsafe bound file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_leaf_errors(cell, files, root):
    """Cross-check claimed facts against a bound source-verification report.

    This is a consistency gate, not a signature of an authoritative publisher.
    Source-specific verifiers and original witnesses remain required.
    """
    errors = []
    name = cell.get('verification_report')
    if name not in files:
        return ['missing_source_verification_report']
    raw = (Path(root)/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != files[name]:
        raise AdmissionError('Source verification report changed during validation')
    report = json.loads(raw.decode('utf-8'))
    checks = ('source_identity', 'field_semantics', 'calendar', 'point_in_time')
    if cell.get('instrument_type') == 'futures':
        checks += ('contract_lifecycle',)
        try:
            listed, session, expiry = (date.fromisoformat(cell[k]) for k in ('listed_on', 'session', 'last_trade_on'))
            if not listed <= session <= expiry:
                errors.append('contract_outside_lifecycle')
        except (ValueError, KeyError, TypeError):
            errors.append('unknown_contract_lifecycle')
    elif cell.get('instrument_type') not in ('spot', 'index'):
        errors.append('unknown_instrument_type')
    if not report.get('verifier_source_id') or any(report.get('checks', {}).get(k) != 'VERIFIED' for k in checks):
        errors.append('incomplete_source_verification')
    expected = report.get('cells', {}).get(cell['id'], {})
    fields = ('provider', 'symbol', 'contract', 'unit', 'field', 'session', 'vintage', 'available_at', 'instrument_type')
    if cell.get('instrument_type') == 'futures':
        fields += ('listed_on', 'last_trade_on')
    if not expected or any(expected.get(k) != cell.get(k) for k in fields):
        errors.append('source_assertion_report_mismatch')
    return errors


def inspect(manifest, root):
    """Validate a complete closure; return reasons without dropping bad rows."""
    if manifest.get('schema') != 1:
        raise AdmissionError('Unsupported admission schema')
    if manifest.get('purpose') not in ('market', 'synthetic'):
        raise AdmissionError('Explicit market or synthetic purpose required')
    required = ('source', 'parser', 'calendar', 'features', 'target', 'split', 'recipe', 'code')
    bindings = manifest.get('bindings', {})
    if set(bindings) != set(required) or any(not bindings[k] for k in required):
        raise AdmissionError('Complete execution bindings required')
    files = manifest.get('files', {})
    if not files:
        raise AdmissionError('Bound evidence files required')
    for name, digest in files.items():
        if file_hash(root, name) != digest:
            raise AdmissionError('Bound file hash changed')
    cells = manifest.get('cells', [])
    nodes = {c['id']: c for c in cells}
    if not nodes or len(nodes) != len(cells):
        raise AdmissionError('Nonempty unique dependency cells required')
    jobs = manifest.get('jobs', [])
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise AdmissionError('Nonempty unique jobs required')
    errors = []
    for defect in manifest.get('open_critical_defects', []):
        errors.append({'job': None, 'cell': None, 'reason': 'critical_defect:' + defect})
    for job in jobs:
        decision, fit = utc(job['decision_at']), utc(job['fit_at'])
        if fit > decision:
            raise AdmissionError('Fit cannot occur after decision')
        roots = job.get('inputs', {})
        if set(roots) != {'feature', 'training_feature', 'training_label', 'evaluation_target'} or any(not x for x in roots.values()):
            raise AdmissionError('All input roles required; coverage cannot be silently reduced')
        for role, ids in roots.items():
            visiting, done = set(), set()

            def visit(key, role=role, job=job, decision=decision, fit=fit, visiting=visiting, done=done):
                if key in visiting:
                    raise AdmissionError('Cyclic source dependency')
                if key in done:
                    return
                if key not in nodes:
                    raise AdmissionError('Missing dependency cell')
                visiting.add(key)
                cell = nodes[key]
                reasons = []
                if cell.get('status') != 'VERIFIED':
                    reasons.append('unverified_source')
                if not cell.get('parents'):
                    if manifest['purpose'] == 'market':
                        reasons.extend(market_leaf_errors(cell, files, root))
                    for field in ('provider', 'symbol', 'contract', 'unit', 'field', 'session', 'vintage'):
                        if not cell.get(field):
                            reasons.append('missing_' + field)
                    if not cell.get('evidence') or any(p not in files for p in cell.get('evidence', [])):
                        reasons.append('unbound_source_evidence')
                if role != 'evaluation_target':
                    cutoff = decision if role == 'feature' else fit
                    available = cell.get('available_at')
                    if available is None:
                        reasons.append('unknown_availability')
                    elif utc(available) >= cutoff:
                        reasons.append('not_available_before_cutoff')
                    if role == 'training_label':
                        mature = cell.get('mature_at')
                        if mature is None or utc(mature) >= fit:
                            reasons.append('label_not_mature_before_fit')
                elif cell.get('target_policy') != manifest.get('target_policy') or not manifest.get('target_policy'):
                    reasons.append('target_vintage_policy_mismatch')
                errors.extend({'job': job['id'], 'cell': key, 'role': role, 'reason': why} for why in reasons)
                for parent in cell.get('parents', []):
                    visit(parent)
                visiting.remove(key)
                done.add(key)

            for key in ids:
                visit(key)
    return {'schema': 1, 'manifest_id': identity(manifest), 'accepted': not errors,
            'job_count': len(jobs), 'cell_count': len(cells), 'errors': errors}


def require(manifest, receipt, root, bindings):
    """Recheck bytes and closure at use time, not just receipt creation time."""
    if bindings != manifest.get('bindings'):
        raise AdmissionError('Execution bindings changed')
    current = inspect(manifest, root)
    if receipt != current or not current['accepted']:
        raise AdmissionError('Admission missing, changed or rejected')
    return current


class ExecutionAdmission:
    """Bind ledger identity and exact job specification to a checked manifest.

    The campaign builder must include ledger_identity and specification_id in
    each authorized job. This deliberately cannot authorize an arbitrary fit.
    """

    def __init__(self, manifest, receipt, root):
        # Freeze caller-owned mutable inputs to avoid silently changed authority.
        self.manifest = json.loads(json.dumps(manifest, allow_nan=False))
        self.receipt = json.loads(json.dumps(receipt, allow_nan=False))
        self.root = Path(root)

    def __call__(self, ledger_identity, specification):
        require(self.manifest, self.receipt, self.root, self.manifest['bindings'])
        if self.manifest.get('ledger_identity') != ledger_identity:
            raise AdmissionError('Ledger identity is not admitted')
        key = identity(specification)
        matching = [j for j in self.manifest['jobs'] if j.get('specification_id') == key]
        if len(matching) != 1:
            raise AdmissionError('Exact fit specification is not admitted')
        return matching[0]['id']

    @contextmanager
    def training_scope(self, ledger_identity, specification):
        job_id = self(ledger_identity, specification)
        job = next(j for j in self.manifest['jobs'] if j['id'] == job_id)
        if not job.get('model_inputs_id'):
            raise AdmissionError('Actual model input identity is not admitted')
        token = _TRAINING_JOB.set(dict(job))
        backend_token = _BACKEND_INPUTS.set(None)
        try:
            yield
        finally:
            _BACKEND_INPUTS.reset(backend_token)
            _TRAINING_JOB.reset(token)


def model_inputs_id(history, train, validation, test, recipe, iterations):
    """Bind sequence context and every split, not just training dates."""
    from cottonlens_ml.cohort import frame_identity
    frames = {'history': history, 'train': train, 'validation': validation, 'test': test}
    return identity({'recipe': recipe, 'iterations': iterations,
                     'frames': {name: None if frame is None else frame_identity(frame, list(frame.columns))
                                for name, frame in frames.items()}})


def require_model_inputs(history, train, validation, test, recipe, iterations):
    job = _TRAINING_JOB.get()
    if job is None:
        raise AdmissionError('Direct training without an admitted job is forbidden')
    actual = model_inputs_id(history, train, validation, test, recipe, iterations)
    if actual != job['model_inputs_id']:
        raise AdmissionError('Actual model inputs changed after admission')
    _BACKEND_INPUTS.set({
        'inputs_id': model_inputs_id(history, train, None, test, recipe, None),
        'qlike_id': qlike_inputs_id(train, recipe['horizon'], recipe.get('variance_features'))
        if recipe.get('family') == 'har' and recipe.get('training_loss') == 'qlike' else None})


def require_backend_inputs(history, train, test, recipe):
    """Support direct adapters and nested dispatch without weakening frame checks."""
    if _TRAINING_JOB.get() is None:
        raise AdmissionError('Direct backend training without admission is forbidden')
    expected = _BACKEND_INPUTS.get()
    if expected is None:
        require_model_inputs(history, train, None, test, recipe, None)
    elif model_inputs_id(history, train, None, test, recipe, None) != expected['inputs_id']:
        raise AdmissionError('Backend inputs changed after model admission')


def qlike_inputs_id(train, horizon, names):
    from cottonlens_ml.cohort import frame_identity
    return identity({'train': frame_identity(train, list(train.columns)), 'horizon': horizon,
                     'features': list(names) if names is not None else ['har_1', 'har_5', 'har_22']})


def require_qlike_inputs(train, horizon, names):
    """The direct optimizer must use the exact admitted backend training split."""
    expected = _BACKEND_INPUTS.get()
    if (_TRAINING_JOB.get() is None or expected is None
            or expected['qlike_id'] != qlike_inputs_id(train, horizon, names)):
        raise AdmissionError('QLIKE optimizer inputs are not admitted')


def require_legacy_fit(fit_identity):
    """Authorize exact legacy subfits; cached model reading needs no new fit."""
    job = _TRAINING_JOB.get()
    if job is None or identity(fit_identity) not in job.get('legacy_fit_ids', []):
        raise AdmissionError('Legacy estimator fit is not admitted')
