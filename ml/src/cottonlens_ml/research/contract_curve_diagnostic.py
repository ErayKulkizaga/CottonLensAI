"""Ex-post T5 error attribution. No fit, feature, origin selection or skill classification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import availability_clock as clock
from cottonlens_ml.research.ledger import freeze_record, read_record


def summarize(frames, panel):
    anchor = frames['mask_D0']
    if (anchor.date.duplicated().any() or not anchor.horizon.eq(5).all()
            or (pd.to_datetime(anchor.target_date) <= pd.to_datetime(anchor.date)).any()):
        raise ValueError('Unique T5 origins and future target dates required')
    for frame in frames.values():
        clock.require_same_targets(anchor, frame)
    quotes = {r['report_date']: r for r in panel['rows']}
    if len(quotes) != len(panel['rows']):
        raise ValueError('Duplicate quote references')
    flags, provenance = [], []
    for row in anchor.itertuples():
        a, b = quotes.get(row.date), quotes.get(row.target_date)
        known = a is not None and b is not None
        flag = 'unknown' if not known else ('transition' if a['first_contract'] != b['first_contract'] else 'unchanged')
        flags.append(flag)
        provenance.append({'date': row.date, 'target_date': row.target_date, 'ex_post_only': True,
                           'future_first_contract': None if b is None else b['first_contract'], 'stratum': flag})
    flags = np.asarray(flags)
    result = {'status': 'post_result_descriptive_only', 'selection_used': False, 'feature_used': False,
              'new_fits': 0, 'confidence_intervals': None, 'market_or_position_skill_claimed': False,
              'origin_count': len(anchor), 'counts': {s: int((flags == s).sum()) for s in ('transition', 'unchanged', 'unknown')},
              'paired': {}, 'ex_post_metadata': provenance,
              'limits': 'Future target contract is oracle metadata only. Subgroups do not change primary full-cohort decision or gates, are not independent tests and cannot drive model selection. Reference-day Final quotes are not verified availability/vintages.'}
    for delay in (0, 1):
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            _, control, _ = clock._metrics(frames[f'mask_D{delay}'], field)
            _, numeric, _ = clock._metrics(frames[f'numeric_D{delay}'], field)
            differences = control - numeric
            summary = {'full_count': len(anchor), 'full_paired_error_sum': float(differences.sum()), 'strata': {}}
            for s in ('transition', 'unchanged', 'unknown'):
                mask = flags == s
                summary['strata'][s] = {'count': int(mask.sum()),
                    'paired_error_sum': float(differences[mask].sum()),
                    'mean_paired_error_difference': float(differences[mask].mean()) if mask.any() else None,
                    'control_mae': float(control[mask].mean()) if mask.any() else None,
                    'numeric_mae': float(numeric[mask].mean()) if mask.any() else None}
            if not np.isclose(sum(v['paired_error_sum'] for v in summary['strata'].values()),
                              summary['full_paired_error_sum'], rtol=0, atol=1e-10):
                raise ValueError('Ex-post attribution lost origins/errors')
            result['paired'][f'{mode}_D{delay}'] = summary
    return result


def report(folder, ready, primary):
    folder = Path(folder)
    if primary['status'] != 'complete' or primary['profile'] != 'contract-curve-t5-pilot-v1':
        raise ValueError('Complete primary T5 comparison required before ex-post attribution')
    frames = {}
    for group in ready['identity']['design']['groups']:
        parts = [pd.DataFrame(read_record(folder / f'contract-curve-t5-outputs/{group}-t5-year{fold["year"]}.json')['records'])
                 for fold in ready['identity']['split']['folds']]
        frames[group] = pd.concat(parts, ignore_index=True)
    panel = json.loads((folder / 'preregistration/inputs/quotes.json').read_bytes())
    body = summarize(frames, panel)
    body.update(registration_id=primary['registration_id'], primary_decision_unchanged=primary['decision'],
                primary_predictions_sha256=primary['predictions_sha256'],
                ready_sha256=digest(folder / 'ready.json'),
                target_integrity_evidence_id=ready['identity']['design']['target_integrity_evidence_id'])
    freeze_record(folder / 'reports' / f'transition-attribution-{content_id(body)[:16]}.json', body)
    return body
