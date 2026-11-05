"""Zero-fit risk measurement feasibility, not a named-contract learner test."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import struct
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_record(path):
    value = json.loads(Path(path).read_bytes())
    body = {k: v for k, v in value.items() if k != 'record_id'}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    if value.get('record_id') != expected:
        raise ValueError('Record identity mismatch')
    return body


def components(previous_close, opening, close):
    if any(not math.isfinite(p) or p <= 0 for p in (previous_close, opening, close)):
        raise ValueError('Positive finite price fields required')
    gap, within = math.log(opening / previous_close), math.log(close / opening)
    total = math.log(close / previous_close)
    interaction = 2 * gap * within
    if not math.isclose(total ** 2, gap ** 2 + within ** 2 + interaction, abs_tol=1e-12):
        raise ValueError('Squared-return component identity failed')
    return {'gap_squared': gap ** 2, 'within_bar_squared': within ** 2,
            'interaction': interaction, 'total_squared': total ** 2}


def assumed_available(quote):
    # This calendar is explicitly NOT a historical publication timestamp.
    reference = date.fromisoformat(quote['report_date'])
    publication = quote.get('publication_local_naive')
    calendar = max(reference, date.fromisoformat(publication[:10])) if publication else reference
    return datetime.combine(calendar + timedelta(days=1), time(), UTC)


def named_label(current, future, origin, target, rank):
    result = {'eligible': False, 'reason': '', 'contract': None, 'risk': None,
              'label_available_assumed': None, 'historical_availability_verified': False}
    if current is None:
        return {**result, 'reason': 'missing_exact_origin_table'}
    if current['report_date'] != origin:
        raise ValueError('Origin report date mismatch; stale quotes forbidden')
    decision = datetime.combine(date.fromisoformat(origin) + timedelta(days=1), time(0, 15), UTC)
    if assumed_available(current) > decision:
        return {**result, 'reason': 'origin_calendar_assumed_late'}
    contract = current[rank + '_contract']
    result['contract'] = contract
    if future is None:
        return {**result, 'reason': 'missing_exact_target_table'}
    if future['report_date'] != target or target <= origin:
        raise ValueError('Target report date mismatch')
    if contract not in future['quotes']:
        return {**result, 'reason': 'origin_contract_absent_at_target'}
    initial, final = current['quotes'][contract], future['quotes'][contract]
    if any(not math.isfinite(v) or v <= 0 for v in (initial, final)):
        raise ValueError('Invalid named price; do not turn corruption into missingness')
    change = math.log(final / initial)
    return {**result, 'eligible': True, 'reason': 'conditional_exact_date_label',
            'risk': change ** 2, 'return': change, 'initial_price': initial, 'final_price': final,
            'origin_document_sha256': current['document_sha256'],
            'target_document_sha256': future['document_sha256'],
            'label_available_assumed': assumed_available(future).isoformat()}


def qlike(actual, prediction):
    if not math.isfinite(actual) or actual < 0 or not math.isfinite(prediction) or prediction <= 0:
        raise ValueError('Valid variance/forecast required')
    return math.log(prediction) + actual / prediction


def scale_projection(count, effect, standard_error, forward_count=126):
    if count <= 0 or standard_error < 0 or forward_count <= 0:
        raise ValueError('Invalid sample/error scale')
    return {'reference_origins': count, 'observed_effect': effect,
            'conditional_origins_at_observed_effect': math.ceil(count * (2.8 * standard_error / effect) ** 2)
            if effect > 0 else None,
            'forward_origins': forward_count,
            'conditional_forward_mde_80pct': 2.8 * standard_error * math.sqrt(count / forward_count),
            'status': 'fixed-noise scaling diagnostic, NOT calibrated power or a time guarantee',
            'assumptions': 'same effect/noise/dependence; square-root sample scaling; no selection adjustment'}


def first_price_parity(quote, price):
    if quote is None:
        return False
    named = quote['quotes'][quote['first_contract']]
    return float(struct.unpack('f', struct.pack('f', named))[0]) == price


def write_json(path, body):
    if path.exists():
        raise ValueError('Immutable output already exists')
    record_id = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps({**body, 'record_id': record_id}, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')


def run(inputs, contract_path, output):
    import numpy as np
    import pandas as pd

    contract = read_record(contract_path)
    if output.exists() or contract['new_fits'] != 0 or contract['primary_rank'] != 'second':
        raise ValueError('New namespace and fixed zero-fit descriptive specification required')
    for name, expected in contract['input_hashes'].items():
        if sha(inputs / name) != expected:
            raise ValueError('Input hash mismatch')
    panel = read_record(inputs / 'quotes.json')
    parent = read_record(inputs / 'risk-report.json')
    proof = read_record(inputs / 'risk-proof.json')
    if (sha(inputs / 'history.parquet') != parent['input_hashes']['original/history.parquet']
            or sha(inputs / 'paired-losses.csv') != parent['paired_losses_sha256']
            or sha(inputs / 'risk-report.json') != proof['report_sha256']
            or sha(inputs / 'quotes.json') != read_record(inputs / 'quote-proof.json')['candidate_panel_sha256']):
        raise ValueError('Parent evidence binding mismatch')
    if panel['model_eligible'] or panel['training_ready'] or panel['release_allowed']:
        raise ValueError('Unadmitted quote candidate must remain unadmitted')
    quotes = {q['report_date']: q for q in panel['rows']}
    if len(quotes) != len(panel['rows']) or any(q['model_eligible'] for q in quotes.values()):
        raise ValueError('Duplicate dates or silently admitted source')
    history = pd.read_parquet(inputs / 'history.parquet')
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted observations required')
    dates = history.date.dt.strftime('%Y-%m-%d').tolist()
    positions = {d: i for i, d in enumerate(dates)}
    raw = list(csv.DictReader((inputs / 'paired-losses.csv').open(encoding='utf-8', newline='')))
    origins = [r for r in raw if r['horizon'] == '1']
    if len(origins) != 2006 or len({r['date'] for r in origins}) != 2006:
        raise ValueError('Keep all2006 parent origins')
    # Reuse the already verified full-ordinal bootstrap; shift BOTH losses to
    # satisfy its nonnegative contract. Only absolute delta/CI are reported.
    spec = importlib.util.spec_from_file_location('frozen_transport_blocks', inputs / 'bootstrap.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, labels, transition = [], [], []
    for origin in origins:
        d, target = origin['date'], origin['target_date']
        i = positions[d]
        if i + 1 >= len(dates) or dates[i + 1] != target:
            raise ValueError('T+1 target-date identity mismatch')
        now, nxt = history.iloc[i], history.iloc[i + 1]
        actual = math.log(nxt.cotton_close / now.cotton_close) ** 2
        if not math.isclose(actual, float(origin['actual_variance']), abs_tol=1e-12, rel_tol=1e-10):
            raise ValueError('Parent actual risk mismatch')
        parts = components(now.cotton_close, nxt.cotton_open, nxt.cotton_close)
        valid_range = bool(nxt.cotton_low > 0 and nxt.cotton_high >= max(nxt.cotton_open, nxt.cotton_close)
                           and nxt.cotton_low <= min(nxt.cotton_open, nxt.cotton_close))
        current, future = quotes.get(d), quotes.get(target)
        changed = current is not None and future is not None and current['first_contract'] != future['first_contract']
        front_status = ('unknown' if current is None or future is None else
                        'observed_changed' if changed else 'observed_unchanged')
        matched = first_price_parity(current, now.cotton_close) and first_price_parity(future, nxt.cotton_close)
        rows.append({'date': d, 'target_date': target, 'year': int(origin['year']), **parts,
                     'ohlc_range_consistent': valid_range, 'candidate_front_name_changed': changed,
                     'front_status': front_status,
                     'both_first_prices_match_ct_float32': matched,
                     'parent_qlike_difference': float(origin['difference'])})
        if changed and matched and future['first_contract'] in current['quotes']:
            a, b = current['first_contract'], future['first_contract']
            gap = math.log(current['quotes'][b] / current['quotes'][a])
            same = math.log(future['quotes'][b] / current['quotes'][b])
            total = math.log(future['quotes'][b] / current['quotes'][a])
            if not math.isclose(total ** 2, gap ** 2 + same ** 2 + 2 * gap * same, abs_tol=1e-12):
                raise ValueError('Expost curve risk identity failed')
            transition.append({'date': d, 'target_date': target, 'origin_front': a, 'future_front': b,
                               'front_mixed_risk': total ** 2, 'gap_squared': gap ** 2,
                               'same_future_contract_squared': same ** 2, 'interaction': 2 * gap * same})
        for rank in ('first', 'second'):
            label = named_label(current, future, d, target, rank)
            entry = {'date': d, 'target_date': target, 'year': int(origin['year']), 'rank': rank, **label,
                     'model_variance': float(origin['variance']), 'ewma_variance': float(origin['ewma_variance']),
                     'ct_qlike_difference': float(origin['difference'])}
            if label['eligible']:
                entry['named_baseline_loss'] = qlike(label['risk'], entry['ewma_variance'])
                entry['named_model_loss'] = qlike(label['risk'], entry['model_variance'])
                entry['named_qlike_difference'] = entry['named_baseline_loss'] - entry['named_model_loss']
            labels.append(entry)
    frame = pd.DataFrame(rows)
    named = pd.DataFrame(labels)
    comparisons = {}
    for rank in ('first', 'second'):
        arm = named.loc[named['rank'].eq(rank)].copy()
        eligible = arm.eligible
        valid = arm.loc[eligible]
        offset = max(0., 1 - min(valid.named_baseline_loss.min(), valid.named_model_loss.min()))
        arm['base_shifted'] = np.where(eligible, arm.named_baseline_loss.fillna(0) + offset, 0)
        arm['model_shifted'] = np.where(eligible, arm.named_model_loss.fillna(0) + offset, 0)
        arm['weight'] = eligible.astype(int)
        parts = [g[['base_shifted', 'model_shifted', 'weight']].to_numpy() for _, g in arm.groupby('year')]
        intervals = {}
        for block in (20, 60):
            bootstrap = module.weighted_bootstrap(parts, block=block)
            intervals[str(block)] = {k: bootstrap[k] for k in ('eligible', 'ordinal_slots', 'block',
                                                               'replicates', 'paired_difference', 'difference_ci_95')}
        comparisons[rank] = {'eligible': len(valid), 'retained_origins': len(arm),
            'missing_reasons': dict(Counter(arm.loc[~eligible, 'reason'])),
            'ct_target_on_same_origins_delta': float(valid.ct_qlike_difference.mean()),
            'named_target_delta': float(valid.named_qlike_difference.mean()), 'bootstrap': intervals,
            'annual': [{'year': int(y), 'eligible': len(g),
                        'named_delta': float(g.named_qlike_difference.mean()),
                        'ct_delta_same_origins': float(g.ct_qlike_difference.mean())} for y, g in valid.groupby('year')],
            'scope': 'Fixed CT-trained forecasts and CT EWMA, different target; NOT named-trained risk skill'}
    # Pure source/label feasibility uses all historical observation slots,
    # not the return magnitude, outside performance or a reduced train gate.
    historical_labels = []
    for i, d in enumerate(dates[:-5]):
        label = named_label(quotes.get(d), quotes.get(dates[i+1]), d, dates[i+1], 'second')
        if label['eligible']:
            historical_labels.append({'date': d, 'target_date_5': dates[i+5],
                                      'available': label['label_available_assumed']})
    feasibility = []
    for year in range(2016, 2024):
        start = next(o['date'] for o in origins if int(o['year']) == year)

        def past_labels(cutoff):
            decision = datetime.combine(date.fromisoformat(cutoff) + timedelta(days=1), time(0,15), UTC)
            return [x for x in historical_labels if x['date'] < cutoff and x['target_date_5'] < cutoff
                    and datetime.fromisoformat(x['available']) <= decision]

        validation = past_labels(start)[-189:]
        counts = [len(past_labels(validation[j]['date'])) for j in (0,63,126)] if len(validation)==189 else []
        feasibility.append({'year':year,'validation_labels':len(validation),'inner_train_counts':counts,
                            'supports_existing_500_train_189_validation_rule': bool(counts and min(counts)>=500)})
    total = frame.total_squared.sum()
    contributions = [{'front_status':key, 'origins':len(group),
                      'risk_sum':float(group.total_squared.sum()),
                      'qlike_difference_sum':float(group.parent_qlike_difference.sum())}
                     for key,group in frame.groupby('front_status')]
    projections = {b: scale_projection(parent['horizons']['1']['count'],
                        parent['horizons']['1']['mean_difference'], v['standard_error'])
                   for b, v in parent['horizons']['1']['bootstrap'].items()}
    report = {'profile': 'risk-measurement-readiness-v1', 'new_fits': 0, 'new_dataset_downloads': 0,
              'classification': 'INCONCLUSIVE', 'classification_scope':'Descriptive measurement and fixed-forecast target sensitivity only',
              'origin_count':2006, 'quote_dates':len(quotes), 'named_comparisons':comparisons,
              'price_field_components':{'gap_squared_sum':float(frame.gap_squared.sum()),
                'within_bar_squared_sum':float(frame.within_bar_squared.sum()), 'interaction_sum':float(frame.interaction.sum()),
                'total_squared_sum':float(total), 'ohlc_consistent_target_bars':int(frame.ohlc_range_consistent.sum()),
                'scope':'Recorded price-field identity; gap is NOT verified tradeable overnight return; inconsistent range NOT proof of corruption'},
              'front_transition':{'changed_origins':int(frame.candidate_front_name_changed.sum()),
                'matched_ct_and_curve_transition_origins':len(transition), 'raw_rows':transition,
                'contributions':contributions,
                'positive_total_contribution_fraction':float(frame.loc[frame.front_status.eq('observed_changed'),
                    'parent_qlike_difference'].sum()/frame.parent_qlike_difference.sum()),
                'scope':'Observed report front-name changes, NOT verified exchange roll dates; future front used only expost identity'},
              'conditional_training_feasibility':feasibility, 'power_scale_diagnostic':projections,
              'intraday_variance_measured':False, 'verified_historical_publication_vintage_rows':
                sum(bool(q.get('timezone_verified') and q.get('first_vintage_verified')) for q in quotes.values()),
              'validated_learning_effect_or_no_signal':False, 'release_allowed':False,
              'conditional_daily_close_label_ready':any(x['supports_existing_500_train_189_validation_rule']
                                                       for x in feasibility),
              'intraday_absence_is_not_a_universal_risk_forecasting_obstacle':True,
              'eight_year_named_training_protocol_ready':all(x['supports_existing_500_train_189_validation_rule']
                                                             for x in feasibility),
              'decision':'NO_GO_CURRENT_126_ORIGIN_CONFIRMATORY_DESIGN',
              'next_single_missing_input':'Same-contract risk target with auditable timestamps and historical coverage adequate for a prespecified useful effect; do not extend current model/grid on this evidence',
              'input_hashes':contract['input_hashes'], 'code_sha256':sha(Path(__file__))}
    for name, expected in contract['input_hashes'].items():
        if sha(inputs/name)!=expected:
            raise ValueError('Inputs changed during analysis')
    output.mkdir(parents=True)
    frame.to_csv(output/'origin-components.csv',index=False,lineterminator='\n')
    named.to_csv(output/'named-target-sensitivity.csv',index=False,lineterminator='\n')
    report['output_hashes']={n:sha(output/n) for n in ('origin-components.csv','named-target-sensitivity.csv')}
    write_json(output/'report.json',report)
    print(json.dumps({'decision':report['decision'],'origins':2006,'new_fits':0,
                      'named_counts':{k:v['eligible'] for k,v in comparisons.items()}}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','contract','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(a.inputs,a.contract,a.output)


if __name__=='__main__':
    main()
