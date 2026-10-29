"""Zero-fit named-risk temporal readiness on the official daily AMS archive."""

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.named_risk_history import plan

from research.ams_upland_futures import sha


def quotes_from(observations):
    ranked = defaultdict(set)
    prices = defaultdict(dict)
    with observations.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['source_date_matches'] != 'True':
                continue
            day, contract = row['date'], row['contract']
            ranked[day].add(contract)
            if row['price_precision_2dp'] == 'True':
                price = float(row['settlement_cents_raw'])
                if not math.isfinite(price) or price <= 0:
                    raise ValueError('Invalid strict named price')
                if contract in prices[day] and prices[day][contract] != price:
                    raise ValueError('Duplicate named price conflict')
                prices[day][contract] = price
    return {day: {'ranked': sorted(contracts), 'prices': prices[day]}
            for day, contracts in ranked.items()}


def attach(history, quotes):
    if (history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or history.date.max() >= pd.Timestamp('2024-01-01')):
        raise ValueError('Original chronological pre-2024 parent required')
    h = history.reset_index(drop=True).copy()
    dates = h.date.dt.strftime('%Y-%m-%d').tolist()
    positions = {day: i for i, day in enumerate(dates)}
    events = defaultdict(int)
    squares = defaultdict(lambda: deque(maxlen=22))
    state = {}
    labels = []
    for i, day in enumerate(dates):
        current = quotes.get(day)
        previous = quotes.get(dates[i-1]) if i else None
        if previous is not None and current is not None:
            for contract in previous['prices'].keys() & current['prices'].keys():
                square = math.log(current['prices'][contract]
                                  / previous['prices'][contract]) ** 2
                state[contract] = (.94*state[contract]+.06*square
                                   if contract in state else square)
                events[contract] += 1
                squares[contract].append((day, square))
        next_day = dates[i+1] if i+1 < len(dates) else None
        future = quotes.get(next_day)
        contract = current['ranked'][1] if current and len(current['ranked']) >= 2 else None
        eligible = bool(contract and contract in current['prices']
                        and future and contract in future['prices'])
        reason = ('conditional_exact_date_label' if eligible else
                  'missing_exact_origin_table' if not current else
                  'origin_second_invalid' if not contract or contract not in current['prices'] else
                  'missing_exact_target_table' if not future else
                  'origin_contract_absent_at_target')
        actual = (math.log(future['prices'][contract]
                           / current['prices'][contract]) ** 2 if eligible else float('nan'))
        baseline_ready = bool(contract and contract in current['prices']
                              and contract in state and events[contract] >= 22)
        past = list(squares[contract]) if baseline_ready else []
        own = {f'own_har_{k}': (math.log(max(sum(value for _, value in past[-k:])/k,
                                            1e-12)) if past else float('nan')) for k in (1, 5, 22)}
        labels.append({'named_eligible': eligible, 'named_variance': actual,
                       'named_contract': contract, 'named_reason': reason,
                       'named_ewma': max(state[contract], 1e-12) if baseline_ready else float('nan'),
                       'named_ewma_observations': events[contract] if contract else 0,
                       'named_ewma_last_date': past[-1][0] if past else None,
                       'named_ewma_eligible': baseline_ready,
                       'own_feature_ready': baseline_ready, **own,
                       'own_observed_pairs': events[contract] if contract else 0,
                       'own_latest_pair_date': past[-1][0] if past else None,
                       'own_latest_pair_age': positions[day]-positions[past[-1][0]] if past else None,
                       'own_22_session_span': positions[past[-1][0]]-positions[past[0][0]]+1
                       if past else None,
                       'named_label_available_at': pd.Timestamp(next_day, tz='UTC')
                       + pd.Timedelta(days=1) if eligible else pd.NaT})
    return pd.concat([h, pd.DataFrame(labels)], axis=1)


def run(source, history_path, split_path, output):
    source = Path(source)
    observations = source.parent/'observations.csv'
    metadata = json.loads(source.read_text(encoding='utf-8'))
    if (metadata['profile'] != 'ams-upland-futures-source-preflight-v1'
            or metadata['observations_sha256'] != sha(observations)
            or metadata['source_clock_verified'] or metadata['training_allowed']):
        raise ValueError('Exact zero-fit source preflight required')
    history = pd.read_parquet(history_path)
    split = json.loads(Path(split_path).read_text(encoding='utf-8'))
    if ([f['year'] for f in split['folds']] != list(range(2016, 2024))
            or sum(len(f['origins']) for f in split['folds']) != 2006):
        raise ValueError('Original full-year cohort required')
    h = attach(history, quotes_from(observations))
    design = plan(h, split)
    yearly = []
    for fold, ready in zip(split['folds'], design['readiness'], strict=True):
        rows = h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[fold['origins']]
        yearly.append({**ready, 'origin_second_pairs': int(rows.named_eligible.sum()),
                       'baseline_22': int(rows.named_ewma_eligible.sum()),
                       'paired_ready': int((rows.named_eligible & rows.named_ewma_eligible).sum())})
    result = {'profile': 'ams-named-risk-full-year-readiness-v1', 'new_fits': 0,
              'source_result_sha256': sha(source), 'observations_sha256': sha(observations),
              'history_sha256': sha(history_path), 'split_sha256': sha(split_path),
              'analysis_sha256': sha(__file__), 'yearly': yearly,
              'supported_years': sum(r['supported'] for r in yearly),
              'source_clock': 'date+1day00:00UTC assumed; not first-vintage verified',
              'source_clock_verified': False, 'training_allowed': False,
              'gate_evaluation_allowed': False, '2024_plus_used_for_selection': False}
    if output.exists():
        raise ValueError('Immutable output namespace already exists')
    output.mkdir(parents=True)
    h.to_parquet(output/'history.parquet', index=False)
    (output/'plan.json').write_text(json.dumps(design, indent=2)+'\n', encoding='utf-8')
    result['history_with_named_sha256'] = sha(output/'history.parquet')
    result['plan_sha256'] = sha(output/'plan.json')
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'supported_years': result['supported_years'], 'yearly': yearly}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.history, args.split, args.output)
