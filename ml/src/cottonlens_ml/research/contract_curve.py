"""Real contract-curve ablation under an explicit, unverified availability assumption."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.full_year import SHRINKAGE, chunks
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import freeze_record, read_record, writer
from cottonlens_ml.research.protocol import full_year_manifest, rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import training_rows

PROFILE = 'contract-curve-t1-pilot-v1'
T5_PROFILE = 'contract-curve-t5-pilot-v1'
RECIPE = {'family': 'ridge', 'params': {'alpha': 1.}, 'horizon': 1, 'task': 'price',
          'device': 'cpu', 'seed': 42, 'target': 'scaled_log', 'loss': 'reg:squarederror',
          'years': None, 'cadence': 21, 'window': 1}
DELAYS = (0, 1)
MAX_AGE = 3
MODE = 'assumption_sensitivity_not_historical_pit'
REFERENCE_HISTORY_SHA256 = 'f46410d913d1dc7bf798a473cc7e37c7567d40a0e5ac2a83b42901f64089bb67'
VERSIONS = ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')
RULES = {
    'primary': 'Selected numeric_D0 versus mask_D0 paired absolute price error; positive favors numeric',
    'secondary': 'D1 robustness; raw/selected, active rate, direction, annual, top1% Naive-error removal',
    'bootstrap': {'repetitions': 10000, 'seed': 42, 'blocks': [20, 60], 'preserve_years': True},
    'source_effect': 'Both D0 difference interval lower bounds >0; D1 is secondary and cannot replace primary',
    'practical_goal': {'naive_gain_pct': 5, 'direction_pct': 53, 'year_wins': '6/8'},
    'gate_evaluated': False, 'automatic_release': False,
    'interpretation': 'Three repeatedly reviewed years, unverified UTC and first vintage; not independent holdout or real-time skill',
}


def definition(profile):
    if profile == PROFILE:
        return RECIPE, RULES
    if profile != T5_PROFILE:
        raise ValueError('Only frozen contract-curve T1 or T5 profiles supported')
    return {**RECIPE, 'horizon': 5}, {
        **RULES, 'practical_goal': {**RULES['practical_goal'], 'direction_pct': 55},
        'transition_guard': 'Ex-post transition/unchanged/unknown error attribution only after full-cohort scoring; never features, cohort filters or selection',
        'overlap': 'T5 labels overlap; preserve year blocks20/60;74 crossed origins are15 observed transitions, not independent events',
        'interpretation': 'Same reviewed three years and UTC/first-vintage assumptions; continuous-target contribution is not same-contract or position skill',
    }


def bind_t5_parent(design, parent_ready, parent_proof, integrity):
    """Pin the horizon comparison to a completed T1 result, never its OOS selections."""
    if (design['profile'] != T5_PROFILE or parent_proof['profile'] != PROFILE
            or parent_proof['status'] != 'completed_assumption_sensitivity'
            or parent_ready['identity']['profile'] != PROFILE
            or parent_ready['identity']['registration_id'] != parent_proof['registration_id']
            or parent_ready['identity']['source_id'] != parent_proof['source_id']
            or design['split'] != parent_ready['identity']['split']
            or design['feature_data_id'] != parent_ready['identity']['research_data_id']
            or design['groups'] != parent_ready['identity']['design']['groups']
            or integrity['parent_registration_id'] != parent_proof['registration_id']
            or integrity['inputs_sha256']['history'] != parent_ready['history_sha256']
            or integrity['new_fits'] != 0 or integrity['market_skill_claimed'] is not False
            or integrity['historical_source_admitted'] is not False):
        raise ValueError('T5 must preserve completed T1 origins, values, groups and target-integrity boundary')
    design.update(parent_t1_registration_id=parent_proof['registration_id'],
                  parent_t1_result_evidence_id=content_id(parent_proof),
                  target_integrity_evidence_id=content_id(integrity))
    return design


def groups():
    return {f'{arm}_D{d}': names(arm, d) for d in DELAYS for arm in ('mask', 'numeric')}


def align(history, records):
    dates = history.date
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or dates.dt.tz is not None:
        raise ValueError('Unique chronological date-only Cotton history required')
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Date-only Cotton observations required')
    decision = pd.to_datetime(dates, utc=True)+pd.Timedelta(days=1, minutes=15)
    events=[]
    seen=set()
    month={'Mar':3,'May':5,'Jul':7,'Oct':10,'Dec':12}
    for row in records:
        ref=pd.Timestamp(row['report_date'])
        displayed=pd.Timestamp(row['publication_local_naive'])
        if (pd.isna(ref) or pd.isna(displayed) or ref.tz is not None or displayed.tz is not None
                or ref!=ref.normalize() or ref in seen or row['published_at'] is not None
                or row['timezone_verified'] is not False or row['first_vintage_verified'] is not False
                or row['model_eligible'] is not False):
            raise ValueError('Unique report dates and explicitly unverified clocks/vintages required')
        seen.add(ref)
        digest=row['document_sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Exact safe source SHA required')
        names=list(row['quotes'])
        if names[:2]!=[row['first_contract'],row['second_contract']]:
            raise ValueError('Original table contract order changed')
        maturities=[(2000+int(n[4:]))*12+month[n[:3]] for n in names]
        values=np.array(list(row['quotes'].values()),dtype=float)
        if maturities!=sorted(set(maturities)) or not np.isfinite(values).all() or (values<=0).any():
            raise ValueError('Ordered real contracts and finite positive quotes required')
        gap=maturities[1]-maturities[0]
        if gap!=row['front_second_gap_months']:
            raise ValueError('Contract-month gap changed')
        # Counterfactual only: this is never an asserted publisher clock.
        assumed_day=max(ref,displayed.normalize())
        assumed=pd.Timestamp(assumed_day,tz='UTC')+pd.Timedelta(days=1)
        first=int(np.searchsorted(decision.astype('int64').to_numpy(),assumed.value,side='left'))
        events.append({'first':first,'reference':ref,'assumed':assumed,'sha':digest,
                       'gap':gap,'spread':float(np.log(values[1]/values[0])),
                       'first_contract':names[0],'second_contract':names[1]})
    events.sort(key=lambda e:(e['first'],e['reference']))
    result=history.copy()
    result['curve_decision_time']=decision.to_numpy()
    for delay in DELAYS:
        age=np.full(len(history),np.nan);gap=np.full(len(history),np.nan);spread=np.full(len(history),np.nan)
        ref=[pd.NaT]*len(history);available=[pd.NaT]*len(history)
        digests=[None]*len(history);first_contract=[None]*len(history);second_contract=[None]*len(history)
        cursor=0;latest=None
        for i in range(len(history)):
            while cursor<len(events) and events[cursor]['first']+delay<=i:
                event=events[cursor]
                if latest is None or event['reference']>latest['reference']:
                    latest=event
                cursor+=1
            if latest is None:
                continue
            age[i]=i-latest['first']
            if age[i]>MAX_AGE:
                continue
            assert latest['assumed']<=decision.iloc[i]
            assert latest['reference']<=dates.iloc[i]
            gap[i]=latest['gap'];spread[i]=latest['spread']
            ref[i]=latest['reference'];available[i]=latest['assumed']
            digests[i]=latest['sha'];first_contract[i]=latest['first_contract'];second_contract[i]=latest['second_contract']
        prefix=f'curve_D{delay}_'
        result[prefix+'age_sessions']=age
        result[prefix+'unavailable']=(~np.isfinite(spread)).astype(float)
        result[prefix+'gap_months']=gap
        result[prefix+'report_date']=pd.to_datetime(ref)
        result[prefix+'assumed_available_at']=pd.to_datetime(available,utc=True)
        result[prefix+'document_sha256']=digests
        result[prefix+'first_contract']=first_contract
        result[prefix+'second_contract']=second_contract
        result[f'mask_D{delay}_spread']=np.where(np.isfinite(spread),0.,np.nan)
        result[f'numeric_D{delay}_spread']=spread
    result.attrs={}
    return result

def names(arm,delay):
    return [*FEATURE_NAMES,*[f'curve_D{delay}_{f}' for f in ('age_sessions','unavailable','gap_months')],f'{arm}_D{delay}_spread']


def make_design(history, records, *, profile=PROFILE):
    """Select years by past source feasibility, never outer performance or row completeness."""
    if history.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('Seen 2024+ audit cannot enter selection')
    recipe, rules = definition(profile)
    expanded = align(history, records)
    reference = full_year_manifest(history)
    coverage, supported = [], []
    for fold in reference['folds']:
        past = []
        for role, origins in [('outer', fold['origins']), *[(f'inner_{i}', b['origins'])
                                                          for i, b in enumerate(fold['inner'])]]:
            test = rows_at(expanded, origins)
            train = training_rows(expanded, test.date.min(), recipe, None)
            for delay in DELAYS:
                key = f'curve_D{delay}_document_sha256'
                values = train[f'numeric_D{delay}_spread']
                row = {'year': fold['year'], 'role': role, 'delay': delay,
                       'origins': len(test), 'mature_train_rows': len(train),
                       'train_reports': int(train[key].nunique()), 'test_reports': int(test[key].nunique()),
                       'test_usable_origins': int(test[f'numeric_D{delay}_spread'].notna().sum()),
                       'train_distinct_spreads': int(values.nunique()),
                       'train_std': None if not values.notna().any() else float(values.std(ddof=0))}
                coverage.append(row)
                if role.startswith('inner'):
                    past.append(row)
        if all(r['train_reports'] >= 2 and r['test_reports'] >= 2
               and r['train_distinct_spreads'] >= 2 and r['train_std'] > 1e-12 for r in past):
            supported.append(fold)
    if not supported:
        raise ValueError('No past-source-supported years')
    body = {k: v for k, v in reference.items() if k not in ('split_id', 'folds', 'label_identity')}
    body.update(version='curve-prior-source-supported-years-v2', folds=supported,
                reference_full_year_split_id=reference['split_id'],
                subset_rule='Each of three past inner blocks, both delays: >=2 train/test reports, >=2 distinct train spreads, std>1e-12; no outer selection',
                gate_evaluation_allowed=False)
    selected = rows_at(history, [d for f in supported for d in f['origins']])
    body['label_identity'] = frame_identity(selected, ['date', 'cotton_close', 'target_return_1',
                                                     'target_return_5', 'target_date_1', 'target_date_5'])
    split = {**body, 'split_id': content_id(body)}
    inner = 4 * sum(len(chunks(history, b['origins'])) for f in supported for b in f['inner'])
    outer = 4 * sum(len(chunks(history, f['origins'])) for f in supported)
    for d in DELAYS:
        np.testing.assert_array_equal(expanded[f'mask_D{d}_spread'].isna(), expanded[f'numeric_D{d}_spread'].isna())
    pd.testing.assert_frame_equal(expanded[list(history)], history, check_exact=True)
    design = {'profile': profile, 'groups': groups(), 'recipe': recipe, 'split': split, 'rules': rules,
              'shrinkage_weights': list(SHRINKAGE), 'mode': MODE,
              'clock': 'Cotton source date +1 calendar day 00:15 UTC',
              'assumed_source_clock': 'max(reference date, unverified displayed publication calendar day)+1 day 00:00 UTC; Final vintage assumed present',
              'delays': list(DELAYS), 'max_age_sessions': MAX_AGE, 'expiry_anchor': 'D0 first eligible decision',
              'selection': 'Fixed recipe; past 3x63 mean relative price-MAE; shrinkage only; ties smallest weight',
              'maturity': 'Original expanding history; common 120 warmup; target_date_5 < cutoff; no source-local reset/drop',
              'feature_data_id': frame_identity(expanded, list(expanded)),
              'fit_budget': {'inner': inner, 'outer': outer, 'total': inner + outer,
                             'synthetic_guard': 1, 'maximum_all_fits': inner + outer + 1,
                             'annual_outputs': 4 * len(supported), 'prediction_rows': 4 * len(selected),
                             'processes': 1, 'threads': 2, 'session_minutes': 30},
              'reference_origins': sum(len(f['origins']) for f in reference['folds']),
              'release_allowed': False, 'historical_source_admitted': False,
              'deliberate_repeat_reason': 'Old AMS spot/Close T+5 is not actual intercontract curve T+1; fixed core re-used solely for matched source ablation'}
    if profile == T5_PROFILE:
        design['deliberate_repeat_reason'] = 'Completed real-curve T1 is below goal; change horizon only, no T1 OOS-weight transfer; old spot-basis T5 never tested actual intercontract values'
    return expanded, design, coverage


def register(repo, output, history_path, quotes_path, proof_path, mode, *, profile=PROFILE, parent_root=None):
    """Freeze code, values, environment and rules before any synthetic or market fit."""
    repo, output, history_path, quotes_path, proof_path = map(Path, (repo, output, history_path, quotes_path, proof_path))
    if mode != MODE:
        raise ValueError('Explicit assumption mode required; no historical source admission')
    validate(repo / 'research')
    if digest(proof_path) != digest(repo / 'research/evidence/information-admission-20261010.json'):
        raise ValueError('Source inventory differs from registered public evidence')
    proof = json.loads(proof_path.read_bytes())
    panel = json.loads(quotes_path.read_bytes())
    if (digest(quotes_path) != proof['private_quote_payload_sha256'] or proof['model_eligible_rows'] != 0
            or panel['model_eligible'] is not False or len(panel['rows']) != proof['reports']):
        raise ValueError('Existing quarantined quotes changed or source admitted')
    history = pd.read_parquet(history_path)
    expanded, design, coverage = make_design(history, panel['rows'], profile=profile)
    if (digest(history_path) != REFERENCE_HISTORY_SHA256
            or [f['year'] for f in design['split']['folds']] != [2021, 2022, 2023]
            or design['fit_budget']['total'] != 252 or design['fit_budget']['prediction_rows'] != 2996):
        raise ValueError('Approved 749-origin / 252-market-fit feasibility changed')
    source = research_source_identity(repo)
    inputs = {'reference.parquet': history_path, 'quotes.json': quotes_path, 'source-proof.json': proof_path,
              'registry.json': repo / 'research/registry.json', 'trials.json': repo / 'research/trials.json'}
    if profile == T5_PROFILE:
        if parent_root is None:
            raise ValueError('Explicit completed T1 parent required before T5 preregistration')
        parent_root = Path(parent_root)
        inputs.update({'parent-ready.json': parent_root / 'ready.json',
                       'parent-history.parquet': parent_root / 'history.parquet',
                       'parent-result-proof.json': repo / 'research/evidence/contract-curve-result-20261010.json',
                       'target-integrity-proof.json': repo / 'research/evidence/curve-target-integrity-20261010.json'})
        parent_ready = read_record(inputs['parent-ready.json'])
        parent_proof = json.loads(inputs['parent-result-proof.json'].read_bytes())
        if (digest(inputs['parent-ready.json']) != parent_proof['ready_sha256']
                or digest(inputs['parent-history.parquet']) != parent_ready['history_sha256']):
            raise ValueError('Completed T1 parent bytes changed')
        pd.testing.assert_frame_equal(expanded, pd.read_parquet(inputs['parent-history.parquet']), check_exact=True)
        bind_t5_parent(design, parent_ready, parent_proof, json.loads(inputs['target-integrity-proof.json'].read_bytes()))
    identity = {'source_id': source['source_id'], 'design': design,
                'input_sha256': {n: digest(p) for n, p in inputs.items()},
                'python': sys.version.split()[0], 'versions': {n: importlib.metadata.version(n) for n in VERSIONS},
                'lock_sha256': digest(repo / 'ml/uv.lock')}
    registration_id = content_id(identity)
    registry, trials = load_registry(repo / 'research')
    scope = {'identity_id': registration_id, 'source_id': source['source_id'],
             'data_id': design['feature_data_id'], 'split_id': design['split']['split_id'], 'profile': profile}
    lookups = []
    for group, features in groups().items():
        proposal = {'scope': scope, 'scope_id': content_id(scope), 'recipe': {**design['recipe'], 'features': features}}
        result = check(registry, trials, proposal=proposal)
        if result['exact_fit_recipes']:
            raise ValueError('Already fitted frozen recipe; deliberate repeat review required')
        lookups.append({'group': group, 'proposal': proposal, 'result': result})
    with writer(output):
        if any(p.name != '.writer-lock' for p in output.iterdir()):
            raise ValueError('Preserve previous/incomplete registration; use new namespace')
        for n, path in inputs.items():
            copy_immutable(path, output / 'inputs' / n)
            if digest(output / 'inputs' / n) != identity['input_sha256'][n]:
                raise ValueError('Input changed during registration')
        expanded.to_parquet(output / 'history.parquet', index=False)
        freeze_record(output / 'source-manifest.json', source)
        for n, expected in source['files'].items():
            copy_immutable(repo / n, output / 'source-snapshot' / n)
            if digest(output / 'source-snapshot' / n) != expected:
                raise ValueError('Code changed during registration')
        freeze_record(output / 'coverage.json', {'rows': coverage, 'fits': 0})
        freeze_record(output / 'history-check.json', {'lookups': lookups, 'no_match_is_novelty_proof': False})
        reg = {'profile': profile, 'identity': identity, 'registration_id': registration_id,
               'completed_experiment': False, 'fits': 0, 'availability_verified': False, 'model_eligible': False}
        freeze_record(output / 'preregistered.json', reg)
        freeze_record(output / 'decision-contract.json', {'registration_id': registration_id, 'rules': design['rules'],
                      'completed_experiment': False, 'fits': 0, 'automatic_release': False})
        if source != research_source_identity(repo) or any(digest(p) != identity['input_sha256'][n] for n, p in inputs.items()):
            raise ValueError('Code/input changed; registration incomplete')
        freeze_record(output / 'complete.json', {'completed': True, 'fits': 0, 'model_eligible': False,
                      'registration_id': registration_id,
                      'files': {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*')
                                if p.is_file() and '.writer-lock' not in p.relative_to(output).parts}})
    return reg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'output', 'history', 'quotes', 'input-proof'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=(MODE,), required=True)
    parser.add_argument('--profile', choices=(PROFILE, T5_PROFILE), default=PROFILE)
    parser.add_argument('--parent-root', type=Path)
    args = parser.parse_args()
    reg = register(args.repo, args.output, args.history, args.quotes, args.input_proof, args.mode,
                   profile=args.profile, parent_root=args.parent_root)
    print(json.dumps({'registration_id': reg['registration_id'], 'fits': 0,
                      'fit_budget': reg['identity']['design']['fit_budget']}, indent=2))


if __name__ == '__main__':
    main()
