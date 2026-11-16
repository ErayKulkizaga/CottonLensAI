"""Build the machine-readable part of the research package (docs/research_package/data/).

Every row comes straight from the project's own ledgers; nothing is estimated here.
  - codex_fit_recipes.csv      one row per recipe in research/trials.json (completed fits, family, params, features)
  - codex_fits_pivot.csv       fits by profile x family x kind
  - codex_experiments.csv      research/registry.json records with flattened metrics
  - feature_catalog.csv        every feature name, how many recipes and which profiles used it
  - hyperparameter_ranges.csv  per family and parameter: n, min, median, max, distinct values
  - docs_catalog.csv           every Markdown document: title, date, size, first result/decision lines
  - evidence_index.csv         every file in research/evidence: size, sha256, top-level keys
Usage: python research/package/build_catalogs.py
"""
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'research_package' / 'data'


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(name, len(rows))


def kind_group(k):
    if k == 'market_fit_receipt':
        return 'market'
    if k.startswith('synthetic'):
        return 'synthetic_control'
    return 'audit_or_replication'


def recipes():
    T = json.loads((ROOT / 'research/trials.json').read_text(encoding='utf-8'))['records']
    rows, pivot, feats, params = [], Counter(), defaultdict(lambda: [0, set()]), defaultdict(list)
    for r in T:
        fits = int(r.get('completed_fits') or 0)
        recs = r.get('recipes') or [{}]
        # recipe families first: some records carry no model_families (e.g. price-only median, 342 fits)
        fams = sorted({rc['family'] for rc in recs if rc.get('family')}) or r.get('model_families') or []
        for i, rc in enumerate(recs):
            fam = rc.get('family') or (fams[0] if len(fams) == 1 else '+'.join(fams) or 'unlabelled')
            f = rc.get('features') or []
            for x in f:
                feats[x][0] += 1
                feats[x][1].add(r.get('profile'))
            for k, v in (rc.get('params') or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    params[(fam, k)].append(v)
            rows.append([r['id'][:16], r.get('profile'), r.get('kind'), kind_group(r.get('kind', '')), fits if i == 0 else 0,
                         r.get('horizon'), fam, rc.get('task'), rc.get('target'), rc.get('loss'), rc.get('window'),
                         rc.get('cadence'), rc.get('seed'), rc.get('device'), len(f), json.dumps(rc.get('params') or {}, sort_keys=True),
                         '|'.join(r.get('roles') or []), ' '.join(f)])
        for fam in (fams or ['unlabelled']):
            pivot[(r.get('profile'), fam, kind_group(r.get('kind', '')))] += fits / max(1, len(fams))
    write('codex_fit_recipes.csv', ['record_id', 'profile', 'kind', 'kind_group', 'completed_fits', 'horizon_days', 'family', 'task',
                                    'target', 'loss', 'window', 'cadence', 'seed', 'device', 'n_features', 'params_json', 'roles', 'features'], rows)
    write('codex_fits_pivot.csv', ['profile', 'family', 'kind_group', 'completed_fits'],
          sorted([[p, f, k, round(v)] for (p, f, k), v in pivot.items()], key=lambda x: -x[3]))
    write('feature_catalog.csv', ['feature', 'recipes_using', 'profiles'],
          sorted([[k, v[0], ' '.join(sorted(str(p) for p in v[1]))] for k, v in feats.items()], key=lambda x: -x[1]))
    write('hyperparameter_ranges.csv', ['family', 'param', 'n_recipes', 'min', 'median', 'max', 'distinct'],
          [[f, k, len(v), min(v), statistics.median(v), max(v), len(set(v))] for (f, k), v in sorted(params.items())])
    return sum(int(r.get('completed_fits') or 0) for r in T)


def experiments():
    R = json.loads((ROOT / 'research/registry.json').read_text(encoding='utf-8'))['records']
    rows = []
    for r in R:
        m = r.get('metrics') or {}
        s, raw = m.get('selected') or {}, m.get('raw') or {}
        rows.append([r.get('experiment'), r.get('profile'), r.get('group'), r.get('horizon'), '+'.join(r.get('model_families') or []),
                     r.get('independent_holdout'), m.get('count'), m.get('start'), m.get('end'), m.get('naive_mae'),
                     s.get('mae'), s.get('gain_pct'), s.get('all_origin_direction_pct'), s.get('active_rate_pct'),
                     raw.get('mae'), raw.get('gain_pct'), raw.get('all_origin_direction_pct'),
                     ' '.join(str(y) for y in r.get('years') or []), (r.get('ready') or '')])
    write('codex_experiments.csv', ['experiment', 'profile', 'group', 'horizon_days', 'families', 'independent_holdout', 'n_origins',
                                    'start', 'end', 'naive_mae', 'selected_mae', 'selected_gain_pct_vs_naive', 'selected_direction_pct',
                                    'selected_active_pct', 'raw_mae', 'raw_gain_pct_vs_naive', 'raw_direction_pct', 'years', 'ready_file'], rows)


def docs():
    rows = []
    paths = sorted((ROOT / 'docs').glob('*.md')) + sorted((ROOT / 'docs/archive').glob('*.md'))
    for p in paths:
        t = p.read_text(encoding='utf-8', errors='ignore')
        title = next((ln.lstrip('# ').strip() for ln in t.splitlines() if ln.startswith('#')), p.stem)
        date = re.search(r'(20\d{6})', p.name)
        hits = []
        for ln in t.splitlines():
            if re.search(r'(?i)\b(result|sonuç|karar|verdict|decision|conclusion|pass|fail|inconclusive|stop)\b', ln) and len(ln.strip()) > 20:
                hits.append(ln.strip())
            if len(' '.join(hits)) > 700:
                break
        rows.append([p.relative_to(ROOT).as_posix(), date.group(1) if date else '', title, len(t), ' / '.join(hits)[:900]])
    write('docs_catalog.csv', ['path', 'date', 'title', 'chars', 'result_and_decision_lines'], rows)


def evidence():
    rows = []
    for p in sorted((ROOT / 'research/evidence').glob('*')):
        if not p.is_file():
            continue
        b = p.read_bytes()
        keys = ''
        if p.suffix == '.json':
            try:
                j = json.loads(b)
                keys = ' '.join(list(j)[:12]) if isinstance(j, dict) else f'list[{len(j)}]'
            except ValueError:
                keys = 'unparsed'
        rows.append([p.name, len(b), hashlib.sha256(b).hexdigest(), keys])
    write('evidence_index.csv', ['file', 'bytes', 'sha256', 'top_level_keys'], rows)


if __name__ == '__main__':
    total = recipes()
    experiments()
    docs()
    evidence()
    print('total completed fits', total)


def profiles():
    """One row per research profile: fits, families, recipe settings, feature sizes, registry results, documents."""
    T = json.loads((ROOT / 'research/trials.json').read_text(encoding='utf-8'))['records']
    R = json.loads((ROOT / 'research/registry.json').read_text(encoding='utf-8'))['records']
    docs = {p: p.read_text(encoding='utf-8', errors='ignore') for p in list((ROOT / 'docs').glob('*.md')) + list((ROOT / 'docs/archive').glob('*.md'))}
    agg = defaultdict(lambda: {'fits': 0, 'records': 0, 'fam': Counter(), 'kind': Counter(), 'h': set(), 'loss': set(), 'target': set(),
                               'nf': [], 'params': Counter(), 'window': set(), 'roles': set()})
    for r in T:
        a = agg[r.get('profile')]
        n = int(r.get('completed_fits') or 0)
        a['fits'] += n
        a['records'] += 1
        a['kind'][kind_group(r.get('kind', ''))] += n
        a['roles'].update(r.get('roles') or [])
        for f in r.get('model_families') or ['unlabelled']:
            a['fam'][f] += n / max(1, len(r.get('model_families') or [1]))
        for rc in r.get('recipes') or []:
            a['h'].add(rc.get('horizon'))
            a['loss'].add(rc.get('loss'))
            a['target'].add(rc.get('target'))
            a['window'].add(rc.get('window'))
            a['nf'].append(len(rc.get('features') or []))
            a['params'][json.dumps(rc.get('params') or {}, sort_keys=True)] += 1
    res = defaultdict(list)
    for r in R:
        s = ((r.get('metrics') or {}).get('selected') or {}).get('gain_pct')
        if s is not None:
            res[r.get('profile')].append((s, r.get('group'), r.get('horizon')))
    rows = []
    for p, a in sorted(agg.items(), key=lambda x: -x[1]['fits']):
        g = res.get(p, [])
        best = max(g) if g else None
        mention = [d.relative_to(ROOT).as_posix() for d, t in docs.items() if p and p in t][:6]
        rows.append([p, a['fits'], a['records'], ' '.join(f'{k}:{round(v)}' for k, v in a['fam'].most_common()),
                     ' '.join(f'{k}:{v}' for k, v in a['kind'].items()), ' '.join(str(x) for x in sorted(a['h'], key=str)),
                     ' '.join(str(x) for x in a['loss'] if x), ' '.join(str(x) for x in a['target'] if x),
                     ' '.join(str(x) for x in sorted(a['window'], key=str)), f"{min(a['nf'])}-{max(a['nf'])}" if a['nf'] else '',
                     len(a['params']), len(g), round(statistics.median(x[0] for x in g), 3) if g else '',
                     f'{best[0]:.3f} ({best[1]}, T+{best[2]})' if best else '', ' '.join(sorted(a['roles']))[:200], ' '.join(mention)])
    write('codex_profiles.csv', ['profile', 'completed_fits', 'ledger_records', 'fits_by_family', 'fits_by_kind', 'horizons_days', 'losses',
                                 'targets', 'windows', 'features_per_recipe', 'distinct_param_sets', 'registry_scored', 'median_gain_pct_vs_naive',
                                 'best_gain_pct_vs_naive', 'roles', 'documents_mentioning'], rows)


if __name__ == '__main__':
    profiles()
