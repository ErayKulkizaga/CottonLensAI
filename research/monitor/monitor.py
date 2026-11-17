"""USDA cotton forecast monitor: point-in-time, gated, logged.

  run [YYYY-MM] [--log PATH]   process the Crop Production report of that month (default: current month)
  verify [--log PATH]          check the hash chain of the log

For one report it:
  1. fetches the report and the previous month's report from ESMIS (public text, no API key);
  2. parses state/US upland production and the Texas harvested area and yield, with the validated parsers;
  3. fetches the latest NASS Crop Progress cotton condition released on or before the survey date (the 1st);
  4. runs gates (states sum to US within 2%, Texas present, area x yield = production within 3%, condition at most
     21 days old, revision within +-50%); any failed gate makes that output fall back to USDA's own number;
  5. applies the frozen model (model_v2.json since 17 Nov 2026): momentum correction for US/Texas production,
     anchoring correction with a real-time-calibrated 80% / 95% interval for Texas harvested area (CAL1);
  6. appends one record to an append-only JSONL log whose records are hash-chained (sha256 of the previous record).
A report already in the log is not processed again.
"""
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.cond_pull import parse as parse_condition
from research.crop_nowcast.t47_blind_old import clean
from research.crop_nowcast.t48_mechanism import rows as parse_area_yield

BASE = 'https://esmis.nal.usda.gov'
MON = {8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
DEFAULT_LOG = ROOT / 'research/monitor/log.jsonl'
MODEL = ROOT / 'research/monitor/model_v2.json'  # v2 = v1 + CAL1 intervals; earlier v1 records stay in the log


def http(url):
    import time
    import urllib.request
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research-client'}),
                                          timeout=120).read()
        except OSError:
            time.sleep(10 * (k + 1))
    raise RuntimeError('fetch failed: ' + url.split('?')[0])


def report_texts(kind, year, month):
    page = http(f'{BASE}/publication/{kind}?date={year}-{month:02d}').decode('utf-8', 'ignore')
    # the newest report is linked under a fixed "current release" path on every page; it is kept here, and the
    # release-date check in crop_production / condition_tx discards it whenever it belongs to another month
    hrefs = sorted(set(re.findall(r'href="(/sites/default/release-files/[^"]+\.txt)"', page)))
    return [(h, http(BASE + h)) for h in hrefs]


TITLE = r'Cotton:?\s+Area Harvested, Yield, and Production,? by Type'


def parse_report(text):
    """Every occurrence of the cotton production table title is tried (the first may be the table of contents, and a
    September report also has an area-only table); the occurrence whose Upland block has the most rows passing
    production = area x yield (within 3%) wins. -> (release date, {state/US: (area_1000ac, yield_lb, prod_1000bales)})."""
    r = re.search(r'Released ([A-Za-z]+ \d{1,2}, \d{4})', text)
    if not r:
        return None
    rel = dt.datetime.strptime(r.group(1), '%B %d, %Y').date()
    best = {}
    for t in re.finditer(TITLE, text):
        try:
            res = parse_area_yield(r.group(0) + '\n' + text[t.start():])
        except ValueError:
            continue
        if res and len(res[1]) > len(best):
            best = res[1]
    return (rel, best) if best else None


def crop_production(year, month):
    """-> (release date, {state/US: bales}, {state/US: (area, yield, prod)}, sha256, href) or None. Values pass the
    area x yield check by construction."""
    for href, raw in report_texts('crop-production', year, month):
        text = clean(raw.decode('latin-1')).replace('Yield, and, Production', 'Yield, and Production')
        res = parse_report(text)
        if not res or (res[0].year, res[0].month) != (year, month):
            continue
        rel, ay = res
        return rel, {k: v[2] * 1000 for k, v in ay.items()}, ay, hashlib.sha256(raw).hexdigest(), href
    return None


def condition_tx(ref):
    """Latest Texas cotton (vp + p) share from Crop Progress weeks ending on or before ref - 1 day."""
    best = None
    for y, mo in {(ref.year, ref.month), ((ref - dt.timedelta(days=20)).year, (ref - dt.timedelta(days=20)).month),
                  ((ref - dt.timedelta(days=45)).year, (ref - dt.timedelta(days=45)).month)}:
        for href, raw in report_texts('crop-progress', y, mo):
            res = parse_condition(raw.decode('latin-1'))
            if not res or 'TX' not in res[1] or res[0] > ref - dt.timedelta(days=1):
                continue
            if best is None or res[0] > best[0]:
                vp, p = res[1]['TX'][:2]
                best = (res[0], (vp + p) / 100, hashlib.sha256(raw).hexdigest(), href)
    return best


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def build_record(year, month):
    model = json.loads(MODEL.read_text(encoding='utf-8'))
    m = MON[month]
    cur = crop_production(year, month)
    if cur is None:
        return {'status': 'no_report', 'report': f'{year}-{month:02d}'}
    rel, vals, ay, sha, href = cur
    prev = None  # latest earlier report of the season, as in momentum.with_revision (e.g. Oct 2025 was cancelled)
    for pm in range(month - 1, 7, -1):
        prev = crop_production(year, pm)
        if prev is not None:
            break
    ref = dt.date(year, month, 1)
    cond = condition_tx(ref)
    states = {k: v for k, v in vals.items() if k != 'US'}
    gates, reasons = {}, []
    gates['states_sum_us'] = 'US' in vals and abs(sum(states.values()) / vals['US'] - 1) < 0.02
    gates['texas_present'] = 'TX' in vals
    tx_ay = ay.get('TX')
    gates['tx_area_yield'] = bool(tx_ay) and abs(tx_ay[2] * 480 / (tx_ay[0] * tx_ay[1]) - 1) < 0.03
    gates['previous_report'] = prev is not None and 'US' in prev[1] and 'TX' in prev[1]
    gates['condition_fresh'] = cond is not None and (ref - cond[0]).days <= 21
    rec = {'report': f'{year}-{month:02d}', 'release': str(rel), 'source': BASE + href, 'source_sha256': sha,
           'previous_source_sha256': prev[3] if prev else None, 'condition_week_ending': str(cond[0]) if cond else None,
           'condition_source_sha256': cond[2] if cond else None, 'model_version': model['version'],
           'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(), 'code_commit': git_commit(),
           'usda': {'us_production_bales': vals.get('US'), 'tx_production_bales': vals.get('TX'),
                    'tx_harvested_acres': round(tx_ay[0] * 1000) if tx_ay else None}, 'outputs': {}, 'gates': gates}
    # momentum: US via state ratio, Texas directly
    if gates['states_sum_us'] and gates['texas_present'] and gates['previous_report'] and m in model['momentum']:
        b, sd = model['momentum'][m]['b'], model['momentum'][m]['sd']
        corr, base = 0.0, 0.0
        for st, v in states.items():
            pv = prev[1].get(st)
            r = np.log(v / pv) if pv else 0.0
            if abs(r) > 0.5:
                reasons.append(f'revision out of range for {st}')
                r = 0.0
            corr += v * np.exp(b * r / sd)
            base += v
        r_tx = np.log(vals['TX'] / prev[1]['TX'])
        rec['outputs']['us_production_bales'] = {'value': round(vals['US'] * corr / base), 'status': 'model'}
        rec['outputs']['tx_production_bales'] = {'value': round(vals['TX'] * np.exp(b * r_tx / sd)), 'status': 'model'}
        rec['outputs']['next_revision_direction'] = {'US': 'up' if prev[1]['US'] < vals['US'] else 'down' if prev[1]['US'] > vals['US'] else 'no call',
                                                    'TX': 'up' if prev[1]['TX'] < vals['TX'] else 'down' if prev[1]['TX'] > vals['TX'] else 'no call'}
    else:
        reasons.append('momentum gates failed: USDA production passed through')
        rec['outputs']['us_production_bales'] = {'value': vals.get('US'), 'status': 'fallback_usda'}
        rec['outputs']['tx_production_bales'] = {'value': vals.get('TX'), 'status': 'fallback_usda'}
    # anchoring: Texas harvested area
    if gates['tx_area_yield'] and gates['condition_fresh'] and m in model['anchoring_tx_area']:
        a, bb = model['anchoring_tx_area'][m]['a'], model['anchoring_tx_area'][m]['b']
        H = tx_ay[0] * 1000
        eh = a + bb * cond[1]
        q80, q95 = model['intervals_tx_area']['q80_log'], model['intervals_tx_area']['q95_log']
        rec['outputs']['tx_harvested_acres'] = {'value': round(H * np.exp(eh)), 'status': 'model', 'pvp': cond[1],
                                               'interval80': [round(H * np.exp(eh - q80)), round(H * np.exp(eh + q80))],
                                               'interval95': [round(H * np.exp(eh - q95)), round(H * np.exp(eh + q95))]}
    else:
        reasons.append('anchoring gates failed: USDA Texas area passed through')
        rec['outputs']['tx_harvested_acres'] = {'value': round(tx_ay[0] * 1000) if tx_ay else None, 'status': 'fallback_usda'}
    rec['reasons'] = reasons
    rec['status'] = 'ok' if not reasons else 'partial_fallback'
    return rec


def read_log(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def append(path, rec):
    log = read_log(path)
    rec['logged_utc'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rec['prev_hash'] = log[-1]['hash'] if log else '0' * 64
    body = json.dumps({k: v for k, v in rec.items() if k != 'hash'}, sort_keys=True, default=str)
    rec['hash'] = hashlib.sha256(body.encode()).hexdigest()
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + '\n')
    return rec


def verify(path):
    prev = '0' * 64
    for i, rec in enumerate(read_log(path)):
        body = json.dumps({k: v for k, v in rec.items() if k != 'hash'}, sort_keys=True, default=str)
        if rec['prev_hash'] != prev or hashlib.sha256(body.encode()).hexdigest() != rec['hash']:
            return False, i
        prev = rec['hash']
    return True, None


def main(argv):
    log = DEFAULT_LOG
    if '--log' in argv:
        log = Path(argv[argv.index('--log') + 1])
        argv = [a for a in argv if a not in ('--log', str(log))]
    cmd = argv[0] if argv else 'run'
    if cmd == 'verify':
        ok, bad = verify(log)
        print('chain ok' if ok else f'chain broken at record {bad}')
        return 0 if ok else 1
    when = argv[1] if len(argv) > 1 else dt.date.today().strftime('%Y-%m')
    y, mo = map(int, when.split('-'))
    if any(r.get('report') == f'{y}-{mo:02d}' for r in read_log(log)):
        print('already logged', when)
        return 0
    rec = build_record(y, mo)
    if rec.get('status') == 'no_report':
        print('no report yet for', when)
        return 0
    rec = append(log, rec)
    print(json.dumps({k: rec[k] for k in ('report', 'status', 'usda', 'outputs', 'reasons')}, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
