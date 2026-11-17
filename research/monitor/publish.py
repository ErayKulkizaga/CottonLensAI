"""Publish the monitor's public performance history from the hash-chained log.

  python research/monitor/publish.py [--log PATH] [--out DIR]

Writes:
  DIR/latest.json  the latest record (machine-readable; a static stand-in for an API)
  DIR/index.html   every logged report: USDA vs the monitor, gates, model version, and errors once finals exist
Finals come from research/monitor/finals.json, filled when NASS publishes them (May of the next year).
"""
import datetime as dt
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.monitor.monitor import DEFAULT_LOG, read_log

FINALS = ROOT / 'research/monitor/finals.json'
OUT = ROOT / 'docs/monitor'
KEYS = (('us_production_bales', 'US production, bales'), ('tx_production_bales', 'Texas production, bales'),
        ('tx_harvested_acres', 'Texas harvested area, acres'))


def latest(rec):
    keep = ('report', 'release', 'status', 'model_version', 'usda', 'outputs', 'reasons', 'logged_utc', 'hash', 'source')
    return {k: rec.get(k) for k in keep}


def err(v, final):
    return f'{100 * (v / final - 1):+.1f}%' if v and final else '—'


def num(v):
    return f'{v:,.0f}' if isinstance(v, (int, float)) else '—'


def rows_html(recs, finals):
    out = []
    for r in reversed(recs):
        fin = finals.get(r['report'][:4], {})
        for k, label in KEYS:
            o = r['outputs'].get(k, {})
            u = r['usda'].get(k)
            iv = o.get('interval80')
            out.append('<tr>' + ''.join(f'<td>{c}</td>' for c in (
                html.escape(r['report']), label, num(u), num(o.get('value')),
                f'{num(iv[0])} – {num(iv[1])}' if iv else '—', html.escape(o.get('status', '—')),
                num(fin.get(k)), err(u, fin.get(k)), err(o.get('value'), fin.get(k)))) + '</tr>')
    return '\n'.join(out)


def page(recs, finals):
    last = recs[-1] if recs else None
    dirs = (last or {}).get('outputs', {}).get('next_revision_direction', {})
    meta = (f"Latest: <b>{html.escape(last['report'])}</b> report (released {html.escape(last['release'])}), status "
            f"<b>{html.escape(last['status'])}</b>, model {html.escape(last['model_version'])}. Next revision called: "
            f"US {html.escape(dirs.get('US', '—'))}, Texas {html.escape(dirs.get('TX', '—'))}.") if last else 'No report logged yet.'
    chain = ' · '.join(f"{html.escape(r['report'])} <code>{r['hash'][:12]}</code>" for r in recs)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>USDA Forecast Monitor</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--rule:#e2dfd6;--soft:#f2efe8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#ecebe6;--muted:#a3a29c;--rule:#33333a;--soft:#202027}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1040px;margin:0 auto}} h1{{font-size:24px;margin:0 0 6px}} p{{margin:8px 0}} .muted{{color:var(--muted)}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}}
th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}} th{{background:var(--soft);font-weight:600}} code{{font-size:12px}}
</style></head><body><main>
<h1>USDA cotton forecast monitor</h1>
<p class="muted">Each USDA Crop Production report (Aug–Dec) is read from the public text, checked by gates, corrected by a frozen
model, and appended to a hash-chained log. A failed gate passes USDA's own number through. Errors appear when NASS publishes
the final (May of the next year). Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>
<p>{meta}</p>
<div class="tbl"><table>
<thead><tr><th>Report</th><th>Quantity</th><th>USDA</th><th>Monitor</th><th>80% interval</th><th>Status</th><th>Final</th>
<th>USDA error</th><th>Monitor error</th></tr></thead>
<tbody>
{rows_html(recs, finals)}
</tbody></table></div>
<p class="muted">Texas area intervals: model_v1 records use hindsight quantiles; from model_v2 on, intervals are calibrated in
real time (CAL1: 83% / 93% coverage of the 80% / 95% interval, 2011–2025) and wide, about ±16% at 80%.
Machine-readable latest record: <a href="latest.json">latest.json</a>.</p>
<p class="muted">Hash chain: {chain or '—'}</p>
</main></body></html>
'''


def main(argv):
    log = Path(argv[argv.index('--log') + 1]) if '--log' in argv else DEFAULT_LOG
    out = Path(argv[argv.index('--out') + 1]) if '--out' in argv else OUT
    recs = read_log(log)
    finals = json.loads(FINALS.read_text(encoding='utf-8')) if FINALS.exists() else {}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'latest.json').write_text(json.dumps(latest(recs[-1]) if recs else {}, indent=1, default=str), encoding='utf-8')
    (out / 'index.html').write_text(page(recs, finals), encoding='utf-8')
    print('published', len(recs), 'records to', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
