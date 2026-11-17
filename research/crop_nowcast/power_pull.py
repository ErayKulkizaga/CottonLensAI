"""Daily NASA POWER weather at each panel county's centroid, development years only.

Archive values are the MERRA-2 reprocessed series (not the real-time GEOS-IT
vintage); this is a documented limitation of historical evaluation.
"""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = 'https://power.larc.nasa.gov/api/temporal/daily/point'
PARAMS = 'T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN'
DEFAULT_YEARS = (2008, 2021)


def main(panel_json, out_dir, first=DEFAULT_YEARS[0], last=DEFAULT_YEARS[1]):
    first, last = int(first), int(last)
    if last >= 2022 and '--unseal' not in sys.argv:
        raise SystemExit('Sealed years need --unseal after preregistration')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = json.loads(Path(panel_json).read_text())['counties']
    receipt = {}
    for i, c in enumerate(panel):
        dest = out / f"{c['fips']}_{first}_{last}.json"
        if dest.exists():
            receipt[c['fips']] = hashlib.sha256(dest.read_bytes()).hexdigest()
            continue
        lon, lat = c['centroid']
        url = (f'{API}?parameters={PARAMS}&community=AG&longitude={lon:.4f}&latitude={lat:.4f}'
               f'&start={first}0101&end={last}1231&format=JSON')
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-client'}), timeout=180) as r:
                    raw = r.read()
                break
            except Exception as exc:               # noqa: BLE001
                if attempt == 4:
                    raise
                print('retry', c['fips'], type(exc).__name__, flush=True)
                time.sleep(10 * (attempt + 1))
        dest.write_bytes(raw)
        receipt[c['fips']] = hashlib.sha256(raw).hexdigest()
        if i % 25 == 0:
            print(i, c['fips'], flush=True)
    (out / f'receipt_{first}_{last}.json').write_text(json.dumps({'api': API, 'params': PARAMS, 'years': [first, last],
                                                                  'files': receipt}, indent=2), encoding='utf-8')
    print('done', len(receipt))


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
