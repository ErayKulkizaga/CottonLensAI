"""US Drought Monitor weekly county statistics for the panel counties (dev years only).

API: usdmdataservices.unl.edu CountyStatistics, categorical area percent (statisticsType=1).
Released maps are not revised, so a map dated D is usable from its release (D + 2 days).
Writes one CSV row per county x map date with none/d0..d4 percentages.
Usage: python usdm_pull.py panel.json out.csv [first last]
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

SEALED_FROM = 2022
URL = ('https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent'
       '?aoi={st}&startdate=3/1/{y}&enddate=10/31/{y}&statisticsType=1')


def main(panel_json, out_csv, first=2011, last=2021):
    first, last = int(first), int(last)
    if last >= SEALED_FROM:
        raise SystemExit('Sealed years need a pushed preregistration')
    panel = json.loads(Path(panel_json).read_text())['counties']
    keep = {c['fips'] for c in panel}
    states = sorted({c['state'] for c in panel})
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['fips', 'map_date', 'none', 'd0', 'd1', 'd2', 'd3', 'd4'])
        for y in range(first, last + 1):
            for st in states:
                for k in range(5):
                    try:
                        req = urllib.request.Request(URL.format(st=st, y=y), headers={'Accept': 'application/json'})
                        rows = json.loads(urllib.request.urlopen(req, timeout=120).read())
                        break
                    except OSError:
                        if k == 4:
                            raise
                        time.sleep(10 * (k + 1))
                n = 0
                for r in rows:
                    if r['fips'] in keep:
                        w.writerow([r['fips'], r['mapDate'][:10], r['none'], r['d0'], r['d1'], r['d2'], r['d3'], r['d4']])
                        n += 1
                print(y, st, n, flush=True)
                time.sleep(1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
