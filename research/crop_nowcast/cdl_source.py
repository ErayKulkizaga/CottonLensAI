"""Where the prior-year CDL cropland raster comes from (T39 declared source change).

Planetary Computer `usda-cdl` covers 2008-2021 only. CDL 2022+ (masks for seasons 2023+) comes from NASS
CropScape state GeoTIFFs of the same product (EPSG:5070, 30 m, class 2 = cotton). Remote strip reads over
HTTP failed intermittently on Kaggle, so each state file is downloaded whole (size-checked, retried) and
read from local disk.
"""
import os
import tempfile
import time
import urllib.request
from pathlib import Path

CROPSCAPE = 'https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_{year}_{st}.tif'
PC_LAST = 2021
CACHE = Path(os.environ.get('CDL_CACHE', Path(tempfile.gettempdir()) / 'cdl_cache'))


def fetch(url, tries=8):
    dest = CACHE / url.rsplit('/', 1)[1]
    if dest.exists():
        return str(dest)
    CACHE.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            part = dest.with_suffix('.part')
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-client'}), timeout=600) as r:
                size = int(r.headers.get('Content-Length', 0))
                with open(part, 'wb') as fh:
                    while chunk := r.read(1 << 22):
                        fh.write(chunk)
            if size and part.stat().st_size != size:
                raise OSError(f'short download {part.stat().st_size} of {size}')
            part.replace(dest)
            return str(dest)
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(20 * (k + 1))
    return None


def cdl_hrefs(search, bbox, cdl_year, state_fips):
    """search(**kw) -> STAC items (Planetary Computer); state_fips: 2-digit codes of the counties in view."""
    if cdl_year <= PC_LAST:
        items = search(collections=['usda-cdl'], bbox=bbox, datetime=f'{cdl_year}-01-01/{cdl_year}-12-31')
        return [i.assets['cropland'].href for i in items if i.properties.get('usda_cdl:type') == 'cropland']
    return [fetch(CROPSCAPE.format(year=cdl_year, st=st)) for st in sorted(set(state_fips))]
