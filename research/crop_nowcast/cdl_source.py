"""Where the prior-year CDL cropland raster comes from (T39 declared source change).

Planetary Computer `usda-cdl` covers 2008-2021 only. CDL 2022+ (masks for seasons 2023+) comes from NASS
CropScape state GeoTIFFs of the same product (EPSG:5070, 30 m, class 2 = cotton), read over HTTP range requests.
"""
CROPSCAPE = 'https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_{year}_{st}.tif'
PC_LAST = 2021


def cdl_hrefs(search, bbox, cdl_year, state_fips):
    """search(**kw) -> STAC items (Planetary Computer); state_fips: 2-digit codes of the counties in view."""
    if cdl_year <= PC_LAST:
        items = search(collections=['usda-cdl'], bbox=bbox, datetime=f'{cdl_year}-01-01/{cdl_year}-12-31')
        return [i.assets['cropland'].href for i in items if i.properties.get('usda_cdl:type') == 'cropland']
    return [CROPSCAPE.format(year=cdl_year, st=st) for st in sorted(set(state_fips))]
