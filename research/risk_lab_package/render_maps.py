"""Render the 2010 vs 2011 late-July NDVI maps over prior-year CDL cotton pixels (PNG)."""
import sys
from pathlib import Path

import numpy as np


def main(sat_dir, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import json

    import matplotlib.pyplot as plt
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from matplotlib.colors import LinearSegmentedColormap
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    sat, out = Path(sat_dir), Path(out_dir)
    meta = json.loads((sat / 'texas_ndvi.json').read_text(encoding='utf-8'))
    bbox = meta['bbox']
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    item = next(i for i in cat.search(collections=['modis-13Q1-061'], bbox=bbox,
                                      datetime='2010-04-01/2010-10-31').item_collection() if '.h09v05.' in i.id)
    with rasterio.open(item.assets['250m_16_days_NDVI'].href) as src:
        to_sin = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        xs, ys = to_sin.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
        wt, crs = src.window_transform(win), src.crs
    H, W = meta['grid']['height'], meta['grid']['width']
    cols, rows = np.meshgrid(np.arange(W + 1), np.arange(H + 1))
    ex, ey = wt * (cols, rows)
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(ex, ey)
    out.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list('ndvi', ['#f2efe2', '#b9d48a', '#5f9d3a', '#1f5a1b'])
    for year in (2010, 2011):
        nd = np.load(sat / f'ndvi_{year}.npy')
        mask = np.load(sat / f'mask_{year}.npy')
        fig = plt.figure(figsize=(4.4, 4.0), dpi=160)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.pcolormesh(lon, lat, np.ma.masked_where(~mask | np.isnan(nd), nd), cmap=cmap, vmin=0.05, vmax=0.75,
                      shading='flat', rasterized=True)
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect(1 / np.cos(np.deg2rad((bbox[1] + bbox[3]) / 2)))
        ax.plot([-101.855], [33.578], marker='o', ms=4, mfc='none', mec='#333333', mew=1)
        fig.savefig(out / f'ndvi_{year}.png', transparent=True)
        plt.close(fig)
        print(year, float(np.nanmedian(nd[mask])))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
