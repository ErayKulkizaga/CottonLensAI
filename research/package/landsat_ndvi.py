import sys, numpy as np, planetary_computer as pc, pystac_client, rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from PIL import Image
BB=[-102.05,33.45,-101.75,33.70]  # Lubbock County cotton belt, Texas High Plains
cat=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',modifier=pc.sign_inplace)
def cmap(v):
    stops=[(-0.1,(120,98,72)),(0.15,(196,170,128)),(0.3,(214,214,150)),(0.45,(142,190,104)),(0.6,(64,146,72)),(0.8,(18,92,48))]
    out=np.zeros(v.shape+(3,),np.uint8)
    xs=[s[0] for s in stops]
    for c in range(3):
        out[...,c]=np.interp(np.nan_to_num(v,nan=-0.1),xs,[s[1][c] for s in stops]).astype(np.uint8)
    out[np.isnan(v)]=(40,40,40)
    return out
for y,(d0,d1) in {2021:('08-01','09-10'),2022:('08-01','09-10'),2023:('08-01','09-10')}.items():
    items=list(cat.search(collections=['landsat-c2-l2'],bbox=BB,datetime=f'{y}-{d0}/{y}-{d1}',query={'eo:cloud_cover':{'lt':5},'platform':{'in':['landsat-8','landsat-9']}}).item_collection())
    items=[i for i in items if i.properties.get('landsat:collection_category')=='T1']
    best=None
    for it in sorted(items,key=lambda i:i.properties['eo:cloud_cover']):
        with rasterio.open(it.assets['red'].href) as s:
            b=transform_bounds('EPSG:4326',s.crs,*BB); w=from_bounds(*b,s.transform)
            red=s.read(1,window=w,boundless=True,fill_value=0).astype(float)
        if (red==0).mean()<0.01: best=it; break
    with rasterio.open(best.assets['nir08'].href) as s:
        nir=s.read(1,window=w,boundless=True,fill_value=0).astype(float)
    red=red*0.0000275-0.2; nir=nir*0.0000275-0.2
    nd=(nir-red)/(nir+red); nd[(red<=-0.19)|(nir<=-0.19)]=np.nan
    Image.fromarray(cmap(nd)).save(sys.argv[1]+f'/ndvi_lubbock_{y}.png',optimize=True)
    print(y,best.id,best.datetime.date(),nd.shape,'median NDVI',round(float(np.nanmedian(nd)),3),'share<0.3',round(float(np.nanmean(nd<0.3)),3))
