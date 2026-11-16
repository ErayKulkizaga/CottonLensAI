import sys, numpy as np, planetary_computer as pc, pystac_client, rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from PIL import Image
BB=[-102.05,33.45,-101.75,33.70]; out=sys.argv[1]
cat=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',modifier=pc.sign_inplace)
it=[i for i in cat.search(collections=['usda-cdl'],bbox=BB,datetime='2021-01-01/2021-12-31').item_collection() if i.properties.get('usda_cdl:type')=='cropland'][0]
with rasterio.open(it.assets['cropland'].href) as s:
    w=from_bounds(*transform_bounds('EPSG:4326',s.crs,*BB),s.transform); a=s.read(1,window=w)
img=np.full(a.shape+(3,),(232,226,214),np.uint8); img[a==2]=(47,111,79); img[np.isin(a,[121,122,123,124])]=(150,150,150); img[a==111]=(90,130,170)
Image.fromarray(img).resize((640,int(640*a.shape[0]/a.shape[1])),Image.NEAREST).save(out+'/cdl_lubbock_2021.png',optimize=True)
print('cotton share of window',round(float((a==2).mean()),3))
for y in (2021,2022,2023):
    im=Image.open(out+f'/ndvi_lubbock_{y}.png'); im=im.resize((640,int(640*im.height/im.width)),Image.LANCZOS); im.save(out+f'/ndvi_lubbock_{y}.jpg',quality=82,optimize=True)
