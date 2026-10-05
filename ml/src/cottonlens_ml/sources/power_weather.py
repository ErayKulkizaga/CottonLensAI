"""Three fixed Southern High Plains weather samples; vintage-unverified POWER."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record

API='https://power.larc.nasa.gov/api/temporal/daily/point'
POINTS={'lubbock':(33.6,-101.9),'hale':(34.2,-101.9),'gaines':(32.75,-102.65)}
UNITS={'PRECTOTCORR':'mm/day','T2M_MAX':'C'}
START,END='20100101','20231231'


def query(point):
    lat,lon=POINTS[point]
    return {'parameters':','.join(UNITS),'community':'AG','latitude':lat,'longitude':lon,
            'start':START,'end':END,'format':'JSON','time-standard':'UTC'}


def parse_weather(raw,point):
    if point not in POINTS:raise ValueError('Fixed preregistered weather point required')
    try:
        body=json.loads(raw);header=body['header'];params=body['properties']['parameter']
        lat,lon=POINTS[point];coordinate=body['geometry']['coordinates']
        if (body['type']!='Feature' or body['geometry']['type']!='Point'
                or not np.allclose(coordinate[:2],(lon,lat),atol=1e-6,rtol=0)
                or header['time_standard']!='UTC' or header['start']!=START or header['end']!=END
                or set(params)!=set(UNITS)):
            raise ValueError('POWER point/time/range/parameter identity changed')
        dates=pd.date_range('2010-01-01','2023-12-31');keys=set(dates.strftime('%Y%m%d'))
        columns={}
        for name,unit in UNITS.items():
            if body['parameters'][name]['units']!=unit or set(params[name])!=keys:
                raise ValueError('Complete daily UTC calendar and exact units required')
            values=np.array([params[name][d] for d in dates.strftime('%Y%m%d')],dtype=float)
            values[values==float(header['fill_value'])]=np.nan
            if np.isinf(values).any():raise ValueError('Invalid weather numeric domain')
            if name=='PRECTOTCORR' and (values[np.isfinite(values)]<0).any():
                raise ValueError('Negative precipitation is not a zero-rain observation')
            if name=='T2M_MAX' and ((values[np.isfinite(values)] < -90) | (values[np.isfinite(values)]>65)).any():
                raise ValueError('Temperature outside Celsius sanity bounds')
            columns[name]=values
    except (KeyError,TypeError,OverflowError,json.JSONDecodeError) as exc:
        raise ValueError('Malformed POWER daily response') from exc
    frame=pd.DataFrame({'date':dates,'point':point,**columns})
    summary={'point':point,'rows':len(frame),'coordinates':coordinate,
        'units':UNITS,'missing_values':{k:int(np.isnan(v).sum()) for k,v in columns.items()},
        'api_version':header['api']['version'],'sources':header['sources'],
        'time_standard':'UTC','fill_value':header['fill_value'],
        'publication_timestamp_verified':False,'first_version_verified':False}
    return frame,summary


def archive(point,root,*,session=None):
    if point not in POINTS:raise ValueError('Unknown weather point')
    client=session or requests.Session();started=datetime.now(UTC).isoformat()
    with client.get(API,params=query(point),timeout=(10,45),stream=True,allow_redirects=False) as response:
        if response.status_code!=200:raise RuntimeError(f'POWER HTTP {response.status_code}; bounded request stopped')
        pieces=[];size=0
        for piece in response.iter_content(65536):
            size+=len(piece)
            if size>8*1024*1024:raise ValueError('POWER point exceeds 8 MiB')
            pieces.append(piece)
        raw=b''.join(pieces)
    _,summary=parse_weather(raw,point);sha=hashlib.sha256(raw).hexdigest()
    folder=Path(root)/point/sha;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'weather.json'
    if path.exists():
        if digest(path)!=sha:raise ValueError('Corrupt weather archive preserved')
    else:
        with path.open('xb') as handle:handle.write(raw)
    receipt=folder/'retrieval.json'
    if not receipt.exists():
        freeze_record(receipt,{'source_url':API,'query':query(point),'sha256':sha,'summary':summary,
            'request_started_at':started,'retrieved_at':datetime.now(UTC).isoformat(),
            'source_tier':'A_exploration_only','model_eligible':False,'release_allowed':False,
            'publication_timestamp_verified':False,'first_version_verified':False,'cost_tl':0,
            'methodology_url':'https://power.larc.nasa.gov/docs/faqs/data/',
            'daily_api_url':'https://power.larc.nasa.gov/docs/services/api/temporal/daily/',
            'interpretation':'source-native gridded assimilation, not three ground stations or Texas crop weights'})
    return folder


def compile_weather(archives,output):
    if set(archives)!=set(POINTS):raise ValueError('All three fixed weather points required')
    frames=[];sources={}
    for point,folder in archives.items():
        folder=Path(folder);path=folder/'weather.json';receipt=folder/'retrieval.json';record=read_record(receipt)
        if (digest(path)!=record['sha256'] or record['query']!=query(point) or record['source_url']!=API
                or record['model_eligible'] or record['release_allowed']
                or record['publication_timestamp_verified'] or record['first_version_verified']):
            raise ValueError('Pinned POWER scope/availability changed')
        frame,summary=parse_weather(path.read_bytes(),point)
        if summary!=record['summary']:raise ValueError('Pinned POWER audit changed')
        frames.append(frame);sources[point]={'source_sha256':digest(path),'receipt_sha256':digest(receipt),
            'source_url':API,'summary':summary}
    frame=pd.concat(frames,ignore_index=True).sort_values(['date','point'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    raw=frame.to_csv(index=False,date_format='%Y-%m-%d').encode()
    if output.exists():
        if output.read_bytes()!=raw:raise ValueError('Conflicting weather table preserved')
    else:output.write_bytes(raw)
    # Use JSON-native coordinates so a frozen manifest compares identically on resume.
    manifest={'table_sha256':digest(output),'rows':len(frame),'sources':sources,
        'points':{point:list(coordinates) for point,coordinates in POINTS.items()},
        'source_tier':'A_exploration_only','model_eligible':False,'release_allowed':False,
        'publication_timestamp_verified':False,'first_version_verified':False,'cost_tl':0,
        'scope':'equal-weight Southern High Plains sample; not production-weighted statewide/global weather',
        'latency':'3 calendar days assumed after UTC daily aggregate, plus Cotton lag1/2/6; not vintage proof'}
    freeze_record(output.with_suffix('.manifest.json'),manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    archives={p:archive(p,args.output/'raw') for p in POINTS}
    print(json.dumps(compile_weather(archives,args.output/'weather-2010-2023.csv')))


if __name__=='__main__':main()
