"""Synthetic complete UTC windows, fixed footprint and causal latency; no fits."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.weather_exploration import (
    add_weather,
    dispatch,
    group_names,
    regional,
)
from cottonlens_ml.sources.power_weather import (
    END,
    POINTS,
    START,
    UNITS,
    archive,
    compile_weather,
    parse_weather,
)


def rows():
    dates=pd.date_range('2020-01-01','2020-03-10')
    return pd.concat([pd.DataFrame({'date':dates,'point':p,'PRECTOTCORR':1.,'T2M_MAX':36.})
        for p in POINTS],ignore_index=True).sort_values(['date','point'])


def payload():
    keys=pd.date_range('2010-01-01','2023-12-31').strftime('%Y%m%d')
    return {'type':'Feature','geometry':{'type':'Point','coordinates':[-101.9,33.6,1000]},
        'properties':{'parameter':{'PRECTOTCORR':dict.fromkeys(keys,1.),'T2M_MAX':dict.fromkeys(keys,36.)}},
        'header':{'start':START,'end':END,'time_standard':'UTC','fill_value':-999.,
            'api':{'version':'fixture'},'sources':['MERRA2','POWER']},
        'parameters':{k:{'units':v} for k,v in UNITS.items()}}


def test_exact_parameter_units_utc_coordinates_and_fill_value():
    b=payload();b['properties']['parameter']['PRECTOTCORR']['20100101']=-999.
    f,a=parse_weather(json.dumps(b).encode(),'lubbock')
    assert len(f)==5113 and pd.isna(f.PRECTOTCORR.iloc[0]) and a['missing_values']['PRECTOTCORR']==1
    b['header']['time_standard']='LST'
    with pytest.raises(ValueError,match='identity'):parse_weather(json.dumps(b).encode(),'lubbock')
    b=payload();b['parameters']['T2M_MAX']['units']='F'
    with pytest.raises(ValueError,match='units'):parse_weather(json.dumps(b).encode(),'lubbock')
    with pytest.raises(ValueError,match='identity'):parse_weather(json.dumps(payload()).encode(),'hale')


def test_missing_calendar_wrong_domains_and_oversized_archive(tmp_path):
    b=payload();b['properties']['parameter']['T2M_MAX'].pop('20100101')
    with pytest.raises(ValueError,match='calendar'):parse_weather(json.dumps(b).encode(),'lubbock')
    b=payload();b['properties']['parameter']['PRECTOTCORR']['20100101']=-1.
    with pytest.raises(ValueError,match='Negative'):parse_weather(json.dumps(b).encode(),'lubbock')
    b=payload();b['properties']['parameter']['T2M_MAX']['20100101']=150.
    with pytest.raises(ValueError,match='Celsius'):parse_weather(json.dumps(b).encode(),'lubbock')
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,*args):yield b'X'*(8*1024*1024+1)
    session=SimpleNamespace(get=lambda *a,**k:Response())
    with pytest.raises(ValueError,match='MiB'):archive('lubbock',tmp_path,session=session)
    assert not list(tmp_path.iterdir())


def test_equal_fixed_footprint_no_available_point_average_or_compressed_days():
    f=rows();out=regional(f)
    assert out.rain_30.iloc[29]==30 and out.heat_excess_7.iloc[6]==7
    changed=f.copy();changed.loc[(changed.point=='hale')&(changed.date=='2020-02-01'),'PRECTOTCORR']=np.nan
    other=regional(changed)
    assert other.loc[other.date=='2020-02-01','rain_7'].isna().all()
    assert other.loc[other.date=='2020-02-29','rain_30'].isna().all()
    assert other.loc[other.date=='2020-03-02','rain_30'].iloc[0]==30
    with pytest.raises(ValueError,match='calendar'):regional(f.drop(f.index[5]))
    with pytest.raises(ValueError):regional(f.iloc[::-1])


def test_three_calendar_day_latency_plus_observation_stresses_and_future_invariance():
    f=rows();h=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-03-20')})
    out=add_weather(h,f)
    # First full 30-day summary is Jan30; assumed availability Feb2, first Cotton Feb3.
    assert out.loc[out.date<'2020-02-03','weather_rain_30_L1'].isna().all()
    for lag,day in ((1,'2020-02-03'),(2,'2020-02-04'),(6,'2020-02-10')):
        assert out.loc[out.date==day,f'weather_rain_30_L{lag}'].iloc[0]==30
    changed=f.copy();changed.loc[changed.date>='2020-02-10','T2M_MAX']=45.
    other=add_weather(h,changed)
    pd.testing.assert_frame_equal(out.loc[out.date<='2020-02-13'],other.loc[other.date<='2020-02-13'])
    assert len(out)==len(h) and out.weather_growing_season_L1.eq(0).all()


def test_missing_daily_sample_does_not_refresh_age_or_erase_origins():
    f=rows();f.loc[f.date>='2020-02-01','T2M_MAX']=np.nan
    h=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-03-20')})
    out=add_weather(h,f)
    assert len(out)==len(h)
    assert out.loc[out.date=='2020-02-04','weather_age_L1'].iloc[0]==0
    assert out.loc[out.date=='2020-02-10','weather_rain_30_L1'].isna().all()
    assert out.loc[out.date=='2020-02-10','weather_age_L1'].iloc[0]==4


def test_matching_season_missing_controls_and_release_block(tmp_path):
    g=group_names()
    for lag in (1,2,6):
        assert g[f'weather_L{lag}'][:-8]==g[f'missing_L{lag}']
        assert len(g[f'weather_L{lag}'])==35
        assert f'weather_growing_season_L{lag}' in g[f'missing_L{lag}']
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())


def test_weather_compile_resume_preserves_exact_table_and_manifest(tmp_path):
    archives = {}
    for point, (latitude, longitude) in POINTS.items():
        data = payload()
        data['geometry']['coordinates'] = [longitude, latitude, 1000]
        raw = json.dumps(data).encode()

        class Response:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_content(self, *args, content=raw):
                yield content

        session = SimpleNamespace(get=lambda *a, **k: Response())
        archives[point] = archive(point, tmp_path / 'raw', session=session)
    output = tmp_path / 'weather.csv'
    first = compile_weather(archives, output)
    files = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert compile_weather(archives, output) == first
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == files
    archived = archives['lubbock'] / 'weather.json'
    archived.write_bytes(archived.read_bytes() + b' ')
    with pytest.raises(ValueError, match='Pinned POWER'):
        compile_weather(archives, output)
    assert output.read_bytes() == files[output]
    manifest = output.with_suffix('.manifest.json')
    assert manifest.read_bytes() == files[manifest]
