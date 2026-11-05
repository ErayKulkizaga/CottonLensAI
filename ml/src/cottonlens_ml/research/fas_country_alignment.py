"""Matched source groups on the existing publication join; no fitting or admission."""
import pandas as pd

from cottonlens_ml.research.publications import (
    attach_package,
    availability_features,
    load_package,
)
from cottonlens_ml.sources.fas_admission import DECISION_CLOCK


def attach_matched(history, package, *, national_features, country_features):
    """Keep every supplied Cotton origin/target; expose identical source masks.

    Returns a history plus source-only group names. A registered experiment must
    explicitly add its frozen Cotton features; this helper never selects models.
    """
    if (history.empty or not pd.api.types.is_datetime64_any_dtype(history.date)
            or history.date.dt.tz is not None or not history.date.eq(history.date.dt.normalize()).all()):
        raise ValueError('FAS history requires nonempty naive-midnight Cotton session labels')
    manifest, rows, features = load_package(package)
    national, country = list(national_features), list(country_features)
    if (manifest['kind'] != 'export_sales' or manifest.get('decision_clock') != DECISION_CLOCK
            or not national or not country or len(set(national + country)) != len(national + country)
            or set(national + country) != set(features)
            or any(not f.startswith('export_sales_national_') for f in national)
            or any(not f.startswith('export_sales_country_') for f in country)):
        raise ValueError('FAS matched national/country partition and explicit decision clock required')
    if type(manifest.get('max_age_days')) is not int or not 1 <= manifest['max_age_days'] <= 366:
        raise ValueError('FAS explicit bounded freshness policy required')
    masks = [name + '_missing' for name in features]
    common = [*availability_features(manifest), *masks]
    if set(common) & (set(history.columns) | set(features)):
        raise ValueError('FAS shared channel collides with an existing history/feature column')
    result = attach_package(history, manifest, rows, features, include_provenance=True)
    for name, mask in zip(features, masks):
        result[mask] = result[name].isna().astype(float)
    if len(result) != len(history) or not result[history.columns].equals(history):
        raise ValueError('FAS alignment changed original origins/targets')
    return result, {'national_control': [*national, *common],
                    'country_candidate': [*national, *country, *common]}
