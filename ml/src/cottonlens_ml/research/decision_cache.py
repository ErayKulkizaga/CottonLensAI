"""Bind derived decisions to their request and actual bounded source frames."""
import pandas as pd

from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.ledger import read_record


def execution_binding(experiment, fold, request, *, selection):
    dates = pd.DatetimeIndex(pd.to_datetime(fold['origins']))
    if (dates.empty or dates.hasnans or dates.has_duplicates
            or not dates.is_monotonic_increasing or not dates.isin(experiment.history.date).all()):
        raise ValueError('Cache origins must be present, unique and chronological')
    cutoff = dates.min() if selection else dates.max()
    frames = {}
    for name in ('history', 'panel'):
        frame = getattr(experiment, name, None)
        if frame is not None:
            past = frame.loc[frame.date.lt(cutoff) if selection else frame.date.le(cutoff)]
            frames[name] = frame_identity(past, list(past.columns))
    return content_id({'schema': 1, 'identity': getattr(experiment, 'identity', {}),
                       'fold': fold, 'request': request, 'frames': frames})


def read_bound(marker, binding):
    result = read_record(marker)
    if result.get('execution_binding') != binding:
        raise ValueError('Cache execution binding missing or changed; preserve old record. '
                         'Inspect historical records with read_record or their frozen source.')
    return result


def read_selection(experiment, marker, fold, namespace, family, horizon):
    """Validate a selection again at consumption, before ranking its score."""
    from cottonlens_ml.research.protocol import validate_inner_fold
    validate_inner_fold(experiment.history, fold)
    result = read_record(marker)
    request = result.get('execution_request', {})
    if (request.get('operation') not in ('tune', 'adaptive', 'adaptive-skip', 'fold-choice')
            or request.get('namespace') != namespace or request.get('family') != family
            or request.get('horizon') != horizon):
        raise ValueError('Selection request missing or changed')
    return read_bound(marker, execution_binding(experiment, fold, request, selection=True))
