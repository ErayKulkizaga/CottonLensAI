"""Causal same-contract history under the existing unverified quote clock.

Only date and quote records are inputs. Targets and varying second-rank prices
cannot enter a historical endpoint. This module does not fit or select models.
"""
import numpy as np
import pandas as pd

from cottonlens_ml.research.contract_curve import MAX_AGE, align

WINDOWS = (1, 5, 10, 20)
FEATURES = tuple(f'named_contract_ret_{w}' for w in WINDOWS)
POLICY = 'named-contract-history-origin-asof-assumed-clock-v1'


def build(history, records):
    """Return four features and their endpoint provenance, keeping all origins.

For each current origin, select its known second contract, then retrieve THAT
    contract at both reference endpoints using sources available at the CURRENT
    decision. A late historical bar already known now must not be discarded.
    Never shift the price of a changing second-ranked contract.
"""
    calendar = history[['date']].copy()
    aligned = align(calendar, records)
    quotes = {pd.Timestamp(r['report_date']): r for r in records}
    selected = [quotes.get(d) if pd.notna(d) else None for d in aligned.curve_D0_report_date]
    ordered = sorted(quotes)
    references = np.array([d.value for d in ordered], dtype=np.int64)
    available = np.array([
        (max(d, pd.Timestamp(quotes[d]['publication_local_naive']).normalize()).tz_localize('UTC')
         + pd.Timedelta(days=1)).value for d in ordered], dtype=np.int64)
    ref_positions = calendar.date.searchsorted(pd.DatetimeIndex(ordered), side='right')-1
    values = np.full((len(calendar), len(WINDOWS)), np.nan)
    provenance = []
    for i, date in enumerate(calendar.date):
        current = selected[i]
        contract = current['second_contract'] if current is not None else None
        for column, window in enumerate(WINDOWS):
            j = i-window
            past = None
            past_age = None
            past_available = None
            if j >= 0:
                position = int(np.searchsorted(references, calendar.date.iloc[j].value, side='right'))-1
                while position >= 0 and available[position] > aligned.curve_decision_time.iloc[i].value:
                    position -= 1
                if position >= 0 and j-int(ref_positions[position]) <= MAX_AGE:
                    past = quotes[ordered[position]]
                    past_age = j-int(ref_positions[position])
                    past_available = pd.Timestamp(int(available[position]), tz='UTC')
            if current is None:
                reason = 'no_current_source'
            elif j < 0:
                reason = 'insufficient_cotton_history'
            elif past is None:
                reason = 'no_past_source'
            elif contract not in past['quotes']:
                reason = 'contract_not_listed_at_past_source'
            else:
                reason = 'available_under_assumed_clock'
                values[i, column] = np.log(current['quotes'][contract]/past['quotes'][contract])
            current_available = aligned.curve_D0_assumed_available_at.iloc[i] if current is not None else None
            if current is not None:
                assert current_available <= aligned.curve_decision_time.iloc[i]
                assert aligned.curve_D0_age_sessions.iloc[i] <= MAX_AGE
            if past is not None:
                assert past_available <= aligned.curve_decision_time.iloc[i]
                assert 0 <= past_age <= MAX_AGE
            provenance.append({
                'date': date.strftime('%Y-%m-%d'), 'contract': contract, 'window': window,
                'lag_origin': None if j < 0 else calendar.date.iloc[j].strftime('%Y-%m-%d'),
                'current_report_date': None if current is None else current['report_date'],
                'past_report_date': None if past is None else past['report_date'],
                'current_document_sha256': None if current is None else current['document_sha256'],
                'past_document_sha256': None if past is None else past['document_sha256'],
                'current_assumed_available_at': None if current_available is None else current_available.isoformat(),
                'past_assumed_available_at': None if past_available is None else past_available.isoformat(),
                'current_age_sessions': None if current is None else int(aligned.curve_D0_age_sessions.iloc[i]),
                'past_age_sessions': past_age,
                'current_quote': None if current is None else current['quotes'][contract],
                'past_quote': None if past is None else past['quotes'].get(contract),
                'reason': reason,
            })
    if np.isinf(values).any():
        raise ValueError('Finite log-domain quote endpoints required')
    features = pd.DataFrame(values, columns=list(FEATURES), index=history.index)
    features.attrs = {'policy': POLICY, 'historical_availability_verified': False,
                      'first_vintage_verified': False, 'new_model_fits': 0}
    return features, provenance


def mask(features):
    """Zero finite numeric values while retaining identical missingness/width.

The existing train-only Preprocessor adds missingness bits in both arms.
No rows are discarded and no transformation is fitted here.
"""
    if list(features.columns) != list(FEATURES) or np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError('Exact named-history feature schema required')
    result = features.copy()
    result.iloc[:, :] = np.where(features.notna(), 0., np.nan)
    return result
