"""Offline cotton narrative corpus; report dates never certify vintage availability."""
import hashlib
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.public import validate_url
from cottonlens_ml.sources.wasde import ReleasePage

RELEASE = re.compile(
    r'https://esmis\.nal\.usda\.gov/publication/'
    r'world-agricultural-supply-and-demand-estimates/(\d{4}-\d{2}-\d{2})(?:-\d+)?$')
HEADER = re.compile(r'^\s*WASDE\s*-\s*\d+\s*-\s*\d+\s*$', re.MULTILINE)
DATE = re.compile(r'\b([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b')
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December')


def cotton_narrative(pages, report_day):
    """Join a paragraph split across pages; reject tables or unbounded extraction."""
    if not pages or len(pages) > 8 or any(not isinstance(p, str) for p in pages):
        raise ValueError('One to eight initial PDF text pages required')
    day = date.fromisoformat(report_day)
    cover_date = f'{MONTHS[day.month-1]} {day.day}, {day.year}'
    dates = [' '.join(value.split()) for value in DATE.findall(pages[0][:1500])]
    if dates != [cover_date]:
        raise ValueError('PDF cover date differs from official release page; no inferred date')
    text = '\n'.join(HEADER.sub('', p) for p in pages)
    starts = list(re.finditer(r'\bCOTTON\s*:', text))
    if len(starts) != 1:
        raise ValueError('One explicit cotton narrative heading required')
    body = text[starts[0].end():]
    end = re.search(r'\b(?:Approved by the (?:Acting )?Secretary|APPROVED BY\s*:)', body)
    if not end:
        raise ValueError('Narrative ending not found; tables must not enter corpus')
    body = unicodedata.normalize('NFKC', ' '.join(body[:end.start()].split()))
    words = re.findall(r'\b[a-zA-Z]+\b', body)
    if not 40 <= len(words) <= 2000 or len(body) > 16000:
        raise ValueError('Cotton narrative size outside reviewed bounds')
    return body


def compile_corpus(archive_root, output, *, extract_pages, extractor_identity):
    """Compile existing checksummed PDFs, recording missing/ambiguous reports explicitly.

    The extractor is an offline callable supplied by the data workbench/tooling.
    No PDF, NLP or training dependency is introduced into backend or ML runtime.
    The signed corpus is Tier A and never constitutes point-in-time admission.
    """
    root, output = Path(archive_root), Path(output)
    if not isinstance(extractor_identity, str) or not extractor_identity.strip():
        raise ValueError('Explicit extractor version identity required')
    index = {}
    alias_hashes = {}
    for path in sorted((root/'wasde').glob('*/retrieval.json')):
        receipt = read_record(path)
        if receipt['kind'] != 'wasde':
            raise ValueError('Non-WASDE receipt inside WASDE archive')
        validate_url('wasde', receipt['source_url'])
        index.setdefault(receipt['source_url'], []).append((path, receipt))
    for path in sorted((root/'wasde_url_aliases').glob('*/alias.json')):
        alias = read_record(path)
        validate_url('wasde', alias['source_url'])
        validate_url('wasde', alias['canonical_source_url'])
        if (not re.fullmatch('[a-f0-9]{64}', alias['sha256'])
                or path.parent.name != hashlib.sha256(alias['source_url'].encode()).hexdigest()):
            raise RuntimeError('Malformed WASDE alias identity')
        receipt_path = root/'wasde'/alias['sha256']/'retrieval.json'
        receipt = read_record(receipt_path)
        if (receipt['kind'] != 'wasde' or receipt['source_url'] != alias['canonical_source_url']
                or receipt['sha256'] != alias['sha256']
                or digest(receipt_path.parent/'source.bin') != alias['sha256']):
            raise RuntimeError('WASDE URL alias provenance/checksum mismatch')
        values = index.setdefault(alias['source_url'], [])
        if all(existing_path != receipt_path for existing_path, _ in values):
            values.append((receipt_path, receipt))
        alias_hashes[alias['source_url']] = digest(path)

    def one(url):
        versions = index.get(url, [])
        if len(versions) != 1:
            raise ValueError('Missing or ambiguous archived source version')
        receipt_path, receipt = versions[0]
        source = receipt_path.parent/'source.bin'
        if digest(source) != receipt['sha256']:
            # Byte corruption is fatal; it is never silently classified as missing.
            raise RuntimeError('WASDE archive checksum mismatch')
        return source, receipt, digest(receipt_path)

    records, exclusions = [], []
    for url in sorted(index):
        match = RELEASE.fullmatch(url)
        if not match or not '2016-01-01' <= match[1] < '2024-01-01':
            continue
        day = match[1]
        try:
            page_source, page_receipt, page_receipt_hash = one(url)
            page = ReleasePage()
            page.feed(page_source.read_text(encoding='utf-8-sig'))
            links = {urljoin(url, link) for link in page.links if link.endswith('.pdf')}
            if len(links) != 1:
                raise ValueError('One PDF link required')
            pdf_url = links.pop()
            validate_url('wasde', pdf_url)
            pdf, receipt, receipt_hash = one(pdf_url)
            pages = extract_pages(pdf)
            narrative = cotton_narrative(pages, day)
        except ValueError as exc:
            exclusions.append({'release_url': url, 'report_day': day, 'reason': str(exc)})
            continue
        records.append({'report_day': day, 'release_url': url, 'page_sha256': page_receipt['sha256'],
            'page_receipt_sha256': page_receipt_hash, 'pdf_url': pdf_url, 'pdf_sha256': receipt['sha256'],
            'page_alias_sha256': alias_hashes.get(url), 'pdf_alias_sha256': alias_hashes.get(pdf_url),
            'pdf_receipt_sha256': receipt_hash, 'retrieved_at': receipt['retrieved_at'],
            'narrative': narrative, 'narrative_id': content_id(narrative),
            'word_count': len(re.findall(r'\b[a-zA-Z]+\b', narrative)),
            'replacement_character_count': narrative.count('\ufffd')})

    # Conflicting dated revisions remain excluded rather than choosing a later version.
    counts = Counter(record['report_day'] for record in records)
    conflicting = {day for day, count in counts.items() if count > 1 and
        len({r['narrative_id'] for r in records if r['report_day'] == day}) > 1}
    for record in records:
        if record['report_day'] in conflicting:
            exclusions.append({'release_url': record['release_url'], 'report_day': record['report_day'],
                               'reason': 'Conflicting same-day narratives; revision review required'})
    unique = {}
    for record in records:
        if record['report_day'] not in conflicting:
            unique.setdefault(record['report_day'], record)
    body = {'kind': 'wasde-cotton-narrative-corpus-v1', 'source_tier': 'A_exploration_only',
        'model_eligible': False, 'release_allowed': False, 'publication_timestamp_verified': False,
        'first_version_verified': False, 'cost_tl': 0, 'extractor_identity': extractor_identity,
        'parser_sha256': digest(Path(__file__)), 'records': list(unique.values()), 'exclusions': exclusions,
        'scope': '2016-2023; cotton narrative only, existing archives; no text model fitted',
        'date_semantics': 'Report cover date, not first publication or availability timestamp',
        'duplicate_policy': 'Identical same-day narratives collapse; conflicting narratives excluded',
        'redistribution_reviewed': False,
        'unique_report_count': len(unique), 'unique_narrative_count': len({r['narrative_id'] for r in unique.values()}),
        'year_counts': dict(sorted(Counter(day[:4] for day in unique).items()))}
    freeze_record(output, body)
    return body


def align_corpus(history, corpus_path, *, lag=1, max_age=45):
    """Causal *assumed-date* alignment for Tier-A discovery; not publication admission.

    Lag and age use the complete recorded Cotton calendar, never a feature-filtered
    calendar. Stale/missing text does not remove an origin or become numeric zero.
    A future training adapter must fit its vocabulary/decomposition on training
    documents only; this function performs no learned transformation.
    """
    if lag not in (1, 2, 6) or not isinstance(max_age, int) or not lag <= max_age <= 126:
        raise ValueError('Registered lag1/+1/+5 stress and bounded age required')
    corpus = read_record(Path(corpus_path))
    if (corpus['kind'] != 'wasde-cotton-narrative-corpus-v1'
            or corpus['source_tier'] != 'A_exploration_only' or corpus['model_eligible']
            or corpus['release_allowed'] or corpus['publication_timestamp_verified']):
        raise ValueError('Explicit unverified Tier-A corpus required')
    dates = pd.DatetimeIndex(history.date)
    if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Complete unique chronological daily Cotton observations required')
    records = corpus['records']
    report_dates = pd.DatetimeIndex([record['report_day'] for record in records])
    if (report_dates.has_duplicates or not report_dates.is_monotonic_increasing
            or any(day >= pd.Timestamp('2024-01-01') for day in report_dates)
            or any(content_id(record['narrative']) != record['narrative_id'] for record in records)):
        raise ValueError('Corrupt, duplicate, unsorted or audit narrative inputs')
    after = np.searchsorted(dates.to_numpy(), report_dates.to_numpy(), side='right')
    starts = after + lag - 1
    selected = np.searchsorted(starts, np.arange(len(dates)), side='right') - 1
    text, ids, report_days = [], [], []
    ages = np.full(len(dates), np.nan)
    for ordinal, report in enumerate(selected):
        age = ordinal-after[report]+1 if report >= 0 else np.nan
        valid = report >= 0 and age <= max_age
        record = records[report] if valid else None
        text.append(record['narrative'] if record else None)
        ids.append(record['narrative_id'] if record else None)
        report_days.append(record['report_day'] if record else None)
        if report >= 0:
            ages[ordinal] = age
    result = history.copy()
    result['wasde_narrative'], result['wasde_narrative_id'] = text, ids
    result['wasde_assumed_report_day'] = report_days
    result['wasde_text_age'], result['wasde_text_missing'] = ages, pd.isna(text).astype(float)
    result.attrs['wasde_text_policy'] = 'Tier A; lagged report date assumption; no publication/vintage admission'
    return result
