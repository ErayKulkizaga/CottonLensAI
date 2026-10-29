"""Strict bounded extraction of dated named settlements, never a training feed.

Observation date, printed publication date and current retrieval are different
clocks. This parser establishes neither first vintage nor daily availability.
PDF loading belongs to the optional document runtime, outside ML dependencies.
"""
import argparse
import calendar
import json
import re
from datetime import date, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

MONTHS = {name: i for i, name in enumerate(calendar.month_abbr) if name}
CONTRACT = re.compile(r"(M\s*a\s*r|M\s*a\s*y|J\s*u\s*l|O\s*c\s*t|D\s*e\s*c)\s*-\s*(\d{2})")
ROW = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s-]+(\d{1,2})\s+(.+)$")


def printed_date(first_page):
    pattern = r"\b(" + "|".join(calendar.month_name[1:]) + r")\s+(\d{1,2}),\s*(20\d{2})\b"
    found = re.search(pattern, first_page)
    if not found:
        raise ValueError("missing printed publication date")
    month, day, year = found.groups()
    return date(int(year), list(calendar.month_name).index(month), int(day))


def parse_table(page_text, publication):
    compact = re.sub(r"\s+", "", page_text).lower()
    if "icefuturescontractsettlement" not in compact:
        raise ValueError("missing settlement caption")
    matches = list(CONTRACT.finditer(page_text))
    runs = []
    for i in range(len(matches) - 5):
        candidate = matches[i:i + 6]
        if all(not page_text[a.end():b.start()].strip()
               for a, b in pairwise(candidate)):
            runs.append(candidate)
    if len(runs) != 1:
        raise ValueError("ambiguous six-contract header")
    header = runs[0]
    contracts = [f"20{m[2]}-{MONTHS[re.sub(r'\s+', '', m[1])]:02d}" for m in header]
    if len(set(contracts)) != 6 or contracts != sorted(contracts):
        raise ValueError("duplicate or unordered delivery months")
    rows = []
    started = False
    for line in page_text[header[-1].end():].splitlines():
        match = ROW.fullmatch(line.strip())
        if not match:
            if started:
                break
            continue
        started = True
        month, day, numbers = match.groups()
        choices = []
        for year in (publication.year - 1, publication.year):
            try:
                observed = date(year, MONTHS[month], int(day))
            except ValueError:
                continue
            if timedelta(0) < publication - observed <= timedelta(days=8):
                choices.append(observed)
        if len(choices) != 1:
            raise ValueError("observation outside strictly prior publication week")
        observed = choices[0]
        tokens = numbers.split()
        if len(tokens) not in (7, 8) or any(not re.fullmatch(r"\d+\.\d{2}", t) for t in tokens):
            raise ValueError("missing or malformed settlement cells; no numerical fallback")
        if any(Decimal(t) <= 0 for t in tokens[:6]):
            raise ValueError("nonpositive settlement")
        if any(contract < observed.strftime('%Y-%m') for contract in contracts):
            raise ValueError("expired delivery month")
        rows.append({"observation_date": observed.isoformat(),
                     "printed_publication_date": publication.isoformat(),
                     "publication_lag_calendar_days": (publication - observed).days,
                     "settlements": dict(zip(contracts, tokens[:6]))})
    dates = [row['observation_date'] for row in rows]
    if not 3 <= len(rows) <= 5 or dates != sorted(set(dates)):
        raise ValueError("invalid unique ordered daily cohort")
    return rows


def extract_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [p.extract_text() or '' for p in reader.pages]
    publication = printed_date(pages[0])
    tables = []
    for i, text in enumerate(pages):
        if "icefuturescontractsettlement" in re.sub(r"\s+", "", text).lower():
            tables.append((i + 1, parse_table(text, publication)))
    if len(tables) != 1:
        raise ValueError("missing or ambiguous settlement page")
    page, rows = tables[0]
    return {"printed_publication_date": publication.isoformat(), "page": page,
            "unit": "cents_per_pound", "field": "ICE_futures_settlement",
            "first_vintage_verified": False, "exact_publication_utc_verified": False,
            "daily_availability_verified": False, "training_admitted": False,
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = {p.name: extract_pdf(p) for p in sorted(args.root.glob('*.pdf'))}
    output = args.root / 'extracted-tables.json'
    if output.exists():
        raise ValueError('immutable result already exists')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pdfs': len(result), 'daily_rows': sum(len(v['rows']) for v in result.values()),
                      'new_fits': 0, 'training_admitted': False}))


if __name__ == '__main__':
    main()
