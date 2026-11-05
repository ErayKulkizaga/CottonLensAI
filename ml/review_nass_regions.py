"""Read-only audit of Texas condition/progress in existing pinned NASS reports."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.sources.nass_regional import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--audits', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.table, args.audits)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'NASS regional audit failed: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
