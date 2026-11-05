"""Read-only, dependency-free audit of the existing pinned FAS country fields."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.sources.fas_country_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('raw-root', 'manifest', 'table'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.raw_root, args.manifest, args.table)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'FAS country audit failed: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
