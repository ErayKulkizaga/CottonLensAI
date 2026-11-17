"""Audit a frozen admission manifest without training or changing input data."""
import argparse
import json
from pathlib import Path

from cottonlens_ml.research.admission import inspect
from cottonlens_ml.sprint import freeze_record, read_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(read_record(args.manifest), args.root)
    freeze_record(args.receipt, result)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
