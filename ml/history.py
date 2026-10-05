"""Dependency-free project command: check prior research before another experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.research.history import main

if __name__ == '__main__':
    raise SystemExit(main())
