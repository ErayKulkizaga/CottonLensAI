"""Project command for the zero-fit historical selection diagnostic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.research.selection_stability import main

if __name__ == '__main__':
    main()
