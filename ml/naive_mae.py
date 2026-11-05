"""Small fixed median research stages; training still uses Experiment/Ledger."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.research.naive_mae import main

if __name__ == '__main__':
    main()
