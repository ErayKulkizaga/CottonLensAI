"""Project command: reconstruct already-completed legacy loss selections without fits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.research.legacy_loss_lineage import main

if __name__ == '__main__':
    main()
