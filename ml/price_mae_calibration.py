"""Project command for fixed past-only calibration of saved Ridge forecasts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.research.price_mae_calibration import main

if __name__ == '__main__':
    main()
