from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import calibrate_from_cv


def main() -> None:
    predictions = pd.read_csv(ROOT / "outputs" / "reports" / "cv_predictions.csv")
    calibration = calibrate_from_cv(predictions)
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()

