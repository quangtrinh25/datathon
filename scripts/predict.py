from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import ForecastPipeline, generate_submission


def main() -> None:
    pipeline = ForecastPipeline.load(ROOT / "outputs" / "models" / "forecast_pipeline.joblib")
    submission = generate_submission(pipeline)
    output_path = ROOT / "outputs" / "submissions" / "submission_uncalibrated.csv"
    submission.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()

