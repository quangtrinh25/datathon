from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-aux", action="store_true")
    parser.add_argument("--disable-specialist", action="store_true")
    args = parser.parse_args()

    fold_metrics, _ = run_backtest(
        use_aux_templates=not args.disable_aux,
        use_specialist=not args.disable_specialist,
    )
    payload = {
        "mean_metrics": fold_metrics.mean(numeric_only=True).to_dict(),
        "folds": fold_metrics.to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

