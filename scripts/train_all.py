from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_training_pipeline


def main() -> None:
    summary = run_training_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

