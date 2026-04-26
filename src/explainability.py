from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def save_feature_importance(model_wrapper, columns: list[str], output_path) -> pd.DataFrame:
    importance = pd.DataFrame(model_wrapper.feature_importance(columns)).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(output_path, index=False)
    return importance
