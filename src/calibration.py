from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import CALIBRATION_GRID


def tune_scalar(y_true: pd.Series, y_pred: pd.Series, grid=CALIBRATION_GRID) -> float:
    best_scalar = 1.0
    best_score = float("inf")
    for scalar in grid:
        score = mean_absolute_error(y_true, y_pred * scalar)
        if score < best_score:
            best_score = score
            best_scalar = float(scalar)
    return best_scalar


def calibrate_from_cv(predictions: pd.DataFrame) -> dict[str, float]:
    return {
        "revenue_scalar": tune_scalar(predictions["Revenue"], predictions["Revenue_pred"]),
        "cogs_scalar": tune_scalar(predictions["COGS"], predictions["COGS_pred"]),
    }

