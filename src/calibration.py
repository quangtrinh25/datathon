from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import CALIBRATION_GRID, CALIBRATION_MIN_SEGMENT_ROWS


def tune_scalar(y_true: pd.Series, y_pred: pd.Series, grid=CALIBRATION_GRID) -> float:
    best_scalar = 1.0
    best_score = float("inf")
    for scalar in grid:
        score = mean_absolute_error(y_true, y_pred * scalar)
        if score < best_score:
            best_score = score
            best_scalar = float(scalar)
    return best_scalar


def _boundary_flag(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    days_to_month_end = dates.dt.days_in_month - dates.dt.day
    return ((dates.dt.day <= 3) | (days_to_month_end <= 2)).astype(int)


def _prepare_calibration_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["quarter"] = frame["Date"].dt.quarter.astype(int)
    if "is_boundary" in frame.columns:
        frame["is_boundary"] = frame["is_boundary"].astype(int)
    else:
        frame["is_boundary"] = _boundary_flag(frame["Date"])
    return frame


def _fit_target_calibration(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    min_segment_rows: int = CALIBRATION_MIN_SEGMENT_ROWS,
) -> dict[str, object]:
    target_calibration = {
        "global_scalar": tune_scalar(frame[actual_column], frame[prediction_column]),
        "quarter_scalars": {},
        "segment_scalars": {},
    }

    for quarter, quarter_frame in frame.groupby("quarter", sort=True):
        target_calibration["quarter_scalars"][str(int(quarter))] = tune_scalar(
            quarter_frame[actual_column],
            quarter_frame[prediction_column],
        )

    for (quarter, is_boundary), segment_frame in frame.groupby(["quarter", "is_boundary"], sort=True):
        if len(segment_frame) < min_segment_rows:
            continue
        target_calibration["segment_scalars"][f"{int(quarter)}|{int(is_boundary)}"] = tune_scalar(
            segment_frame[actual_column],
            segment_frame[prediction_column],
        )

    return target_calibration


def _select_scalar(target_calibration: dict[str, object], quarter: int, is_boundary: int) -> float:
    segment_key = f"{int(quarter)}|{int(is_boundary)}"
    quarter_key = str(int(quarter))
    if segment_key in target_calibration["segment_scalars"]:
        return float(target_calibration["segment_scalars"][segment_key])
    if quarter_key in target_calibration["quarter_scalars"]:
        return float(target_calibration["quarter_scalars"][quarter_key])
    return float(target_calibration["global_scalar"])


def apply_calibration(predictions: pd.DataFrame, calibration: dict[str, object]) -> pd.DataFrame:
    frame = _prepare_calibration_frame(predictions)
    revenue_scalars = [
        _select_scalar(calibration["revenue"], quarter, is_boundary)
        for quarter, is_boundary in zip(frame["quarter"], frame["is_boundary"])
    ]
    cogs_scalars = [
        _select_scalar(calibration["cogs"], quarter, is_boundary)
        for quarter, is_boundary in zip(frame["quarter"], frame["is_boundary"])
    ]
    frame["Revenue_pred_calibrated"] = frame["Revenue_pred"] * pd.Series(revenue_scalars, index=frame.index)
    frame["COGS_pred_calibrated"] = frame["COGS_pred"] * pd.Series(cogs_scalars, index=frame.index)
    return frame


def calibrate_from_cv(predictions: pd.DataFrame) -> dict[str, object]:
    frame = _prepare_calibration_frame(predictions)
    return {
        "revenue": _fit_target_calibration(frame, "Revenue", "Revenue_pred"),
        "cogs": _fit_target_calibration(frame, "COGS", "COGS_pred"),
    }
