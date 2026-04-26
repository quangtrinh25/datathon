from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.data_loader import load_sales


def load_submission_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    required = {"Date", "Revenue", "COGS"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["year"] = frame["Date"].dt.year.astype(int)
    frame["quarter"] = frame["Date"].dt.quarter.astype(int)
    frame["month"] = frame["Date"].dt.month.astype(int)
    frame["ratio"] = frame["COGS"] / frame["Revenue"]
    return frame


def align_submission_frames(anchor: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(anchor) != len(other) or not anchor["Date"].equals(other["Date"]):
        raise ValueError("Submission frames must have identical ordered Date columns.")
    return anchor.reset_index(drop=True), other.reset_index(drop=True)


def _weight_series(frame: pd.DataFrame, weight: float | Mapping[int, float]) -> pd.Series:
    if isinstance(weight, Mapping):
        return frame["quarter"].map(weight).fillna(0.0).astype(float)
    return pd.Series(float(weight), index=frame.index, dtype=float)


def preserve_group_mean_blend(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    target: str,
    weight: float | Mapping[int, float],
    group_keys: list[str],
) -> pd.Series:
    anchor_mean = anchor.groupby(group_keys)[target].transform("mean")
    shape_mean = shape_source.groupby(group_keys)[target].transform("mean")
    scaled_shape = shape_source[target] * anchor_mean / shape_mean.replace(0.0, np.nan)
    scaled_shape = scaled_shape.fillna(anchor[target])
    weights = _weight_series(anchor, weight)
    return anchor[target] + weights * (scaled_shape - anchor[target])


def preserve_year_quarter_mean_blend(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    target: str,
    weight: float | Mapping[int, float],
) -> pd.Series:
    return preserve_group_mean_blend(
        anchor=anchor,
        shape_source=shape_source,
        target=target,
        weight=weight,
        group_keys=["year", "quarter"],
    )


def preserve_year_month_mean_blend(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    target: str,
    weight: float | Mapping[int, float],
) -> pd.Series:
    return preserve_group_mean_blend(
        anchor=anchor,
        shape_source=shape_source,
        target=target,
        weight=weight,
        group_keys=["year", "month"],
    )


def blend_day_share_within_month(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    target: str,
    weight: float | Mapping[int, float],
) -> pd.Series:
    month_keys = [anchor["year"], anchor["month"]]
    anchor_month_total = anchor.groupby(["year", "month"])[target].transform("sum")
    shape_month_total = shape_source.groupby(["year", "month"])[target].transform("sum")
    anchor_share = anchor[target] / anchor_month_total.replace(0.0, np.nan)
    shape_share = shape_source[target] / shape_month_total.replace(0.0, np.nan)

    weights = _weight_series(anchor, weight)
    blended = anchor_share + weights * (shape_share - anchor_share)
    blended = blended / blended.groupby(month_keys).transform("sum").replace(0.0, np.nan)
    return blended.fillna(anchor_share)


def blend_month_share_within_quarter(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    target: str,
    weight: float | Mapping[int, float],
) -> pd.DataFrame:
    month_keys = ["year", "quarter", "month"]

    anchor_month = (
        anchor.groupby(month_keys, as_index=False)[target]
        .sum()
        .rename(columns={target: "anchor_total"})
    )
    shape_month = (
        shape_source.groupby(month_keys, as_index=False)[target]
        .sum()
        .rename(columns={target: "shape_total"})
    )
    month_frame = anchor_month.merge(shape_month, on=month_keys, how="left")
    month_frame["anchor_share"] = month_frame["anchor_total"] / month_frame.groupby(["year", "quarter"])[
        "anchor_total"
    ].transform("sum").replace(0.0, np.nan)
    month_frame["shape_share"] = month_frame["shape_total"] / month_frame.groupby(["year", "quarter"])[
        "shape_total"
    ].transform("sum").replace(0.0, np.nan)

    weights = _weight_series(month_frame, weight)
    month_frame["blended_share"] = month_frame["anchor_share"] + weights * (
        month_frame["shape_share"] - month_frame["anchor_share"]
    )
    month_frame["blended_share"] = month_frame["blended_share"] / month_frame.groupby(["year", "quarter"])[
        "blended_share"
    ].transform("sum").replace(0.0, np.nan)
    month_frame["blended_share"] = month_frame["blended_share"].fillna(month_frame["anchor_share"])
    return month_frame[month_keys + ["blended_share"]]


def factorized_quarter_month_day_blend(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    target: str,
    month_weight: float | Mapping[int, float],
    day_weight: float | Mapping[int, float],
) -> pd.Series:
    month_share = blend_month_share_within_quarter(anchor, shape_source, target, month_weight)
    work = anchor[["Date", "year", "quarter", "month"]].merge(
        month_share,
        on=["year", "quarter", "month"],
        how="left",
    )
    quarter_total = anchor.groupby(["year", "quarter"])[target].transform("sum")
    day_share = blend_day_share_within_month(anchor, shape_source, target, day_weight)
    return quarter_total * work["blended_share"].fillna(0.0) * day_share


def _candidate_frame(anchor: pd.DataFrame) -> pd.DataFrame:
    return anchor[["Date", "year", "quarter", "month"]].copy()


def _finalize_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["Date", "Revenue", "COGS"]].copy()


def make_quarter_shape_candidate(
    anchor: pd.DataFrame,
    shape_source: pd.DataFrame,
    revenue_weight: float,
    cogs_weight: float,
) -> pd.DataFrame:
    frame = _candidate_frame(anchor)
    frame["Revenue"] = preserve_year_quarter_mean_blend(anchor, shape_source, "Revenue", revenue_weight)
    frame["COGS"] = preserve_year_quarter_mean_blend(anchor, shape_source, "COGS", cogs_weight)
    return _finalize_candidate(frame)


def historical_quarter_margin_map() -> dict[int, float]:
    sales = load_sales()
    sales = sales.loc[sales["Date"] >= "2020-01-01"].copy()
    sales["quarter"] = sales["Date"].dt.quarter.astype(int)
    grouped = sales.groupby("quarter", as_index=True).agg(revenue=("Revenue", "sum"), cogs=("COGS", "sum"))
    ratio = grouped["cogs"] / grouped["revenue"]
    return {int(quarter): float(value) for quarter, value in ratio.items()}


def blend_cogs_toward_historical_margin(
    frame: pd.DataFrame,
    revenue: pd.Series,
    cogs: pd.Series,
    beta: float,
) -> pd.Series:
    quarter_margin = historical_quarter_margin_map()
    historical_cogs = revenue * frame["quarter"].map(quarter_margin).astype(float)
    blended = (1.0 - float(beta)) * cogs + float(beta) * historical_cogs

    target_group_mean = frame.groupby(["year", "quarter"])["COGS"].transform("mean")
    blended_group_mean = blended.groupby([frame["year"], frame["quarter"]]).transform("mean")
    return blended * target_group_mean / blended_group_mean.replace(0.0, np.nan)


def build_lb_candidates(
    notebook_raw: pd.DataFrame,
    notebook_v57: pd.DataFrame,
    model_shape: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    notebook_raw, model_shape = align_submission_frames(notebook_raw, model_shape)
    notebook_v57, model_shape = align_submission_frames(notebook_v57, model_shape)

    candidates: dict[str, pd.DataFrame] = {}

    candidates["candidate_00_notebook_raw"] = notebook_raw[["Date", "Revenue", "COGS"]].copy()
    candidates["candidate_01_notebook_v57"] = notebook_v57[["Date", "Revenue", "COGS"]].copy()

    shape15 = _candidate_frame(notebook_v57)
    shape15["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 0.15)
    shape15["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.10)
    candidates["candidate_02_shape15"] = _finalize_candidate(shape15)

    shape25 = _candidate_frame(notebook_v57)
    shape25["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 0.25)
    shape25["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.15)
    candidates["candidate_03_shape25"] = _finalize_candidate(shape25)

    qaware = _candidate_frame(notebook_v57)
    qaware["Revenue"] = preserve_year_quarter_mean_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        {1: 0.22, 2: 0.10, 3: 0.08, 4: 0.15},
    )
    qaware_cogs = preserve_year_quarter_mean_blend(
        notebook_v57,
        model_shape,
        "COGS",
        {1: 0.18, 2: 0.08, 3: 0.10, 4: 0.12},
    )
    qaware["COGS"] = blend_cogs_toward_historical_margin(notebook_v57, qaware["Revenue"], qaware_cogs, beta=0.15)
    candidates["candidate_04_qaware_margin15"] = _finalize_candidate(qaware)

    qaware_plus = _candidate_frame(notebook_v57)
    qaware_plus["Revenue"] = preserve_year_quarter_mean_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        {1: 0.28, 2: 0.12, 3: 0.10, 4: 0.18},
    )
    qaware_plus_cogs = preserve_year_quarter_mean_blend(
        notebook_v57,
        model_shape,
        "COGS",
        {1: 0.22, 2: 0.10, 3: 0.12, 4: 0.15},
    )
    qaware_plus["COGS"] = blend_cogs_toward_historical_margin(
        notebook_v57,
        qaware_plus["Revenue"],
        qaware_plus_cogs,
        beta=0.22,
    )
    candidates["candidate_05_qaware_margin22"] = _finalize_candidate(qaware_plus)

    shape35 = _candidate_frame(notebook_v57)
    shape35["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 0.35)
    shape35["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.25)
    candidates["candidate_06_shape35"] = _finalize_candidate(shape35)

    shape45 = _candidate_frame(notebook_v57)
    shape45["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 0.45)
    shape45["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.30)
    candidates["candidate_07_shape45"] = _finalize_candidate(shape45)

    shape60 = _candidate_frame(notebook_v57)
    shape60["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 0.60)
    shape60["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.40)
    candidates["candidate_08_shape60"] = _finalize_candidate(shape60)

    shape35_month = shape35.copy()
    shape35_month["Revenue"] = preserve_year_month_mean_blend(shape35, model_shape, "Revenue", 1.0)
    shape35_month["COGS"] = preserve_year_month_mean_blend(shape35, model_shape, "COGS", 1.0)
    candidates["candidate_09_shape35_monthrefine"] = _finalize_candidate(shape35_month)

    shape45_month = shape45.copy()
    shape45_month["Revenue"] = preserve_year_month_mean_blend(shape45, model_shape, "Revenue", 1.0)
    shape45_month["COGS"] = preserve_year_month_mean_blend(shape45, model_shape, "COGS", 1.0)
    candidates["candidate_10_shape45_monthrefine"] = _finalize_candidate(shape45_month)

    shape60_month = shape60.copy()
    shape60_month["Revenue"] = preserve_year_month_mean_blend(shape60, model_shape, "Revenue", 1.0)
    shape60_month["COGS"] = preserve_year_month_mean_blend(shape60, model_shape, "COGS", 1.0)
    candidates["candidate_11_shape60_monthrefine"] = _finalize_candidate(shape60_month)

    shape80 = _candidate_frame(notebook_v57)
    shape80["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 0.80)
    shape80["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.55)
    candidates["candidate_12_shape80"] = _finalize_candidate(shape80)

    shape100 = _candidate_frame(notebook_v57)
    shape100["Revenue"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "Revenue", 1.00)
    shape100["COGS"] = preserve_year_quarter_mean_blend(notebook_v57, model_shape, "COGS", 0.70)
    candidates["candidate_13_shape100"] = _finalize_candidate(shape100)

    month90_day00 = _candidate_frame(notebook_v57)
    month90_day00["Revenue"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        month_weight=0.90,
        day_weight=0.00,
    )
    month90_day00["COGS"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "COGS",
        month_weight=0.75,
        day_weight=0.00,
    )
    candidates["candidate_14_month90_day00"] = _finalize_candidate(month90_day00)

    month100_day00 = _candidate_frame(notebook_v57)
    month100_day00["Revenue"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        month_weight=1.00,
        day_weight=0.00,
    )
    month100_day00["COGS"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "COGS",
        month_weight=0.85,
        day_weight=0.00,
    )
    candidates["candidate_15_month100_day00"] = _finalize_candidate(month100_day00)

    month90_day10 = _candidate_frame(notebook_v57)
    month90_day10["Revenue"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        month_weight=0.90,
        day_weight=0.10,
    )
    month90_day10["COGS"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "COGS",
        month_weight=0.75,
        day_weight=0.05,
    )
    candidates["candidate_16_month90_day10"] = _finalize_candidate(month90_day10)

    month100_day10 = _candidate_frame(notebook_v57)
    month100_day10["Revenue"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        month_weight=1.00,
        day_weight=0.10,
    )
    month100_day10["COGS"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "COGS",
        month_weight=0.85,
        day_weight=0.05,
    )
    candidates["candidate_17_month100_day10"] = _finalize_candidate(month100_day10)

    month100_day20 = _candidate_frame(notebook_v57)
    month100_day20["Revenue"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "Revenue",
        month_weight=1.00,
        day_weight=0.20,
    )
    month100_day20["COGS"] = factorized_quarter_month_day_blend(
        notebook_v57,
        model_shape,
        "COGS",
        month_weight=0.85,
        day_weight=0.10,
    )
    candidates["candidate_18_month100_day20"] = _finalize_candidate(month100_day20)

    candidates["candidate_19_shape70"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.70,
        cogs_weight=0.48,
    )
    candidates["candidate_20_shape75"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.75,
        cogs_weight=0.51,
    )
    candidates["candidate_21_shape85"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.85,
        cogs_weight=0.59,
    )
    candidates["candidate_22_shape90"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.90,
        cogs_weight=0.62,
    )
    candidates["candidate_23_rev80_cogs50"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.80,
        cogs_weight=0.50,
    )
    candidates["candidate_24_rev80_cogs60"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.80,
        cogs_weight=0.60,
    )
    candidates["candidate_25_rev85_cogs50"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.85,
        cogs_weight=0.50,
    )
    candidates["candidate_26_rev90_cogs55"] = make_quarter_shape_candidate(
        notebook_v57,
        model_shape,
        revenue_weight=0.90,
        cogs_weight=0.55,
    )
    return candidates


def summarize_submission(frame: pd.DataFrame) -> dict[str, object]:
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["year"] = work["Date"].dt.year.astype(int)
    work["quarter"] = work["Date"].dt.quarter.astype(int)
    work["ratio"] = work["COGS"] / work["Revenue"]
    return {
        "mean_revenue": float(work["Revenue"].mean()),
        "mean_cogs": float(work["COGS"].mean()),
        "mean_ratio": float(work["ratio"].mean()),
        "year_quarter_mean": {
            f"{int(year)}Q{int(quarter)}": {
                "Revenue": float(group["Revenue"].mean()),
                "COGS": float(group["COGS"].mean()),
                "ratio": float(group["ratio"].mean()),
            }
            for (year, quarter), group in work.groupby(["year", "quarter"], sort=True)
        },
    }
