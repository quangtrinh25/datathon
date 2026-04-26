from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import LAGS, ROLLING_WINDOWS, TET_DATES, VN_FIXED_HOLIDAYS
from src.data_loader import build_aux_daily_observations, build_daily_promo_features


def _month_end_day(date_series: pd.Series) -> pd.Series:
    return date_series.dt.days_in_month - date_series.dt.day


def _make_fourier(date_series: pd.Series, period: float, order: int, prefix: str) -> pd.DataFrame:
    steps = (date_series - pd.Timestamp("2012-01-01")).dt.days.astype(float).to_numpy()
    values: dict[str, np.ndarray] = {}
    for n in range(1, order + 1):
        angle = 2 * np.pi * n * steps / period
        values[f"{prefix}_sin_{n}"] = np.sin(angle)
        values[f"{prefix}_cos_{n}"] = np.cos(angle)
    return pd.DataFrame(values, index=date_series.index)


@dataclass
class FeatureBuilder:
    history_end: pd.Timestamp | None = None
    aux_observations: pd.DataFrame | None = None
    target_profile_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    aux_profile_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    static_feature_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    aux_columns: tuple[str, ...] = (
        "order_count",
        "refund_amount",
        "return_quantity",
        "review_count",
        "review_rating_mean",
        "shipping_fee",
        "days_of_supply",
        "fill_rate",
        "stockout_rate",
        "reorder_rate",
        "sessions",
        "unique_visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
    )

    def fit(self, history_sales: pd.DataFrame) -> "FeatureBuilder":
        history_sales = history_sales.sort_values("Date").reset_index(drop=True).copy()
        self.history_end = pd.Timestamp(history_sales["Date"].max())

        static_history = self._build_calendar_frame(history_sales[["Date"]])
        promo_history = build_daily_promo_features(static_history["Date"], self.history_end)
        self.aux_observations = build_aux_daily_observations()

        history = (
            static_history.merge(promo_history, on="Date", how="left")
            .merge(self.aux_observations, on="Date", how="left")
            .merge(history_sales, on="Date", how="left")
        )
        history["ratio"] = history["COGS"] / history["Revenue"]

        target_specs = {
            "template_revenue_dom": ("Revenue", ["day"]),
            "template_revenue_dte": ("Revenue", ["days_to_month_end"]),
            "template_revenue_month_dow": ("Revenue", ["month", "dow"]),
            "template_revenue_doy": ("Revenue", ["month", "day"]),
            "template_ratio_dom": ("ratio", ["day"]),
            "template_ratio_dte": ("ratio", ["days_to_month_end"]),
            "template_ratio_month_dow": ("ratio", ["month", "dow"]),
            "template_ratio_doy": ("ratio", ["month", "day"]),
        }
        self.target_profile_tables = {
            name: history.groupby(keys, as_index=False)[value].mean()
            for name, (value, keys) in target_specs.items()
        }

        aux_key = ["month", "dow"]
        self.aux_profile_tables = {
            column: history.groupby(aux_key, as_index=False)[column].mean()
            for column in self.aux_columns
            if column in history.columns
        }
        return self

    def build_static_frame(self, dates: pd.DataFrame) -> pd.DataFrame:
        if self.history_end is None:
            raise RuntimeError("FeatureBuilder.fit must run before build_static_frame.")

        frame = self._build_calendar_frame(dates)
        frame = frame.merge(build_daily_promo_features(frame["Date"], self.history_end), on="Date", how="left")

        for name, table in self.target_profile_tables.items():
            if name.endswith("_month_dow"):
                keys = ["month", "dow"]
            elif name.endswith("_doy"):
                keys = ["month", "day"]
            elif name.endswith("_dte"):
                keys = ["days_to_month_end"]
            else:
                keys = ["day"]
            frame = frame.merge(table.rename(columns={table.columns[-1]: name}), on=keys, how="left")

        for column, table in self.aux_profile_tables.items():
            frame = frame.merge(
                table.rename(columns={column: f"expected_{column}"}),
                on=["month", "dow"],
                how="left",
            )

        for column in frame.columns:
            if column != "Date":
                frame[column] = frame[column].fillna(0.0)

        frame["specialist_flag"] = (
            frame["is_first_3_days"] + frame["is_last_3_days"] + (frame["promo_active"] > 0).astype(float)
        ).clip(0, 1)
        frame["promo_boundary_pressure"] = frame["promo_weighted_discount"] * (
            frame["is_first_3_days"] + frame["is_last_3_days"]
        )
        frame["promo_tet_pressure"] = frame["promo_weighted_discount"] * frame["is_tet_window_14"]

        fourier = _make_fourier(frame["Date"], period=365.25, order=3, prefix="yearly")
        frame = pd.concat([frame, fourier], axis=1)

        self.static_feature_columns = [column for column in frame.columns if column != "Date"]
        return frame

    def make_training_frame(self, history_sales: pd.DataFrame) -> pd.DataFrame:
        history_sales = history_sales.sort_values("Date").reset_index(drop=True).copy()
        frame = self.build_static_frame(history_sales[["Date"]]).merge(history_sales, on="Date", how="left")
        frame["ratio"] = frame["COGS"] / frame["Revenue"]

        dynamic_columns: dict[str, pd.Series] = {}
        for lag in LAGS:
            dynamic_columns[f"revenue_lag_{lag}"] = frame["Revenue"].shift(lag)
            dynamic_columns[f"ratio_lag_{lag}"] = frame["ratio"].shift(lag)

        for window in ROLLING_WINDOWS:
            dynamic_columns[f"revenue_roll_mean_{window}"] = frame["Revenue"].shift(1).rolling(window).mean()
            dynamic_columns[f"revenue_roll_std_{window}"] = frame["Revenue"].shift(1).rolling(window).std()
            dynamic_columns[f"ratio_roll_mean_{window}"] = frame["ratio"].shift(1).rolling(window).mean()

        dynamic_frame = pd.DataFrame(dynamic_columns)
        frame = pd.concat([frame, dynamic_frame], axis=1)
        frame["revenue_profile_gap_28"] = frame["revenue_lag_28"] - frame["template_revenue_dom"]
        frame["ratio_profile_gap_28"] = frame["ratio_lag_28"] - frame["template_ratio_dom"]
        frame["recent_growth_7_28"] = frame["revenue_lag_7"] / frame["revenue_lag_28"].replace(0, np.nan)
        frame["recent_growth_7_28"] = frame["recent_growth_7_28"].replace([np.inf, -np.inf], np.nan)

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        self.feature_columns = [
            column
            for column in frame.columns
            if column not in {"Date", "Revenue", "COGS", "ratio"}
        ]
        return frame

    def build_dynamic_row(
        self,
        date_row: pd.Series,
        revenue_history: list[float],
        ratio_history: list[float],
    ) -> dict[str, float]:
        values = date_row.to_dict()
        for lag in LAGS:
            values[f"revenue_lag_{lag}"] = revenue_history[-lag]
            values[f"ratio_lag_{lag}"] = ratio_history[-lag]
        for window in ROLLING_WINDOWS:
            revenue_slice = revenue_history[-window:]
            ratio_slice = ratio_history[-window:]
            values[f"revenue_roll_mean_{window}"] = float(np.mean(revenue_slice))
            values[f"revenue_roll_std_{window}"] = float(np.std(revenue_slice, ddof=0))
            values[f"ratio_roll_mean_{window}"] = float(np.mean(ratio_slice))
        values["revenue_profile_gap_28"] = values["revenue_lag_28"] - values["template_revenue_dom"]
        values["ratio_profile_gap_28"] = values["ratio_lag_28"] - values["template_ratio_dom"]
        values["recent_growth_7_28"] = values["revenue_lag_7"] / max(values["revenue_lag_28"], 1.0)
        return values

    def _build_calendar_frame(self, dates: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame({"Date": pd.to_datetime(dates["Date"])}).sort_values("Date").reset_index(drop=True)
        frame["year"] = frame["Date"].dt.year
        frame["quarter"] = frame["Date"].dt.quarter
        frame["month"] = frame["Date"].dt.month
        frame["week"] = frame["Date"].dt.isocalendar().week.astype(int)
        frame["dow"] = frame["Date"].dt.dayofweek
        frame["day"] = frame["Date"].dt.day
        frame["days_in_month"] = frame["Date"].dt.days_in_month
        frame["days_to_month_end"] = _month_end_day(frame["Date"])
        frame["days_from_month_start"] = frame["Date"].dt.day - 1
        frame["week_of_month"] = ((frame["day"] - 1) // 7) + 1
        frame["is_weekend"] = frame["dow"].isin([5, 6]).astype(float)
        frame["is_month_start"] = (frame["day"] == 1).astype(float)
        frame["is_month_end"] = (frame["days_to_month_end"] == 0).astype(float)
        frame["is_first_3_days"] = (frame["day"] <= 3).astype(float)
        frame["is_last_3_days"] = (frame["days_to_month_end"] <= 2).astype(float)
        frame["is_pay_cycle_window"] = (
            frame["is_first_3_days"] + frame["is_last_3_days"]
        ).clip(0, 1)

        for name, (month, day) in VN_FIXED_HOLIDAYS.items():
            frame[f"holiday_{name}"] = ((frame["month"] == month) & (frame["day"] == day)).astype(float)

        frame["days_to_tet"] = 0.0
        frame["days_from_tet"] = 0.0
        frame["is_tet_window_14"] = 0.0
        frame["is_post_tet_7"] = 0.0
        for index, date in frame["Date"].items():
            tet = pd.Timestamp(TET_DATES.get(int(date.year), TET_DATES[max(TET_DATES)]))
            delta = (date - tet).days
            frame.at[index, "days_to_tet"] = float(max(-40, min(40, -delta)))
            frame.at[index, "days_from_tet"] = float(max(-40, min(40, delta)))
            frame.at[index, "is_tet_window_14"] = float(abs(delta) <= 14)
            frame.at[index, "is_post_tet_7"] = float(0 <= delta <= 7)
        return frame
