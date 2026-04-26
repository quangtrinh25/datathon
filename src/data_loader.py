from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DATA_RAW, FORECAST_END, TRAIN_END, TRAIN_START


def load_sales() -> pd.DataFrame:
    sales = pd.read_csv(DATA_RAW / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    full_range = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    sales = sales.set_index("Date").reindex(full_range).rename_axis("Date").reset_index()
    sales[["Revenue", "COGS"]] = sales[["Revenue", "COGS"]].fillna(0.0)
    return sales


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "sample_submission.csv", parse_dates=["Date"]).sort_values("Date")


def load_promotions() -> pd.DataFrame:
    promotions = pd.read_csv(
        DATA_RAW / "promotions.csv",
        parse_dates=["start_date", "end_date"],
    ).sort_values("start_date")
    promotions["duration_days"] = (
        promotions["end_date"] - promotions["start_date"]
    ).dt.days + 1
    promotions["applicable_category"] = (
        promotions["applicable_category"].fillna("ALL").replace({"nan": "ALL"})
    )
    return promotions


def estimate_category_weights(history_end: pd.Timestamp) -> dict[str, float]:
    orders = pd.read_csv(DATA_RAW / "orders.csv", parse_dates=["order_date"])
    orders = orders.loc[orders["order_date"] <= history_end, ["order_id", "order_date"]]

    items = pd.read_csv(DATA_RAW / "order_items.csv", low_memory=False)
    products = pd.read_csv(DATA_RAW / "products.csv", usecols=["product_id", "category"])

    merged = items.merge(orders, on="order_id", how="inner").merge(products, on="product_id", how="left")
    merged["line_revenue"] = (
        merged["quantity"].fillna(0).astype(float) * merged["unit_price"].fillna(0).astype(float)
        - merged["discount_amount"].fillna(0).astype(float)
    )
    shares = merged.groupby("category", dropna=False)["line_revenue"].sum()
    total = float(shares.sum())
    if total <= 0:
        return {"ALL": 1.0, "Streetwear": 0.5, "Outdoor": 0.5}
    normalized = (shares / total).to_dict()
    normalized["ALL"] = 1.0
    normalized.setdefault("Streetwear", 0.0)
    normalized.setdefault("Outdoor", 0.0)
    return {str(key): float(value) for key, value in normalized.items()}


def infer_recurring_promotions(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    promotions = load_promotions()
    prototypes: list[dict[str, object]] = []

    for promo_name, group in promotions.groupby("promo_name", sort=False):
        years = sorted(group["start_date"].dt.year.unique().tolist())
        odd_year_only = all(year % 2 == 1 for year in years)
        start_month = int(group["start_date"].dt.month.mode().iloc[0])
        start_day = int(round(group["start_date"].dt.day.median()))
        duration_days = int(round(group["duration_days"].median()))
        prototypes.append(
            {
                "promo_name": promo_name,
                "start_month": start_month,
                "start_day": start_day,
                "duration_days": duration_days,
                "odd_year_only": odd_year_only,
                "applicable_category": str(group["applicable_category"].mode().iloc[0]),
                "promo_type": str(group["promo_type"].mode().iloc[0]),
                "discount_value": float(group["discount_value"].median()),
                "promo_channel": str(group["promo_channel"].mode().iloc[0]),
                "stackable_flag": float(group["stackable_flag"].mean()),
            }
        )

    rows: list[dict[str, object]] = []
    for proto in prototypes:
        for year in range(start_date.year, end_date.year + 1):
            if proto["odd_year_only"] and year % 2 == 0:
                continue
            start = pd.Timestamp(year=year, month=int(proto["start_month"]), day=int(proto["start_day"]))
            end = start + pd.Timedelta(days=int(proto["duration_days"]) - 1)
            if end < start_date or start > end_date:
                continue
            active_dates = pd.date_range(max(start, start_date), min(end, end_date), freq="D")
            for active_date in active_dates:
                rows.append(
                    {
                        "Date": active_date,
                        "promo_name": proto["promo_name"],
                        "applicable_category": proto["applicable_category"],
                        "promo_type": proto["promo_type"],
                        "discount_value": proto["discount_value"],
                        "promo_channel": proto["promo_channel"],
                        "stackable_flag": proto["stackable_flag"],
                        "days_since_start": (active_date - start).days,
                        "days_to_end": (end - active_date).days,
                    }
                )

    calendar = pd.DataFrame(rows)
    if calendar.empty:
        return pd.DataFrame(columns=["Date"])
    return calendar.sort_values(["Date", "promo_name"]).reset_index(drop=True)


def build_daily_promo_features(
    dates: Iterable[pd.Timestamp],
    history_end: pd.Timestamp,
) -> pd.DataFrame:
    date_frame = pd.DataFrame({"Date": pd.to_datetime(pd.Index(dates))}).sort_values("Date")
    category_weights = estimate_category_weights(history_end)
    promo_calendar = infer_recurring_promotions(date_frame["Date"].min(), pd.Timestamp(FORECAST_END))
    promo_calendar = promo_calendar.loc[promo_calendar["Date"].isin(date_frame["Date"])]

    if promo_calendar.empty:
        empty = date_frame.copy()
        empty["promo_active"] = 0.0
        return empty

    category_map = {
        "ALL": category_weights.get("ALL", 1.0),
        "Streetwear": category_weights.get("Streetwear", 0.0),
        "Outdoor": category_weights.get("Outdoor", 0.0),
    }
    promo_calendar = promo_calendar.copy()
    promo_calendar["category_weight"] = promo_calendar["applicable_category"].map(category_map).fillna(0.15)
    promo_calendar["weighted_discount"] = promo_calendar["discount_value"] * promo_calendar["category_weight"]
    promo_calendar["is_percentage"] = (promo_calendar["promo_type"] == "percentage").astype(float)
    promo_calendar["is_fixed"] = (promo_calendar["promo_type"] == "fixed").astype(float)
    promo_calendar["cat_all"] = (promo_calendar["applicable_category"] == "ALL").astype(float)
    promo_calendar["cat_streetwear"] = (promo_calendar["applicable_category"] == "Streetwear").astype(float)
    promo_calendar["cat_outdoor"] = (promo_calendar["applicable_category"] == "Outdoor").astype(float)

    for channel in ["all_channels", "email", "in_store", "online", "social_media"]:
        promo_calendar[f"channel_{channel}"] = (promo_calendar["promo_channel"] == channel).astype(float)

    grouped = (
        promo_calendar.groupby("Date", as_index=False)
        .agg(
            promo_active=("promo_name", "size"),
            promo_percentage_count=("is_percentage", "sum"),
            promo_fixed_count=("is_fixed", "sum"),
            promo_stackable_score=("stackable_flag", "sum"),
            promo_max_discount=("discount_value", "max"),
            promo_mean_discount=("discount_value", "mean"),
            promo_weighted_discount=("weighted_discount", "sum"),
            promo_days_since_start=("days_since_start", "mean"),
            promo_days_to_end=("days_to_end", "mean"),
            promo_cat_all=("cat_all", "sum"),
            promo_cat_streetwear=("cat_streetwear", "sum"),
            promo_cat_outdoor=("cat_outdoor", "sum"),
            channel_all_channels=("channel_all_channels", "sum"),
            channel_email=("channel_email", "sum"),
            channel_in_store=("channel_in_store", "sum"),
            channel_online=("channel_online", "sum"),
            channel_social_media=("channel_social_media", "sum"),
        )
    )
    return date_frame.merge(grouped, on="Date", how="left").fillna(0.0)


def build_aux_daily_observations() -> pd.DataFrame:
    sales = load_sales()[["Date"]]

    orders = pd.read_csv(DATA_RAW / "orders.csv", parse_dates=["order_date"])
    orders_daily = (
        orders.groupby("order_date", as_index=False)
        .agg(order_count=("order_id", "nunique"))
        .rename(columns={"order_date": "Date"})
    )

    returns = pd.read_csv(DATA_RAW / "returns.csv", parse_dates=["return_date"])
    returns_daily = (
        returns.groupby("return_date", as_index=False)
        .agg(
            refund_amount=("refund_amount", "sum"),
            return_quantity=("return_quantity", "sum"),
        )
        .rename(columns={"return_date": "Date"})
    )

    reviews = pd.read_csv(DATA_RAW / "reviews.csv", parse_dates=["review_date"])
    reviews_daily = (
        reviews.groupby("review_date", as_index=False)
        .agg(review_count=("review_id", "count"), review_rating_mean=("rating", "mean"))
        .rename(columns={"review_date": "Date"})
    )

    shipments = pd.read_csv(DATA_RAW / "shipments.csv", parse_dates=["delivery_date"])
    shipments_daily = (
        shipments.groupby("delivery_date", as_index=False)
        .agg(shipping_fee=("shipping_fee", "sum"))
        .rename(columns={"delivery_date": "Date"})
    )

    inventory = pd.read_csv(DATA_RAW / "inventory.csv", parse_dates=["snapshot_date"])
    inventory_daily = (
        inventory.groupby("snapshot_date", as_index=False)
        .agg(
            days_of_supply=("days_of_supply", "mean"),
            fill_rate=("fill_rate", "mean"),
            stockout_rate=("stockout_flag", "mean"),
            reorder_rate=("reorder_flag", "mean"),
        )
        .rename(columns={"snapshot_date": "Date"})
    )

    web = pd.read_csv(DATA_RAW / "web_traffic.csv", parse_dates=["date"]).rename(columns={"date": "Date"})
    web = web[
        [
            "Date",
            "sessions",
            "unique_visitors",
            "page_views",
            "bounce_rate",
            "avg_session_duration_sec",
        ]
    ]

    merged = sales.merge(orders_daily, on="Date", how="left")
    for frame in [returns_daily, reviews_daily, shipments_daily, inventory_daily, web]:
        merged = merged.merge(frame, on="Date", how="left")

    for column in merged.columns:
        if column != "Date":
            merged[column] = merged[column].astype(float).fillna(0.0)
    return merged
