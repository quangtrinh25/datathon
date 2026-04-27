from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import build_aux_daily_observations, build_daily_promo_features, load_promotions, load_sales


def _round_frame(frame: pd.DataFrame, digits: int = 4) -> list[dict[str, object]]:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].round(digits)
    return rounded.to_dict(orient="records")


def main() -> None:
    sales = load_sales().sort_values("Date").reset_index(drop=True)
    sales["year"] = sales["Date"].dt.year.astype(int)
    sales["quarter"] = sales["Date"].dt.quarter.astype(int)
    sales["month"] = sales["Date"].dt.month.astype(int)
    sales["ratio"] = sales["COGS"] / sales["Revenue"]
    sales["dom"] = sales["Date"].dt.day.astype(int)
    sales["dte"] = (sales["Date"].dt.days_in_month - sales["Date"].dt.day).astype(int)

    year_summary = sales.groupby("year", as_index=False).agg(
        revenue=("Revenue", "sum"),
        cogs=("COGS", "sum"),
        ratio=("ratio", "mean"),
    )

    quarter_summary = sales.groupby(["year", "quarter"], as_index=False).agg(
        revenue=("Revenue", "mean"),
        cogs=("COGS", "mean"),
        ratio=("ratio", "mean"),
    )

    quarter_share = sales.groupby(["year", "quarter"], as_index=False)["Revenue"].sum()
    quarter_totals = quarter_share.groupby("year", as_index=False)["Revenue"].sum().rename(columns={"Revenue": "year_total"})
    quarter_share = quarter_share.merge(quarter_totals, on="year", how="left")
    quarter_share["share"] = quarter_share["Revenue"] / quarter_share["year_total"]
    quarter_share_pivot = quarter_share.pivot(index="year", columns="quarter", values="share").sort_index()

    ratio_pivot = (
        sales.groupby(["year", "quarter"], as_index=False)["ratio"]
        .mean()
        .pivot(index="year", columns="quarter", values="ratio")
        .sort_index()
    )

    regime_similarity: dict[str, dict[str, float]] = {}
    for target_year in [2021, 2022]:
        share_distance = ((quarter_share_pivot.sub(quarter_share_pivot.loc[target_year], axis=1)) ** 2).sum(axis=1) ** 0.5
        ratio_distance = ((ratio_pivot.sub(ratio_pivot.loc[target_year], axis=1)) ** 2).sum(axis=1) ** 0.5
        regime_similarity[str(target_year)] = {
            "quarter_share_distance": {str(int(year)): float(value) for year, value in share_distance.sort_values().items()},
            "ratio_distance": {str(int(year)): float(value) for year, value in ratio_distance.sort_values().items()},
        }

    month_share = sales.groupby(["year", "quarter", "month"], as_index=False)["Revenue"].sum()
    month_quarter_totals = month_share.groupby(["year", "quarter"], as_index=False)["Revenue"].sum().rename(
        columns={"Revenue": "quarter_total"}
    )
    month_share = month_share.merge(month_quarter_totals, on=["year", "quarter"], how="left")
    month_share["month_of_quarter"] = ((month_share["month"] - 1) % 3) + 1
    month_share["share"] = month_share["Revenue"] / month_share["quarter_total"]
    month_of_quarter_summary = month_share.groupby("month_of_quarter", as_index=False)["share"].agg(["mean", "std", "min", "max"])
    month_of_quarter_summary = month_of_quarter_summary.reset_index()

    edge_effects = []
    for name, mask in [
        ("first3", sales["dom"] <= 3),
        ("last3", sales["dte"] <= 2),
        ("middle", (sales["dom"] > 3) & (sales["dte"] > 2)),
    ]:
        edge_effects.append(
            {
                "segment": name,
                "mean_revenue": float(sales.loc[mask, "Revenue"].mean()),
                "mean_ratio": float(sales.loc[mask, "ratio"].mean()),
                "rows": int(mask.sum()),
            }
        )

    promo_features = build_daily_promo_features(sales["Date"], sales["Date"].max())
    promo_frame = sales.merge(
        promo_features[["Date", "promo_active", "promo_weighted_discount"]],
        on="Date",
        how="left",
    )
    promo_effect = []
    for label, mask in [("promo", promo_frame["promo_active"] > 0), ("nonpromo", promo_frame["promo_active"] == 0)]:
        promo_effect.append(
            {
                "segment": label,
                "mean_revenue": float(promo_frame.loc[mask, "Revenue"].mean()),
                "mean_ratio": float(promo_frame.loc[mask, "ratio"].mean()),
                "rows": int(mask.sum()),
            }
        )

    aux = build_aux_daily_observations()
    merged = sales.merge(aux, on="Date", how="left")
    aux_columns = [
        "order_count",
        "refund_amount",
        "return_quantity",
        "review_count",
        "review_rating_mean",
        "shipping_fee",
        "days_of_supply",
        "fill_rate",
        "stockout_rate",
        "sessions",
        "unique_visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
    ]
    corr = merged[["Revenue", "COGS", *aux_columns]].corr(numeric_only=True)["Revenue"].sort_values(ascending=False)

    report = {
        "date_range": {
            "start": sales["Date"].min().date().isoformat(),
            "end": sales["Date"].max().date().isoformat(),
            "rows": int(len(sales)),
        },
        "year_summary": _round_frame(year_summary, digits=3),
        "quarter_summary": _round_frame(quarter_summary, digits=3),
        "regime_similarity": regime_similarity,
        "month_of_quarter_share": _round_frame(month_of_quarter_summary, digits=4),
        "edge_effects": _round_frame(pd.DataFrame(edge_effects), digits=4),
        "promo_effect": _round_frame(pd.DataFrame(promo_effect), digits=4),
        "promo_family_counts": load_promotions()["promo_family"].value_counts().to_dict(),
        "revenue_correlations": {str(key): round(float(value), 4) for key, value in corr.items()},
    }

    out_path = ROOT / "outputs" / "reports" / "data_signal_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
