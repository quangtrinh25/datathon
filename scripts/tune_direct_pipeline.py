from __future__ import annotations

import json
import argparse
import sys
import warnings
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ALLOCATION_LGB_PARAMS, DIRECT_BLEND, DIRECT_LGB_PARAMS, ALLOCATION_BLEND, REPORTS_DIR
from src.pipeline import run_backtest


def score_fold_metrics(metrics) -> dict[str, float]:
    mean_metrics = metrics.mean(numeric_only=True)
    return {
        "score": float(
            mean_metrics["revenue_mae"] + mean_metrics["cogs_mae"] + 0.15 * mean_metrics["boundary_revenue_mae"]
        ),
        "revenue_mae": float(mean_metrics["revenue_mae"]),
        "cogs_mae": float(mean_metrics["cogs_mae"]),
        "boundary_revenue_mae": float(mean_metrics["boundary_revenue_mae"]),
        "revenue_r2": float(mean_metrics["revenue_r2"]),
        "cogs_r2": float(mean_metrics["cogs_r2"]),
    }


def main() -> None:
    warnings.simplefilter("ignore")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", action="append", dest="folds", default=None)
    args = parser.parse_args()

    candidates = [
        {
            "name": "baseline",
            "direct_lgb_params": deepcopy(DIRECT_LGB_PARAMS),
            "allocation_lgb_params": deepcopy(ALLOCATION_LGB_PARAMS),
            "direct_blend": deepcopy(DIRECT_BLEND),
            "allocation_blend": deepcopy(ALLOCATION_BLEND),
        },
        {
            "name": "shape_heavier",
            "direct_lgb_params": {**DIRECT_LGB_PARAMS, "min_child_samples": 28, "reg_lambda": 0.8},
            "allocation_lgb_params": {
                **ALLOCATION_LGB_PARAMS,
                "n_estimators": 650,
                "learning_rate": 0.025,
                "num_leaves": 47,
                "min_child_samples": 18,
                "reg_lambda": 0.75,
            },
            "direct_blend": {"lgb": 0.6, "ridge": 0.4, "specialist": 0.6},
            "allocation_blend": {"revenue": 0.68, "cogs": 0.45},
        },
        {
            "name": "regularized_anchor",
            "direct_lgb_params": {
                **DIRECT_LGB_PARAMS,
                "n_estimators": 700,
                "learning_rate": 0.025,
                "num_leaves": 47,
                "min_child_samples": 36,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.25,
            },
            "allocation_lgb_params": {**ALLOCATION_LGB_PARAMS, "num_leaves": 47, "min_child_samples": 20},
            "direct_blend": {"lgb": 0.65, "ridge": 0.35, "specialist": 0.55},
            "allocation_blend": {"revenue": 0.62, "cogs": 0.4},
        },
        {
            "name": "smoother_allocation",
            "direct_lgb_params": deepcopy(DIRECT_LGB_PARAMS),
            "allocation_lgb_params": {
                **ALLOCATION_LGB_PARAMS,
                "n_estimators": 750,
                "learning_rate": 0.0225,
                "num_leaves": 31,
                "min_child_samples": 30,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
            "direct_blend": {"lgb": 0.6, "ridge": 0.4, "specialist": 0.6},
            "allocation_blend": {"revenue": 0.7, "cogs": 0.42},
        },
        {
            "name": "balanced_tuned",
            "direct_lgb_params": {
                **DIRECT_LGB_PARAMS,
                "n_estimators": 650,
                "learning_rate": 0.0275,
                "num_leaves": 55,
                "min_child_samples": 30,
                "reg_lambda": 0.9,
            },
            "allocation_lgb_params": {
                **ALLOCATION_LGB_PARAMS,
                "n_estimators": 650,
                "learning_rate": 0.025,
                "num_leaves": 39,
                "min_child_samples": 24,
                "reg_lambda": 0.85,
            },
            "direct_blend": {"lgb": 0.65, "ridge": 0.35, "specialist": 0.6},
            "allocation_blend": {"revenue": 0.66, "cogs": 0.43},
        },
    ]

    results: list[dict[str, object]] = []
    for candidate in candidates:
        fold_metrics, _ = run_backtest(
            direct_lgb_params=candidate["direct_lgb_params"],
            allocation_lgb_params=candidate["allocation_lgb_params"],
            direct_blend=candidate["direct_blend"],
            allocation_blend=candidate["allocation_blend"],
            fold_names=None if not args.folds else set(args.folds),
        )
        row = {
            "name": candidate["name"],
            **score_fold_metrics(fold_metrics),
            "folds": fold_metrics.to_dict(orient="records"),
            "direct_lgb_params": candidate["direct_lgb_params"],
            "allocation_lgb_params": candidate["allocation_lgb_params"],
            "direct_blend": candidate["direct_blend"],
            "allocation_blend": candidate["allocation_blend"],
        }
        print(json.dumps({"candidate": candidate["name"], "score": row["score"]}), flush=True)
        results.append(row)

    results.sort(key=lambda row: row["score"])
    payload = {
        "best": results[0],
        "results": results,
    }
    with open(REPORTS_DIR / "tuning_direct_pipeline.json", "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
