from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lb_blending import load_submission_frame, make_quarter_shape_candidate, summarize_submission
from src.pipeline import run_training_pipeline


def main() -> None:
    summary = run_training_pipeline(
        allocation_model_type="xgboost",
        submission_filename="submission_xgb_shape_sharefe.csv",
        uncalibrated_submission_filename="submission_xgb_shape_sharefe_uncalibrated.csv",
        model_filename="forecast_pipeline_xgb_shape_sharefe.joblib",
        summary_filename="summary_xgb_shape_sharefe.json",
        fold_metrics_filename="fold_metrics_xgb_shape_sharefe.csv",
        cv_predictions_filename="cv_predictions_xgb_shape_sharefe.csv",
    )

    notebook_v57_path = ROOT / "VinDatathon" / "outputs" / "submission_v57_mp_blend30.csv"
    model_shape_path = ROOT / "outputs" / "submissions" / "submission_xgb_shape_sharefe_uncalibrated.csv"
    out_dir = ROOT / "outputs" / "submissions" / "lb_candidates_xgb_shape_sharefe"
    out_dir.mkdir(parents=True, exist_ok=True)

    notebook_v57 = load_submission_frame(notebook_v57_path)
    model_shape = load_submission_frame(model_shape_path)

    blend_specs = [
        ("candidate_xgb_sharefe_shape62", 0.62, 0.41),
        ("candidate_xgb_sharefe_shape64", 0.64, 0.43),
        ("candidate_xgb_sharefe_shape65", 0.65, 0.44),
        ("candidate_xgb_sharefe_shape66", 0.66, 0.45),
        ("candidate_xgb_sharefe_shape68", 0.68, 0.46),
    ]

    report: dict[str, object] = {
        "summary": summary,
        "anchor_submission": str(notebook_v57_path),
        "shape_submission": str(model_shape_path),
        "candidates": {},
    }

    for name, revenue_weight, cogs_weight in blend_specs:
        frame = make_quarter_shape_candidate(
            notebook_v57,
            model_shape,
            revenue_weight=revenue_weight,
            cogs_weight=cogs_weight,
        )
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        report["candidates"][name] = {
            "path": str(path),
            "weights": {"revenue": revenue_weight, "cogs": cogs_weight},
            "summary": summarize_submission(frame),
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
