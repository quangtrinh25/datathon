from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lb_blending import build_lb_candidates, load_submission_frame, summarize_submission
from src.pipeline import run_training_pipeline


def main() -> None:
    summary = run_training_pipeline(
        allocation_model_type="xgboost",
        submission_filename="submission_xgb_shape.csv",
        uncalibrated_submission_filename="submission_xgb_shape_uncalibrated.csv",
        model_filename="forecast_pipeline_xgb_shape.joblib",
        summary_filename="summary_xgb_shape.json",
        fold_metrics_filename="fold_metrics_xgb_shape.csv",
        cv_predictions_filename="cv_predictions_xgb_shape.csv",
    )

    notebook_raw_path = ROOT / "VinDatathon" / "outputs" / "submission.csv"
    notebook_v57_path = ROOT / "VinDatathon" / "outputs" / "submission_v57_mp_blend30.csv"
    model_shape_path = ROOT / "outputs" / "submissions" / "submission_xgb_shape_uncalibrated.csv"
    out_dir = ROOT / "outputs" / "submissions" / "lb_candidates_xgb_shape"
    out_dir.mkdir(parents=True, exist_ok=True)

    notebook_raw = load_submission_frame(notebook_raw_path)
    notebook_v57 = load_submission_frame(notebook_v57_path)
    model_shape = load_submission_frame(model_shape_path)
    candidates = build_lb_candidates(
        notebook_raw=notebook_raw,
        notebook_v57=notebook_v57,
        model_shape=model_shape,
    )

    report: dict[str, object] = {
        "summary": summary,
        "anchor_submission": str(notebook_v57_path),
        "shape_submission": str(model_shape_path),
        "candidates": {},
    }
    for name, frame in candidates.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        report["candidates"][name] = {
            "path": str(path),
            "summary": summarize_submission(frame),
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
