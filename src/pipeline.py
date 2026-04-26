from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.calibration import calibrate_from_cv
from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RATIO_BLEND,
    RATIO_CLIP,
    RATIO_LGB_PARAMS,
    REPORTS_DIR,
    REVENUE_BLEND,
    REVENUE_LGB_PARAMS,
    RIDGE_ALPHA,
    SUBMISSIONS_DIR,
)
from src.data_loader import load_sales, load_sample_submission
from src.ensemble import ForecastEnsemble
from src.explainability import save_feature_importance
from src.feature_engineering import FeatureBuilder
from src.models.lgb_model import LightGBMForecaster
from src.models.q_specialists import BoundarySpecialistForecaster
from src.models.ridge_model import RidgeForecaster
from src.validators import TimeFold, iter_folds


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


@dataclass
class ForecastPipeline:
    use_aux_templates: bool = True
    use_specialist: bool = True
    feature_builder: FeatureBuilder | None = None
    revenue_ensemble: ForecastEnsemble | None = None
    ratio_ensemble: ForecastEnsemble | None = None
    feature_columns: list[str] | None = None
    train_frame_: pd.DataFrame | None = None
    history_sales_: pd.DataFrame | None = None

    def fit(self, history_sales: pd.DataFrame) -> "ForecastPipeline":
        self.history_sales_ = history_sales.sort_values("Date").reset_index(drop=True).copy()
        self.feature_builder = FeatureBuilder().fit(self.history_sales_)
        train_frame = self.feature_builder.make_training_frame(self.history_sales_)
        if not self.use_aux_templates:
            aux_columns = [column for column in train_frame.columns if column.startswith("expected_")]
            train_frame = train_frame.drop(columns=aux_columns)
            self.feature_builder.static_feature_columns = [
                column for column in self.feature_builder.static_feature_columns if not column.startswith("expected_")
            ]
        self.train_frame_ = train_frame
        self.feature_columns = list(self.feature_builder.feature_columns)
        if not self.use_aux_templates:
            self.feature_columns = [column for column in self.feature_columns if not column.startswith("expected_")]

        self.revenue_ensemble = self._fit_target_models(
            train_frame=train_frame,
            target_column="Revenue",
            params=REVENUE_LGB_PARAMS,
            blend=REVENUE_BLEND,
        )
        self.ratio_ensemble = self._fit_target_models(
            train_frame=train_frame,
            target_column="ratio",
            params=RATIO_LGB_PARAMS,
            blend=RATIO_BLEND,
        )
        return self

    def predict(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        if self.feature_builder is None or self.feature_columns is None:
            raise RuntimeError("Pipeline is not fitted.")
        if self.history_sales_ is None:
            raise RuntimeError("Training history is not available.")

        static_future = self.feature_builder.build_static_frame(future_dates[["Date"]])
        if not self.use_aux_templates:
            static_future = static_future[[column for column in static_future.columns if not column.startswith("expected_")]]

        revenue_history = self.history_sales_["Revenue"].astype(float).tolist()
        ratio_history = (self.history_sales_["COGS"] / self.history_sales_["Revenue"]).astype(float).tolist()
        rows: list[dict[str, float]] = []

        for _, date_row in static_future.iterrows():
            feature_row = self.feature_builder.build_dynamic_row(date_row, revenue_history, ratio_history)
            feature_frame = pd.DataFrame([feature_row])[self.feature_columns]
            revenue_pred = float(self.revenue_ensemble.predict(feature_frame)[0])
            ratio_pred = float(self.ratio_ensemble.predict(feature_frame)[0])
            ratio_pred = float(np.clip(ratio_pred, RATIO_CLIP[0], RATIO_CLIP[1]))
            cogs_pred = revenue_pred * ratio_pred

            revenue_history.append(revenue_pred)
            ratio_history.append(ratio_pred)
            rows.append(
                {
                    "Date": pd.Timestamp(date_row["Date"]),
                    "Revenue_pred": revenue_pred,
                    "COGS_pred": cogs_pred,
                    "ratio_pred": ratio_pred,
                }
            )

        return pd.DataFrame(rows)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "ForecastPipeline":
        return joblib.load(path)

    def _fit_target_models(self, train_frame: pd.DataFrame, target_column: str, params: dict, blend: dict) -> ForecastEnsemble:
        X = train_frame[self.feature_columns]
        y = train_frame[target_column]
        weights = self._sample_weights(train_frame)

        lgb_model = LightGBMForecaster(params=params).fit(X, y, sample_weight=weights)
        ridge_model = RidgeForecaster(alpha=RIDGE_ALPHA).fit(X, y, sample_weight=weights)

        specialist_model = None
        if self.use_specialist:
            specialist = BoundarySpecialistForecaster(params=params).fit(
                X,
                y,
                train_frame["specialist_flag"] > 0,
                sample_weight=weights,
            )
            specialist_model = specialist

        return ForecastEnsemble(
            lgb_model=lgb_model,
            ridge_model=ridge_model,
            specialist_model=specialist_model,
            base_weights=blend,
        )

    @staticmethod
    def _sample_weights(train_frame: pd.DataFrame) -> np.ndarray:
        recent_weight = np.where(train_frame["year"] >= 2019, 1.35, 1.0)
        boundary_weight = 1.0 + 0.45 * train_frame["specialist_flag"].to_numpy()
        promo_weight = 1.0 + 0.02 * train_frame["promo_weighted_discount"].to_numpy()
        return recent_weight * boundary_weight * promo_weight


def run_backtest(use_aux_templates: bool = True, use_specialist: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = load_sales()
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in iter_folds():
        train_sales = sales.loc[sales["Date"] <= fold.train_end].reset_index(drop=True)
        val_sales = sales.loc[(sales["Date"] >= fold.val_start) & (sales["Date"] <= fold.val_end)].reset_index(drop=True)

        pipeline = ForecastPipeline(
            use_aux_templates=use_aux_templates,
            use_specialist=use_specialist,
        ).fit(train_sales)
        pred = pipeline.predict(val_sales[["Date"]])
        merged = val_sales.merge(pred, on="Date", how="left")
        merged["fold"] = fold.name
        merged["is_boundary"] = ((merged["Date"].dt.day <= 3) | ((merged["Date"].dt.days_in_month - merged["Date"].dt.day) <= 2)).astype(int)

        revenue_metrics = compute_metrics(merged["Revenue"], merged["Revenue_pred"])
        cogs_metrics = compute_metrics(merged["COGS"], merged["COGS_pred"])
        boundary_metrics = compute_metrics(
            merged.loc[merged["is_boundary"] == 1, "Revenue"],
            merged.loc[merged["is_boundary"] == 1, "Revenue_pred"],
        )

        fold_rows.append(
            {
                "fold": fold.name,
                "train_end": fold.train_end.date().isoformat(),
                "val_start": fold.val_start.date().isoformat(),
                "val_end": fold.val_end.date().isoformat(),
                "revenue_mae": revenue_metrics["mae"],
                "revenue_rmse": revenue_metrics["rmse"],
                "revenue_r2": revenue_metrics["r2"],
                "cogs_mae": cogs_metrics["mae"],
                "cogs_rmse": cogs_metrics["rmse"],
                "cogs_r2": cogs_metrics["r2"],
                "boundary_revenue_mae": boundary_metrics["mae"],
            }
        )
        prediction_frames.append(merged)

    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def fit_final_pipeline(use_aux_templates: bool = True, use_specialist: bool = True) -> ForecastPipeline:
    return ForecastPipeline(
        use_aux_templates=use_aux_templates,
        use_specialist=use_specialist,
    ).fit(load_sales())


def save_processed_features(pipeline: ForecastPipeline) -> None:
    if pipeline.feature_builder is None or pipeline.train_frame_ is None:
        raise RuntimeError("Pipeline is not fitted.")
    sample = load_sample_submission()
    static_future = pipeline.feature_builder.build_static_frame(sample[["Date"]])
    if not pipeline.use_aux_templates:
        static_future = static_future[[column for column in static_future.columns if not column.startswith("expected_")]]

    pipeline.train_frame_.to_parquet(DATA_PROCESSED / "features_train.parquet", index=False)
    static_future.to_parquet(DATA_PROCESSED / "features_test.parquet", index=False)


def generate_submission(
    pipeline: ForecastPipeline,
    calibration: dict[str, float] | None = None,
) -> pd.DataFrame:
    sample = load_sample_submission()
    pred = pipeline.predict(sample[["Date"]])
    submission = sample[["Date"]].copy()
    revenue_scalar = 1.0 if calibration is None else calibration["revenue_scalar"]
    cogs_scalar = 1.0 if calibration is None else calibration["cogs_scalar"]
    submission["Revenue"] = pred["Revenue_pred"] * revenue_scalar
    submission["COGS"] = pred["COGS_pred"] * cogs_scalar
    return submission


def run_training_pipeline(use_aux_templates: bool = True, use_specialist: bool = True) -> dict[str, object]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    fold_metrics, cv_predictions = run_backtest(
        use_aux_templates=use_aux_templates,
        use_specialist=use_specialist,
    )
    calibration = calibrate_from_cv(cv_predictions)

    final_pipeline = fit_final_pipeline(
        use_aux_templates=use_aux_templates,
        use_specialist=use_specialist,
    )
    final_pipeline.save(MODELS_DIR / "forecast_pipeline.joblib")
    save_processed_features(final_pipeline)

    revenue_importance = save_feature_importance(
        final_pipeline.revenue_ensemble.lgb_model,
        final_pipeline.feature_columns,
        REPORTS_DIR / "feature_importance_revenue.csv",
    )
    ratio_importance = save_feature_importance(
        final_pipeline.ratio_ensemble.lgb_model,
        final_pipeline.feature_columns,
        REPORTS_DIR / "feature_importance_ratio.csv",
    )

    submission = generate_submission(final_pipeline, calibration=calibration)
    submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)

    fold_metrics.to_csv(REPORTS_DIR / "fold_metrics.csv", index=False)
    cv_predictions.to_csv(REPORTS_DIR / "cv_predictions.csv", index=False)
    summary = {
        "fold_metrics_mean": fold_metrics.mean(numeric_only=True).to_dict(),
        "calibration": calibration,
        "revenue_top_features": revenue_importance.head(10).to_dict(orient="records"),
        "ratio_top_features": ratio_importance.head(10).to_dict(orient="records"),
        "submission_path": str(SUBMISSIONS_DIR / "submission.csv"),
        "model_path": str(MODELS_DIR / "forecast_pipeline.joblib"),
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2)
    return summary

