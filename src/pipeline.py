from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.calibration import apply_calibration, calibrate_from_cv
from src.config import (
    ALLOCATION_BLEND,
    BASE_ERA_WEIGHT,
    DATA_PROCESSED,
    DIRECT_BLEND,
    DIRECT_SPECIALIST_BOOST,
    HIGH_ERA_END,
    HIGH_ERA_START,
    HIGH_ERA_WEIGHT,
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
from src.ensemble import ForecastEnsemble, QuarterBlendedEnsemble
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
    direct_revenue_ensemble: QuarterBlendedEnsemble | None = None
    direct_cogs_ensemble: QuarterBlendedEnsemble | None = None
    revenue_allocation_ensemble: QuarterBlendedEnsemble | None = None
    cogs_allocation_ensemble: QuarterBlendedEnsemble | None = None
    feature_columns: list[str] | None = None
    static_feature_columns: list[str] | None = None
    train_frame_: pd.DataFrame | None = None
    history_sales_: pd.DataFrame | None = None

    def fit(self, history_sales: pd.DataFrame) -> "ForecastPipeline":
        self.history_sales_ = history_sales.sort_values("Date").reset_index(drop=True).copy()
        self.feature_builder = FeatureBuilder().fit(self.history_sales_)
        static_train_frame = self.feature_builder.build_static_frame(self.history_sales_[["Date"]]).merge(
            self.history_sales_,
            on="Date",
            how="left",
        )
        if not self.use_aux_templates:
            static_aux_columns = [column for column in static_train_frame.columns if column.startswith("expected_")]
            static_train_frame = static_train_frame.drop(columns=static_aux_columns)
            self.feature_builder.static_feature_columns = [
                column for column in self.feature_builder.static_feature_columns if not column.startswith("expected_")
            ]
        self.train_frame_ = static_train_frame
        self.feature_columns = []
        self.static_feature_columns = list(self.feature_builder.static_feature_columns)
        if not self.use_aux_templates:
            self.static_feature_columns = [
                column for column in self.static_feature_columns if not column.startswith("expected_")
            ]

        self.revenue_ensemble = None
        self.ratio_ensemble = None
        self.direct_revenue_ensemble = self._fit_direct_target_models(static_train_frame, "Revenue")
        self.direct_cogs_ensemble = self._fit_direct_target_models(static_train_frame, "COGS")
        self.revenue_allocation_ensemble = self._fit_direct_allocation_models(static_train_frame, "Revenue")
        self.cogs_allocation_ensemble = self._fit_direct_allocation_models(static_train_frame, "COGS")
        return self

    def predict(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        if self.feature_builder is None or self.static_feature_columns is None:
            raise RuntimeError("Pipeline is not fitted.")
        if self.history_sales_ is None:
            raise RuntimeError("Training history is not available.")

        static_future = self.feature_builder.build_static_frame(future_dates[["Date"]])
        if not self.use_aux_templates:
            static_future = static_future[[column for column in static_future.columns if not column.startswith("expected_")]]

        if self.direct_revenue_ensemble is not None and self.direct_cogs_ensemble is not None:
            feature_frame = static_future[self.static_feature_columns]
            revenue_anchor = self.direct_revenue_ensemble.predict(feature_frame)
            cogs_anchor = self.direct_cogs_ensemble.predict(feature_frame)
            revenue_pred = revenue_anchor
            cogs_pred = cogs_anchor
            if self.revenue_allocation_ensemble is not None:
                revenue_share = self.revenue_allocation_ensemble.predict(feature_frame)
                revenue_pred = self._blend_with_allocation_head(static_future, revenue_anchor, revenue_share, "revenue")
            if self.cogs_allocation_ensemble is not None:
                cogs_share = self.cogs_allocation_ensemble.predict(feature_frame)
                cogs_pred = self._blend_with_allocation_head(static_future, cogs_anchor, cogs_share, "cogs")
            ratio_pred = np.divide(
                cogs_pred,
                np.clip(revenue_pred, a_min=1.0, a_max=None),
            )
            return pd.DataFrame(
                {
                    "Date": static_future["Date"].astype("datetime64[ns]"),
                    "Revenue_pred": revenue_pred,
                    "COGS_pred": cogs_pred,
                    "ratio_pred": ratio_pred,
                }
            )

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

    def _fit_direct_target_models(self, train_frame: pd.DataFrame, target_column: str) -> QuarterBlendedEnsemble:
        if self.static_feature_columns is None:
            raise RuntimeError("Static feature columns are not available.")

        X = train_frame[self.static_feature_columns]
        y = train_frame[target_column]
        weights = self._direct_sample_weights(train_frame)
        lgb_model = LightGBMForecaster(params=REVENUE_LGB_PARAMS).fit(X, y, sample_weight=weights)
        ridge_model = RidgeForecaster(alpha=RIDGE_ALPHA).fit(X, y)

        specialist_models: dict[int, LightGBMForecaster] = {}
        for quarter in [1, 2, 3, 4]:
            quarter_weights = weights.copy()
            quarter_weights[train_frame["quarter"].to_numpy() == quarter] *= DIRECT_SPECIALIST_BOOST
            specialist_models[quarter] = LightGBMForecaster(params=REVENUE_LGB_PARAMS).fit(
                X,
                y,
                sample_weight=quarter_weights,
            )

        return QuarterBlendedEnsemble(
            lgb_model=lgb_model,
            ridge_model=ridge_model,
            specialist_models=specialist_models,
            base_weights=DIRECT_BLEND,
        )

    def _fit_direct_allocation_models(self, train_frame: pd.DataFrame, target_column: str) -> QuarterBlendedEnsemble:
        if self.static_feature_columns is None:
            raise RuntimeError("Static feature columns are not available.")

        X = train_frame[self.static_feature_columns]
        y = self._quarter_share_target(train_frame, target_column)
        weights = self._direct_sample_weights(train_frame)
        lgb_model = LightGBMForecaster(params=REVENUE_LGB_PARAMS).fit(X, y, sample_weight=weights)
        ridge_model = RidgeForecaster(alpha=RIDGE_ALPHA).fit(X, y)

        specialist_models: dict[int, LightGBMForecaster] = {}
        for quarter in [1, 2, 3, 4]:
            quarter_weights = weights.copy()
            quarter_weights[train_frame["quarter"].to_numpy() == quarter] *= DIRECT_SPECIALIST_BOOST
            specialist_models[quarter] = LightGBMForecaster(params=REVENUE_LGB_PARAMS).fit(
                X,
                y,
                sample_weight=quarter_weights,
            )

        return QuarterBlendedEnsemble(
            lgb_model=lgb_model,
            ridge_model=ridge_model,
            specialist_models=specialist_models,
            base_weights=DIRECT_BLEND,
        )

    @staticmethod
    def _sample_weights(train_frame: pd.DataFrame) -> np.ndarray:
        recent_weight = np.where(train_frame["year"] >= 2019, 1.35, 1.0)
        boundary_weight = 1.0 + 0.45 * train_frame["specialist_flag"].to_numpy()
        promo_weight = 1.0 + 0.02 * train_frame["promo_weighted_discount"].to_numpy()
        return recent_weight * boundary_weight * promo_weight

    @staticmethod
    def _direct_sample_weights(train_frame: pd.DataFrame) -> np.ndarray:
        years = train_frame["year"].to_numpy()
        weights = np.full(len(train_frame), BASE_ERA_WEIGHT, dtype=float)
        high_era_mask = (years >= HIGH_ERA_START) & (years <= HIGH_ERA_END)
        weights[high_era_mask] = HIGH_ERA_WEIGHT
        return weights

    @staticmethod
    def _quarter_share_target(train_frame: pd.DataFrame, target_column: str) -> pd.Series:
        quarter_total = train_frame.groupby(["year", "quarter"])[target_column].transform("sum")
        share = train_frame[target_column] / quarter_total.replace(0.0, np.nan)
        return share.fillna(0.0)

    @staticmethod
    def _blend_with_allocation_head(
        future_frame: pd.DataFrame,
        anchor_pred: np.ndarray,
        share_pred: np.ndarray,
        target_name: str,
    ) -> np.ndarray:
        group_keys = [future_frame["year"], future_frame["quarter"]]
        anchor_series = pd.Series(anchor_pred, index=future_frame.index, dtype=float)
        anchor_total = anchor_series.groupby(group_keys).transform("sum")
        anchor_share = anchor_series / anchor_total.replace(0.0, np.nan)

        share_series = pd.Series(np.clip(share_pred, a_min=0.0, a_max=None), index=future_frame.index, dtype=float)
        share_total = share_series.groupby(group_keys).transform("sum")
        normalized_share = share_series / share_total.replace(0.0, np.nan)
        normalized_share = normalized_share.fillna(anchor_share).fillna(0.0)

        allocation_pred = anchor_total * normalized_share
        blend_weight = float(ALLOCATION_BLEND[target_name])
        blended = (1.0 - blend_weight) * anchor_series + blend_weight * allocation_pred
        return blended.to_numpy()


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


def summarize_cv_predictions(
    predictions: pd.DataFrame,
    revenue_prediction_column: str = "Revenue_pred",
    cogs_prediction_column: str = "COGS_pred",
) -> pd.DataFrame:
    fold_rows: list[dict[str, object]] = []
    for fold_name, fold_frame in predictions.groupby("fold", sort=False):
        revenue_metrics = compute_metrics(fold_frame["Revenue"], fold_frame[revenue_prediction_column])
        cogs_metrics = compute_metrics(fold_frame["COGS"], fold_frame[cogs_prediction_column])
        boundary_frame = fold_frame.loc[fold_frame["is_boundary"] == 1]
        boundary_metrics = compute_metrics(
            boundary_frame["Revenue"],
            boundary_frame[revenue_prediction_column],
        )
        fold_rows.append(
            {
                "fold": fold_name,
                "revenue_mae": revenue_metrics["mae"],
                "revenue_rmse": revenue_metrics["rmse"],
                "revenue_r2": revenue_metrics["r2"],
                "cogs_mae": cogs_metrics["mae"],
                "cogs_rmse": cogs_metrics["rmse"],
                "cogs_r2": cogs_metrics["r2"],
                "boundary_revenue_mae": boundary_metrics["mae"],
            }
        )
    return pd.DataFrame(fold_rows)


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
    calibration: dict[str, object] | None = None,
) -> pd.DataFrame:
    sample = load_sample_submission()
    pred = pipeline.predict(sample[["Date"]])
    if calibration is not None:
        pred = apply_calibration(pred, calibration)
    submission = sample[["Date"]].copy()
    if calibration is None:
        submission["Revenue"] = pred["Revenue_pred"]
        submission["COGS"] = pred["COGS_pred"]
    else:
        submission["Revenue"] = pred["Revenue_pred_calibrated"]
        submission["COGS"] = pred["COGS_pred_calibrated"]
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
    calibrated_cv_predictions = apply_calibration(cv_predictions, calibration)
    calibrated_fold_metrics = summarize_cv_predictions(
        calibrated_cv_predictions,
        revenue_prediction_column="Revenue_pred_calibrated",
        cogs_prediction_column="COGS_pred_calibrated",
    )

    final_pipeline = fit_final_pipeline(
        use_aux_templates=use_aux_templates,
        use_specialist=use_specialist,
    )
    final_pipeline.save(MODELS_DIR / "forecast_pipeline.joblib")
    save_processed_features(final_pipeline)

    revenue_importance = save_feature_importance(
        final_pipeline.direct_revenue_ensemble.lgb_model,
        final_pipeline.static_feature_columns,
        REPORTS_DIR / "feature_importance_revenue.csv",
    )
    cogs_importance = save_feature_importance(
        final_pipeline.direct_cogs_ensemble.lgb_model,
        final_pipeline.static_feature_columns,
        REPORTS_DIR / "feature_importance_cogs.csv",
    )

    uncalibrated_submission = generate_submission(final_pipeline, calibration=None)
    uncalibrated_submission.to_csv(SUBMISSIONS_DIR / "submission_uncalibrated.csv", index=False)

    submission = generate_submission(final_pipeline, calibration=calibration)
    submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)

    fold_metrics.to_csv(REPORTS_DIR / "fold_metrics.csv", index=False)
    cv_predictions.to_csv(REPORTS_DIR / "cv_predictions.csv", index=False)
    summary = {
        "fold_metrics_mean": fold_metrics.mean(numeric_only=True).to_dict(),
        "calibrated_fold_metrics_mean": calibrated_fold_metrics.mean(numeric_only=True).to_dict(),
        "calibration": calibration,
        "revenue_top_features": revenue_importance.head(10).to_dict(orient="records"),
        "cogs_top_features": cogs_importance.head(10).to_dict(orient="records"),
        "submission_path": str(SUBMISSIONS_DIR / "submission.csv"),
        "model_path": str(MODELS_DIR / "forecast_pipeline.joblib"),
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2)
    return summary
