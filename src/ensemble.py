from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastEnsemble:
    lgb_model: object
    ridge_model: object
    specialist_model: object | None
    base_weights: dict[str, float]

    def predict(self, X) -> np.ndarray:
        lgb_pred = self.lgb_model.predict(X)
        ridge_pred = self.ridge_model.predict(X)
        base = self.base_weights["lgb"] * lgb_pred + self.base_weights["ridge"] * ridge_pred

        if self.specialist_model is None:
            return base

        specialist_pred = self.specialist_model.predict(X)
        mask = X["specialist_flag"].to_numpy().astype(bool)
        valid = mask & ~np.isnan(specialist_pred)
        if valid.any():
            weight = self.base_weights["specialist"]
            base[valid] = (1.0 - weight) * base[valid] + weight * specialist_pred[valid]
        return base


@dataclass
class QuarterBlendedEnsemble:
    lgb_model: object
    ridge_model: object
    specialist_models: dict[int, object]
    base_weights: dict[str, float]

    def predict(self, X) -> np.ndarray:
        lgb_pred = self.lgb_model.predict(X)
        ridge_pred = self.ridge_model.predict(X)
        base = self.base_weights["lgb"] * lgb_pred + self.base_weights["ridge"] * ridge_pred

        if not self.specialist_models:
            return base

        blended = base.copy()
        quarter_values = X["quarter"].astype(int).to_numpy()
        weight = self.base_weights["specialist"]
        for quarter, model in self.specialist_models.items():
            mask = quarter_values == int(quarter)
            if not mask.any():
                continue
            specialist_pred = model.predict(X.loc[mask])
            blended[mask] = (1.0 - weight) * base[mask] + weight * specialist_pred
        return blended
