from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor


@dataclass
class LightGBMForecaster:
    params: dict[str, Any]
    model: LGBMRegressor | None = None

    def fit(self, X, y, sample_weight=None) -> "LightGBMForecaster":
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X, np.log1p(y), sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return np.clip(np.expm1(self.model.predict(X)), a_min=0.0, a_max=None)

    def feature_importance(self, columns: list[str]):
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return {
            "feature": columns,
            "importance": self.model.feature_importances_,
        }

