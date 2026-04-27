from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from xgboost import XGBRegressor


@dataclass
class XGBoostForecaster:
    params: dict[str, Any]
    model: XGBRegressor | None = None

    def fit(self, X, y, sample_weight=None) -> "XGBoostForecaster":
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, np.log1p(y), sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return np.clip(np.expm1(self.model.predict(X)), a_min=0.0, a_max=None)
