from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeForecaster:
    alpha: float
    model: Pipeline | None = None

    def fit(self, X, y, sample_weight=None) -> "RidgeForecaster":
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.model.fit(X, np.log1p(y), ridge__sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return np.clip(np.expm1(self.model.predict(X)), a_min=0.0, a_max=None)

