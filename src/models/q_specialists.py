from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMRegressor

from src.config import SPECIALIST_MIN_ROWS


@dataclass
class BoundarySpecialistForecaster:
    params: dict
    min_rows: int = SPECIALIST_MIN_ROWS
    model: LGBMRegressor | None = None

    def fit(self, X, y, mask, sample_weight=None) -> "BoundarySpecialistForecaster":
        mask = np.asarray(mask).astype(bool)
        if mask.sum() < self.min_rows:
            self.model = None
            return self
        self.model = LGBMRegressor(**self.params)
        weights = None if sample_weight is None else np.asarray(sample_weight)[mask]
        self.model.fit(X.loc[mask], np.log1p(y.loc[mask]), sample_weight=weights)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), np.nan)
        return np.clip(np.expm1(self.model.predict(X)), a_min=0.0, a_max=None)

