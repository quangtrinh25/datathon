from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.config import FOLDS


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def iter_folds() -> Iterable[TimeFold]:
    for fold in FOLDS:
        yield TimeFold(
            name=fold["name"],
            train_end=pd.Timestamp(fold["train_end"]),
            val_start=pd.Timestamp(fold["val_start"]),
            val_end=pd.Timestamp(fold["val_end"]),
        )

