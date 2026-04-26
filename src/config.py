from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

RANDOM_SEED = 42

TRAIN_START = "2012-07-04"
TRAIN_END = "2022-12-31"
FORECAST_START = "2023-01-01"
FORECAST_END = "2024-07-01"

LAGS = (1, 7, 14, 28, 56, 91, 182, 364)
ROLLING_WINDOWS = (7, 28, 56)

FOLDS = (
    {
        "name": "fold_2021",
        "train_end": "2020-12-31",
        "val_start": "2021-01-01",
        "val_end": "2021-12-31",
    },
    {
        "name": "fold_2021h2_2022h1",
        "train_end": "2021-06-30",
        "val_start": "2021-07-01",
        "val_end": "2022-06-30",
    },
    {
        "name": "fold_2022",
        "train_end": "2021-12-31",
        "val_start": "2022-01-01",
        "val_end": "2022-12-31",
    },
)

TET_DATES = {
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
}

VN_FIXED_HOLIDAYS = {
    "new_year": (1, 1),
    "womens_day": (3, 8),
    "reunification": (4, 30),
    "labor_day": (5, 1),
    "national_day": (9, 2),
    "vn_womens_day": (10, 20),
    "double_1111": (11, 11),
    "double_1212": (12, 12),
    "christmas_eve": (12, 24),
    "christmas": (12, 25),
}

REVENUE_LGB_PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 24,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "reg_lambda": 0.5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

RATIO_LGB_PARAMS = {
    "objective": "regression",
    "n_estimators": 350,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "reg_lambda": 0.25,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

RIDGE_ALPHA = 3.0

REVENUE_BLEND = {
    "lgb": 0.8,
    "ridge": 0.2,
    "specialist": 0.35,
}

RATIO_BLEND = {
    "lgb": 0.75,
    "ridge": 0.25,
    "specialist": 0.25,
}

DIRECT_BLEND = {
    "lgb": 0.6,
    "ridge": 0.4,
    "specialist": 0.6,
}

ALLOCATION_BLEND = {
    "revenue": 0.6,
    "cogs": 0.4,
}

SPECIALIST_MIN_ROWS = 180
DIRECT_SPECIALIST_BOOST = 2.0
HIGH_ERA_START = 2014
HIGH_ERA_END = 2018
HIGH_ERA_WEIGHT = 1.0
BASE_ERA_WEIGHT = 0.01
RATIO_CLIP = (0.65, 1.30)
CALIBRATION_GRID = tuple(round(x, 3) for x in [0.90 + i * 0.01 for i in range(41)])
CALIBRATION_MIN_SEGMENT_ROWS = 45
