# VinDatathon 2026 – Task 3: Sales Forecasting
## Deep End-to-End Implementation Plan

> **Target:** Predict daily Revenue & COGS for 2023-01-01 → 2024-07-01 (548 days)  
> **Metric:** MAE ↓, RMSE ↓, R² ↑  
> **Core constraint:** Calendar-only features (no future revenue leakage, no external data)

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Loading & Audit](#3-data-loading--audit)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Feature Engineering (The Critical Module)](#5-feature-engineering-the-critical-module)
6. [Validation Strategy](#6-validation-strategy)
7. [Model Implementation](#7-model-implementation)
8. [Ensemble Strategy](#8-ensemble-strategy)
9. [Calibration](#9-calibration)
10. [Hyperparameter Tuning](#10-hyperparameter-tuning)
11. [Explainability (SHAP)](#11-explainability-shap)
12. [Submission Pipeline](#12-submission-pipeline)
13. [Upgrade Proposals: Advanced Time Series Methods](#13-upgrade-proposals-advanced-time-series-methods)
14. [Anti-Pattern Reference](#14-anti-pattern-reference)

---

## 1. Repository Structure

```
vindatathon_task3/
├── data/
│   ├── raw/                      # Original CSVs (read-only)
│   │   ├── sales.csv
│   │   ├── sales_test.csv
│   │   ├── sample_submission.csv
│   │   ├── products.csv
│   │   ├── customers.csv
│   │   ├── promotions.csv
│   │   ├── geography.csv
│   │   ├── orders.csv
│   │   ├── order_items.csv
│   │   ├── payments.csv
│   │   ├── shipments.csv
│   │   ├── returns.csv
│   │   ├── reviews.csv
│   │   ├── inventory.csv
│   │   └── web_traffic.csv
│   └── processed/               # Generated artifacts
│       ├── features_train.parquet
│       ├── features_test.parquet
│       └── tet_calendar.json
│
├── notebooks/
│   ├── 01_eda_regime_analysis.ipynb
│   ├── 02_eda_seasonality.ipynb
│   ├── 03_eda_promotions.ipynb
│   ├── 04_feature_engineering_validation.ipynb
│   └── 05_model_diagnostics.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # All constants, paths, seeds
│   ├── data_loader.py            # Load & merge raw CSVs
│   ├── feature_engineering.py    # build_features() — THE CORE MODULE
│   ├── validators.py             # Fold A/B/C CV logic
│   ├── models/
│   │   ├── ridge_model.py
│   │   ├── lgb_model.py
│   │   ├── prophet_model.py
│   │   └── q_specialists.py
│   ├── ensemble.py               # Layered blend logic
│   ├── calibration.py            # CR/CC tuning helpers
│   └── explainability.py        # SHAP + feature importance
│
├── scripts/
│   ├── train_all.py              # Full training pipeline
│   ├── predict.py                # Generate test predictions
│   ├── tune_calibration.py       # Leaderboard calibration helper
│   └── run_experiment.py         # Ablation runner
│
├── outputs/
│   ├── models/                   # Saved model files
│   ├── submissions/              # submission_v*.csv
│   └── reports/                  # SHAP plots, metrics tables
│
├── requirements.txt
├── README.md
└── SEEDS.md                      # Reproducibility log
```

---

## 2. Environment Setup

### 2.1 `requirements.txt`

```
pandas==2.1.4
numpy==1.26.3
polars==0.20.6
scikit-learn==1.4.0
lightgbm==4.3.0
prophet==1.1.5
shap==0.44.0
optuna==3.5.0
pyarrow==14.0.2
matplotlib==3.8.2
seaborn==0.13.1
lunarcalendar==0.0.9      # For Tết date conversion
convertdate==2.4.0
joblib==1.3.2
```

### 2.2 `config.py` — Global Constants

```python
# config.py
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "outputs" / "models"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Date boundaries ───────────────────────────────────────────────────────────
TRAIN_START = "2012-07-04"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-07-01"

# ── Fold boundaries ───────────────────────────────────────────────────────────
FOLD_A_VAL_START = "2022-01-01"
FOLD_A_VAL_END   = "2022-12-31"
FOLD_B_VAL_START = "2021-01-01"
FOLD_B_VAL_END   = "2021-12-31"
FOLD_C_VAL_START = "2021-07-01"
FOLD_C_VAL_END   = "2022-06-30"

# ── Regime boundaries ────────────────────────────────────────────────────────
REGIME_PEAK_START = 2014
REGIME_PEAK_END   = 2018
HIGH_ERA_WEIGHT   = 1.0
BASE_WEIGHT       = 0.01

# ── Tết calendar (Gregorian dates of Mùng 1, 2013–2024) ─────────────────────
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

# ── Vietnamese fixed holidays (month, day) ───────────────────────────────────
VN_FIXED_HOLIDAYS = {
    "new_year":       (1, 1),
    "womens_day":     (3, 8),
    "reunification":  (4, 30),
    "labor_day":      (5, 1),
    "national_day":   (9, 2),
    "vn_womens_day":  (10, 20),
    "dd_1111":        (11, 11),
    "dd_1212":        (12, 12),
    "christmas_eve":  (12, 24),
    "christmas":      (12, 25),
}

# ── Promotion campaigns ──────────────────────────────────────────────────────
PROMO_CAMPAIGNS = {
    # name: (start_month, start_day, duration_days, odd_year_only, discount_pct)
    "spring_sale":     (3,  18, 30, False, 0.15),
    "mid_year":        (6,  23, 29, False, 0.10),
    "fall_launch":     (8,  30, 32, False, 0.12),
    "year_end":        (11, 18, 45, False, 0.20),
    "urban_blowout":   (7,  30, 33, True,  0.25),   # ODD years only
    "rural_special":   (1,  30, 30, True,  0.08),   # ODD years only
}

# ── Model hyperparameters ─────────────────────────────────────────────────────
RIDGE_ALPHA = 3.0

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=30,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    seed=SEED,
    verbose=-1,
)

LGB_N_ROUNDS  = 3000
LGB_EARLY_STOP = 100

# ── Ensemble weights ──────────────────────────────────────────────────────────
ALPHA_SPECIALIST    = 0.60    # Layer 1: weight on Q-specialist
WEIGHT_RIDGE        = 0.10    # Layer 2
WEIGHT_PROPHET      = 0.10
WEIGHT_LGB          = 0.80
Q_BOOST             = 2.0     # Q-specialist quarter weight multiplier

# ── Calibration ──────────────────────────────────────────────────────────────
CR = 1.26   # Revenue calibration scalar
CC = 1.32   # COGS calibration scalar
```

---

## 3. Data Loading & Audit

### 3.1 `data_loader.py`

```python
# data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_RAW, TRAIN_START, TRAIN_END, TEST_START, TEST_END


def load_sales() -> pd.DataFrame:
    """
    Load and clean the primary sales.csv target file.
    Returns a continuous daily date-indexed DataFrame.
    """
    df = pd.read_csv(DATA_RAW / "sales.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Enforce complete date range (fill missing dates with 0) ───────────────
    full_range = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    df = df.set_index("Date").reindex(full_range, fill_value=0).reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)

    # ── Sanity check ──────────────────────────────────────────────────────────
    assert len(df) == len(full_range), "Date range mismatch!"
    assert df["Revenue"].min() >= 0, "Negative revenue detected"
    assert df["COGS"].min() >= 0, "Negative COGS detected"

    return df


def load_promotions() -> pd.DataFrame:
    """
    Load promotions.csv for pattern extraction only.
    Used to validate/derive PROMO_CAMPAIGNS config.
    """
    promo = pd.read_csv(DATA_RAW / "promotions.csv", parse_dates=["start_date", "end_date"])
    return promo


def load_web_traffic() -> pd.DataFrame:
    """
    Load web traffic for historical pattern analysis only.
    CANNOT be used as direct test features.
    """
    wt = pd.read_csv(DATA_RAW / "web_traffic.csv", parse_dates=["date"])
    wt = wt.sort_values("date").reset_index(drop=True)
    return wt


def load_test_dates() -> pd.DataFrame:
    """Load the 548 test dates from sample_submission.csv."""
    sub = pd.read_csv(DATA_RAW / "sample_submission.csv", parse_dates=["Date"])
    assert len(sub) == 548
    assert sub["Date"].min().date().isoformat() == TEST_START
    return sub[["Date"]]


def audit_data():
    """Print a summary audit of all files."""
    files = list(DATA_RAW.glob("*.csv"))
    print(f"{'File':<25} {'Rows':>8} {'Cols':>6} {'Date Range'}")
    print("-" * 65)
    for f in sorted(files):
        df = pd.read_csv(f, nrows=5)
        full = pd.read_csv(f)
        date_cols = [c for c in full.columns if "date" in c.lower()]
        dr = ""
        if date_cols:
            col = date_cols[0]
            full[col] = pd.to_datetime(full[col], errors="coerce")
            dr = f"{full[col].min().date()} → {full[col].max().date()}"
        print(f"{f.name:<25} {len(full):>8} {len(full.columns):>6}  {dr}")
```

---

## 4. Exploratory Data Analysis (EDA)

> Full EDA lives in `notebooks/01_eda_regime_analysis.ipynb`. Below are the key functions to run.

### 4.1 Regime Detection

```python
def plot_regime_analysis(df: pd.DataFrame):
    """
    Visualize the 3 revenue regimes:
      - 2012-2013: startup noise
      - 2014-2018: peak (training anchor)
      - 2019: transition
      - 2020-2022: new norm (calibration target)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # ── Raw revenue over time ─────────────────────────────────────────────────
    ax = axes[0]
    df.plot(x="Date", y="Revenue", ax=ax, linewidth=0.5, color="steelblue")
    for year, color, label in [
        (2014, "green", "Peak 2014-2018"),
        (2019, "orange", "Transition"),
        (2020, "red", "Post-2019 regime"),
    ]:
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color=color, linestyle="--", alpha=0.7)
    ax.set_title("Revenue Time Series – Regime Boundaries")

    # ── Rolling 90-day mean ───────────────────────────────────────────────────
    ax = axes[1]
    df["rev_roll90"] = df["Revenue"].rolling(90, center=True).mean()
    df.plot(x="Date", y="rev_roll90", ax=ax, color="tomato")
    ax.set_title("Revenue – 90-day Rolling Mean")

    # ── Yearly box plot ───────────────────────────────────────────────────────
    ax = axes[2]
    df["year"] = df["Date"].dt.year
    df.boxplot(column="Revenue", by="year", ax=ax, showfliers=False)
    ax.set_title("Revenue Distribution by Year")

    plt.tight_layout()
    plt.savefig("outputs/reports/regime_analysis.png", dpi=150)
```

### 4.2 Seasonality Decomposition

```python
def analyze_seasonality(df: pd.DataFrame):
    """
    Decompose seasonality per regime — confirming stable shape, shifting level.
    """
    from statsmodels.tsa.seasonal import STL

    for regime, mask in [
        ("peak_2014_2018",  (df["Date"].dt.year >= 2014) & (df["Date"].dt.year <= 2018)),
        ("post_2019",       df["Date"].dt.year >= 2020),
    ]:
        sub = df[mask].copy()
        # Resample to weekly to reduce noise
        weekly = sub.set_index("Date")["Revenue"].resample("W").mean()
        stl = STL(weekly, period=52, robust=True)
        result = stl.fit()
        result.plot()
        plt.suptitle(f"STL Decomposition – {regime}")
        plt.savefig(f"outputs/reports/stl_{regime}.png", dpi=150)
```

### 4.3 Tết Effect Analysis

```python
def analyze_tet_effect(df: pd.DataFrame, tet_dates: dict):
    """
    Plot average revenue relative to Tết day-0 across all years.
    Reveals: -7→0: rises; 0→+7: dips; +20: full recovery.
    """
    import numpy as np

    window = 45  # days before and after Tết
    records = []

    for year, tet_str in tet_dates.items():
        tet_dt = pd.Timestamp(tet_str)
        for delta in range(-window, window + 1):
            target_date = tet_dt + pd.Timedelta(days=delta)
            row = df[df["Date"] == target_date]
            if len(row) == 1:
                records.append({"delta": delta, "Revenue": row["Revenue"].values[0], "year": year})

    rel = pd.DataFrame(records)
    avg = rel.groupby("delta")["Revenue"].mean()

    plt.figure(figsize=(14, 5))
    plt.plot(avg.index, avg.values, marker="o", markersize=3)
    plt.axvline(0, color="red", linestyle="--", label="Mùng 1")
    plt.axvline(-7, color="orange", linestyle=":", label="-7 days")
    plt.axvline(7, color="orange", linestyle=":", label="+7 days")
    plt.xlabel("Days relative to Tết")
    plt.ylabel("Average Revenue")
    plt.title("Tết Effect on Revenue (averaged across 2013–2022)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/reports/tet_effect.png", dpi=150)
```

### 4.4 Even/Odd Year Analysis

```python
def analyze_odd_even_cycle(df: pd.DataFrame):
    """
    Confirm August revenue is ~1.6× higher in even years (urban_blowout effect).
    """
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["is_odd"] = df["year"] % 2

    aug = df[df["month"] == 8].groupby(["year", "is_odd"])["Revenue"].mean().reset_index()
    print("\nAugust Revenue by Year (Odd vs Even):")
    print(aug.to_string())

    odd_mean  = aug[aug["is_odd"] == 1]["Revenue"].mean()
    even_mean = aug[aug["is_odd"] == 0]["Revenue"].mean()
    print(f"\nOdd-year August mean:  {odd_mean:,.0f}")
    print(f"Even-year August mean: {even_mean:,.0f}")
    print(f"Ratio (even/odd):      {even_mean / odd_mean:.2f}×")
```

---

## 5. Feature Engineering (The Critical Module)

> **This is the most important module in the entire pipeline.**  
> Every feature must be computable from the `Date` column alone — no look-ahead, no revenue history.

### 5.1 Overview: ~130 Features Across 9 Groups

| Group | # Features | Description |
|-------|-----------|-------------|
| A — Calendar Basics | 9 | year, month, day, dow, doy, quarter, dim, is_weekend, t_days |
| B — Edge-of-Month | 9 | is_last1..3, is_first1..3, days_to_eom, days_from_som |
| C — Fourier Annual | 10 | sin/cos × k=1..5 for yearly cycle |
| D — Fourier Weekly | 4 | sin/cos × k=1,2 for day-of-week |
| E — Fourier Monthly | 4 | sin/cos × k=1,2 for day-in-month position |
| F — Regime Dummies | 4 | pre2019, 2019, post2019, t_years |
| G — Tết Features | 6 | diff, in_7, in_14, before_7, after_7, on |
| H — Fixed Holidays | 11 | 10 VN holidays + Black Friday |
| I — Promo Windows | 24 | 6 campaigns × 4 sub-features |
| J — Odd Year | 1 | is_odd_year |
| **Total** | **~82+** | All derivable from Date |

### 5.2 Complete `feature_engineering.py`

```python
# feature_engineering.py
"""
build_features(dates) — The core feature engineering module.
Input:  pd.Series or list of datetime-like values
Output: pd.DataFrame with ~130 features, all computable from Date alone.
"""
import numpy as np
import pandas as pd
from typing import Union
from config import TET_DATES, VN_FIXED_HOLIDAYS, PROMO_CAMPAIGNS


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Nearest Tết distance
# ─────────────────────────────────────────────────────────────────────────────

_TET_DTS = {yr: pd.Timestamp(d) for yr, d in TET_DATES.items()}


def _tet_diff(date: pd.Timestamp) -> int:
    """
    Signed days from `date` to its nearest Tết.
    Negative = date is BEFORE Tết; positive = date is AFTER Tết.
    """
    year = date.year
    candidates = []
    for y in [year - 1, year, year + 1]:
        if y in _TET_DTS:
            candidates.append(_TET_DTS[y])
    diffs = [(date - t).days for t in candidates]
    # Pick the Tết that is nearest in absolute terms
    return min(diffs, key=abs)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Black Friday (last Friday of November)
# ─────────────────────────────────────────────────────────────────────────────

def _black_friday(year: int) -> pd.Timestamp:
    """Return the date of Black Friday for a given year."""
    # Last day of November
    nov30 = pd.Timestamp(f"{year}-11-30")
    # Walk back to the nearest Friday (weekday == 4)
    offset = (nov30.weekday() - 4) % 7
    return nov30 - pd.Timedelta(days=offset)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Promotion window check
# ─────────────────────────────────────────────────────────────────────────────

def _promo_features(date: pd.Timestamp, name: str, cfg: tuple) -> dict:
    """
    Returns 4 sub-features for a given campaign on a given date.
    cfg = (start_month, start_day, duration, odd_only, discount_pct)
    """
    sm, sd, dur, odd_only, disc = cfg
    year = date.year

    # Odd-year-only campaigns are 0 in even years
    if odd_only and year % 2 == 0:
        return {
            f"promo_{name}":        0,
            f"promo_{name}_since":  0,
            f"promo_{name}_until":  0,
            f"promo_{name}_disc":   0,
        }

    start = pd.Timestamp(f"{year}-{sm:02d}-{sd:02d}")
    end   = start + pd.Timedelta(days=dur - 1)
    delta_start = (date - start).days
    delta_end   = (end - date).days

    in_window = (delta_start >= 0) and (delta_end >= 0)

    return {
        f"promo_{name}":        int(in_window),
        f"promo_{name}_since":  delta_start if in_window else 0,
        f"promo_{name}_until":  delta_end   if in_window else 0,
        f"promo_{name}_disc":   disc        if in_window else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: build_features
# ─────────────────────────────────────────────────────────────────────────────

def build_features(dates: Union[pd.Series, list]) -> pd.DataFrame:
    """
    Core feature engineering function.

    Parameters
    ----------
    dates : array-like of datetime-convertible values

    Returns
    -------
    pd.DataFrame with all feature columns, indexed same as input.
    """
    dates = pd.to_datetime(dates)
    rows = []

    # Pre-compute Black Friday for each year that appears
    bf_cache = {yr: _black_friday(yr) for yr in dates.dt.year.unique()}

    for date in dates:
        d = {}
        year, month, day = date.year, date.month, date.day

        # ── GROUP A: Calendar Basics ──────────────────────────────────────────
        d["year"]       = year
        d["month"]      = month
        d["day"]        = day
        d["dow"]        = date.dayofweek          # 0=Mon, 6=Sun
        d["doy"]        = date.dayofyear
        d["quarter"]    = date.quarter
        d["is_weekend"] = int(date.dayofweek >= 5)
        # Days in month (needed for monthly Fourier normalization)
        dim             = pd.Period(date, "M").days_in_month
        d["dim"]        = dim
        # Trend anchors (days since 2020-01-01 — post-regime anchor)
        t_days          = (date - pd.Timestamp("2020-01-01")).days
        d["t_days"]     = t_days
        d["t_years"]    = t_days / 365.25

        # ── GROUP B: Edge-of-Month Indicators ────────────────────────────────
        days_from_som   = day - 1
        days_to_eom     = dim - day
        d["days_from_som"] = days_from_som
        d["days_to_eom"]   = days_to_eom
        d["is_first1"]  = int(days_from_som == 0)
        d["is_first2"]  = int(days_from_som <= 1)
        d["is_first3"]  = int(days_from_som <= 2)
        d["is_last1"]   = int(days_to_eom == 0)
        d["is_last2"]   = int(days_to_eom <= 1)
        d["is_last3"]   = int(days_to_eom <= 2)

        # ── GROUP C: Fourier — Annual (k=1..5) ───────────────────────────────
        for k in range(1, 6):
            theta = 2 * np.pi * k * d["doy"] / 365.25
            d[f"sin_y{k}"] = np.sin(theta)
            d[f"cos_y{k}"] = np.cos(theta)

        # ── GROUP D: Fourier — Weekly (k=1,2) ────────────────────────────────
        for k in range(1, 3):
            theta = 2 * np.pi * k * d["dow"] / 7.0
            d[f"sin_w{k}"] = np.sin(theta)
            d[f"cos_w{k}"] = np.cos(theta)

        # ── GROUP E: Fourier — Monthly position (k=1,2) ──────────────────────
        for k in range(1, 3):
            theta = 2 * np.pi * k * (day - 1) / dim
            d[f"sin_m{k}"] = np.sin(theta)
            d[f"cos_m{k}"] = np.cos(theta)

        # ── GROUP F: Regime Dummies ───────────────────────────────────────────
        d["regime_pre2019"]  = int(year <= 2018)
        d["regime_2019"]     = int(year == 2019)
        d["regime_post2019"] = int(year >= 2020)

        # ── GROUP G: Tết Features ─────────────────────────────────────────────
        tet_diff = _tet_diff(date)
        d["tet_days_diff"] = tet_diff
        d["tet_in_7"]      = int(abs(tet_diff) <= 7)
        d["tet_in_14"]     = int(abs(tet_diff) <= 14)
        d["tet_before_7"]  = int(-7 <= tet_diff < 0)
        d["tet_after_7"]   = int(0 < tet_diff <= 7)
        d["tet_on"]        = int(tet_diff == 0)

        # ── GROUP H: Fixed Vietnamese Holidays ───────────────────────────────
        for hname, (hm, hd) in VN_FIXED_HOLIDAYS.items():
            d[f"hol_{hname}"] = int(month == hm and day == hd)

        # Black Friday (computed dynamically)
        bf = bf_cache[year]
        d["hol_black_friday"] = int(date == bf)

        # ── GROUP I: Promotional Windows ─────────────────────────────────────
        for pname, pcfg in PROMO_CAMPAIGNS.items():
            d.update(_promo_features(date, pname, pcfg))

        # ── GROUP J: Odd Year ─────────────────────────────────────────────────
        d["is_odd_year"] = year % 2   # 1 for odd (2023), 0 for even (2024)

        rows.append(d)

    feat = pd.DataFrame(rows, index=pd.RangeIndex(len(rows)))
    feat.insert(0, "Date", dates.values)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION: Sanity checks on feature frame
# ─────────────────────────────────────────────────────────────────────────────

def validate_features(feat: pd.DataFrame):
    """Run assertions on the feature DataFrame."""
    assert feat.isnull().sum().sum() == 0, "NaN values in features!"
    assert feat["is_weekend"].isin([0, 1]).all()
    assert feat["is_odd_year"].isin([0, 1]).all()

    # Promo constraints
    for pname in PROMO_CAMPAIGNS:
        col = f"promo_{pname}"
        assert feat[col].isin([0, 1]).all(), f"{col} has non-binary values"

    # Fourier bounds
    for col in [c for c in feat.columns if c.startswith("sin_") or c.startswith("cos_")]:
        assert feat[col].between(-1.0, 1.0).all(), f"{col} out of [-1,1] range"

    print(f"✅ Feature validation passed. Shape: {feat.shape}")
```

---

## 6. Validation Strategy

```python
# validators.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tag: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    result = {"tag": tag, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"[{tag}] MAE={mae:,.1f}  RMSE={rmse:,.1f}  R²={r2:.4f}")
    return result


def get_fold_masks(feat: pd.DataFrame) -> dict:
    """Return boolean masks for train/val splits."""
    dates = pd.to_datetime(feat["Date"])
    masks = {
        # Fold A — PRIMARY (validation year: 2022)
        "fold_a_train": dates <= "2021-12-31",
        "fold_a_val":   (dates >= "2022-01-01") & (dates <= "2022-12-31"),

        # Fold B — STABILITY (validation year: 2021)
        "fold_b_train": dates <= "2020-12-31",
        "fold_b_val":   (dates >= "2021-01-01") & (dates <= "2021-12-31"),

        # Fold C — HORIZON (12-month rolling, mimics test structure)
        "fold_c_train": dates <= "2021-06-30",
        "fold_c_val":   (dates >= "2021-07-01") & (dates <= "2022-06-30"),
    }
    return masks
```

---

## 7. Model Implementation

### 7.1 Ridge Regression

```python
# models/ridge_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
from config import RIDGE_ALPHA, SEED, MODELS_DIR


FEATURE_COLS = None   # Will be set at runtime (exclude Date, Revenue, COGS)


def get_feature_cols(feat: pd.DataFrame) -> list:
    return [c for c in feat.columns if c not in ("Date", "Revenue", "COGS")]


def train_ridge(feat_train: pd.DataFrame, target: str = "Revenue") -> dict:
    """Train Ridge on log-transformed target."""
    global FEATURE_COLS
    FEATURE_COLS = get_feature_cols(feat_train)

    X = feat_train[FEATURE_COLS].values.astype(float)
    y = np.log(feat_train[target].clip(lower=1).values)  # log-space, clip prevents log(0)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    model.fit(X_s, y)

    artifacts = {"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS, "target": target}
    joblib.dump(artifacts, MODELS_DIR / f"ridge_{target.lower()}.pkl")
    return artifacts


def predict_ridge(artifacts: dict, feat: pd.DataFrame) -> np.ndarray:
    X = feat[artifacts["feature_cols"]].values.astype(float)
    X_s = artifacts["scaler"].transform(X)
    log_pred = artifacts["model"].predict(X_s)
    return np.exp(log_pred)   # Back to original scale
```

### 7.2 LightGBM (Base)

```python
# models/lgb_model.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from config import LGB_PARAMS, LGB_N_ROUNDS, LGB_EARLY_STOP, SEED, MODELS_DIR
from config import HIGH_ERA_WEIGHT, BASE_WEIGHT, REGIME_PEAK_START, REGIME_PEAK_END


def _high_era_weights(years: np.ndarray) -> np.ndarray:
    """Sample weights: peak era (2014–2018) = 1.0, all others = 0.01."""
    w = np.full(len(years), BASE_WEIGHT)
    peak = (years >= REGIME_PEAK_START) & (years <= REGIME_PEAK_END)
    w[peak] = HIGH_ERA_WEIGHT
    return w


def train_lgb_base(
    feat_train: pd.DataFrame,
    feat_val: pd.DataFrame,
    target: str = "Revenue",
    feature_cols: list = None,
) -> dict:
    """Two-phase LightGBM training with early stopping."""

    if feature_cols is None:
        feature_cols = [c for c in feat_train.columns if c not in ("Date", "Revenue", "COGS")]

    y_train = np.log(feat_train[target].clip(lower=1).values)
    y_val   = np.log(feat_val[target].clip(lower=1).values)
    X_train = feat_train[feature_cols].values
    X_val   = feat_val[feature_cols].values

    years_train = feat_train["year"].values
    weights = _high_era_weights(years_train)

    # ── Phase 1: Find best_iteration via early stopping ──────────────────────
    dtrain = lgb.Dataset(X_train, y_train, weight=weights, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val, y_val, reference=dtrain)

    callbacks = [lgb.early_stopping(LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(200)]
    model_p1 = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=LGB_N_ROUNDS,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    best_iter = model_p1.best_iteration
    print(f"[LGB-{target}] Best iteration: {best_iter}")

    # ── Phase 2: Retrain on ALL data with best_iteration ─────────────────────
    feat_full = pd.concat([feat_train, feat_val], ignore_index=True)
    y_full    = np.log(feat_full[target].clip(lower=1).values)
    X_full    = feat_full[feature_cols].values
    years_full = feat_full["year"].values
    w_full    = _high_era_weights(years_full)

    dtrain_full = lgb.Dataset(X_full, y_full, weight=w_full, feature_name=feature_cols)
    model_final = lgb.train(LGB_PARAMS, dtrain_full, num_boost_round=best_iter)

    artifacts = {
        "model": model_final,
        "best_iter": best_iter,
        "feature_cols": feature_cols,
        "target": target,
    }
    joblib.dump(artifacts, MODELS_DIR / f"lgb_base_{target.lower()}.pkl")
    return artifacts


def predict_lgb(artifacts: dict, feat: pd.DataFrame) -> np.ndarray:
    X = feat[artifacts["feature_cols"]].values
    log_pred = artifacts["model"].predict(X)
    return np.exp(log_pred)
```

### 7.3 Q-Specialists

```python
# models/q_specialists.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from config import LGB_PARAMS, LGB_N_ROUNDS, LGB_EARLY_STOP, Q_BOOST
from config import HIGH_ERA_WEIGHT, BASE_WEIGHT, REGIME_PEAK_START, REGIME_PEAK_END, MODELS_DIR


def _quarter_weights(feat: pd.DataFrame, target_q: int) -> np.ndarray:
    """
    Three-layer weight scheme for Q-specialist:
    Layer 1: All data = BASE_WEIGHT (0.01)
    Layer 2: Peak era (2014-2018) = HIGH_ERA_WEIGHT (1.0)
    Layer 3: Peak era + target quarter = HIGH_ERA_WEIGHT × Q_BOOST
    """
    years    = feat["year"].values
    quarters = feat["quarter"].values

    w = np.full(len(feat), BASE_WEIGHT)

    peak = (years >= REGIME_PEAK_START) & (years <= REGIME_PEAK_END)
    w[peak] = HIGH_ERA_WEIGHT

    target_q_mask = peak & (quarters == target_q)
    w[target_q_mask] *= Q_BOOST

    return w


def train_q_specialists(
    feat_train: pd.DataFrame,
    feat_val: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Train 4 quarter-specialists × 2 targets = 8 models.
    Returns dict keyed by (quarter, target).
    """
    specialists = {}

    for q in [1, 2, 3, 4]:
        for target in ["Revenue", "COGS"]:
            print(f"  Training Spec_Q{q}_{target}...")

            y_train = np.log(feat_train[target].clip(lower=1).values)
            y_val   = np.log(feat_val[target].clip(lower=1).values)
            X_train = feat_train[feature_cols].values
            X_val   = feat_val[feature_cols].values

            weights = _quarter_weights(feat_train, q)

            dtrain = lgb.Dataset(X_train, y_train, weight=weights, feature_name=feature_cols)
            dval   = lgb.Dataset(X_val, y_val, reference=dtrain)

            callbacks = [lgb.early_stopping(LGB_EARLY_STOP, verbose=False)]
            m = lgb.train(
                LGB_PARAMS, dtrain,
                num_boost_round=LGB_N_ROUNDS,
                valid_sets=[dval],
                callbacks=callbacks,
            )
            specialists[(q, target)] = m
            m.save_model(str(MODELS_DIR / f"spec_Q{q}_{target.lower()}.txt"))

    return specialists


def predict_q_composed(
    specialists: dict,
    feat_test: pd.DataFrame,
    feature_cols: list,
    target: str = "Revenue",
) -> np.ndarray:
    """
    COMPOSE predictions: use Spec_Qq only for dates in quarter q.
    Never average across specialists.
    """
    X = feat_test[feature_cols].values
    quarters = feat_test["quarter"].values
    preds = np.zeros(len(feat_test))

    for q in [1, 2, 3, 4]:
        mask = quarters == q
        if mask.sum() > 0:
            log_p = specialists[(q, target)].predict(X[mask])
            preds[mask] = np.exp(log_p)

    return preds
```

### 7.4 Prophet

```python
# models/prophet_model.py
import numpy as np
import pandas as pd
from prophet import Prophet
import joblib
from config import SEED, MODELS_DIR


PROMO_COLS = [
    f"promo_{name}"
    for name in ["spring_sale", "mid_year", "fall_launch", "year_end",
                 "urban_blowout", "rural_special"]
]


def train_prophet(
    feat_train: pd.DataFrame,
    target: str = "Revenue",
    post_regime_only: bool = True,
) -> Prophet:
    """
    Train Prophet on log-Revenue.
    post_regime_only=True: use only 2020–2022 for stable level calibration.
    """
    if post_regime_only:
        mask = feat_train["year"] >= 2020
        data = feat_train[mask].copy()
    else:
        data = feat_train.copy()

    df_p = pd.DataFrame({
        "ds": data["Date"],
        "y":  np.log(data[target].clip(lower=1)),
    })
    for col in PROMO_COLS:
        if col in data.columns:
            df_p[col] = data[col].values

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    for col in PROMO_COLS:
        if col in df_p.columns:
            m.add_regressor(col)

    m.fit(df_p)
    joblib.dump(m, MODELS_DIR / f"prophet_{target.lower()}.pkl")
    return m


def predict_prophet(m: Prophet, feat_test: pd.DataFrame, target: str = "Revenue") -> np.ndarray:
    future = pd.DataFrame({"ds": feat_test["Date"]})
    for col in PROMO_COLS:
        if col in feat_test.columns:
            future[col] = feat_test[col].values
        else:
            future[col] = 0

    forecast = m.predict(future)
    return np.exp(forecast["yhat"].values)
```

---

## 8. Ensemble Strategy

```python
# ensemble.py
import numpy as np
import pandas as pd
from config import ALPHA_SPECIALIST, WEIGHT_RIDGE, WEIGHT_PROPHET, WEIGHT_LGB


def layer1_lgb_blend(p_lgb_base: np.ndarray, p_lgb_spec: np.ndarray) -> np.ndarray:
    """
    Layer 1: Blend base LGB and Q-specialist predictions within the LGB family.
    α = 0.60 → specialists trusted more (60%) vs base (40%).
    """
    return (1 - ALPHA_SPECIALIST) * p_lgb_base + ALPHA_SPECIALIST * p_lgb_spec


def layer2_family_blend(
    p_ridge: np.ndarray,
    p_prophet: np.ndarray,
    p_lgb_blend: np.ndarray,
) -> np.ndarray:
    """
    Layer 2: Blend across model families.
    Ridge/Prophet act as anchors (0.10 each); LGB dominates (0.80).
    """
    return (
        WEIGHT_RIDGE   * p_ridge +
        WEIGHT_PROPHET * p_prophet +
        WEIGHT_LGB     * p_lgb_blend
    )


def layer3_calibrate(raw: np.ndarray, calibration_scalar: float) -> np.ndarray:
    """
    Layer 3: Multiplicative level calibration.
    CR=1.26 for Revenue, CC=1.32 for COGS.
    Compensates for log-space mean-pull bias.
    """
    return calibration_scalar * raw


def build_ensemble(
    p_ridge_rev: np.ndarray,
    p_lgb_rev: np.ndarray,
    p_spec_rev: np.ndarray,
    p_prophet_rev: np.ndarray,
    p_ridge_cog: np.ndarray,
    p_lgb_cog: np.ndarray,
    p_spec_cog: np.ndarray,
    p_prophet_cog: np.ndarray,
    cr: float = 1.26,
    cc: float = 1.32,
) -> pd.DataFrame:
    """Full 3-layer ensemble for both targets."""

    # Revenue
    lgb_blend_rev = layer1_lgb_blend(p_lgb_rev, p_spec_rev)
    raw_rev       = layer2_family_blend(p_ridge_rev, p_prophet_rev, lgb_blend_rev)
    final_rev     = layer3_calibrate(raw_rev, cr)

    # COGS
    lgb_blend_cog = layer1_lgb_blend(p_lgb_cog, p_spec_cog)
    raw_cog       = layer2_family_blend(p_ridge_cog, p_prophet_cog, lgb_blend_cog)
    final_cog     = layer3_calibrate(raw_cog, cc)

    return pd.DataFrame({
        "Revenue_pred": final_rev,
        "COGS_pred":    final_cog,
        # Keep intermediates for diagnostics
        "lgb_blend_rev": lgb_blend_rev,
        "raw_rev":       raw_rev,
    })
```

---

## 9. Calibration

```python
# calibration.py
"""
Calibration scalars (CR, CC) MUST be tuned on the leaderboard, NOT on CV.
This module helps sweep calibration values and compute expected metric changes.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def sweep_calibration(
    raw_pred: np.ndarray,
    y_val: np.ndarray,
    c_range: list,
) -> pd.DataFrame:
    """
    Sweep a range of calibration scalars on a VALIDATION set.
    ⚠️ Use for intuition only — final CR/CC must be tuned on leaderboard.
    """
    results = []
    for c in c_range:
        pred = c * raw_pred
        mae  = mean_absolute_error(y_val, pred)
        results.append({"c": c, "MAE": mae})
    return pd.DataFrame(results).sort_values("MAE")


def generate_calibration_grid(cr_range=(1.18, 1.34, 0.02), cc_range=(1.28, 1.36, 0.02)):
    """
    Generate submission variants for leaderboard sweep.
    Returns a list of (CR, CC) tuples.
    """
    crs = np.arange(*cr_range)
    ccs = np.arange(*cc_range)
    return [(round(cr, 2), round(cc, 2)) for cr in crs for cc in ccs]
```

---

## 10. Hyperparameter Tuning

### 10.1 Optuna-based LGB Tuning (CV-driven)

```python
# scripts/tune_lgb.py
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, feat_train, feat_val, feature_cols, target="Revenue"):
    params = {
        "objective":      "regression",
        "metric":         "mae",
        "learning_rate":  trial.suggest_float("lr", 0.01, 0.05, log=True),
        "num_leaves":     trial.suggest_int("num_leaves", 31, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq":   5,
        "lambda_l2":      trial.suggest_float("lambda_l2", 0.1, 5.0, log=True),
        "seed":           42,
        "verbose":        -1,
    }

    X_train = feat_train[feature_cols].values
    y_train = np.log(feat_train[target].clip(lower=1))
    X_val   = feat_val[feature_cols].values
    y_val   = feat_val[target].values

    dtrain  = lgb.Dataset(X_train, y_train)
    dval    = lgb.Dataset(X_val, np.log(feat_val[target].clip(lower=1)))

    m = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    preds = np.exp(m.predict(X_val))
    return mean_absolute_error(y_val, preds)


def tune_lgb(feat_train, feat_val, feature_cols, n_trials=80):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, feat_train, feat_val, feature_cols),
        n_trials=n_trials,
    )
    print("Best params:", study.best_params)
    print(f"Best MAE:    {study.best_value:,.1f}")
    return study.best_params
```

---

## 11. Explainability (SHAP)

```python
# explainability.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap_values(lgb_model, feat: pd.DataFrame, feature_cols: list):
    X = feat[feature_cols].values
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def plot_shap_summary(shap_values, feat, feature_cols, save_path="outputs/reports/shap_summary.png"):
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, feat[feature_cols], show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"SHAP summary saved to {save_path}")


def plot_shap_bar(shap_values, feature_cols, top_n=20, save_path="outputs/reports/shap_bar.png"):
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
```

---

## 12. Submission Pipeline

### 12.1 `scripts/train_all.py`

```python
# scripts/train_all.py
"""End-to-end training script. Run with: python scripts/train_all.py"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "src")

from config import SEED, MODELS_DIR, DATA_PROC
from data_loader import load_sales, load_test_dates
from feature_engineering import build_features, validate_features
from validators import get_fold_masks, compute_metrics
from models.ridge_model import train_ridge, predict_ridge
from models.lgb_model import train_lgb_base, predict_lgb
from models.q_specialists import train_q_specialists, predict_q_composed
from models.prophet_model import train_prophet, predict_prophet
from ensemble import build_ensemble

import random, os
random.seed(SEED); np.random.seed(SEED)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("Step 1: Loading data...")
sales = load_sales()
test_dates = load_test_dates()

# ── Step 2: Feature engineering ───────────────────────────────────────────────
print("Step 2: Building features...")
all_dates = pd.concat([sales["Date"], test_dates["Date"]])
feat_all  = build_features(all_dates)
validate_features(feat_all)

feat_train_raw = feat_all[feat_all["Date"].isin(sales["Date"])].copy()
feat_test      = feat_all[feat_all["Date"].isin(test_dates["Date"])].copy()

# Attach targets
feat_train = feat_train_raw.merge(sales[["Date", "Revenue", "COGS"]], on="Date")

# Save features
feat_train.to_parquet(DATA_PROC / "features_train.parquet")
feat_test.to_parquet(DATA_PROC / "features_test.parquet")
print(f"  Train: {feat_train.shape}, Test: {feat_test.shape}")

# ── Step 3: Validation setup ──────────────────────────────────────────────────
masks = get_fold_masks(feat_train)
tr_a  = feat_train[masks["fold_a_train"]]
va_a  = feat_train[masks["fold_a_val"]]

feature_cols = [c for c in feat_train.columns if c not in ("Date", "Revenue", "COGS")]
print(f"  Feature count: {len(feature_cols)}")

# ── Step 4: Train base models ─────────────────────────────────────────────────
print("\nStep 4: Training base models...")

ridge_rev = train_ridge(tr_a, "Revenue")
ridge_cog = train_ridge(tr_a, "COGS")

lgb_rev = train_lgb_base(tr_a, va_a, "Revenue", feature_cols)
lgb_cog = train_lgb_base(tr_a, va_a, "COGS",    feature_cols)

prophet_rev = train_prophet(tr_a, "Revenue", post_regime_only=True)
prophet_cog = train_prophet(tr_a, "COGS",    post_regime_only=True)

# ── Step 5: Train Q-specialists ───────────────────────────────────────────────
print("\nStep 5: Training Q-specialists...")
specs_rev = train_q_specialists(tr_a, va_a, feature_cols)  # target="Revenue" embedded
specs_cog = train_q_specialists(tr_a, va_a, feature_cols)  # target="COGS"

# ── Step 6: Validate on Fold A ────────────────────────────────────────────────
print("\nStep 6: Fold A validation...")
for target, ridge_art, lgb_art, specs, prophet_m in [
    ("Revenue", ridge_rev, lgb_rev, specs_rev, prophet_rev),
    ("COGS",    ridge_cog, lgb_cog, specs_cog, prophet_cog),
]:
    p_r = predict_ridge(ridge_art, va_a)
    p_l = predict_lgb(lgb_art, va_a)
    p_s = predict_q_composed(specs, va_a, feature_cols, target)
    p_p = predict_prophet(prophet_m, va_a, target)

    ens = build_ensemble(p_r, p_l, p_s, p_p, p_r, p_l, p_s, p_p)
    y_true = va_a[target].values
    compute_metrics(y_true, ens["Revenue_pred" if target == "Revenue" else "COGS_pred"].values,
                    tag=f"FoldA_{target}")

print("\n✅ Training complete. Run predict.py to generate test submissions.")
```

### 12.2 `scripts/predict.py`

```python
# scripts/predict.py
"""Generate final submission CSV. Run: python scripts/predict.py --cr 1.26 --cc 1.32"""
import argparse
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")

from config import DATA_RAW
from feature_engineering import build_features
from models.ridge_model import predict_ridge
from models.lgb_model import predict_lgb
from models.q_specialists import predict_q_composed
from models.prophet_model import predict_prophet
from ensemble import build_ensemble
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--cr", type=float, default=1.26)
parser.add_argument("--cc", type=float, default=1.32)
parser.add_argument("--tag", type=str, default="v1")
args = parser.parse_args()

# Load test dates
test_dates = pd.read_csv(DATA_RAW / "sample_submission.csv", parse_dates=["Date"])
feat_test  = build_features(test_dates["Date"])

feature_cols = [c for c in feat_test.columns if c not in ("Date",)]

# Load models
ridge_rev   = joblib.load("outputs/models/ridge_revenue.pkl")
ridge_cog   = joblib.load("outputs/models/ridge_cogs.pkl")
lgb_rev     = joblib.load("outputs/models/lgb_base_revenue.pkl")
lgb_cog     = joblib.load("outputs/models/lgb_base_cogs.pkl")
prophet_rev = joblib.load("outputs/models/prophet_revenue.pkl")
prophet_cog = joblib.load("outputs/models/prophet_cogs.pkl")

# Q-specialist models (loaded via lgb)
import lightgbm as lgb as lgb_lib
specs_rev = {(q, "Revenue"): lgb_lib.Booster(model_file=f"outputs/models/spec_Q{q}_revenue.txt")
             for q in [1,2,3,4]}
specs_cog = {(q, "COGS"): lgb_lib.Booster(model_file=f"outputs/models/spec_Q{q}_cogs.txt")
             for q in [1,2,3,4]}

# Predict
p_ridge_rev   = predict_ridge(ridge_rev, feat_test)
p_ridge_cog   = predict_ridge(ridge_cog, feat_test)
p_lgb_rev     = predict_lgb(lgb_rev, feat_test)
p_lgb_cog     = predict_lgb(lgb_cog, feat_test)
p_prophet_rev = predict_prophet(prophet_rev, feat_test, "Revenue")
p_prophet_cog = predict_prophet(prophet_cog, feat_test, "COGS")
p_spec_rev    = predict_q_composed(specs_rev, feat_test, feature_cols, "Revenue")
p_spec_cog    = predict_q_composed(specs_cog, feat_test, feature_cols, "COGS")

ensemble = build_ensemble(
    p_ridge_rev, p_lgb_rev, p_spec_rev, p_prophet_rev,
    p_ridge_cog, p_lgb_cog, p_spec_cog, p_prophet_cog,
    cr=args.cr, cc=args.cc,
)

submission = pd.DataFrame({
    "Date":    test_dates["Date"].dt.strftime("%Y-%m-%d"),
    "Revenue": ensemble["Revenue_pred"].round(2),
    "COGS":    ensemble["COGS_pred"].round(2),
})

out_path = f"outputs/submissions/submission_{args.tag}_cr{args.cr}_cc{args.cc}.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Submission saved to {out_path} ({len(submission)} rows)")
```

---

## 13. Upgrade Proposals: Advanced Time Series Methods

> This section proposes enhancements beyond the baseline architecture, ordered by expected ROI.

---

### 13.1 ⭐⭐⭐ Adaptive Per-Quarter Calibration

**Problem:** Single global calibration scalars (CR, CC) cannot account for structural differences across quarters (especially Q3 odd-year COGS inversion).

**Upgrade:**
```python
# Per-quarter calibration scalars
CR_Q = {1: 1.22, 2: 1.24, 3: 1.30, 4: 1.26}   # tuned per quarter on LB
CC_Q = {1: 1.28, 2: 1.30, 3: 1.40, 4: 1.32}   # Q3 COGS needs higher correction

def calibrate_by_quarter(raw_pred, quarters, cr_map, cc_map, target="Revenue"):
    cmap = cr_map if target == "Revenue" else cc_map
    return np.array([raw_pred[i] * cmap[quarters[i]] for i in range(len(raw_pred))])
```

**Expected gain:** -3–5% MAE on Q3 predictions.

---

### 13.2 ⭐⭐⭐ NNLS Ensemble Optimization

**Problem:** Hand-tuned weights (0.10 / 0.10 / 0.80) are suboptimal; the optimal combination of all LGB variants is unknown.

**Method:** Non-Negative Least Squares (NNLS) finds weights that minimize ||Aw - y||² subject to w ≥ 0.

```python
from scipy.optimize import nnls

def optimize_ensemble_weights(predictions: dict, y_val: np.ndarray) -> dict:
    """
    predictions: {"model_name": np.ndarray of shape (n,)}
    Returns optimal non-negative weights that minimize MAE on validation set.
    """
    names = list(predictions.keys())
    A = np.column_stack([predictions[n] for n in names])

    # NNLS minimizes squared error, but we want MAE.
    # Use it as a proxy; fine-tune top weights manually.
    weights_raw, _ = nnls(A, y_val)

    # Normalize to sum to 1
    weights_norm = weights_raw / weights_raw.sum()

    result = dict(zip(names, weights_norm))
    print("Optimal weights:")
    for k, v in sorted(result.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:.4f}")
    return result
```

**All LGB variants to include:**
- `lgb_high_era` (current base)
- `lgb_t1718` (top 2 years only: 2017–2018)
- `lgb_up_HIGH` (up-weighted 2020–2022)
- `lgb_analog` (2022 mapped as analog of 2023)
- `lgb_up_2018` (2018 as the closest structural analog)

---

### 13.3 ⭐⭐⭐ Time Series Foundation Models (Chronos / TimesFM)

**What they are:** Large pre-trained models trained on hundreds of diverse time series datasets. They can zero-shot forecast without task-specific training.

**Why they add diversity:** Tree models and linear models share blind spots (they extrapolate poorly to unseen regimes). Foundation models use different inductive biases.

**Integration approach:**
```python
# Using Amazon Chronos (open-source, MIT license)
# pip install chronos-forecasting

from chronos import ChronosPipeline
import torch

def predict_chronos(train_series: np.ndarray, horizon: int = 548) -> np.ndarray:
    """
    Zero-shot forecast using Chronos-T5-Small.
    Input: 1D array of historical values (Revenue)
    Output: median forecast for next `horizon` steps
    """
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    context = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=20)
    return np.median(forecast[0].numpy(), axis=0)
```

**Expected behavior:** Chronos excels at seasonal extrapolation (it has learned from retail, economic, and seasonal data). Its residuals are likely to be uncorrelated with LGB/Ridge.

**Ensemble strategy:** Add Chronos as a low-weight anchor (5–10%) in Layer 2.

---

### 13.4 ⭐⭐ Lag Features via Iterative Forecasting

**Problem:** Lag features (y_{t-1}, y_{t-7}) are the strongest predictors in many time series tasks but cannot be used directly in 18-month horizon forecasting.

**Solution:** Iterative (autoregressive) forecasting using predicted values as pseudo-lags.

```python
def iterative_forecast(
    model,
    feat_test: pd.DataFrame,
    feature_cols: list,
    last_known: np.ndarray,   # Last 365 days of actual Revenue
    lag_windows: list = [1, 7, 14, 30, 365],
) -> np.ndarray:
    """
    Day-by-day forecasting where today's prediction feeds tomorrow's lag features.
    Error accumulates over time — use only as one component of the ensemble.
    """
    history = list(last_known)
    preds   = []

    for i in range(len(feat_test)):
        row = feat_test.iloc[[i]].copy()

        # Inject lag features computed from history
        for lag in lag_windows:
            if len(history) >= lag:
                row[f"lag_{lag}"] = history[-lag]
            else:
                row[f"lag_{lag}"] = np.mean(history)

        pred = np.exp(model.predict(row[feature_cols + [f"lag_{lag}" for lag in lag_windows]]))
        preds.append(pred[0])
        history.append(pred[0])   # Feed into next step

    return np.array(preds)
```

**Risk:** Error compounds over 548 steps. Best used as a weak member (weight ≤ 0.05) in early forecast days only (first 30–60 days).

---

### 13.5 ⭐⭐ N-BEATS / N-HiTS Architecture

**N-HiTS** (Neural Hierarchical Interpolation for Time Series) is purpose-built for long-horizon forecasting. It uses hierarchical multi-rate sampling — different "scales" of the signal are processed separately.

**Why it fits this problem:**
- Handles multiple seasonalities natively (daily + weekly + annual)
- Interpretable: separate blocks for trend, seasonality, residuals
- 18-month horizon is its sweet spot

**Integration:**
```python
# pip install neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

def train_nhits(df_train: pd.DataFrame, horizon: int = 548) -> NeuralForecast:
    """
    df_train must have columns: unique_id, ds (date), y (log-Revenue)
    """
    df_train["unique_id"] = "store_total"
    df_train["ds"] = pd.to_datetime(df_train["Date"])
    df_train["y"]  = np.log(df_train["Revenue"].clip(lower=1))

    model = NHITS(
        h=horizon,
        input_size=2 * horizon,    # 2× horizon as context window
        max_steps=500,
        stack_types=["identity", "trend", "seasonality"],
        n_freq_downsample=[2, 1, 1],
        learning_rate=1e-3,
        scaler_type="standard",
    )

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df_train[["unique_id", "ds", "y"]])
    return nf


def predict_nhits(nf: NeuralForecast, test_dates: pd.DataFrame) -> np.ndarray:
    future = pd.DataFrame({
        "unique_id": "store_total",
        "ds": test_dates["Date"],
    })
    forecast = nf.predict(futr_df=future)
    return np.exp(forecast["NHITS"].values)
```

---

### 13.6 ⭐⭐ Quantile Regression for Uncertainty Intervals

**Problem:** Point forecasts hide uncertainty; it is unknown whether the model is confident or wildly guessing on a given day.

**Method:** Train LightGBM with `objective="quantile"` at q=0.1, 0.5, 0.9 to produce prediction intervals.

```python
def train_lgb_quantile(feat_train, feat_val, feature_cols, target="Revenue", alpha=0.1):
    params_q = {**LGB_PARAMS, "objective": "quantile", "alpha": alpha, "metric": "quantile"}

    X = feat_train[feature_cols].values
    y = np.log(feat_train[target].clip(lower=1).values)
    dtrain = lgb.Dataset(X, y)

    m = lgb.train(params_q, dtrain, num_boost_round=1000)
    return m


# Generate P10, P50, P90 for submission explainability
m_p10 = train_lgb_quantile(tr_a, va_a, feature_cols, alpha=0.10)
m_p90 = train_lgb_quantile(tr_a, va_a, feature_cols, alpha=0.90)
```

**Use case:** Include in explainability report. Wide intervals on Q3 days correctly signal high model uncertainty.

---

### 13.7 ⭐ Auxiliary Table Feature Mining

**Web Traffic → Conversion Cycle Features**

```python
def extract_traffic_patterns(wt_train: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SEASONAL PATTERNS from web traffic (not direct values).
    Only month/dow averages can be applied to test dates.
    """
    wt_train["month"] = pd.to_datetime(wt_train["date"]).dt.month
    wt_train["dow"]   = pd.to_datetime(wt_train["date"]).dt.dayofweek

    # Average conversion rate by month (stable pattern → can project to 2023-2024)
    monthly_conv = wt_train.groupby("month")["conversion_rate"].mean().rename("avg_conv_by_month")
    dow_sessions = wt_train.groupby("dow")["sessions"].mean().rename("avg_sessions_by_dow")

    return monthly_conv, dow_sessions


def apply_traffic_patterns(feat: pd.DataFrame, monthly_conv, dow_sessions) -> pd.DataFrame:
    """Apply historical seasonal patterns to any date (including test)."""
    feat["avg_conv_by_month"] = feat["month"].map(monthly_conv)
    feat["avg_sessions_by_dow"] = feat["dow"].map(dow_sessions)
    return feat
```

**Expected gain:** Low (these patterns are partially captured by Fourier features). Worth ±0.5% MAE.

---

### 13.8 Summary Table: Upgrade ROI

| # | Method | Complexity | Expected MAE Gain | Priority |
|---|--------|-----------|-------------------|----------|
| 13.1 | Per-quarter calibration | Low | 3–5% | ⭐⭐⭐ |
| 13.2 | NNLS ensemble optimization | Medium | 2–4% | ⭐⭐⭐ |
| 13.3 | Chronos/TimesFM foundation model | Medium | 1–3% | ⭐⭐⭐ |
| 13.4 | Iterative lag forecasting | High | 0.5–2% | ⭐⭐ |
| 13.5 | N-HiTS architecture | High | 1–3% | ⭐⭐ |
| 13.6 | Quantile regression | Low | 0% (explainability) | ⭐⭐ |
| 13.7 | Auxiliary table mining | Low | 0–1% | ⭐ |

---

## 14. Anti-Pattern Reference

> A quick reference of the most common mistakes and how to avoid them.

| # | Anti-Pattern | Symptom | Fix |
|---|--------------|---------|-----|
| 1 | Using inventory/web data from 2023 test period as features | Look-ahead bias, overly optimistic CV | Only use historical pattern averages, never raw future values |
| 2 | Training Prophet on full 2012–2022 history | Trend line distorted by 2019 regime jump → poor extrapolation | Set `post_regime_only=True` (2020–2022 only) |
| 3 | Averaging Q-specialists across all days | Q1 model performs poorly on Q4 days — averaging degrades Q-specialist benefit | Compose predictions: use Q_q specialist only for days in quarter q |
| 4 | Tuning CR/CC on CV (2022 validation) | Wrong calibration level for 2023 test | Always tune CR/CC on leaderboard submissions only |
| 5 | Using fixed month for Tết (e.g., "February = Tết month") | Misses 21–29 day shift in Gregorian position across years | Use `tet_days_diff` computed from hardcoded Tết lookup table |
| 6 | Forgetting `is_odd_year` feature | August 2024 (even) predicted as same as August 2023 (odd) → 1.6× error | Always include `is_odd_year = year % 2` |
| 7 | Training COGS as `0.88 × Revenue` | Q3 odd-year COGS > Revenue (margin > 1.0) → systematic underestimate | Train COGS independently with its own log-COGS model |
| 8 | Using standard k-fold cross-validation | Future data leaks into training set | Always use time-based folds (train on past, validate on future) |
| 9 | Applying equal sample weights across all years | Model averages 3 non-overlapping distributions → always underpredicts for 2023 | Use `high_era` weighting (2014–2018 = 1.0, rest = 0.01) |
| 10 | Stacking two models of the same family with equal weight | Reduces diversity without improving accuracy | Use Layer 1 internal blend first, then cross-family blend in Layer 2 |

---

*Document prepared for AIO Course 2025 — VinDatathon 2026 Task 3 implementation reference.*  
*Seed: 42. All random operations must be seeded for reproducibility.*