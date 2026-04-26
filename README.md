# VinDatathon Forecasting Pipeline

This repository rebuilds the forecasting solution around the architecture in `plan.md`:

- `src/`: data loading, feature engineering, model wrappers, ensembles, calibration, and pipeline orchestration
- `scripts/`: train, predict, calibration, and experiment entrypoints
- `data/raw/`: provided competition data
- `data/processed/`: generated feature artifacts
- `outputs/`: trained models, reports, and submissions

## Run

```bash
./.venv/bin/python scripts/train_all.py
```

That command will:

1. run rolling backtests
2. tune simple revenue and COGS calibration scalars
3. fit the final models on all training data
4. save feature artifacts, reports, the serialized pipeline, and `outputs/submissions/submission.csv`

## Notes

- The implementation uses only provided data.
- Promotion features for 2023-2024 are generated from the recurring historical campaign patterns present in `promotions.csv`.
- The primary models are LightGBM plus Ridge with a month-boundary specialist layer.

