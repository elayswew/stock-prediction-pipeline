# 📈 Stock High/Low Prediction Pipeline

An end-to-end machine learning pipeline that predicts a stock's **next-week high and low**
from historical price data — covering data ingestion, feature engineering, model selection,
and evaluation.

## Overview

This project fetches historical stock data, engineers technical features, compares multiple
ML models using time-series cross-validation, and predicts the high/low range for the next
5 trading days.

> **Note on scope:** This project demonstrates an end-to-end ML engineering workflow.
> It is **not** intended as a trading tool — short-term stock prices are close to a random
> walk and not reliably predictable. The value is in the methodology: data pipelines,
> feature engineering, leakage-free validation, and honest evaluation.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `yfinance` · `SQLite` · `matplotlib`

## How to Run

```bash
pip install -r requirements.txt
python model.py
```

Output: each model's cross-validated error, the selected best model, the predicted
next-week high/low, and a chart saved to `prediction_plot.png`.

## Project Structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Fetch stock data via the yfinance API; store/load in SQLite |
| `features.py` | Technical feature engineering + prediction target construction |
| `model.py` | Model comparison, selection, prediction, and visualization |

## Pipeline Steps

1. **Data ingestion** — Fetch ~5 years of daily OHLCV data and cache it in SQLite
2. **Feature engineering** — Build 9 technical features (moving averages, volatility,
   daily return, range position, volume change, momentum, RSI)
3. **Target construction** — Define the next-5-day high and low as regression targets
4. **Model selection** — Compare 4 models via time-series cross-validation, pick lowest MAE
5. **Prediction & visualization** — Forecast the next-week range and plot it

## Key Design Decisions

- **Time-series cross-validation, not random splits.** Stock data is ordered in time, so
  training folds always precede test folds. Random splits would leak future information into
  training (lookahead bias), inflating performance unrealistically.

- **Separated feature and target NaN handling.** Features need past data (NaN at the start
  from rolling windows); targets need future data (NaN at the end, since the future hasn't
  happened). Training drops rows without targets, but **prediction keeps the latest rows** —
  because prediction doesn't need a label, and those most recent days are exactly what we
  want to forecast.

- **Idempotent, per-symbol storage.** Re-running shouldn't create duplicate rows, but the
  database should still hold multiple stocks. I use a delete-then-insert pattern: before
  inserting a symbol, I delete only that symbol's existing rows.

- **Parameterized SQL queries** to prevent SQL injection — the database treats inputs
  strictly as data, never as executable SQL.

- **Empirical model selection over assumptions.** Rather than assuming a complex model wins,
  I let cross-validation decide.

## Results & Findings

A notable finding: **simple linear models (Linear Regression, Ridge) clearly outperformed
the tree-based ensembles (Random Forest, Gradient Boosting)** — several times lower error
(roughly 3x–7x depending on the stock). On noisy financial data, the more flexible models
overfit the noise, while the simpler models generalized better.

Tree-based models also **cannot extrapolate beyond the value range seen in training**: a
stock making new highs pushes test-period prices above anything in the training window, so
the trees systematically underestimate. This effect is larger for high-priced, trending
stocks (e.g. MU) than for more moderate ones (e.g. AAPL), which is why the gap between
linear and tree models varies by stock. A concrete reminder that more complex isn't always
better.

## Limitations

- **Short-term prices are near-random.** A low MAE here doesn't imply real predictive power;
  a naive "tomorrow ≈ today" baseline achieves comparable error because prices move little
  day-to-day. This pipeline is a methodology demonstration, not a trading signal.
- **Price-only features.** The model uses only historical prices, ignoring fundamentals,
  news, and macro factors that drive real markets.
- **Tree models extrapolate poorly (in fact, not at all).** For a stock at all-time highs,
  they cannot predict beyond the highest value seen in training, leading to large errors.

## Future Work

- Add a **naive baseline** for honest benchmarking (does the model actually beat "tomorrow ≈ today"?)
- Add **XGBoost / LightGBM** as stronger ensemble options
- Add **cross-asset features** (correlated stocks or a sector index), since related stocks move together
- Backtest predicted ranges against actuals over time

---

*Originally built in 2023; reconstructed and refined as a portfolio project.*
