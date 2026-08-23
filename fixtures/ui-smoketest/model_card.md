# Model Card — Ridge

## Intended Use
Weekly sales forecasting per (Store, Dept) for the Retail Data Analytics workflow. Not intended for financial reporting or any decision outside retail demand planning.

## Training Data
- Source: Kaggle 'Retail Data Analytics' (stores.csv, features.csv, train.csv).
- Chronological split cutoff: 2012-04-13 (22,857 train rows / 5,763 test rows).
- Target `Weekly_Sales` contract bounds: [-5000.0, 800000.0].

## Selected Model & Performance
- **Model:** Ridge
- **MAE:** 1714.79
- **RMSE:** 3067.94
- **R2:** 0.9878
- Selection rule: lowest RMSE on the chronological hold-out test set, computed deterministically in src/services/ml_trainer.py (no LLM involved in model choice or metric computation).

## Candidate Models Considered
Exactly two candidates were trained and compared: `Ridge(alpha=1.0)` and `RandomForestRegressor(n_estimators=100, random_state=42)`. No other model families were evaluated.

## Limitations
- Trained on historical US retail data (2010-2012); may not generalize to other markets or eras.
- Chronological hold-out approximates production drift but does not guarantee future accuracy.
- Lag/rolling features require at least 4 prior weeks of history per (Store, Dept); cold-start series are dropped during feature engineering.
