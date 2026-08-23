# Model Evaluation Report

- **Train rows:** 22,857
- **Test rows:** 5,763
- **Chronological split cutoff date:** 2012-04-13
- **Feature columns (17):** IsHoliday, Size, Temperature, Fuel_Price, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI, Unemployment, Weekly_Sales_Lag1, Weekly_Sales_Lag4, Weekly_Sales_RollingMean4, Weekly_Sales_RollingStd4, Type_A, Type_B

## Candidate Comparison (chronological hold-out test set)

| Model | MAE | RMSE | R2 | Selected |
|---|---|---|---|---|
| Ridge | 1714.79 | 3067.94 | 0.9878 | selected |
| RandomForestRegressor | 1627.04 | 3132.01 | 0.9873 |  |

**Selected model:** `Ridge` (deterministically chosen: lowest RMSE on the chronological test split).
