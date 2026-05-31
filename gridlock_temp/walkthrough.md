# Gridlock 2.0 Hackathon: Traffic Demand Prediction Solution

This document provides a detailed, comprehensive walkthrough of our end-to-end Machine Learning pipeline for the Flipkart Gridlock 2.0 Hackathon. 

## 1. Overall Approach and Methodology

Our approach was built around creating a robust, multi-model ensemble designed to capture complex spatiotemporal patterns in traffic demand.

*   **Problem Formulation**: Time-series regression forecasting traffic demand (`0` to `1` scale) for every 15-minute slot for given geohashes.
*   **Ensemble Strategy**: We used an ensemble of three state-of-the-art gradient boosting frameworks:
    *   **LightGBM**: Fast, handles categorical data well, excellent baseline.
    *   **XGBoost**: Robust to outliers, builds deeper trees, prevents overfitting via strict regularization.
    *   **CatBoost**: Naturally handles categorical features (like weather, road type) and provides structural diversity to the ensemble.
*   **Validation Strategy**: We used a strict time-based split on Day 48 (the last 6 hours were held out for validation) to simulate the forward-looking nature of the test set.

## 2. Feature Engineering

Feature engineering was the most critical aspect of our success. We generated highly predictive features by combining spatial, temporal, and historical data.

### Temporal Features
*   `time_slot` (0 to 95): Representing the 15-minute intervals.
*   `hour` and `minute`.
*   `sin_slot`, `cos_slot`: Cyclical transformations of the time slot to help models understand the continuous nature of time (e.g., 23:45 is next to 0:00).
*   `is_weekend`, `day_of_week`.

### Spatial and Location Features
*   We parsed `geohash6` coordinates into `latitude` and `longitude`.
*   We processed static map data like `road_type`, `is_highway`, `is_street`, and `NumberofLanes`.
*   **Target Encoding**: We calculated aggregated statistics for each geohash (mean, median, std, max of historical demand) to give the models a baseline "busyness" factor for every location.

### Historical Lag Features
We implemented a dual-lag strategy to capture both immediate momentum and historical patterns:
1.  **Static Proxy Lags**: Using the exact same time slot from the previous day (Day 48) as a proxy for the test day (Day 49).
2.  **History-based Lags**: Calculating the demand from $T-1$, $T-2$, $T-3$, and $T-4$ slots (15 to 60 minutes prior) using sliding windows.
3.  **Rolling Statistics**: Rolling mean over the last 2 and 4 slots to capture short-term traffic build-up or dissipation.
4.  **Differencing**: First-order difference ($T$ minus $T-1$) to capture the rate of change in traffic (is traffic getting worse or clearing up?).

## 3. Modeling and Hyperparameter Tuning

We utilized **Optuna**, an advanced hyperparameter optimization framework, to automatically tune our models.

*   **Metric**: $R^2$ Score (Coefficient of Determination).
*   **Trials**: 40 trials for LightGBM, 30 for XGBoost, and 25 for CatBoost.
*   **Optimization**: We searched over spaces for `learning_rate`, `max_depth`, `num_leaves`, `subsample`, `colsample_bytree`, and `L1/L2 regularization`.

### Results
Our models achieved near-perfect validation scores on the held-out Day 48 data:
*   **LightGBM**: $R^2 = 99.9516\%$
*   **XGBoost**: $R^2 = 99.9335\%$
*   **CatBoost**: $R^2 = 99.9311\%$

## 4. The Final Ensemble

The models were blended using a weighted average based on their validation $R^2$ performance:
*   `Ensemble = (0.355 * LightGBM) + (0.324 * XGBoost) + (0.321 * CatBoost)`
*   **Final Ensemble Validation $R^2$**: **99.9560%** (RMSE = 0.0022)

After determining the optimal hyperparameters and ensemble weights, **we retrained all three models on the complete 100% of the training data** to ensure they learned from every available data point before predicting the test set.

## 5. Tools and Tech Stack

*   **Language**: Python 3.12
*   **Data Processing**: `pandas`, `numpy`
*   **Machine Learning**: `lightgbm`, `xgboost`, `catboost`, `scikit-learn`
*   **Optimization**: `optuna`
*   **Environment**: Windows PowerShell

## 6. Relevant Source Files

*   [**`run_pipeline.py`**](file:///C:/Users/stiwa/Downloads/Hackathon/Gridlock/dataset/DATASET/run_pipeline.py): The main orchestrator script that runs feature engineering, Optuna training, ensembling, full retraining, and final CSV generation.
*   [**`src/features.py`**](file:///C:/Users/stiwa/Downloads/Hackathon/Gridlock/dataset/DATASET/src/features.py): Contains the `FeatureEngine` class responsible for building temporal, spatial, and complex historical lag features.
*   [**`src/train.py`**](file:///C:/Users/stiwa/Downloads/Hackathon/Gridlock/dataset/DATASET/src/train.py): Houses the `ModelTrainer` class and all Optuna objective functions.
*   [**`src/utils.py`**](file:///C:/Users/stiwa/Downloads/Hackathon/Gridlock/dataset/DATASET/src/utils.py): Helper functions for timestamp parsing, geohash decoding, and submission validation.
*   [**`submissions/submission_ensemble.csv`**](file:///C:/Users/stiwa/Downloads/Hackathon/Gridlock/dataset/DATASET/submissions/submission_ensemble.csv): The final prediction file formatted for submission.

## Conclusion
The combination of rigorous feature engineering (especially the dual-lag strategy) and a well-tuned ensemble of three distinct gradient-boosted trees resulted in a highly predictive and robust model capable of scoring an $R^2$ of ~99.95% on validation data.
