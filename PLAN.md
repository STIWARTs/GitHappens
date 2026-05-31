# Gridlock Demand Model Plan

Build a rules-compliant, generalizable demand prediction pipeline focused on time-aware validation and strong tabular modeling. Use CatBoost on GPU as the primary model (best for mixed categorical + numeric data), add LightGBM only if it improves validation, and produce a clean submission plus an approach document for the source upload. Tools planned: Python with pandas, numpy, scikit-learn for preprocessing; CatBoost GPU and LightGBM for modeling; Optuna for tuning; matplotlib or seaborn for EDA.

## Steps
1. Phase 1 - Data audit and schema confirmation: Load train, test, and sample submission; verify column names, types, and missingness; confirm target range and distribution; check for duplicates across (geohash, day, timestamp) and for day distribution shift (train vs test).
2. Phase 1 - Evaluation and split strategy: Define validation splits that respect time (hold out latest day or latest time blocks) and group by geohash where possible; compute baseline R2; use out-of-fold encoding to prevent target leakage during feature engineering.
3. Phase 2 - Feature engineering (core): Parse timestamp into hour and minute; add cyclical time-of-day features (sin, cos), time buckets (15-min or 30-min), and day trend features.
4. Phase 2 - Feature engineering (spatial and categorical): Decode geohash to lat/lon; add geohash prefix features (coarser spatial bins) and spatial clusters; treat RoadType, Weather, LargeVehicles, Landmarks as categoricals for CatBoost; add interactions such as RoadType x Lanes, Weather x Temperature, RoadType x LargeVehicles.
5. Phase 2 - Missingness handling: Add missing-value indicators for RoadType, Weather, Temperature; impute Temperature with median per geohash or geohash+time bucket; set missing categoricals to an explicit Unknown category.
6. Phase 3 - Modeling: Train CatBoostRegressor on GPU with early stopping; tune depth, learning_rate, l2_leaf_reg, subsample; compare to LightGBM with encoded categoricals if needed; track R2 on time-aware validation.
7. Phase 4 - Ensembling (optional): If LightGBM adds measurable validation gain, blend predictions via weighted average; choose weights by validation R2; keep it simple to avoid overfitting.
8. Phase 5 - Submission and packaging: Retrain best model on full training data; generate predictions for test; validate submission shape (41778 x 2), column names, and Index alignment; produce a concise approach document listing features, validation, tools, and run instructions; package notebook and any helper scripts in the required archive.

## Relevant files
- [DATASET/train.csv](DATASET/train.csv) - training data and target distribution checks
- [DATASET/test.csv](DATASET/test.csv) - scoring data for prediction
- [DATASET/sample_submission.csv](DATASET/sample_submission.csv) - submission schema validation
- [PS.md](PS.md) - task requirements and metric definition
- [COMMENTSECTION.md](COMMENTSECTION.md) - notes about leaderboard anomalies and leakage risk
- [AboutHackahton.md](AboutHackahton.md) - hackathon context and submission expectations

## Verification
1. Report missingness and basic stats for each column; confirm no unexpected schema drift between train and test.
2. Validate that time-aware splits produce stable R2 and avoid leakage (no future data in training folds).
3. Confirm model performance with a locked validation split and seed for reproducibility.
4. Check submission file: exact row count, correct column names, no NaNs, and Index order matching test.

## Decisions
- Use a rules-compliant approach with no external or leaked datasets; optimize for generalization to hidden tests.
- Primary model is CatBoost on GPU; add LightGBM only if it improves validation.
- Deliverables: notebook plus scripts and a short approach document in the source archive.

## Further considerations
1. If geohash decoding is noisy or unstable, prefer geohash prefix features over raw lat/lon.
2. If validation variance is high, reduce feature set or increase regularization before ensembling.
3. If target values are tightly bounded, consider mild clipping only when validation indicates improved R2.
