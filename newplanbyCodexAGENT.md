<!-- read @file:PLAN.md , @file:ChatgptThinking.md and @file:PS.md and make a newplanbyCodexAGENT.md file which contain detailed plan approach etc -->

# Gridlock Demand Prediction - Detailed Plan

## 1. Objective and constraints
Build a rules-compliant model to predict `demand` for each row in test data. Focus on generalization to hidden tests, not leaderboard leakage.

- Metric: `score = max(0, 100 * R2(actual, predicted))`
- Output format: CSV with columns `Index` and `demand`, shape 41778 x 2.
- Submission requires a notebook (.ipynb) plus a short approach document and source files.

## 2. Dataset summary
From the provided problem statement and data:

- Train: 77299 rows, 11 columns (includes `demand`).
- Test: 41778 rows, 10 columns (no `demand`).
- Key columns: `geohash`, `day`, `timestamp`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`.

Hypothesis from inspection: demand is mostly driven by location and time, with weather and road context as secondary modifiers.

## 3. High-level approach
Use a strong tabular model with native categorical handling (CatBoost) plus careful feature engineering focused on:

- Spatial hierarchy (geohash prefixes)
- Temporal structure (hour/minute + cyclical encoding)
- Robust handling of missing data
- Day-aware validation that simulates the test distribution

Only add a second model (LightGBM) if it provides stable validation gains.

## 4. Validation and evaluation strategy
The validation plan prevents leakage and mimics test distribution.

### 4.1 Primary validation (Day-based split)
- Train: Day 48
- Validate: Day 49

This aligns with the test scenario (test is day 49). It is the main metric for model selection.

### 4.2 Secondary validation (Time-based split within Day 48)
- Train: earlier timestamps
- Validate: later timestamps

This checks temporal robustness and overfitting to specific time intervals.

### 4.3 Notes
- Use a fixed random seed for reproducibility.
- Track both validation schemes; pick model hyperparameters that perform well across both.

## 5. Feature engineering plan
Feature engineering is expected to contribute more to performance than model choice.

### 5.1 Geohash hierarchy
Create coarse spatial features using geohash prefixes (no lat/lon required):

- `geo_2`, `geo_4`, `geo_5`, `geo_6` from full geohash.
- Example: `qp02zt` -> `qp`, `qp02`, `qp02z`, `qp02zt`.

Rationale: captures neighborhood-level patterns without noisy decoding.

### 5.2 Time features
Parse `timestamp` into `hour` and `minute`.

- Numerical: `hour`, `minute`, `time_minutes = hour * 60 + minute`.
- Cyclical: `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`.
- Buckets: 15-min or 30-min bin (`time_bin`).

### 5.3 Day interaction features
- `day x hour` (as a categorical or combined string).
- `day x weather`.

This helps model shifts between Day 48 and Day 49.

### 5.4 Missing-value indicators
Create binary indicators:

- `temp_missing`, `weather_missing`, `road_missing`.

Impute:

- `Temperature`: median by geohash or geohash + time_bin.
- Categoricals: fill with `Unknown`.

### 5.5 Demand priors (out-of-fold)
These are powerful but must be computed without leakage.

- `mean_demand_by_geohash`, `median_demand_by_geohash`, `std_demand_by_geohash`.
- `mean_demand_by_hour`.
- `mean_demand_by_geohash_hour`.

Use out-of-fold (OOF) computation within training folds to avoid target leakage.

## 6. Modeling plan
### 6.1 Baseline
Start with a simple model using only core features (geohash, day, timestamp) to establish a baseline R2.

### 6.2 Main model: CatBoostRegressor
- Use GPU mode (RTX 3050 supports this).
- Categorical features: `geohash`, geohash prefixes, `RoadType`, `Weather`, `LargeVehicles`, `Landmarks`, and interaction strings.
- Early stopping on the validation set.

Key hyperparameters to tune (Optuna or manual sweeps):

- `depth`
- `learning_rate`
- `l2_leaf_reg`
- `subsample`
- `iterations`

### 6.3 Optional second model (LightGBM)
Only if CatBoost gains plateau.

- Encode categoricals with target encoding or CatBoost encoding.
- Compare R2 under the same validation schemes.

### 6.4 Optional ensemble
If LightGBM adds measurable gain, blend predictions:

- Weighted average optimized on validation R2.
- Keep weights simple to reduce overfitting.

## 7. Leakage prevention and robustness
- No external datasets or hidden test probing.
- All target-driven features must be computed OOF.
- Time-aware validation prevents future leakage.

## 8. Experiment tracking
Log all experiments with:

- Feature set version
- Validation split used
- Hyperparameters
- R2 scores for both validation schemes

This ensures reproducibility and confidence when retraining on full data.

## 9. Deliverables
### 9.1 Required files
- `notebook.ipynb`: end-to-end pipeline (EDA, features, training, prediction)
- `submission.csv`: final predictions
- `approach.md`: summary of method, features, tools, and run instructions
- Optional helper scripts: `features.py`, `train.py`, `predict.py`

### 9.2 Submission checks
- Confirm CSV columns: `Index`, `demand`
- Row count: 41778
- No missing values
- Index order matches test file

## 10. Tools and environment
- Python 3.10+
- pandas, numpy
- scikit-learn
- catboost (GPU)
- lightgbm (optional)
- optuna (optional)
- matplotlib or seaborn

## 11. Risks and mitigations
- Risk: Day 49 shift vs Day 48.
  - Mitigation: Day-based validation + interactions with `day`.
- Risk: Overfitting to geohash + time priors.
  - Mitigation: OOF priors and validation on multiple splits.
- Risk: Leakage in categorical encodings.
  - Mitigation: Use fold-aware encodings only.

## 12. Execution checklist
1. Load data and verify schema.
2. Generate feature set v1 (geohash prefixes + time features).
3. Train CatBoost baseline and evaluate on Day-based validation.
4. Add missing indicators and priors; re-evaluate.
5. Tune hyperparameters; lock the best model.
6. Train final model on full training data.
7. Generate submission and validate format.
8. Package notebook and approach document for upload.
