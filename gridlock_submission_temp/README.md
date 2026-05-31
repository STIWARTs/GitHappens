# Gridlock 2.0 Traffic Demand Prediction - Submission Package

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Validate Model (Optional - Test on Day 49 from training data)
```bash
python -m src.train --mode validate --model sklearn --config configs/model.yaml
```

**Expected Output:**
```
R² Score: 0.765
MAE: 0.0399
RMSE: 0.0702
```

### 3. Generate Submission
```bash
python -m src.train --mode full --model sklearn --config configs/model.yaml
python -m src.predict \
  --model-path outputs/model_full.sklearn \
  --builder-path outputs/feature_builder.pkl \
  --output-path outputs/submission.csv
```

**Output:** `outputs/submission.csv` (41,778 predictions, ready for submission)

---

## Solution Overview

**Problem:** Predict normalized traffic demand across 41,778 test locations/times

**Approach:**
- **Feature Engineering:** Temporal (time-of-day cycles), Spatial (geohash decoding), Infrastructure (road type, lanes), Environmental (weather, temperature)
- **Modeling:** sklearn HistGradientBoostingRegressor (no heavy dependencies)
- **Validation:** Day 48 train → Day 49 test (mimics distribution shift)
- **Performance:** R² = 0.765 on validation

**Key Features:**
- K-Fold target encoding (prevents leakage, regularizes)
- Cyclical time encoding (24-hour periodicity)
- Geohash hierarchical decoding (1km to 100km scales)
- Temperature imputation by weather context
- ~28 engineered features from 11 raw columns

---

## Package Structure

```
gridlock_submission/
├── APPROACH.txt              ← Ultra-detailed methodology (THIS FILE)
├── README.md                 ← Quick start guide
├── requirements.txt          ← Dependencies
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py    (FeatureBuilder, TargetEncoder, geohash)
│   ├── metrics.py                (regression_metrics)
│   ├── train.py                  (main training pipeline)
│   └── predict.py                (generate submission.csv)
├── configs/
│   └── model.yaml            (HistGradientBoosting hyperparameters)
├── notebooks/
│   └── 01_eda.ipynb          (exploratory data analysis + examples)
└── outputs/
    ├── model_validate.sklearn    (pickled model, validate mode)
    ├── model_full.sklearn        (pickled model, full mode)
    ├── feature_builder.pkl       (pickled FeatureBuilder)
    ├── model_meta.json           (metadata)
    └── submission.csv            (final predictions)
```

---

## Key Files Explained

### `src/feature_engineering.py` (350+ lines)
Orchestrates all feature engineering:
- **`decode_geohash()`** → Converts 6-char geohash to (lat, lon)
- **`TargetEncoder`** → K-Fold target encoding with James-Stein smoothing
- **`FeatureBuilder`** → Main class that:
  - Parses timestamps ("H:M" → minutes, cyclical encoding)
  - Decodes geohashes and creates hierarchical keys
  - Imputes missing temperatures (by weather context)
  - Applies target encoding (prevents leakage)
  - Produces ~28 engineered features

### `src/train.py` (270 lines)
Training orchestration:
- `split_by_day()` → Train on Day 48, validate on Day 49
- `train_sklearn()` → HistGradientBoostingRegressor trainer
- Handles config loading, model serialization, metrics reporting

### `src/predict.py` (60 lines)
Inference pipeline:
- Loads trained model and FeatureBuilder
- Transforms test.csv using fitted statistics
- Generates `submission.csv` with Index + demand columns

### `configs/model.yaml`
Hyperparameters for HistGradientBoosting:
- `n_estimators: 2500`
- `learning_rate: 0.05`
- `max_depth: 8`

### `notebooks/01_eda.ipynb`
Exploratory data analysis + feature demo

---

## Feature Engineering Details

### 1. **Temporal Features** (8 total)
- `time_bin_15`: Quarter-hour index (0–95)
- `time_sin`, `time_cos`: Cyclical encoding (prevents midnight boundary artifact)
- `hour`, `minute`: Time components
- `day`, `is_day49`: Day indicators

### 2. **Spatial Features** (6 total)
- `latitude`, `longitude`: Decoded from geohash
- `geo_distance`: Distance from dataset centroid
- `geohash_4`, `geohash_5`: Hierarchical grouping keys (100km → 10km scales)

### 3. **Infrastructure Features** (5 total)
- `large_vehicles_allowed`: Binary (LargeVehicles == "Allowed")
- `has_landmarks`: Binary (Landmarks == "Yes")
- `NumberofLanes`: Continuous [1, 3]
- `RoadType`: Categorical {Residential, Street}
- Plus interaction keys (`geohash_time`, `roadtype_time`, etc.)

### 4. **Environmental Features** (2 total)
- `Temperature`: Imputed by (geohash, weather) groups
- `Weather`: Categorical {Sunny, Rainy, Foggy, Snowy}

### 5. **Target-Encoded Features** (4 total)
- `geohash_te`: Average demand per location
- `geohash_time_te`: Average demand per (location, time)
- `roadtype_time_te`: Average demand per (road type, time)
- `weather_time_te`: Average demand per (weather, time)

**K-Fold Target Encoding Process:**
```
Per unique value in column:
  encoded_value = (sum(targets) + global_mean × smoothing) / (count + smoothing)

K-Fold: Training fold targets never used to encode same fold (prevents leakage)
```

---

## Data Flow

### Training:
```
train.csv (77,299 rows)
    ↓
[FeatureBuilder.fit_transform]
  • Parse timestamps, decode geohashes
  • Impute temperatures
  • Create temporal/spatial/infrastructure features
  • Apply K-Fold target encoding (5 folds)
    ↓
engineered features (Day 48 + Day 49)
    ↓
[Train/Val Split]: Day 48 → train, Day 49 → validate
    ↓
[HistGradientBoostingRegressor.fit]
    ↓
outputs/model_full.sklearn (entire train.csv)
outputs/model_validate.sklearn (Day 48 only)
```

### Prediction:
```
test.csv (41,778 rows, Day 49)
    ↓
[FeatureBuilder.transform]
  (using fitted statistics from training)
    ↓
engineered test features
    ↓
[model.predict]
    ↓
submission.csv [Index, demand]
```

---

## Model Performance

### Validation (Day 48 train → Day 49 test split):
- **R² = 0.765** (76.5% variance explained)
- **MAE = 0.0399** (mean absolute error)
- **RMSE = 0.0702** (root mean squared error)

### Interpretation:
- Strong generalization to unseen day (addresses distribution shift)
- Average prediction error ~0.04 on [0, 1] normalized scale
- Model captures major demand patterns

---

## Tools & Dependencies

**Installed:**
- `numpy` 2.3.5 — numerical computations
- `pandas` 2.3.3 — data manipulation
- `scikit-learn` 1.7.2 — ML models + metrics
- `joblib` 1.5.2 — serialize/deserialize objects
- `pyyaml` 6.0.3 — config parsing

**Why no CatBoost/LightGBM?**
Disk space constraints (~100MB each). sklearn's HistGradientBoosting is:
- ✓ Built-in (no extra download)
- ✓ Efficient histogram-based boosting
- ✓ Achieves competitive R² (0.765)

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy train.csv, test.csv to DATASET/ (download from competition)

# 3. Validate on Day 49 (optional)
python -m src.train --mode validate --model sklearn --config configs/model.yaml

# 4. Train on full data
python -m src.train --mode full --model sklearn --config configs/model.yaml

# 5. Generate submission
python -m src.predict \
  --model-path outputs/model_full.sklearn \
  --builder-path outputs/feature_builder.pkl \
  --output-path outputs/submission.csv

# 6. Submit outputs/submission.csv to Flipkart Gridlock
```

---

## Validation Checklist

- ✓ All 77,299 training rows processed (Day 48 + Day 49)
- ✓ All 41,778 test rows predicted (Day 49)
- ✓ Timestamps parsed correctly (96 unique 15-min intervals per day)
- ✓ Geohash decoding verified (6-char → lat/lon)
- ✓ Temperature imputation by weather context
- ✓ K-Fold target encoding prevents leakage
- ✓ Validation set created from Day 49 (mimics test distribution)
- ✓ Features applied identically to train/test
- ✓ No NaN values in model input (all imputed/encoded)
- ✓ Submission shape matches test.csv (41,778 rows, 2 columns: Index, demand)

---

## Key Insights

1. **Day 48 → Day 49 Distribution Shift**: Explicitly validated on Day 49 to ensure generalization. Many naive models overfit to Day 48 and fail on Day 49.

2. **K-Fold Target Encoding**: Prevents target leakage while creating compact (~4 features) replacements for high-cardinality categoricals (geohash, road type, weather).

3. **Geohash Hierarchical Decoding**: Captures spatial patterns at multiple scales (1km to 100km).

4. **Contextual Temperature Imputation**: Group by weather before global mean ensures weather-temperature relationships are preserved.

5. **Cyclical Time Encoding**: Models learn 24-hour wraparound naturally without artificial midnight boundary.

---

## Future Improvements

- **Ensemble**: Combine CatBoost + LightGBM + HistGradientBoosting (often +1–3% R²)
- **Hyperparameter tuning**: Optuna for learning_rate, max_depth, smoothing
- **Time-series cross-validation**: Respect temporal ordering
- **External data**: Weather forecasts, event calendars, public transport schedules
- **Neural networks**: Graph neural networks over geohash regions, LSTM/Transformer for temporal dynamics

---

## Contact & Metadata

**Submission:** Individual participant
**Hackathon:** Flipkart Gridlock 2.0 (HackerEarth)
**Date:** May 31, 2026
**Validation R²:** 0.765
**Python:** 3.11+
**Model:** sklearn HistGradientBoostingRegressor

---

**For detailed technical documentation, see `APPROACH.txt` in this package.**
