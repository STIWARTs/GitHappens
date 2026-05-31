# Flipkart Gridlock 2.0 — Traffic Demand Prediction: Implementation Plan

## 1. Problem Summary

**Goal**: Predict normalized traffic `demand` (float in [0, 1]) for **41,778 rows** on **Day 49** (timestamps `2:15` → `13:45`) across ~1,190 geohash locations.

**Metric**: `score = max(0, 100 * R²(actual, predicted))` — we maximize R² score.

**Training data**: 77,299 rows — **Day 48** (all 96 timestamps: `0:00`–`23:45`) + **Day 49** (first 9 timestamps: `0:00`–`2:00`).

**Test data**: Day 49, timestamps `2:15` → `13:45` (47 timestamps × ~889 geohashes per slot).

---

## 2. Critical EDA Findings

These findings fundamentally shape the modeling approach:

### 2.1. Geohash is the Dominant Predictor (~80% of variance)

| Metric | Value |
|--------|-------|
| Between-geohash variance explained | **79.9%** |
| Within-geohash variance | 0.004 |
| Total variance | 0.020 |
| Lag-1 autocorrelation (within geohash) | **0.9594** |

> [!IMPORTANT]
> **The single most important feature is the geohash identity itself.** Each location has a characteristic demand level. Time-of-day modulates this baseline, but the location baseline dominates.

### 2.2. "Static" Features are Synthetically Noisy — NOT Truly Static

| Feature | Geohashes with Inconsistent Values |
|---------|-----------------------------------|
| RoadType | 255 / 1,249 (20%) |
| NumberofLanes | **1,204 / 1,249 (96%)** |
| LargeVehicles | **1,183 / 1,249 (95%)** |
| Landmarks | **1,182 / 1,249 (95%)** |

> [!WARNING]
> **NumberofLanes, LargeVehicles, and Landmarks change across rows for the SAME geohash.** These were synthetically added with noise (confirmed by community analysis showing this dataset derives from the Grab AI 2019 challenge). Treat them as **noisy categorical features**, not ground truth. The model should learn to discount their noise.

### 2.3. Temperature and Weather are Deterministic

| Weather | Mean Temp | Std | Range |
|---------|-----------|-----|-------|
| Snowy | 3.59 | 3.00 | ≤7.0 |
| Rainy | 10.93 | 1.97 | ≤14.0 |
| Foggy | 16.49 | 1.43 | ~14–19 |
| Sunny | 24.02 | 4.03 | ≥19 |

Temperature and Weather are **perfectly correlated** — Weather is essentially a binned version of Temperature. Despite this, **Weather has almost zero impact on demand** (all categories ≈ 0.092 mean demand). Temperature/Weather are distractors.

### 2.4. RoadType and Lanes DO Matter (through interaction with geohash identity)

| RoadType | Mean Demand |
|----------|-------------|
| Highway | **0.616** |
| Street | **0.273** |
| Residential | 0.057 |

| Lanes | Mean Demand |
|-------|-------------|
| 4 | **0.606** |
| 5 | **0.611** |
| 1 | 0.087 |
| 2 | 0.076 |
| 3 | 0.077 |

Highway/high-lane locations have dramatically higher demand. But since these are noisy per-row values, the model should learn the **modal** road type per geohash.

### 2.5. Temporal Pattern — Clear Diurnal Cycle

```
Hour  Demand
 0:   0.057  (trough)
 5:   0.104
11:   0.117  (peak)
14:   0.107
18:   0.049  (evening trough)
23:   0.093
```

There's a bimodal pattern: demand rises 0→11 hours, dips through afternoon, troughs at ~18h, and rises again overnight. Test window (2:15→13:45) covers the **rising demand phase**.

### 2.6. Geohash Spatial Clustering

| Prefix | Count | Description |
|--------|-------|-------------|
| qp09 | 41,391 (53.5%) | Majority cluster |
| qp03 | 23,835 (30.8%) | Second cluster |
| qp0d | 6,175 (8.0%) | Sparse |
| qp06 | 2,511 | Very sparse |
| qp08 | 1,953 | Very sparse |
| qp02 | 1,434 | Very sparse |

### 2.7. Missing Values

| Feature | Train Missing | Test Missing |
|---------|---------------|-------------|
| RoadType | 600 (0.8%) | 324 (0.8%) |
| Temperature | 2,495 (3.2%) | 1,349 (3.2%) |
| Weather | 797 (1.0%) | 431 (1.0%) |

### 2.8. Geohash Coverage Overlap

- Train has **1,249** unique geohashes, Test has **1,190**
- **1,180** overlap — 10 geohashes in test are UNSEEN in training
- 69 geohashes in train don't appear in test

> [!CAUTION]
> 10 test geohashes have zero training history. For these, we must fall back to spatial neighbor averages or global statistics.

---

## 3. Modeling Strategy

### 3.1. Core Approach: Gradient Boosting Ensemble

Given the tabular nature with strong categorical/geohash features and the competition setting, **gradient boosting** is the clear winner:

| Model | Role |
|-------|------|
| **LightGBM** | Primary model — fast, handles categoricals natively, excellent with high-cardinality features |
| **XGBoost** | Secondary model — different tree-building strategy provides diversity |
| **CatBoost** | Tertiary model — native categorical handling, ordered boosting reduces overfitting |
| **Weighted Ensemble** | Final prediction = weighted average of all three |

### 3.2. Optional Neural Baseline

If time permits, a **TabNet** or **simple MLP** with entity embeddings for geohash could add ensemble diversity.

---

## 4. Feature Engineering Plan

### 4.1. Temporal Features (from `timestamp`)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `hour` | Hour of day (0–23) | Primary temporal driver |
| `minute` | Minute (0, 15, 30, 45) | Sub-hourly resolution |
| `time_slot` | `hour * 4 + minute // 15` (0–95) | Unique time index for ordered encoding |
| `sin_hour`, `cos_hour` | Cyclical encoding of hour | Captures 24h periodicity |
| `sin_slot`, `cos_slot` | Cyclical encoding of 96-slot index | Finer cyclical pattern |
| `is_morning_rush` | 7 ≤ hour ≤ 10 | Rush hour indicator |
| `is_evening_low` | 16 ≤ hour ≤ 20 | Low-demand period |
| `is_peak_demand` | 9 ≤ hour ≤ 13 | Peak demand window |

### 4.2. Geohash Identity Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `geohash_encoded` | Label/target encoding of geohash | Core predictor (80% variance) |
| `geohash_mean_demand` | Mean demand per geohash (Day 48) | Strongest single feature |
| `geohash_median_demand` | Median demand per geohash | Robust center |
| `geohash_std_demand` | Demand variability per geohash | Captures volatility |
| `geohash_max_demand` | Peak demand per geohash | Scale indicator |
| `geohash_min_demand` | Floor demand per geohash | Baseline |
| `geohash_q25`, `geohash_q75` | Quartiles | Distribution shape |
| `geohash_count` | Number of observations per geohash | Popularity/coverage proxy |

### 4.3. Geohash × Time Interaction Features (THE KEY FEATURES)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `geohash_hour_mean` | Mean demand for (geohash, hour) from Day 48 | **Most predictive interaction** |
| `geohash_hour_median` | Median demand for (geohash, hour) | Robust version |
| `geohash_hour_std` | Std demand for (geohash, hour) | Volatility at that time |
| `geohash_slot_mean` | Mean demand for (geohash, time_slot) from Day 48 | Finest grain — exact 15-min slot |
| `geohash_slot_last` | Last known demand at this (geohash, time_slot) | Direct observation |

> [!TIP]
> Since Day 48 has all 96 timestamps, we can compute exact (geohash, time_slot) lookup values. This alone could yield R² > 0.90 as a baseline.

### 4.4. Lag / Autoregressive Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `demand_lag_1` | Demand at previous 15-min slot for same geohash | Autocorr = 0.96! |
| `demand_lag_2` to `demand_lag_4` | 30, 45, 60 min lags | Short-term trend |
| `demand_rolling_mean_4` | Rolling mean of last 4 slots (1 hour) | Smoothed recent trend |
| `demand_rolling_mean_8` | Rolling mean of last 8 slots (2 hours) | Medium-term |
| `demand_diff_1` | demand_t - demand_{t-1} | Momentum/acceleration |
| `demand_day48_same_slot` | Demand on Day 48 at same (geohash, slot) | **Day-over-day** baseline |

> [!IMPORTANT]
> **Lag features require careful construction at inference time.** For the test set, only Day 49 0:00–2:00 is available as history. We'll use Day 49 training data as the seed, then can potentially do **iterative prediction** (predict 2:15, use it as lag for 2:30, etc.) or simply use Day 48 same-slot values as proxy lags.

### 4.5. Spatial / Geohash Neighbor Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `gh_prefix5` | First 5 chars of geohash (broader area) | Spatial cluster |
| `gh_prefix4` | First 4 chars (even broader) | Regional identity |
| `neighbor_mean_demand` | Mean demand of geohash neighbors (same prefix5) | Spatial smoothing |
| `spatial_rank` | Rank of geohash within its region by avg demand | Relative position |

### 4.6. Road Infrastructure Features (Noisy)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `road_type_encoded` | Ordinal: Residential=0, Street=1, Highway=2 | Major demand driver |
| `modal_road_type` | Most common road type per geohash | Denoise row-level noise |
| `modal_lanes` | Most common lane count per geohash | Denoised |
| `modal_large_vehicles` | Most common value per geohash | Denoised |
| `modal_landmarks` | Most common value per geohash | Denoised |
| `is_highway` | Binary: Highway road type | High-demand indicator |
| `is_high_capacity` | Lanes ≥ 4 | Infrastructure flag |

### 4.7. Weather Features (Low Impact but Include)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `temperature_filled` | Temperature with missing imputed by Weather group mean | Clean numeric |
| `weather_encoded` | Ordinal encoding (Snowy=0, Rainy=1, Foggy=2, Sunny=3) | Ordered by temperature |

---

## 5. Training Strategy

### 5.1. Data Split

```
┌──────────────────────────────────────────────────────────────┐
│  Day 48: All 96 timestamps (0:00–23:45) — 69,427 rows       │
│  → PRIMARY TRAINING DATA                                     │
├──────────────────────────────────────────────────────────────┤
│  Day 49: 9 timestamps (0:00–2:00) — 7,872 rows              │
│  → SECONDARY TRAINING DATA + LAG SEED                        │
├──────────────────────────────────────────────────────────────┤
│  Day 49: 47 timestamps (2:15–13:45) — 41,778 rows           │
│  → TEST SET (predict demand)                                 │
└──────────────────────────────────────────────────────────────┘
```

### 5.2. Validation Strategy

> [!IMPORTANT]
> Standard random K-fold will leak temporal information. Use **temporal validation**:

- **Approach A — Time-Based Split**: Train on Day 48 (0:00–17:45), validate on Day 48 (18:00–23:45). This simulates predicting future timestamps.
- **Approach B — Day-Based Split**: Train on Day 48, validate on Day 49 (0:00–2:00). Simulates the actual cross-day prediction task.
- **Approach C — Grouped Time K-Fold**: 5-fold where each fold holds out a contiguous block of 4 hours from Day 48.

We will use **Approach A + B combined** to tune hyperparameters and select features.

### 5.3. Handling Unseen Geohashes (10 in test)

For the 10 test geohashes with no training history:
1. Use **5-character prefix match** — find geohashes sharing the first 5 characters and use their aggregated stats
2. If no prefix5 match, use **prefix4 match**
3. Final fallback: global median demand for that time slot

### 5.4. Lag Feature Construction for Test

**Strategy**: Two-pass approach:
1. **Pass 1**: Build features using only Day 48 same-slot values as lag proxies → predict all test rows at once
2. **Pass 2** (if iterative improves): Sort test by timestamp, predict sequentially, feeding predictions back as lags

Pass 1 is safer and avoids error propagation. We'll benchmark both.

---

## 6. Model Hyperparameter Ranges

### LightGBM
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': [500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [6, 8, 10, -1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
    'categorical_feature': ['geohash_encoded', 'road_type_encoded', ...]
}
```

### XGBoost
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [6, 8, 10],
    'min_child_weight': [5, 10, 20],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 5]
}
```

### CatBoost
```python
{
    'iterations': [1000, 2000],
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [6, 8, 10],
    'l2_leaf_reg': [1, 3, 5],
    'cat_features': [geohash, RoadType, Weather, ...]
}
```

---

## 7. Ensemble Strategy

```
Final_Prediction = w1 * LightGBM + w2 * XGBoost + w3 * CatBoost
```

Weights determined by **validation R²** using inverse-error weighting or Optuna optimization.

Post-processing: clip predictions to `[0, 1]` range.

---

## 8. Tools & Libraries

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Runtime |
| **pandas** | Data manipulation |
| **numpy** | Numerical ops |
| **scikit-learn** | Preprocessing, metrics, base utilities |
| **LightGBM** | Primary model |
| **XGBoost** | Secondary model |
| **CatBoost** | Tertiary model |
| **Optuna** | Hyperparameter optimization |
| **matplotlib / seaborn** | Visualization & EDA plots |
| **python-geohash** | Geohash decoding (lat/lon extraction, neighbor lookup) |

---

## 9. File Structure

```
DATASET/
├── train.csv                  # Given training data
├── test.csv                   # Given test data
├── sample_submission.csv      # Submission format
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_modeling.ipynb      # Model training & evaluation
├── src/
│   ├── features.py            # Feature engineering pipeline
│   ├── train.py               # Model training script
│   ├── predict.py             # Inference / submission generation
│   ├── ensemble.py            # Ensemble & post-processing
│   └── utils.py               # Helper functions (geohash, validation, etc.)
├── models/                    # Saved model artifacts
├── submissions/               # Generated CSV submissions
└── approach.txt               # Text file explaining approach (required for upload)
```

---

## 10. Execution Roadmap

| Step | Task | ETA |
|------|------|-----|
| 1 | Feature engineering pipeline (`features.py`) | Core |
| 2 | Validation framework with temporal splits | Core |
| 3 | LightGBM baseline → target R² > 0.85 | Core |
| 4 | Add lag features + geohash×time interactions | Core |
| 5 | XGBoost + CatBoost training | Enhancement |
| 6 | Hyperparameter tuning with Optuna | Enhancement |
| 7 | Ensemble optimization | Enhancement |
| 8 | Handle unseen geohashes | Robustness |
| 9 | Generate final submission CSV | Deliverable |
| 10 | Write approach.txt + package source code | Deliverable |

---

## 11. Expected Baseline Performance

| Approach | Expected R² |
|----------|-------------|
| Global mean prediction | ~0.00 |
| Per-geohash mean demand (Day 48) | ~0.80 |
| Per-geohash × hour mean demand | ~0.90 |
| Per-geohash × time_slot mean demand | ~0.92 |
| + Lag features + GBDT model | ~0.94–0.96 |
| + Ensemble + tuning | ~0.96–0.98 |

---

## 12. Verification Plan

### Automated Checks
1. **Submission format validation**: Ensure output is 41,778 × 2 with columns `Index, demand`
2. **Cross-validation R² on temporal hold-out**: Must exceed 0.85 before submission
3. **Sanity checks**: All predictions in [0, 1], no NaN values
4. **Feature importance analysis**: Confirm geohash and time features dominate

### Manual Verification
1. Plot predicted vs actual for validation set
2. Inspect predictions for known high-demand geohashes (e.g., `qp09d9` should predict near 1.0)
3. Check predictions for unseen geohashes are reasonable
4. Compare submission statistics (mean, std, distribution) with training data patterns

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Should I prioritize a single strong LightGBM model first and iterate, or build the full 3-model ensemble from the start? A single LightGBM with strong features may be sufficient for a top score. The ensemble adds complexity but typically 1–3% improvement.

> [!IMPORTANT]
> **Q2**: For lag features at inference time, do you prefer:
> - **(A) Static proxy** — Use Day 48 same-slot demand as lag (simpler, no error propagation)
> - **(B) Iterative prediction** — Predict slot-by-slot, feeding predictions as lags (riskier but potentially more accurate)
> - **(C) Both** — Try both, pick whichever validates better

> [!NOTE]
> **Re: Leaderboard situation**: The community has identified that 100+ teams achieved perfect R²=100 through reverse engineering the test labels from the public Grab AI 2019 dataset. The final evaluation will use **hidden test data** with a distributional shift. Our strategy of building a genuinely robust model with strong features will perform well on the hidden set, unlike overfitted/leaked submissions.
