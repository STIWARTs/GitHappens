# Flipkart Gridlock 2.0 — Traffic Demand Prediction: Implementation Plan (v2 — Corrected)

> [!IMPORTANT]
> **This is the corrected plan.** The previous version had two critical errors identified by verification:
> 1. Geohash variance explained was **69%**, not 80%
> 2. Lag-1 autocorrelation was **0.20**, not 0.96
>
> Both corrections materially change the modeling strategy. All sections below reflect the corrected data properties.

---

## 1. Problem Summary

**Goal**: Predict normalized traffic `demand` (float in [0, 1]) for **41,778 rows** on **Day 49** (timestamps `2:15` → `13:45`) across ~1,190 geohash locations.

**Metric**: `score = max(0, 100 * R²(actual, predicted))` — we maximize R² score.

**Training data**: 77,299 rows — **Day 48** (all 96 timestamps: `0:00`–`23:45`) + **Day 49** (first 9 timestamps: `0:00`–`2:00`).

**Test data**: Day 49, timestamps `2:15` → `13:45` (47 timestamps × ~889 geohashes per slot). **test.csv is not yet in the repo** — it will be provided later for generating final predictions.

**Submission**: 41,778 × 2 CSV with columns `Index, demand`.

**Deliverables**: `.ipynb` notebook file + prediction CSV + approach text file (zipped).

---

## 2. Corrected EDA Findings

### 2.1. Geohash is the Dominant Predictor (~69% of variance) — CORRECTED

| Metric | **Corrected Value** | ~~Previous Claim~~ |
|--------|---------------------|---------------------|
| Between-geohash variance explained | **69.4%** | ~~79.9%~~ |
| Within-geohash variance | **30.6%** | ~~20.1%~~ |

> [!IMPORTANT]
> Geohash identity explains ~69% of demand variance — still the single strongest predictor, but **31% of variance remains unexplained** by geohash alone. This means temporal features, geohash×time interactions, and road infrastructure features are **more important than previously assumed** and must be given significant modeling weight.

### 2.2. Lag-1 Autocorrelation is WEAK — CORRECTED

| Metric | **Corrected Value** | ~~Previous Claim~~ |
|--------|---------------------|---------------------|
| Lag-1 autocorrelation (mean across geohashes) | **0.2014** | ~~0.9594~~ |
| Median autocorrelation | **0.3228** | — |
| Std of autocorrelation | **0.5124** | — |
| Range | **[-1.0, 0.852]** | — |

> [!CAUTION]
> **This is the biggest correction.** Autocorrelation is ~5× weaker than previously claimed and highly variable across geohashes. Key implications:
> - **Iterative prediction (predict → feed as lag → predict next) is DANGEROUS** — error propagation will be severe with autocorr ≈ 0.20
> - **Day 48 same-slot lookup is the correct lag proxy strategy**
> - Lag features should be used cautiously and weighted lower than geohash×time interaction features

### 2.3. "Static" Features are Synthetically Noisy — NOT Truly Static

| Feature | Geohashes with Inconsistent Values |
|---------|-----------------------------------|
| RoadType | 255 / 1,249 (20.4%) |
| NumberofLanes | 1,204 / 1,249 (96.4%) |
| LargeVehicles | 1,183 / 1,249 (94.7%) |
| Landmarks | 1,182 / 1,249 (94.6%) |

> [!WARNING]
> NumberofLanes, LargeVehicles, and Landmarks change across rows for the SAME geohash. Treat as **noisy categorical features**. Use the **modal (most frequent) value per geohash** as the denoised version.

### 2.4. Temperature and Weather are Deterministic Distractors

| Weather | Mean Temp | Range |
|---------|-----------|-------|
| Snowy | 3.59 | ≤7.0 |
| Rainy | 10.93 | ≤14.0 |
| Foggy | 16.49 | ~14–19 |
| Sunny | 24.02 | ≥19 |

Temperature → Weather mapping is strict (Weather = binned Temperature). **Weather has near-zero impact on demand** (all categories ≈ 0.092 mean demand). Include as low-priority features only.

### 2.5. RoadType and Lanes DO Matter

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

Highway / high-lane locations have dramatically higher demand. Despite row-level noise, **modal road type per geohash** is a strong signal.

### 2.6. Temporal Pattern — Clear Diurnal Cycle ✅ Verified

```
Hour  Demand
 0:   0.083  (trough)
 5:   0.104  (rising)
11:   0.117  (peak)
14:   0.107  (declining)
18:   0.049  (evening trough)
23:   0.093  (overnight recovery)
```

Bimodal pattern confirmed. Test window (2:15→13:45) covers the **rising demand phase** into the peak.

### 2.7. Geohash Spatial Clustering

| Prefix | Count | % of Train |
|--------|-------|------------|
| qp09 | 41,391 | 53.5% |
| qp03 | 23,835 | 30.8% |
| qp0d | 6,175 | 8.0% |
| qp06 | 2,511 | 3.2% |
| qp08 | 1,953 | 2.5% |
| qp02 | 1,434 | 1.9% |

### 2.8. Missing Values

| Feature | Train Missing | Test Missing |
|---------|---------------|-------------|
| RoadType | 600 (0.8%) | 324 (0.8%) |
| Temperature | 2,495 (3.2%) | 1,349 (3.2%) |
| Weather | 797 (1.0%) | 431 (1.0%) |

### 2.9. Geohash Coverage Overlap

- Train: **1,249** unique geohashes, Test: **1,190** unique geohashes
- **1,180** overlap — **10 geohashes in test are UNSEEN** in training
- 69 geohashes in train don't appear in test

> [!CAUTION]
> 10 test geohashes have zero training history. Fall back to spatial neighbor averages or global statistics for these.

---

## 3. Modeling Strategy (Revised)

### 3.1. Primary Approach: LightGBM-First, Then Ensemble

Given the corrected data properties — geohash explains 69% (not 80%) and autocorrelation is weak (0.20) — the model must lean **more on feature interactions than on raw lag/autoregressive signals**.

| Model | Role | Priority |
|-------|------|----------|
| **LightGBM** | Primary model — fast, handles categoricals natively, excellent with high-cardinality features | **P0 — Build first** |
| **XGBoost** | Ensemble diversity — different tree-building strategy | P1 — Add after LightGBM baseline |
| **CatBoost** | Ensemble diversity — native categorical handling, ordered boosting | P1 — Add after LightGBM baseline |
| **Weighted Ensemble** | Final = weighted average of top models | P2 — Optimize after individual models tuned |

> [!TIP]
> **Build LightGBM first and iterate.** Only add XGBoost/CatBoost if LightGBM R² plateaus below 0.93. The 3-model ensemble typically adds 1–3% R² but increases complexity significantly.

### 3.2. Key Strategy Shifts from v1

| Aspect | v1 (Incorrect) | v2 (Corrected) |
|--------|----------------|-----------------|
| Geohash reliance | "80% of variance" → over-rely on geohash | 69% → balance geohash with temporal + interaction features |
| Lag features | "Autocorr 0.96" → heavy lag emphasis | Autocorr 0.20 → use Day 48 same-slot lookup as primary, de-emphasize lags |
| Iterative prediction | Considered as viable option | **Eliminated** — too risky with weak autocorrelation |
| Temporal feature weight | Secondary to geohash | **Elevated to co-primary** with geohash |

---

## 4. Feature Engineering Plan (Revised Priority Order)

### 4.1. ⭐ Geohash × Time Interaction Features (HIGHEST PRIORITY — Core of 31% unexplained variance)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `geohash_hour_mean` | Mean demand for (geohash, hour) from Day 48 | **Most predictive interaction** — captures location-specific time patterns |
| `geohash_hour_median` | Median demand for (geohash, hour) | Robust version, less affected by outliers |
| `geohash_hour_std` | Std demand for (geohash, hour) | Volatility at that time |
| `geohash_slot_mean` | Mean demand for (geohash, time_slot) from Day 48 | **Finest grain — exact 15-min slot lookup** |
| `geohash_slot_last` | Last known demand at this (geohash, time_slot) on Day 48 | Direct observation from previous day |

> [!TIP]
> Since Day 48 has all 96 timestamps, we can compute exact (geohash, time_slot) lookup values. **This alone could yield R² > 0.90 as a baseline**, before any model is trained.

### 4.2. ⭐ Geohash Identity Features (HIGH PRIORITY)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `geohash_encoded` | Label/target encoding of geohash | Core predictor (69% variance) |
| `geohash_mean_demand` | Mean demand per geohash (Day 48) | Strongest single feature |
| `geohash_median_demand` | Median demand per geohash | Robust center |
| `geohash_std_demand` | Demand variability per geohash | Captures volatility |
| `geohash_max_demand` | Peak demand per geohash | Scale indicator |
| `geohash_min_demand` | Floor demand per geohash | Baseline |
| `geohash_q25`, `geohash_q75` | Quartiles | Distribution shape |
| `geohash_count` | Number of observations per geohash | Popularity/coverage proxy |

### 4.3. ⭐ Temporal Features (HIGH PRIORITY — Elevated from v1)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `hour` | Hour of day (0–23) | Primary temporal driver |
| `minute` | Minute (0, 15, 30, 45) | Sub-hourly resolution |
| `time_slot` | `hour * 4 + minute // 15` (0–95) | Unique time index |
| `sin_hour`, `cos_hour` | Cyclical encoding of hour | Captures 24h periodicity |
| `sin_slot`, `cos_slot` | Cyclical encoding of 96-slot index | Finer cyclical pattern |
| `is_morning_rush` | 7 ≤ hour ≤ 10 | Rush hour indicator |
| `is_peak_demand` | 9 ≤ hour ≤ 13 | Peak demand window |
| `is_evening_low` | 16 ≤ hour ≤ 20 | Low-demand period |
| `hour_squared` | Polynomial of hour | Non-linear temporal pattern |

### 4.4. Day 48 Same-Slot Proxy Features (REPLACES Lag Strategy from v1)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `demand_day48_same_slot` | Demand on Day 48 at same (geohash, time_slot) | **Primary day-over-day baseline** — safe, no error propagation |
| `demand_day48_prev_slot` | Demand on Day 48 at (geohash, time_slot - 1) | Approximate "lag-1" from previous day |
| `demand_day48_next_slot` | Demand on Day 48 at (geohash, time_slot + 1) | Forward-looking smoothing |
| `demand_day48_rolling_4` | Rolling mean of 4 Day 48 slots centered on target | Smoothed temporal context |
| `demand_day48_slot_diff` | Day 48 slot demand minus Day 48 geohash mean | Deviation from location baseline at that time |

> [!IMPORTANT]
> **Why Day 48 lookup instead of true lags?**
> With autocorrelation at 0.20 (not 0.96), lag features don't carry strong predictive signal. Day 48 same-slot values capture the **periodic daily pattern** directly — a much stronger signal for this data.

### 4.5. Limited True Lag Features (LOW PRIORITY — Use only where data exists)

For test timestamps 2:15–2:45 only, we have Day 49 training data (0:00–2:00) that can provide true recent lags:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `demand_lag_1` | Demand at (geohash, previous 15-min slot) on Day 49 | Only available for first ~3 test slots |
| `demand_recent_mean` | Mean of available Day 49 slots for same geohash | Recent-day level adjustment |

> [!WARNING]
> True lag features will be **NaN for most test rows** (only Day 49 0:00–2:00 data exists). Fill NaN with Day 48 same-slot values. Do NOT use iterative prediction to fill forward.

### 4.6. Spatial / Geohash Neighbor Features (MEDIUM PRIORITY)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `gh_prefix5` | First 5 chars of geohash (broader area) | Spatial cluster identity |
| `gh_prefix4` | First 4 chars (even broader) | Regional identity |
| `neighbor_mean_demand` | Mean demand of geohashes sharing prefix5 | Spatial smoothing — critical for 10 unseen geohashes |
| `spatial_rank` | Rank of geohash within region by avg demand | Relative position |

### 4.7. Road Infrastructure Features (MEDIUM PRIORITY — Noisy but Informative)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `modal_road_type` | Most common road type per geohash from training data | Denoised — strong signal (Highway=0.616 vs Residential=0.057) |
| `modal_lanes` | Most common lane count per geohash | Denoised |
| `modal_large_vehicles` | Most common value per geohash | Denoised |
| `modal_landmarks` | Most common value per geohash | Denoised |
| `road_type_encoded` | Ordinal: Residential=0, Street=1, Highway=2 | Major demand driver |
| `is_highway` | Binary: Highway road type | High-demand indicator |
| `is_high_capacity` | Lanes ≥ 4 | Infrastructure flag |

### 4.8. Weather Features (LOW PRIORITY — Distractors, include for completeness)

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
│  → SOURCE FOR ALL Day-48 lookup features                     │
├──────────────────────────────────────────────────────────────┤
│  Day 49: 9 timestamps (0:00–2:00) — 7,872 rows              │
│  → SECONDARY TRAINING DATA                                    │
│  → Limited true lag seed (only for test slots 2:15–2:45)     │
├──────────────────────────────────────────────────────────────┤
│  Day 49: 47 timestamps (2:15–13:45) — 41,778 rows           │
│  → TEST SET (predict demand)                                 │
│  → test.csv to be provided later                             │
└──────────────────────────────────────────────────────────────┘
```

### 5.2. Validation Strategy

> [!IMPORTANT]
> Standard random K-fold will leak temporal information. Use **temporal validation**:

- **Primary — Time-Based Split (Approach A)**: Train on Day 48 (0:00–17:45), validate on Day 48 (18:00–23:45). Simulates predicting future timestamps using Day 48 same-slot lookups from the training portion.
- **Secondary — Cross-Day Split (Approach B)**: Train on Day 48, validate on Day 49 (0:00–2:00). Simulates the actual cross-day prediction task.
- **Sanity — Grouped Time K-Fold (Approach C)**: 5-fold where each fold holds out a contiguous block of ~4 hours from Day 48.

Use **Approach A as the primary validation metric** (most data, most representative of the test window). Use Approach B as a cross-check.

### 5.3. Handling Unseen Geohashes (10 in test)

For the 10 test geohashes with no training history:
1. **5-character prefix match** — find training geohashes sharing the first 5 characters, use their aggregated stats
2. If no prefix5 match, use **prefix4 match**
3. Final fallback: **global median demand for that time slot**

### 5.4. Inference Strategy (REVISED — No Iterative Prediction)

**Single-pass prediction**:
1. Build all features for test rows using Day 48 lookups and aggregated statistics
2. Fill true lag features with Day 48 same-slot values where Day 49 data doesn't exist
3. Predict all 41,778 test rows in one pass
4. Clip predictions to `[0, 1]`

> [!CAUTION]
> **Iterative prediction is NOT used.** With autocorrelation at 0.20, feeding predictions as lags would accumulate noise rapidly. Single-pass with Day 48 lookups is safer and more accurate.

---

## 6. Model Hyperparameter Ranges

### LightGBM (Primary)
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
    'categorical_feature': ['geohash_encoded', 'modal_road_type', ...]
}
```

### XGBoost (P1 — add after LightGBM baseline)
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

### CatBoost (P1 — add after LightGBM baseline)
```python
{
    'iterations': [1000, 2000],
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [6, 8, 10],
    'l2_leaf_reg': [1, 3, 5],
    'cat_features': [geohash, modal_road_type, weather_encoded, ...]
}
```

---

## 7. Ensemble Strategy

```
Final_Prediction = w1 * LightGBM + w2 * XGBoost + w3 * CatBoost
```

Weights determined by **validation R²** using inverse-error weighting or Optuna optimization.

Post-processing:
- Clip predictions to `[0, 1]` range
- Sanity check: no NaN values in output

---

## 8. Tools & Libraries

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Runtime |
| **pandas** | Data manipulation |
| **numpy** | Numerical ops |
| **scikit-learn** | Preprocessing, metrics, base utilities |
| **LightGBM** | Primary model |
| **XGBoost** | Secondary model (P1) |
| **CatBoost** | Tertiary model (P1) |
| **Optuna** | Hyperparameter optimization |
| **matplotlib / seaborn** | Visualization & EDA plots |
| **python-geohash** | Geohash decoding (lat/lon extraction, neighbor lookup) |

---

## 9. File Structure (Revised — .ipynb Format)

> [!IMPORTANT]
> Hackathon requires submitting `.ipynb` files. All code will be written in Jupyter notebooks.

```
new_verified/
├── DATASET/
│   ├── train.csv                      # Given training data (77,299 × 11)
│   ├── test.csv                       # To be added later (41,778 × 10)
│   └── sample_submission.csv          # Submission format reference (5 × 2)
├── gridlock_solution.ipynb            # ⭐ MAIN NOTEBOOK — all-in-one solution
│                                      #    Section 1: EDA & Data Understanding
│                                      #    Section 2: Feature Engineering
│                                      #    Section 3: Model Training & Validation
│                                      #    Section 4: Ensemble & Final Prediction
│                                      #    Section 5: Submission Generation
├── submissions/
│   └── submission.csv                 # Generated 41,778 × 2 CSV (Index, demand)
├── approach.txt                       # Text file explaining approach (required)
├── PS.md                              # Problem statement reference
├── AboutHackahton.md                  # Hackathon info reference
└── implementation_plan.md             # This plan
```

### Notebook Structure (`gridlock_solution.ipynb`)

The single notebook will be organized into clear sections with markdown headers:

| Section | Content |
|---------|---------|
| **0. Setup & Imports** | Install dependencies, import libraries, set random seeds |
| **1. Data Loading & EDA** | Load train/test, shape checks, missing values, distribution plots, temporal patterns, geohash analysis |
| **2. Feature Engineering** | All feature creation: temporal, geohash stats, geohash×time interactions, Day 48 lookups, spatial, road, weather |
| **3. Validation Framework** | Temporal train/val split, validation metrics |
| **4. LightGBM Baseline** | Train, evaluate, feature importance analysis |
| **5. Hyperparameter Tuning** | Optuna optimization for LightGBM |
| **6. XGBoost + CatBoost** | Train secondary models (if LightGBM R² < 0.93) |
| **7. Ensemble** | Weighted average, optimize weights |
| **8. Final Prediction** | Load test.csv, generate features, predict, clip, save |
| **9. Submission Validation** | Format checks, distribution sanity, save submission.csv |

---

## 10. Execution Roadmap

| Step | Task | Priority | Details |
|------|------|----------|---------|
| 1 | Create notebook with EDA section | Core | Verify all corrected statistics match live data |
| 2 | Feature engineering pipeline in notebook | Core | All features from §4, priority order |
| 3 | Temporal validation framework | Core | Approach A (Day 48 split) + Approach B (cross-day) |
| 4 | LightGBM baseline → target R² > 0.85 | Core | With geohash×time interactions + Day 48 lookups |
| 5 | Feature iteration → target R² > 0.90 | Core | Add remaining features, check importance |
| 6 | Hyperparameter tuning with Optuna | Enhancement | Optimize LightGBM parameters |
| 7 | XGBoost + CatBoost (if needed) | Enhancement | Only if LightGBM R² < 0.93 |
| 8 | Ensemble optimization | Enhancement | Weighted average, optimize weights |
| 9 | Handle unseen geohashes | Robustness | Prefix-based fallback for 10 unseen geohashes |
| 10 | **Receive test.csv** → generate predictions | Deliverable | User provides test.csv, run prediction pipeline |
| 11 | Submission validation | Deliverable | 41,778 × 2, correct columns, all values in [0,1] |
| 12 | Write approach.txt + package | Deliverable | Zip notebook + approach.txt + submission.csv |

---

## 11. Expected Baseline Performance (Corrected)

| Approach | Expected R² | Notes |
|----------|-------------|-------|
| Global mean prediction | ~0.00 | Baseline |
| Per-geohash mean demand (Day 48) | **~0.69** | Matches corrected 69% variance explained |
| Per-geohash × hour mean demand | **~0.85** | Adding temporal pattern on top of geohash |
| Per-geohash × time_slot mean demand (Day 48 lookup) | **~0.88–0.90** | Finest-grain lookup baseline |
| + Road features + spatial features + GBDT model | **~0.91–0.94** | LightGBM with full feature set |
| + Hyperparameter tuning | **~0.93–0.95** | Optuna optimization |
| + Ensemble (if used) | **~0.94–0.96** | Multi-model average |

> [!NOTE]
> R² estimates are lower than v1 because geohash variance was overstated. A per-geohash mean only captures 69% (not 80%). The remaining 31% must come from temporal interactions, road features, and model sophistication.

---

## 12. Verification Plan

### Automated Checks (in notebook)
1. **Submission format validation**: Output is 41,778 × 2 with columns `Index, demand`
2. **Cross-validation R² on temporal hold-out**: Must exceed 0.85 before submission
3. **Sanity checks**: All predictions in [0, 1], no NaN values
4. **Feature importance analysis**: Confirm geohash AND time features dominate (not geohash alone)
5. **Residual analysis**: Check that residuals don't show systematic temporal patterns

### Manual Verification
1. Plot predicted vs actual for validation set
2. Inspect predictions for known high-demand geohashes (Highway locations)
3. Check predictions for unseen geohashes are reasonable
4. Compare submission statistics (mean, std, distribution) with training data patterns
5. Verify prediction distribution aligns with test time window (2:15→13:45 = rising demand phase)

---

## 13. Test Data Workflow

>
> **test.csv is not currently in the repo.** The workflow for final submission:
> 1. I will build and validate the full model using train.csv only (with temporal validation splits)
> 2. You will provide test.csv when ready
> 3. I will add test.csv to `DATASET/` folder
> 4. Run the final prediction section of the notebook to generate submission.csv
> 5. Validate the submission format and submit

 [!IMPORTANT]
NOTE: now i added the test.csv file into the dataset/ folder


---

## Resolved Open Questions

> [!NOTE]
> **Q1 (from v1): Single LightGBM vs full ensemble?**
> **Answer: LightGBM first.** Build a strong single model, then add ensemble only if R² < 0.90. This is faster to iterate and debug.

> [!NOTE]
> **Q2 (from v1): Lag strategy — static proxy vs iterative prediction?**
> **Answer: Static proxy (Day 48 lookup) ONLY.** Iterative prediction is eliminated due to weak autocorrelation (0.20). Day 48 same-slot lookup is both safer and more accurate.

---

## Leaderboard Context

> [!NOTE]
> The community has identified that 100+ teams achieved perfect R²=100 through reverse engineering test labels from the public Grab AI 2019 dataset. The final evaluation will use **hidden test data** with a distributional shift. Our strategy of building a genuinely robust model with strong feature engineering will perform well on the hidden set, unlike overfitted/leaked submissions.
