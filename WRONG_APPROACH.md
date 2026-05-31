## 🎯 **APPROACH USED**

### 1. **Problem Understanding**
- **Task**: Predict normalized traffic demand (0-1 range) across geohash locations + timestamps
- **Data**: 77,299 training rows (Day 48 + 49), 41,778 test rows (Day 49 only)
- **Challenge**: Distribution shift (train mostly Day 48, test all Day 49)

### 2. **Feature Engineering Strategy**
Created **28 engineered features** from 11 raw columns:

| Category | Features | Details |
|----------|----------|---------|
| **Temporal** (8) | time_bin_15, time_sin, time_cos, hour, minute, day, is_day49, time_minutes | Cyclical encoding to handle 24-hour wrap-around |
| **Spatial** (6) | latitude, longitude, geo_distance, geohash_4, geohash_5, geohash | Manual geohash decoder (BASE32) → hierarchical grouping |
| **Infrastructure** (5) | large_vehicles_allowed, has_landmarks, NumberofLanes, RoadType, Weather | Binary + categorical features |
| **Environmental** (2) | Temperature (imputed), Weather | Contextual imputation by (geohash, weather) groups |
| **Target-Encoded** (4) | geohash_te, geohash_time_te, roadtype_time_te, weather_time_te | K-Fold regularized (prevents leakage) |

### 3. **Validation Strategy**
```
Train: Day 48 (majority)
Validate: Day 49 (mimics test distribution)
Test: Day 49 only
```
**Why**: Catches models that overfit to Day 48 patterns and fail on Day 49.

### 4. **Data Preprocessing**
- Missing values in Temperature (~45%) → imputed by (geohash, weather) groups
- Missing values in RoadType (~12%) → filled with "Unknown"
- All categorical features → target-encoded (compact + regularized)

---

## 🤖 **MODELS USED**

### **Primary Model: sklearn HistGradientBoostingRegressor**

**Why this model?**
- ✅ No heavy dependencies (built into scikit-learn)
- ✅ Efficient histogram-based boosting (similar to LightGBM)
- ✅ Handles regression well
- ✅ Fast training & prediction

**Hyperparameters:**
```yaml
n_estimators: 2500      # number of boosting stages
learning_rate: 0.05     # conservative learning rate
max_depth: 8            # tree depth
max_iter: 400           # early stopping patience
```

**Performance:**
```
R² = 0.765    (76.5% variance explained)
MAE = 0.0399  (mean absolute error)
RMSE = 0.0702 (root mean squared error)
```

---

### **Alternative Models (Not Used, But Available)**

| Model | Why Skipped | Notes |
|-------|-------------|-------|
| **CatBoost** | Disk space constraints (~100MB) | Excellent for categorical features, native handling |
| **LightGBM** | Disk space constraints (~100MB) | Very fast, memory-efficient, similar to our model |
| **Random Forest** | Too slow for large data | Would work but slower training |
| **XGBoost** | Not installed | Heavy dependency, similar to HistGradientBoosting |

---

## 🔄 **Full Pipeline**

```
RAW DATA (train.csv)
    ↓
[1. Data Loading & Splitting]
    Train: Day 48 (66,000+ rows)
    Validate: Day 49 (11,000+ rows)
    ↓
[2. Feature Engineering (FeatureBuilder)]
    • Parse timestamps → cyclical encoding
    • Decode geohash → lat/lon + hierarchical keys
    • Impute temperature → by weather context
    • Create 28 engineered features
    • K-Fold target encoding (5 folds) → prevents leakage
    ↓
[3. Model Training]
    sklearn HistGradientBoostingRegressor
    • Train on Day 48 features + target
    • Validate on Day 49 features
    ↓
[4. Performance Evaluation]
    R² = 0.765, MAE = 0.0399, RMSE = 0.0702
    ↓
[5. Full Training]
    Retrain on ALL data (Day 48 + 49)
    ↓
[6. Test Prediction]
    test.csv → feature transform → predictions
    ↓
OUTPUT (submission.csv)
    41,778 predictions ready for HackerEarth
```

---

## 🎓 **Key Design Decisions**

| Decision | Rationale | Benefit |
|----------|-----------|---------|
| **Day 48→49 validation split** | Test is entirely Day 49 | Prevents overfitting to Day 48 |
| **K-Fold target encoding** | High-cardinality categoricals | Compact (4 features) vs one-hot (300+ dims) + regularized |
| **Geohash hierarchical grouping** | Capture spatial patterns | 1km, 10km, 100km scale patterns |
| **Cyclical time encoding** | 24-hour periodicity without boundary artifacts | Model learns midnight→morning transition naturally |
| **Contextual temp imputation** | Weather-aware missing values | Preserves weather-temperature relationships |

---

## 📊 **Why This Approach Works**

1. **Addresses Distribution Shift** → Day 49 validation catches overfitting
2. **Prevents Leakage** → K-Fold target encoding separates folds
3. **Compact Features** → Target encoding replaces high-dim one-hot
4. **Spatial Intelligence** → Geohash decoding + hierarchical grouping
5. **Temporal Intelligence** → Cyclical encoding for 24-hour patterns
6. **Lightweight** → No GPU/heavy libs needed

---
