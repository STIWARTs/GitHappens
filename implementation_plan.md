**Approach overview (researcher-grade plan)**  
The problem is a supervised regression task predicting `demand` (continuous) using spatiotemporal and contextual features. The evaluation metric is **R²**, so we prioritize models that capture non‑linear interactions and generalize across a distribution shift (day 48 → day 49).  
Given your dataset facts: **train has mostly day 48 with some day 49, test is day 49 only**, and **timestamps cover all 96 quarter‑hour slots**, the plan focuses on robust time + location modeling and careful validation that mimics the day‑shift.

---

**Data understanding and preprocessing strategy**  
**Input files** (from `DATASET/`): `train.csv`, `test.csv`, `sample_submission.csv` (plus existing `.md` notes in repo root).  
Key columns:  
- **Numeric**: `demand` (target), `NumberofLanes`, `Temperature`, `day`  
- **Categorical**: `geohash`, `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`  
- **Temporal**: `timestamp` (quarter‑hour string)

Planned preprocessing steps:  
1. **Timestamp parsing**  
   - Normalize `timestamp` strings (e.g., `0:0` → `00:00`)  
   - Convert to **minutes since midnight**, **hour**, **minute**, and **quarter‑hour index (0–95)**  
2. **Missing values**  
   - `RoadType`/`Weather`: fill with `"Unknown"`  
   - `Temperature`: impute using median by `(geohash, Weather)` fallback to global median  
   - Ensure `NumberofLanes` is numeric with safe casting  
3. **Binary encoding**  
   - `LargeVehicles`, `Landmarks` → {0,1}  

---

**Feature engineering (detailed, competition‑grade)**  
We will engineer a layered feature set to capture **space, time, infrastructure, and weather interactions**, while avoiding leakage.

**A. Temporal features**  
- `time_minutes`: minutes since midnight  
- `time_bin_15`: quarter‑hour index (0–95)  
- `hour`, `minute`  
- Cyclical time encoding:  
  - `sin(2π * time_bin_15/96)`, `cos(2π * time_bin_15/96)`  
- `day` and `is_day49` (binary) to model shift  

**B. Spatial features from geohash**  
- **Decode geohash** to `latitude`, `longitude` (6‑char geohash → ~1km grid)  
- **Geohash prefix levels**: `geohash_4`, `geohash_5` (coarser spatial grouping)  
- Distance‑based features: distance to dataset centroid  

**C. Infrastructure and context interactions**  
- `RoadType` × `NumberofLanes` interaction  
- `LargeVehicles` × `RoadType`  
- `Landmarks` × `RoadType`  
- `Weather` × `Temperature`  

**D. Aggregated demand statistics (target‑based, leakage‑safe)**  
Use **K‑Fold target encoding** strictly on training folds:  
- Mean demand by `geohash`  
- Mean demand by `geohash + time_bin_15`  
- Mean demand by `RoadType + time_bin_15`  
- Mean demand by `Weather + time_bin_15`  
Smoothing (e.g., James‑Stein / Bayesian shrinkage) to prevent overfitting on sparse groups.

---

**Modeling strategy**  
We will implement **two strong baselines** and choose the best (or ensemble):

1. **CatBoost Regressor**  
   - Handles categorical features natively  
   - Strong on mixed tabular data  
   - Built‑in handling of missing values  
2. **LightGBM / XGBoost**  
   - Excellent for engineered numeric features  
   - Fast iteration and easy ensembling  

**Ensemble option**:  
- Weighted average of CatBoost + LightGBM predictions (optimize weights on validation).

---

**Validation design (critical because of day shift)**  
We will **avoid random split**.  
Primary validation split:  
- **Train on day 48**, validate on day 49 rows inside train.csv (your dataset already has some day 49).  
Secondary validation:  
- Time‑based split within day 48 to check stability.

This mimics the **true test distribution** (day 49 only) and reduces over‑optimistic results.

---

**Metrics and monitoring**  
- Primary metric: **R²** (as per evaluation)  
- Secondary: MAE, RMSE for sanity  
- Residual checks by time of day and geohash to detect bias.

---

**Tools to be used**  
- **Python 3.11**  
- **Pandas / NumPy** for data processing  
- **Scikit‑learn** for pipelines, splits, metrics  
- **CatBoost** for baseline model  
- **LightGBM or XGBoost** for alternative model  
- **Geohash decoding** (`geohash2` or `pygeohash`)  
- **Matplotlib / Seaborn** for EDA  
- **Optuna** for tuning (only if baseline is stable)  

---

**Relevant source files (planned structure)**  
These are the minimal files needed for a clean, auditable submission package:

1. `DATASET/train.csv`, `DATASET/test.csv`, `DATASET/sample_submission.csv`  
2. `notebooks/01_eda.ipynb` — data inspection, missing values, distribution shift  
3. `src/feature_engineering.py` — all transformations + target encoding  
4. `src/train.py` — training pipeline + validation  
5. `src/predict.py` — generates `submission.csv`  
6. `src/metrics.py` — R², MAE, RMSE  
7. `configs/model.yaml` — hyperparameters (keeps experiments reproducible)  
8. `outputs/submission.csv` — final submission file  
9. `approach.txt` — the required explanation text (this plan condensed)

---

**Risk controls (aligned with leaderboard leakage concerns)**  
- **No external data** usage  
- Validation mimics day‑shift (avoids accidental leakage)  
- K‑Fold target encoding prevents leakage from using the same row’s target  

---

If you approve this plan, I will proceed with implementation and produce the full pipeline plus the submission file.



Notebook for analysis + .py for the final pipeline. Start with a notebook for EDA and feature prototyping, then move stable code into src/*.py so the submission and reruns are clean and reproducible