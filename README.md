# Flipkart Grid 2.0 — Traffic Demand Prediction

> **Final Score:** 91.75866 / 100 (Public Leaderboard)  
> A two-stage ensemble combining LightGBM and Seq2Seq Transformers for citywide traffic demand forecasting.

## Overview

This solution predicts normalized traffic demand at the granularity of `(geohash, 15-min timestamp)` for urban road cells. Given a full day (day-48) of traffic history and early-morning observations (day-49 slots 0–8), we forecast demand for the remaining slots (9–55) of day-49.

**Challenge:** Most road cells (geohash×slot pairs) appear only once in the training data, so naive approaches overfit. We solve this through:
1. Bayesian-smoothed target encodings and neighbourhood fallbacks
2. Heavy regularisation to combat distribution shift
3. Blending complementary models (tree-based + sequence)

## Approach

### Stage A: LightGBM Ensemble (10 seeds)
- **Features:** ~41 engineered features including spatial (lat/lng, geohash aggregates), temporal (cyclical sin/cos, slot encoding), and target-encoded aggregates.
- **Regularisation:** `lambda_l2=2.0`, `min_data_in_leaf=100`, feature/row subsampling → protects against overfitting and day-level shifts.
- **Output:** `10-LightGBM.csv` (~91.24 leaderboard)

### Stage B: Seq2Seq Transformer (3 seeds)
- **Architecture:** Small Transformer encoder (d_model=48, 1 layer, 4 heads) trained on per-geohash curves (length 152: 96 day-48 + 56 day-49).
- **Training:** Self-supervised masked reconstruction (random chunk masking) mimics inference task.
- **Blending:** Averaged 3 seeds, blended at 15.2% into GBM → (~91.70 leaderboard)

### Stage C: Post-processing
- **Anti-5seed extrapolation:** Used empirical direction from models to nudge final blend → (~91.76 leaderboard)

## Results

| Stage | Model | Leaderboard Score |
|---|---|---:|
| Stage A | 10-seed LightGBM | 90.34 |
| Stage A+B | + 15.2% Seq2Seq blend | 91.24 |
| Final | + Anti-5seed extrapolation | **91.76** |

## Key Insights

1. **Day-48 lag dominates:** Same geohash, same slot yesterday explains ~50% of variance.
2. **Bayesian smoothing is crucial:** Transforms noisy 1-observation encodings into stable predictors.
3. **Heavy regularisation beats low CV loss:** Empirically validated against leaderboard behavior; protects generalization.
4. **Complementary models:** Trees excel at engineered aggregates; Transformers capture curve shape → combined lift is consistent.

## Project Structure

```
GitHappens/
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns
├── sub.md                             # Detailed submission report (judges)
├── final_notebook.ipynb               # Full code, EDA, training pipeline
├── dataset/
│   ├── train.csv                      # Training data (day-48 + day-49 slots 0-8)
│   ├── test.csv                       # Test data (day-49 slots 9-55)
│   └── sample_submission.csv
└── outputs/
    └── final_submission.csv            # Final submission
```

## Feature Engineering Highlights

- **Spatial:** Geohash decoding (lat/lng), per-geohash aggregates, prefix-level (gh4/gh5) neighbourhood fallbacks
- **Temporal:** Slot index, cyclical sin/cos encoding, monotonic time trends
- **Target-encoded aggregates:** 
  - Bayesian-smoothed per-(geohash, slot) from day-48
  - Cross-day d49/d48 calibration (ratio + delta)
  - Lag features (same-slot + neighbours ±1, ±2)
- **Recent context:** Early-morning day-49 observations

## Getting Started

### Requirements
- Python 3.8+
- `pandas`, `numpy`, `lightgbm`, `torch`, `scikit-learn`
- `matplotlib`, `seaborn` (for plotting)

### Setup
```bash
pip install pandas numpy lightgbm torch scikit-learn matplotlib seaborn
```

### Run Training
Open and execute `final_notebook.ipynb` in Jupyter:
```bash
jupyter notebook final_notebook.ipynb
```

The notebook trains both stages and generates submissions in `outputs/`.

### Quick Inference
```python
import pandas as pd

# Load pre-trained predictions
gbm_preds = pd.read_csv('outputs/10-LightGBM.csv')
seq_preds = pd.read_csv('outputs/3-Transformer.csv')
final_blend = pd.read_csv('outputs/anti5_extrapolation.csv')

print(final_blend.head())
```

## Tools & Libraries

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data wrangling, matrix ops |
| `lightgbm` | Gradient boosted tree ensemble |
| `torch` | Seq2Seq Transformer model |
| `scikit-learn` | Metrics (R²), utilities |
| `matplotlib`, `seaborn` | EDA & visualisation |

## Next Steps (Future Work)

- Validate blend weights with geohash-grouped cross-validation
- Explore spatial encoders (graph conv / attention) for inter-geohash signal sharing
- Add learned calibration layers (per-geohash scaling) regularised by neighbourhood priors
- Fine-tune hyperparameters on larger holdout set

## GIT HAPPENS - Team Details
- [Anvesha Yadav](https://github.com/Anveshayadav28)
- [Ashika Agrawal](https://github.com/ashikaagrawal28)
- [Piyush Verma](https://github.com/piyerx)
- [Stiwart Stance Saxena](https://github.com/STIWARTs)

---

**Disclaimer:** This solution leverages public leaderboard feedback for final tuning (blend weights, extrapolation). When deploying in practice, ensure proper cross-validation grouped by geohash and time to reduce data leakage risk.