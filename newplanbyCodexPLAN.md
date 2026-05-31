<!-- read @file:PLAN.md , @file:ChatgptThinking.md and @file:PS.md and make a newplanbyCodexPLAN.md file which contain detailed plan approach etc -->

## Plan: Gridlock Demand Prediction (Detailed)

Build a leakage-safe, time-aware tabular modeling pipeline centered on CatBoost with strong feature engineering (geohash hierarchy, time cycles, and demand priors). Emphasize validation that mirrors the test day, then package a clean submission and a concise approach document for source upload.

**Steps**
1. Phase 1 - Data audit and schema lock: Load train, test, and sample submission; confirm column names, dtypes, and missingness; verify unique index counts and detect duplicates across (geohash, day, timestamp); profile target distribution and outliers; validate train days (48/49) vs test day (49). *depends on none*
2. Phase 1 - Split design (time-aware): Create a primary split with Day 48 as train and Day 49 as validation to simulate the test scenario; add a secondary temporal split within Day 48 (earlier timestamps train, later timestamps valid) to test temporal robustness; fix random seeds and document fold definitions. *depends on 1*
3. Phase 2 - Core time features: Parse timestamp into hour and minute; add cyclical encodings for hour and minute; add coarse time buckets (15-min or 30-min), and day-hour interaction features to capture day shift effects. *depends on 1, parallel with 4*
4. Phase 2 - Spatial and categorical features: Generate geohash prefix features at multiple lengths (2, 4, 5, 6); treat RoadType, Weather, LargeVehicles, Landmarks as categorical for CatBoost; add limited, high-signal interactions (RoadType x NumberofLanes, Weather x Temperature, day x Weather). *depends on 1, parallel with 3*
5. Phase 2 - Missingness strategy: Create missing indicators for Temperature, Weather, and RoadType; impute Temperature using median by geohash or geohash plus time bucket; convert missing categoricals to an explicit Unknown category. *depends on 1, parallel with 3*
6. Phase 2 - Demand priors with leakage guard: Compute demand priors (mean, median, std) for geohash and for geohash plus hour; generate out-of-fold priors for training data using the Day 48 training fold, then recompute priors on full train for test. *depends on 2, blocks 7*
7. Phase 3 - Baseline model and diagnostics: Train a `CatBoostRegressor` baseline with categorical features and early stopping on the Day 48 to 49 split; analyze R2, residuals by hour and geohash prefix, and feature importance; adjust features if validation is unstable. *depends on 3-6*
8. Phase 3 - Tuning and robustness: Tune depth, learning rate, l2_leaf_reg, subsample; compare Day 48 to 49 and within-Day 48 temporal splits; lock a final configuration that performs consistently across both validations. *depends on 7*
9. Phase 4 - Optional secondary model: Train a `LightGBM` model only if CatBoost leaves clear gaps; use leakage-safe encoding for categoricals; evaluate on the same splits and blend only if it improves validation materially. *depends on 7, parallel with 8 if resources allow*
10. Phase 5 - Final training and submission: Retrain the best model on full train with fixed features; generate test predictions; validate submission shape (41778 x 2), column names, and index alignment; consider mild clipping only if it improves R2 on validation. *depends on 8 and 9*
11. Phase 5 - Packaging: Prepare a short approach document describing features, validation, and tools; bundle notebook and any helper scripts for the source archive; note that only training data was used for priors. *depends on 10*

**Relevant files**
- [DATASET/train.csv](DATASET/train.csv) - primary training data and target distribution checks
- [DATASET/test.csv](DATASET/test.csv) - scoring data for predictions
- [DATASET/sample_submission.csv](DATASET/sample_submission.csv) - column names and submission format
- [PS.md](PS.md) - official task description and submission rules
- [PLAN.md](PLAN.md) - prior high-level plan to expand and align
- [DATASET/ChatgptThinking.md](DATASET/ChatgptThinking.md) - feature and validation priorities

**Verification**
1. Produce a data audit summary (missingness, duplicates, day distributions) and confirm no schema drift between train and test.
2. Report R2 for both validations (Day 48 to 49 and within-Day 48 temporal split) using the same seed and feature set.
3. Validate that demand priors are out-of-fold for training data and computed from full train only for test data.
4. Confirm submission file row count, column names, and index ordering match the sample submission.
5. Run a full notebook execution from top to bottom to ensure reproducibility.

**Decisions**
- Prioritize time and location features; treat weather and road attributes as secondary modifiers.
- Use CatBoost as the primary model; introduce `LightGBM` only if it improves validation.
- Use leakage-safe demand priors and day-aware validation to generalize to the test day.

**Further Considerations**
1. If geohash decoding to lat/lon offers no gain, keep only geohash prefix features.
2. If Day 49 validation is highly volatile, reduce feature complexity and increase regularization before blending.
3. If predictions show negative or extreme values, evaluate mild clipping only when it improves validation.
