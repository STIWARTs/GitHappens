"""
Main Pipeline for Gridlock 2.0 Traffic Demand Prediction.
Orchestrates: Data Loading → Feature Engineering → Training → Prediction → Submission.

Usage:
    python run_pipeline.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import (
    load_data, validate_submission, timestamp_to_slot,
    temporal_train_val_split, day_based_split
)
from features import FeatureEngine, build_lag_features_static, build_lag_features_from_history
from train import ModelTrainer


def main():
    start_time = time.time()
    print("=" * 70)
    print("  GRIDLOCK 2.0 — TRAFFIC DEMAND PREDICTION PIPELINE")
    print("  Full Ensemble: LightGBM + XGBoost + CatBoost")
    print("=" * 70)

    # =====================================================================
    # STEP 1: LOAD DATA
    # =====================================================================
    print("\n[STEP 1] Loading data...")
    data_dir = os.path.dirname(__file__)
    train_raw, test_raw = load_data(data_dir)
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

    # Separate Day 48 (full) and Day 49 train (0:00-2:00)
    d48 = train_raw[train_raw['day'] == 48].copy()
    d49_train = train_raw[train_raw['day'] == 49].copy()
    print(f"  Day 48: {len(d48)} rows, Day 49 train: {len(d49_train)} rows")
    print(f"  Test (Day 49, 2:15-13:45): {len(test_raw)} rows")

    # =====================================================================
    # STEP 2: FEATURE ENGINEERING
    # =====================================================================
    print("\n[STEP 2] Feature engineering...")
    fe = FeatureEngine()
    fe.fit(train_raw)
    print(f"  Fitted on {len(train_raw)} training rows")
    print(f"  Geohash stats computed for {len(fe.geohash_stats)} geohashes")

    # Transform all datasets
    print("  Transforming train data...")
    train_featured = fe.transform(train_raw, is_train=True)
    print("  Transforming test data...")
    test_featured = fe.transform(test_raw, is_train=False)

    # Build lag features (static proxy from Day 48)
    print("  Building static lag features for train...")
    train_featured = build_lag_features_static(
        train_raw, train_featured, fe.geohash_slot_stats
    )
    print("  Building static lag features for test...")
    test_featured = build_lag_features_static(
        train_raw, test_featured, fe.geohash_slot_stats
    )

    # Build lag features from Day 49 history
    print("  Building history-based lag features for test...")
    test_featured = build_lag_features_from_history(
        d49_train, test_featured
    )
    # Fill NaN in history lags with static proxy values
    for lag in [1, 2, 3, 4]:
        hist_col = f'demand_lag_{lag}_hist'
        static_col = f'demand_lag_{lag}_static'
        if hist_col in test_featured.columns:
            test_featured[hist_col] = test_featured[hist_col].fillna(
                test_featured[static_col] if static_col in test_featured.columns else 0
            )

    # Also compute hist lag features for train (using train itself as history for proper val)
    train_featured = build_lag_features_from_history(
        train_raw, train_featured
    )
    for lag in [1, 2, 3, 4]:
        hist_col = f'demand_lag_{lag}_hist'
        static_col = f'demand_lag_{lag}_static'
        if hist_col in train_featured.columns:
            train_featured[hist_col] = train_featured[hist_col].fillna(
                train_featured[static_col] if static_col in train_featured.columns else 0
            )

    # Fill remaining NaN in hist-based rolling features
    for col in ['demand_rolling_mean_4_hist', 'demand_rolling_mean_2_hist', 'demand_diff_1_hist']:
        if col in train_featured.columns:
            train_featured[col] = train_featured[col].fillna(0)
        if col in test_featured.columns:
            test_featured[col] = test_featured[col].fillna(0)

    # Determine feature columns (intersection of train and test)
    feature_cols = fe.get_feature_cols()
    # Add lag feature columns
    lag_feature_cols = [c for c in train_featured.columns if 'lag_' in c or 'rolling_' in c or 'diff_1' in c]
    feature_cols = list(set(feature_cols + lag_feature_cols))
    # Only keep columns present in both
    feature_cols = [c for c in feature_cols if c in train_featured.columns and c in test_featured.columns]
    feature_cols = sorted(feature_cols)
    print(f"  Total features: {len(feature_cols)}")

    # =====================================================================
    # STEP 3: VALIDATION SPLIT
    # =====================================================================
    print("\n[STEP 3] Creating validation split...")

    # Use temporal split: train on Day48 early, validate on Day48 late
    d48_featured = train_featured[train_featured['day'] == 48]
    d48_featured = d48_featured.copy()
    d48_featured['_slot'] = d48_featured['timestamp'].apply(timestamp_to_slot)

    # Split: train on slots 0-71 (0:00-17:45), validate on slots 72-95 (18:00-23:45)
    val_cutoff = 72
    train_split = d48_featured[d48_featured['_slot'] < val_cutoff]
    val_split = d48_featured[d48_featured['_slot'] >= val_cutoff]

    X_train = train_split[feature_cols]
    y_train = train_split['demand']
    X_val = val_split[feature_cols]
    y_val = val_split['demand']

    print(f"  Train split: {len(X_train)} rows (slots 0-71)")
    print(f"  Val split:   {len(X_val)} rows (slots 72-95)")

    # Also test Day-based validation
    d49_featured = train_featured[train_featured['day'] == 49]
    if len(d49_featured) > 0:
        X_val_d49 = d49_featured[feature_cols]
        y_val_d49 = d49_featured['demand']
        print(f"  Day 49 val:  {len(X_val_d49)} rows")

    # =====================================================================
    # STEP 4: TRAIN MODELS
    # =====================================================================
    print("\n[STEP 4] Training models...")
    trainer = ModelTrainer(feature_cols, model_dir=os.path.join(data_dir, 'models'))

    # Handle any remaining NaN in features
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # --- LightGBM ---
    trainer.train_lightgbm(X_train, y_train, X_val, y_val, n_trials=40)

    # --- XGBoost ---
    trainer.train_xgboost(X_train, y_train, X_val, y_val, n_trials=30)

    # --- CatBoost ---
    trainer.train_catboost(X_train, y_train, X_val, y_val, n_trials=25)

    # =====================================================================
    # STEP 5: ENSEMBLE & VALIDATION
    # =====================================================================
    print("\n[STEP 5] Computing ensemble...")
    weights = trainer.compute_ensemble_weights()

    # Validate ensemble on both splits
    for name, (X_v, y_v) in [("Time-split", (X_val, y_val))]:
        X_v = X_v.fillna(0)
        pred = trainer.predict_ensemble(pd.DataFrame(X_v, columns=feature_cols), weights)
        score = max(0, 100 * r2_score(y_v, pred))
        rmse = np.sqrt(mean_squared_error(y_v, pred))
        print(f"\n  Ensemble on {name}: R² = {score:.4f}, RMSE = {rmse:.6f}")

    if len(d49_featured) > 0:
        X_val_d49_clean = d49_featured[feature_cols].fillna(0)
        pred_d49 = trainer.predict_ensemble(pd.DataFrame(X_val_d49_clean.values, columns=feature_cols), weights)
        score_d49 = max(0, 100 * r2_score(y_val_d49, pred_d49))
        print(f"  Ensemble on Day-49 train: R² = {score_d49:.4f}")

    # =====================================================================
    # STEP 6: RETRAIN ON FULL DATA
    # =====================================================================
    print("\n[STEP 6] Retraining on ALL training data...")

    # Use ALL training data (Day48 + Day49 train) for final models
    X_full = train_featured[feature_cols].fillna(0)
    y_full = train_featured['demand']
    print(f"  Full training set: {len(X_full)} rows")

    # Retrain each model with best params on full data
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb_lib

    final_models = {}

    # LightGBM final
    lgbm_params = trainer.best_params.get('lightgbm', {})
    lgbm_params['n_estimators'] = int(lgbm_params.get('n_estimators', 1000))
    lgbm_final = lgb.LGBMRegressor(**lgbm_params)
    lgbm_final.fit(X_full, y_full)
    final_models['lightgbm'] = lgbm_final
    print("  [OK] LightGBM retrained on full data")

    # XGBoost final
    xgb_params = trainer.best_params.get('xgboost', {}).copy()
    xgb_params['n_estimators'] = int(xgb_params.get('n_estimators', 1000))
    # Remove early stopping for final retrain (no validation set)
    xgb_params.pop('early_stopping_rounds', None)
    xgb_final = xgb.XGBRegressor(**xgb_params)
    xgb_final.fit(X_full, y_full, verbose=False)
    final_models['xgboost'] = xgb_final
    print("  [OK] XGBoost retrained on full data")

    # CatBoost final
    cb_params = trainer.best_params.get('catboost', {})
    cb_params['iterations'] = int(cb_params.get('iterations', 1000))
    cb_final = cb_lib.CatBoostRegressor(**cb_params)
    cb_final.fit(X_full, y_full, verbose=0)
    final_models['catboost'] = cb_final
    print("  [OK] CatBoost retrained on full data")

    # =====================================================================
    # STEP 7: GENERATE PREDICTIONS
    # =====================================================================
    print("\n[STEP 7] Generating predictions...")

    X_test = test_featured[feature_cols].fillna(0)

    # Individual predictions
    predictions = {}
    for name, model in final_models.items():
        pred = np.clip(model.predict(X_test), 0, 1)
        predictions[name] = pred
        print(f"  {name:12s}: mean={pred.mean():.6f}, std={pred.std():.6f}, range=[{pred.min():.6f}, {pred.max():.6f}]")

    # Ensemble
    ensemble_pred = np.zeros(len(X_test))
    for name, pred in predictions.items():
        ensemble_pred += weights.get(name, 1/3) * pred
    ensemble_pred = np.clip(ensemble_pred, 0, 1)

    print(f"\n  Ensemble:     mean={ensemble_pred.mean():.6f}, std={ensemble_pred.std():.6f}, range=[{ensemble_pred.min():.6f}, {ensemble_pred.max():.6f}]")
    print(f"  Train mean demand: {y_full.mean():.6f}")

    # =====================================================================
    # STEP 8: CREATE SUBMISSION
    # =====================================================================
    print("\n[STEP 8] Creating submission...")

    submissions_dir = os.path.join(data_dir, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    submission = pd.DataFrame({
        'Index': test_raw['Index'],
        'demand': ensemble_pred
    })

    # Validate
    is_valid = validate_submission(submission, test_raw)

    # Save
    submission_path = os.path.join(submissions_dir, 'submission_ensemble.csv')
    submission.to_csv(submission_path, index=False)
    print(f"  Saved to: {submission_path}")

    # Also save individual model submissions for comparison
    for name, pred in predictions.items():
        sub = pd.DataFrame({'Index': test_raw['Index'], 'demand': pred})
        sub.to_csv(os.path.join(submissions_dir, f'submission_{name}.csv'), index=False)

    # =====================================================================
    # STEP 9: FEATURE IMPORTANCE
    # =====================================================================
    print("\n[STEP 9] Feature importance (top 30)...")
    trainer.models = final_models
    importance = trainer.get_feature_importance(top_n=30)
    if len(importance) > 0:
        for _, row in importance.iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.6f}")

    # =====================================================================
    # STEP 10: SUMMARY
    # =====================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Validation R² scores:")
    for name, score in sorted(trainer.val_scores.items(), key=lambda x: -x[1]):
        print(f"    {name:12s}: {score:.4f}")
    print(f"  Ensemble weights: {weights}")
    print(f"  Submission: {submission_path}")
    print(f"  Rows: {len(submission)}, Valid: {is_valid}")
    print("=" * 70)

    # Save metadata
    trainer.save_metadata()


if __name__ == '__main__':
    main()
