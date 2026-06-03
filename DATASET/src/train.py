"""
Model Training Pipeline for Gridlock 2.0.
Trains LightGBM, XGBoost, and CatBoost with Optuna hyperparameter optimization.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """
    Trains and optimizes LightGBM, XGBoost, and CatBoost models.
    """

    def __init__(self, feature_cols: List[str], model_dir: str = '../models'):
        self.feature_cols = feature_cols
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.val_scores = {}
        self.best_params = {}

    def _r2_score(self, y_true, y_pred):
        return max(0, 100 * r2_score(y_true, y_pred))

    # =========================================================================
    #  LIGHTGBM
    # =========================================================================

    def _lgbm_objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for LightGBM."""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0001, 5.0, log=True),
            'n_jobs': -1,
            'random_state': 42
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 1)
        return self._r2_score(y_val, y_pred)

    def train_lightgbm(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        n_trials: int = 30
    ) -> lgb.LGBMRegressor:
        """Train LightGBM with Optuna optimization."""
        print("=" * 60)
        print("Training LightGBM with Optuna ({} trials)...".format(n_trials))
        print("=" * 60)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: self._lgbm_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update({
            'objective': 'regression', 'metric': 'rmse',
            'verbosity': -1, 'n_jobs': -1, 'random_state': 42
        })

        print(f"\nBest LightGBM params: R² = {study.best_value:.4f}")
        for k, v in best_params.items():
            if k not in ['objective', 'metric', 'verbosity', 'n_jobs', 'random_state']:
                print(f"  {k}: {v}")

        # Retrain with best params
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        val_pred = np.clip(model.predict(X_val), 0, 1)
        score = self._r2_score(y_val, val_pred)
        print(f"Final LightGBM validation R²: {score:.4f}")

        self.models['lightgbm'] = model
        self.val_scores['lightgbm'] = score
        self.best_params['lightgbm'] = best_params

        # Save model
        with open(os.path.join(self.model_dir, 'lightgbm.pkl'), 'wb') as f:
            pickle.dump(model, f)

        return model

    # =========================================================================
    #  XGBOOST
    # =========================================================================

    def _xgb_objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for XGBoost."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 40),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 5.0, log=True),
            'early_stopping_rounds': 50,
            'n_jobs': -1,
            'random_state': 42
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 1)
        return self._r2_score(y_val, y_pred)

    def train_xgboost(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        n_trials: int = 25
    ) -> xgb.XGBRegressor:
        """Train XGBoost with Optuna optimization."""
        print("=" * 60)
        print("Training XGBoost with Optuna ({} trials)...".format(n_trials))
        print("=" * 60)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=43))
        study.optimize(
            lambda trial: self._xgb_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update({
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'verbosity': 0, 'n_jobs': -1, 'random_state': 42
        })

        print(f"\nBest XGBoost params: R² = {study.best_value:.4f}")

        best_params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_pred = np.clip(model.predict(X_val), 0, 1)
        score = self._r2_score(y_val, val_pred)
        print(f"Final XGBoost validation R²: {score:.4f}")

        self.models['xgboost'] = model
        self.val_scores['xgboost'] = score
        self.best_params['xgboost'] = best_params

        with open(os.path.join(self.model_dir, 'xgboost.pkl'), 'wb') as f:
            pickle.dump(model, f)

        return model

    # =========================================================================
    #  CATBOOST
    # =========================================================================

    def _catboost_objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for CatBoost."""
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 9),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 64, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.01, 10.0, log=True),
            'verbose': 0,
            'random_seed': 42,
            'loss_function': 'RMSE',
            'task_type': 'CPU'
        }

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0
        )

        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 1)
        return self._r2_score(y_val, y_pred)

    def train_catboost(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        n_trials: int = 20
    ) -> cb.CatBoostRegressor:
        """Train CatBoost with Optuna optimization."""
        print("=" * 60)
        print("Training CatBoost with Optuna ({} trials)...".format(n_trials))
        print("=" * 60)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=44))
        study.optimize(
            lambda trial: self._catboost_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update({
            'verbose': 0, 'random_seed': 42,
            'loss_function': 'RMSE', 'task_type': 'CPU'
        })

        print(f"\nBest CatBoost params: R² = {study.best_value:.4f}")

        model = cb.CatBoostRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0
        )

        val_pred = np.clip(model.predict(X_val), 0, 1)
        score = self._r2_score(y_val, val_pred)
        print(f"Final CatBoost validation R²: {score:.4f}")

        self.models['catboost'] = model
        self.val_scores['catboost'] = score
        self.best_params['catboost'] = best_params

        model.save_model(os.path.join(self.model_dir, 'catboost.cbm'))

        return model

    # =========================================================================
    #  ENSEMBLE
    # =========================================================================

    def compute_ensemble_weights(self, X_val=None, y_val=None) -> Dict[str, float]:
        """
        Compute ensemble weights.
        If X_val/y_val provided: optimize via brute-force grid search on validation R².
        Otherwise: use softmax of validation scores.
        """
        scores = self.val_scores.copy()
        if not scores:
            return {}

        # If validation data provided, optimize weights directly
        if X_val is not None and y_val is not None and len(self.models) >= 2:
            print("\n  Optimizing ensemble weights on validation data...")
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = np.clip(model.predict(X_val[self.feature_cols]), 0, 1)

            model_names = list(predictions.keys())
            n_models = len(model_names)
            best_score = -1
            best_weights = {}

            # Grid search over weight combinations (step=0.05)
            step = 0.05
            if n_models == 3:
                for w1 in np.arange(0, 1.01, step):
                    for w2 in np.arange(0, 1.01 - w1, step):
                        w3 = 1.0 - w1 - w2
                        if w3 < -0.001:
                            continue
                        w3 = max(0, w3)
                        pred = w1 * predictions[model_names[0]] + w2 * predictions[model_names[1]] + w3 * predictions[model_names[2]]
                        pred = np.clip(pred, 0, 1)
                        score = r2_score(y_val, pred)
                        if score > best_score:
                            best_score = score
                            best_weights = {model_names[0]: w1, model_names[1]: w2, model_names[2]: w3}
            elif n_models == 2:
                for w1 in np.arange(0, 1.01, step):
                    w2 = 1.0 - w1
                    pred = w1 * predictions[model_names[0]] + w2 * predictions[model_names[1]]
                    pred = np.clip(pred, 0, 1)
                    score = r2_score(y_val, pred)
                    if score > best_score:
                        best_score = score
                        best_weights = {model_names[0]: w1, model_names[1]: w2}

            print(f"  Optimized R²: {max(0, 100*best_score):.4f}")
            weights = best_weights
        else:
            # Softmax-like weighting fallback
            max_score = max(scores.values())
            exp_scores = {k: np.exp(5 * (v - max_score)) for k, v in scores.items()}
            total = sum(exp_scores.values())
            weights = {k: v / total for k, v in exp_scores.items()}

        print("\n" + "=" * 60)
        print("ENSEMBLE WEIGHTS")
        print("=" * 60)
        for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {name:12s}: weight = {weight:.4f}  (val R² = {scores.get(name, 0):.4f})")

        self.ensemble_weights = weights
        return weights

    def predict_ensemble(
        self,
        X: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Generate ensemble predictions."""
        if weights is None:
            weights = self.compute_ensemble_weights()

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X[self.feature_cols])
            pred = np.clip(pred, 0, 1)
            predictions[name] = pred

        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += weights.get(name, 0) * pred

        return np.clip(ensemble_pred, 0, 1)

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Get aggregated feature importance across models."""
        importances = []
        if 'lightgbm' in self.models:
            imp = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.models['lightgbm'].feature_importances_,
                'model': 'lightgbm'
            })
            imp['importance'] = imp['importance'] / imp['importance'].sum()
            importances.append(imp)

        if 'xgboost' in self.models:
            imp = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.models['xgboost'].feature_importances_,
                'model': 'xgboost'
            })
            imp['importance'] = imp['importance'] / imp['importance'].sum()
            importances.append(imp)

        if importances:
            all_imp = pd.concat(importances)
            avg_imp = all_imp.groupby('feature')['importance'].mean().sort_values(ascending=False)
            return avg_imp.head(top_n).reset_index()

        return pd.DataFrame()

    def save_metadata(self):
        """Save training metadata."""
        meta = {
            'val_scores': self.val_scores,
            'best_params': {k: {pk: str(pv) for pk, pv in v.items()} for k, v in self.best_params.items()},
            'feature_cols': self.feature_cols,
            'ensemble_weights': getattr(self, 'ensemble_weights', {})
        }
        with open(os.path.join(self.model_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
