from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.feature_engineering import (
    CATEGORICAL_COLUMNS,
    FeatureBuilder,
    get_feature_columns,
)
from src.metrics import regression_metrics


DEFAULT_CATBOOST_PARAMS: Dict[str, object] = {
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 4.0,
    "random_seed": 42,
    "verbose": 200,
}

DEFAULT_LIGHTGBM_PARAMS: Dict[str, object] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
}

DEFAULT_SKLEARN_PARAMS: Dict[str, object] = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "max_iter": 400,
    "l2_regularization": 0.0,
    "random_state": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("DATASET"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--mode", choices=["validate", "full"], default="validate")
    parser.add_argument(
        "--model",
        choices=["sklearn", "catboost", "lightgbm"],
        default="sklearn",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--te-splits", type=int, default=5)
    parser.add_argument("--te-smoothing", type=float, default=10.0)
    return parser.parse_args()


def load_config(path: Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def split_by_day(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (df["day"] == 49).any():
        train_df = df[df["day"] == 48].copy()
        val_df = df[df["day"] == 49].copy()
        if train_df.empty or val_df.empty:
            raise ValueError("Day-based split failed; empty train or validation set.")
        return train_df, val_df
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
    return train_df, val_df


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    params: Dict[str, object],
    feature_cols: list[str],
) -> Tuple[object, Dict[str, float]]:
    from catboost import CatBoostRegressor, Pool

    train_pool = Pool(
        X_train[feature_cols],
        y_train,
        cat_features=[
            feature_cols.index(col)
            for col in CATEGORICAL_COLUMNS
            if col in feature_cols
        ],
    )
    metrics: Dict[str, float] = {}

    if X_val is not None and y_val is not None:
        val_pool = Pool(
            X_val[feature_cols],
            y_val,
            cat_features=[
                feature_cols.index(col)
                for col in CATEGORICAL_COLUMNS
                if col in feature_cols
            ],
        )
        model = CatBoostRegressor(**params, use_best_model=True)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200)
        preds = model.predict(X_val[feature_cols])
        metrics = regression_metrics(y_val, preds)
    else:
        model = CatBoostRegressor(**params)
        model.fit(train_pool)

    return model, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    params: Dict[str, object],
    feature_cols: list[str],
) -> Tuple[object, Dict[str, float]]:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    metrics: Dict[str, float] = {}

    if X_val is not None and y_val is not None:
        model.fit(
            X_train[feature_cols],
            y_train,
            eval_set=[(X_val[feature_cols], y_val)],
            eval_metric="r2",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        preds = model.predict(X_val[feature_cols])
        metrics = regression_metrics(y_val, preds)
    else:
        model.fit(X_train[feature_cols], y_train)

    return model, metrics


def train_sklearn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    params: Dict[str, object],
    feature_cols: list[str],
) -> Tuple[object, Dict[str, float]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(**params)
    metrics: Dict[str, float] = {}

    model.fit(X_train[feature_cols], y_train)
    if X_val is not None and y_val is not None:
        preds = model.predict(X_val[feature_cols])
        metrics = regression_metrics(y_val, preds)
    return model, metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_path = args.data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {args.data_dir}")

    df = pd.read_csv(train_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_builder = FeatureBuilder(
        use_target_encoding=True,
        te_splits=args.te_splits,
        te_smoothing=args.te_smoothing,
        random_state=args.seed,
    )

    if args.mode == "validate":
        train_df, val_df = split_by_day(df, args.seed)
        y_train = train_df["demand"]
        y_val = val_df["demand"]
        X_train = feature_builder.fit_transform(train_df, y_train)
        X_val = feature_builder.transform(val_df)
    else:
        y_train = df["demand"]
        X_train = feature_builder.fit_transform(df, y_train)
        X_val = None
        y_val = None

    model_params = config.get("model_params", {})
    feature_cols = get_feature_columns(X_train)
    numeric_feature_cols = [
        col for col in feature_cols if col not in CATEGORICAL_COLUMNS
    ]

    if args.model == "catboost":
        params = {**DEFAULT_CATBOOST_PARAMS, **model_params.get("catboost", {})}
        model, metrics = train_catboost(
            X_train, y_train, X_val, y_val, params, feature_cols
        )
        model_path = args.output_dir / f"model_{args.mode}.cbm"
        model.save_model(model_path)
        model_feature_cols = feature_cols
    elif args.model == "lightgbm":
        params = {**DEFAULT_LIGHTGBM_PARAMS, **model_params.get("lightgbm", {})}
        model, metrics = train_lightgbm(
            X_train, y_train, X_val, y_val, params, feature_cols
        )
        model_path = args.output_dir / f"model_{args.mode}.lgbm"
        joblib.dump(model, model_path)
        model_feature_cols = feature_cols
    else:
        params = {**DEFAULT_SKLEARN_PARAMS, **model_params.get("sklearn", {})}
        model, metrics = train_sklearn(
            X_train, y_train, X_val, y_val, params, numeric_feature_cols
        )
        model_path = args.output_dir / f"model_{args.mode}.sklearn"
        joblib.dump(model, model_path)
        model_feature_cols = numeric_feature_cols

    builder_path = args.output_dir / "feature_builder.pkl"
    joblib.dump(feature_builder, builder_path)

    meta = {
        "model_type": args.model,
        "train_mode": args.mode,
        "feature_columns": feature_cols,
        "model_feature_columns": model_feature_cols,
        "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in X_train.columns],
        "metrics": metrics,
        "model_path": str(model_path),
        "builder_path": str(builder_path),
    }
    meta_path = args.output_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if metrics:
        print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved feature builder to {builder_path}")


if __name__ == "__main__":
    main()

