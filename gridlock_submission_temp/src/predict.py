from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("DATASET"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--builder-path", type=Path, required=True)
    parser.add_argument("--meta-path", type=Path, default=Path("outputs/model_meta.json"))
    parser.add_argument(
        "--output-path", type=Path, default=Path("outputs/submission.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = args.data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found in {args.data_dir}")

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    feature_columns = meta.get("model_feature_columns", meta["feature_columns"])
    model_type = meta["model_type"]

    feature_builder = joblib.load(args.builder_path)
    test_df = pd.read_csv(test_path)
    features = feature_builder.transform(test_df)

    if model_type == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(args.model_path)
        preds = model.predict(features[feature_columns])
    elif model_type == "lightgbm":
        model = joblib.load(args.model_path)
        preds = model.predict(features[feature_columns])
    elif model_type == "sklearn":
        model = joblib.load(args.model_path)
        preds = model.predict(features[feature_columns])
    else:
        raise ValueError(f"unsupported model_type: {model_type}")

    submission = pd.DataFrame({"Index": test_df["Index"], "demand": preds})
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    print(f"Saved submission to {args.output_path}")


if __name__ == "__main__":
    main()

