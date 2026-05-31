from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BITS = (16, 8, 4, 2, 1)

CATEGORICAL_COLUMNS = [
    "geohash",
    "geohash_4",
    "geohash_5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "geohash_time",
    "roadtype_time",
    "weather_time",
]

TARGET_ENCODER_COLUMNS = [
    "geohash",
    "geohash_time",
    "roadtype_time",
    "weather_time",
]


def parse_timestamp_to_minutes(value: str) -> int:
    if pd.isna(value):
        raise ValueError("timestamp is missing")
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid timestamp: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"invalid timestamp: {value!r}")
    return hour * 60 + minute


def decode_geohash(geohash: str) -> Tuple[float, float]:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    even = True
    for char in geohash.lower():
        idx = BASE32.find(char)
        if idx == -1:
            raise ValueError(f"invalid geohash character: {char!r}")
        for mask in BITS:
            if even:
                mid = (lon_range[0] + lon_range[1]) / 2.0
                if idx & mask:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2.0
                if idx & mask:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            even = not even
    lat = (lat_range[0] + lat_range[1]) / 2.0
    lon = (lon_range[0] + lon_range[1]) / 2.0
    return lat, lon


def decode_geohash_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    unique_values = series.dropna().unique()
    mapping: Dict[str, Tuple[float, float]] = {}
    for value in unique_values:
        mapping[value] = decode_geohash(value)
    lat = series.map(lambda x: mapping.get(x, (np.nan, np.nan))[0])
    lon = series.map(lambda x: mapping.get(x, (np.nan, np.nan))[1])
    return lat, lon


class TargetEncoder:
    def __init__(
        self,
        cols: Iterable[str],
        n_splits: int = 5,
        smoothing: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.cols = list(cols)
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_: float = np.nan
        self.mapping_: Dict[str, pd.Series] = {}

    def _fit_column(self, series: pd.Series, target: pd.Series) -> pd.Series:
        stats = target.groupby(series).agg(["sum", "count"])
        enc = (stats["sum"] + self.global_mean_ * self.smoothing) / (
            stats["count"] + self.smoothing
        )
        return enc

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TargetEncoder":
        self.global_mean_ = float(target.mean())
        self.mapping_ = {}
        for col in self.cols:
            self.mapping_[col] = self._fit_column(features[col], target)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=features.index)
        for col in self.cols:
            mapping = self.mapping_.get(col)
            if mapping is None:
                raise ValueError(f"target encoder not fitted for column {col}")
            encoded[col + "_te"] = features[col].map(mapping).fillna(self.global_mean_)
        return encoded

    def fit_transform(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.global_mean_ = float(target.mean())
        oof = pd.DataFrame(index=features.index)
        for col in self.cols:
            oof[col + "_te"] = np.nan

        kf = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for train_idx, val_idx in kf.split(features):
            train_features = features.iloc[train_idx]
            train_target = target.iloc[train_idx]
            val_features = features.iloc[val_idx]
            for col in self.cols:
                mapping = self._fit_column(train_features[col], train_target)
                oof.loc[val_features.index, col + "_te"] = val_features[col].map(
                    mapping
                )

        for col in self.cols:
            oof[col + "_te"] = oof[col + "_te"].fillna(self.global_mean_)

        self.fit(features, target)
        return oof


@dataclass
class FeatureBuilder:
    use_target_encoding: bool = True
    te_cols: Optional[List[str]] = None
    te_splits: int = 5
    te_smoothing: float = 10.0
    random_state: int = 42
    temp_median_by_geo_weather_: Dict[Tuple[str, str], float] = field(
        default_factory=dict
    )
    global_temp_median_: float = np.nan
    target_encoder_: Optional[TargetEncoder] = None

    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()

        features["timestamp"] = features["timestamp"].astype(str).str.strip()
        features["time_minutes"] = features["timestamp"].apply(
            parse_timestamp_to_minutes
        )
        features["time_bin_15"] = (features["time_minutes"] // 15).astype(int)
        features["hour"] = (features["time_minutes"] // 60).astype(int)
        features["minute"] = (features["time_minutes"] % 60).astype(int)
        angle = 2 * np.pi * features["time_bin_15"] / 96.0
        features["time_sin"] = np.sin(angle)
        features["time_cos"] = np.cos(angle)
        features["is_day49"] = (features["day"] == 49).astype(int)

        features["RoadType"] = features["RoadType"].fillna("Unknown").astype(str)
        features["LargeVehicles"] = (
            features["LargeVehicles"].fillna("Unknown").astype(str)
        )
        features["Landmarks"] = features["Landmarks"].fillna("Unknown").astype(str)
        features["Weather"] = features["Weather"].fillna("Unknown").astype(str)

        features["large_vehicles_allowed"] = (
            features["LargeVehicles"].str.lower().eq("allowed").astype(int)
        )
        features["has_landmarks"] = (
            features["Landmarks"].str.lower().eq("yes").astype(int)
        )

        features["Temperature"] = pd.to_numeric(
            features["Temperature"], errors="coerce"
        )

        features["geohash_4"] = features["geohash"].str.slice(0, 4)
        features["geohash_5"] = features["geohash"].str.slice(0, 5)

        lat, lon = decode_geohash_series(features["geohash"])
        features["latitude"] = lat
        features["longitude"] = lon
        features["geo_distance"] = np.sqrt(
            (features["latitude"] - features["latitude"].mean()) ** 2
            + (features["longitude"] - features["longitude"].mean()) ** 2
        )

        features["geohash_time"] = (
            features["geohash"].astype(str)
            + "_"
            + features["time_bin_15"].astype(str)
        )
        features["roadtype_time"] = (
            features["RoadType"].astype(str)
            + "_"
            + features["time_bin_15"].astype(str)
        )
        features["weather_time"] = (
            features["Weather"].astype(str)
            + "_"
            + features["time_bin_15"].astype(str)
        )

        return features

    def _fit_temperature_stats(self, features: pd.DataFrame) -> None:
        self.global_temp_median_ = float(features["Temperature"].median())
        grouped = features.groupby(["geohash", "Weather"])["Temperature"].median()
        self.temp_median_by_geo_weather_ = grouped.dropna().to_dict()

    def _impute_temperature(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.temp_median_by_geo_weather_:
            return features
        keys = list(zip(features["geohash"], features["Weather"]))
        mapped = pd.Series(keys, index=features.index).map(self.temp_median_by_geo_weather_)
        features["Temperature"] = (
            features["Temperature"].fillna(mapped).fillna(self.global_temp_median_)
        )
        return features

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "FeatureBuilder":
        base = self._basic_features(df)
        self._fit_temperature_stats(base)
        if self.use_target_encoding and target is not None:
            cols = self.te_cols or TARGET_ENCODER_COLUMNS
            self.target_encoder_ = TargetEncoder(
                cols=cols,
                n_splits=self.te_splits,
                smoothing=self.te_smoothing,
                random_state=self.random_state,
            )
            self.target_encoder_.fit(base, target)
        return self

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        base = self._basic_features(df)
        self._fit_temperature_stats(base)
        base = self._impute_temperature(base)
        if self.use_target_encoding:
            cols = self.te_cols or TARGET_ENCODER_COLUMNS
            self.target_encoder_ = TargetEncoder(
                cols=cols,
                n_splits=self.te_splits,
                smoothing=self.te_smoothing,
                random_state=self.random_state,
            )
            te_features = self.target_encoder_.fit_transform(base, target)
            base = pd.concat([base, te_features], axis=1)
        return base

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if np.isnan(self.global_temp_median_):
            raise ValueError("FeatureBuilder must be fitted before calling transform.")
        base = self._basic_features(df)
        base = self._impute_temperature(base)
        if self.use_target_encoding:
            if self.target_encoder_ is None:
                raise ValueError("Target encoder is not fitted.")
            te_features = self.target_encoder_.transform(base)
            base = pd.concat([base, te_features], axis=1)
        return base


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {"demand", "Index", "timestamp"}
    return [col for col in df.columns if col not in drop_cols]

