"""
Feature Engineering Pipeline for Gridlock 2.0 Traffic Demand Prediction.
Builds all features from raw train/test data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils import (
    parse_timestamp, timestamp_to_slot, geohash_decode
)


class FeatureEngine:
    """
    Stateful feature engineering pipeline.
    Fits on training data (computes aggregates), transforms both train and test.
    """

    def __init__(self):
        self.geohash_stats: Dict = {}          # Per-geohash demand statistics
        self.geohash_hour_stats: Dict = {}     # Per-(geohash, hour) demand stats
        self.geohash_slot_stats: Dict = {}     # Per-(geohash, slot) demand stats
        self.geohash_modal_features: Dict = {} # Modal static features per geohash
        self.prefix5_stats: Dict = {}          # Spatial region stats (5-char prefix)
        self.prefix4_stats: Dict = {}          # Broader spatial region stats
        self.global_hour_stats: Dict = {}      # Global demand by hour
        self.global_slot_stats: Dict = {}      # Global demand by slot
        self.global_mean_demand: float = 0.0
        self.global_median_demand: float = 0.0
        self.weather_temp_map: Dict = {}       # Weather -> mean temperature
        self.feature_cols: list = []
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> 'FeatureEngine':
        """
        Compute all aggregate statistics from training data.
        Uses Day 48 as the primary reference.
        """
        df = train_df.copy()
        df['_hour'] = df['timestamp'].apply(lambda x: parse_timestamp(x)[0])
        df['_slot'] = df['timestamp'].apply(timestamp_to_slot)

        # Use Day 48 for stable aggregate computation (full 24h)
        d48 = df[df['day'] == 48]

        # ---------- Global statistics ----------
        self.global_mean_demand = d48['demand'].mean()
        self.global_median_demand = d48['demand'].median()

        # ---------- Per-geohash demand stats ----------
        gh_agg = d48.groupby('geohash')['demand'].agg(
            ['mean', 'median', 'std', 'min', 'max', 'count']
        )
        gh_agg['q25'] = d48.groupby('geohash')['demand'].quantile(0.25)
        gh_agg['q75'] = d48.groupby('geohash')['demand'].quantile(0.75)
        gh_agg['iqr'] = gh_agg['q75'] - gh_agg['q25']
        gh_agg['cv'] = gh_agg['std'] / (gh_agg['mean'] + 1e-10)  # Coefficient of variation
        self.geohash_stats = gh_agg.to_dict('index')

        # ---------- Per-(geohash, hour) stats ----------
        gh_hr = d48.groupby(['geohash', '_hour'])['demand'].agg(['mean', 'median', 'std', 'min', 'max'])
        self.geohash_hour_stats = {}
        for (gh, hr), row in gh_hr.iterrows():
            self.geohash_hour_stats[(gh, hr)] = row.to_dict()

        # ---------- Per-(geohash, slot) stats ----------
        gh_slot = d48.groupby(['geohash', '_slot'])['demand'].agg(['mean', 'median', 'std', 'last'])
        self.geohash_slot_stats = {}
        for (gh, sl), row in gh_slot.iterrows():
            self.geohash_slot_stats[(gh, sl)] = row.to_dict()

        # ---------- Global hour/slot stats ----------
        self.global_hour_stats = d48.groupby('_hour')['demand'].agg(['mean', 'median', 'std']).to_dict('index')
        self.global_slot_stats = d48.groupby('_slot')['demand'].agg(['mean', 'median']).to_dict('index')

        # ---------- Modal static features per geohash ----------
        # Use ALL training data for modal computation (more samples = more robust mode)
        for gh in df['geohash'].unique():
            gh_rows = df[df['geohash'] == gh]
            modal = {}
            for col in ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']:
                mode_vals = gh_rows[col].dropna().mode()
                modal[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else None
            self.geohash_modal_features[gh] = modal

        # ---------- Spatial prefix stats ----------
        d48_with_prefix = d48.copy()
        d48_with_prefix['_prefix5'] = d48_with_prefix['geohash'].str[:5]
        d48_with_prefix['_prefix4'] = d48_with_prefix['geohash'].str[:4]

        p5_agg = d48_with_prefix.groupby('_prefix5')['demand'].agg(['mean', 'median', 'std', 'count'])
        self.prefix5_stats = p5_agg.to_dict('index')

        p4_agg = d48_with_prefix.groupby('_prefix4')['demand'].agg(['mean', 'median', 'std', 'count'])
        self.prefix4_stats = p4_agg.to_dict('index')

        # ---------- Weather -> Temperature mapping ----------
        wt = df.dropna(subset=['Weather', 'Temperature'])
        self.weather_temp_map = wt.groupby('Weather')['Temperature'].mean().to_dict()

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Apply all feature transformations to a DataFrame.
        """
        assert self._fitted, "Must call fit() before transform()"

        result = df.copy()

        # ===== TEMPORAL FEATURES =====
        result['hour'] = result['timestamp'].apply(lambda x: parse_timestamp(x)[0])
        result['minute'] = result['timestamp'].apply(lambda x: parse_timestamp(x)[1])
        result['time_slot'] = result['timestamp'].apply(timestamp_to_slot)

        # Cyclical encoding
        result['sin_hour'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['cos_hour'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['sin_slot'] = np.sin(2 * np.pi * result['time_slot'] / 96)
        result['cos_slot'] = np.cos(2 * np.pi * result['time_slot'] / 96)

        # Binary time indicators
        result['is_morning_rush'] = ((result['hour'] >= 7) & (result['hour'] <= 10)).astype(int)
        result['is_evening_low'] = ((result['hour'] >= 16) & (result['hour'] <= 20)).astype(int)
        result['is_peak_demand'] = ((result['hour'] >= 9) & (result['hour'] <= 13)).astype(int)
        result['is_night'] = ((result['hour'] >= 0) & (result['hour'] <= 5)).astype(int)

        # ===== GEOHASH IDENTITY FEATURES =====
        # Per-geohash aggregate stats from Day 48
        for stat in ['mean', 'median', 'std', 'min', 'max', 'count', 'q25', 'q75', 'iqr', 'cv']:
            result[f'gh_{stat}'] = result['geohash'].map(
                lambda gh, s=stat: self.geohash_stats.get(gh, {}).get(s, self.global_mean_demand if s == 'mean' else 0)
            )

        # ===== GEOHASH × TIME INTERACTION FEATURES =====
        # Per-(geohash, hour) stats
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            result[f'gh_hour_{stat}'] = result.apply(
                lambda row, s=stat: self.geohash_hour_stats.get(
                    (row['geohash'], row['hour']), {}
                ).get(s, self.geohash_stats.get(row['geohash'], {}).get(s, self.global_mean_demand if s == 'mean' else 0)),
                axis=1
            )

        # Per-(geohash, time_slot) stats — finest grain
        for stat in ['mean', 'median', 'last']:
            result[f'gh_slot_{stat}'] = result.apply(
                lambda row, s=stat: self.geohash_slot_stats.get(
                    (row['geohash'], row['time_slot']), {}
                ).get(s, self.geohash_stats.get(row['geohash'], {}).get('mean' if s == 'last' else s, self.global_mean_demand)),
                axis=1
            )

        # Demand at Day 48 same slot (direct lookup for day-over-day comparison)
        result['gh_d48_same_slot'] = result.apply(
            lambda row: self.geohash_slot_stats.get(
                (row['geohash'], row['time_slot']), {}
            ).get('mean', self.geohash_stats.get(row['geohash'], {}).get('mean', self.global_mean_demand)),
            axis=1
        )

        # ===== GLOBAL TIME FEATURES =====
        for stat in ['mean', 'median', 'std']:
            result[f'global_hour_{stat}'] = result['hour'].map(
                lambda h, s=stat: self.global_hour_stats.get(h, {}).get(s, self.global_mean_demand if s == 'mean' else 0)
            )

        result['global_slot_mean'] = result['time_slot'].map(
            lambda s: self.global_slot_stats.get(s, {}).get('mean', self.global_mean_demand)
        )

        # Ratio: geohash demand vs global demand at same time
        result['gh_vs_global_ratio'] = result['gh_hour_mean'] / (result['global_hour_mean'] + 1e-10)

        # ===== SPATIAL FEATURES =====
        result['gh_prefix5'] = result['geohash'].str[:5]
        result['gh_prefix4'] = result['geohash'].str[:4]
        result['gh_prefix3'] = result['geohash'].str[:3]

        for stat in ['mean', 'median', 'std', 'count']:
            result[f'prefix5_{stat}'] = result['gh_prefix5'].map(
                lambda p, s=stat: self.prefix5_stats.get(p, {}).get(s, self.global_mean_demand if s == 'mean' else 0)
            )

        for stat in ['mean', 'median']:
            result[f'prefix4_{stat}'] = result['gh_prefix4'].map(
                lambda p, s=stat: self.prefix4_stats.get(p, {}).get(s, self.global_mean_demand if s == 'mean' else 0)
            )

        # Spatial rank: how does this geohash compare to its prefix5 neighbors?
        result['spatial_rank_ratio'] = result['gh_mean'] / (result['prefix5_mean'] + 1e-10)

        # ===== GEOHASH COORDINATES =====
        # Decode lat/lon (cached per unique geohash)
        unique_ghs = result['geohash'].unique()
        gh_coords = {gh: geohash_decode(gh) for gh in unique_ghs}
        result['latitude'] = result['geohash'].map(lambda gh: gh_coords[gh][0])
        result['longitude'] = result['geohash'].map(lambda gh: gh_coords[gh][1])

        # ===== ROAD INFRASTRUCTURE FEATURES =====
        # Row-level features (noisy)
        road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
        result['road_type_encoded'] = result['RoadType'].map(road_map).fillna(-1).astype(int)
        result['is_highway'] = (result['RoadType'] == 'Highway').astype(int)
        result['is_street'] = (result['RoadType'] == 'Street').astype(int)
        result['is_high_capacity'] = (result['NumberofLanes'] >= 4).astype(int)
        result['large_vehicles_encoded'] = (result['LargeVehicles'] == 'Allowed').astype(int)
        result['landmarks_encoded'] = (result['Landmarks'] == 'Yes').astype(int)

        # Modal (denoised) features per geohash
        result['modal_road_type'] = result['geohash'].map(
            lambda gh: road_map.get(self.geohash_modal_features.get(gh, {}).get('RoadType'), -1)
        )
        result['modal_lanes'] = result['geohash'].map(
            lambda gh: self.geohash_modal_features.get(gh, {}).get('NumberofLanes', 2)
        )
        result['modal_large_vehicles'] = result['geohash'].map(
            lambda gh: 1 if self.geohash_modal_features.get(gh, {}).get('LargeVehicles') == 'Allowed' else 0
        )
        result['modal_landmarks'] = result['geohash'].map(
            lambda gh: 1 if self.geohash_modal_features.get(gh, {}).get('Landmarks') == 'Yes' else 0
        )
        result['modal_is_highway'] = (result['modal_road_type'] == 2).astype(int)
        result['modal_is_high_capacity'] = (result['modal_lanes'] >= 4).astype(int)

        # ===== WEATHER FEATURES =====
        # Fill missing temperature from weather group mean
        result['temperature_filled'] = result['Temperature'].copy()
        for weather, mean_temp in self.weather_temp_map.items():
            mask = result['Temperature'].isnull() & (result['Weather'] == weather)
            result.loc[mask, 'temperature_filled'] = mean_temp
        # Fill remaining NaN with global mean
        result['temperature_filled'] = result['temperature_filled'].fillna(
            result['temperature_filled'].median()
        )

        weather_map = {'Snowy': 0, 'Rainy': 1, 'Foggy': 2, 'Sunny': 3}
        result['weather_encoded'] = result['Weather'].map(weather_map).fillna(-1).astype(int)

        # ===== INTERACTION FEATURES =====
        result['gh_mean_x_hour'] = result['gh_mean'] * result['sin_hour']
        result['modal_road_x_demand'] = result['modal_road_type'] * result['gh_mean']
        result['lanes_x_demand'] = result['modal_lanes'] * result['gh_mean']

        # ===== DEMAND DEVIATION FEATURES =====
        result['demand_deviation_from_gh_mean'] = result['gh_d48_same_slot'] - result['gh_mean']
        result['demand_deviation_from_global'] = result['gh_d48_same_slot'] - self.global_mean_demand
        result['demand_percentile_in_gh'] = (result['gh_d48_same_slot'] - result['gh_min']) / (result['gh_max'] - result['gh_min'] + 1e-10)

        # Store feature column list
        self.feature_cols = self._get_feature_columns(result)

        return result

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns (exclude identifiers and target)."""
        exclude = {
            'Index', 'geohash', 'day', 'timestamp', 'demand',
            'RoadType', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
            'gh_prefix5', 'gh_prefix4', 'gh_prefix3'
        }
        return [c for c in df.columns if c not in exclude]

    def get_feature_cols(self) -> list:
        """Return the list of feature column names."""
        return self.feature_cols


def build_lag_features_static(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    geohash_slot_stats: Dict
) -> pd.DataFrame:
    """
    Build lag features using Day 48 same-slot values as static proxies.
    For each test row at (geohash, slot), look up Day 48 demand at (geohash, slot-1), (slot-2), etc.
    """
    result = target_df.copy()
    result['time_slot'] = result['timestamp'].apply(timestamp_to_slot)

    for lag in [1, 2, 3, 4]:
        col_name = f'demand_lag_{lag}_static'
        result[col_name] = result.apply(
            lambda row, l=lag: geohash_slot_stats.get(
                (row['geohash'], row['time_slot'] - l), {}
            ).get('mean', geohash_slot_stats.get(
                (row['geohash'], (row['time_slot'] - l) % 96), {}
            ).get('mean', 0)),
            axis=1
        )

    # Rolling mean of lag values
    lag_cols = [f'demand_lag_{i}_static' for i in [1, 2, 3, 4]]
    result['demand_rolling_mean_4_static'] = result[lag_cols].mean(axis=1)
    result['demand_rolling_mean_2_static'] = result[[f'demand_lag_{i}_static' for i in [1, 2]]].mean(axis=1)

    # Momentum
    result['demand_diff_1_static'] = result['demand_lag_1_static'] - result['demand_lag_2_static']

    if 'time_slot' in result.columns and 'time_slot' not in target_df.columns:
        result = result.drop(columns=['time_slot'])

    return result


def build_lag_features_from_history(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build lag features using actual Day 49 history (0:00-2:00) where available,
    falling back to Day 48 same-slot values.
    
    history_df: Day 49 training rows (0:00-2:00) with demand values
    target_df: rows to compute lag features for
    """
    result = target_df.copy()
    result['time_slot'] = result['timestamp'].apply(timestamp_to_slot)

    # Build a lookup: (geohash, slot) -> demand from Day 49 history
    hist = history_df.copy()
    hist['time_slot'] = hist['timestamp'].apply(timestamp_to_slot)
    d49_lookup = hist.groupby(['geohash', 'time_slot'])['demand'].last().to_dict()

    for lag in [1, 2, 3, 4]:
        col_name = f'demand_lag_{lag}_hist'
        result[col_name] = result.apply(
            lambda row, l=lag: d49_lookup.get(
                (row['geohash'], row['time_slot'] - l),
                np.nan  # Will be filled later
            ),
            axis=1
        )

    lag_cols = [f'demand_lag_{i}_hist' for i in [1, 2, 3, 4]]
    result['demand_rolling_mean_4_hist'] = result[lag_cols].mean(axis=1)
    result['demand_rolling_mean_2_hist'] = result[[f'demand_lag_{i}_hist' for i in [1, 2]]].mean(axis=1)
    result['demand_diff_1_hist'] = result['demand_lag_1_hist'] - result['demand_lag_2_hist']

    if 'time_slot' in result.columns and 'time_slot' not in target_df.columns:
        result = result.drop(columns=['time_slot'])

    return result
