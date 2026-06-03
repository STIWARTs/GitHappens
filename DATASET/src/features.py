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

        # ---------- Per-geohash temporal profile stats ----------
        # Morning (slots 8-48, ~2:00-12:00) vs afternoon (slots 48-72, ~12:00-18:00)
        d48['_morning'] = ((d48['_slot'] >= 8) & (d48['_slot'] <= 48)).astype(int)
        d48['_afternoon'] = ((d48['_slot'] > 48) & (d48['_slot'] <= 72)).astype(int)
        d48['_early_morning'] = ((d48['_slot'] >= 0) & (d48['_slot'] < 24)).astype(int)
        d48['_late_morning'] = ((d48['_slot'] >= 32) & (d48['_slot'] <= 56)).astype(int)

        gh_morning = d48[d48['_morning'] == 1].groupby('geohash')['demand'].agg(['mean', 'std']).rename(
            columns={'mean': 'gh_morning_mean', 'std': 'gh_morning_std'})
        gh_afternoon = d48[d48['_afternoon'] == 1].groupby('geohash')['demand'].agg(['mean']).rename(
            columns={'mean': 'gh_afternoon_mean'})
        gh_early = d48[d48['_early_morning'] == 1].groupby('geohash')['demand'].agg(['mean']).rename(
            columns={'mean': 'gh_early_morning_mean'})
        gh_late_morning = d48[d48['_late_morning'] == 1].groupby('geohash')['demand'].agg(['mean']).rename(
            columns={'mean': 'gh_late_morning_mean'})

        self.gh_temporal_profile = gh_morning.join(gh_afternoon, how='outer').join(
            gh_early, how='outer').join(gh_late_morning, how='outer').to_dict('index')

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

        # ===== TEMPORAL PROFILE FEATURES =====
        # These capture time-of-day shape and generalize across days
        # Morning ramp: demand change from slot 8 to current slot's D48 value
        result['morning_ramp'] = result['gh_d48_same_slot'] - result['gh_mean']
        # Normalized time-of-day position within geohash's range
        result['tod_position'] = (result['gh_d48_same_slot'] - result['gh_min']) / (result['gh_max'] - result['gh_min'] + 1e-10)
        # Distance from start of known Day49 history (slot 9 = first test slot)
        result['slot_distance_from_history'] = np.clip(result['time_slot'] - 8, 0, 60)
        # Proximity to morning peak hours (slot 44 = 11:00, slot 52 = 13:00)
        result['morning_peak_proximity'] = 1.0 - np.minimum(
            np.abs(result['time_slot'] - 44),
            np.abs(result['time_slot'] - 52)
        ) / 48.0
        # Is the slot in the rising phase (morning) or falling phase
        result['is_rising_phase'] = ((result['time_slot'] >= 0) & (result['time_slot'] <= 48)).astype(int)
        # Hour-of-day normalized demand expectation (ratio of hour mean to geohash mean)
        result['hour_demand_ratio'] = result['gh_hour_mean'] / (result['gh_mean'] + 1e-10)
        # Slot normalized demand (ratio of slot mean to hour mean)
        result['slot_hour_ratio'] = result['gh_slot_mean'] / (result['gh_hour_mean'] + 1e-10)

        # ===== GEOHASH TEMPORAL PROFILE (from fit) =====
        if hasattr(self, 'gh_temporal_profile'):
            tp = self.gh_temporal_profile
            global_morning = self.global_mean_demand
            for col, default in [('gh_morning_mean', global_morning), ('gh_morning_std', 0),
                                  ('gh_afternoon_mean', global_morning), ('gh_early_morning_mean', global_morning),
                                  ('gh_late_morning_mean', global_morning)]:
                result[col] = result['geohash'].map(
                    lambda gh, c=col, d=default: tp.get(gh, {}).get(c, d)
                )
            # Morning-to-overall ratio (how much higher is morning demand vs average)
            result['gh_morning_ratio'] = result['gh_morning_mean'] / (result['gh_mean'] + 1e-10)
            # Morning-to-afternoon ratio (shape of the demand curve)
            result['gh_morning_afternoon_ratio'] = result['gh_morning_mean'] / (result['gh_afternoon_mean'] + 1e-10)
            # Late morning boost (how much higher is late morning vs early morning)
            result['gh_late_morning_boost'] = result['gh_late_morning_mean'] - result['gh_early_morning_mean']

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
