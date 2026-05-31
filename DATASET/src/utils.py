"""
Utility functions for Gridlock 2.0 Traffic Demand Prediction.
Handles geohash processing, validation splits, and common transformations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import math
import warnings
warnings.filterwarnings('ignore')


# ========== GEOHASH UTILITIES ==========

# Base32 character set used by geohash encoding
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}


def geohash_decode(ghash: str) -> Tuple[float, float]:
    """Decode a geohash string into (latitude, longitude) center coordinates."""
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_lon = True
    for ch in ghash:
        bits = _BASE32_MAP.get(ch, 0)
        for mask in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bits & mask:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bits & mask:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_lon = not is_lon
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon


def geohash_neighbors(ghash: str) -> List[str]:
    """Get the 8 neighboring geohashes of a given geohash."""
    lat, lon = geohash_decode(ghash)
    # Approximate cell dimensions for 6-char geohash
    # ~1.2km x 0.6km
    dlat = 0.006
    dlon = 0.012
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nlat = lat + dy * dlat
            nlon = lon + dx * dlon
            neighbors.append(geohash_encode(nlat, nlon, precision=len(ghash)))
    return neighbors


def geohash_encode(lat: float, lon: float, precision: int = 6) -> str:
    """Encode latitude/longitude into a geohash string."""
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    chars = []
    bits = 0
    bit_count = 0
    is_lon = True
    while len(chars) < precision:
        if is_lon:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon >= mid:
                bits = bits * 2 + 1
                lon_interval[0] = mid
            else:
                bits = bits * 2
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat >= mid:
                bits = bits * 2 + 1
                lat_interval[0] = mid
            else:
                bits = bits * 2
                lat_interval[1] = mid
        is_lon = not is_lon
        bit_count += 1
        if bit_count == 5:
            chars.append(_BASE32[bits])
            bits = 0
            bit_count = 0
    return ''.join(chars)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ========== TIMESTAMP UTILITIES ==========

def parse_timestamp(ts: str) -> Tuple[int, int]:
    """Parse 'H:M' or 'H:MM' timestamp into (hour, minute)."""
    parts = ts.split(':')
    return int(parts[0]), int(parts[1])


def timestamp_to_slot(ts: str) -> int:
    """Convert timestamp to 15-min slot index (0-95)."""
    h, m = parse_timestamp(ts)
    return h * 4 + m // 15


def slot_to_timestamp(slot: int) -> str:
    """Convert slot index back to timestamp string."""
    h = slot // 4
    m = (slot % 4) * 15
    return f"{h}:{m}"


# ========== VALIDATION SPLITS ==========

def temporal_train_val_split(
    df: pd.DataFrame,
    val_hours: int = 6,
    day_col: str = 'day',
    timestamp_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Day 48 data into train/val by time.
    Last `val_hours` hours become validation.
    """
    d48 = df[df[day_col] == 48].copy()
    d48['_slot'] = d48[timestamp_col].apply(timestamp_to_slot)
    cutoff_slot = 96 - val_hours * 4  # e.g., for 6 hours: slot 72 (18:00)
    train = d48[d48['_slot'] < cutoff_slot].drop(columns=['_slot'])
    val = d48[d48['_slot'] >= cutoff_slot].drop(columns=['_slot'])
    return train, val


def day_based_split(
    df: pd.DataFrame,
    day_col: str = 'day'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train on Day 48, validate on Day 49 (0:00-2:00).
    """
    train = df[df[day_col] == 48].copy()
    val = df[df[day_col] == 49].copy()
    return train, val


def grouped_time_kfold(
    df: pd.DataFrame,
    n_folds: int = 5,
    timestamp_col: str = 'timestamp'
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    K-fold where each fold holds out a contiguous block of hours from Day 48.
    Returns list of (train_df, val_df) tuples.
    """
    d48 = df[df['day'] == 48].copy()
    d48['_slot'] = d48[timestamp_col].apply(timestamp_to_slot)
    slots_per_fold = 96 // n_folds
    folds = []
    for i in range(n_folds):
        start_slot = i * slots_per_fold
        end_slot = (i + 1) * slots_per_fold if i < n_folds - 1 else 96
        val = d48[(d48['_slot'] >= start_slot) & (d48['_slot'] < end_slot)].drop(columns=['_slot'])
        train = d48[(d48['_slot'] < start_slot) | (d48['_slot'] >= end_slot)].drop(columns=['_slot'])
        folds.append((train, val))
    return folds


# ========== DATA LOADING ==========

def load_data(data_dir: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs."""
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    return train, test


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> bool:
    """Validate submission format and contents."""
    expected_shape = (len(test), 2)
    if submission.shape != expected_shape:
        print(f"ERROR: Shape {submission.shape} != expected {expected_shape}")
        return False
    if list(submission.columns) != ['Index', 'demand']:
        print(f"ERROR: Columns {list(submission.columns)} != ['Index', 'demand']")
        return False
    if submission['demand'].isnull().any():
        print(f"ERROR: {submission['demand'].isnull().sum()} NaN values in demand")
        return False
    if (submission['demand'] < 0).any() or (submission['demand'] > 1).any():
        print(f"WARNING: {((submission['demand'] < 0) | (submission['demand'] > 1)).sum()} predictions outside [0, 1]")
    if not (submission['Index'].values == test['Index'].values).all():
        print("ERROR: Index mismatch with test file")
        return False
    print(f"✓ Submission valid: {submission.shape[0]} rows, demand range [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
    return True
