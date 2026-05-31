import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check if same geohash has consistent static features
print("=== Feature consistency check ===")
static_cols = ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']
for c in static_cols:
    nuniq = train.groupby('geohash')[c].nunique()
    inconsistent = nuniq[nuniq > 1]
    print(f"{c}: {len(inconsistent)} geohashes have INCONSISTENT values")
    if len(inconsistent) > 0:
        for gh in inconsistent.index[:3]:
            vals = train[train['geohash']==gh][c].unique()
            print(f"  {gh}: {vals}")

# Also check across train+test
print("\n=== Cross train/test feature consistency ===")
combined = pd.concat([train[['geohash'] + static_cols], test[['geohash'] + static_cols]])
for c in static_cols:
    nuniq = combined.groupby('geohash')[c].nunique()
    inconsistent = nuniq[nuniq > 1]
    print(f"{c}: {len(inconsistent)} geohashes inconsistent across train+test")

# Check temporal pattern for a high-demand geohash
print("\n=== Temporal demand for qp09d9 (highest avg) ===")
gh = 'qp09d9'
sub = train[train['geohash']==gh].copy()
sub['hour'] = sub['timestamp'].apply(lambda x: int(x.split(':')[0]))
sub['minute'] = sub['timestamp'].apply(lambda x: int(x.split(':')[1]))
sub = sub.sort_values(['day','hour','minute'])
print(sub[['day','timestamp','demand','RoadType','NumberofLanes']].head(20).to_string())

# Check geohash prefix patterns (spatial structure)
print("\n=== Geohash prefix distribution ===")
train['gh_prefix4'] = train['geohash'].str[:4]
print(train['gh_prefix4'].value_counts().to_dict())

# Check if demand varies by timestamp within same geohash
print("\n=== Demand variance within geohash vs between ===")
d48 = train[train['day']==48]
within_var = d48.groupby('geohash')['demand'].var().mean()
total_var = d48['demand'].var()
print(f"Mean within-geohash variance: {within_var:.6f}")
print(f"Total variance: {total_var:.6f}")
print(f"Between-geohash explains: {(total_var - within_var)/total_var*100:.1f}% of variance")

# Check autocorrelation: does demand at t-1 predict demand at t?
print("\n=== Lag features importance check ===")
d48_sorted = d48.sort_values(['geohash', 'timestamp']).copy()
d48_sorted['demand_lag1'] = d48_sorted.groupby('geohash')['demand'].shift(1)
valid = d48_sorted.dropna(subset=['demand_lag1'])
corr = valid['demand'].corr(valid['demand_lag1'])
print(f"Lag-1 autocorrelation within geohash: {corr:.4f}")

# Check geohash neighbor structure
print("\n=== Sample geohashes (first 30) ===")
ghs = sorted(train['geohash'].unique())[:30]
print(ghs)

# Demand distribution check
print("\n=== Demand percentiles ===")
print(train['demand'].quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))

# Check Temperature-Weather correlation
print("\n=== Temperature by Weather ===")
tw = train.dropna(subset=['Temperature', 'Weather'])
print(tw.groupby('Weather')['Temperature'].describe())
