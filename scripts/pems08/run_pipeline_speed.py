import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


PROCESSED    = Path("data/processed")
GRAPH_PATH   = PROCESSED / "graph"    / "pems08.graphml"
SPEED_PATH   = PROCESSED / "labels"   / "pems08_speed_targets.npy"
FEATURES_NPZ = PROCESSED / "features" / "pems08_features_speed.npz"
METRICS_PATH = Path("reports") / "pems08_baselines_speed.json"

STEPS_PER_DAY    = 288
HORIZON          = 12
TRAIN_FRACTION   = 0.6
VAL_FRACTION     = 0.2
DATASET_START    = "2016-07-01 00:00:00"


t0 = time.time()

print("\n=== Loading PEMS08 ===")
data  = np.load("data/PEMS08/PEMS08.npz")["data"].astype(np.float32)
edges = pd.read_csv("data/PEMS08/PEMS08.csv").rename(columns={"cost": "distance"})
T, N, C = data.shape
timestamps = pd.date_range(DATASET_START, periods=T, freq="5min")
flow = data[:, :, 0]
occ  = data[:, :, 1]
spd  = data[:, :, 2]
print(f"data: T={T}  N={N}  C={C}  edges={len(edges)}")
print(f"time range: {timestamps[0]} → {timestamps[-1]}")

print("\n=== Building graph ===")
G = nx.DiGraph()
means = data.mean(axis=0)
stds  = data.std(axis=0)
channel_names = ["flow", "occupancy", "speed"]
for n in range(N):
    attrs = {"index": int(n)}
    for c, name in enumerate(channel_names):
        attrs[f"mean_{name}"] = float(means[n, c])
        attrs[f"std_{name}"]  = float(stds[n, c])
    G.add_node(n, **attrs)
distances = edges["distance"].to_numpy()
sigma = float(distances.std())
for _, row in edges.iterrows():
    d = float(row["distance"])
    w = float(np.exp(-(d ** 2) / (sigma ** 2)))
    G.add_edge(int(row["from"]), int(row["to"]), distance=d, weight=w)
GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
nx.write_graphml(G, GRAPH_PATH)
print(f"graph saved → {GRAPH_PATH}  (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")

print("\n=== Preparing speed targets ===")
y_speed = spd.copy()
speed_stats = {
    "mean": float(spd.mean()),
    "std": float(spd.std()),
    "min": float(spd.min()),
    "median": float(np.median(spd)),
    "max": float(spd.max()),
}
SPEED_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(SPEED_PATH, y_speed)
print(f"speed targets saved → {SPEED_PATH}")
print(f"speed stats: mean={speed_stats['mean']:.1f}  std={speed_stats['std']:.1f}  "
      f"min={speed_stats['min']:.1f}  median={speed_stats['median']:.1f}  max={speed_stats['max']:.1f} mph")

print("\n=== Building feature matrix ===")
LAGS = (1, 3, 12, STEPS_PER_DAY, 7 * STEPS_PER_DAY)
feats = [flow[..., None], occ[..., None], spd[..., None]]
names = ["flow", "occupancy", "speed"]
for lag in LAGS:
    lagged       = np.roll(spd, lag, axis=0)
    lagged[:lag] = spd[:lag]
    feats.append(lagged[..., None])
    names.append(f"speed_lag_{lag}")

tod     = np.arange(T) % STEPS_PER_DAY
sin_tod = np.sin(2 * np.pi * tod / STEPS_PER_DAY)
cos_tod = np.cos(2 * np.pi * tod / STEPS_PER_DAY)
feats.append(np.broadcast_to(sin_tod[:, None, None], (T, N, 1)).copy())
feats.append(np.broadcast_to(cos_tod[:, None, None], (T, N, 1)).copy())
names += ["sin_tod", "cos_tod"]

dow     = (np.arange(T) // STEPS_PER_DAY) % 7
sin_dow = np.sin(2 * np.pi * dow / 7)
cos_dow = np.cos(2 * np.pi * dow / 7)
feats.append(np.broadcast_to(sin_dow[:, None, None], (T, N, 1)).copy())
feats.append(np.broadcast_to(cos_dow[:, None, None], (T, N, 1)).copy())
names += ["sin_dow", "cos_dow"]

A_in                = nx.to_numpy_array(G, nodelist=range(N), weight=None).T
deg_in              = A_in.sum(axis=1, keepdims=True)
deg_in[deg_in == 0] = 1.0
A_in_norm           = A_in / deg_in
for arr, label in [(flow, "flow"), (occ, "occupancy"), (spd, "speed")]:
    feats.append((arr @ A_in_norm.T)[..., None])
    names.append(f"nbr_in_mean_{label}")

X_full    = np.concatenate(feats, axis=-1).astype(np.float32)
X         = X_full[:-HORIZON]
y_shifted = y_speed[HORIZON:]
T_eff     = X.shape[0]

n_train   = int(T_eff * TRAIN_FRACTION)
n_val     = int(T_eff * (TRAIN_FRACTION + VAL_FRACTION))
train_idx = np.zeros(T_eff, dtype=bool); train_idx[:n_train]    = True
val_idx   = np.zeros(T_eff, dtype=bool); val_idx[n_train:n_val] = True
test_idx  = np.zeros(T_eff, dtype=bool); test_idx[n_val:]       = True

y_current = y_speed[:T_eff]
FEATURES_NPZ.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    FEATURES_NPZ,
    X=X, y=y_shifted, y_current=y_current,
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    feature_names=np.array(names),
    horizon=np.int32(HORIZON),
)
print(f"features saved → {FEATURES_NPZ}  (X shape = {X.shape}, F = {len(names)}, horizon = {HORIZON} steps = {HORIZON*5} min)")
print(f"split sizes (timesteps): train={train_idx.sum()}  val={val_idx.sum()}  test={test_idx.sum()}")

print("\n=== Training baselines (test-set metrics, 1-hour-ahead speed prediction) ===")
X_train = X[train_idx].reshape(-1, X.shape[-1])
y_train = y_shifted[train_idx].reshape(-1)
X_test  = X[test_idx].reshape(-1, X.shape[-1])
y_test  = y_shifted[test_idx].reshape(-1)

def compute_mape(y_true, y_pred, epsilon=1e-3):
    """Mean Absolute Percentage Error with small epsilon to avoid division by zero"""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)

metrics = []
for name in ["mean", "persistence", "linreg"]:
    ts = time.time()
    if name == "mean":
        # Predict the training set mean for all test samples
        pred     = np.full_like(y_test, fill_value=y_train.mean(), dtype=np.float32)
        out_name = "mean_baseline"
    elif name == "persistence":
        # Use current speed as prediction for future speed
        pred     = y_current[test_idx].reshape(-1).astype(np.float32)
        out_name = "persistence"
    else:
        # Linear regression
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        model     = LinearRegression(n_jobs=-1)
        model.fit(X_train_s, y_train)
        pred      = model.predict(X_test_s).astype(np.float32)
        out_name  = "linear_regression"

    mae  = mean_absolute_error(y_test, pred)
    mse  = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, pred)
    mape = compute_mape(y_test, pred)

    m = {
        "name":     out_name,
        "mae":      float(mae),
        "mse":      float(mse),
        "rmse":     float(rmse),
        "r2":       float(r2),
        "mape":     float(mape),
        "n_total":  int(y_test.size),
    }
    print(f"[{name:11s}] fit+eval in {time.time()-ts:5.1f}s  →  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  MAPE={m['mape']:.2f}%")
    metrics.append(m)

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"\nmetrics saved → {METRICS_PATH}")
print(f"\nTotal pipeline time: {time.time()-t0:.1f}s")
