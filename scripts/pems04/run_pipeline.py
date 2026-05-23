import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROCESSED = Path("data/processed")
GRAPH_PATH   = PROCESSED / "graph"    / "pems04.graphml"
LABELS_PATH  = PROCESSED / "labels"   / "pems04_speed_labels.npy"
VFREE_PATH   = PROCESSED / "labels"   / "pems04_v_free.npy"
LABELS_META  = PROCESSED / "labels"   / "pems04_labels_meta.json"
FEATURES_NPZ = PROCESSED / "features" / "pems04_features.npz"
METRICS_PATH = Path("reports") / "pems04_baselines.json"

STEPS_PER_DAY = 288
HORIZON = 12
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
THRESHOLD_RATIO = 0.6
FREE_FLOW_PERCENTILE = 95.0


t0 = time.time()

print("\n=== Loading PEMS04 ===")
data = np.load("data/PEMS04/PEMS04.npz")["data"].astype(np.float32)
edges = pd.read_csv("data/PEMS04/PEMS04.csv").rename(columns={"cost": "distance"})
T, N, C = data.shape
timestamps = pd.date_range(pd.Timestamp("2018-01-01 00:00:00"), periods=T, freq="5min")
flow = data[:, :, 0]
occ = data[:, :, 1]
spd = data[:, :, 2]
print(f"data: T={T}  N={N}  C={C}  edges={len(edges)}")
print(f"time range: {timestamps[0]} → {timestamps[-1]}")

print("\n=== Building graph ===")
G = nx.DiGraph()
means = data.mean(axis=0)
stds = data.std(axis=0)
channel_names = ["flow", "occupancy", "speed"]
for n in range(N):
    attrs = {"index": int(n)}
    for c, name in enumerate(channel_names):
        attrs[f"mean_{name}"] = float(means[n, c])
        attrs[f"std_{name}"] = float(stds[n, c])
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

print("\n=== Computing congestion labels ===")
train_end = int(T * TRAIN_FRACTION)
v_free = np.percentile(spd[:train_end], FREE_FLOW_PERCENTILE, axis=0)
threshold = v_free * THRESHOLD_RATIO
y_lab = (spd < threshold[None, :]).astype(np.uint8)
positive_rate = float(y_lab.mean())
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(LABELS_PATH, y_lab)
np.save(VFREE_PATH, v_free)
LABELS_META.write_text(json.dumps({
    "rule": "speed < threshold_ratio * v_free",
    "threshold_ratio": THRESHOLD_RATIO,
    "free_flow_percentile": FREE_FLOW_PERCENTILE,
    "train_fraction": TRAIN_FRACTION,
    "positive_rate": positive_rate,
    "v_free_min": float(v_free.min()),
    "v_free_median": float(np.median(v_free)),
    "v_free_max": float(v_free.max()),
}, indent=2))
print(f"labels saved → {LABELS_PATH}  (positive rate = {positive_rate:.4f})")
print(f"v_free per sensor: min={v_free.min():.1f}  median={np.median(v_free):.1f}  max={v_free.max():.1f} mph")

print("\n=== Building feature matrix ===")
LAGS = (1, 3, 12, STEPS_PER_DAY, 7 * STEPS_PER_DAY)
feats = [flow[..., None], occ[..., None], spd[..., None]]
names = ["flow", "occupancy", "speed"]
for lag in LAGS:
    lagged = np.roll(spd, lag, axis=0)
    lagged[:lag] = spd[:lag]
    feats.append(lagged[..., None])
    names.append(f"speed_lag_{lag}")

tod = np.arange(T) % STEPS_PER_DAY
sin_tod = np.sin(2 * np.pi * tod / STEPS_PER_DAY)
cos_tod = np.cos(2 * np.pi * tod / STEPS_PER_DAY)
feats.append(np.broadcast_to(sin_tod[:, None, None], (T, N, 1)).copy())
feats.append(np.broadcast_to(cos_tod[:, None, None], (T, N, 1)).copy())
names += ["sin_tod", "cos_tod"]

dow = (np.arange(T) // STEPS_PER_DAY) % 7
sin_dow = np.sin(2 * np.pi * dow / 7)
cos_dow = np.cos(2 * np.pi * dow / 7)
feats.append(np.broadcast_to(sin_dow[:, None, None], (T, N, 1)).copy())
feats.append(np.broadcast_to(cos_dow[:, None, None], (T, N, 1)).copy())
names += ["sin_dow", "cos_dow"]

A_in = nx.to_numpy_array(G, nodelist=range(N), weight=None).T
deg_in = A_in.sum(axis=1, keepdims=True)
deg_in[deg_in == 0] = 1.0
A_in_norm = A_in / deg_in
for arr, label in [(flow, "flow"), (occ, "occupancy"), (spd, "speed")]:
    feats.append((arr @ A_in_norm.T)[..., None])
    names.append(f"nbr_in_mean_{label}")

X_full = np.concatenate(feats, axis=-1).astype(np.float32)
X = X_full[:-HORIZON]
y_shifted = y_lab[HORIZON:]
T_eff = X.shape[0]

n_train = int(T_eff * TRAIN_FRACTION)
n_val = int(T_eff * (TRAIN_FRACTION + VAL_FRACTION))
train_idx = np.zeros(T_eff, dtype=bool); train_idx[:n_train] = True
val_idx   = np.zeros(T_eff, dtype=bool); val_idx[n_train:n_val] = True
test_idx  = np.zeros(T_eff, dtype=bool); test_idx[n_val:] = True

y_current = y_lab[: T_eff]
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

print("\n=== Training baselines (test-set metrics, 1-hour-ahead congestion) ===")
X_train = X[train_idx].reshape(-1, X.shape[-1])
y_train = y_shifted[train_idx].reshape(-1)
X_test = X[test_idx].reshape(-1, X.shape[-1])
y_test = y_shifted[test_idx].reshape(-1)

metrics = []

for name in ["majority", "persistence", "logreg"]:
    ts = time.time()
    if name == "majority":
        p = np.zeros_like(y_test, dtype=np.float32)
        out_name = "majority(0)"
    elif name == "persistence":
        p = y_current[test_idx].reshape(-1).astype(np.float32)
        out_name = "persistence"
    else:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1, solver="lbfgs")
        model.fit(X_train_s, y_train)
        p = model.predict_proba(X_test_s)[:, 1]
        out_name = "logreg(balanced)"

    y_pred = (p >= 0.5).astype(np.uint8)
    m = {
        "name": out_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan"),
        "n_positive": int(y_test.sum()),
        "n_total": int(y_test.size),
    }
    print(f"[{name:11s}] fit+eval in {time.time()-ts:5.1f}s  →  acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    metrics.append(m)

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"\nmetrics saved → {METRICS_PATH}")

print(f"\nTotal pipeline time: {time.time()-t0:.1f}s")
