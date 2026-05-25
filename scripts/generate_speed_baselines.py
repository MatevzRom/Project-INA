"""
Generate speed prediction baselines with proper features (no data leakage)
───────────────────────────────────────────────────────────────────────────
Features REMOVED to avoid trivial solutions:
  - current speed
  - speed_lag_1 (5 min ago)
  - speed_lag_3 (15 min ago)
  - speed_lag_12 (1 hour ago - same as prediction horizon)

Features KEPT to learn meaningful patterns:
  - flow, occupancy (current - indirect indicators)
  - speed_lag_288 (same time yesterday)
  - speed_lag_2016 (same time 7 days ago)
  - temporal features (time-of-day, day-of-week)
  - spatial features (neighbor flow/occupancy/speed)

Produces baselines for all three split strategies:
  - {dataset}_baselines_speed_temporal.json
  - {dataset}_baselines_speed_walk_forward.json
  - {dataset}_baselines_speed_sliding_window.json
"""

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def compute_mape(y_true, y_pred, epsilon=1e-3):
    """Mean Absolute Percentage Error"""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def build_features_and_targets(dataset, horizon=12):
    """
    Build feature matrix and speed targets from raw data.

    Returns:
        X: (T_eff, N, F) features
        y_speed: (T_eff, N) speed targets (horizon-ahead)
        y_current: (T_eff, N) current speed
        G: networkx graph
        feature_names: list of feature names
    """
    print(f"\n=== Loading {dataset.upper()} ===")

    # Load raw data
    if dataset == "pems04":
        data = np.load("data/PEMS04/PEMS04.npz")["data"].astype(np.float32)
        edges = pd.read_csv("data/PEMS04/PEMS04.csv")
        start_date = "2018-01-01 00:00:00"
    elif dataset == "pems08":
        data = np.load("data/PEMS08/PEMS08.npz")["data"].astype(np.float32)
        edges = pd.read_csv("data/PEMS08/PEMS08.csv")
        start_date = "2016-07-01 00:00:00"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    T, N, C = data.shape
    timestamps = pd.date_range(start_date, periods=T, freq="5min")
    flow = data[:, :, 0]
    occ = data[:, :, 1]
    spd = data[:, :, 2]

    print(f"data: T={T}  N={N}  C={C}")
    print(f"time range: {timestamps[0]} → {timestamps[-1]}")

    # Build graph for spatial features
    print("\n=== Building graph ===")
    G = nx.DiGraph()
    for n in range(N):
        G.add_node(n, index=int(n))

    for _, row in edges.iterrows():
        G.add_edge(int(row["from"]), int(row["to"]))

    print(f"graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Build features (NO DATA LEAKAGE)
    print("\n=== Building features (no recent speed lags) ===")
    STEPS_PER_DAY = 288
    feats = []
    names = []

    # 1. Current flow and occupancy (NOT speed!)
    feats.append(flow[..., None])
    feats.append(occ[..., None])
    names += ["flow", "occupancy"]

    # 2. Pattern-based speed lags ONLY (no recent lags!)
    for lag in [STEPS_PER_DAY, 7 * STEPS_PER_DAY]:
        lagged = np.roll(spd, lag, axis=0)
        lagged[:lag] = spd[:lag]  # fill early timesteps
        feats.append(lagged[..., None])
        names.append(f"speed_lag_{lag}")

    print(f"  ✓ Using speed_lag_288 (1 day ago) and speed_lag_2016 (7 days ago)")
    print(f"  ✗ Removed: current speed, lag_1, lag_3, lag_12 (avoid data leakage)")

    # 3. Temporal features
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

    # 4. Spatial features (neighbor averages)
    A_in = nx.to_numpy_array(G, nodelist=range(N), weight=None).T
    deg_in = A_in.sum(axis=1, keepdims=True)
    deg_in[deg_in == 0] = 1.0
    A_in_norm = A_in / deg_in

    for arr, label in [(flow, "flow"), (occ, "occupancy"), (spd, "speed")]:
        feats.append((arr @ A_in_norm.T)[..., None])
        names.append(f"nbr_in_mean_{label}")

    # Concatenate all features
    X_full = np.concatenate(feats, axis=-1).astype(np.float32)
    X = X_full[:-horizon]
    y_speed = spd[horizon:]
    y_current = spd[:-horizon]
    T_eff = X.shape[0]

    print(f"features: X={X.shape}  (F={len(names)} features)")
    print(f"feature names: {names}")

    return X, y_speed, y_current, G, names


def temporal_split(X, y_speed, y_current, train_frac=0.6, val_frac=0.2):
    """Single temporal train/val/test split"""
    T_eff = X.shape[0]
    n_train = int(T_eff * train_frac)
    n_val = int(T_eff * (train_frac + val_frac))

    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_val)
    test_idx = np.arange(n_val, T_eff)

    return [(X, y_speed, y_current, train_idx, val_idx, test_idx)]


def walk_forward_split(X, y_speed, y_current, n_folds=5, val_frac=0.1, test_frac=0.1, window=12):
    """Walk-forward expanding window split"""
    T_eff = X.shape[0]
    val_size = int(T_eff * val_frac)
    test_size = int(T_eff * test_frac)
    block = val_size + test_size + 2 * window

    print(f"  walk_forward: n_folds={n_folds}, val_size={val_size}, test_size={test_size}, gap={window}")

    folds = []
    for k in range(n_folds):
        test_end = T_eff - k * block
        test_start = test_end - test_size
        val_end = test_start - window
        val_start = val_end - val_size
        train_end = val_start - window

        if train_end < window:
            print(f"  fold {k+1}: insufficient data, stopping")
            break

        train_idx = np.arange(window, train_end)
        val_idx = np.arange(val_start, val_end)
        test_idx = np.arange(test_start, test_end)

        print(f"  fold {k+1}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        folds.append((X, y_speed, y_current, train_idx, val_idx, test_idx))

    return folds


def sliding_window_split(X, y_speed, y_current, n_folds=5, train_frac=0.6,
                         val_frac=0.1, test_frac=0.1, window=12):
    """Sliding window split"""
    T_eff = X.shape[0]
    train_size = int(T_eff * train_frac)
    val_size = int(T_eff * val_frac)
    test_size = int(T_eff * test_frac)
    block = train_size + val_size + test_size + 2 * window

    print(f"  sliding_window: n_folds={n_folds}, train={train_size}, val={val_size}, test={test_size}, gap={window}")

    folds = []
    for k in range(n_folds):
        offset = k * (val_size + test_size + 2 * window)
        train_start = offset
        train_end = train_start + train_size
        val_start = train_end + window
        val_end = val_start + val_size
        test_start = val_end + window
        test_end = test_start + test_size

        if test_end > T_eff:
            print(f"  fold {k+1}: exceeds T_eff, stopping")
            break

        train_idx = np.arange(train_start + window, train_end)
        val_idx = np.arange(val_start, val_end)
        test_idx = np.arange(test_start, test_end)

        print(f"  fold {k+1}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        folds.append((X, y_speed, y_current, train_idx, val_idx, test_idx))

    return folds


def train_baselines(X_train, y_train, X_test, y_test, y_current_test):
    """Train three baseline models and evaluate on test set"""
    metrics = []

    for name in ["mean", "persistence", "linreg"]:
        t_start = time.time()

        if name == "mean":
            pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=np.float32)
            model_name = "mean_baseline"
        elif name == "persistence":
            pred = y_current_test.astype(np.float32)
            model_name = "persistence"
        else:  # linreg
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s).astype(np.float32)
            model_name = "linear_regression"

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        mape = compute_mape(y_test, pred)

        m = {
            "name": model_name,
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "n_total": int(y_test.size),
        }

        print(f"  [{name:12s}] {time.time()-t_start:5.1f}s → "
              f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

        metrics.append(m)

    return metrics


def average_metrics(all_fold_metrics):
    """Average metrics across folds"""
    n_models = len(all_fold_metrics[0])
    n_folds = len(all_fold_metrics)

    avg_metrics = []
    for model_idx in range(n_models):
        model_name = all_fold_metrics[0][model_idx]["name"]

        mae_vals = [fold[model_idx]["mae"] for fold in all_fold_metrics]
        rmse_vals = [fold[model_idx]["rmse"] for fold in all_fold_metrics]
        r2_vals = [fold[model_idx]["r2"] for fold in all_fold_metrics]
        mape_vals = [fold[model_idx]["mape"] for fold in all_fold_metrics]

        avg_m = {
            "name": model_name,
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "r2_mean": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)),
            "mape_mean": float(np.mean(mape_vals)),
            "mape_std": float(np.std(mape_vals)),
            "n_folds": n_folds,
        }

        print(f"  [{model_name:20s}] MAE={avg_m['mae_mean']:.4f}±{avg_m['mae_std']:.4f}  "
              f"RMSE={avg_m['rmse_mean']:.4f}±{avg_m['rmse_std']:.4f}  "
              f"R²={avg_m['r2_mean']:.4f}±{avg_m['r2_std']:.4f}")

        avg_metrics.append(avg_m)

    return avg_metrics


def generate_baselines(dataset, split_mode, **split_kwargs):
    """Generate baselines for one dataset and one split mode"""
    print(f"\n{'='*70}")
    print(f"{dataset.upper()} - {split_mode.upper()}")
    print(f"{'='*70}")

    t0 = time.time()

    # Build features
    X, y_speed, y_current, G, feat_names = build_features_and_targets(dataset)

    # Get folds
    print(f"\n=== Creating {split_mode} splits ===")
    if split_mode == "temporal":
        folds = temporal_split(X, y_speed, y_current, **split_kwargs)
    elif split_mode == "walk_forward":
        folds = walk_forward_split(X, y_speed, y_current, **split_kwargs)
    elif split_mode == "sliding_window":
        folds = sliding_window_split(X, y_speed, y_current, **split_kwargs)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    # Train baselines for each fold
    print(f"\n=== Training baselines ===")
    all_fold_metrics = []

    for fold_idx, (X_fold, y_fold, y_curr_fold, train_idx, val_idx, test_idx) in enumerate(folds, 1):
        if len(folds) > 1:
            print(f"\n--- Fold {fold_idx}/{len(folds)} ---")

        # Flatten (timesteps, nodes) into samples
        X_train = X_fold[train_idx].reshape(-1, X_fold.shape[-1])
        y_train = y_fold[train_idx].reshape(-1)
        X_test = X_fold[test_idx].reshape(-1, X_fold.shape[-1])
        y_test = y_fold[test_idx].reshape(-1)
        y_current_test = y_curr_fold[test_idx].reshape(-1)

        fold_metrics = train_baselines(X_train, y_train, X_test, y_test, y_current_test)
        all_fold_metrics.append(fold_metrics)

    # Average across folds if needed
    if len(folds) == 1:
        final_metrics = all_fold_metrics[0]
    else:
        print(f"\n=== Averaging across {len(folds)} folds ===")
        final_metrics = average_metrics(all_fold_metrics)

    # Save results
    output_path = Path("reports") / f"{dataset}_baselines_speed_{split_mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_metrics, indent=2))

    print(f"\n✓ Saved → {output_path}")
    print(f"✓ Total time: {time.time()-t0:.1f}s")

    return final_metrics


if __name__ == "__main__":
    HORIZON = 12
    WINDOW = 12

    for dataset in ["pems08", "pems04"]:
        # Temporal
        generate_baselines(
            dataset, "temporal",
            train_frac=0.6, val_frac=0.2
        )

        # Walk-forward
        generate_baselines(
            dataset, "walk_forward",
            n_folds=5, val_frac=0.1, test_frac=0.1, window=WINDOW
        )

        # Sliding window
        generate_baselines(
            dataset, "sliding_window",
            n_folds=5, train_frac=0.6, val_frac=0.1, test_frac=0.1, window=WINDOW
        )

    print("\n" + "="*70)
    print("ALL BASELINES GENERATED SUCCESSFULLY!")
    print("="*70)
