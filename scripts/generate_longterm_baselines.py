"""
Generate LONG-TERM speed prediction baselines (6-12 hours ahead)
───────────────────────────────────────────────────────────────────────────
Scenario: Long-term spatiotemporal forecasting for traffic planning

Features for REAL forecasting (no current measurements):
  ✓ Multiple historical speed lags (1-7 days ago)
  ✓ Temporal features (time-of-day, day-of-week)
  ✗ NO current flow/occupancy (not available for future dates!)
  ✗ NO neighbor current measurements

Prediction horizon: 72 timesteps ahead (6 hours)
Input window: 576 timesteps (48 hours of history)

Use case: "Given a future date/time, predict traffic speed"
Example: "What will traffic be like tomorrow at 8am?"
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


def build_longterm_features(dataset, horizon=72):
    """
    Build features for LONG-TERM forecasting (6-12 hours ahead).

    Features ONLY include historical patterns + time features.
    NO current measurements (not available for future dates).

    Returns:
        X: (T_eff, N, F) features
        y_speed: (T_eff, N) speed targets (horizon-ahead)
        G: networkx graph
        feature_names: list of feature names
    """
    print(f"\n=== Loading {dataset.upper()} (Long-term Forecasting) ===")

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
    spd = data[:, :, 2]

    print(f"data: T={T}  N={N}  C={C}")
    print(f"time range: {timestamps[0]} → {timestamps[-1]}")
    print(f"prediction horizon: {horizon} steps ({horizon*5/60:.1f} hours ahead)")

    # Build graph for topology
    print("\n=== Building graph ===")
    G = nx.DiGraph()
    for n in range(N):
        G.add_node(n, index=int(n))

    for _, row in edges.iterrows():
        G.add_edge(int(row["from"]), int(row["to"]))

    print(f"graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Build features for LONG-TERM forecasting
    print("\n=== Building long-term forecasting features ===")
    STEPS_PER_DAY = 288
    feats = []
    names = []

    # 1. Historical speed lags (multiple time scales)
    # Use: 6h, 12h, 24h, 48h, 7days ago
    lags = [36, 72, 144, 288, 576, 7*288]  # 3h, 6h, 12h, 24h, 48h, 7d
    for lag in lags:
        lagged = np.roll(spd, lag, axis=0)
        lagged[:lag] = spd[:lag]  # fill early timesteps
        feats.append(lagged[..., None])
        hours = lag * 5 / 60
        if hours >= 24:
            names.append(f"speed_lag_{lag}_{int(hours/24)}d")
        else:
            names.append(f"speed_lag_{lag}_{int(hours)}h")

    print(f"  ✓ Historical speed lags: {len(lags)} features")
    print(f"    {', '.join([f'{lag*5/60:.0f}h' if lag*5/60 < 24 else f'{lag*5/60/24:.0f}d' for lag in lags])}")

    # 2. Temporal features (for TARGET time, not current time)
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

    print(f"  ✓ Temporal features: 4 (sin/cos tod, sin/cos dow)")
    print(f"  ✗ Removed: flow, occupancy, neighbor features (not available for future!)")

    # Concatenate all features
    X_full = np.concatenate(feats, axis=-1).astype(np.float32)
    X = X_full[:-horizon]
    y_speed = spd[horizon:]
    T_eff = X.shape[0]

    print(f"features: X={X.shape}  (F={len(names)} features)")
    print(f"feature names: {names}")

    return X, y_speed, G, names


def temporal_split(X, y_speed, train_frac=0.6, val_frac=0.2):
    """Single temporal train/val/test split"""
    T_eff = X.shape[0]
    n_train = int(T_eff * train_frac)
    n_val = int(T_eff * (train_frac + val_frac))

    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_val)
    test_idx = np.arange(n_val, T_eff)

    print(f"  temporal: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return [(X, y_speed, train_idx, val_idx, test_idx)]


def walk_forward_split(X, y_speed, window=576, n_folds=4):
    """Walk-forward cross-validation with longer window"""
    T_eff = X.shape[0]
    total_available = T_eff - window

    fold_size = total_available // (n_folds + 2)  # +2 for val and test
    val_size = fold_size
    test_size = fold_size
    train_size = total_available - (n_folds * (val_size + test_size + 2 * window))

    print(f"  walk_forward: n_folds={n_folds}, val_size={val_size}, test_size={test_size}, gap={window}")

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
            print(f"  fold {k+1}: insufficient data, stopping")
            break

        train_idx = np.arange(train_start + window, train_end)
        val_idx = np.arange(val_start, val_end)
        test_idx = np.arange(test_start, test_end)

        print(f"  fold {k+1}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        folds.append((X, y_speed, train_idx, val_idx, test_idx))

    return folds


def train_baselines(X_train, y_train, X_test, y_test):
    """Train baseline models for long-term forecasting"""
    metrics = []

    # Flatten data for sklearn
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    y_train_flat = y_train.ravel()
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    y_test_flat = y_test.ravel()

    for name in ["mean", "last_week", "linreg"]:
        t_start = time.time()

        if name == "mean":
            pred = np.full_like(y_test_flat, fill_value=y_train_flat.mean(), dtype=np.float32)
            model_name = "mean_baseline"
        elif name == "last_week":
            # Use speed from 7 days ago (last feature is speed_lag_7d)
            pred = X_test_flat[:, -5].astype(np.float32)  # speed_lag_2016_7d
            model_name = "last_week_persistence"
        else:  # linreg
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_flat)
            X_test_s = scaler.transform(X_test_flat)
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train_s, y_train_flat)
            pred = model.predict(X_test_s).astype(np.float32)
            model_name = "linear_regression"

        mae = mean_absolute_error(y_test_flat, pred)
        mse = mean_squared_error(y_test_flat, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_flat, pred)
        mape = compute_mape(y_test_flat, pred)

        m = {
            "name": model_name,
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "n_total": int(y_test_flat.size),
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

        print(f"  [{model_name:25s}] MAE={avg_m['mae_mean']:.4f}±{avg_m['mae_std']:.4f}  "
              f"RMSE={avg_m['rmse_mean']:.4f}±{avg_m['rmse_std']:.4f}  "
              f"R²={avg_m['r2_mean']:.4f}±{avg_m['r2_std']:.4f}")

        avg_metrics.append(avg_m)

    return avg_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pems08", choices=["pems04", "pems08"])
    parser.add_argument("--split", type=str, default="temporal", choices=["temporal", "walk_forward"])
    parser.add_argument("--horizon", type=int, default=72, help="Prediction horizon in timesteps (72=6h)")
    parser.add_argument("--window", type=int, default=576, help="Input window for walk_forward (576=48h)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"LONG-TERM SPEED FORECASTING - {args.dataset.upper()} - {args.split.upper()}")
    print(f"{'='*70}")

    # Build features
    X, y_speed, G, feat_names = build_longterm_features(args.dataset, horizon=args.horizon)

    # Get splits
    print(f"\n=== Creating {args.split} splits ===")
    if args.split == "temporal":
        folds = temporal_split(X, y_speed)
    else:
        folds = walk_forward_split(X, y_speed, window=args.window)

    # Train baselines
    all_fold_metrics = []
    for fold_idx, (X_fold, y_fold, train_idx, val_idx, test_idx) in enumerate(folds, 1):
        if len(folds) > 1:
            print(f"\n{'='*70}")
            print(f"Fold {fold_idx}/{len(folds)}")
            print(f"{'='*70}")

        X_train, X_test = X_fold[train_idx], X_fold[test_idx]
        y_train, y_test = y_fold[train_idx], y_fold[test_idx]

        fold_metrics = train_baselines(X_train, y_train, X_test, y_test)
        all_fold_metrics.append(fold_metrics)

    # Average and save
    if len(folds) > 1:
        print(f"\n{'='*70}")
        print(f"Average across {len(folds)} folds:")
        print(f"{'='*70}")
        final_metrics = average_metrics(all_fold_metrics)
    else:
        final_metrics = all_fold_metrics[0]

    output_path = Path("reports") / f"{args.dataset}_longterm_baselines_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_metrics, indent=2))

    print(f"\n✓ Saved → {output_path}")
    print(f"✓ Total time: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
