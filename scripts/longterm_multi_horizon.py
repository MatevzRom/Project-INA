"""
Long-term multi-horizon speed forecasting for PEMS traffic datasets.

The script builds leakage-safe features for each current time t, predicts future
speed at several horizons, and can run either simple baselines or a shared GRU
neural model. All reported metrics are computed after converting predictions
back to mph.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STEPS_PER_DAY = 288
DATASETS = {
    "pems04": {
        "npz": Path("data/PEMS04/PEMS04.npz"),
        "csv": Path("data/PEMS04/PEMS04.csv"),
        "start": "2018-01-01 00:00:00",
    },
    "pems08": {
        "npz": Path("data/PEMS08/PEMS08.npz"),
        "csv": Path("data/PEMS08/PEMS08.csv"),
        "start": "2016-07-01 00:00:00",
    },
}


@dataclass
class MultiHorizonData:
    """Container for the built graph-time dataset and its chronological splits."""

    dataset: str
    X: np.ndarray
    y: np.ndarray
    current_times: np.ndarray
    timestamps: pd.DatetimeIndex
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: list[str]
    horizon_steps: list[int]
    horizon_names: list[str]
    speed_lags: list[int]
    target_feature_names: list[list[str]]
    raw_shape: tuple[int, int, int]


def parse_steps(value: str) -> list[int]:
    """Parse comma-separated 5-minute step counts from a CLI argument."""

    steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not steps:
        raise ValueError("At least one step value is required.")
    if any(step <= 0 for step in steps):
        raise ValueError("Step values must be positive.")
    return steps


def step_name(steps: int) -> str:
    """Convert a number of 5-minute steps into a readable horizon name."""

    minutes = steps * 5
    if minutes % (24 * 60) == 0:
        return f"{minutes // (24 * 60)}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def load_raw(dataset: str) -> tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    """Load raw PEMS measurements, graph CSV, and inferred timestamps."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}.")

    cfg = DATASETS[dataset]
    if not cfg["npz"].exists():
        raise FileNotFoundError(f"{cfg['npz']} not found.")
    if not cfg["csv"].exists():
        raise FileNotFoundError(f"{cfg['csv']} not found.")

    data = np.load(cfg["npz"])["data"].astype(np.float32)
    edges = pd.read_csv(cfg["csv"])
    timestamps = pd.date_range(cfg["start"], periods=data.shape[0], freq="5min")
    return data, edges, timestamps


def make_time_features(indices: np.ndarray, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    """Create cyclic time-of-day and day-of-week features for timestep indices."""

    tod = indices % STEPS_PER_DAY
    dow = (indices // STEPS_PER_DAY) % 7
    arrays = [
        np.sin(2 * np.pi * tod / STEPS_PER_DAY).astype(np.float32),
        np.cos(2 * np.pi * tod / STEPS_PER_DAY).astype(np.float32),
        np.sin(2 * np.pi * dow / 7).astype(np.float32),
        np.cos(2 * np.pi * dow / 7).astype(np.float32),
    ]
    names = [
        f"{prefix}_sin_tod",
        f"{prefix}_cos_tod",
        f"{prefix}_sin_dow",
        f"{prefix}_cos_dow",
    ]
    return arrays, names


def broadcast_time_feature(values: np.ndarray, n_nodes: int) -> np.ndarray:
    """Repeat a global time feature so every sensor receives the same value."""

    return np.broadcast_to(values[:, None, None], (len(values), n_nodes, 1)).copy()


def build_multi_horizon_data(
    dataset: str = "pems08",
    horizon_steps: list[int] | None = None,
    speed_lags: list[int] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> MultiHorizonData:
    """Build feature and target tensors for long-term speed forecasting.

    X has shape [time, sensors, features]. y has shape [time, sensors, horizons].
    Measured traffic features only use timestamps <= current time t; target-time
    features are limited to calendar information that would be known in advance.
    """

    if horizon_steps is None:
        horizon_steps = [12, 36, 72, 144]
    if speed_lags is None:
        speed_lags = [36, 72, 144, 288, 576, 2016]

    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must satisfy train_frac > 0, val_frac > 0, train_frac + val_frac < 1.")

    data, _edges, timestamps = load_raw(dataset)
    T, N, C = data.shape
    if C < 3:
        raise ValueError(f"{dataset} must contain flow, occupancy, speed channels; got C={C}.")

    max_lag = max(speed_lags)
    max_horizon = max(horizon_steps)
    if T <= max_lag + max_horizon:
        raise ValueError(
            f"Not enough timesteps T={T} for max_lag={max_lag} and max_horizon={max_horizon}."
        )

    current_times = np.arange(max_lag, T - max_horizon)
    flow = data[:, :, 0]
    occupancy = data[:, :, 1]
    speed = data[:, :, 2]

    # Current measurements are legal because they are observed at prediction time t.
    feature_arrays = [
        flow[current_times, :, None],
        occupancy[current_times, :, None],
        speed[current_times, :, None],
    ]
    feature_names = ["flow_t", "occupancy_t", "speed_t"]

    measured_feature_indices: list[np.ndarray] = [
        current_times,
        current_times,
        current_times,
    ]

    # Historical speed lags give the model daily/weekly context without looking ahead.
    for lag in speed_lags:
        source_times = current_times - lag
        feature_arrays.append(speed[source_times, :, None])
        feature_names.append(f"speed_lag_{lag}_{step_name(lag)}")
        measured_feature_indices.append(source_times)

    current_time_arrays, current_time_names = make_time_features(current_times, "current")
    for values, name in zip(current_time_arrays, current_time_names):
        feature_arrays.append(broadcast_time_feature(values, N))
        feature_names.append(name)

    # Future calendar features are allowed because the forecast timestamp is known.
    target_feature_names: list[list[str]] = []
    for horizon in horizon_steps:
        target_times = current_times + horizon
        target_arrays, names = make_time_features(target_times, f"target_{step_name(horizon)}")
        target_feature_names.append(names)
        for values, name in zip(target_arrays, names):
            feature_arrays.append(broadcast_time_feature(values, N))
            feature_names.append(name)

    X = np.concatenate(feature_arrays, axis=-1).astype(np.float32)
    y = np.stack([speed[current_times + horizon] for horizon in horizon_steps], axis=-1).astype(np.float32)

    # Fail early if any feature accidentally uses future traffic measurements.
    validate_no_leakage(current_times, measured_feature_indices)
    validate_targets(current_times, horizon_steps, y, speed)

    # Chronological split matches deployment: train on the past, test on the future.
    T_eff = len(current_times)
    train_end = int(T_eff * train_frac)
    val_end = int(T_eff * (train_frac + val_frac))
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, T_eff)

    return MultiHorizonData(
        dataset=dataset,
        X=X,
        y=y,
        current_times=current_times,
        timestamps=timestamps,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_names=feature_names,
        horizon_steps=horizon_steps,
        horizon_names=[step_name(step) for step in horizon_steps],
        speed_lags=speed_lags,
        target_feature_names=target_feature_names,
        raw_shape=(T, N, C),
    )


def validate_no_leakage(current_times: np.ndarray, measured_feature_indices: list[np.ndarray]) -> None:
    """Ensure measured traffic features never come from after the current time."""

    for source_times in measured_feature_indices:
        if np.any(source_times > current_times):
            raise AssertionError("Measured feature uses a future timestamp.")
        if np.any(source_times < 0):
            raise AssertionError("Measured feature uses a negative timestamp.")


def validate_targets(
    current_times: np.ndarray,
    horizon_steps: list[int],
    y: np.ndarray,
    speed: np.ndarray,
) -> None:
    """Check that each target slice equals the raw speed at t + horizon."""

    for h_idx, horizon in enumerate(horizon_steps):
        expected = speed[current_times + horizon]
        if not np.array_equal(y[:, :, h_idx], expected):
            raise AssertionError(f"Target mismatch for horizon {horizon}.")


def print_summary(data: MultiHorizonData) -> None:
    """Print dataset shapes, split ranges, target statistics, and leakage status."""

    T, N, C = data.raw_shape
    print(f"\n=== {data.dataset.upper()} Long-Term Multi-Horizon Data ===")
    print(f"raw: T={T:,}  N={N:,}  C={C}")
    print(f"raw time range: {data.timestamps[0]} -> {data.timestamps[-1]}")
    print(f"effective current times: {len(data.current_times):,}")
    print(f"X: {data.X.shape}  y: {data.y.shape}")
    print(f"horizons: {list(zip(data.horizon_names, data.horizon_steps))}")
    print(f"speed lags: {[(step_name(lag), lag) for lag in data.speed_lags]}")
    print(f"features ({len(data.feature_names)}):")
    for i, name in enumerate(data.feature_names):
        print(f"  {i:02d}: {name}")

    print("\nSplit ranges:")
    max_horizon = max(data.horizon_steps)
    for split_name, idx in [
        ("train", data.train_idx),
        ("val", data.val_idx),
        ("test", data.test_idx),
    ]:
        current = data.current_times[idx]
        first_t = data.timestamps[current[0]]
        last_t = data.timestamps[current[-1]]
        last_target = data.timestamps[current[-1] + max_horizon]
        print(
            f"  {split_name:<5} snapshots={len(idx):,}  "
            f"sensor_examples/horizon={len(idx) * data.X.shape[1]:,}  "
            f"current={first_t} -> {last_t}  "
            f"last_target={last_target}"
        )

    print("\nTarget speed summary by split and horizon:")
    for split_name, idx in [
        ("train", data.train_idx),
        ("val", data.val_idx),
        ("test", data.test_idx),
    ]:
        print(f"  {split_name}:")
        values = data.y[idx]
        for h_idx, h_name in enumerate(data.horizon_names):
            y_h = values[:, :, h_idx]
            print(
                f"    {h_name:>4}: mean={y_h.mean():.3f}  std={y_h.std():.3f}  "
                f"min={y_h.min():.3f}  p05={np.percentile(y_h, 5):.3f}  "
                f"median={np.median(y_h):.3f}  p95={np.percentile(y_h, 95):.3f}  max={y_h.max():.3f}"
            )

    print("\nLeakage checks: OK")


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    """Compute MAPE while avoiding division by zero for very low speeds."""

    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_names: list[str],
) -> dict:
    """Compute regression metrics separately for each forecast horizon."""

    per_horizon = []
    for h_idx, horizon_name in enumerate(horizon_names):
        true_h = y_true[:, :, h_idx].reshape(-1)
        pred_h = y_pred[:, :, h_idx].reshape(-1)
        mse = mean_squared_error(true_h, pred_h)
        per_horizon.append({
            "horizon": horizon_name,
            "mae": float(mean_absolute_error(true_h, pred_h)),
            "rmse": float(np.sqrt(mse)),
            "mape": compute_mape(true_h, pred_h),
            "r2": float(r2_score(true_h, pred_h)),
            "n_total": int(true_h.size),
        })

    return {
        "per_horizon": per_horizon,
        "mae_mean": float(np.mean([m["mae"] for m in per_horizon])),
        "rmse_mean": float(np.mean([m["rmse"] for m in per_horizon])),
        "mape_mean": float(np.mean([m["mape"] for m in per_horizon])),
        "r2_mean": float(np.mean([m["r2"] for m in per_horizon])),
    }


def repeated_feature_prediction(data: MultiHorizonData, feature_name: str, split_idx: np.ndarray) -> np.ndarray:
    """Use one feature column as a naive prediction for every target horizon."""

    feature_idx = data.feature_names.index(feature_name)
    values = data.X[split_idx, :, feature_idx]
    return np.repeat(values[:, :, None], len(data.horizon_names), axis=-1)


def run_baselines(
    data: MultiHorizonData,
    outdir: Path,
    ridge_alpha: float = 1.0,
) -> dict:
    """Evaluate simple non-neural baselines on the test split."""

    outdir.mkdir(parents=True, exist_ok=True)

    y_train = data.y[data.train_idx]
    y_test = data.y[data.test_idx]
    models = []

    train_mean = y_train.reshape(-1, len(data.horizon_names)).mean(axis=0)
    mean_pred = np.broadcast_to(train_mean[None, None, :], y_test.shape).copy()
    models.append({
        "name": "mean_train_global",
        **compute_regression_metrics(y_test, mean_pred, data.horizon_names),
    })

    current_pred = repeated_feature_prediction(data, "speed_t", data.test_idx)
    models.append({
        "name": "current_speed",
        **compute_regression_metrics(y_test, current_pred, data.horizon_names),
    })

    # Lag baselines test whether a single historical speed pattern is enough.
    for lag in data.speed_lags:
        feature_name = f"speed_lag_{lag}_{step_name(lag)}"
        lag_pred = repeated_feature_prediction(data, feature_name, data.test_idx)
        models.append({
            "name": feature_name,
            **compute_regression_metrics(y_test, lag_pred, data.horizon_names),
        })

    # Ridge uses all engineered features while staying interpretable and fast.
    train_X = data.X[data.train_idx].reshape(-1, data.X.shape[-1])
    train_y = data.y[data.train_idx].reshape(-1, len(data.horizon_names))
    test_X = data.X[data.test_idx].reshape(-1, data.X.shape[-1])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    ridge.fit(train_X, train_y)
    ridge_pred_flat = ridge.predict(test_X).astype(np.float32)
    ridge_pred = ridge_pred_flat.reshape(len(data.test_idx), data.X.shape[1], len(data.horizon_names))
    models.append({
        "name": f"ridge_alpha_{ridge_alpha:g}",
        **compute_regression_metrics(y_test, ridge_pred, data.horizon_names),
    })

    report = {
        "dataset": data.dataset,
        "task": "long_term_multi_horizon_speed_forecasting",
        "horizon_steps": data.horizon_steps,
        "horizon_names": data.horizon_names,
        "feature_names": data.feature_names,
        "split": {
            "train_snapshots": int(len(data.train_idx)),
            "val_snapshots": int(len(data.val_idx)),
            "test_snapshots": int(len(data.test_idx)),
            "train_start": str(data.timestamps[data.current_times[data.train_idx[0]]]),
            "train_end": str(data.timestamps[data.current_times[data.train_idx[-1]]]),
            "val_start": str(data.timestamps[data.current_times[data.val_idx[0]]]),
            "val_end": str(data.timestamps[data.current_times[data.val_idx[-1]]]),
            "test_start": str(data.timestamps[data.current_times[data.test_idx[0]]]),
            "test_end": str(data.timestamps[data.current_times[data.test_idx[-1]]]),
        },
        "models": models,
    }

    print("\n=== Baseline Test Metrics ===")
    print(f"{'model':<24} {'avg_MAE':>8} {'1h':>8} {'3h':>8} {'6h':>8} {'12h':>8}")
    print("-" * 72)
    for model in models:
        maes = {m["horizon"]: m["mae"] for m in model["per_horizon"]}
        print(
            f"{model['name']:<24} {model['mae_mean']:>8.4f} "
            f"{maes.get('1h', float('nan')):>8.4f} "
            f"{maes.get('3h', float('nan')):>8.4f} "
            f"{maes.get('6h', float('nan')):>8.4f} "
            f"{maes.get('12h', float('nan')):>8.4f}"
        )

    output_path = outdir / f"{data.dataset}_longterm_multi_horizon_baselines.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nBaseline report saved -> {output_path}")

    return report


def import_torch():
    """Import PyTorch lazily so baseline/summary modes do not require it."""

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch could not be imported. If libcudnn.so.9 is missing, run with LD_LIBRARY_PATH "
            "pointing to the nvidia/*/lib folders installed with torch."
        ) from exc
    return torch, nn, DataLoader, Dataset


def set_seed(seed: int, torch) -> None:
    """Set random seeds for reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(torch, device_choice: str):
    """Resolve the requested training device, with CUDA used in auto mode if present."""

    if device_choice == "cpu":
        return torch.device("cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_choice != "auto":
        raise ValueError("device_choice must be one of: auto, cpu, cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalise_train_only(data: MultiHorizonData) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalise features and targets using training split statistics only."""

    x_train = data.X[data.train_idx]
    y_train = data.y[data.train_idx]

    x_mean = x_train.reshape(-1, data.X.shape[-1]).mean(axis=0).astype(np.float32)
    x_std = x_train.reshape(-1, data.X.shape[-1]).std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0

    y_mean = y_train.reshape(-1, len(data.horizon_names)).mean(axis=0).astype(np.float32)
    y_std = y_train.reshape(-1, len(data.horizon_names)).std(axis=0).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0

    X_norm = ((data.X - x_mean[None, None, :]) / x_std[None, None, :]).astype(np.float32)
    y_norm = ((data.y - y_mean[None, None, :]) / y_std[None, None, :]).astype(np.float32)

    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    return X_norm, y_norm, stats


def make_window_dataset_class(torch, Dataset):
    class MultiHorizonWindowDataset(Dataset):
        """Return rolling input windows ending at each target timestamp."""

        def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, window: int):
            if window <= 0:
                raise ValueError("window must be positive.")
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            self.indices = indices[indices >= window - 1]
            self.window = window

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, item: int):
            t = int(self.indices[item])
            # The label is at t; the input window contains only times up to t.
            return self.X[t - self.window + 1 : t + 1], self.y[t]

    return MultiHorizonWindowDataset


def make_shared_sensor_gru_class(torch, nn):
    class SharedSensorGRU(nn.Module):
        """
        Shared-weight GRU used independently for every sensor.

        Input:  [B, W, N, F]
        Output: [B, N, H]
        """
        def __init__(
            self,
            in_features: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            n_horizons: int,
        ):
            super().__init__()
            self.gru = nn.GRU(
                in_features,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim, n_horizons)

        def forward(self, x):
            B, W, N, F = x.shape
            # Treat sensors as extra batch items so one GRU is shared by all nodes.
            x = x.permute(0, 2, 1, 3).reshape(B * N, W, F)
            out, _ = self.gru(x)
            out = self.dropout(out[:, -1, :])
            return self.head(out).reshape(B, N, -1)

    return SharedSensorGRU


def evaluate_neural_model(model, loader, device, y_mean: np.ndarray, y_std: np.ndarray, horizon_names: list[str]) -> dict:
    """Evaluate the neural model and convert normalised outputs back to mph."""

    import torch

    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            preds.append(pred)
            targets.append(y_batch.numpy())

    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(targets, axis=0)
    pred = pred_norm * y_std[None, None, :] + y_mean[None, None, :]
    true = true_norm * y_std[None, None, :] + y_mean[None, None, :]
    return compute_regression_metrics(true, pred, horizon_names)


def run_neural_model(
    data: MultiHorizonData,
    outdir: Path,
    checkpoint_dir: Path,
    run_name: str,
    window: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    device_choice: str,
) -> dict:
    """Train the shared-sensor GRU and save a JSON report plus checkpoint."""

    torch, nn, DataLoader, Dataset = import_torch()
    set_seed(seed, torch)

    device = choose_device(torch, device_choice)
    X_norm, y_norm, stats = normalise_train_only(data)
    y_mean = np.array(stats["y_mean"], dtype=np.float32)
    y_std = np.array(stats["y_std"], dtype=np.float32)

    WindowDataset = make_window_dataset_class(torch, Dataset)
    train_ds = WindowDataset(X_norm, y_norm, data.train_idx, window)
    val_ds = WindowDataset(X_norm, y_norm, data.val_idx, window)
    test_ds = WindowDataset(X_norm, y_norm, data.test_idx, window)

    loader_kwargs = {"pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    SharedSensorGRU = make_shared_sensor_gru_class(torch, nn)
    model = SharedSensorGRU(
        in_features=data.X.shape[-1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        n_horizons=len(data.horizon_names),
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", patience=2, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Neural Model Training ===", flush=True)
    print(
        f"device={device}  cuda_available={torch.cuda.is_available()}  params={n_params:,}",
        flush=True,
    )
    print(f"window={window}  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}", flush=True)

    best_state = None
    best_val_mae = float("inf")
    patience_ctr = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        try:
            # Optimise on normalised targets; evaluation below is reported in mph.
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimiser.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item() * x_batch.size(0)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                print(
                    "\nCUDA ran out of memory. Try a smaller --batch-size, a smaller --window, "
                    "or --device cpu. For this 4GB GPU, good next attempts are "
                    "--batch-size 32 or --batch-size 16.",
                    flush=True,
                )
                raise
            raise
        train_loss /= len(train_loader.dataset)

        val_metrics = evaluate_neural_model(model, val_loader, device, y_mean, y_std, data.horizon_names)
        val_mae = val_metrics["mae_mean"]
        scheduler.step(val_mae)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_mae_mean": float(val_mae),
            "val_rmse_mean": float(val_metrics["rmse_mean"]),
        })

        print(
            f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_MAE={val_mae:.4f}  val_RMSE={val_metrics['rmse_mean']:.4f}",
            flush=True,
        )

        if val_mae < best_val_mae:
            # Keep a CPU copy so it is safe even if CUDA memory is tight later.
            best_val_mae = val_mae
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_metrics = evaluate_neural_model(model, test_loader, device, y_mean, y_std, data.horizon_names)

    default_run_name = f"gru_w{window}_h{hidden_dim}_l{num_layers}_bs{batch_size}"
    safe_run_name = (run_name or default_run_name).replace("/", "_").replace(" ", "_")

    report = {
        "dataset": data.dataset,
        "task": "long_term_multi_horizon_speed_forecasting",
        "model": "shared_sensor_gru",
        "run_name": safe_run_name,
        "config": {
            "window": window,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "device": str(device),
            "device_choice": device_choice,
            "cuda_available": bool(torch.cuda.is_available()),
            "n_params": int(n_params),
        },
        "horizon_steps": data.horizon_steps,
        "horizon_names": data.horizon_names,
        "feature_names": data.feature_names,
        "normalisation": stats,
        "history": history,
        "test_metrics": test_metrics,
        "elapsed_seconds": float(time.time() - t0),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"{data.dataset}_longterm_multi_horizon_{safe_run_name}.json"
    checkpoint_path = checkpoint_dir / f"{data.dataset}_longterm_multi_horizon_{safe_run_name}.pt"
    output_path.write_text(json.dumps(report, indent=2))
    # The checkpoint is optional for git, but useful for reloading without retraining.
    torch.save({
        "config": report["config"],
        "feature_names": data.feature_names,
        "horizon_steps": data.horizon_steps,
        "horizon_names": data.horizon_names,
        "normalisation": stats,
        "state_dict": best_state,
        "test_metrics": test_metrics,
    }, checkpoint_path)

    print("\n=== Neural Test Metrics ===", flush=True)
    print(f"{'model':<24} {'avg_MAE':>8} {'1h':>8} {'3h':>8} {'6h':>8} {'12h':>8}", flush=True)
    print("-" * 72, flush=True)
    maes = {m["horizon"]: m["mae"] for m in test_metrics["per_horizon"]}
    print(
        f"{'shared_sensor_gru':<24} {test_metrics['mae_mean']:>8.4f} "
        f"{maes.get('1h', float('nan')):>8.4f} "
        f"{maes.get('3h', float('nan')):>8.4f} "
        f"{maes.get('6h', float('nan')):>8.4f} "
        f"{maes.get('12h', float('nan')):>8.4f}"
    )
    print(f"\nNeural report saved -> {output_path}", flush=True)
    print(f"Checkpoint saved -> {checkpoint_path}", flush=True)

    return report


def main() -> None:
    """Parse CLI arguments, build the dataset, and run the requested mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pems08", choices=sorted(DATASETS))
    parser.add_argument("--horizons", default="12,36,72,144")
    parser.add_argument("--speed-lags", default="36,72,144,288,576,2016")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--mode", default="summary", choices=["summary", "baselines", "train_nn"])
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    data = build_multi_horizon_data(
        dataset=args.dataset,
        horizon_steps=parse_steps(args.horizons),
        speed_lags=parse_steps(args.speed_lags),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    print_summary(data)
    if args.mode == "baselines":
        run_baselines(data, Path(args.outdir), ridge_alpha=args.ridge_alpha)
    elif args.mode == "train_nn":
        run_neural_model(
            data,
            outdir=Path(args.outdir),
            checkpoint_dir=Path(args.checkpoint_dir),
            run_name=args.run_name,
            window=args.window,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            device_choice=args.device,
        )


if __name__ == "__main__":
    main()
