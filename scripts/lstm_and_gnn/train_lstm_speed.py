"""
train_lstm_speed.py — LSTM for speed prediction (regression, 1-hour ahead)
Uses same features and splitting as speed baselines (no data leakage).

Features:
  - flow, occupancy (current)
  - speed_lag_288, speed_lag_2016 (pattern-based, no recent lags)
  - temporal (sin/cos time-of-day, day-of-week)
  - spatial (neighbor averages)

Supports all three split modes: temporal, walk_forward, sliding_window

Run:
    python train_lstm_speed.py --dataset pems08 --split temporal
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Import feature building from our baseline script
import sys
sys.path.append('scripts')
from generate_speed_baselines import (
    build_features_and_targets,
    temporal_split,
    walk_forward_split,
    sliding_window_split,
    compute_mape,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable cuDNN to avoid version incompatibility (still uses CUDA)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Dataset ────────────────────────────────────────────────────────────────────
class SpeedWindowDataset(Dataset):
    """Dataset for speed prediction with time windows"""
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, window: int):
        self.indices = indices[indices >= window]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.window = window

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        return self.X[t - self.window : t], self.y[t]


# ── Model ──────────────────────────────────────────────────────────────────────
class SensorLSTM(nn.Module):
    """
    LSTM for speed prediction (regression).
    Treats each sensor as independent time series.

    [B, W, N, F] -> reshape [B*N, W, F] -> LSTM -> Linear -> [B, N]
    """
    def __init__(self, in_features: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, W, F)  # [B*N, W, F]
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Last timestep
        return self.fc(out).reshape(B, N)  # [B, N]


# ── Training ───────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)
    mape = compute_mape(targets, preds)

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ── Main ───────────────────────────────────────────────────────────────────────
def train_fold(X, y_speed, y_current, train_idx, val_idx, test_idx,
               window, hidden_dim, num_layers, dropout, lr, batch_size, epochs, patience, seed):
    """Train LSTM on one fold"""
    set_seed(seed)

    # Normalize features using training data only
    T, N, F = X.shape
    X_flat = X.reshape(T * N, F)  # [T*N, F]

    # Fit scaler on training timesteps only
    train_mask = np.zeros(T, dtype=bool)
    train_mask[train_idx] = True
    train_flat_idx = np.repeat(train_mask, N)  # Repeat for each sensor

    scaler = StandardScaler()
    scaler.fit(X_flat[train_flat_idx])
    X_normalized = scaler.transform(X_flat).reshape(T, N, F)

    print(f"  Feature normalization: mean={X_flat[train_flat_idx].mean():.4f}, "
          f"std={X_flat[train_flat_idx].std():.4f}")

    # Create datasets
    train_ds = SpeedWindowDataset(X_normalized, y_speed, train_idx, window)
    val_ds = SpeedWindowDataset(X_normalized, y_speed, val_idx, window)
    test_ds = SpeedWindowDataset(X_normalized, y_speed, test_idx, window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Loaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Model
    F = X.shape[-1]
    model = SensorLSTM(F, hidden_dim, num_layers, dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training loop
    best_val_mae = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics["mae"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_MAE={val_metrics['mae']:.4f}  val_R²={val_metrics['r2']:.4f}")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_ctr = 0
            best_state = model.state_dict().copy()
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, DEVICE)

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pems08", choices=["pems04", "pems08"])
    parser.add_argument("--split", type=str, default="temporal",
                        choices=["temporal", "walk_forward", "sliding_window"])
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"LSTM Speed Prediction - {args.dataset.upper()} - {args.split.upper()}")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}\n")

    # Build features (same as baselines)
    X, y_speed, y_current, G, feat_names = build_features_and_targets(args.dataset)

    # Get splits
    print(f"\n=== Creating {args.split} splits ===")
    if args.split == "temporal":
        folds = temporal_split(X, y_speed, y_current)
    elif args.split == "walk_forward":
        folds = walk_forward_split(X, y_speed, y_current, window=args.window)
    elif args.split == "sliding_window":
        folds = sliding_window_split(X, y_speed, y_current, window=args.window)

    # Train on each fold
    all_results = []
    for fold_idx, (X_fold, y_fold, y_curr, train_idx, val_idx, test_idx) in enumerate(folds, 1):
        if len(folds) > 1:
            print(f"\n{'='*70}")
            print(f"Fold {fold_idx}/{len(folds)}")
            print(f"{'='*70}")

        result = train_fold(
            X_fold, y_fold, y_curr, train_idx, val_idx, test_idx,
            args.window, args.hidden_dim, args.num_layers, args.dropout,
            args.lr, args.batch_size, args.epochs, args.patience, args.seed
        )

        result["fold"] = fold_idx
        all_results.append(result)

        print(f"\n  Test Results (Fold {fold_idx}):")
        print(f"    MAE:  {result['mae']:.4f}")
        print(f"    RMSE: {result['rmse']:.4f}")
        print(f"    R²:   {result['r2']:.4f}")
        print(f"    MAPE: {result['mape']:.2f}%")

    # Average across folds
    if len(folds) > 1:
        print(f"\n{'='*70}")
        print(f"Average across {len(folds)} folds:")
        print(f"{'='*70}")
        avg_result = {
            "mae_mean": float(np.mean([r["mae"] for r in all_results])),
            "mae_std": float(np.std([r["mae"] for r in all_results])),
            "rmse_mean": float(np.mean([r["rmse"] for r in all_results])),
            "rmse_std": float(np.std([r["rmse"] for r in all_results])),
            "r2_mean": float(np.mean([r["r2"] for r in all_results])),
            "r2_std": float(np.std([r["r2"] for r in all_results])),
            "mape_mean": float(np.mean([r["mape"] for r in all_results])),
            "mape_std": float(np.std([r["mape"] for r in all_results])),
            "n_folds": len(folds),
        }
        print(f"  MAE:  {avg_result['mae_mean']:.4f} ± {avg_result['mae_std']:.4f}")
        print(f"  RMSE: {avg_result['rmse_mean']:.4f} ± {avg_result['rmse_std']:.4f}")
        print(f"  R²:   {avg_result['r2_mean']:.4f} ± {avg_result['r2_std']:.4f}")
        print(f"  MAPE: {avg_result['mape_mean']:.2f}% ± {avg_result['mape_std']:.2f}%")

        final_result = avg_result
    else:
        final_result = all_results[0]

    # Save results
    output_path = Path("reports") / f"{args.dataset}_lstm_speed_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_result, indent=2))

    print(f"\n✓ Saved → {output_path}")
    print(f"✓ Total time: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
