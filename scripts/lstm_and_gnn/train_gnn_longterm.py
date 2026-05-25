"""
train_gnn_longterm.py — GCN-GRU for long-term speed prediction (6-12 hours ahead)

Long-term spatiotemporal forecasting scenario:
  - Input: 48-72 hours of historical data (window=576-864)
  - Predict: 6 hours ahead (horizon=72)
  - Features: Historical speed lags + time features ONLY
  - NO current measurements (flow, occupancy, neighbors)

Run:
    python scripts/lstm_and_gnn/train_gnn_longterm.py --dataset pems08 --split temporal
"""

import argparse
import json
import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import networkx as nx

import sys
sys.path.append('scripts')
from generate_longterm_baselines import build_longterm_features, temporal_split, walk_forward_split, compute_mape

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    """Dataset for long-term speed prediction with time windows"""
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

class GraphConvLayer(nn.Module):
    """
    Graph Convolution: H' = ReLU( A_norm @ H @ W )

    Accepts batched input of any shape [..., N, F] — no Python loops needed.
    torch.matmul broadcasts the [N, N] adjacency over all leading batch dims.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   [*, N, F]   (any number of leading dims)
        # adj: [N, N]      (broadcast over leading dims automatically)
        return torch.relu(self.linear(torch.matmul(adj, x)))


class GCNGRU(nn.Module):
    """
    Spatio-Temporal GCN-GRU for speed prediction (regression).

    Key change vs original: GCN is applied to ALL timesteps and ALL batch
    samples in a single vectorized call by reshaping [B, W, N, F] -> [B*W, N, F]
    before the GCN stack, then reshaping back for the GRU.

    This eliminates the two nested Python loops (for t in W, for b in B)
    that caused ~18,000 tiny GPU kernel launches per batch.
    """

    def __init__(self, in_features, hidden_dim, gcn_layers, gru_layers, dropout, adj):
        super().__init__()
        self.register_buffer("adj", adj)

        gcn = []
        for i in range(gcn_layers):
            gcn.append(GraphConvLayer(in_features if i == 0 else hidden_dim, hidden_dim))
        self.gcn_layers = nn.ModuleList(gcn)

        self.gru = nn.GRU(
            hidden_dim, hidden_dim, gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, N, F = x.shape

        # ── GCN: vectorized over batch AND time ───────────────────────────────
        # Merge batch + time into one leading dim so GCN runs in ONE GPU call
        h = x.reshape(B * W, N, F)            # [B*W, N, F]

        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, self.adj)             # [B*W, N, H]
            if i < len(self.gcn_layers) - 1:
                h = self.dropout(h)

        # ── GRU: across time per node ─────────────────────────────────────────
        # Rearrange to [B*N, W, H] so GRU processes each node's time-series
        h = h.reshape(B, W, N, -1)            # [B, W, N, H]
        h = h.permute(0, 2, 1, 3)             # [B, N, W, H]
        h = h.reshape(B * N, W, -1)           # [B*N, W, H]

        gru_out, _ = self.gru(h)              # [B*N, W, H]
        gru_out = self.dropout(gru_out[:, -1, :])  # [B*N, H]  (last timestep)

        return self.fc(gru_out).reshape(B, N) # [B, N]


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", unit="batch", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  eval ", unit="batch", leave=False,
                         bar_format="{l_bar}{bar:30}{r_bar}"):
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()

    mae  = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2   = r2_score(targets, preds)
    mape = compute_mape(targets, preds)

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ── Fold ───────────────────────────────────────────────────────────────────────

def train_fold(X, y_speed, G, train_idx, val_idx, test_idx,
               window, hidden_dim, gcn_layers, gru_layers, dropout,
               lr, batch_size, epochs, patience, seed):

    set_seed(seed)

    # Normalise adjacency matrix
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(N), weight=None)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_norm = A / deg
    adj = torch.from_numpy(A_norm).float().to(DEVICE)

    # Datasets & loaders
    train_ds = SpeedWindowDataset(X, y_speed, train_idx, window)
    val_ds   = SpeedWindowDataset(X, y_speed, val_idx,   window)
    test_ds  = SpeedWindowDataset(X, y_speed, test_idx,  window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=DEVICE == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=DEVICE == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=DEVICE == "cuda")

    print(f"  Samples — train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")

    # Model
    F = X.shape[-1]
    model = GCNGRU(F, hidden_dim, gcn_layers, gru_layers, dropout, adj).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_mae = float("inf")
    patience_ctr = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        train_loss  = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics["mae"])
        elapsed = time.time() - t_ep

        print(
            f"  Epoch {epoch:02d}/{epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_MAE={val_metrics['mae']:.4f}  "
            f"val_R²={val_metrics['r2']:.4f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_ctr = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return evaluate(model, test_loader, DEVICE)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   default="pems08",
                        choices=["pems04", "pems08"])
    parser.add_argument("--split",      type=str,   default="temporal",
                        choices=["temporal", "walk_forward"])
    parser.add_argument("--horizon",    type=int,   default=72,
                        help="Prediction horizon steps (72 = 6 h at 5-min intervals)")
    parser.add_argument("--window",     type=int,   default=576,
                        help="Input window steps (576 = 48 h)")
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--gcn_layers", type=int,   default=2)
    parser.add_argument("--gru_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"GCN-GRU Long-term Speed Forecasting — {args.dataset.upper()} — {args.split.upper()}")
    print(f"{'='*70}")
    print(f"Device : {DEVICE}")
    print(f"Window : {args.window} steps ({args.window * 5 // 60} h)")
    print(f"Horizon: {args.horizon} steps ({args.horizon * 5 // 60} h)\n")

    # Build features
    X, y_speed, G, feat_names = build_longterm_features(args.dataset, horizon=args.horizon)

    # Splits
    print(f"=== Creating {args.split} splits ===")
    if args.split == "temporal":
        folds = temporal_split(X, y_speed)
    else:
        folds = walk_forward_split(X, y_speed, window=args.window)

    all_results = []
    for fold_idx, (X_fold, y_fold, train_idx, val_idx, test_idx) in enumerate(folds, 1):
        if len(folds) > 1:
            print(f"\n{'='*70}")
            print(f"Fold {fold_idx}/{len(folds)}")
            print(f"{'='*70}")

        result = train_fold(
            X_fold, y_fold, G, train_idx, val_idx, test_idx,
            args.window, args.hidden_dim, args.gcn_layers, args.gru_layers,
            args.dropout, args.lr, args.batch_size, args.epochs, args.patience, args.seed,
        )

        result["fold"] = fold_idx
        all_results.append(result)

        print(f"\n  Test results (fold {fold_idx}):")
        print(f"    MAE  : {result['mae']:.4f}")
        print(f"    RMSE : {result['rmse']:.4f}")
        print(f"    R²   : {result['r2']:.4f}")
        print(f"    MAPE : {result['mape']:.2f}%")

    # Average across folds
    if len(folds) > 1:
        print(f"\n{'='*70}")
        print(f"Average across {len(folds)} folds:")
        print(f"{'='*70}")
        final_result = {
            "mae_mean":  float(np.mean([r["mae"]  for r in all_results])),
            "mae_std":   float(np.std( [r["mae"]  for r in all_results])),
            "rmse_mean": float(np.mean([r["rmse"] for r in all_results])),
            "rmse_std":  float(np.std( [r["rmse"] for r in all_results])),
            "r2_mean":   float(np.mean([r["r2"]   for r in all_results])),
            "r2_std":    float(np.std( [r["r2"]   for r in all_results])),
            "mape_mean": float(np.mean([r["mape"] for r in all_results])),
            "mape_std":  float(np.std( [r["mape"] for r in all_results])),
            "n_folds":   len(folds),
        }
        print(f"  MAE  : {final_result['mae_mean']:.4f} ± {final_result['mae_std']:.4f}")
        print(f"  RMSE : {final_result['rmse_mean']:.4f} ± {final_result['rmse_std']:.4f}")
        print(f"  R²   : {final_result['r2_mean']:.4f} ± {final_result['r2_std']:.4f}")
        print(f"  MAPE : {final_result['mape_mean']:.2f}% ± {final_result['mape_std']:.2f}%")
    else:
        final_result = all_results[0]

    # Save
    output_path = Path("reports") / f"{args.dataset}_gnn_longterm_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_result, indent=2))

    print(f"\n✓ Saved → {output_path}")
    print(f"✓ Total time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()