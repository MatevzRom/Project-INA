from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_data(dataset: str) -> dict:
    path = Path(f"data/processed/features/{dataset}_features.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/{dataset}/run_pipeline.py first."
        )
    npz = np.load(path)
    X   = npz["X"].astype(np.float32)
    y   = npz["y"].astype(np.float32)
    train_idx = npz["train_idx"]
    val_idx   = npz["val_idx"]
    test_idx  = npz["test_idx"]
    F = X.shape[-1]

    def split(mask):
        Xs = X[mask].reshape(-1, F)
        ys = y[mask].reshape(-1)
        return Xs, ys

    X_train, y_train = split(train_idx)
    X_val,   y_val   = split(val_idx)
    X_test,  y_test  = split(test_idx)

    pos_rate   = float(y_train.mean())
    pos_weight = float((1 - pos_rate) / pos_rate)

    print(f"  Train:  {X_train.shape[0]:>9,} samples  ({y_train.mean():.3%} positive)")
    print(f"  Val:    {X_val.shape[0]:>9,} samples  ({y_val.mean():.3%} positive)")
    print(f"  Test:   {X_test.shape[0]:>9,} samples  ({y_test.mean():.3%} positive)")
    print(f"  pos_weight (n_neg/n_pos): {pos_weight:.1f}")

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        F=F,
        pos_weight=pos_weight,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                device: torch.device) -> DataLoader:
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: MLP, loader: DataLoader, criterion: nn.Module,
                optimiser: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        optimiser.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_y = [], []
    for Xb, yb in loader:
        all_logits.append(model(Xb).cpu().numpy())
        all_y.append(yb.cpu().numpy())
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_y).astype(np.int32)
    proba  = 1 / (1 + np.exp(-logits))
    y_pred = (proba >= 0.5).astype(np.int32)
    has_both = len(np.unique(y_true)) > 1
    return {
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_true, proba)) if has_both else float("nan"),
        "pr_auc":     float(average_precision_score(y_true, proba)) if has_both else float("nan"),
        "n_positive": int(y_true.sum()),
        "n_total":    int(y_true.size),
    }


def train_config(cfg: dict, data: dict, device: torch.device,
                 max_epochs: int, patience: int, verbose: bool = False) -> dict:
    model = MLP(
        in_features=data["F"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([data["pos_weight"]], device=device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=2
    )

    train_loader = make_loader(data["X_train"], data["y_train"], cfg["batch_size"], shuffle=True,  device=device)
    val_loader   = make_loader(data["X_val"],   data["y_val"],   cfg["batch_size"], shuffle=False, device=device)

    best_val_pr_auc   = -1.0
    best_epoch        = 0
    best_state        = None
    epochs_no_improve = 0
    history           = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimiser)
        val_m      = evaluate(model, val_loader, device)
        val_pr     = val_m["pr_auc"]
        scheduler.step(val_pr)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val_m.items()}})

        if val_pr > best_val_pr_auc:
            best_val_pr_auc   = val_pr
            best_epoch        = epoch
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"    epoch {epoch:3d}  loss={train_loss:.4f}  val_pr_auc={val_pr:.4f}  "
                  f"val_f1={val_m['f1']:.4f}  {'*' if epochs_no_improve == 0 else ''}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (best={best_epoch})")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_loader = make_loader(data["X_test"], data["y_test"], cfg["batch_size"], shuffle=False, device=device)
    test_m = evaluate(model, test_loader, device)

    return {
        "config":       cfg,
        "best_epoch":   best_epoch,
        "val_pr_auc":   best_val_pr_auc,
        "val_metrics":  history[best_epoch - 1],
        "test_metrics": test_m,
        "history":      history,
        "model_state":  best_state,
    }


GRID = {
    "hidden_dims": [
        (128,),
        (256, 128),
        (256, 128, 64),
    ],
    "dropout":    [0.2, 0.4],
    "lr":         [1e-3, 3e-4],
    "batch_size": [4096],
}


def grid_search(data: dict, device: torch.device, max_epochs: int, patience: int) -> list[dict]:
    keys    = list(GRID.keys())
    values  = list(GRID.values())
    configs = [dict(zip(keys, combo)) for combo in product(*values)]
    n       = len(configs)

    print(f"\nGrid search: {n} configurations  (device={device})\n")
    print(f"  {'#':>3}  {'hidden_dims':>20}  {'drop':>5}  {'lr':>8}  {'batch':>6}  "
          f"{'val_pr_auc':>10}  {'val_f1':>7}  {'time':>6}")
    print("  " + "-" * 78)

    results = []
    for i, cfg in enumerate(configs, 1):
        t0      = time.time()
        res     = train_config(cfg, data, device, max_epochs, patience, verbose=False)
        elapsed = time.time() - t0
        val_m   = res["val_metrics"]
        print(f"  {i:>3}  {str(cfg['hidden_dims']):>20}  {cfg['dropout']:>5.1f}  "
              f"{cfg['lr']:>8.0e}  {cfg['batch_size']:>6}  "
              f"{res['val_pr_auc']:>10.4f}  {val_m['val_f1']:>7.4f}  {elapsed:>5.0f}s")
        results.append(res)

    results.sort(key=lambda r: r["val_pr_auc"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       default="pems04", choices=["pems04", "pems08"])
    parser.add_argument("--no-gridsearch", action="store_true")
    parser.add_argument("--epochs",        type=int, default=40)
    parser.add_argument("--patience",      type=int, default=6)
    parser.add_argument("--outdir",        default="reports")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = _best_device()
    print(f"Dataset : {args.dataset.upper()}")
    print(f"Device  : {device}")

    print("\nLoading data …")
    data = load_data(args.dataset)

    if args.no_gridsearch:
        default_cfg = {"hidden_dims": (256, 128), "dropout": 0.2, "lr": 1e-3, "batch_size": 4096}
        print(f"\nTraining with default config: {default_cfg}")
        results = [train_config(default_cfg, data, device, args.epochs, args.patience, verbose=True)]
    else:
        results = grid_search(data, device, args.epochs, args.patience)

    best = results[0]
    print(f"\n{'─'*60}")
    print(f"Best config   : {best['config']}")
    print(f"Best val epoch: {best['best_epoch']}")
    print(f"Val  PR-AUC   : {best['val_pr_auc']:.4f}")

    tm = best["test_metrics"]
    print(f"\nTest-set results (1-hour-ahead congestion):")
    print(f"  Accuracy  : {tm['accuracy']:.4f}")
    print(f"  Precision : {tm['precision']:.4f}")
    print(f"  Recall    : {tm['recall']:.4f}")
    print(f"  F1        : {tm['f1']:.4f}")
    print(f"  ROC-AUC   : {tm['roc_auc']:.4f}")
    print(f"  PR-AUC    : {tm['pr_auc']:.4f}  ← headline metric")
    print(f"  Positives : {tm['n_positive']:,} / {tm['n_total']:,}")

    summary = []
    for r in results:
        summary.append({
            "config":       r["config"],
            "best_epoch":   r["best_epoch"],
            "val_pr_auc":   r["val_pr_auc"],
            "val_metrics":  {k: v for k, v in r["val_metrics"].items() if k != "history"},
            "test_metrics": r["test_metrics"],
        })
    gs_path = out / f"{args.dataset}_nn_gridsearch.json"
    gs_path.write_text(json.dumps(summary, indent=2))
    print(f"\nGrid search results saved → {gs_path}")

    model_path = out / f"{args.dataset}_best_nn.pt"
    torch.save({
        "config":       best["config"],
        "F":            data["F"],
        "state_dict":   best["model_state"],
        "test_metrics": tm,
    }, model_path)
    print(f"Best model saved        → {model_path}")


if __name__ == "__main__":
    main()
