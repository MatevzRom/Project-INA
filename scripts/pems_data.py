"""
pems_data.py — Canonical data loading, splitting, and evaluation for PEMS datasets
───────────────────────────────────────────────────────────────────────────────────
Supports pems04, pems08 (or any dataset following the pipeline naming convention).
Supports three split strategies:
  - "temporal"      : single train/val/test split by time (fast, standard)
  - "walk_forward"  : expanding train window, val+test shift right (K folds)
  - "sliding_window": fixed-size train window shifts right (K folds)

Usage
─────
    from pems_data import load_data, make_loaders, compute_metrics, DEVICE

    # single temporal split, pems04
    data = load_data(dataset="pems04", split="temporal")
    train_loader, val_loader, test_loader = make_loaders(data, window=12)

    # walk-forward or sliding window, pems08
    folds = load_data(dataset="pems08", split="walk_forward", n_folds=5)
    for fold in folds:
        train_loader, val_loader, test_loader = make_loaders(fold, window=12)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED = Path("data/processed")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def _paths(dataset: str) -> dict:
    d = dataset.lower()
    return {
        "features": PROCESSED / "features" / f"{d}_features.npz",
        "graph":    PROCESSED / "graph"    / f"{d}.graphml",
        "meta":     PROCESSED / "labels"   / f"{d}_labels_meta.json",
    }


# ── Data container ─────────────────────────────────────────────────────────────
@dataclass
class PemsData:
    dataset:       str
    split:         str                  # "temporal" | "walk_forward" | "sliding_window"
    X:             np.ndarray           # [T_eff, N, F]
    y:             np.ndarray           # [T_eff, N]  horizon-ahead congestion
    y_current:     np.ndarray           # [T_eff, N]  current congestion
    train_idx:     np.ndarray           # int indices into T_eff
    val_idx:       np.ndarray           # int indices into T_eff
    test_idx:      np.ndarray           # int indices into T_eff
    feature_names: list
    horizon:       int
    pos_weight:    float
    N:             int
    F:             int
    T:             int
    fold:          int = 0              # 0 = temporal, 1..K = multi-fold strategies
    n_folds:       int = 1
    adj:           Optional[torch.Tensor] = None


# ── Main loader ────────────────────────────────────────────────────────────────
def load_data(
    dataset:    str   = "pems04",
    split:      str   = "temporal",
    load_graph: bool  = True,
    window:     int   = 12,
    # multi-fold options
    n_folds:    int   = 5,
    val_frac:   float = 0.1,
    test_frac:  float = 0.1,
    train_frac: float = 0.6,            # sliding_window only
) -> Union["PemsData", list]:
    """
    Load pipeline outputs and return split data.

    Parameters
    ----------
    dataset     : "pems04" or "pems08"
    split       : "temporal"       → returns a single PemsData
                  "walk_forward"   → returns list[PemsData], one per fold
                  "sliding_window" → returns list[PemsData], one per fold
    load_graph  : set False for non-GNN models to skip adjacency construction
    window      : look-back window — used to compute boundary gaps
    n_folds     : number of folds (walk_forward / sliding_window only)
    val_frac    : val window as fraction of T
    test_frac   : test window as fraction of T
    train_frac  : train window as fraction of T (sliding_window only)
    """
    if split == "temporal":
        return _load_temporal(dataset, load_graph, window)
    elif split == "walk_forward":
        return _load_walk_forward(dataset, load_graph, window, n_folds, val_frac, test_frac)
    elif split == "sliding_window":
        return _load_sliding_window(dataset, load_graph, window, n_folds, train_frac, val_frac, test_frac)
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'temporal', 'walk_forward', or 'sliding_window'.")


# ── Strategy 1: single temporal split ─────────────────────────────────────────
def _load_temporal(dataset: str, load_graph: bool, window: int) -> PemsData:
    """
    |<──────── train (60%) ────────>|gap|<── val (20%) ──>|gap|<── test (20%) ──>|

    Single fixed cut. Gap of `window` at each boundary prevents look-ahead.
    """
    paths = _paths(dataset)
    npz   = np.load(paths["features"])
    X, y, y_current, feat_names, horizon = _unpack_npz(npz)
    T, N, F = X.shape

    raw_train = np.where(npz["train_idx"])[0]
    raw_val   = np.where(npz["val_idx"])[0]
    raw_test  = np.where(npz["test_idx"])[0]

    train_idx = raw_train
    val_idx   = raw_val[raw_val   >= raw_val[0]  + window]
    test_idx  = raw_test[raw_test >= raw_test[0] + window]

    pos_weight, meta = _pos_weight(paths)
    adj = _build_adjacency(paths["graph"], N) if load_graph else None
    _print_summary(dataset, "temporal", X.shape, train_idx, val_idx, test_idx,
                   meta["positive_rate"], pos_weight, adj)

    return PemsData(
        dataset=dataset, split="temporal", fold=0, n_folds=1,
        X=X, y=y, y_current=y_current,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        feature_names=feat_names, horizon=horizon,
        pos_weight=pos_weight, N=N, F=F, T=T, adj=adj,
    )


# ── Strategy 2: walk-forward (expanding window) ────────────────────────────────
def _load_walk_forward(
    dataset: str, load_graph: bool, window: int,
    n_folds: int, val_frac: float, test_frac: float,
) -> list:
    """
    fold 1:  |██ train ██|gap|val|gap|test|
    fold 2:  |████ train ████|gap|val|gap|test|
    fold 3:  |██████ train ██████|gap|val|gap|test|

    Train always starts at t=0 and grows. Old data is never thrown away.
    Best choice when patterns are stable across the full dataset (typical for PEMS).
    """
    paths = _paths(dataset)
    npz   = np.load(paths["features"])
    X, y, y_current, feat_names, horizon = _unpack_npz(npz)
    T, N, F = X.shape

    pos_weight, meta = _pos_weight(paths)
    adj = _build_adjacency(paths["graph"], N) if load_graph else None

    val_size  = int(T * val_frac)
    test_size = int(T * test_frac)
    block     = val_size + test_size + 2 * window

    print(f"\n[{dataset}] walk_forward  folds={n_folds}  "
          f"val={val_size}  test={test_size}  gap={window}")

    folds = []
    for k in range(n_folds):
        test_end   = T - k * block
        test_start = test_end  - test_size
        val_end    = test_start - window
        val_start  = val_end   - val_size
        train_end  = val_start - window

        if train_end < window:
            print(f"  fold {k+1}: not enough data, stopping at {k} folds.")
            break

        train_idx = np.arange(window, train_end)
        val_idx   = np.arange(val_start, val_end)
        test_idx  = np.arange(test_start, test_end)

        print(f"  fold {k+1}/{n_folds}  "
              f"train=[0,{train_end})={len(train_idx)}  "
              f"val=[{val_start},{val_end})={len(val_idx)}  "
              f"test=[{test_start},{test_end})={len(test_idx)}")

        folds.append(PemsData(
            dataset=dataset, split="walk_forward", fold=k + 1, n_folds=n_folds,
            X=X, y=y, y_current=y_current,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            feature_names=feat_names, horizon=horizon,
            pos_weight=pos_weight, N=N, F=F, T=T, adj=adj,
        ))

    return folds


# ── Strategy 3: sliding window ─────────────────────────────────────────────────
def _load_sliding_window(
    dataset: str, load_graph: bool, window: int,
    n_folds: int, train_frac: float, val_frac: float, test_frac: float,
) -> list:
    """
    fold 1:  |████ train ████|gap|val|gap|test|
    fold 2:          |████ train ████|gap|val|gap|test|
    fold 3:                  |████ train ████|gap|val|gap|test|

    Train is a fixed-size window that shifts right each fold.
    Old data is discarded — only useful if you suspect concept drift
    (e.g. traffic patterns changed significantly over the dataset period).
    """
    paths = _paths(dataset)
    npz   = np.load(paths["features"])
    X, y, y_current, feat_names, horizon = _unpack_npz(npz)
    T, N, F = X.shape

    pos_weight, meta = _pos_weight(paths)
    adj = _build_adjacency(paths["graph"], N) if load_graph else None

    train_size = int(T * train_frac)
    val_size   = int(T * val_frac)
    test_size  = int(T * test_frac)
    block      = train_size + val_size + test_size + 2 * window

    if block * n_folds > T:
        raise ValueError(
            f"Not enough timesteps ({T}) for {n_folds} sliding folds of size {block}. "
            f"Reduce n_folds or fractions."
        )

    print(f"\n[{dataset}] sliding_window  folds={n_folds}  "
          f"train={train_size}  val={val_size}  test={test_size}  gap={window}")

    folds = []
    for k in range(n_folds):
        offset      = k * (val_size + test_size + 2 * window)
        train_start = offset
        train_end   = train_start + train_size
        val_start   = train_end   + window
        val_end     = val_start   + val_size
        test_start  = val_end     + window
        test_end    = test_start  + test_size

        if test_end > T:
            print(f"  fold {k+1}: exceeds T={T}, stopping at {k} folds.")
            break

        train_idx = np.arange(train_start + window, train_end)
        val_idx   = np.arange(val_start, val_end)
        test_idx  = np.arange(test_start, test_end)

        print(f"  fold {k+1}/{n_folds}  "
              f"train=[{train_start},{train_end})={len(train_idx)}  "
              f"val=[{val_start},{val_end})={len(val_idx)}  "
              f"test=[{test_start},{test_end})={len(test_idx)}")

        folds.append(PemsData(
            dataset=dataset, split="sliding_window", fold=k + 1, n_folds=n_folds,
            X=X, y=y, y_current=y_current,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            feature_names=feat_names, horizon=horizon,
            pos_weight=pos_weight, N=N, F=F, T=T, adj=adj,
        ))

    return folds


# ── Helpers ────────────────────────────────────────────────────────────────────
def _unpack_npz(npz) -> tuple:
    return (
        npz["X"], npz["y"], npz["y_current"],
        list(npz["feature_names"]), int(npz["horizon"]),
    )


def _pos_weight(paths: dict) -> tuple:
    meta     = json.loads(paths["meta"].read_text())
    pos_rate = meta["positive_rate"]
    return (1.0 - pos_rate) / pos_rate, meta


def _print_summary(dataset, split, shape, train_idx, val_idx, test_idx,
                   pos_rate, pos_weight, adj):
    print(f"\n[{dataset}]  split={split}  X={shape}")
    print(f"[split]  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"[label]  positive_rate={pos_rate:.4f}  pos_weight={pos_weight:.2f}")
    if adj is not None:
        print(f"[graph]  adj={tuple(adj.shape)}  non-zero={int((adj > 0).sum())}")


def _build_adjacency(graph_path: Path, n_nodes: int) -> torch.Tensor:
    """Directed, self-loop, D^{-1} row-normalised adjacency."""
    G      = nx.read_graphml(graph_path)
    id_map = {str(n): n for n in range(n_nodes)}
    A      = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        i, j    = id_map[u], id_map[v]
        A[i, j] = float(d.get("weight", 1.0))
    A += np.eye(n_nodes, dtype=np.float32)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return torch.tensor(A / deg, dtype=torch.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────
class GraphWindowDataset(Dataset):
    """One sample = one time window, all sensors. Returns x:[W,N,F], y:[N]"""
    def __init__(self, X: np.ndarray, y: np.ndarray, time_indices: np.ndarray, window: int):
        self.t_idx  = time_indices[time_indices >= window]
        self.X      = torch.from_numpy(X)
        self.y      = torch.from_numpy(y.astype(np.float32))
        self.window = window

    def __len__(self):
        return len(self.t_idx)

    def __getitem__(self, i: int):
        t = self.t_idx[i]
        return self.X[t - self.window : t], self.y[t]


# ── DataLoader factory ─────────────────────────────────────────────────────────
def make_loaders(
    data:        PemsData,
    window:      int = 12,
    batch_size:  int = 64,
    num_workers: int = 0,
) -> tuple:
    """Build train / val / test DataLoaders from a PemsData instance."""
    train_ds = GraphWindowDataset(data.X, data.y, data.train_idx, window)
    val_ds   = GraphWindowDataset(data.X, data.y, data.val_idx,   window)
    test_ds  = GraphWindowDataset(data.X, data.y, data.test_idx,  window)

    fold_str = f"fold={data.fold}/{data.n_folds}  " if data.n_folds > 1 else ""
    print(f"[loaders] {fold_str}train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  W={window}")

    kw = dict(num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────
def find_best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Sweep thresholds on val logits, return the one maximising F1."""
    probs  = torch.sigmoid(logits).numpy().ravel()
    y_true = labels.numpy().ravel().astype(np.uint8)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s    = [f1_score(y_true, (probs >= t).astype(np.uint8), zero_division=0)
              for t in thresholds]
    best_t = float(thresholds[np.argmax(f1s)])
    print(f"[threshold] best={best_t:.2f}  val_F1={max(f1s):.4f}")
    return best_t


def compute_metrics(
    logits:    torch.Tensor,
    labels:    torch.Tensor,
    name:      str   = "",
    threshold: float = 0.5,
) -> dict:
    """Standard binary classification metrics, identical to the baseline script."""
    probs  = torch.sigmoid(logits).numpy().ravel()
    y_true = labels.numpy().ravel().astype(np.uint8)
    y_pred = (probs >= threshold).astype(np.uint8)
    unique = np.unique(y_true)
    return {
        "name":       name,
        "threshold":  threshold,
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_true, probs)) if len(unique) > 1 else float("nan"),
        "pr_auc":     float(average_precision_score(y_true, probs)) if len(unique) > 1 else float("nan"),
        "n_positive": int(y_true.sum()),
        "n_total":    int(y_true.size),
    }


def evaluate_loader(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str = DEVICE,
) -> tuple:
    """Run model over a DataLoader in eval mode. Returns (avg_loss, logits, labels)."""
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits  = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    return (
        total_loss / len(loader.dataset),
        torch.cat(all_logits),
        torch.cat(all_labels),
    )


#Tutorial use:

# temporal — single object, no loop
# data = load_data(dataset=DATASET, split="temporal", ...)
# train_loader, val_loader, test_loader = make_loaders(data, ...)



# # walk_forward — no train_frac needed
# folds = load_data(dataset=DATASET, split="walk_forward", load_graph=True,
#                   window=WINDOW, n_folds=5,
#                   val_frac=0.1, test_frac=0.1)
# for fold in folds:
#     train_loader, val_loader, test_loader = make_loaders(fold, ...)
##     reinitialise model + optimizer here

# # sliding_window — needs train_frac
# folds = load_data(dataset=DATASET, split="sliding_window", load_graph=True,
#                   window=WINDOW, n_folds=5,
#                   train_frac=0.6, val_frac=0.1, test_frac=0.1)
# for fold in folds:
#     train_loader, val_loader, test_loader = make_loaders(fold, ...)
##    reinitialise model + optimizer here

# usefull tip: SETSEED -> check /scripts/train_gnn.py for example