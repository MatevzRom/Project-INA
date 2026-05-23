"""
pems_data.py — Canonical data loading, splitting, and evaluation for PEMS datasets
───────────────────────────────────────────────────────────────────────────────────
Supports any dataset that follows the pipeline naming convention (pems04, pems08, ...).

Usage
─────
    from pems_data import load_data, make_loaders, compute_metrics, DEVICE

    # PEMS04 (default)
    data = load_data()

    # PEMS08
    data = load_data(dataset="pems08")

    # Both use identical splits / metrics / loaders
    train_loader, val_loader, test_loader = make_loaders(data, window=12, batch_size=64)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    """Derive all file paths from dataset name (e.g. 'pems04', 'pems08')."""
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
    X:             np.ndarray           # [T_eff, N, F]
    y:             np.ndarray           # [T_eff, N]  1hr-ahead congestion
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
    adj:           Optional[torch.Tensor] = None   # [N, N] normalised adjacency


# ── Main loader ────────────────────────────────────────────────────────────────
def load_data(
    dataset:    str  = "pems04",
    load_graph: bool = True,
    window:     int  = 12,
) -> PemsData:
    """
    Load pipeline outputs for any PEMS dataset into a PemsData container.

    Parameters
    ----------
    dataset     : e.g. "pems04" or "pems08" — drives all file paths
    load_graph  : set False for non-GNN models to skip adjacency construction
    window      : look-back window — used to compute boundary gaps between splits
    """
    paths = _paths(dataset)

    # ── Features ──────────────────────────────────────────────────────────────
    npz          = np.load(paths["features"])
    X            = npz["X"]                     # [T, N, F]
    y            = npz["y"]                     # [T, N]
    y_current    = npz["y_current"]             # [T, N]
    feat_names   = list(npz["feature_names"])
    horizon      = int(npz["horizon"])
    T, N, F      = X.shape

    # ── Splits with boundary gap ───────────────────────────────────────────────
    # Remove `window` steps from the start of val and test so no sliding-window
    # sample straddles two splits.
    raw_train = np.where(npz["train_idx"])[0]
    raw_val   = np.where(npz["val_idx"])[0]
    raw_test  = np.where(npz["test_idx"])[0]

    train_idx = raw_train
    val_idx   = raw_val[raw_val   >= raw_val[0]  + window]
    test_idx  = raw_test[raw_test >= raw_test[0] + window]

    # ── Class imbalance weight ─────────────────────────────────────────────────
    meta       = json.loads(paths["meta"].read_text())
    pos_rate   = meta["positive_rate"]
    pos_weight = (1.0 - pos_rate) / pos_rate

    # ── Adjacency ──────────────────────────────────────────────────────────────
    adj = _build_adjacency(paths["graph"], N) if load_graph else None

    print(f"[{dataset}]  X={X.shape}  F={F}  horizon={horizon}")
    print(f"[split]  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}  gap={window}")
    print(f"[label]  positive_rate={pos_rate:.4f}  pos_weight={pos_weight:.2f}")
    if adj is not None:
        print(f"[graph]  adj={tuple(adj.shape)}  non-zero={int((adj > 0).sum())}")

    return PemsData(
        dataset=dataset,
        X=X, y=y, y_current=y_current,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        feature_names=feat_names, horizon=horizon,
        pos_weight=pos_weight,
        N=N, F=F, T=T, adj=adj,
    )


def _build_adjacency(graph_path: Path, n_nodes: int) -> torch.Tensor:
    """
    Load graphml → directed, self-loop, D^{-1} row-normalised adjacency.
    Kept directed (no symmetrisation) so upstream/downstream flow is preserved.
    """
    G      = nx.read_graphml(graph_path)
    id_map = {str(n): n for n in range(n_nodes)}
    A      = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for u, v, d in G.edges(data=True):
        i, j    = id_map[u], id_map[v]
        A[i, j] = float(d.get("weight", 1.0))   # directed: keep as-is

    A += np.eye(n_nodes, dtype=np.float32)       # self-loops

    # Row normalisation: D^{-1} A  (simpler than symmetric, fine for directed)
    deg    = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_norm = A / deg

    return torch.tensor(A_norm, dtype=torch.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────
class GraphWindowDataset(Dataset):
    """
    One sample = one time window covering ALL sensors.
    Returns  x: [W, N, F],  y: [N]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, time_indices: np.ndarray, window: int):
        valid      = time_indices[time_indices >= window]
        self.t_idx = valid
        self.X     = torch.from_numpy(X)
        self.y     = torch.from_numpy(y.astype(np.float32))
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

    print(f"[loaders] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  W={window}")

    kw = dict(num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────
def find_best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Sweep thresholds on val logits and return the one maximising F1.
    Call this after val evaluation; use the result for test metrics.
    """
    probs  = torch.sigmoid(logits).numpy().ravel()
    y_true = labels.numpy().ravel().astype(np.uint8)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (probs >= t).astype(np.uint8), zero_division=0)
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
    """
    Standard binary classification metrics, identical to the baseline script.
    Pass the threshold returned by find_best_threshold() for fairer F1.
    """
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