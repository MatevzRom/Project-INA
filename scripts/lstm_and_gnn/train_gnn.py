"""
train_gnn.py — Spatio-Temporal GCN-GRU for PEMS04 congestion prediction (1hr ahead)
Uses pems4_data.py for standardised splits, datasets, and metrics.

Run:
    python train_gnn.py
"""
import sys
print(sys.path)
# print("ss")

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from scripts.pems_data import (
    DEVICE,
    compute_metrics,
    evaluate_loader,
    load_data,
    make_loaders,
)
# at the top of train_lstm.py and train_gnn.py, replace the hardcoded paths:
DATASET      = "pems08"   # change to "pems08" when needed

METRICS_PATH = Path(f"reports/{DATASET}_gnn_metrics.json")
CKPT_PATH    = Path(f"checkpoints/{DATASET}_gnn_best.pt")
# ── Paths ──────────────────────────────────────────────────────────────────────
# METRICS_PATH = Path("reports/pems04_gnn_metrics.json")
# CKPT_PATH    = Path("checkpoints/gnn_best.pt")

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW     = 12
HIDDEN_DIM = 64
GCN_LAYERS = 2
GRU_LAYERS = 2
DROPOUT    = 0.3
LR         = 1e-3
BATCH_SIZE = 64
EPOCHS     = 30
PATIENCE   = 5


# ── Model ──────────────────────────────────────────────────────────────────────
class GraphConvLayer(nn.Module):
    """H' = ReLU( A_norm @ H @ W )"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(adj @ x))


class GCNGRU(nn.Module):
    """
    Spatio-Temporal GCN-GRU.

    Per timestep  -> GCN stack : [N, F]    ->  [N, H]
    Across time   -> GRU       : [N, W, H] ->  [N, H]
    Output        -> Linear    : [N, H]    ->  [N, 1]
    """

    def __init__(self, in_features, hidden_dim, gcn_layers, gru_layers, dropout, adj):
        super().__init__()
        self.register_buffer("adj", adj)

        gcn = []
        for i in range(gcn_layers):
            gcn.append(GraphConvLayer(in_features if i == 0 else hidden_dim, hidden_dim))
        self.gcn_layers = nn.ModuleList(gcn)

        self.gru     = nn.GRU(hidden_dim, hidden_dim, gru_layers,
                              batch_first=True,
                              dropout=dropout if gru_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def _gcn_step(self, x):
        h = x
        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, self.adj)
            if i < len(self.gcn_layers) - 1:
                h = self.dropout(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, N, F = x.shape

        gcn_seq = []
        for t in range(W):
            xt = x[:, t]                              # [B, N, F]
            ht = torch.stack([self._gcn_step(xt[b]) for b in range(B)])  # [B, N, H]
            gcn_seq.append(ht)
        gcn_seq = torch.stack(gcn_seq, dim=2)         # [B, N, W, H]

        gcn_seq      = gcn_seq.reshape(B * N, W, -1)  # [B*N, W, H]
        gru_out, _   = self.gru(gcn_seq)              # [B*N, W, H]
        gru_out      = self.dropout(gru_out[:, -1, :])# [B*N, H]
        return self.fc(gru_out).reshape(B, N)          # [B, N]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"\n=== GCN-GRU Training  (device: {DEVICE}) ===\n")

    data = load_data(dataset=DATASET,load_graph=True, window=WINDOW)
    train_loader, val_loader, test_loader = make_loaders(data, window=WINDOW, batch_size=BATCH_SIZE)

    adj   = data.adj.to(DEVICE)
    model = GCNGRU(data.F, HIDDEN_DIM, GCN_LAYERS, GRU_LAYERS, DROPOUT, adj).to(DEVICE)
    pw    = torch.tensor([data.pos_weight], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    print(f"\nParameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss, patience_ctr = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y_batch in train_loader:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_logits, val_labels = evaluate_loader(model, val_loader, criterion, DEVICE)
        val_m = compute_metrics(val_logits, val_labels)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}"
              f"  F1={val_m['f1']:.4f}  ROC={val_m['roc_auc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\n── Test metrics ──────────────────────────────────────────────────")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    _, test_logits, test_labels = evaluate_loader(model, test_loader, criterion, DEVICE)
    test_m = compute_metrics(test_logits, test_labels,
                             name=f"GCN-GRU(W={WINDOW},H={HIDDEN_DIM},GCN={GCN_LAYERS},GRU={GRU_LAYERS})")

    for k, v in test_m.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.4f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps([test_m], indent=2))
    print(f"\nMetrics -> {METRICS_PATH}  |  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
