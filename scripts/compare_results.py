"""
compare_results.py — Compare baseline, LSTM, and GNN test-set metrics
───────────────────────────────────────────────────────────────────────
Reads all reports/*.json files produced by the three training scripts
and prints a formatted comparison table. Highlights the best value in
each metric column and saves a combined JSON.

Run:
    python compare_results.py
"""

import json
from pathlib import Path
DATASET = "pems08"

# ── Paths ──────────────────────────────────────────────────────────────────────
REPORT_FILES = {
    "Baselines":  Path(f"reports/{DATASET}_baselines.json"),
    "LSTM":       Path(f"reports/{DATASET}_lstm_metrics.json"),
    "GCN-GRU":   Path(f"reports/{DATASET}_gnn_metrics.json"),
}
COMBINED_PATH = Path(f"reports/{DATASET}_all_metrics.json")

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1",
    "roc_auc":   "ROC-AUC",
    "pr_auc":    "PR-AUC",
}


def load_all() -> list[dict]:
    rows = []
    for source, path in REPORT_FILES.items():
        if not path.exists():
            print(f"  [skip] {path} not found — run the corresponding script first.")
            continue
        entries = json.loads(path.read_text())
        for entry in entries:
            entry["_source"] = source
            rows.append(entry)
    return rows


def col_width(rows: list[dict], col: str, header: str) -> int:
    vals = [str(r.get("name", "")) for r in rows] if col == "name" else \
           [f"{r[col]:.4f}" for r in rows if col in r]
    return max(len(header), *(len(v) for v in vals)) + 2


def print_table(rows: list[dict]):
    # Column setup
    name_w = col_width(rows, "name", "Model")
    metric_ws = {m: col_width(rows, m, METRIC_LABELS[m]) for m in METRICS}

    # Find best value per metric (higher = better for all these metrics)
    best: dict[str, float] = {}
    for m in METRICS:
        vals = [r[m] for r in rows if m in r and not (isinstance(r[m], float) and r[m] != r[m])]
        if vals:
            best[m] = max(vals)

    # Header
    sep = "─" * (name_w + sum(metric_ws.values()) + len(METRICS) + 1)
    header = f"{'Model':<{name_w}}" + "".join(
        f"{METRIC_LABELS[m]:>{metric_ws[m]}}" for m in METRICS
    )
    print()
    print(sep)
    print(header)
    print(sep)

    prev_source = None
    for row in rows:
        # Group separator between baseline / LSTM / GNN sections
        if row["_source"] != prev_source:
            if prev_source is not None:
                print()
            prev_source = row["_source"]

        name = row.get("name", "?")
        line = f"{name:<{name_w}}"
        for m in METRICS:
            if m not in row or (isinstance(row[m], float) and row[m] != row[m]):
                val_str = f"{'N/A':>{metric_ws[m]}}"
            else:
                val_str = f"{row[m]:.4f}"
                # Mark best value with an asterisk
                if m in best and abs(row[m] - best[m]) < 1e-9:
                    val_str = f"{val_str}*"
                val_str = f"{val_str:>{metric_ws[m]}}"
            line += val_str
        print(line)

    print(sep)
    print("  * = best in column\n")


def print_improvements(rows: list[dict]):
    """Show % improvement of best NN over best baseline."""
    baseline_rows = [r for r in rows if r["_source"] == "Baselines"]
    nn_rows       = [r for r in rows if r["_source"] != "Baselines"]
    if not baseline_rows or not nn_rows:
        return

    print("── Improvement over best baseline ────────────────────────────────────")
    for m in ["f1", "roc_auc", "pr_auc"]:
        best_base = max((r[m] for r in baseline_rows if m in r and r[m] == r[m]), default=None)
        if best_base is None or best_base == 0:
            continue
        for row in nn_rows:
            if m not in row or row[m] != row[m]:
                continue
            delta = (row[m] - best_base) / best_base * 100
            sign  = "+" if delta >= 0 else ""
            print(f"  {row['name']:<40} {METRIC_LABELS[m]}: {sign}{delta:.1f}%  ({best_base:.4f} → {row[m]:.4f})")
    print()


def main():
    print(f"\n=== {DATASET} Model Comparison ===")

    rows = load_all()
    if not rows:
        print("No report files found. Run the pipeline and training scripts first.")
        return

    print_table(rows)
    print_improvements(rows)

    # Save combined report
    clean = [{k: v for k, v in r.items() if k != "_source"} for r in rows]
    COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_PATH.write_text(json.dumps(clean, indent=2))
    print(f"Combined report saved → {COMBINED_PATH}")


if __name__ == "__main__":
    main()
