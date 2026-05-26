"""
Create a compact summary table and MAE figure for long-term PEMS08 forecasting.

This script does not train models. It reads the JSON files produced by
scripts/longterm_multi_horizon.py and writes report-ready comparison outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_REPORT = Path("reports/pems08_longterm_multi_horizon_baselines.json")
DEFAULT_NEURAL_GLOB = "reports/pems08_longterm_multi_horizon_gru*.json"
DEFAULT_SUMMARY_OUT = Path("reports/pems08_longterm_multi_horizon_summary.json")
DEFAULT_FIGURE_OUT = Path("reports/figures/pems08_longterm_multi_horizon_mae.png")


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON report file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def per_horizon_mae(metrics: dict[str, Any]) -> list[float]:
    """Extract MAE values in the original horizon order."""

    return [float(item["mae"]) for item in metrics["per_horizon"]]


def per_horizon_rmse(metrics: dict[str, Any]) -> list[float]:
    """Extract RMSE values in the original horizon order."""

    return [float(item["rmse"]) for item in metrics["per_horizon"]]


def short_neural_label(report: dict[str, Any]) -> str:
    """Build a compact label from the neural report configuration."""

    run_name = report.get("run_name", "gru")
    config = report.get("config", {})
    window = config.get("window")
    hidden = config.get("hidden_dim")
    layers = config.get("num_layers")
    dropout = config.get("dropout")
    epochs = config.get("epochs")

    parts = ["GRU"]
    if window is not None:
        parts.append(f"w{window}")
    if hidden is not None:
        parts.append(f"h{hidden}")
    if layers is not None:
        parts.append(f"L{layers}")
    if dropout not in (None, 0, 0.0):
        parts.append(f"d{dropout}")
    if epochs is not None:
        parts.append(f"e{epochs}")
    label = " ".join(parts)
    return label if len(label) > 3 else f"GRU {run_name}"


def collect_baselines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect baseline rows from the baseline report."""

    report = load_json(path)
    horizon_names = report["horizon_names"]
    rows = []
    for model in report["models"]:
        metrics = {
            "per_horizon": model["per_horizon"],
            "mae_mean": model["mae_mean"],
            "rmse_mean": model["rmse_mean"],
            "mape_mean": model["mape_mean"],
            "r2_mean": model["r2_mean"],
        }
        rows.append(
            {
                "name": model["name"],
                "kind": "baseline",
                "source": str(path),
                "mae_mean": float(model["mae_mean"]),
                "rmse_mean": float(model["rmse_mean"]),
                "mape_mean": float(model["mape_mean"]),
                "r2_mean": float(model["r2_mean"]),
                "mae_by_horizon": per_horizon_mae(metrics),
                "rmse_by_horizon": per_horizon_rmse(metrics),
            }
        )
    return rows, horizon_names


def collect_neural_reports(pattern: str) -> list[dict[str, Any]]:
    """Collect neural result rows from all reports matching a glob pattern."""

    rows = []
    for path in sorted(Path(".").glob(pattern)):
        report = load_json(path)
        metrics = report["test_metrics"]
        config = report.get("config", {})
        rows.append(
            {
                "name": report.get("run_name", path.stem),
                "label": short_neural_label(report),
                "kind": "neural",
                "source": str(path),
                "mae_mean": float(metrics["mae_mean"]),
                "rmse_mean": float(metrics["rmse_mean"]),
                "mape_mean": float(metrics["mape_mean"]),
                "r2_mean": float(metrics["r2_mean"]),
                "mae_by_horizon": per_horizon_mae(metrics),
                "rmse_by_horizon": per_horizon_rmse(metrics),
                "config": {
                    "window": config.get("window"),
                    "hidden_dim": config.get("hidden_dim"),
                    "num_layers": config.get("num_layers"),
                    "dropout": config.get("dropout"),
                    "batch_size": config.get("batch_size"),
                    "n_params": config.get("n_params"),
                },
            }
        )
    return rows


def best_row(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Return the row with the lowest average MAE for one model kind."""

    candidates = [row for row in rows if row["kind"] == kind]
    if not candidates:
        raise ValueError(f"No {kind} rows found.")
    return min(candidates, key=lambda row: row["mae_mean"])


def print_table(rows: list[dict[str, Any]], horizon_names: list[str]) -> None:
    """Print a terminal-friendly comparison table sorted by average MAE."""

    sorted_rows = sorted(rows, key=lambda row: row["mae_mean"])
    headers = ["model", "kind", "avg_MAE", *horizon_names]
    widths = [32, 10, 8, *([8] * len(horizon_names))]

    def fmt_row(values: list[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    print("\n=== Long-Term Multi-Horizon Result Summary ===")
    print(fmt_row(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in sorted_rows:
        label = row.get("label", row["name"])
        values = [
            label[: widths[0]],
            row["kind"],
            f"{row['mae_mean']:.4f}",
            *[f"{value:.4f}" for value in row["mae_by_horizon"]],
        ]
        print(fmt_row(values))


def write_summary(
    rows: list[dict[str, Any]],
    horizon_names: list[str],
    output_path: Path,
) -> None:
    """Write the best models, improvements, and all rows to one JSON summary."""

    best_baseline = best_row(rows, "baseline")
    best_neural = best_row(rows, "neural")
    six_h_idx = horizon_names.index("6h")
    ridge_or_best = best_baseline

    summary = {
        "horizon_names": horizon_names,
        "best_baseline": ridge_or_best,
        "best_neural": best_neural,
        "improvements": {
            "avg_mae_vs_best_baseline": ridge_or_best["mae_mean"] - best_neural["mae_mean"],
            "avg_mae_relative_vs_best_baseline": (
                (ridge_or_best["mae_mean"] - best_neural["mae_mean"]) / ridge_or_best["mae_mean"]
            ),
            "mae_6h_vs_best_baseline": (
                ridge_or_best["mae_by_horizon"][six_h_idx] - best_neural["mae_by_horizon"][six_h_idx]
            ),
        },
        "all_models": sorted(rows, key=lambda row: row["mae_mean"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {output_path}")


def plot_mae(
    rows: list[dict[str, Any]],
    horizon_names: list[str],
    output_path: Path,
) -> None:
    """Plot MAE by forecast horizon for the main baseline and best GRU."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for the figure. Install it with: python3 -m pip install matplotlib"
        ) from exc

    horizon_hours = [1, 3, 6, 12]
    best_neural = best_row(rows, "neural")
    # Keep the figure readable by showing only the strongest comparison lines.
    selected_names = {"current_speed", "ridge_alpha_1", best_neural["name"]}
    selected = [row for row in rows if row["name"] in selected_names]

    label_map = {
        "current_speed": "Current-speed persistence",
        "ridge_alpha_1": "Ridge baseline",
        best_neural["name"]: best_neural.get("label", best_neural["name"]),
    }

    plt.figure(figsize=(8.0, 4.8))
    for row in selected:
        label = label_map.get(row["name"], row.get("label", row["name"]))
        marker = "o" if row["kind"] == "baseline" else "s"
        linewidth = 2.4 if row["name"] == best_neural["name"] else 1.8
        plt.plot(horizon_hours, row["mae_by_horizon"], marker=marker, linewidth=linewidth, label=label)

    plt.title("PEMS08 long-term multi-horizon speed forecasting")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Test MAE (mph)")
    plt.xticks(horizon_hours, horizon_names)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Figure saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse report paths and output destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--neural-glob", default=DEFAULT_NEURAL_GLOB)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)
    return parser.parse_args()


def main() -> None:
    """Load reports, print the table, write summary JSON, and save the figure."""

    args = parse_args()
    if not args.baseline_report.exists():
        raise FileNotFoundError(f"Missing baseline report: {args.baseline_report}")

    baseline_rows, horizon_names = collect_baselines(args.baseline_report)
    neural_rows = collect_neural_reports(args.neural_glob)
    if not neural_rows:
        raise FileNotFoundError(f"No neural reports matched pattern: {args.neural_glob}")

    rows = baseline_rows + neural_rows
    print_table(rows, horizon_names)
    write_summary(rows, horizon_names, args.summary_out)
    plot_mae(rows, horizon_names, args.figure_out)

    best_baseline = best_row(rows, "baseline")
    best_neural = best_row(rows, "neural")
    six_h_idx = horizon_names.index("6h")
    print("\nBest baseline:", best_baseline["name"], f"avg MAE={best_baseline['mae_mean']:.4f}")
    print("Best neural:", best_neural.get("label", best_neural["name"]), f"avg MAE={best_neural['mae_mean']:.4f}")
    print(
        "6h improvement vs best baseline:",
        f"{best_baseline['mae_by_horizon'][six_h_idx] - best_neural['mae_by_horizon'][six_h_idx]:.4f} MAE",
    )

if __name__ == "__main__":
    main()
