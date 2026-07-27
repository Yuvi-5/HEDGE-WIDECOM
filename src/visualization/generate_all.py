"""Generate all paper figures from a completed experiment results directory.

Usage:
    python src/visualization/generate_all.py \\
        --results-dir outputs/paper_results/ \\
        --output-dir outputs/paper_results/figures/ \\
        --dpi 300 --format png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_dir = _Path(__file__).parent.parent
if str(_src_dir) not in _sys.path:
    _sys.path.insert(0, str(_src_dir))

import argparse
import csv
from pathlib import Path
from typing import Any

from loguru import logger

from visualization.ablation_plots import (
    plot_ablation_bars,
    plot_acceptance_vs_kmax,
    plot_rejection_vs_kmax,
)
from visualization.cost_plots import plot_carbon_vs_kmax, plot_cost_vs_kmax
from visualization.latency_plots import plot_latency_cdf, plot_latency_vs_kmax


def _load_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    """Load metrics.csv as a list of row dicts.

    Args:
        metrics_csv: Path to metrics CSV file.

    Returns:
        List of dicts; empty list if file not found.
    """
    if not metrics_csv.exists():
        logger.warning(f"metrics.csv not found: {metrics_csv}")
        return []
    with metrics_csv.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _has_sweep_col(rows: list[dict[str, Any]], col: str) -> bool:
    """Return True if col exists in rows and has more than one distinct value."""
    if not rows or col not in rows[0]:
        return False
    vals = {r.get(col) for r in rows}
    return len(vals) > 1


def generate_all_figures(
    results_dir: Path,
    output_dir: Path,
    dpi: int = 300,
    fmt: str = "png",
) -> list[Path]:
    """Generate all figures from a results directory.

    Reads metrics.csv and generates plots appropriate to the data:
    - If K_max sweep detected: rejection, latency, cost, acceptance vs K_max
    - Otherwise: ablation bar chart of available metrics

    Args:
        results_dir: Directory containing metrics.csv.
        output_dir: Destination directory for figure files.
        dpi: Output DPI (default 300 for IEEE publication).
        fmt: File format (default 'png').

    Returns:
        List of paths to generated figure files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = results_dir / "metrics.csv"
    rows = _load_rows(metrics_csv)

    if not rows:
        logger.warning(f"No data in {metrics_csv}; generating placeholder figure")
        return [_placeholder_figure(output_dir, fmt, dpi)]

    generated: list[Path] = []

    def _fig(name: str) -> Path:
        return output_dir / f"{name}.{fmt}"

    # Detect sweep parameter
    sweep_col = "K_max"
    for candidate in ["K_max", "lambda_max", "theta_risk"]:
        if _has_sweep_col(rows, candidate):
            sweep_col = candidate
            break

    if _has_sweep_col(rows, sweep_col):
        # Sweep figures
        generated.append(
            plot_rejection_vs_kmax(rows, _fig("m6_rejection_vs_kmax"), sweep_col=sweep_col, dpi=dpi)
        )
        generated.append(
            plot_latency_vs_kmax(rows, _fig("m1_latency_vs_kmax"), sweep_col=sweep_col, dpi=dpi)
        )
        generated.append(
            plot_cost_vs_kmax(rows, _fig("m2_cost_vs_kmax"), sweep_col=sweep_col, dpi=dpi)
        )
        generated.append(
            plot_carbon_vs_kmax(rows, _fig("m7_carbon_vs_kmax"), sweep_col=sweep_col, dpi=dpi)
        )
        generated.append(
            plot_acceptance_vs_kmax(
                rows, _fig("m6_acceptance_vs_kmax"), sweep_col=sweep_col, dpi=dpi
            )
        )
    else:
        # Non-sweep: ablation bar chart
        generated.append(plot_ablation_bars(rows, _fig("ablation_comparison"), dpi=dpi))

    # Latency CDF always generated when latency data exists
    lat_col = next(
        (c for c in ["M1_mean_latency_s", "M3_p90_latency_s"] if rows and c in rows[0]),
        None,
    )
    if lat_col and _has_sweep_col(rows, sweep_col):
        generated.append(
            plot_latency_cdf(
                rows, _fig("m1_latency_cdf"), metric_col=lat_col, group_col=sweep_col, dpi=dpi
            )
        )

    logger.info(f"Generated {len(generated)} figure(s) in {output_dir}")
    return generated


def _placeholder_figure(output_dir: Path, fmt: str, dpi: int) -> Path:
    """Write a minimal placeholder figure when no data is available."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("HEDGE results (no data)", fontsize=8)
    out = output_dir / f"placeholder.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved placeholder figure: {out}")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate HEDGE paper figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        dest="results_dir",
        help="Directory containing metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Destination for figures (default: results_dir/figures/)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", type=str, default="png", dest="fmt")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "figures")
    generated = generate_all_figures(
        results_dir=args.results_dir,
        output_dir=output_dir,
        dpi=args.dpi,
        fmt=args.fmt,
    )
    logger.info(f"Done: {len(generated)} figure(s) written to {output_dir}")


if __name__ == "__main__":
    main()
