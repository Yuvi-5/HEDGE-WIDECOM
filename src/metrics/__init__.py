"""HEDGE Metrics M1-M16: collector, statistical utilities, and exports."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from hedge.core.constants import DELTA_Q
from metrics.carbon import compute_carbon_per_task
from metrics.completion import (
    compute_completion_rate,
    compute_market_revenue,
    compute_quote_acceptance_rate,
)
from metrics.convergence import compute_stackelberg_convergence
from metrics.cost import compute_mean_cost
from metrics.energy import compute_fleet_energy_per_task
from metrics.latency import compute_latency_metrics
from metrics.myerson import compute_myerson_efficiency
from metrics.offload import compute_primary_offload_fraction
from metrics.prediction import compute_prediction_rmse
from metrics.rejection import compute_rejection_rate
from metrics.resale import compute_err
from metrics.sla import compute_tsfr
from metrics.utilisation import compute_jains_fairness, compute_utilisation_variance


def wilcoxon_test(
    hedge_values: list[float],
    baseline_values: list[float],
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test between HEDGE and a baseline.

    Args:
        hedge_values: Per-seed metric values for HEDGE.
        baseline_values: Per-seed metric values for the baseline.

    Returns:
        Dict with keys: statistic (float), p_value (float), significant (bool).
    """
    stat, p_value = stats.wilcoxon(hedge_values, baseline_values)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: Observed scalar values (e.g., per-seed metric results).
        confidence: Confidence level (default 0.95 = 95% CI).
        n_boot: Number of bootstrap resamples.
        rng: Seeded Generator for reproducibility; if None, one is derived from `seed`
             so results are reproducible even when the caller passes no generator.
        seed: Fallback seed used only when rng is None.

    Returns:
        (lower, upper) bounds of the CI.
    """
    gen = rng if rng is not None else np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    boot_means: list[float] = [
        float(np.mean(gen.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)
    ]
    alpha = 1.0 - confidence
    return (
        float(np.percentile(boot_means, 100.0 * alpha / 2.0)),
        float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
    )


class MetricsCollector:
    """Central collector for all 16 HEDGE metrics.

    Usage:
        collector = MetricsCollector()
        collector.record_event({"event_type": "completed", "latency": 0.05, ...})
        collector.record_snapshot({"time": 0.1, "node_id": "n0", "l_current": 2e9, ...})
        collector.record_quote({"time": 0.2, "p_dagger_ref": 1.5e-11})
        results = collector.compute_all()
    """

    def __init__(
        self,
        burst_start: float = -1.0,
        burst_duration: float = 0.0,
    ) -> None:
        """Initialise collector with optional burst window parameters.

        Args:
            burst_start: Simulation time (s) when Pareto burst begins. Use -1.0 for
                         steady-state (no burst), which makes M12 return 0.0.
            burst_duration: Duration of the burst in seconds.
        """
        self.events: list[dict[str, Any]] = []
        self.node_snapshots: list[dict[str, Any]] = []
        self.quote_history: list[dict[str, Any]] = []
        self._burst_start = burst_start
        self._burst_duration = burst_duration

    def set_burst_params(self, burst_start: float, burst_duration: float) -> None:
        """Update burst window parameters for M12 computation.

        Args:
            burst_start: Simulation time when burst begins (seconds).
            burst_duration: Duration of the burst (seconds).
        """
        self._burst_start = burst_start
        self._burst_duration = burst_duration

    def record_event(self, event: dict[str, Any]) -> None:
        """Record a task completion or rejection event.

        Args:
            event: Dict with at minimum "event_type" ("completed" or "rejected").
                   Completed events should also include: latency, price_paid,
                   executor_id, carbon_g, energy_joules, is_resale, arrival_time, deadline.
        """
        self.events.append(event)

    def record_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Record a per-node state snapshot for utilisation and prediction metrics.

        Args:
            snapshot: Dict with keys: time (s), node_id (str), l_current (Hz),
                      f_max (Hz), l_hat (Hz), R_hat (USD/RFQ).
                      R_realised (USD/RFQ) is optional for M14.
        """
        self.node_snapshots.append(snapshot)

    def record_quote(self, quote: dict[str, Any]) -> None:
        """Record a standing-quote broadcast for convergence tracking (M8).

        Args:
            quote: Dict with keys: time (s), p_dagger_ref (USD/cycle).
        """
        self.quote_history.append(quote)

    def _compute_m3(self) -> float:
        """Compute M3 system utilisation from per-node snapshots grouped by time."""
        if not self.node_snapshots:
            return 0.0
        tick_load: dict[float, float] = defaultdict(float)
        tick_cap: dict[float, float] = defaultdict(float)
        for snap in self.node_snapshots:
            t = float(snap.get("time", 0.0))
            tick_load[t] += float(snap["l_current"])
            tick_cap[t] += float(snap["f_max"])
        utils = [tick_load[t] / max(tick_cap[t], 1.0) for t in tick_load]
        return float(np.mean(utils)) if utils else 0.0

    def compute_all(self) -> dict[str, Any]:
        """Compute all 17 metrics and return them keyed by 'M1' through 'M17'.

        Returns:
            Dict where values are floats (M2-M7, M9-M13, M15-M17) or dicts
            (M1: latency percentiles, M8: convergence, M14: RMSE pair).
        """
        return {
            "M1": compute_latency_metrics(self.events),
            "M2": compute_mean_cost(self.events),
            "M3": self._compute_m3(),
            "M4": compute_utilisation_variance(self.node_snapshots),
            "M5": compute_market_revenue(self.events),
            "M6": compute_rejection_rate(self.events),
            "M7": compute_carbon_per_task(self.events),
            "M8": compute_stackelberg_convergence(self.quote_history),
            "M9": compute_quote_acceptance_rate(self.events),
            "M10": compute_completion_rate(self.events),
            "M11": compute_jains_fairness(self.node_snapshots),
            "M12": compute_tsfr(
                self.events, self._burst_start, self._burst_duration, delta_q=DELTA_Q
            ),
            "M13": compute_err(self.events),
            "M14": compute_prediction_rmse(self.node_snapshots),
            "M15": compute_fleet_energy_per_task(self.events),
            "M16": compute_myerson_efficiency(self.events, self.node_snapshots),
            "M17": compute_primary_offload_fraction(self.events),
        }

    def export_csv(self, output_path: Path | str) -> None:
        """Export flat metric summary to CSV (one row, one column per scalar metric).

        Dict-valued metrics (M1, M8, M14) are flattened with underscore-joined keys.

        Args:
            output_path: Destination CSV file path.
        """
        results = self.compute_all()
        flat: dict[str, float] = {}
        for key, val in results.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    flat_key = f"{key}_{sub_key}"
                    num = float(sub_val) if not isinstance(sub_val, bool) else float(sub_val)
                    flat[flat_key] = num if math.isfinite(num) else 0.0
            else:
                num = float(val)
                flat[key] = num if math.isfinite(num) else 0.0

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)


__all__ = [
    "MetricsCollector",
    "wilcoxon_test",
    "bootstrap_ci",
    "compute_latency_metrics",
    "compute_mean_cost",
    "compute_utilisation_variance",
    "compute_jains_fairness",
    "compute_market_revenue",
    "compute_rejection_rate",
    "compute_carbon_per_task",
    "compute_stackelberg_convergence",
    "compute_quote_acceptance_rate",
    "compute_completion_rate",
    "compute_tsfr",
    "compute_err",
    "compute_prediction_rmse",
    "compute_fleet_energy_per_task",
    "compute_myerson_efficiency",
    "compute_primary_offload_fraction",
]
