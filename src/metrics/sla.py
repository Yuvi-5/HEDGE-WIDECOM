"""M12: Transient SLA Failure Rate (TSFR) during Pareto burst windows."""

from __future__ import annotations

from typing import Any

from hedge.core.constants import DELTA_Q


def compute_tsfr(
    events: list[dict[str, Any]],
    burst_start: float,
    burst_duration: float,
    delta_q: float = DELTA_Q,
) -> float:
    """Fraction of tasks that miss deadline during and immediately after a burst (M12).

    Measured over the burst window [burst_start, burst_start + burst_duration + delta_q].
    Returns 0.0 when burst_start is set below all event arrival times (steady-state mode).

    Args:
        events: Event log dicts; each must have "arrival_time", "latency", "deadline".
        burst_start: Simulation time (seconds) when the Pareto burst begins.
        burst_duration: Duration of the burst in seconds.
        delta_q: One additional quote-window grace period after the burst (seconds).

    Returns:
        TSFR in [0, 1]; 0.0 if no tasks fall in the burst window.
    """
    window_end = burst_start + burst_duration + delta_q
    burst_tasks = [
        e for e in events if burst_start <= float(e.get("arrival_time", -1.0)) <= window_end
    ]
    violated = [
        e
        for e in burst_tasks
        if float(e.get("latency", 0.0)) > float(e.get("deadline", float("inf")))
    ]
    return float(len(violated)) / max(1, len(burst_tasks)) if burst_tasks else 0.0
