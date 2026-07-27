"""M1: End-to-End Latency metrics (P50, P95, P99, mean)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_latency_metrics(events: list[dict[str, Any]]) -> dict[str, float]:
    """Compute P50, P95, P99 and mean latency from completed tasks (M1).

    Args:
        events: Event log dicts; each completed event must have a "latency" key.

    Returns:
        Dict with keys p50, p95, p99, mean (seconds). All zeros if no completed tasks.
    """
    latencies = [float(e["latency"]) for e in events if e["event_type"] == "completed"]
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.array(latencies, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }
