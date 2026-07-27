"""M5 Market Revenue, M9 Quote Acceptance Rate, M10 Task Completion Rate."""

from __future__ import annotations

from typing import Any

from metrics.rejection import compute_rejection_rate


def compute_market_revenue(events: list[dict[str, Any]]) -> float:
    """Total payments received by edge sellers (USD), excluding cloud revenue (M5).

    Args:
        events: Event log dicts; each completed event must have "price_paid" and "executor_id".

    Returns:
        Total edge revenue in USD; 0.0 if no edge-completed tasks.
    """
    edge_payments = [
        float(e["price_paid"])
        for e in events
        if e["event_type"] == "completed" and str(e.get("executor_id", "")) != "cloud"
    ]
    return float(sum(edge_payments))


def compute_quote_acceptance_rate(events: list[dict[str, Any]]) -> float:
    """Fraction of completed tasks served by edge peers, not cloud (M9).

    Args:
        events: Event log dicts.

    Returns:
        Edge service fraction in [0, 1].
    """
    completed = [e for e in events if e["event_type"] == "completed"]
    edge_served = [e for e in completed if str(e.get("executor_id", "")) != "cloud"]
    return float(len(edge_served)) / max(1, len(completed))


def compute_completion_rate(events: list[dict[str, Any]]) -> float:
    """Fraction of tasks that complete successfully; complement of M6 (M10).

    Args:
        events: Event log dicts.

    Returns:
        Completion rate in [0, 1].
    """
    return 1.0 - compute_rejection_rate(events)
