"""M6: Task Rejection Rate."""

from __future__ import annotations

from typing import Any


def compute_rejection_rate(events: list[dict[str, Any]]) -> float:
    """Fraction of tasks that were unmatched (M6).

    Includes all-reject, deadline-infeasible, and budget-infeasible rejections.

    Args:
        events: Event log dicts; event_type must be "completed" or "rejected".

    Returns:
        Rejection rate in [0, 1]; 0.0 if no tasks.
    """
    total = sum(1 for e in events if e["event_type"] in ("completed", "rejected"))
    rejected = sum(1 for e in events if e["event_type"] == "rejected")
    return float(rejected) / max(1, total)
