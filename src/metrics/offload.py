"""M17: Primary Inter-Edge Offload Fraction.

The horizontal-market signal: the fraction of served (completed) tasks that
execute on a different edge node than the one they arrived at, via the
primary RFQ/AFGM match -- distinct from M13 (edge-resale ratio), which counts
only the narrower, second-order wholesale-resale layer (Section 4).
"""

from __future__ import annotations

from typing import Any


def compute_primary_offload_fraction(events: list[dict[str, Any]]) -> float:
    """Fraction of completed tasks executed away from their home/arrival node (M17).

    Args:
        events: Event log dicts; completed events must include "executor_id"
                and "home_node_id". Rejected and cloud-executed events are
                excluded from the offload count (cloud is not a peer edge node).

    Returns:
        Offload fraction in [0, 1]; 0.0 if no completed tasks.
    """
    completed = [e for e in events if e.get("event_type") == "completed"]
    if not completed:
        return 0.0
    offloaded = sum(
        1
        for e in completed
        if e.get("executor_id") not in ("", "cloud")
        and e.get("executor_id") != e.get("home_node_id")
    )
    return float(offloaded) / len(completed)
