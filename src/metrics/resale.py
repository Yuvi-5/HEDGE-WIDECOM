"""M13: Edge-Resale Ratio (ERR)."""

from __future__ import annotations

from typing import Any


def compute_err(events: list[dict[str, Any]]) -> float:
    """Fraction of tasks cleared via edge-to-edge wholesale resale (GC play) (M13).

    ERR > 0 under heterogeneous load validates the wholesale-market claim.
    Near-zero ERR under heavy heterogeneous load would invalidate that claim.

    Args:
        events: Event log dicts; completed events may have an "is_resale" bool key.

    Returns:
        ERR in [0, 1]; 0.0 if no completed tasks.
    """
    completed = [e for e in events if e["event_type"] == "completed"]
    resale = [e for e in completed if bool(e.get("is_resale", False))]
    return float(len(resale)) / max(1, len(completed))
