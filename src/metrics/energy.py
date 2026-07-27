"""M15: Fleet Energy Per Task."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_fleet_energy_per_task(events: list[dict[str, Any]]) -> float:
    """Mean energy (Joules) consumed per completed task across all executors (M15).

    Reports separately from M7 (carbon) to distinguish energy efficiency from
    grid-mix effects. A node can be low-carbon (low beta) but energy-wasteful (high kappa).

    Args:
        events: Event log dicts; each completed event must have an "energy_joules" key.

    Returns:
        Mean energy per task in Joules; 0.0 if no completed tasks.
    """
    energy = [float(e["energy_joules"]) for e in events if e["event_type"] == "completed"]
    return float(np.mean(energy)) if energy else 0.0
