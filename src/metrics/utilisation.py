"""M3 system utilisation, M4 utilisation variance, M11 Jain's Fairness Index."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_utilisation_variance(node_snapshots: list[dict[str, Any]]) -> float:
    """Variance of per-node utilisation fractions (M4). Lower = better load balance.

    Args:
        node_snapshots: Per-(node, tick) dicts each with l_current and f_max keys.

    Returns:
        Variance of l_current/f_max fractions; 0.0 if no data.
    """
    if not node_snapshots:
        return 0.0
    fracs = [float(snap["l_current"]) / max(float(snap["f_max"]), 1.0) for snap in node_snapshots]
    return float(np.var(fracs))


def compute_jains_fairness(node_snapshots: list[dict[str, Any]]) -> float:
    """Jain's Fairness Index on per-node utilisation fractions (M11).

    JFI = 1.0 means perfectly fair; near 1/N means one node carries all load.

    Args:
        node_snapshots: Per-(node, tick) dicts each with l_current and f_max keys.

    Returns:
        JFI in [0, 1]; 0.0 if no data.
    """
    if not node_snapshots:
        return 0.0
    fracs = [float(snap["l_current"]) / max(float(snap["f_max"]), 1.0) for snap in node_snapshots]
    n = len(fracs)
    sum_f = sum(fracs)
    sum_f2 = sum(f**2 for f in fracs)
    if sum_f2 <= 0.0:
        return 1.0
    return float(max(0.0, min(1.0, (sum_f**2) / (n * sum_f2))))
