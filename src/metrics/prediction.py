"""M14: Prediction RMSE for Kalman load forecast and EMA revenue forecast."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_prediction_rmse(node_snapshots: list[dict[str, Any]]) -> dict[str, float]:
    """RMSE of Kalman load forecast and EMA revenue forecast (M14).

    RMSE_l: load forecast error (cycles/s).
    RMSE_R: revenue forecast error (USD/RFQ). Falls back to 0.0 if R_realised absent.

    Args:
        node_snapshots: Per-(node, tick) dicts each with l_current, l_hat, R_hat keys.
                        R_realised is optional (falls back to R_hat when absent).

    Returns:
        Dict with keys RMSE_load (cycles/s) and RMSE_revenue (USD/RFQ).
    """
    if not node_snapshots:
        return {"RMSE_load": 0.0, "RMSE_revenue": 0.0}

    l_actual = np.array([float(s["l_current"]) for s in node_snapshots], dtype=np.float64)
    l_hat = np.array([float(s["l_hat"]) for s in node_snapshots], dtype=np.float64)
    r_actual = np.array(
        [float(s.get("R_realised", s["R_hat"])) for s in node_snapshots], dtype=np.float64
    )
    r_hat = np.array([float(s["R_hat"]) for s in node_snapshots], dtype=np.float64)

    rmse_l = float(np.sqrt(np.mean((l_actual - l_hat) ** 2)))
    rmse_r = float(np.sqrt(np.mean((r_actual - r_hat) ** 2)))
    return {"RMSE_load": rmse_l, "RMSE_revenue": rmse_r}
