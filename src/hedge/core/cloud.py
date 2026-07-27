"""CloudNode - the logical cloud fallback with infinite capacity."""

from __future__ import annotations

from dataclasses import dataclass

from hedge.core.constants import (
    BETA_CLOUD,
    F_MAX_CLOUD,
    KAPPA_CLOUD,
    MU_C,
    PI_E_CLOUD,
    RHO_CLOUD,
    TAU_C_MEAN,
)


@dataclass
class CloudNode:
    """Logical cloud fallback node.

    Single logical entity above the metro mesh. Never capacity-infeasible.
    RTT tau_c is sampled from N(TAU_C_MEAN, TAU_C_STD) at simulation init and
    stored here as tau_c. Not an EdgeSimPy entity; standalone dataclass.
    """

    f_max: float = F_MAX_CLOUD  # effective compute (Hz)
    kappa: float = KAPPA_CLOUD  # DVFS capacitance constant
    rho: float = RHO_CLOUD  # CAPEX rate (USD/s)
    beta: float = BETA_CLOUD  # carbon intensity (gCO2/kWh)
    pi_E: float = PI_E_CLOUD  # electricity price (USD/kWh)
    P_idle: float = 0.0  # hyperscale idle power not tracked
    mu_c: float = MU_C  # markup multiplier over Layer-1 cost
    tau_c: float = TAU_C_MEAN  # RTT (seconds), sampled per simulation run
