"""Layer-1 pricing: DVFS energy + CAPEX amortisation + carbon liability.

All three components are exactly linear in w (CPU cycles), so C1_i(w) = c0_i * w.
This linearity is the batching-arbitrage removal guarantee (Proposition 1).
"""

from __future__ import annotations

from hedge.core.constants import JOULES_PER_KWH, PI_CO2


def compute_layer1_cost(
    w: float,
    f_max: float,
    kappa: float,
    P_idle: float,
    rho: float,
    pi_E: float,
    beta: float,
    pi_co2: float = PI_CO2,
) -> tuple[float, float, float, float]:
    """Compute Layer-1 cost components for a task requiring w CPU cycles.

    Args:
        w: CPU workload (cycles).
        f_max: Maximum node CPU frequency (Hz). Node always runs at f_max.
        kappa: DVFS switched-capacitance constant (farads).
        P_idle: Idle baseline power draw (W).
        rho: CAPEX depreciation rate (USD/s).
        pi_E: Local electricity price (USD/kWh).
        beta: Local carbon intensity (gCO2/kWh).
        pi_co2: Internal carbon price (USD/g). Defaults to PI_CO2.

    Returns:
        (C1_total, C1_dvfs, C1_capex, C1_co2) all in USD.
        C1_total = C1_dvfs + C1_capex + C1_co2.
    """
    # Total energy to execute w cycles at f_max (Joules), Equation 12
    E: float = kappa * f_max**2 * w + P_idle * w / f_max

    # DVFS energy cost (Equation 29)
    C1_dvfs: float = pi_E / JOULES_PER_KWH * E  # USD

    # CAPEX amortisation (Equation 30): rho [USD/s] * execution_time [s]
    C1_capex: float = rho * w / f_max  # USD

    # Carbon liability (Equation 31)
    C1_co2: float = beta / JOULES_PER_KWH * E * pi_co2  # USD

    # Aggregate (Equation 32)
    C1_total: float = C1_dvfs + C1_capex + C1_co2  # USD

    return C1_total, C1_dvfs, C1_capex, C1_co2


def compute_layer1_per_cycle(
    f_max: float,
    kappa: float,
    P_idle: float,
    rho: float,
    pi_E: float,
    beta: float,
    pi_co2: float = PI_CO2,
) -> float:
    """Compute per-cycle Layer-1 cost c0_i = C1_i(w)/w (USD/cycle).

    Since C1 is linear in w, this constant is independent of task size.
    Used as the Bertrand floor in Layer-2.5: p_dagger_i >= c0_i.

    Args:
        f_max: Maximum node CPU frequency (Hz).
        kappa: DVFS switched-capacitance constant (farads).
        P_idle: Idle baseline power draw (W).
        rho: CAPEX depreciation rate (USD/s).
        pi_E: Local electricity price (USD/kWh).
        beta: Local carbon intensity (gCO2/kWh).
        pi_co2: Internal carbon price (USD/g). Defaults to PI_CO2.

    Returns:
        c0_i: Per-cycle cost (USD/cycle). Strictly positive.
    """
    total, _, _, _ = compute_layer1_cost(
        w=1.0,
        f_max=f_max,
        kappa=kappa,
        P_idle=P_idle,
        rho=rho,
        pi_E=pi_E,
        beta=beta,
        pi_co2=pi_co2,
    )
    return total  # USD/cycle (C1(1 cycle) / 1 cycle)
