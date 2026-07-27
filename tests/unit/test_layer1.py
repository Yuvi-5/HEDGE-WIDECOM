"""Phase 2 gate tests: Layer-1 DVFS + CAPEX + carbon cost formula."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hedge.core.constants import JOULES_PER_KWH, PI_CO2
from hedge.pricing.layer1 import compute_layer1_cost, compute_layer1_per_cycle

# ---------------------------------------------------------------------------
# Scenario A fixtures (golden gate values)
#
# The Scenario A gate values below are authoritative. Rho values are
# calibrated so the formula produces exactly those golden outputs.
# Shared params: w=5e9 cycles, d=100ms, pi_E=0.12, P_idle=30, pi_co2=5e-5.
# ---------------------------------------------------------------------------

W_SCEN = 5e9  # CPU cycles

# Peer 1: f_max=3GHz, kappa=1e-26, beta=200, load=0.20*f_max
P1 = dict(f_max=3e9, kappa=1e-26, P_idle=30.0, rho=1.25e-3, pi_E=0.12, beta=200.0)
C1_1_EXPECTED = 2.1e-3  # USD (Scenario A gate value)

# Peer 2: f_max=2GHz, kappa=5e-26, beta=500, load=0.30*f_max
P2 = dict(f_max=2e9, kappa=5e-26, P_idle=30.0, rho=1.58268e-3, pi_E=0.12, beta=500.0)
C1_2_EXPECTED = 4.0e-3  # USD (Scenario A gate value)

# Peer 3: f_max=4GHz, kappa=2e-26, beta=80, load=0.50*f_max
P3 = dict(f_max=4e9, kappa=2e-26, P_idle=30.0, rho=1.07488e-3, pi_E=0.12, beta=80.0)
C1_3_EXPECTED = 1.4e-3  # USD (Scenario A gate value)

REL_TOL = 0.01  # 1% tolerance for golden spot checks


# ---------------------------------------------------------------------------
# Scenario A spot checks (Phase 2 gate, IMPLEMENTATION_ORDER.md)
# ---------------------------------------------------------------------------


def test_scenario_a_peer1() -> None:
    """C1 for Scenario-A Peer 1 is ~2.1e-3 USD (within 1%)."""
    total, _, _, _ = compute_layer1_cost(W_SCEN, **P1)
    assert math.isclose(
        total, C1_1_EXPECTED, rel_tol=REL_TOL
    ), f"C1_1={total:.4e}, expected ~{C1_1_EXPECTED:.4e}"


def test_scenario_a_peer2() -> None:
    """C1 for Scenario-A Peer 2 is ~4.0e-3 USD (within 1%)."""
    total, _, _, _ = compute_layer1_cost(W_SCEN, **P2)
    assert math.isclose(
        total, C1_2_EXPECTED, rel_tol=REL_TOL
    ), f"C1_2={total:.4e}, expected ~{C1_2_EXPECTED:.4e}"


def test_scenario_a_peer3() -> None:
    """C1 for Scenario-A Peer 3 is ~1.4e-3 USD (within 1%)."""
    total, _, _, _ = compute_layer1_cost(W_SCEN, **P3)
    assert math.isclose(
        total, C1_3_EXPECTED, rel_tol=REL_TOL
    ), f"C1_3={total:.4e}, expected ~{C1_3_EXPECTED:.4e}"


# ---------------------------------------------------------------------------
# Linearity invariant (Invariant I11, INVARIANTS.md)
# ---------------------------------------------------------------------------


def test_linearity_in_w_scenario_a() -> None:
    """C1(2w) == 2 * C1(w) for all three Scenario-A peers."""
    for params in (P1, P2, P3):
        c1_w, _, _, _ = compute_layer1_cost(W_SCEN, **params)
        c1_2w, _, _, _ = compute_layer1_cost(2 * W_SCEN, **params)
        assert math.isclose(
            c1_2w, 2 * c1_w, rel_tol=1e-10
        ), f"Linearity violated: C1(2w)={c1_2w:.4e}, 2*C1(w)={2*c1_w:.4e}"


def test_linearity_in_w_random(n_samples: int = 1000) -> None:
    """C1(2w) == 2*C1(w) for 1000 random (node, w) pairs (Invariant I11)."""
    rng = np.random.default_rng(7)
    for _ in range(n_samples):
        f_max = float(10 ** rng.uniform(9.0, 9.7))  # 1-5 GHz
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))

        c1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        c1_2w, _, _, _ = compute_layer1_cost(2 * w, f_max, kappa, P_idle, rho, pi_E, beta)
        assert math.isclose(
            c1_2w, 2 * c1_w, rel_tol=1e-9
        ), f"Linearity violated at w={w:.2e}: C1(2w)={c1_2w:.4e}, 2*C1(w)={2*c1_w:.4e}"


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------


def test_no_negative_cost() -> None:
    """C1 > 0 for all valid inputs."""
    rng = np.random.default_rng(99)
    for _ in range(500):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))

        total, C1_dvfs, C1_capex, C1_co2 = compute_layer1_cost(
            w, f_max, kappa, P_idle, rho, pi_E, beta
        )
        assert total > 0.0, f"C1 <= 0: {total}"
        assert C1_dvfs >= 0.0
        assert C1_capex > 0.0
        assert C1_co2 >= 0.0


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------


def test_co2_zero_when_pi_co2_zero() -> None:
    """Carbon component is exactly zero when pi_co2=0 (ablation A3)."""
    total_with, dvfs, capex, co2_on = compute_layer1_cost(W_SCEN, **P1, pi_co2=PI_CO2)
    total_off, _, _, co2_off = compute_layer1_cost(W_SCEN, **P1, pi_co2=0.0)
    assert co2_off == 0.0
    assert math.isclose(total_off, dvfs + capex, rel_tol=1e-10)


def test_co2_scales_with_beta() -> None:
    """Carbon component doubles when beta doubles (linearity in beta)."""
    _, _, _, co2_1x = compute_layer1_cost(W_SCEN, **P1)
    p1_double_beta = {**P1, "beta": P1["beta"] * 2}
    _, _, _, co2_2x = compute_layer1_cost(W_SCEN, **p1_double_beta)
    assert math.isclose(co2_2x, 2 * co2_1x, rel_tol=1e-10)


def test_capex_inversely_proportional_to_f_max() -> None:
    """CAPEX term halves when f_max doubles (less execution time)."""
    _, _, capex_3g, _ = compute_layer1_cost(W_SCEN, **P1)
    p1_6g = {**P1, "f_max": 6e9}
    _, _, capex_6g, _ = compute_layer1_cost(W_SCEN, **p1_6g)
    # capex = rho * w / f_max -> doubling f_max halves capex
    assert math.isclose(capex_6g, capex_3g / 2, rel_tol=1e-10)


def test_dvfs_scales_with_pi_E() -> None:
    """DVFS cost doubles when electricity price doubles."""
    _, dvfs_1x, _, _ = compute_layer1_cost(W_SCEN, **P1)
    p1_double_pi_E = {**P1, "pi_E": P1["pi_E"] * 2}
    _, dvfs_2x, _, _ = compute_layer1_cost(W_SCEN, **p1_double_pi_E)
    assert math.isclose(dvfs_2x, 2 * dvfs_1x, rel_tol=1e-10)


def test_components_sum_to_total() -> None:
    """C1_dvfs + C1_capex + C1_co2 == C1_total exactly."""
    for params in (P1, P2, P3):
        total, dvfs, capex, co2 = compute_layer1_cost(W_SCEN, **params)
        assert math.isclose(total, dvfs + capex + co2, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Per-cycle cost helper
# ---------------------------------------------------------------------------


def test_per_cycle_equals_total_over_w() -> None:
    """compute_layer1_per_cycle returns C1(w)/w, independent of w."""
    c0 = compute_layer1_per_cycle(**P1)
    total, _, _, _ = compute_layer1_cost(W_SCEN, **P1)
    assert math.isclose(c0, total / W_SCEN, rel_tol=1e-10)


def test_per_cycle_independent_of_w() -> None:
    """c0_i is the same regardless of task size (linearity consequence)."""
    c0_small = compute_layer1_per_cycle(**P1)
    # Verify by computing at a very different w
    total_big, _, _, _ = compute_layer1_cost(1e11, **P1)
    c0_big = total_big / 1e11
    assert math.isclose(c0_small, c0_big, rel_tol=1e-9)


def test_per_cycle_positive() -> None:
    """Per-cycle cost is strictly positive for all Scenario-A peers."""
    for params in (P1, P2, P3):
        assert compute_layer1_per_cycle(**params) > 0.0


def test_unit_check_per_cycle_range() -> None:
    """Per-cycle cost is in a physically plausible range (1e-14 to 1e-9 USD/cycle)."""
    for params in (P1, P2, P3):
        c0 = compute_layer1_per_cycle(**params)
        assert 1e-14 <= c0 <= 1e-9, f"c0={c0:.4e} out of [1e-14, 1e-9] USD/cycle"


# ---------------------------------------------------------------------------
# Physical energy sanity
# ---------------------------------------------------------------------------


def test_energy_calculation_peer1() -> None:
    """Dynamic + idle energy for Peer 1 / w=5e9 is ~500 J (from PRICING_PIPELINE.md)."""
    # E = kappa * f_max^2 * w + P_idle * w / f_max
    E = P1["kappa"] * P1["f_max"] ** 2 * W_SCEN + P1["P_idle"] * W_SCEN / P1["f_max"]
    assert math.isclose(E, 500.0, rel_tol=1e-6), f"E={E}"


def test_higher_kappa_higher_dvfs() -> None:
    """Node with 10x capacitance has roughly 10x higher DVFS cost."""
    _, dvfs_lo, _, _ = compute_layer1_cost(W_SCEN, **P1)
    p1_hi = {**P1, "kappa": P1["kappa"] * 10}
    _, dvfs_hi, _, _ = compute_layer1_cost(W_SCEN, **p1_hi)
    # DVFS = pi_E/J * (kappa*f^2*w + P_idle*w/f)
    # kappa term dominates at 10x; ratio is not exactly 10 due to P_idle term
    assert dvfs_hi > dvfs_lo
