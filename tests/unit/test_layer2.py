"""Phase 4 gate tests: Layer-2 Stackelberg price and congestion markup."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hedge.core.constants import A_REF, B_SHAPE, ETA, LAMBDA_MAX
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price

# ---------------------------------------------------------------------------
# Scenario A fixtures (same peers as test_layer1.py)
# Load fractions (Scenario A): 0.20, 0.30, 0.50 of f_max respectively.
# ---------------------------------------------------------------------------

W_SCEN = 5e9  # CPU cycles

P1 = dict(f_max=3e9, kappa=1e-26, P_idle=30.0, rho=1.25e-3, pi_E=0.12, beta=200.0)
P2 = dict(f_max=2e9, kappa=5e-26, P_idle=30.0, rho=1.58268e-3, pi_E=0.12, beta=500.0)
P3 = dict(f_max=4e9, kappa=2e-26, P_idle=30.0, rho=1.07488e-3, pi_E=0.12, beta=80.0)

LOAD_1 = 0.20  # fraction of f_max for Peer 1
LOAD_2 = 0.30
LOAD_3 = 0.50

# Expected Layer-2 totals (USD), Scenario A gate values
P_TOTAL_1_EXPECTED = 0.079
P_TOTAL_2_EXPECTED = 0.114
P_TOTAL_3_EXPECTED = 0.072

REL_TOL = 0.01  # 1% tolerance for Scenario A spot checks


# ---------------------------------------------------------------------------
# Gate test 1-3: Scenario A spot checks (IMPLEMENTATION_ORDER.md)
# ---------------------------------------------------------------------------


def test_scenario_a_peer1() -> None:
    """p_star_1 * w ~= 0.079 USD for Scenario-A Peer 1 (within 1%)."""
    C1_1, _, _, _ = compute_layer1_cost(W_SCEN, **P1)
    l_hat_1 = LOAD_1 * P1["f_max"]
    m_1 = compute_markup(l_hat_1, P1["f_max"])
    p_star_1 = compute_stackelberg_price(C1_1, W_SCEN, m_1)
    total_price = p_star_1 * W_SCEN
    assert math.isclose(
        total_price, P_TOTAL_1_EXPECTED, rel_tol=REL_TOL
    ), f"p_star_1*w={total_price:.4e}, expected~{P_TOTAL_1_EXPECTED}"


def test_scenario_a_peer2() -> None:
    """p_star_2 * w ~= 0.114 USD for Scenario-A Peer 2 (within 1%)."""
    C1_2, _, _, _ = compute_layer1_cost(W_SCEN, **P2)
    l_hat_2 = LOAD_2 * P2["f_max"]
    m_2 = compute_markup(l_hat_2, P2["f_max"])
    p_star_2 = compute_stackelberg_price(C1_2, W_SCEN, m_2)
    total_price = p_star_2 * W_SCEN
    assert math.isclose(
        total_price, P_TOTAL_2_EXPECTED, rel_tol=REL_TOL
    ), f"p_star_2*w={total_price:.4e}, expected~{P_TOTAL_2_EXPECTED}"


def test_scenario_a_peer3() -> None:
    """p_star_3 * w ~= 0.072 USD for Scenario-A Peer 3 (within 1%)."""
    C1_3, _, _, _ = compute_layer1_cost(W_SCEN, **P3)
    l_hat_3 = LOAD_3 * P3["f_max"]
    m_3 = compute_markup(l_hat_3, P3["f_max"])
    p_star_3 = compute_stackelberg_price(C1_3, W_SCEN, m_3)
    total_price = p_star_3 * W_SCEN
    assert math.isclose(
        total_price, P_TOTAL_3_EXPECTED, rel_tol=REL_TOL
    ), f"p_star_3*w={total_price:.4e}, expected~{P_TOTAL_3_EXPECTED}"


# ---------------------------------------------------------------------------
# Gate test 4: markup bounds (Invariant I6)
# ---------------------------------------------------------------------------


def test_markup_bounds() -> None:
    """m_i in [1, 1+eta] for all l_hat in [0, f_max] and 1000 random inputs."""
    f_max = 3e9
    eta = ETA

    # Boundary conditions
    assert math.isclose(compute_markup(0.0, f_max), 1.0, rel_tol=1e-12)
    assert math.isclose(compute_markup(f_max, f_max), 1.0 + eta, rel_tol=1e-12)

    # Random l_hat values in [0, f_max]
    rng = np.random.default_rng(12)
    for _ in range(1000):
        l_hat = float(rng.uniform(0.0, f_max))
        m = compute_markup(l_hat, f_max)
        assert (
            1.0 <= m <= 1.0 + eta + 1e-12
        ), f"Markup out of bounds: m={m:.6f}, l_hat={l_hat:.3e}, f_max={f_max:.3e}"


# ---------------------------------------------------------------------------
# Gate test 5: price above Bertrand floor (C1/w)
# ---------------------------------------------------------------------------


def test_price_above_floor() -> None:
    """p_star >= C1/w for 1000 random valid (node, task, load) inputs."""
    rng = np.random.default_rng(55)
    for _ in range(1000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))
        l_hat = float(rng.uniform(0.0, f_max))

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(l_hat, f_max)
        p_star = compute_stackelberg_price(C1_w, w, m_i)
        floor = C1_w / w

        assert (
            p_star >= floor - 1e-20
        ), f"p_star={p_star:.4e} < floor={floor:.4e} (f_max={f_max:.2e}, C1/w={floor:.2e})"


# ---------------------------------------------------------------------------
# Gate test 6: batching invariance (p_star * 2w == 2 * p_star * w)
# ---------------------------------------------------------------------------


def test_batching_invariance() -> None:
    """p_star * (2w) == 2 * (p_star * w) within 1e-10 for Scenario-A peers."""
    for params, load_frac in [(P1, LOAD_1), (P2, LOAD_2), (P3, LOAD_3)]:
        f_max = params["f_max"]
        l_hat = load_frac * f_max
        m_i = compute_markup(l_hat, f_max)

        C1_w, _, _, _ = compute_layer1_cost(W_SCEN, **params)
        C1_2w, _, _, _ = compute_layer1_cost(2 * W_SCEN, **params)

        p_star_w = compute_stackelberg_price(C1_w, W_SCEN, m_i)
        p_star_2w = compute_stackelberg_price(C1_2w, 2 * W_SCEN, m_i)

        total_w = p_star_w * W_SCEN
        total_2w = p_star_2w * (2 * W_SCEN)

        assert math.isclose(
            total_2w, 2 * total_w, rel_tol=1e-10
        ), f"Batching violated: 2x_total={total_2w:.4e}, 2*1x_total={2*total_w:.4e}"


# ---------------------------------------------------------------------------
# Additional markup tests
# ---------------------------------------------------------------------------


def test_markup_zero_load() -> None:
    """m_i = 1.0 exactly when l_hat = 0 (idle node)."""
    assert compute_markup(0.0, F_MAX := 3e9) == 1.0


def test_markup_full_load() -> None:
    """m_i = 1 + eta exactly when l_hat = f_max (saturated node)."""
    f_max = 3e9
    m = compute_markup(f_max, f_max)
    assert math.isclose(m, 1.0 + ETA, rel_tol=1e-12)


def test_markup_monotone_in_load() -> None:
    """m_i is strictly increasing in l_hat (higher load -> higher markup)."""
    f_max = 3e9
    loads = np.linspace(0, f_max, 50)
    markups = [compute_markup(float(l), f_max) for l in loads]
    for i in range(len(markups) - 1):
        assert (
            markups[i] <= markups[i + 1] + 1e-15
        ), f"Markup not monotone at index {i}: {markups[i]:.6f} > {markups[i+1]:.6f}"


def test_markup_clamp_above_f_max() -> None:
    """l_hat > f_max is clipped to 1 + eta (not > 1+eta)."""
    f_max = 3e9
    m = compute_markup(2.0 * f_max, f_max)  # l_hat exceeds f_max
    assert math.isclose(m, 1.0 + ETA, rel_tol=1e-12)


def test_markup_clamp_below_zero() -> None:
    """l_hat < 0 is clipped to 1.0 (not < 1.0)."""
    f_max = 3e9
    m = compute_markup(-1e10, f_max)
    assert math.isclose(m, 1.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Additional Stackelberg price tests
# ---------------------------------------------------------------------------


def test_price_positive() -> None:
    """p_star > 0 for all valid inputs."""
    rng = np.random.default_rng(88)
    for _ in range(500):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))
        l_hat = float(rng.uniform(0.0, f_max))

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(l_hat, f_max)
        p_star = compute_stackelberg_price(C1_w, w, m_i)
        assert p_star > 0.0, f"p_star={p_star:.4e} not positive"


def test_price_increases_with_markup() -> None:
    """Higher m_i -> higher p_star (monotone in markup, Remark 9 in paper)."""
    C1_1, _, _, _ = compute_layer1_cost(W_SCEN, **P1)
    m_low = compute_markup(0.1 * P1["f_max"], P1["f_max"])
    m_high = compute_markup(0.9 * P1["f_max"], P1["f_max"])
    p_low = compute_stackelberg_price(C1_1, W_SCEN, m_low)
    p_high = compute_stackelberg_price(C1_1, W_SCEN, m_high)
    assert (
        p_high > p_low
    ), f"Higher markup did not raise price: p_low={p_low:.4e}, p_high={p_high:.4e}"


def test_price_increases_with_C1() -> None:
    """Higher C1 -> higher p_star (more expensive nodes charge more)."""
    m_i = 1.2
    p_cheap = compute_stackelberg_price(1.4e-3, W_SCEN, m_i)  # Peer 3 C1
    p_expensive = compute_stackelberg_price(4.0e-3, W_SCEN, m_i)  # Peer 2 C1
    assert p_expensive > p_cheap, f"p_expensive={p_expensive:.4e} not > p_cheap={p_cheap:.4e}"


def test_price_constant_in_w() -> None:
    """p_star is constant across different task sizes (batching invariant corollary)."""
    C1_w, _, _, _ = compute_layer1_cost(W_SCEN, **P1)
    C1_2w, _, _, _ = compute_layer1_cost(2 * W_SCEN, **P1)
    m_i = 1.2
    p1 = compute_stackelberg_price(C1_w, W_SCEN, m_i)
    p2 = compute_stackelberg_price(C1_2w, 2 * W_SCEN, m_i)
    assert math.isclose(
        p1, p2, rel_tol=1e-10
    ), f"p_star differs with task size: p(w)={p1:.6e}, p(2w)={p2:.6e}"


def test_scenario_a_peer3_wins() -> None:
    """Peer 3 has the lowest shaded quote (Scenario A outcome)."""
    results = []
    for params, load_frac in [(P1, LOAD_1), (P2, LOAD_2), (P3, LOAD_3)]:
        C1, _, _, _ = compute_layer1_cost(W_SCEN, **params)
        m = compute_markup(load_frac * params["f_max"], params["f_max"])
        p_star = compute_stackelberg_price(C1, W_SCEN, m)
        results.append((p_star * W_SCEN, params))

    # Peer 3 must have the lowest price
    min_price, min_params = min(results, key=lambda x: x[0])
    assert min_params is P3, f"Peer 3 did not win: prices={[f'{r[0]:.4f}' for r in results]}"
