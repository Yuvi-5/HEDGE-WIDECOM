"""Phase 5 gate tests: Layer-2.5 SPA shading, aggressiveness, and cascade monotonicity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hedge.core.constants import A_REF, B_SHAPE, ETA, LAMBDA_MAX, R_REF
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price
from hedge.pricing.layer2_5 import (
    compute_aggressiveness,
    compute_competitiveness_signal,
    compute_spa_price,
)

# ---------------------------------------------------------------------------
# Scenario A fixtures (identical to test_layer2.py)
# Load fractions (Scenario A): 0.20, 0.30, 0.50 of f_max respectively.
# R_hat for Scenario A Peer 1: 0.10 * R_REF, giving lambda_SPA = 0.5*0.8*0.9 = 0.36
# ---------------------------------------------------------------------------

W_SCEN = 5e9  # CPU cycles

P1 = dict(f_max=3e9, kappa=1e-26, P_idle=30.0, rho=1.25e-3, pi_E=0.12, beta=200.0)
P2 = dict(f_max=2e9, kappa=5e-26, P_idle=30.0, rho=1.58268e-3, pi_E=0.12, beta=500.0)
P3 = dict(f_max=4e9, kappa=2e-26, P_idle=30.0, rho=1.07488e-3, pi_E=0.12, beta=80.0)

LOAD_1 = 0.20
LOAD_2 = 0.30
LOAD_3 = 0.50

# R_hat for Scenario A: well below R_ref so (1 - R_hat/R_ref) is large.
# From PRICING_PIPELINE.md: lambda_SPA_1 = 0.5 * (1-0.20) * (1-0.10) = 0.36
# -> R_hat = 0.10 * R_REF = 0.10 USD/RFQ
R_HAT_SCEN_A = 0.10 * R_REF

REL_TOL = 0.01  # 1% tolerance for gate checks


def _peer_stackelberg(params: dict, load_frac: float, w: float = W_SCEN) -> float:
    """Return p_star per-cycle price (USD/cycle) for a given peer configuration."""
    C1, _, _, _ = compute_layer1_cost(w, **params)
    m = compute_markup(load_frac * params["f_max"], params["f_max"])
    return compute_stackelberg_price(C1, w, m)


# ---------------------------------------------------------------------------
# Gate test 1: Scenario A Peer 1 shaded price (IMPLEMENTATION_ORDER.md)
# ---------------------------------------------------------------------------


def test_scenario_a_peer1_shaded() -> None:
    """p_dagger_1 * w ~= 0.077 USD for Scenario-A Peer 1 (within 1%)."""
    # Layer-2 prices for all three peers (USD/cycle)
    p_star_1 = _peer_stackelberg(P1, LOAD_1)
    p_star_2 = _peer_stackelberg(P2, LOAD_2)
    p_star_3 = _peer_stackelberg(P3, LOAD_3)

    # Competitiveness: Peer 1 competes against Peers 2 and 3
    delta_comp_1 = compute_competitiveness_signal(p_star_1, [p_star_2, p_star_3])

    # Aggressiveness for Peer 1 (R_hat well below R_ref)
    lambda_SPA_1 = compute_aggressiveness(
        l_hat=LOAD_1 * P1["f_max"],
        f_max=P1["f_max"],
        R_hat=R_HAT_SCEN_A,
        R_ref=R_REF,
        lambda_max=LAMBDA_MAX,
    )

    # SPA price for Peer 1
    C1_1, _, _, _ = compute_layer1_cost(W_SCEN, **P1)
    p_dagger_1 = compute_spa_price(C1_1, W_SCEN, p_star_1, lambda_SPA_1, delta_comp_1)
    total = p_dagger_1 * W_SCEN

    assert math.isclose(
        total, 0.077, rel_tol=REL_TOL
    ), f"p_dagger_1*w={total:.4e}, expected~0.077 USD"


# ---------------------------------------------------------------------------
# Gate test 2 reference: Bertrand floor (also covered in invariants file)
# ---------------------------------------------------------------------------


def test_bertrand_floor_1k() -> None:
    """p_dagger >= C1/w for 1,000 random (node, task, load, revenue) inputs."""
    rng = np.random.default_rng(31)
    for _ in range(1000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))
        l_hat = float(rng.uniform(0.0, f_max))
        R_hat = float(rng.uniform(0.0, 5.0))

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(l_hat, f_max)
        p_star = compute_stackelberg_price(C1_w, w, m_i)
        lambda_SPA = compute_aggressiveness(l_hat, f_max, R_hat)
        delta_comp = float(rng.uniform(0.0, 0.99))
        p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA, delta_comp)
        floor = C1_w / w

        assert (
            p_dagger >= floor - 1e-20
        ), f"Bertrand floor violated: p_dagger={p_dagger:.4e}, floor={floor:.4e}"


# ---------------------------------------------------------------------------
# Gate test 3: cascade monotonicity (Invariant I5)
# ---------------------------------------------------------------------------


def test_cascade_monotonicity() -> None:
    """C1/w <= p_dagger <= p_star for 1,000 random inputs."""
    rng = np.random.default_rng(77)
    for _ in range(1000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))
        l_hat = float(rng.uniform(0.0, f_max))
        R_hat = float(rng.uniform(0.0, 5.0))

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(l_hat, f_max)
        p_star = compute_stackelberg_price(C1_w, w, m_i)
        lambda_SPA = compute_aggressiveness(l_hat, f_max, R_hat)
        delta_comp = float(rng.uniform(0.0, 0.99))
        p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA, delta_comp)
        floor = C1_w / w

        assert (
            p_dagger >= floor - 1e-20
        ), f"Lower cascade violated: p_dagger={p_dagger:.4e} < C1/w={floor:.4e}"
        assert (
            p_dagger <= p_star + 1e-20
        ), f"Upper cascade violated: p_dagger={p_dagger:.4e} > p_star={p_star:.4e}"


# ---------------------------------------------------------------------------
# Gate test 4: no negative aggressiveness (Invariant I7)
# ---------------------------------------------------------------------------


def test_no_negative_aggressiveness() -> None:
    """lambda_SPA >= 0 for all inputs including adversarial ones."""
    f_max = 3e9
    rng = np.random.default_rng(44)

    # Random valid inputs
    for _ in range(1000):
        l_hat = float(rng.uniform(0.0, f_max))
        R_hat = float(rng.uniform(0.0, 20.0))
        lam = compute_aggressiveness(l_hat, f_max, R_hat)
        assert lam >= 0.0, f"Negative aggressiveness: lambda_SPA={lam:.4e}"

    # Adversarial: R_hat >> R_ref (should clamp to 0, not go negative)
    assert compute_aggressiveness(0.0, f_max, R_hat=100.0) == 0.0
    assert compute_aggressiveness(0.0, f_max, R_hat=R_REF) == 0.0

    # Adversarial: l_hat > f_max (filter clamps, but defensive check)
    assert compute_aggressiveness(2.0 * f_max, f_max, R_hat=0.0) == 0.0


# ---------------------------------------------------------------------------
# Gate test 5: no shading at full load or high revenue (Invariant I7 boundary)
# ---------------------------------------------------------------------------


def test_no_shading_at_full_load() -> None:
    """lambda_SPA = 0 when l_hat = f_max (saturated node cannot shade further)."""
    f_max = 3e9
    lam = compute_aggressiveness(l_hat=f_max, f_max=f_max, R_hat=0.0)
    assert lam == 0.0, f"Expected 0.0 at full load, got {lam:.4e}"


def test_no_shading_at_high_revenue() -> None:
    """lambda_SPA = 0 when R_hat >= R_ref (profitable node does not shade)."""
    f_max = 3e9
    lam_at_ref = compute_aggressiveness(l_hat=0.0, f_max=f_max, R_hat=R_REF)
    assert lam_at_ref == 0.0, f"Expected 0.0 at R_hat == R_ref, got {lam_at_ref:.4e}"

    lam_above_ref = compute_aggressiveness(l_hat=0.0, f_max=f_max, R_hat=2.0 * R_REF)
    assert lam_above_ref == 0.0, f"Expected 0.0 at R_hat > R_ref, got {lam_above_ref:.4e}"


# ---------------------------------------------------------------------------
# Additional competitiveness signal tests
# ---------------------------------------------------------------------------


def test_delta_comp_zero_when_cheapest() -> None:
    """delta_comp = 0 when node i has the lowest price among all peers."""
    p_self = 0.05e-9  # cheapest
    peers = [0.07e-9, 0.09e-9]
    delta = compute_competitiveness_signal(p_self, peers)
    assert delta == 0.0, f"Expected 0.0 when cheapest, got {delta:.4e}"


def test_delta_comp_positive_when_undercut() -> None:
    """delta_comp > 0 when a peer offers a lower price."""
    p_self = 0.09e-9
    peers = [0.07e-9, 0.11e-9]
    delta = compute_competitiveness_signal(p_self, peers)
    expected = (0.09e-9 - 0.07e-9) / 0.09e-9
    assert math.isclose(
        delta, expected, rel_tol=1e-12
    ), f"delta_comp={delta:.6e}, expected={expected:.6e}"


def test_delta_comp_zero_when_no_peers() -> None:
    """delta_comp = 0 when there are no peer quotes (monopoly scenario)."""
    delta = compute_competitiveness_signal(0.09e-9, [])
    assert delta == 0.0


def test_delta_comp_bounded_below_one() -> None:
    """delta_comp is always < 1 by construction (p_self > 0, min_peer >= 0)."""
    rng = np.random.default_rng(21)
    for _ in range(500):
        p_self = float(rng.uniform(1e-13, 1e-10))
        n_peers = int(rng.integers(1, 6))
        # Peers may be lower or higher
        peers = [float(rng.uniform(0.5e-13, 2e-10)) for _ in range(n_peers)]
        delta = compute_competitiveness_signal(p_self, peers)
        assert 0.0 <= delta < 1.0 + 1e-12, f"delta_comp out of [0,1): {delta:.4e}"


def test_delta_comp_only_uses_minimum_peer() -> None:
    """delta_comp depends only on the minimum peer quote, not all peers."""
    p_self = 0.10e-9
    p_min = 0.06e-9
    # Adding a higher peer should not change delta_comp
    delta_one_peer = compute_competitiveness_signal(p_self, [p_min])
    delta_two_peers = compute_competitiveness_signal(p_self, [p_min, 0.20e-9])
    assert math.isclose(delta_one_peer, delta_two_peers, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Additional aggressiveness tests
# ---------------------------------------------------------------------------


def test_aggressiveness_maximum_at_idle_low_revenue() -> None:
    """lambda_SPA = lambda_max when l_hat = 0 and R_hat = 0 (max shading scenario)."""
    f_max = 3e9
    lam = compute_aggressiveness(l_hat=0.0, f_max=f_max, R_hat=0.0, R_ref=R_REF)
    assert math.isclose(
        lam, LAMBDA_MAX, rel_tol=1e-12
    ), f"Expected LAMBDA_MAX={LAMBDA_MAX}, got {lam:.4e}"


def test_aggressiveness_scenario_a_peer1() -> None:
    """lambda_SPA_1 = 0.36 for Scenario A Peer 1 (gate value)."""
    lam = compute_aggressiveness(
        l_hat=LOAD_1 * P1["f_max"],
        f_max=P1["f_max"],
        R_hat=R_HAT_SCEN_A,
        R_ref=R_REF,
        lambda_max=LAMBDA_MAX,
    )
    assert math.isclose(lam, 0.36, rel_tol=1e-10), f"lambda_SPA_1={lam:.6f}, expected=0.36"


def test_aggressiveness_bounds_1k() -> None:
    """lambda_SPA in [0, lambda_max] for 1,000 random inputs (Invariant I7)."""
    f_max = 3e9
    rng = np.random.default_rng(99)
    for _ in range(1000):
        l_hat = float(rng.uniform(0.0, 1.5 * f_max))  # intentionally beyond f_max
        R_hat = float(rng.uniform(-1.0, 5.0))  # intentionally negative R_hat
        lam = compute_aggressiveness(l_hat, f_max, R_hat)
        assert (
            0.0 <= lam <= LAMBDA_MAX + 1e-12
        ), f"I7 violated: lambda_SPA={lam:.4e} outside [0, {LAMBDA_MAX}]"


# ---------------------------------------------------------------------------
# Additional SPA price tests
# ---------------------------------------------------------------------------


def test_spa_price_equals_p_star_when_no_shading() -> None:
    """p_dagger = p_star when lambda_SPA = 0 (no competitive pressure)."""
    C1_w = 2.1e-3
    w = W_SCEN
    p_star = compute_stackelberg_price(C1_w, w, 1.2)
    p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA=0.0, delta_comp=0.5)
    assert math.isclose(
        p_dagger, p_star, rel_tol=1e-12
    ), f"Expected p_dagger==p_star when lambda=0: p_dagger={p_dagger:.4e}, p_star={p_star:.4e}"


def test_spa_price_equals_p_star_when_no_competition() -> None:
    """p_dagger = p_star when delta_comp = 0 (node already cheapest)."""
    C1_w = 2.1e-3
    w = W_SCEN
    p_star = compute_stackelberg_price(C1_w, w, 1.2)
    p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA=0.4, delta_comp=0.0)
    assert math.isclose(p_dagger, p_star, rel_tol=1e-12)


def test_spa_floor_binds_at_extreme_shading() -> None:
    """Bertrand floor is enforced even at maximum shading (lambda*delta close to 1)."""
    C1_w = 1.0e-3
    w = W_SCEN
    p_star = compute_stackelberg_price(C1_w, w, 1.0)
    # Near-maximum shading: lambda=0.5, delta=0.9999
    p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA=LAMBDA_MAX, delta_comp=0.9999)
    floor = C1_w / w
    assert (
        p_dagger >= floor - 1e-20
    ), f"Floor not enforced: p_dagger={p_dagger:.4e}, floor={floor:.4e}"


def test_spa_price_decreasing_in_delta_comp() -> None:
    """Higher delta_comp -> lower or equal p_dagger (more competitive -> shade more)."""
    C1_w = 2.1e-3
    w = W_SCEN
    p_star = compute_stackelberg_price(C1_w, w, 1.2)
    lambda_SPA = 0.3
    deltas = [0.0, 0.1, 0.3, 0.5, 0.8]
    prices = [compute_spa_price(C1_w, w, p_star, lambda_SPA, d) for d in deltas]
    for i in range(len(prices) - 1):
        assert (
            prices[i] >= prices[i + 1] - 1e-20
        ), f"p_dagger not monotone: prices[{i}]={prices[i]:.4e} < prices[{i+1}]={prices[i+1]:.4e}"


def test_spa_price_batching_invariance() -> None:
    """p_dagger * (2w) == 2 * (p_dagger * w): SPA preserves Layer-1 linearity."""
    params = P1
    load_frac = LOAD_1
    f_max = params["f_max"]
    m_i = compute_markup(load_frac * f_max, f_max)

    C1_w, _, _, _ = compute_layer1_cost(W_SCEN, **params)
    C1_2w, _, _, _ = compute_layer1_cost(2 * W_SCEN, **params)

    p_star_w = compute_stackelberg_price(C1_w, W_SCEN, m_i)
    p_star_2w = compute_stackelberg_price(C1_2w, 2 * W_SCEN, m_i)

    lambda_SPA = 0.3
    delta_comp = 0.05

    p_dagger_w = compute_spa_price(C1_w, W_SCEN, p_star_w, lambda_SPA, delta_comp)
    p_dagger_2w = compute_spa_price(C1_2w, 2 * W_SCEN, p_star_2w, lambda_SPA, delta_comp)

    total_w = p_dagger_w * W_SCEN
    total_2w = p_dagger_2w * (2 * W_SCEN)

    assert math.isclose(
        total_2w, 2 * total_w, rel_tol=1e-10
    ), f"SPA batching violated: 2x={total_2w:.4e}, 2*1x={2*total_w:.4e}"
