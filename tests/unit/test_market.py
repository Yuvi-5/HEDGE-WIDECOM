"""Phase 6 gate tests: market mechanism (Phase 0, AFGM, RFQ round)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from hedge.core.constants import K_MAX, MU_C, R_REF
from hedge.core.task import HEDGETask
from hedge.market.afgm import afgm_select, build_affordable_set, compute_node_latency
from hedge.market.phase0 import (
    compute_aeco_score,
    compute_gc_entry_bid,
    is_capacity_feasible,
    run_phase0,
)
from hedge.market.rfq import run_rfq_round
from hedge.market.tier3_inversion import invert_load_from_price
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price

# ---------------------------------------------------------------------------
# Lightweight proxies (avoid EdgeSimPy init overhead in unit tests)
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """Minimal node proxy for market unit tests."""

    unique_id: int
    f_max: float
    kappa: float
    rho: float
    P_idle: float
    beta: float
    pi_E: float
    w_pending: float = 0.0
    l_hat: float = 0.0
    R_hat: float = 0.0
    p_star_ref: float = 0.0
    p_dagger_ref: float = 0.0

    @property
    def W_q(self) -> float:
        """Queue wait scalar (seconds)."""
        return self.w_pending / self.f_max


@dataclass
class _Cloud:
    """Minimal cloud proxy for market unit tests."""

    f_max: float = 1e12
    kappa: float = 1e-27
    rho: float = 1e-4
    P_idle: float = 0.0
    beta: float = 80.0
    pi_E: float = 0.06
    mu_c: float = MU_C
    tau_c: float = 0.035


# ---------------------------------------------------------------------------
# Scenario A fixtures (same as test_layer2.py for consistency)
# ---------------------------------------------------------------------------

W_SCEN: float = 5e9
S_SCEN: float = 2e6  # 2 MB in bits
D_SCEN: float = 0.1  # 100 ms

P1 = _Node(1, f_max=3e9, kappa=1e-26, rho=1.25e-3, P_idle=30.0, beta=200.0, pi_E=0.12)
P2 = _Node(2, f_max=2e9, kappa=5e-26, rho=1.58268e-3, P_idle=30.0, beta=500.0, pi_E=0.12)
P3 = _Node(3, f_max=4e9, kappa=2e-26, rho=1.07488e-3, P_idle=30.0, beta=80.0, pi_E=0.12)

LOAD_1: float = 0.20
LOAD_2: float = 0.30
LOAD_3: float = 0.50

CLOUD = _Cloud()

# Peer delays (user to each node, seconds)
TAU: dict[int, float] = {1: 0.001, 2: 0.002, 3: 0.001}
BANDWIDTH: float = 1e9  # 1 Gbps metro


def _make_task(a_buyer: float = 5.0) -> HEDGETask:
    return HEDGETask(
        task_id="T_test",
        s=S_SCEN,
        w=W_SCEN,
        d=D_SCEN,
        a_buyer=a_buyer,
        created_at=0.0,
    )


def _seed_node_state(node: _Node, load_frac: float, R_hat: float = 0.10) -> None:
    """Set l_hat, p_star_ref, p_dagger_ref from load fraction for Phase 0 tests."""
    node.l_hat = load_frac * node.f_max
    node.R_hat = R_hat
    C1, _, _, _ = compute_layer1_cost(
        W_SCEN, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
    )
    m = compute_markup(node.l_hat, node.f_max)
    node.p_star_ref = compute_stackelberg_price(C1, W_SCEN, m)
    node.p_dagger_ref = node.p_star_ref * 0.98  # approximate shaded ref


# ---------------------------------------------------------------------------
# Gate test 1: Scenario A - Peer 3 wins RFQ
# ---------------------------------------------------------------------------


def test_scenario_a_peer3_wins_rfq() -> None:
    """Scenario A: 3 peers, Peer 3 wins RFQ with lowest p_dagger*w.

    Cloud is made unaffordable (mu_c=100) to isolate edge-peer selection.
    Peer 3 wins because it has both the fastest CPU (lowest latency) and
    the lowest shaded price (0.072 USD) among the three edge peers.
    Deadline is relaxed to 5.0s so all edge peers are execution-feasible.
    """
    _seed_node_state(P1, LOAD_1)
    _seed_node_state(P2, LOAD_2)
    _seed_node_state(P3, LOAD_3)

    # Use 5s deadline: execution at 4 GHz for 5e9 cycles = 1.25s < 5.0s (all peers feasible)
    task = HEDGETask("T_A", s=S_SCEN, w=W_SCEN, d=5.0, a_buyer=5.0, created_at=0.0)
    standing = {n.unique_id: n.p_star_ref for n in [P1, P2, P3]}
    # Make cloud unaffordable: p_c*w >> a_buyer=5.0
    expensive_cloud = _Cloud(mu_c=100.0)

    result = run_rfq_round(
        peer_pool=[P1, P2, P3],
        task=task,
        cloud=expensive_cloud,
        tau_dict=TAU,
        peer_standing_quotes=standing,
        k_max=3,
        bandwidth_metro=BANDWIDTH,
    )

    assert result.executor_type == "edge", f"Expected edge, got {result.executor_type}"
    assert result.winner is P3, (
        f"Expected Peer 3 (fastest CPU + cheapest price), "
        f"got id={getattr(result.winner, 'unique_id', '?')}"
    )
    assert result.price_total < 5.0, f"price_total={result.price_total} exceeds a_buyer=5.0"


# ---------------------------------------------------------------------------
# Gate test 2: Scenario B - all peers overloaded, cloud wins
# ---------------------------------------------------------------------------


def test_scenario_b_cloud_wins() -> None:
    """Scenario B: peers at full load -> latency infeasible, cloud wins."""
    p1_over = _Node(11, f_max=3e9, kappa=1e-26, rho=1.25e-3, P_idle=30.0, beta=200.0, pi_E=0.12)
    p2_over = _Node(12, f_max=2e9, kappa=5e-26, rho=1.58268e-3, P_idle=30.0, beta=500.0, pi_E=0.12)

    # Saturated queue: W_q will exceed deadline
    for n in [p1_over, p2_over]:
        n.w_pending = n.f_max * 1.0  # 1 second of backlog >> d=0.1s
        n.l_hat = n.f_max
        n.R_hat = 0.0

    task = _make_task(a_buyer=5.0)
    standing: dict[int, float] = {}

    result = run_rfq_round(
        peer_pool=[p1_over, p2_over],
        task=task,
        cloud=CLOUD,
        tau_dict={11: 0.001, 12: 0.001},
        peer_standing_quotes=standing,
        k_max=2,
    )

    assert result.executor_type == "cloud", f"Expected cloud, got {result.executor_type}"


# ---------------------------------------------------------------------------
# Gate test 3: Scenario C - a_buyer too small, all-reject
# ---------------------------------------------------------------------------


def test_scenario_c_all_reject() -> None:
    """Scenario C: a_buyer << any quote -> empty F_b -> unmatched."""
    _seed_node_state(P1, LOAD_1)
    _seed_node_state(P2, LOAD_2)
    _seed_node_state(P3, LOAD_3)

    task = _make_task(a_buyer=1e-6)  # essentially zero budget

    result = run_rfq_round(
        peer_pool=[P1, P2, P3],
        task=task,
        cloud=CLOUD,
        tau_dict=TAU,
        k_max=3,
    )

    assert result.executor_type == "unmatched", f"Expected unmatched, got {result.executor_type}"
    assert math.isinf(result.price_total)


# ---------------------------------------------------------------------------
# Gate test 5: K_max=0 recovery - only cloud
# ---------------------------------------------------------------------------


def test_kmax_zero_recovery() -> None:
    """K_max=0: empty peer pool -> cloud wins or unmatched (no edge selection)."""
    task = _make_task(a_buyer=5.0)

    result = run_rfq_round(
        peer_pool=[],
        task=task,
        cloud=CLOUD,
        tau_dict={},
        k_max=0,
    )

    assert result.executor_type in {
        "cloud",
        "unmatched",
    }, f"K_max=0 should only produce cloud or unmatched, got {result.executor_type}"
    if result.executor_type == "cloud":
        assert result.winner is CLOUD


# ---------------------------------------------------------------------------
# Gate test 6: message count <= 4*K_max + 10
# ---------------------------------------------------------------------------


def test_rfq_message_count() -> None:
    """Message count per RFQ round does not exceed 4*K_max + 10."""
    _seed_node_state(P1, LOAD_1)
    _seed_node_state(P2, LOAD_2)
    _seed_node_state(P3, LOAD_3)

    task = _make_task(a_buyer=5.0)
    standing = {n.unique_id: n.p_star_ref for n in [P1, P2, P3]}

    result = run_rfq_round(
        peer_pool=[P1, P2, P3],
        task=task,
        cloud=CLOUD,
        tau_dict=TAU,
        peer_standing_quotes=standing,
        k_max=K_MAX,
    )

    limit = 4 * K_MAX + 10
    assert (
        result.message_count <= limit
    ), f"message_count={result.message_count} exceeds limit {limit}"


# ---------------------------------------------------------------------------
# Phase 0 tests
# ---------------------------------------------------------------------------


def test_phase0_selects_highest_aeco() -> None:
    """run_phase0 returns the node with highest A_eco score."""
    _seed_node_state(P1, LOAD_1)
    _seed_node_state(P2, LOAD_2)
    _seed_node_state(P3, LOAD_3)

    task = _make_task()
    tau_dict = {1: 0.001, 2: 0.003, 3: 0.002}

    orchestrator, _ = run_phase0([P1, P2, P3], task, tau_dict)
    # Must be one of the three (compare by unique_id; dataclass is not hashable)
    assert orchestrator.unique_id in {1, 2, 3}


def test_phase0_peer_pool_feasible_only() -> None:
    """Peer pool P0 contains only capacity-feasible losers."""
    _seed_node_state(P1, LOAD_1)
    _seed_node_state(P2, LOAD_2)
    _seed_node_state(P3, LOAD_3)

    # Overload P2 so it fails feasibility
    P2.l_hat = P2.f_max  # full load: l_hat + w/d > f_max for any w>0

    task = _make_task()
    tau_dict = {1: 0.002, 2: 0.001, 3: 0.002}

    _orch, peer_pool = run_phase0([P1, P2, P3], task, tau_dict)

    overloaded = [n for n in peer_pool if n.unique_id == 2]
    assert not overloaded, "Overloaded node P2 should not be in peer pool"


def test_phase0_empty_candidate_raises() -> None:
    """run_phase0 raises ValueError for empty candidate list."""
    task = _make_task()
    with pytest.raises(ValueError, match="non-empty"):
        run_phase0([], task, {})


def test_phase0_gc_risk_premium_disabled_by_default() -> None:
    """Default run_phase0 call (no enable_gc_risk_premium) excludes overloaded
    losers exactly as before -- backward compatible with the pre-Eq6 behaviour."""
    _seed_node_state(P1, LOAD_1)
    _seed_node_state(P3, LOAD_3)
    P2.l_hat = P2.f_max  # fully overloaded

    task = _make_task()
    tau_dict = {1: 0.002, 2: 0.001, 3: 0.002}

    _orch, peer_pool = run_phase0([P1, P2, P3], task, tau_dict)
    assert all(n.unique_id != 2 for n in peer_pool)


# ---------------------------------------------------------------------------
# GC risk-premium entry (Section 1e, Eq 6-8)
# ---------------------------------------------------------------------------


def _gc_test_node(p_star_ref: float, p_dagger_ref: float) -> _Node:
    """A node proxy with pre-set standing quotes for compute_gc_entry_bid tests."""
    node = _Node(9, f_max=3e9, kappa=1e-26, rho=1.25e-3, P_idle=30.0, beta=200.0, pi_E=0.12)
    node.p_star_ref = p_star_ref
    node.p_dagger_ref = p_dagger_ref
    return node


def test_gc_entry_bid_theta_risk_boundary() -> None:
    """Eq 6: p_bid - p_hat_dagger_j >= theta_risk * C1_i/w gates GC-play entry."""
    task = _make_task()
    C1, _, _, _ = compute_layer1_cost(task.w, 3e9, 1e-26, 30.0, 1.25e-3, 0.12, 200.0)
    floor = C1 / task.w

    p_hat_dagger_j = floor * 1.5
    node = _gc_test_node(p_star_ref=floor * 3.0, p_dagger_ref=p_hat_dagger_j + 0.2 * floor)

    # gap = 0.2*floor >= theta_risk(0.1)*floor -> entry allowed
    assert compute_gc_entry_bid(node, task, p_hat_dagger_j, theta_risk=0.1) == node.p_dagger_ref
    # gap = 0.2*floor < theta_risk(0.3)*floor -> entry refused
    assert compute_gc_entry_bid(node, task, p_hat_dagger_j, theta_risk=0.3) is None


def test_gc_entry_bid_rejects_below_floor_or_above_p_star() -> None:
    """Eq 6 also requires C1_i/w <= p_bid_i < p_star_i regardless of theta_risk."""
    task = _make_task()
    C1, _, _, _ = compute_layer1_cost(task.w, 3e9, 1e-26, 30.0, 1.25e-3, 0.12, 200.0)
    floor = C1 / task.w

    below_floor = _gc_test_node(p_star_ref=floor * 3.0, p_dagger_ref=floor * 0.5)
    assert compute_gc_entry_bid(below_floor, task, p_hat_dagger_j=0.0, theta_risk=0.0) is None

    at_or_above_p_star = _gc_test_node(p_star_ref=floor * 2.0, p_dagger_ref=floor * 2.0)
    assert compute_gc_entry_bid(at_or_above_p_star, task, p_hat_dagger_j=0.0, theta_risk=0.0) is None


def test_phase0_gc_risk_premium_admits_overloaded_node_when_enabled() -> None:
    """With enable_gc_risk_premium=True and a satisfied Eq 6 gap, an overloaded
    node (excluded by is_capacity_feasible) still enters the peer pool."""
    task = _make_task()
    C1, _, _, _ = compute_layer1_cost(task.w, 3e9, 1e-26, 30.0, 1.25e-3, 0.12, 200.0)
    floor = C1 / task.w

    # Orchestrator: low (competitive) price -> wins A_eco outright.
    orchestrator = _gc_test_node(p_star_ref=floor * 1.1, p_dagger_ref=floor * 1.1)
    orchestrator.unique_id = 100
    orchestrator.l_hat = 0.0

    # Overloaded: asks a modest premium (gap = 0.2*floor) over the orchestrator's
    # price, and fails is_capacity_feasible outright (l_hat pinned at f_max).
    overloaded = _gc_test_node(p_star_ref=floor * 3.0, p_dagger_ref=floor * 1.3)
    overloaded.unique_id = 101
    overloaded.l_hat = overloaded.f_max

    tau_dict = {100: 0.001, 101: 0.001}
    _orch, peer_pool = run_phase0(
        [orchestrator, overloaded],
        task,
        tau_dict,
        k_max=4,
        enable_gc_risk_premium=True,
        theta_risk=0.1,
    )
    assert _orch.unique_id == 100, "expected the low-price node to win orchestrator role"
    assert any(n.unique_id == 101 for n in peer_pool), "GC-play entrant should be admitted"

    # Same scenario with enable_gc_risk_premium=False must exclude it.
    _orch2, peer_pool_off = run_phase0(
        [orchestrator, overloaded], task, tau_dict, k_max=4, enable_gc_risk_premium=False
    )
    assert all(n.unique_id != 101 for n in peer_pool_off)

    # A stricter theta_risk (0.3 > the 0.2*floor gap) must refuse the same entrant.
    _orch3, peer_pool_strict = run_phase0(
        [orchestrator, overloaded],
        task,
        tau_dict,
        k_max=4,
        enable_gc_risk_premium=True,
        theta_risk=0.3,
    )
    assert all(n.unique_id != 101 for n in peer_pool_strict)


def test_aeco_score_normalised() -> None:
    """A_eco scores for all candidates sum to 1."""
    all_p = [0.01, 0.02, 0.015]
    all_t = [0.001, 0.002, 0.001]
    total = sum(compute_aeco_score(p, t, all_p, all_t) for p, t in zip(all_p, all_t))
    assert math.isclose(total, 1.0, rel_tol=1e-9), f"Scores do not sum to 1: {total}"


def test_aeco_zero_for_invalid_inputs() -> None:
    """A_eco returns 0 when p_dagger_ref or tau_ui is non-positive."""
    assert compute_aeco_score(0.0, 0.001, [0.01], [0.001]) == 0.0
    assert compute_aeco_score(0.01, 0.0, [0.01], [0.001]) == 0.0


def test_capacity_feasibility() -> None:
    """is_capacity_feasible correctly enforces l_hat + w/d <= f_max."""
    assert is_capacity_feasible(0.0, 3e9, 1e9, 0.5)  # 0 + 2e9 <= 3e9
    assert not is_capacity_feasible(3e9, 3e9, 1e9, 0.5)  # 3e9 + 2e9 > 3e9


# ---------------------------------------------------------------------------
# AFGM tests
# ---------------------------------------------------------------------------


def test_afgm_returns_none_for_empty_set() -> None:
    """afgm_select returns None for empty affordable set."""
    assert afgm_select([]) is None


def test_afgm_selects_minimum_phi() -> None:
    """afgm_select picks min(alpha*L + gamma*p) over affordable set."""
    n1 = object()
    n2 = object()
    affordable = [(n1, 0.01, 0.050), (n2, 0.005, 0.080)]
    # Phi1 = 0.5*0.050 + 0.5*0.01 = 0.030
    # Phi2 = 0.5*0.080 + 0.5*0.005 = 0.0425
    winner, price, _ = afgm_select(affordable, alpha_u=0.5, gamma_u=0.5)
    assert winner is n1 and math.isclose(price, 0.01)


def test_build_affordable_set_excludes_overpriced() -> None:
    """build_affordable_set excludes sellers with p_dagger*w > a_buyer."""
    n1 = _Node(50, f_max=3e9, kappa=1e-26, rho=1.25e-3, P_idle=30.0, beta=200.0, pi_E=0.12)
    task = HEDGETask("t", s=1e6, w=1e9, d=1.0, a_buyer=0.001, created_at=0.0)
    peers = [(n1, 0.01, 0.050)]  # p_dagger=0.01 USD/cycle * 1e9 cycles = 1e7 >> a_buyer
    result = build_affordable_set(peers, None, task, task.a_buyer)
    assert not result


def test_build_affordable_set_excludes_latency_infeasible() -> None:
    """build_affordable_set excludes sellers with L_i > d."""
    n1 = _Node(51, f_max=3e9, kappa=1e-26, rho=1.25e-3, P_idle=30.0, beta=200.0, pi_E=0.12)
    task = HEDGETask("t", s=1e6, w=1e9, d=0.001, a_buyer=100.0, created_at=0.0)
    peers = [(n1, 1e-12, 5.0)]  # L_i=5.0 >> d=0.001
    result = build_affordable_set(peers, None, task, task.a_buyer)
    assert not result


# ---------------------------------------------------------------------------
# Tier-3 inversion test
# ---------------------------------------------------------------------------


def test_tier3_inversion_roundtrip() -> None:
    """invert_load_from_price recovers l_hat used to generate p_star."""
    from hedge.pricing.layer1 import compute_layer1_cost

    f_max = 3e9
    l_hat_orig = 0.5 * f_max
    w = 5e9

    C1_w, _, _, _ = compute_layer1_cost(w, 3e9, 1e-26, 30.0, 1.25e-3, 0.12, 200.0)
    m = compute_markup(l_hat_orig, f_max)
    p_star = compute_stackelberg_price(C1_w, w, m)
    l_hat_recovered = invert_load_from_price(p_star, C1_w, w, f_max)

    assert math.isclose(
        l_hat_recovered, l_hat_orig, rel_tol=1e-8
    ), f"Round-trip failed: orig={l_hat_orig:.3e}, recovered={l_hat_recovered:.3e}"
