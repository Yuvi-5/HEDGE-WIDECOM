"""Integration test: full RFQ round from Phase 0 to settlement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hedge.core.constants import K_MAX, MU_C, R_REF
from hedge.core.task import HEDGETask
from hedge.market.phase0 import run_phase0
from hedge.market.rfq import run_rfq_round
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price


@dataclass
class _Node:
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
    f_max: float = 1e12
    kappa: float = 1e-27
    rho: float = 1e-4
    P_idle: float = 0.0
    beta: float = 80.0
    pi_E: float = 0.06
    mu_c: float = MU_C
    tau_c: float = 0.035


def _setup_node(uid: int, f_max: float, load_frac: float = 0.3, R_hat: float = 0.1) -> _Node:
    n = _Node(
        unique_id=uid,
        f_max=f_max,
        kappa=1e-26,
        rho=1.25e-3,
        P_idle=30.0,
        beta=200.0,
        pi_E=0.12,
    )
    n.l_hat = load_frac * f_max
    n.R_hat = R_hat
    return n


CLOUD = _Cloud()


# ---------------------------------------------------------------------------
# Full RFQ round: Phase 0 -> orchestrator -> RFQ -> settlement
# ---------------------------------------------------------------------------


def test_full_rfq_round_phase0_to_settlement() -> None:
    """Full pipeline: Phase 0 selects orchestrator, RFQ picks winner, IR holds."""
    rng = np.random.default_rng(1)
    nodes = [
        _setup_node(i, float(rng.uniform(2e9, 4e9)), float(rng.uniform(0.1, 0.6))) for i in range(5)
    ]

    for n in nodes:
        C1, _, _, _ = compute_layer1_cost(5e9, n.f_max, n.kappa, n.P_idle, n.rho, n.pi_E, n.beta)
        n.p_star_ref = compute_stackelberg_price(C1, 5e9, compute_markup(n.l_hat, n.f_max))
        n.p_dagger_ref = n.p_star_ref * 0.97

    task = HEDGETask("T1", s=2e6, w=5e9, d=0.5, a_buyer=5.0, created_at=0.0)
    tau_dict = {n.unique_id: 0.001 + 0.001 * n.unique_id for n in nodes}

    orchestrator, peer_pool = run_phase0(nodes, task, tau_dict)

    result = run_rfq_round(
        peer_pool=peer_pool,
        task=task,
        cloud=CLOUD,
        tau_dict=tau_dict,
        peer_standing_quotes={n.unique_id: n.p_star_ref for n in nodes},
        k_max=K_MAX,
    )

    # Winner must be one of the nodes or cloud
    assert result.executor_type in {"edge", "cloud", "unmatched"}
    if result.executor_type != "unmatched":
        # Buyer IR
        assert (
            result.price_total <= task.a_buyer + 1e-10
        ), f"Buyer IR violated: price_total={result.price_total:.4f} > a_buyer={task.a_buyer}"
        # Seller IR (Bertrand floor)
        assert (
            result.price_total >= result.C1_winner - 1e-10
        ), f"Seller IR violated: price_total={result.price_total:.4f} < C1={result.C1_winner:.4f}"
        # WBB: seller_receipt == price_total
        assert abs(result.seller_receipt - result.price_total) < 1e-10


def test_rfq_wbb_holds_over_many_rounds() -> None:
    """WBB: seller_receipt == price_total for 100 random RFQ rounds."""
    rng = np.random.default_rng(7)
    violations: list[str] = []

    for i in range(100):
        n_nodes = int(rng.integers(1, 5))
        nodes = [
            _setup_node(j, float(rng.uniform(1e9, 4e9)), float(rng.uniform(0.0, 0.9)))
            for j in range(n_nodes)
        ]
        task = HEDGETask(
            f"T{i}",
            s=float(rng.uniform(1e6, 2e7)),
            w=float(rng.uniform(1e9, 1e10)),
            d=float(rng.uniform(0.05, 1.0)),
            a_buyer=float(rng.uniform(0.01, 10.0)),
            created_at=0.0,
        )
        tau_dict = {n.unique_id: float(rng.uniform(0.001, 0.005)) for n in nodes}

        result = run_rfq_round(
            peer_pool=nodes,
            task=task,
            cloud=CLOUD,
            tau_dict=tau_dict,
            k_max=K_MAX,
        )

        if result.executor_type != "unmatched":
            diff = abs(result.seller_receipt - result.price_total)
            if diff > 1e-10:
                violations.append(f"Round {i}: WBB diff={diff:.2e}")

    assert not violations, f"WBB violated {len(violations)} times. First: {violations[0]}"


def test_rfq_message_count_within_budget_random() -> None:
    """message_count <= 4*K_max + 10 for 50 random RFQ rounds."""
    rng = np.random.default_rng(42)
    limit = 4 * K_MAX + 10

    for i in range(50):
        k = int(rng.integers(1, K_MAX + 1))
        nodes = [
            _setup_node(j, float(rng.uniform(1e9, 4e9)), float(rng.uniform(0.0, 0.9)))
            for j in range(k)
        ]
        task = HEDGETask(
            f"T{i}",
            s=float(rng.uniform(1e6, 2e7)),
            w=float(rng.uniform(1e9, 1e10)),
            d=float(rng.uniform(0.1, 2.0)),
            a_buyer=float(rng.uniform(0.01, 10.0)),
            created_at=0.0,
        )
        tau_dict = {n.unique_id: 0.002 for n in nodes}

        result = run_rfq_round(
            peer_pool=nodes,
            task=task,
            cloud=CLOUD,
            tau_dict=tau_dict,
            k_max=K_MAX,
        )

        assert (
            result.message_count <= limit
        ), f"Round {i}: messages={result.message_count} > {limit}"
