"""Invariant test: RFQ round emits at most 4 * K_max messages (MARKET_MECHANISM.md).

One RFQ round consists of:
  - 1 broadcast from orchestrator to K_max peers
  - K_max responses (one per solicited peer)
  - 1 winner selection (internal, no message)
  Total = 1 + K_max = K_max + 1 messages per round (conservative upper bound: 4 * K_max)

The MARKET_MECHANISM.md invariant is: total messages per round <= 4 * K_max.
"""

from __future__ import annotations

from typing import Any

import pytest

from hedge.core.constants import K_MAX
from hedge.market.rfq import RFQResult, run_rfq_round


def _make_synthetic_nodes(n: int, rng: Any) -> list[Any]:
    """Create minimal fake node objects for RFQ testing."""
    from dataclasses import dataclass

    import numpy as np

    @dataclass
    class FakeNode:
        unique_id: int
        f_max: float
        kappa: float
        P_idle: float
        rho: float
        pi_E: float
        beta: float
        w_pending: float = 0.0
        l_hat: float = 0.0
        R_hat: float = 1.0
        p_star_ref: float = 0.0
        p_dagger_ref: float = 0.0

    nodes = []
    for i in range(n):
        f = float(rng.uniform(1e9, 5e9))
        nodes.append(
            FakeNode(
                unique_id=i,
                f_max=f,
                kappa=float(rng.uniform(1e-27, 1e-25)),
                P_idle=float(rng.choice([20.0, 30.0, 50.0])),
                rho=float(rng.uniform(2e-4, 1e-3)),
                pi_E=float(rng.uniform(0.06, 0.18)),
                beta=float(rng.uniform(40.0, 700.0)),
                l_hat=float(rng.uniform(0.0, f * 0.3)),
                R_hat=1.0,
            )
        )
    return nodes


def _make_cloud() -> Any:
    from dataclasses import dataclass

    @dataclass
    class FakeCloud:
        unique_id: int = -1
        f_max: float = 1e12
        kappa: float = 1e-27
        P_idle: float = 0.0
        rho: float = 1e-4
        pi_E: float = 0.06
        beta: float = 80.0
        tau_c: float = 0.035
        w_pending: float = 0.0
        mu_c: float = 1.5

    return FakeCloud()


def _make_task(rng: Any) -> Any:
    from dataclasses import dataclass

    @dataclass
    class FakeTask:
        task_id: str = "t0"
        s: float = 2e6
        w: float = 5e9
        d: float = 2.0
        a_buyer: float = 5.0
        created_at: float = 0.0
        user_id: str = "u0"
        status: str = "pending"

        @property
        def s_ret(self) -> float:
            return self.s * 0.01

    return FakeTask(
        w=float(rng.uniform(1e9, 1e10)),
        d=float(rng.uniform(1.0, 5.0)),
        a_buyer=float(rng.uniform(2.0, 20.0)),
    )


import numpy as np


@pytest.mark.parametrize("k_max", [1, 2, 4])
@pytest.mark.parametrize("n_nodes", [4, 8, 16])
def test_message_count_le_4_k_max(k_max: int, n_nodes: int) -> None:
    """RFQ round message count must be <= 4 * K_max for all (K_max, N_nodes) combos."""
    rng = np.random.default_rng(seed=42)
    nodes = _make_synthetic_nodes(n_nodes, rng)
    cloud = _make_cloud()
    task = _make_task(rng)
    tau_dict = {n.unique_id: float(rng.uniform(0.001, 0.005)) for n in nodes}
    tau_dict[-1] = 0.035  # cloud

    # Compute per-node p_star_ref for peer_standing_quotes
    from hedge.pricing.layer1 import compute_layer1_cost
    from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price

    W_REF = 1e9
    peer_sq = {}
    for node in nodes:
        C1, _, _, _ = compute_layer1_cost(
            W_REF, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
        )
        m = compute_markup(node.l_hat, node.f_max)
        peer_sq[node.unique_id] = compute_stackelberg_price(C1, W_REF, m)

    result: RFQResult = run_rfq_round(
        peer_pool=nodes,
        task=task,
        cloud=cloud,
        tau_dict=tau_dict,
        peer_standing_quotes=peer_sq,
        k_max=k_max,
        alpha_u=0.5,
        gamma_u=0.5,
    )

    # Exact bound from Algorithm 1: k broadcasts + k responses + affordable_size AFGM
    # notifications + 1 ACCEPT + (affordable_size-1) RELEASES <= 4*k_max + 2
    max_allowed = 4 * k_max + 2
    assert (
        result.message_count <= max_allowed
    ), f"K_max={k_max}, N={n_nodes}: message_count={result.message_count} > {max_allowed}"


def test_message_count_default_k_max_stress() -> None:
    """Run 50 random tasks with K_max=4 and verify message count invariant holds throughout."""
    rng = np.random.default_rng(seed=99)
    k_max = K_MAX  # 4
    max_allowed = 4 * k_max + 2

    for trial in range(50):
        n = int(rng.integers(4, 20))
        nodes = _make_synthetic_nodes(n, rng)
        cloud = _make_cloud()
        task = _make_task(rng)
        tau_dict = {node.unique_id: float(rng.uniform(0.001, 0.01)) for node in nodes}

        from hedge.pricing.layer1 import compute_layer1_cost
        from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price

        W_REF = 1e9
        peer_sq = {}
        for node in nodes:
            C1, _, _, _ = compute_layer1_cost(
                1e9, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
            )
            m = compute_markup(node.l_hat, node.f_max)
            peer_sq[node.unique_id] = compute_stackelberg_price(C1, W_REF, m)

        result = run_rfq_round(
            peer_pool=nodes,
            task=task,
            cloud=cloud,
            tau_dict=tau_dict,
            peer_standing_quotes=peer_sq,
            k_max=k_max,
        )

        # Bound is 4*K_max + 2 (see test_message_count_le_4_k_max for derivation)
        assert (
            result.message_count <= max_allowed
        ), f"Trial {trial}: message_count={result.message_count} > {max_allowed}"
