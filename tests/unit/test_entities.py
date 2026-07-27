"""Phase 1 gate tests: core entity instantiation and topology validity."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hedge.core.cloud import CloudNode
from hedge.core.constants import (
    A_REF,
    ALPHA_R,
    B_SHAPE,
    BETA_CLOUD,
    BETA_MAX,
    BETA_MIN,
    DELTA_Q,
    DELTA_TP,
    EPSILON_TRANS,
    ETA,
    F_MAX_CLOUD,
    F_MAX_MAX,
    F_MAX_MIN,
    H_MAX,
    JOULES_PER_KWH,
    K_MAX,
    KAPPA_CLOUD,
    KAPPA_MAX,
    KAPPA_MIN,
    LAMBDA_MAX,
    MU_C,
    N_NODES_DEFAULT,
    P_IDLE_OPTIONS,
    PI_CO2,
    PI_E_CLOUD,
    PI_E_MAX,
    PI_E_MIN,
    R_REF,
    RHO_CLOUD,
    RHO_MAX,
    RHO_MIN,
    S_RET_FRACTION,
    TAU_C_MEAN,
    TAU_C_STD,
    TAU_LOCK,
    TAU_MESH_MAX,
    THETA_RISK,
    W_FAIL,
    WARMUP_DURATION,
)
from hedge.core.node import HEDGEEdgeServer
from hedge.core.task import HEDGETask
from hedge.core.topology import create_topology
from hedge.core.user import HEDGEUser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG_SEED = 42
CFG_SYNTHETIC_50 = {
    "topology": {"source": "synthetic", "N_nodes": 50, "tau_c_mean": 0.035, "tau_c_std": 0.008},
    "arrivals": {"n_hotspots": 5, "hotspot_rate_multiplier": 5.0, "quiet_rate_multiplier": 0.2},
}
CFG_SYNTHETIC_125 = {
    "topology": {"source": "synthetic", "N_nodes": 125, "tau_c_mean": 0.035, "tau_c_std": 0.008},
    "arrivals": {"n_hotspots": 5, "hotspot_rate_multiplier": 5.0, "quiet_rate_multiplier": 0.2},
}
CFG_EUA = {
    "topology": {
        "source": "eua_melbourne",
        "N_nodes": 125,
        "tau_c_mean": 0.035,
        "tau_c_std": 0.008,
        "eua_data_path": "data/eua_melbourne_DOES_NOT_EXIST.csv",
    },
    "arrivals": {"n_hotspots": 5, "hotspot_rate_multiplier": 5.0, "quiet_rate_multiplier": 0.2},
}


def _make_node(obj_id: int = 1) -> HEDGEEdgeServer:
    """Return a HEDGEEdgeServer with typical Scenario-A Peer-1 parameters."""
    return HEDGEEdgeServer(
        obj_id=obj_id,
        f_max=3e9,
        kappa=1e-26,
        rho=5e-4,
        P_idle=30.0,
        beta=200.0,
        pi_E=0.12,
    )


def _make_task() -> HEDGETask:
    """Return a typical HEDGETask for testing."""
    return HEDGETask(
        task_id="t1",
        s=2e6,
        w=5e9,
        d=0.1,
        a_buyer=5.0,
        created_at=0.0,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_pricing_constants() -> None:
    """All 9 pricing constants exist with correct values (within 0.1%)."""
    assert math.isclose(A_REF, 5e-10, rel_tol=1e-3)
    assert math.isclose(B_SHAPE, 1.0, rel_tol=1e-3)
    assert math.isclose(ETA, 1.0, rel_tol=1e-3)
    assert math.isclose(LAMBDA_MAX, 0.5, rel_tol=1e-3)
    assert math.isclose(THETA_RISK, 0.12, rel_tol=1e-3)
    assert math.isclose(ALPHA_R, 0.2, rel_tol=1e-3)
    assert math.isclose(R_REF, 1.0, rel_tol=1e-3)
    assert math.isclose(PI_CO2, 5e-5, rel_tol=1e-3)
    assert math.isclose(MU_C, 1.5, rel_tol=1e-3)


def test_protocol_constants() -> None:
    """All 7 protocol constants exist with correct values."""
    assert K_MAX == 4
    assert H_MAX == 2
    assert W_FAIL == 20
    assert math.isclose(EPSILON_TRANS, 0.05, rel_tol=1e-3)
    assert math.isclose(TAU_LOCK, 0.02, rel_tol=1e-3)
    assert math.isclose(DELTA_Q, 0.1, rel_tol=1e-3)
    assert math.isclose(DELTA_TP, 0.1, rel_tol=1e-3)


def test_energy_constant() -> None:
    """JOULES_PER_KWH is 3.6e6."""
    assert math.isclose(JOULES_PER_KWH, 3.6e6, rel_tol=1e-3)


def test_idle_power_options() -> None:
    """P_IDLE_OPTIONS contains exactly [20, 30, 50] watts."""
    assert sorted(P_IDLE_OPTIONS) == [20.0, 30.0, 50.0]


# ---------------------------------------------------------------------------
# HEDGETask
# ---------------------------------------------------------------------------


def test_task_instantiation() -> None:
    """HEDGETask can be created with all required fields."""
    task = _make_task()
    assert task.task_id == "t1"
    assert task.s == 2e6
    assert task.w == 5e9
    assert task.d == 0.1
    assert task.a_buyer == 5.0
    assert task.created_at == 0.0


def test_task_status_default() -> None:
    """HEDGETask.status defaults to 'pending'."""
    assert _make_task().status == "pending"


def test_task_user_id_default() -> None:
    """HEDGETask.user_id defaults to empty string."""
    assert _make_task().user_id == ""


def test_task_s_ret_property() -> None:
    """s_ret equals s * S_RET_FRACTION."""
    task = _make_task()
    assert math.isclose(task.s_ret, task.s * S_RET_FRACTION, rel_tol=1e-10)
    assert math.isclose(task.s_ret, 2e6 * 0.01, rel_tol=1e-10)


def test_task_s_ret_scales_with_s() -> None:
    """s_ret is proportional to s."""
    task_a = HEDGETask("a", s=1e6, w=1e9, d=0.1, a_buyer=1.0, created_at=0.0)
    task_b = HEDGETask("b", s=2e6, w=1e9, d=0.1, a_buyer=1.0, created_at=0.0)
    assert math.isclose(task_b.s_ret / task_a.s_ret, 2.0, rel_tol=1e-10)


def test_task_status_mutability() -> None:
    """HEDGETask.status can be updated (not frozen dataclass)."""
    task = _make_task()
    task.status = "completed"
    assert task.status == "completed"


# ---------------------------------------------------------------------------
# CloudNode
# ---------------------------------------------------------------------------


def test_cloud_node_defaults() -> None:
    """CloudNode() uses constants from constants.py."""
    cloud = CloudNode()
    assert math.isclose(cloud.f_max, F_MAX_CLOUD, rel_tol=1e-3)
    assert math.isclose(cloud.kappa, KAPPA_CLOUD, rel_tol=1e-3)
    assert math.isclose(cloud.rho, RHO_CLOUD, rel_tol=1e-3)
    assert math.isclose(cloud.beta, BETA_CLOUD, rel_tol=1e-3)
    assert math.isclose(cloud.pi_E, PI_E_CLOUD, rel_tol=1e-3)
    assert math.isclose(cloud.mu_c, MU_C, rel_tol=1e-3)
    assert math.isclose(cloud.tau_c, TAU_C_MEAN, rel_tol=1e-3)
    assert cloud.P_idle == 0.0


def test_cloud_node_tau_c_settable() -> None:
    """CloudNode.tau_c can be overridden at construction."""
    cloud = CloudNode(tau_c=0.040)
    assert math.isclose(cloud.tau_c, 0.040, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# HEDGEEdgeServer
# ---------------------------------------------------------------------------


def test_edge_server_instantiation() -> None:
    """HEDGEEdgeServer instantiates with hardware parameters."""
    server = _make_node(obj_id=100)
    assert server.id == 100
    assert math.isclose(server.f_max, 3e9, rel_tol=1e-10)
    assert math.isclose(server.kappa, 1e-26, rel_tol=1e-10)
    assert math.isclose(server.rho, 5e-4, rel_tol=1e-10)
    assert server.P_idle == 30.0
    assert server.beta == 200.0
    assert math.isclose(server.pi_E, 0.12, rel_tol=1e-10)


def test_edge_server_w_pending_init() -> None:
    """w_pending initialises to 0."""
    assert _make_node().w_pending == 0.0


def test_edge_server_W_q_zero_initially() -> None:
    """W_q returns 0 when w_pending is 0."""
    assert _make_node().W_q == 0.0


def test_edge_server_W_q_property() -> None:
    """W_q equals w_pending / f_max after queue update."""
    server = _make_node()
    server.w_pending = 1.5e9
    expected = 1.5e9 / 3e9
    assert math.isclose(server.W_q, expected, rel_tol=1e-10)


def test_edge_server_tier2_state_init() -> None:
    """Tier-2 private state initialises to zero."""
    server = _make_node()
    assert server.l_hat == 0.0
    assert server.R_hat == 0.0
    assert server.lambda_SPA == 0.0


def test_edge_server_tier1_state_init() -> None:
    """Tier-1 public quote cache initialises to zero."""
    server = _make_node()
    assert server.p_star_ref == 0.0
    assert server.p_dagger_ref == 0.0


def test_edge_server_connectivity_init() -> None:
    """Connectivity containers initialise empty."""
    server = _make_node()
    assert server.peer_delays == {}
    assert server.coverage_users == []


def test_edge_server_arrival_rate_multiplier_default() -> None:
    """arrival_rate_multiplier defaults to 1.0."""
    assert _make_node().arrival_rate_multiplier == 1.0


# ---------------------------------------------------------------------------
# HEDGEUser
# ---------------------------------------------------------------------------


def test_user_instantiation() -> None:
    """HEDGEUser instantiates with an obj_id."""
    user = HEDGEUser(obj_id=200)
    assert user.id == 200


def test_user_coverage_set_default() -> None:
    """coverage_set defaults to empty list."""
    assert HEDGEUser(obj_id=1).coverage_set == []


def test_user_task_queue_default() -> None:
    """task_queue defaults to empty list."""
    assert HEDGEUser(obj_id=1).task_queue == []


def test_user_rng_default() -> None:
    """rng defaults to None (injected by runner)."""
    assert HEDGEUser(obj_id=1).rng is None


def test_user_coverage_set_settable() -> None:
    """coverage_set can be populated after construction."""
    user = HEDGEUser(obj_id=1)
    user.coverage_set = [10, 11, 12]
    assert len(user.coverage_set) == 3




# ---------------------------------------------------------------------------
# Topology - 50-node synthetic
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def topo_50() -> tuple:
    """Build 50-node synthetic topology once for this module."""
    rng = np.random.default_rng(RNG_SEED)
    return create_topology(CFG_SYNTHETIC_50, rng)


def test_topology_50_node_count(topo_50: tuple) -> None:
    """Synthetic 50-node topology returns exactly 50 nodes."""
    nodes, tau, cloud = topo_50
    assert len(nodes) == 50


def test_topology_50_delay_shape(topo_50: tuple) -> None:
    """Delay matrix is 50x50."""
    nodes, tau, cloud = topo_50
    assert tau.shape == (50, 50)


def test_topology_50_cloud_is_cloud_node(topo_50: tuple) -> None:
    """Third return value is a CloudNode."""
    nodes, tau, cloud = topo_50
    assert isinstance(cloud, CloudNode)


# ---------------------------------------------------------------------------
# Topology - 125-node synthetic (Phase 1 gate: EUA-equivalent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def topo_125() -> tuple:
    """Build 125-node synthetic topology once for this module."""
    rng = np.random.default_rng(RNG_SEED)
    return create_topology(CFG_SYNTHETIC_125, rng)


def test_topology_125_node_count(topo_125: tuple) -> None:
    """Synthetic 125-node topology returns exactly 125 nodes."""
    nodes, tau, cloud = topo_125
    assert len(nodes) == 125


def test_delay_matrix_diagonal_zero(topo_125: tuple) -> None:
    """Delay matrix diagonal is all zeros (no self-latency)."""
    nodes, tau, cloud = topo_125
    assert np.all(np.diag(tau) == 0.0)


def test_delay_matrix_nonneg(topo_125: tuple) -> None:
    """All delay values are non-negative."""
    nodes, tau, cloud = topo_125
    assert np.all(tau >= 0.0)


def test_delay_matrix_within_mesh_max(topo_125: tuple) -> None:
    """All off-diagonal delays are at most TAU_MESH_MAX."""
    nodes, tau, cloud = topo_125
    off_diag = tau[~np.eye(tau.shape[0], dtype=bool)]
    assert np.all(off_diag <= TAU_MESH_MAX + 1e-12)


def test_delay_matrix_symmetric(topo_125: tuple) -> None:
    """Delay matrix is symmetric (tau_ij == tau_ji)."""
    nodes, tau, cloud = topo_125
    assert np.allclose(tau, tau.T)


def test_node_f_max_in_range(topo_125: tuple) -> None:
    """All node f_max values are in [F_MAX_MIN, F_MAX_MAX]."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert (
            F_MAX_MIN <= node.f_max <= F_MAX_MAX
        ), f"Node {node.id}: f_max={node.f_max} out of [{F_MAX_MIN}, {F_MAX_MAX}]"


def test_node_kappa_in_range(topo_125: tuple) -> None:
    """All node kappa values are in [KAPPA_MIN, KAPPA_MAX]."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert (
            KAPPA_MIN <= node.kappa <= KAPPA_MAX
        ), f"Node {node.id}: kappa={node.kappa} out of [{KAPPA_MIN}, {KAPPA_MAX}]"


def test_node_rho_in_range(topo_125: tuple) -> None:
    """All node rho values are in [RHO_MIN, RHO_MAX]."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert RHO_MIN <= node.rho <= RHO_MAX


def test_node_pi_E_in_range(topo_125: tuple) -> None:
    """All node pi_E values are in [PI_E_MIN, PI_E_MAX]."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert PI_E_MIN <= node.pi_E <= PI_E_MAX


def test_node_beta_in_range(topo_125: tuple) -> None:
    """All node beta values are in [BETA_MIN, BETA_MAX]."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert BETA_MIN <= node.beta <= BETA_MAX


def test_node_P_idle_is_valid_option(topo_125: tuple) -> None:
    """All node P_idle values are drawn from P_IDLE_OPTIONS."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert node.P_idle in P_IDLE_OPTIONS, f"Node {node.id}: P_idle={node.P_idle}"


def test_nodes_are_heterogeneous(topo_125: tuple) -> None:
    """Not all nodes have the same f_max (confirms heterogeneous sampling)."""
    nodes, tau, cloud = topo_125
    f_max_values = {n.f_max for n in nodes}
    assert len(f_max_values) > 1


def test_hotspot_assignment(topo_125: tuple) -> None:
    """First 5 nodes are hotspots (arrival_rate_multiplier == 5.0)."""
    nodes, tau, cloud = topo_125
    for node in nodes[:5]:
        assert node.arrival_rate_multiplier == 5.0


def test_quiet_node_assignment(topo_125: tuple) -> None:
    """Nodes 5-24 are quiet (arrival_rate_multiplier == 0.2)."""
    nodes, tau, cloud = topo_125
    for node in nodes[5:25]:
        assert node.arrival_rate_multiplier == 0.2


def test_normal_node_assignment(topo_125: tuple) -> None:
    """Remaining nodes have arrival_rate_multiplier == 1.0."""
    nodes, tau, cloud = topo_125
    for node in nodes[25:]:
        assert node.arrival_rate_multiplier == 1.0


def test_peer_delays_populated(topo_125: tuple) -> None:
    """Each node's peer_delays dict has n_nodes-1 entries."""
    nodes, tau, cloud = topo_125
    for node in nodes:
        assert len(node.peer_delays) == len(nodes) - 1


def test_cloud_tau_c_positive(topo_125: tuple) -> None:
    """Cloud RTT is strictly positive."""
    nodes, tau, cloud = topo_125
    assert cloud.tau_c > 0.0


def test_cloud_tau_c_reasonable(topo_125: tuple) -> None:
    """Cloud RTT is within a realistic range (5 ms to 200 ms)."""
    nodes, tau, cloud = topo_125
    assert 0.005 <= cloud.tau_c <= 0.200


# ---------------------------------------------------------------------------
# EUA topology - missing file
# ---------------------------------------------------------------------------


def test_eua_topology_raises_on_missing_file() -> None:
    """create_topology with eua_melbourne source raises FileNotFoundError if CSV absent."""
    rng = np.random.default_rng(RNG_SEED)
    with pytest.raises(FileNotFoundError):
        create_topology(CFG_EUA, rng)


# ---------------------------------------------------------------------------
# Topology - reproducibility
# ---------------------------------------------------------------------------


def test_topology_reproducible_with_same_seed() -> None:
    """Same seed produces identical f_max values."""
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    nodes1, _, _ = create_topology(CFG_SYNTHETIC_50, rng1)
    nodes2, _, _ = create_topology(CFG_SYNTHETIC_50, rng2)
    for n1, n2 in zip(nodes1, nodes2):
        assert math.isclose(n1.f_max, n2.f_max, rel_tol=1e-10)


def test_topology_different_seeds_differ() -> None:
    """Different seeds produce different f_max values."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    nodes1, _, _ = create_topology(CFG_SYNTHETIC_50, rng1)
    nodes2, _, _ = create_topology(CFG_SYNTHETIC_50, rng2)
    f_max_1 = [n.f_max for n in nodes1]
    f_max_2 = [n.f_max for n in nodes2]
    assert f_max_1 != f_max_2
