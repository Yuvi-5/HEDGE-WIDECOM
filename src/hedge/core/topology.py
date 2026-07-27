"""Topology builder: creates heterogeneous edge node graphs for HEDGE simulations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from hedge.core.cloud import CloudNode
from hedge.core.constants import (
    BETA_MAX,
    BETA_MIN,
    COVERAGE_RADIUS_KM_DEFAULT,
    F_MAX_MAX,
    F_MAX_MIN,
    KAPPA_MAX,
    KAPPA_MIN,
    N_NODES_DEFAULT,
    P_IDLE_OPTIONS,
    PI_E_MAX,
    PI_E_MIN,
    RHO_MAX,
    RHO_MIN,
    TAU_C_MEAN,
    TAU_C_STD,
    TAU_MESH_MAX,
)
from hedge.core.node import HEDGEEdgeServer


def create_topology(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[HEDGEEdgeServer], np.ndarray, CloudNode]:
    """Build the simulation topology from config.

    Routes to the synthetic generator or EUA Melbourne loader based on
    config['topology']['source']. Returns nodes, delay matrix, and cloud node.

    Args:
        config: Experiment configuration dict (loaded from configs/default.yaml).
        rng: Seeded random number generator for reproducible heterogeneity.

    Returns:
        nodes: List of HEDGEEdgeServer instances (length = N_nodes).
        tau: Symmetric delay matrix, shape (N_nodes, N_nodes), diagonal = 0 (seconds).
        cloud: CloudNode with RTT sampled from config values.
    """
    topo_cfg = config.get("topology", {})
    source = topo_cfg.get("source", "synthetic")
    n_nodes = int(topo_cfg.get("N_nodes", N_NODES_DEFAULT))

    if source == "synthetic":
        return _create_synthetic_topology(n_nodes=n_nodes, config=config, rng=rng)
    elif source == "eua_melbourne":
        data_path = Path(topo_cfg.get("eua_data_path", "data/eua_melbourne.csv"))
        return _load_eua_topology(data_path=data_path, n_nodes=n_nodes, config=config, rng=rng)
    else:
        raise ValueError(
            f"Unknown topology source: '{source}'. Use 'synthetic' or 'eua_melbourne'."
        )


def _create_synthetic_topology(
    n_nodes: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[HEDGEEdgeServer], np.ndarray, CloudNode]:
    """Generate a synthetic heterogeneous topology with log-uniform hardware params.

    Args:
        n_nodes: Number of edge servers to create.
        config: Experiment configuration dict.
        rng: Seeded random number generator.

    Returns:
        nodes, tau, cloud as described in create_topology.
    """
    load_cfg = config.get("arrivals", {})
    topo_cfg = config.get("topology", {})
    n_hotspots: int = int(load_cfg.get("n_hotspots", 5))
    hotspot_mult: float = float(load_cfg.get("hotspot_rate_multiplier", 5.0))
    quiet_mult: float = float(load_cfg.get("quiet_rate_multiplier", 0.2))
    n_quiet: int = 20
    heterogeneous: bool = bool(topo_cfg.get("heterogeneous", True))
    fat_tail: bool = bool(load_cfg.get("fat_tail", True))

    # Homogeneous mode (A9 ablation): every node gets identical hardware, the
    # geometric/arithmetic centre of the normal log-uniform/uniform ranges.
    homog_f_max = math.sqrt(F_MAX_MIN * F_MAX_MAX)
    homog_kappa = math.sqrt(KAPPA_MIN * KAPPA_MAX)
    homog_rho = (RHO_MIN + RHO_MAX) / 2.0
    homog_pi_E = (PI_E_MIN + PI_E_MAX) / 2.0
    homog_beta = (BETA_MIN + BETA_MAX) / 2.0
    homog_P_idle = float(np.mean(P_IDLE_OPTIONS))

    nodes: list[HEDGEEdgeServer] = []
    for i in range(n_nodes):
        if heterogeneous:
            f_max = float(10 ** rng.uniform(math.log10(F_MAX_MIN), math.log10(F_MAX_MAX)))
            kappa = float(10 ** rng.uniform(math.log10(KAPPA_MIN), math.log10(KAPPA_MAX)))
            rho = float(rng.uniform(RHO_MIN, RHO_MAX))
            pi_E = float(rng.uniform(PI_E_MIN, PI_E_MAX))
            beta = float(rng.uniform(BETA_MIN, BETA_MAX))
            p_idle_idx = int(rng.integers(0, len(P_IDLE_OPTIONS)))
            P_idle = P_IDLE_OPTIONS[p_idle_idx]
        else:
            f_max = homog_f_max
            kappa = homog_kappa
            rho = homog_rho
            pi_E = homog_pi_E
            beta = homog_beta
            P_idle = homog_P_idle

        server = HEDGEEdgeServer(
            obj_id=i + 1,
            f_max=f_max,
            kappa=kappa,
            rho=rho,
            P_idle=P_idle,
            beta=beta,
            pi_E=pi_E,
        )

        # Load tier assignment (SYSTEM_MODEL.md Section 2)
        if not fat_tail:
            server.arrival_rate_multiplier = 1.0
        elif i < n_hotspots:
            server.arrival_rate_multiplier = hotspot_mult
        elif i < n_hotspots + n_quiet:
            server.arrival_rate_multiplier = quiet_mult
        else:
            server.arrival_rate_multiplier = 1.0

        nodes.append(server)

    tau = _build_delay_matrix(n_nodes=n_nodes, rng=rng)
    _assign_peer_delays(nodes=nodes, tau=tau)
    _assign_coverage_sets(
        nodes=nodes,
        radius_km=float(topo_cfg.get("coverage_radius_km", COVERAGE_RADIUS_KM_DEFAULT)),
    )
    cloud = _sample_cloud(config=config, rng=rng)

    return nodes, tau, cloud


def _load_eua_topology(
    data_path: Path,
    n_nodes: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[HEDGEEdgeServer], np.ndarray, CloudNode]:
    """Load EUA Melbourne topology from CSV using the Phase 8 loader.

    Args:
        data_path: Path to eua_melbourne.csv.
        n_nodes: Number of nodes to use (125 for full dataset, 50 for default).
        config: Experiment configuration dict.
        rng: Seeded random number generator.

    Returns:
        nodes, tau, cloud as described in create_topology.

    Raises:
        FileNotFoundError: If data_path does not exist.
    """
    from data.loaders.eua_loader import load_eua_topology as _eua_load

    nodes, tau, cloud = _eua_load(data_path=data_path, n_nodes=n_nodes, config=config, rng=rng)
    _assign_peer_delays(nodes=nodes, tau=tau)
    topo_cfg = config.get("topology", {})
    _assign_coverage_sets(
        nodes=nodes,
        radius_km=float(topo_cfg.get("coverage_radius_km", COVERAGE_RADIUS_KM_DEFAULT)),
    )
    return nodes, tau, cloud


def _build_delay_matrix(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Build a symmetric propagation delay matrix with diagonal = 0.

    Args:
        n_nodes: Number of nodes.
        rng: Seeded random number generator.

    Returns:
        tau: float64 array of shape (n_nodes, n_nodes), values in [0, TAU_MESH_MAX].
    """
    upper = rng.uniform(0.0, TAU_MESH_MAX, size=(n_nodes, n_nodes))
    tau = (upper + upper.T) / 2.0
    np.fill_diagonal(tau, 0.0)
    return tau


def _assign_peer_delays(nodes: list[HEDGEEdgeServer], tau: np.ndarray) -> None:
    """Populate each node's peer_delays dict from the delay matrix.

    Args:
        nodes: List of HEDGEEdgeServer instances.
        tau: Delay matrix of shape (len(nodes), len(nodes)).
    """
    for i, node in enumerate(nodes):
        node.peer_delays = {nodes[j].id: float(tau[i, j]) for j in range(len(nodes)) if j != i}


def _assign_coverage_sets(nodes: list[HEDGEEdgeServer], radius_km: float) -> None:
    """Populate each node's coverage_peers: nodes within radius_km of its own location.

    This is C_u, the Phase-0 physical radio-coverage-radius constraint on
    which nodes may compete for the
    orchestrator role for a task arriving at this node. Every node always
    includes itself. Falls back to the full node list (today's exact
    pre-fix behavior) when coordinates are unavailable (synthetic topology
    has no geography), so this is a provable no-op for every synthetic
    config.

    Args:
        nodes: List of HEDGEEdgeServer instances (coordinates may be None).
        radius_km: Coverage radius in kilometres.
    """
    if any(node.coordinates is None for node in nodes):
        for node in nodes:
            node.coverage_peers = list(nodes)
        return

    from data.loaders.eua_loader import haversine_distance_km  # noqa: PLC0415

    for node in nodes:
        lat_i, lon_i = node.coordinates
        peers = [
            other
            for other in nodes
            if other is node
            or haversine_distance_km(lat_i, lon_i, other.coordinates[0], other.coordinates[1])
            <= radius_km
        ]
        node.coverage_peers = peers


def _sample_cloud(config: dict[str, Any], rng: np.random.Generator) -> CloudNode:
    """Sample cloud RTT and return a CloudNode.

    Args:
        config: Experiment configuration dict.
        rng: Seeded random number generator.

    Returns:
        CloudNode with tau_c drawn from N(TAU_C_MEAN, TAU_C_STD), clamped > 0.
    """
    topo_cfg = config.get("topology", {})
    tau_c_mean: float = float(topo_cfg.get("tau_c_mean", TAU_C_MEAN))
    tau_c_std: float = float(topo_cfg.get("tau_c_std", TAU_C_STD))
    tau_c = max(0.001, float(rng.normal(tau_c_mean, tau_c_std)))
    return CloudNode(tau_c=tau_c)
