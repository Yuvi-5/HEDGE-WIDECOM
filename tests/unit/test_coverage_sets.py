"""Coverage-set (C_u) architecture gate tests.

Gate criteria:
- _assign_coverage_sets: symmetric, self-inclusive, never empty, synthetic no-op.
- eua_loader.load_eua_topology populates node.coordinates end-to-end.
- Real-data C_u size distribution lands near the paper's "5 to 15 typical" target.
- Coverage gating actually narrows the Phase-0 candidate pool below N_nodes.
- I9 (K_max=0 => cloud-only) still holds under coverage gating.
- cluster_hotspots produces a spatially coherent hotspot cluster and composes
  correctly with correlate_hotspot_weak_hardware.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.loaders.eua_loader import haversine_distance_km, load_eua_topology
from hedge.core.topology import _assign_coverage_sets, create_topology

_EUA_CSV = Path("data/eua_melbourne.csv")

CFG_SYNTHETIC = {
    "topology": {"source": "synthetic", "N_nodes": 20, "tau_c_mean": 0.035, "tau_c_std": 0.008},
    "arrivals": {"n_hotspots": 3, "hotspot_rate_multiplier": 5.0, "quiet_rate_multiplier": 0.2},
}


def _cfg_eua(n_nodes: int, radius_km: float = 0.25, **arrivals_extra: object) -> dict:
    return {
        "topology": {
            "source": "eua_melbourne",
            "N_nodes": n_nodes,
            "eua_data_path": str(_EUA_CSV),
            "coverage_radius_km": radius_km,
        },
        "arrivals": {
            "n_hotspots": 5,
            "hotspot_rate_multiplier": 8.0,
            "quiet_rate_multiplier": 0.2,
            **arrivals_extra,
        },
    }


# ---------------------------------------------------------------------------
# _assign_coverage_sets unit correctness
# ---------------------------------------------------------------------------


def test_synthetic_topology_coverage_is_full_list_noop() -> None:
    """Synthetic topology (no coordinates) falls back to the full node list."""
    nodes, _tau, _cloud = create_topology(CFG_SYNTHETIC, np.random.default_rng(0))
    assert all(node.coordinates is None for node in nodes)
    for node in nodes:
        assert len(node.coverage_peers) == len(nodes)
        assert set(n.unique_id for n in node.coverage_peers) == {
            n.unique_id for n in nodes
        }


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_eua_loader_populates_coordinates() -> None:
    """load_eua_topology sets real (lat, lon) coordinates on every node."""
    rng = np.random.default_rng(0)
    nodes, _tau, _cloud = load_eua_topology(
        _EUA_CSV, n_nodes=50, config={"arrivals": {}, "topology": {}}, rng=rng
    )
    assert all(node.coordinates is not None for node in nodes)
    for node in nodes:
        lat, lon = node.coordinates
        # Melbourne CBD bounding box (generous margin)
        assert -38.0 <= lat <= -37.5
        assert 144.5 <= lon <= 145.5


def test_assign_coverage_sets_self_inclusion_and_symmetry() -> None:
    """Every node includes itself; coverage is symmetric under a uniform radius."""
    from hedge.core.node import HEDGEEdgeServer

    nodes = [
        HEDGEEdgeServer(
            obj_id=i + 1, f_max=3e9, kappa=1e-26, rho=5e-4, P_idle=30.0, beta=200.0,
            pi_E=0.12, coordinates=(-37.81 + 0.001 * i, 144.96 + 0.001 * i),
        )
        for i in range(10)
    ]
    _assign_coverage_sets(nodes, radius_km=0.5)
    for node in nodes:
        assert node in node.coverage_peers, "self must always be in own coverage_peers"
        assert len(node.coverage_peers) >= 1
    for i, node_i in enumerate(nodes):
        for node_j in node_i.coverage_peers:
            j = nodes.index(node_j)
            assert nodes[i] in nodes[j].coverage_peers, (
                f"asymmetric coverage: {i} sees {j} but not vice versa"
            )


def test_assign_coverage_sets_never_empty_even_at_tiny_radius() -> None:
    """A radius smaller than any real inter-node distance still yields self-only coverage."""
    from hedge.core.node import HEDGEEdgeServer

    nodes = [
        HEDGEEdgeServer(
            obj_id=i + 1, f_max=3e9, kappa=1e-26, rho=5e-4, P_idle=30.0, beta=200.0,
            pi_E=0.12, coordinates=(-37.81 + i, 144.96 + i),  # ~100km+ apart
        )
        for i in range(5)
    ]
    _assign_coverage_sets(nodes, radius_km=0.001)
    for node in nodes:
        assert node.coverage_peers == [node], "isolated node must still see itself"


# ---------------------------------------------------------------------------
# Empirical C_u size distribution (real loader path, not a hand-rolled recompute)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_coverage_set_size_distribution_at_full_dataset() -> None:
    """At N_nodes=125 (full dataset) and the calibrated radius, C_u sizes land
    near the paper's '5 to 15 typical candidate nodes' claim (Section subsec:phase0).
    """
    nodes, _tau, _cloud = create_topology(_cfg_eua(125), np.random.default_rng(0))
    sizes = [len(n.coverage_peers) for n in nodes]
    mean_size = float(np.mean(sizes))
    assert 5.0 <= mean_size <= 20.0, f"mean C_u size {mean_size} far outside target band"
    assert min(sizes) >= 1, "every node must see at least itself"


# ---------------------------------------------------------------------------
# Integration: coverage gating actually narrows the candidate pool
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_coverage_gating_narrows_phase0_candidates() -> None:
    """Under coverage gating, a real-data engine run's Phase-0 candidate pool
    is bounded by the home node's own coverage_peers, and typically smaller
    than the full topology -- proves the gate is active, not silently
    falling through to the full node list."""
    import sys

    sys.path.insert(0, "src")
    from experiments.runner import load_config
    from hedge.simulation import engine as engine_mod
    from hedge.market import phase0 as phase0_mod

    cfg = load_config(Path("configs/real_heavy.yaml"))
    cfg["simulation"]["duration"] = 20.0
    cfg["simulation"]["warmup_duration"] = 0.0
    cfg["output"]["save_events"] = False
    cfg["output"]["save_snapshots"] = False
    cfg["experiment"] = {"algorithm": "HEDGE"}

    engine = engine_mod.HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
    n_total = len(engine.nodes)

    orig_run_phase0 = phase0_mod.run_phase0
    seen_sizes: list[int] = []

    def traced_run_phase0(candidate_nodes, *args, **kwargs):
        seen_sizes.append(len(candidate_nodes))
        return orig_run_phase0(candidate_nodes, *args, **kwargs)

    engine_mod.run_phase0 = traced_run_phase0
    engine.run(n_tasks=200)

    assert seen_sizes, "Phase-0 was never called"
    assert all(size <= n_total for size in seen_sizes)
    assert any(size < n_total for size in seen_sizes), (
        "coverage gating never narrowed the candidate pool below the full topology"
    )


def test_i9_kmax_zero_still_cloud_only_under_coverage_gating() -> None:
    """K_max=0 => empty peer pool => cloud-only, even with coverage gating active
    (self-inclusion guarantees Phase-0 itself never raises on an empty pool)."""
    import sys

    sys.path.insert(0, "src")
    from experiments.runner import load_config
    from hedge.simulation import engine as engine_mod

    cfg = load_config(Path("configs/default.yaml"))
    cfg["simulation"]["duration"] = 10.0
    cfg["simulation"]["warmup_duration"] = 0.0
    cfg["output"]["save_events"] = False
    cfg["output"]["save_snapshots"] = False
    cfg["experiment"] = {"algorithm": "HEDGE"}
    cfg.setdefault("hedge", {})["K_max"] = 0

    engine = engine_mod.HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
    metrics = engine.run(n_tasks=100)
    assert metrics["M9_edge_service_fraction"] == 0.0, "K_max=0 must never serve at the edge"


# ---------------------------------------------------------------------------
# Hotspot spatial clustering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_cluster_hotspots_produces_spatially_coherent_cluster() -> None:
    """cluster_hotspots=True: hotspot nodes are physically closer together
    than a random same-size sample of the topology."""
    cfg = _cfg_eua(125, n_hotspots=5, hotspot_rate_multiplier=8.0, cluster_hotspots=True)
    nodes, _tau, _cloud = create_topology(cfg, np.random.default_rng(0))
    hotspots = [n for n in nodes if n.arrival_rate_multiplier == 8.0]
    assert len(hotspots) == 5

    def mean_pairwise_km(ns: list) -> float:
        dists = [
            haversine_distance_km(*ns[i].coordinates, *ns[j].coordinates)
            for i in range(len(ns))
            for j in range(i + 1, len(ns))
        ]
        return float(np.mean(dists))

    hotspot_dist = mean_pairwise_km(hotspots)
    rng = np.random.default_rng(1)
    random_sample = list(rng.choice(nodes, size=len(hotspots), replace=False))
    random_dist = mean_pairwise_km(random_sample)
    assert hotspot_dist < random_dist, "clustered hotspots should be tighter than random"


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_cluster_hotspots_composes_with_weak_hardware_seed() -> None:
    """cluster_hotspots + correlate_hotspot_weak_hardware: the cluster seed is
    the weakest-f_max node, not a random one."""
    cfg = _cfg_eua(
        125,
        n_hotspots=5,
        hotspot_rate_multiplier=8.0,
        cluster_hotspots=True,
        correlate_hotspot_weak_hardware=True,
    )
    nodes, _tau, _cloud = create_topology(cfg, np.random.default_rng(0))
    weakest = min(nodes, key=lambda n: n.f_max)
    assert weakest.arrival_rate_multiplier == 8.0, (
        "weakest-f_max node should be the hotspot cluster seed and thus a hotspot itself"
    )


def test_cluster_hotspots_falls_back_when_no_coordinates() -> None:
    """cluster_hotspots is a no-op (falls back to scattered assignment) for
    synthetic topology, which has no geography to cluster over."""
    cfg = {
        "topology": {"source": "synthetic", "N_nodes": 20},
        "arrivals": {
            "n_hotspots": 3,
            "hotspot_rate_multiplier": 5.0,
            "quiet_rate_multiplier": 0.2,
            "cluster_hotspots": True,
        },
    }
    nodes, _tau, _cloud = create_topology(cfg, np.random.default_rng(0))
    hotspots = [n for n in nodes if n.arrival_rate_multiplier == 5.0]
    assert len(hotspots) == 3
