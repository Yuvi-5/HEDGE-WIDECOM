"""HEDGE simulation engine: discrete-time cadence loop with RFQ market protocol.

Usage (CLI):
    python src/hedge/simulation/engine.py \\
        --config configs/default.yaml --n-tasks 100 --seed 0 \\
        --output outputs/smoke_test/

Internal design:
    - One simulation tick = delta_tp seconds (cadence period).
    - Each tick: generate Poisson arrivals -> run RFQ per task -> cadence_tick (Kalman + quotes).
    - No SimPy or Mesa event loop; pure loop for determinism and testability.
    - Kalman predictors and EMA revenue trackers are held in engine-level dicts keyed by
      node.unique_id and synced to node.l_hat / node.R_hat each cadence tick.
    - Algorithm is pluggable: set config["experiment"]["algorithm"] to "HEDGE" (default)
      or a baseline key from BASELINE_REGISTRY. The admission gate is gated by
      config["market"]["enable_admission_gate"].
"""

from __future__ import annotations

import argparse
import heapq
import inspect
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from loguru import logger

from hedge.core.constants import (
    ALPHA_R,
    DELTA_TP,
    JOULES_PER_KWH,
    K_MAX,
    LAMBDA_MAX,
    R_REF,
    THETA_RISK,
    W_FAIL,
)
from hedge.core.topology import create_topology
from hedge.market.afgm import compute_cloud_latency, compute_node_latency
from hedge.market.phase0 import run_phase0
from hedge.market.rfq import RFQResult, run_rfq_round
from hedge.predictor.ema import EMARevenueTracker
from hedge.predictor.instantaneous import InstantaneousLoadFilter
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price
from hedge.pricing.layer2_5 import (
    compute_aggressiveness,
    compute_competitiveness_signal,
    compute_spa_price,
)
from hedge.simulation.scheduler import (
    PoissonScheduler,
    TraceScheduler,
)
from metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Unified allocation outcome (normalises RFQResult and baseline dict returns)
# ---------------------------------------------------------------------------


@dataclass
class _AllocOutcome:
    """Minimal normalised view of an allocation result for engine dispatch."""

    executor_type: str  # "edge" | "cloud" | "unmatched"
    winner: Any  # edge node if executor_type=="edge", else None
    price_total: float
    C1_winner: float
    latency_est: float
    message_count: int = 0


def _rfq_to_outcome(result: RFQResult) -> _AllocOutcome:
    """Convert RFQResult to _AllocOutcome."""
    return _AllocOutcome(
        executor_type=result.executor_type,
        winner=result.winner if result.executor_type == "edge" else None,
        price_total=result.price_total if math.isfinite(result.price_total) else 0.0,
        C1_winner=result.C1_winner,
        latency_est=0.0,  # filled in by caller after latency computation
        message_count=result.message_count,
    )


def _dict_to_outcome(
    result_dict: dict[str, Any],
    node_by_uid: dict[int, Any],
) -> _AllocOutcome:
    """Convert baseline dict result {status, executor_id, price_total, latency, C1_total}.

    Args:
        result_dict: Dict returned by B2-B7 baselines.
        node_by_uid: Map from unique_id (int) to node object.

    Returns:
        Normalised _AllocOutcome.
    """
    status = str(result_dict.get("status", "rejected"))
    executor_id = str(result_dict.get("executor_id", ""))
    price_total = float(result_dict.get("price_total", 0.0))
    latency = float(result_dict.get("latency", 0.0))
    C1_total = float(result_dict.get("C1_total", 0.0))

    if status != "completed":
        return _AllocOutcome(
            executor_type="unmatched",
            winner=None,
            price_total=0.0,
            C1_winner=0.0,
            latency_est=latency,
        )

    if executor_id == "cloud" or executor_id == "":
        return _AllocOutcome(
            executor_type="cloud",
            winner=None,
            price_total=price_total,
            C1_winner=C1_total,
            latency_est=latency,
        )

    # Edge win: resolve node by unique_id
    try:
        uid = int(executor_id)
        winner_node = node_by_uid.get(uid)
    except (ValueError, TypeError):
        winner_node = None

    if winner_node is None:
        return _AllocOutcome(
            executor_type="cloud",
            winner=None,
            price_total=price_total,
            C1_winner=C1_total,
            latency_est=latency,
        )

    return _AllocOutcome(
        executor_type="edge",
        winner=winner_node,
        price_total=price_total,
        C1_winner=C1_total,
        latency_est=latency,
    )


# ---------------------------------------------------------------------------
# Baseline signature adapter
# ---------------------------------------------------------------------------


def _make_baseline_adapter(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a baseline callable so it accepts the engine's standard kwargs.

    The engine always calls allocators with:
        peer_pool, task, cloud, tau_dict, peer_standing_quotes, k_max, alpha_u, gamma_u

    Baselines have varying subsets of these parameters. This function inspects the
    baseline's signature and creates a wrapper that only forwards accepted args.

    Args:
        fn: Baseline callable (may be a function or a bound method).

    Returns:
        Wrapper callable with signature (*engine_args, **engine_kwargs) -> Any.
    """
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
    except (ValueError, TypeError):
        accepted = set()

    # Only explicitly named parameters from the engine's standard call set.
    # We NEVER forward k_max through **kwargs to avoid conflicting with baselines
    # that set their own k_max internally (e.g., B1 sets k_max=0 unconditionally).
    _engine_kwargs: dict[str, Any] = {
        "peer_pool": None,
        "task": None,
        "cloud": None,
        "tau_dict": None,
        "peer_standing_quotes": None,
        "k_max": None,
        "alpha_u": None,
        "gamma_u": None,
        "lambda_max": None,
    }
    # Determine which engine kwargs the function accepts by name (ignore VAR_KEYWORD)
    _passthrough = {k for k in _engine_kwargs if k in accepted}

    def _adapted(
        peer_pool: list[Any],
        task: Any,
        cloud: Any,
        tau_dict: dict[int, float],
        peer_standing_quotes: dict[int, float] | None = None,
        k_max: int = K_MAX,
        alpha_u: float = 0.5,
        gamma_u: float = 0.5,
        lambda_max: float = LAMBDA_MAX,
        **_ignored: Any,
    ) -> Any:
        all_kwargs: dict[str, Any] = {
            "peer_pool": peer_pool,
            "task": task,
            "cloud": cloud,
            "tau_dict": tau_dict,
            "peer_standing_quotes": peer_standing_quotes,
            "k_max": k_max,
            "alpha_u": alpha_u,
            "gamma_u": gamma_u,
            "lambda_max": lambda_max,
        }
        # Pass only what the function explicitly names; never blindly forward **kwargs
        # to avoid conflicts (e.g., B1 sets k_max=0 internally and would conflict)
        call_kwargs = {k: v for k, v in all_kwargs.items() if k in _passthrough}
        return fn(**call_kwargs)

    return _adapted


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------


class HEDGESimulationEngine:
    """Discrete-time HEDGE simulation engine.

    Drives the cadence loop, task scheduling, RFQ market rounds, and metric collection.
    Does not use the EdgeSimPy/Mesa event loop; all timing is explicit.

    Algorithm is selectable via config["experiment"]["algorithm"]:
    - "HEDGE" (default): full HEDGE protocol with SPA pricing.
    - a baseline key from BASELINE_REGISTRY.
    """

    # Reference workload for standing-quote computation (1 billion cycles)
    _W_REF: float = 1e9

    def __init__(
        self,
        config: dict[str, Any],
        seed: int = 42,
        output_dir: Path | None = None,
    ) -> None:
        """Initialise engine from config.

        Args:
            config: Experiment config dict (loaded from configs/default.yaml or override).
            seed: Random seed for full reproducibility.
            output_dir: If set, write events.jsonl and summary.json here after run.
        """
        self.config = config
        self.output_dir = output_dir
        self.rng = np.random.default_rng(seed)
        # Propagate the actual per-run seed into config["simulation"]["random_seed"]
        # so downstream readers (e.g. TraceScheduler's own base_seed lookup, used
        # for trace sampling and task-mix splitting) vary correctly across seeds
        # instead of always reading whatever static value (or absence, defaulting
        # to 0) was in the loaded config file. setdefault ensures the "simulation"
        # dict is the same object subsequent config.get("simulation", {}) calls
        # see, not a fresh empty dict if the key was previously absent.
        config.setdefault("simulation", {})["random_seed"] = seed

        # Output IO flags (gate disk writes, not in-memory metric computation;
        # large sweeps should disable save_events/save_snapshots to avoid
        # writing an events.jsonl per seed x variant).
        output_cfg = config.get("output", {})
        self._save_events: bool = bool(output_cfg.get("save_events", True))
        self._save_snapshots: bool = bool(output_cfg.get("save_snapshots", False))

        sim_cfg = config.get("simulation", {})
        self.delta_tp: float = float(sim_cfg.get("delta_tp", DELTA_TP))
        self.sim_duration: float = float(sim_cfg.get("duration", 3600.0))
        self.warmup_duration: float = float(sim_cfg.get("warmup_duration", 300.0))

        hedge_cfg = config.get("hedge", {})
        self.k_max: int = int(hedge_cfg.get("K_max", K_MAX))
        self.alpha_u: float = float(hedge_cfg.get("alpha_u", 0.5))
        self.gamma_u: float = float(hedge_cfg.get("gamma_u", 0.5))
        # theta_risk / W_fail live under the existing `hedge:` config section
        # (see configs/default.yaml and configs/ablations/a8_risk_premium.yaml,
        # which sweeps the bare `theta_risk`/`W_fail` keys) -- NOT under `market:`.
        self._theta_risk: float = float(hedge_cfg.get("theta_risk", THETA_RISK))
        self._w_fail: int = int(hedge_cfg.get("W_fail", W_FAIL))

        self._lambda_max: float = float(config.get("pricing", {}).get("lambda_max", LAMBDA_MAX))

        # Market control flags (gated by config)
        market_cfg = config.get("market", {})
        self._enable_admission_gate: bool = bool(market_cfg.get("enable_admission_gate", True))
        self._enable_phase0: bool = bool(market_cfg.get("enable_phase0", True))
        self._enable_gc_risk_premium: bool = bool(
            market_cfg.get("enable_gc_risk_premium", True)
        )
        # Coverage-set (C_u) gating: restrict Phase-0's candidate pool to
        # physical radio-coverage neighbours instead of the full topology.
        # Defensive fallback: node.coverage_peers is always populated (falls
        # back to the full node list for synthetic topology / when disabled),
        # so this flag only controls whether the engine READS that narrower
        # list or reads self.nodes directly.
        self._enable_coverage_gating: bool = bool(
            market_cfg.get("enable_coverage_gating", True)
        )

        # Algorithm selection
        exp_cfg = config.get("experiment", {})
        algorithm_name: str = str(exp_cfg.get("algorithm", "HEDGE")).upper()
        self._algorithm_name = algorithm_name
        self._allocate: Callable[..., Any] = self._resolve_allocator(algorithm_name)

        # Topology
        self.nodes, self.tau_matrix, self.cloud = create_topology(config, self.rng)
        logger.info(f"Topology: {len(self.nodes)} nodes, cloud tau_c={self.cloud.tau_c:.4f}s")

        # Per-node predictors keyed by node.unique_id
        pred_cfg = config.get("predictor", {})
        self._load_predictors: dict[int, Any] = {}
        self._emas: dict[int, EMARevenueTracker] = {}
        self._init_predictors(pred_cfg)

        # Task generation
        arrivals_mode = str(config.get("arrivals", {}).get("mode", "poisson")).lower()
        _VALID_ARRIVALS_MODES = {"poisson", "alibaba"}
        if arrivals_mode not in _VALID_ARRIVALS_MODES:
            raise ValueError(
                f"Unknown arrivals.mode '{arrivals_mode}'; expected one of "
                f"{sorted(_VALID_ARRIVALS_MODES)}."
            )
        self.scheduler: PoissonScheduler | TraceScheduler
        if arrivals_mode == "alibaba":
            self.scheduler = TraceScheduler(self.nodes, config, self.rng)
        else:
            self.scheduler = PoissonScheduler(self.nodes, config, self.rng)

        # Real metrics collector (MetricsCollector from src/metrics/)
        # Burst window for M12 (TSFR): only set via an explicit config["metrics"]
        # override; defaults to no burst window (steady-state arrivals only).
        metrics_cfg = config.get("metrics", {})
        burst_start = float(metrics_cfg.get("burst_start", -1.0))
        burst_duration = float(metrics_cfg.get("burst_duration", 0.0))
        self._mc = MetricsCollector(burst_start=burst_start, burst_duration=burst_duration)

        # Min-heap for task completions: (completion_time, node_uid, w_cycles, task_id)
        self._completion_heap: list[tuple[float, int, float, str]] = []

        # Map node_uid -> node for O(1) lookup
        self._node_by_uid: dict[int, Any] = {n.unique_id: n for n in self.nodes}

        # --- Phase 2 performance caches ---
        # Pre-compute per-home-node tau_dicts (avoids per-RFQ dict rebuild)
        self._tau_dicts: list[dict[int, float]] = [
            {self.nodes[j].unique_id: float(self.tau_matrix[i][j]) for j in range(len(self.nodes))}
            for i in range(len(self.nodes))
        ]

        # Cached peer standing quotes: updated in _cadence_tick, not per-RFQ
        self._peer_standing_quotes: dict[int, float] = {
            n.unique_id: n.p_star_ref for n in self.nodes
        }

        # Realised per-node settlement revenue accumulated since the last cadence
        # tick snapshot (M14 RMSE_revenue); cleared at the end of each _cadence_tick.
        self._revenue_since_tick: dict[int, float] = {}

        # Online P_fail(i) sliding-window estimator (Eq 9): fraction of a
        # node's last W_fail completed tasks that missed
        # their deadline. Tracked and exposed on node.p_fail_hat for reporting/A8
        # diagnostics; NOT used to gate the GC risk-premium entry condition itself
        # (Eq 6, the only fully-specified inequality) since Eqs 7-8 relating
        # theta_risk to P_fail are not given in full in the available docs.
        self._sla_outcome_window: dict[int, deque[bool]] = {
            n.unique_id: deque(maxlen=self._w_fail) for n in self.nodes
        }
        for n in self.nodes:
            n.p_fail_hat = 0.0

    # ------------------------------------------------------------------
    # Algorithm resolution
    # ------------------------------------------------------------------

    def _resolve_allocator(
        self,
        algorithm_name: str,
    ) -> Callable[..., Any]:
        """Return the allocation callable for the chosen algorithm.

        For HEDGE: returns run_rfq_round.
        For a baseline: returns the corresponding function from BASELINE_REGISTRY.

        Args:
            algorithm_name: "HEDGE" or a baseline key from BASELINE_REGISTRY.

        Returns:
            Callable with signature compatible with _dispatch_rfq.
        """
        if algorithm_name == "HEDGE":
            return _make_baseline_adapter(run_rfq_round)

        from hedge.baselines import BASELINE_REGISTRY  # noqa: PLC0415

        if algorithm_name not in BASELINE_REGISTRY:
            logger.warning(f"Unknown algorithm '{algorithm_name}'; falling back to HEDGE.")
            return run_rfq_round

        entry = BASELINE_REGISTRY[algorithm_name]
        return _make_baseline_adapter(entry)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_predictors(self, pred_cfg: dict[str, Any]) -> None:
        """Create one load predictor and one EMARevenueTracker per node.

        The load predictor is InstantaneousLoadFilter (raw y_i as l_hat_i,
        A4 "No Kalman" in ablation terms): predictor.mode is otherwise
        unused since this repo only ships that one predictor implementation.
        """
        mode: str = str(pred_cfg.get("mode", "instantaneous")).lower()
        if mode != "instantaneous":
            logger.warning(f"Unknown predictor.mode '{mode}'; falling back to instantaneous.")
        pricing_cfg = self.config.get("pricing", {})
        alpha_R: float = float(pricing_cfg.get("alpha_R", ALPHA_R))
        R_ref: float = float(pricing_cfg.get("R_ref", R_REF))

        for node in self.nodes:
            self._load_predictors[node.unique_id] = InstantaneousLoadFilter(f_max=node.f_max)
            self._emas[node.unique_id] = EMARevenueTracker(R_ref=R_ref, alpha_R=alpha_R)
            node.R_hat = R_ref  # initialise EMA to R_ref

    # ------------------------------------------------------------------
    # Cadence tick (Phase 2: O(N) standing-quote update)
    # ------------------------------------------------------------------

    def _update_standing_quotes(self, sim_time: float) -> None:
        """Recompute p_star_ref and p_dagger_ref for all nodes.

        Phase 2 optimisation: precompute global min/second-min of p_star_ref to
        reduce per-node competitiveness signal from O(N) filter to O(1) lookup.

        Also records per-node snapshots and quote history for MetricsCollector.

        Args:
            sim_time: Current simulation time (used for snapshot timestamps).
        """
        # Step 1: compute p_star for all nodes (O(N))
        for node in self.nodes:
            C1_ref, _, _, _ = compute_layer1_cost(
                self._W_REF,
                node.f_max,
                node.kappa,
                node.P_idle,
                node.rho,
                node.pi_E,
                node.beta,
            )
            m_i = compute_markup(node.l_hat, node.f_max)
            p_star = compute_stackelberg_price(C1_ref, self._W_REF, m_i)
            node.p_star_ref = p_star

        # Step 2: precompute global min and second-min (O(N))
        if len(self.nodes) >= 2:
            p_vals = sorted((n.p_star_ref, n.unique_id) for n in self.nodes)
            global_min, min_uid = p_vals[0]
            second_min = p_vals[1][0]
        elif len(self.nodes) == 1:
            global_min = self.nodes[0].p_star_ref
            second_min = global_min
            min_uid = self.nodes[0].unique_id
        else:
            return

        # Step 3: compute p_dagger using O(1) per-node peer min (O(N) total)
        for node in self.nodes:
            peer_min = second_min if node.unique_id == min_uid else global_min
            delta_comp = compute_competitiveness_signal(node.p_star_ref, [peer_min])
            lambda_spa = compute_aggressiveness(
                node.l_hat, node.f_max, node.R_hat, lambda_max=self._lambda_max
            )
            C1_ref, _, _, _ = compute_layer1_cost(
                self._W_REF,
                node.f_max,
                node.kappa,
                node.P_idle,
                node.rho,
                node.pi_E,
                node.beta,
            )
            p_dagger = compute_spa_price(
                C1_ref, self._W_REF, node.p_star_ref, lambda_spa, delta_comp
            )
            node.p_dagger_ref = p_dagger

            # Record per-node snapshot for MetricsCollector (M3, M4, M11, M14)
            if sim_time >= self.warmup_duration:
                self._mc.record_snapshot(
                    {
                        "time": sim_time,
                        "node_id": str(node.unique_id),
                        "l_current": node.l_observed,
                        "f_max": node.f_max,
                        "l_hat": node.l_hat,
                        "R_hat": node.R_hat,
                        "R_realised": self._revenue_since_tick.get(node.unique_id, 0.0),
                    }
                )

        # Step 4: update cached peer_standing_quotes dict (O(N))
        for node in self.nodes:
            self._peer_standing_quotes[node.unique_id] = node.p_star_ref

        # Step 5: record system mean p_dagger for convergence tracking (M8)
        if sim_time >= self.warmup_duration and self.nodes:
            mean_p_dagger = sum(n.p_dagger_ref for n in self.nodes) / len(self.nodes)
            self._mc.record_quote({"time": sim_time, "p_dagger_ref": mean_p_dagger})

    def _cadence_tick(self, sim_time: float, w_arrived_per_uid: dict[int, float]) -> None:
        """Run Kalman update for each node and refresh standing quotes.

        Args:
            sim_time: Current simulation time (seconds).
            w_arrived_per_uid: Cycles submitted to each node this tick (cycles).
                               Used to compute observed load l_observed = w / delta_tp.
        """
        for node in self.nodes:
            w_tick = w_arrived_per_uid.get(node.unique_id, 0.0)
            l_observed = min(w_tick / self.delta_tp, node.f_max)
            node.l_observed = l_observed
            node.l_hat = self._load_predictors[node.unique_id].update(l_observed)
            node.R_hat = self._emas[node.unique_id].get_latest_forecast()
        self._update_standing_quotes(sim_time)
        # Algorithm-agnostic hook: any stateful allocator may expose
        # on_cadence_tick for per-tick state updates (e.g. a leased-capacity
        # auction); no-op for stateless allocators.
        if hasattr(self._allocate, "on_cadence_tick"):
            self._allocate.on_cadence_tick(sim_time, self.nodes, self.cloud)
        self._revenue_since_tick.clear()

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def _process_completions(self, sim_time: float) -> None:
        """Decrement w_pending for tasks that finished at or before sim_time."""
        while self._completion_heap and self._completion_heap[0][0] <= sim_time:
            _t, node_uid, w_task, _tid = heapq.heappop(self._completion_heap)
            node = self._node_by_uid.get(node_uid)
            if node is not None:
                node.w_pending = max(0.0, node.w_pending - w_task)

    # ------------------------------------------------------------------
    # Admission gate
    # ------------------------------------------------------------------

    def _passes_admission_gate(self, task: Any) -> bool:
        """Return True if task passes the shared IR admission gate.

        Applied uniformly across all algorithms when enable_admission_gate is True.

        Args:
            task: HEDGETask with a_buyer attribute.

        Returns:
            True if task should proceed to allocation; False to reject immediately.
        """
        if not self._enable_admission_gate:
            return True
        from hedge.baselines import admission_gate  # noqa: PLC0415

        return admission_gate(task, self.nodes, self.config, cloud=self.cloud)

    # ------------------------------------------------------------------
    # RFQ dispatch (Phase 1 core: pluggable algorithm + rich events)
    # ------------------------------------------------------------------

    def _compute_energy_carbon(self, node: Any, w: float) -> tuple[float, float]:
        """Compute energy (J) and carbon (gCO2) for executing w cycles on node.

        Args:
            node: Edge/cloud node with hardware attributes.
            w: Task workload (cycles).

        Returns:
            (energy_joules, carbon_g) both non-negative.
        """
        E = node.kappa * node.f_max**2 * w + node.P_idle * w / max(node.f_max, 1.0)
        carbon_g = node.beta / JOULES_PER_KWH * E
        return float(E), float(carbon_g)

    def _dispatch_rfq(
        self,
        sim_time: float,
        task: Any,
        home_node_idx: int,
        w_arrived_per_uid: dict[int, float],
    ) -> None:
        """Run one allocation round and record the result.

        Applies admission gate, runs the chosen algorithm, computes energy/carbon,
        feeds MetricsCollector.

        Args:
            sim_time: Current simulation time (seconds).
            task: HEDGETask arriving at home_node.
            home_node_idx: Index of arrival node in self.nodes.
            w_arrived_per_uid: Accumulator; updated in-place for matched edge tasks.
        """
        # Admission gate check (uniform IR pre-filter)
        if not self._passes_admission_gate(task):
            if sim_time >= self.warmup_duration:
                self._mc.record_event(
                    {
                        "event_type": "rejected",
                        "task_id": task.task_id,
                        "arrival_time": sim_time,
                        "deadline": task.d,
                        "latency": 0.0,
                        "price_paid": 0.0,
                        "executor_id": "",
                        "home_node_id": str(self.nodes[home_node_idx].unique_id),
                        "carbon_g": 0.0,
                        "energy_joules": 0.0,
                        "is_resale": False,
                        "a_buyer": task.a_buyer,
                        "C1_total": 0.0,
                    }
                )
            return

        # Use pre-computed tau_dict (Phase 2: no per-RFQ dict rebuild)
        tau_dict = self._tau_dicts[home_node_idx]

        # Phase 0 orchestrator election + peer-pool curation (Section 1, only for
        # the HEDGE algorithm itself; each baseline runs its own candidate
        # selection over the full node list per its own spec). This
        # replaces the previous placeholder of handing the ENTIRE node list to
        # run_rfq_round (which then silently sliced peer_pool[:k_max] by raw list
        # order every time, so the same first-k_max nodes were queried for every
        # task regardless of where it arrived). Capping the final pool at k_max
        # (not k_max+1) preserves I9 (K_max=0 => empty pool => cloud-only).
        #
        # Candidate pool is coverage-gated to C_u ("only nodes within
        # physical radio contact of u are probed") when enabled --
        # node.coverage_peers falls back to the full node list for
        # synthetic topology, so this is a no-op there.
        home_node = self.nodes[home_node_idx]
        phase0_candidates = (
            home_node.coverage_peers if self._enable_coverage_gating else self.nodes
        )
        if self._algorithm_name == "HEDGE" and self._enable_phase0:
            orchestrator, p0_pool = run_phase0(
                candidate_nodes=phase0_candidates,
                task=task,
                tau_dict=tau_dict,
                k_max=self.k_max,
                enable_gc_risk_premium=self._enable_gc_risk_premium,
                theta_risk=self._theta_risk,
                all_nodes=self.nodes if self._enable_coverage_gating else None,
            )
            peer_pool_for_rfq = ([orchestrator] + p0_pool)[: self.k_max]
        else:
            peer_pool_for_rfq = self.nodes

        # Run allocation (HEDGE or selected baseline)
        raw_result = self._allocate(
            peer_pool=peer_pool_for_rfq,
            task=task,
            cloud=self.cloud,
            tau_dict=tau_dict,
            peer_standing_quotes=self._peer_standing_quotes,
            k_max=self.k_max,
            alpha_u=self.alpha_u,
            gamma_u=self.gamma_u,
            lambda_max=self._lambda_max,
        )

        # Normalise result to _AllocOutcome
        if isinstance(raw_result, RFQResult):
            outcome = _rfq_to_outcome(raw_result)
            # Fill latency (RFQ result has no latency cached; compute it)
            if outcome.executor_type == "edge" and outcome.winner is not None:
                tau_i = tau_dict.get(outcome.winner.unique_id, 0.005)
                outcome.latency_est = compute_node_latency(task, outcome.winner, tau_i)
            elif outcome.executor_type == "cloud":
                outcome.latency_est = compute_cloud_latency(task, self.cloud)
        else:
            # dict-returning baseline
            outcome = _dict_to_outcome(raw_result, self._node_by_uid)

        # Update winner state (w_pending, completion heap, EMA)
        energy_joules = 0.0
        carbon_g = 0.0

        if outcome.executor_type == "edge" and outcome.winner is not None:
            winner = outcome.winner
            winner.w_pending += task.w
            w_arrived_per_uid[winner.unique_id] = (
                w_arrived_per_uid.get(winner.unique_id, 0.0) + task.w
            )
            # Drain time from now until this task (and everything queued ahead of
            # it) finishes, i.e. the same FIFO queue_wait + execution model used by
            # compute_node_latency for the admission gate. w_pending here is
            # POST-increment, so it already includes this task's own w on top of
            # whatever backlog preceded it. Pushing exec-time-only (ignoring the
            # backlog) would let the node's tracked backlog drain faster than the
            # model used to admit tasks assumes, silently under-counting queue
            # delay for every subsequent quote on this node.
            drain_time = winner.w_pending / max(winner.f_max, 1.0)
            heapq.heappush(
                self._completion_heap,
                (sim_time + drain_time, winner.unique_id, task.w, task.task_id),
            )
            # Executor's own settlement: the full buyer payment.
            settlement_amount = outcome.price_total
            self._emas[winner.unique_id].on_task_settlement(settlement_amount)
            winner.R_hat = self._emas[winner.unique_id].get_latest_forecast()
            self._revenue_since_tick[winner.unique_id] = (
                self._revenue_since_tick.get(winner.unique_id, 0.0) + settlement_amount
            )
            energy_joules, carbon_g = self._compute_energy_carbon(winner, task.w)

            # Online P_fail(i) sliding window (Eq 9): did this task miss its
            # deadline once actually assigned. Tracked for reporting/A8
            # diagnostics only (see the enable_gc_risk_premium docstring in
            # phase0.py for why it does not gate GC-play entry directly).
            window = self._sla_outcome_window.setdefault(
                winner.unique_id, deque(maxlen=self._w_fail)
            )
            window.append(outcome.latency_est > task.d)
            winner.p_fail_hat = sum(window) / len(window) if window else 0.0

        elif outcome.executor_type == "cloud":
            energy_joules, carbon_g = self._compute_energy_carbon(self.cloud, task.w)

        # Feed real MetricsCollector (post-warmup only for paper metrics)
        if sim_time >= self.warmup_duration:
            if outcome.executor_type == "unmatched":
                self._mc.record_event(
                    {
                        "event_type": "rejected",
                        "task_id": task.task_id,
                        "arrival_time": sim_time,
                        "deadline": task.d,
                        "latency": 0.0,
                        "price_paid": 0.0,
                        "executor_id": "",
                        "home_node_id": str(self.nodes[home_node_idx].unique_id),
                        "carbon_g": 0.0,
                        "energy_joules": 0.0,
                        "is_resale": False,
                        "a_buyer": task.a_buyer,
                        "C1_total": 0.0,
                    }
                )
            else:
                executor_id = (
                    "cloud"
                    if outcome.executor_type == "cloud"
                    else str(outcome.winner.unique_id) if outcome.winner is not None else "cloud"
                )
                self._mc.record_event(
                    {
                        "event_type": "completed",
                        "task_id": task.task_id,
                        "arrival_time": sim_time,
                        "deadline": task.d,
                        "latency": outcome.latency_est,
                        "price_paid": outcome.price_total,
                        "executor_id": executor_id,
                        "home_node_id": str(self.nodes[home_node_idx].unique_id),
                        "carbon_g": carbon_g,
                        "energy_joules": energy_joules,
                        "is_resale": False,
                        "a_buyer": task.a_buyer,
                        "C1_total": outcome.C1_winner,
                    }
                )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, n_tasks: int | None = None) -> dict[str, float]:
        """Run the simulation.

        Args:
            n_tasks: If set, stop after this many tasks have been submitted.
                     If None, run for sim_duration seconds.

        Returns:
            Flat metrics dict (M1_p50_latency_s through M16_myerson_efficiency).
        """
        max_steps = int(self.sim_duration / self.delta_tp) + 1 if n_tasks is None else int(2e7)

        n_submitted = 0

        for step in range(max_steps):
            sim_time = step * self.delta_tp

            arrivals = self.scheduler.generate_tick(sim_time, self.delta_tp)
            w_arrived_per_uid: dict[int, float] = {}

            for arrival in arrivals:
                self._dispatch_rfq(sim_time, arrival.task, arrival.home_node_idx, w_arrived_per_uid)
                n_submitted += 1
                if n_tasks is not None and n_submitted >= n_tasks:
                    break

            self._process_completions(sim_time)
            self._cadence_tick(sim_time, w_arrived_per_uid)

            if n_tasks is not None and n_submitted >= n_tasks:
                logger.info(f"Reached n_tasks={n_tasks} at t={sim_time:.3f}s")
                break

        logger.info(
            f"Simulation done: {n_submitted} tasks submitted, "
            f"{len(self._mc.events)} events recorded"
        )

        metrics = self._compute_final_metrics()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self._save_events:
                self._write_events_jsonl()
            if self._save_snapshots:
                self._write_snapshots_jsonl()
            self._write_summary_json(metrics)

        return metrics

    def _write_events_jsonl(self) -> None:
        """Write the rich per-task event log (arrival_time, deadline, is_resale,
        a_buyer, etc.) to events.jsonl."""
        assert self.output_dir is not None
        path = self.output_dir / "events.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ev in self._mc.events:
                fh.write(json.dumps(ev) + "\n")
        logger.info(f"Wrote {len(self._mc.events)} events to {path}")

    def _write_snapshots_jsonl(self) -> None:
        """Write per-(node, tick) snapshots (l_current, l_hat, R_hat, f_max) to
        node_snapshots.jsonl. Off by default (output.save_snapshots) since it is
        O(N_nodes * ticks) and only needed for burst-tracking / M14 analysis."""
        assert self.output_dir is not None
        path = self.output_dir / "node_snapshots.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for snap in self._mc.node_snapshots:
                fh.write(json.dumps(snap) + "\n")
        logger.info(f"Wrote {len(self._mc.node_snapshots)} snapshots to {path}")

    def _write_summary_json(self, metrics: dict[str, float]) -> None:
        """Write the real flat M1-M16 metrics dict (not the legacy stub) to
        summary.json."""
        assert self.output_dir is not None
        path = self.output_dir / "summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"Wrote summary metrics to {path}")

    def _compute_final_metrics(self) -> dict[str, float]:
        """Compute final flat metrics from MetricsCollector.

        Returns:
            Flat dict with all metric values as finite floats.
        """
        # Real metrics from MetricsCollector
        real = self._mc.compute_all()

        # Flatten nested metrics (M1 dict, M8 dict, M14 dict)
        m1: dict[str, float] = real.get("M1", {})
        m8: dict[str, Any] = real.get("M8", {})
        m14: dict[str, float] = real.get("M14", {})

        def _f(v: Any) -> float:
            """Return finite float or 0.0."""
            try:
                x = float(v)
                return x if math.isfinite(x) else 0.0
            except (TypeError, ValueError):
                return 0.0

        metrics: dict[str, float] = {
            # M1: Latency percentiles
            "M1_p50_latency_s": _f(m1.get("p50", 0.0)),
            "M1_p95_latency_s": _f(m1.get("p95", 0.0)),
            "M1_p99_latency_s": _f(m1.get("p99", 0.0)),
            "M1_mean_latency_s": _f(m1.get("mean", 0.0)),
            # M2: Mean cost per accepted task
            "M2_mean_cost_usd": _f(real.get("M2", 0.0)),
            # M3: System utilisation
            "M3_system_utilisation": _f(real.get("M3", 0.0)),
            # M4: Utilisation variance
            "M4_utilisation_variance": _f(real.get("M4", 0.0)),
            # M5: Total market revenue (edge sellers only)
            "M5_market_revenue_usd": _f(real.get("M5", 0.0)),
            # M6: Rejection rate
            "M6_rejection_rate": _f(real.get("M6", 0.0)),
            # M7: Carbon emissions per completed task
            "M7_carbon_g_per_task": _f(real.get("M7", 0.0)),
            # M8: Price convergence (quote variance scalar)
            "M8_quote_variance": _f(m8.get("quote_variance", 0.0)),
            # M9: Edge service fraction (fraction of completed tasks served by edge)
            "M9_edge_service_fraction": _f(real.get("M9", 0.0)),
            # M10: Completion rate
            "M10_completion_rate": _f(real.get("M10", 0.0)),
            # M11: Jain's Fairness Index on per-node utilisation
            "M11_jfi": _f(real.get("M11", 1.0)),
            # M12: Transient SLA Failure Rate (TSFR)
            "M12_tsfr": _f(real.get("M12", 0.0)),
            # M13: Edge-Resale Ratio
            "M13_err": _f(real.get("M13", 0.0)),
            # M14: Prediction RMSE
            "M14_rmse_load": _f(m14.get("RMSE_load", 0.0)),
            "M14_rmse_revenue": _f(m14.get("RMSE_revenue", 0.0)),
            # M15: Fleet energy per task
            "M15_energy_joules_per_task": _f(real.get("M15", 0.0)),
            # M16: Myerson efficiency
            "M16_myerson_efficiency": _f(real.get("M16", 1.0)),
            # M17: Primary inter-edge offload fraction (horizontal-market signal)
            "M17_primary_offload_fraction": _f(real.get("M17", 0.0)),
        }

        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from path."""
    with config_path.open("r", encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh))


def main() -> None:
    """CLI entry point: run HEDGE simulation and write outputs."""
    parser = argparse.ArgumentParser(description="HEDGE simulation engine")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--n-tasks", type=int, default=None, dest="n_tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Override algorithm: HEDGE | B0 | B1 | ... | B7",
    )
    parser.add_argument(
        "--topology-source",
        default=None,
        dest="topology_source",
        help="Override config topology source ('synthetic' or 'eua_melbourne').",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=None,
        dest="n_nodes",
        help="Override config N_nodes.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.topology_source is not None:
        config.setdefault("topology", {})["source"] = args.topology_source
    if args.n_nodes is not None:
        config.setdefault("topology", {})["N_nodes"] = args.n_nodes
    if args.algorithm is not None:
        config.setdefault("experiment", {})["algorithm"] = args.algorithm

    output_dir = Path(args.output) if args.output else None

    engine = HEDGESimulationEngine(config=config, seed=args.seed, output_dir=output_dir)
    metrics = engine.run(n_tasks=args.n_tasks)

    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
