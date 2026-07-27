"""Poisson task arrival scheduler for HEDGE simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from loguru import logger

from hedge.core.constants import (
    A_BUYER_MAX,
    A_BUYER_MIN,
    B_SHAPE,
    D_MAX,
    D_MIN,
    S_MAX,
    S_MIN,
    W_MAX,
    W_MIN,
)
from hedge.core.task import HEDGETask
from hedge.pricing.layer1 import compute_layer1_cost


@dataclass
class TaskArrival:
    """A task that arrived at a specific node during a cadence tick."""

    task: HEDGETask
    home_node_idx: int  # index into nodes list; used for tau_dict row selection


class PoissonScheduler:
    """Generates Poisson-distributed task arrivals across all nodes per cadence tick.

    Each node i gets lambda_bar * arrival_rate_multiplier_i tasks/second on average.
    Task parameters (w, s, d, a_buyer) are drawn uniform from config ranges.
    """

    def __init__(
        self,
        nodes: list[Any],
        config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Initialise scheduler.

        Args:
            nodes: List of edge nodes with arrival_rate_multiplier attribute.
            config: Experiment config dict (reads arrivals and task_model sections).
            rng: Seeded random generator for reproducibility.
        """
        self.nodes = nodes
        self.config = config
        self.rng = rng
        self._task_counter: int = 0

        arrivals_cfg = config.get("arrivals", {})
        lambda_bar = float(arrivals_cfg.get("lambda_bar", 1.0))
        self._lambda_per_node: list[float] = [
            lambda_bar * float(node.arrival_rate_multiplier) for node in nodes
        ]

        task_cfg = config.get("task_model", {})
        self._w_min = float(task_cfg.get("w_min", W_MIN))
        self._w_max = float(task_cfg.get("w_max", W_MAX))
        self._s_min = float(task_cfg.get("s_min", S_MIN))
        self._s_max = float(task_cfg.get("s_max", S_MAX))
        self._d_min = float(task_cfg.get("d_min", D_MIN))
        self._d_max = float(task_cfg.get("d_max", D_MAX))
        self._a_min = float(task_cfg.get("a_buyer_min", A_BUYER_MIN))
        self._a_max = float(task_cfg.get("a_buyer_max", A_BUYER_MAX))

        # Adv3: low-valuation parasite injection. A fraction of tasks declare
        # a_buyer in [low_mult, high_mult] * C1_min/b_shape,
        # i.e. barely below/at the cheapest node's Bertrand floor, to exercise the
        # economic admission gate. Disabled by default (parasite_fraction=0.0).
        self._parasite_fraction: float = float(arrivals_cfg.get("parasite_fraction", 0.0))
        self._parasite_low_mult: float = float(arrivals_cfg.get("parasite_low_mult", 0.5))
        self._parasite_high_mult: float = float(arrivals_cfg.get("parasite_high_mult", 1.0))
        self._b_shape: float = float(config.get("pricing", {}).get("b_shape", B_SHAPE))

    def _draw_a_buyer(self, w: float) -> float:
        """Draw a buyer valuation for a task of workload w cycles.

        With probability parasite_fraction, draws a low-valuation "parasite"
        value near the cheapest node's Bertrand floor instead of the normal
        uniform range (Adv3 sub-experiment).

        Args:
            w: Task workload (cycles); needed to compute the floor for parasites.

        Returns:
            Buyer valuation in USD.
        """
        if self._parasite_fraction > 0.0 and self.rng.random() < self._parasite_fraction:
            c1_min = min(
                compute_layer1_cost(w, n.f_max, n.kappa, n.P_idle, n.rho, n.pi_E, n.beta)[0]
                for n in self.nodes
            )
            floor = c1_min / max(self._b_shape, 1e-30)
            lo = self._parasite_low_mult * floor
            hi = self._parasite_high_mult * floor
            if hi <= lo:
                hi = lo * 1.0000001 + 1e-30  # guard against a degenerate zero-width range
            return float(self.rng.uniform(lo, hi))
        return float(self.rng.uniform(self._a_min, self._a_max))

    def generate_tick(self, sim_time: float, delta_tp: float) -> list[TaskArrival]:
        """Generate all task arrivals for one cadence tick.

        Args:
            sim_time: Current simulation time (seconds).
            delta_tp: Tick duration (seconds).

        Returns:
            List of TaskArrival, one entry per arriving task (may be empty).
        """
        arrivals: list[TaskArrival] = []
        for i, (node, lam) in enumerate(zip(self.nodes, self._lambda_per_node)):
            n = int(self.rng.poisson(lam * delta_tp))
            for _ in range(n):
                self._task_counter += 1
                # Draw order (s, w, d, a_buyer) matches the pre-Adv3 draw sequence
                # exactly when parasite_fraction=0.0, so default-config runs remain
                # byte-identical.
                s = float(self.rng.uniform(self._s_min, self._s_max))
                w = float(self.rng.uniform(self._w_min, self._w_max))
                d = float(self.rng.uniform(self._d_min, self._d_max))
                a_buyer = self._draw_a_buyer(w)
                task = HEDGETask(
                    task_id=f"T{self._task_counter}",
                    s=s,
                    w=w,
                    d=d,
                    a_buyer=a_buyer,
                    created_at=sim_time,
                )
                arrivals.append(TaskArrival(task=task, home_node_idx=i))
        return arrivals


class TraceScheduler:
    """Real-trace task arrivals (Alibaba 2018 batch trace), replayed against the
    simulated topology instead of a synthetic Poisson process.

    Loads a trace once at construction (via data.loaders.alibaba_loader), then:
    1. Rescales the trace's timestamps so its system-wide arrival rate matches
       arrivals.trace_target_lambda * N_nodes (same per-node-rate semantics as
       PoissonScheduler's lambda_bar, so heavy-load calibration stays meaningful).
    2. Applies a seeded circular phase rotation so different seeds see a
       different starting point in the trace (inter-arrival structure, i.e. the
       trace's real burstiness, is preserved -- only WHERE in the cycle each
       seed starts differs).
    3. Tiles the (rescaled, rotated) trace to cover the full simulation duration
       when the natural trace span is shorter than sim_duration.
    4. Assigns each trace task a home node by a single seeded categorical draw
       weighted by node.arrival_rate_multiplier, so the existing hotspot/quiet/
       normal fat-tail tiers apply to trace-driven arrivals the same way they
       apply to synthetic Poisson arrivals.

    See data/README.md for the trace's field-level provenance; this class
    only adds the simulation-arrival-process wiring on top of that.
    """

    def __init__(
        self,
        nodes: list[Any],
        config: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Load, rescale, rotate, tile, and assign home nodes for the trace.

        Args:
            nodes: Edge nodes with arrival_rate_multiplier attribute.
            config: Experiment config dict (reads arrivals, task_model, simulation).
            rng: Seeded random generator (consumed once here for rotation phase
                 and home-node assignment; generate_tick draws nothing further).
        """
        from data.loaders.alibaba_loader import (  # noqa: PLC0415
            load_alibaba_trace,
            remap_to_heavy_envelope,
        )

        self.nodes = nodes
        self.config = config
        self.rng = rng

        arrivals_cfg = config.get("arrivals", {})
        trace_path = str(arrivals_cfg.get("trace_path", "data/alibaba_1h_subset.parquet"))
        max_tasks = int(arrivals_cfg.get("max_tasks", 100_000))
        target_lambda = float(arrivals_cfg.get("trace_target_lambda", arrivals_cfg.get("lambda_bar", 1.0)))
        sim_duration = float(config.get("simulation", {}).get("duration", 3600.0))
        delta_tp = float(config.get("simulation", {}).get("delta_tp", 0.1))
        base_seed = int(config.get("simulation", {}).get("random_seed", 0))

        tasks = load_alibaba_trace(trace_path, max_tasks=max_tasks, seed=base_seed, sim_time_scale=1.0)
        if len(tasks) < 2:
            raise ValueError(
                f"Alibaba trace at '{trace_path}' yielded {len(tasks)} tasks; need >= 2 "
                "to derive an inter-arrival rate for rescaling."
            )

        # Optional: rank-transform w/s/d into an edge-native heavy-load envelope
        # (see remap_to_heavy_envelope docstring). Off by default -- the raw
        # trace, preprocessed via the light task_model constants, is
        # structurally cloud-dominated (large compute, generous deadlines) at
        # ANY arrival rate, so this is the calibration lever that actually
        # matters for a real-data HEAVY campaign, not trace_target_lambda alone.
        if bool(arrivals_cfg.get("remap_to_heavy_envelope", False)):
            # float() cast guards against PyYAML parsing unsigned-exponent
            # scientific notation (e.g. "1.0e7", no explicit "+") as a string.
            w_lo, w_hi = arrivals_cfg.get("heavy_w_range", [1.0e7, 5.0e7])
            d_lo, d_hi = arrivals_cfg.get("heavy_d_range", [0.15, 0.5])
            s_lo, s_hi = arrivals_cfg.get("heavy_s_range", [4.5e7, 8.0e7])
            # Bimodal task mix: remapping 100% of tasks into the tight
            # edge-native envelope makes cloud transfer-time-infeasible for
            # EVERY task by construction (see docstring), which is why cloud
            # usage is always exactly 0 downstream. cloud_eligible_fraction
            # carves out a disclosed fraction of tasks (default 0.0, byte-
            # identical to prior behavior) that are left at their natural,
            # light-scale w/s/d values instead -- larger compute, more
            # generous relative deadline, naturally cloud-competitive,
            # consistent with a realistic mixed IoT+batch workload where not
            # every job is edge-appropriate. This only makes cloud POSSIBLE
            # for that fraction; AFGM still decides case by case, so the
            # realized cloud-usage fraction need not equal this config value.
            cloud_eligible_fraction = float(arrivals_cfg.get("cloud_eligible_fraction", 0.0))
            if cloud_eligible_fraction > 0.0:
                split_rng = np.random.default_rng(base_seed * 104729 + 7)
                is_cloud_eligible = split_rng.random(len(tasks)) < cloud_eligible_fraction
                edge_native = [t for t, m in zip(tasks, is_cloud_eligible) if not m]
                cloud_eligible = [t for t, m in zip(tasks, is_cloud_eligible) if m]
                edge_native = remap_to_heavy_envelope(
                    edge_native,
                    (float(w_lo), float(w_hi)),
                    (float(d_lo), float(d_hi)),
                    (float(s_lo), float(s_hi)),
                )
                # a_buyer is drawn independently of w (load_alibaba_trace /
                # _task_from_row: uniform over A_BUYER_MIN/MAX regardless of
                # task size). That is fine for edge-native tasks (remapped w
                # is small enough that any a_buyer in range clears the edge
                # cost floor), but cloud-eligible tasks keep their natural
                # w up to W_MAX (1e11 cycles) while a_buyer stays capped at
                # A_BUYER_MAX -- for large w this falls below cloud's own
                # marked-up price (p_c = mu_c * C1_cloud(w)), so the task
                # fails AFGM's affordable-set check (Eq. affordable_set) even
                # though it was left at natural scale specifically so cloud
                # COULD serve it. Rescale a_buyer for this subset only to a
                # multiple of cloud's own price, drawn from a fixed range
                # (a disclosed modeling choice: a job explicitly identified
                # as cloud-appropriate is assumed to carry a valuation
                # consistent with cloud pricing, not an IoT-ping-scale one).
                from hedge.core.constants import (  # noqa: PLC0415
                    BETA_CLOUD,
                    F_MAX_CLOUD,
                    MU_C,
                    KAPPA_CLOUD,
                    PI_E_CLOUD,
                    RHO_CLOUD,
                    S_RET_FRACTION,
                    TAU_C_MEAN,
                )
                from hedge.pricing.layer1 import compute_layer1_cost  # noqa: PLC0415

                # d is ALSO drawn independently of w in the raw loader
                # (d_raw = duration_s * DEADLINE_SLACK, clipped to
                # [D_MIN,D_MAX], while w is clipped separately from the SAME
                # duration_s combined with cpu_avg) -- for a short-duration,
                # high-cpu_avg raw job this can clip to w=W_MAX with
                # d=D_MIN simultaneously, a combination even CLOUD cannot
                # execute (w/F_MAX_CLOUD alone can exceed D_MIN). Floor d for
                # the cloud-eligible subset at cloud's own minimum feasible
                # latency plus margin, for the same reason a_buyer is
                # rescaled above: a task deliberately left at natural scale
                # to be cloud-routable must have a deadline cloud can meet.
                bandwidth_cloud = 1e8
                a_buyer_rng = np.random.default_rng(base_seed * 104729 + 11)
                rescaled_cloud_eligible = []
                for t in cloud_eligible:
                    c1_cloud, _, _, _ = compute_layer1_cost(
                        t.w, F_MAX_CLOUD, KAPPA_CLOUD, 0.0, RHO_CLOUD, PI_E_CLOUD, BETA_CLOUD
                    )
                    p_cloud = MU_C * c1_cloud
                    new_a_buyer = p_cloud * float(a_buyer_rng.uniform(1.3, 3.0))
                    cloud_min_latency = (
                        (t.s + t.s * S_RET_FRACTION) / bandwidth_cloud
                        + 2.0 * TAU_C_MEAN
                        + t.w / F_MAX_CLOUD
                    )
                    new_d = max(t.d, cloud_min_latency * 1.5)
                    rescaled_cloud_eligible.append(
                        replace(t, a_buyer=new_a_buyer, d=new_d)
                    )
                tasks = sorted(edge_native + rescaled_cloud_eligible, key=lambda t: t.created_at)
            else:
                tasks = remap_to_heavy_envelope(
                    tasks,
                    (float(w_lo), float(w_hi)),
                    (float(d_lo), float(d_hi)),
                    (float(s_lo), float(s_hi)),
                )

        t_min = tasks[0].created_at
        natural_span = max(tasks[-1].created_at - t_min, 1e-6)
        natural_system_rate = len(tasks) / natural_span
        target_system_rate = max(target_lambda * len(nodes), 1e-9)
        time_scale = natural_system_rate / target_system_rate

        rescaled = [replace(t, created_at=(t.created_at - t_min) * time_scale) for t in tasks]
        scaled_span = natural_span * time_scale

        # Seeded circular rotation: randomises which phase of the trace each seed
        # starts at, without altering the trace's own inter-arrival structure.
        shift = float(self.rng.uniform(0.0, scaled_span))
        rotated = [replace(t, created_at=(t.created_at - shift) % scaled_span) for t in rescaled]
        rotated.sort(key=lambda t: t.created_at)

        weights = np.array([float(n.arrival_rate_multiplier) for n in nodes], dtype=np.float64)
        weights = weights / weights.sum()
        home_indices = self.rng.choice(len(nodes), size=len(rotated), p=weights)

        n_tiles_estimate = max(1, math.ceil((sim_duration + delta_tp) / scaled_span))
        if n_tiles_estimate > 50:
            logger.warning(
                f"TraceScheduler will cycle through the trace ~{n_tiles_estimate}x to "
                f"cover {sim_duration:.0f}s (natural scaled span {scaled_span:.2f}s); "
                "tiles are generated lazily (constant memory), but consider raising "
                "arrivals.max_tasks for a longer natural span per cycle."
            )

        # Base single-cycle queue (NOT pre-tiled: tiles are generated lazily in
        # generate_tick by tracking which cycle-of-the-base-queue the current
        # sim_time falls into). Precomputing every tile upfront does not scale:
        # at a high trace_target_lambda the natural span can compress to a
        # fraction of a second, requiring thousands of repetitions to cover a
        # full simulation, which would materialise tens of millions of
        # HEDGETask objects for a single seed.
        self._base_queue: list[tuple[HEDGETask, int]] = list(
            zip(rotated, (int(h) for h in home_indices))
        )
        self._scaled_span: float = scaled_span
        self._base_cursor: int = 0
        self._tile_idx: int = 0

    def generate_tick(self, sim_time: float, delta_tp: float) -> list[TaskArrival]:
        """Return all trace arrivals due in [sim_time, sim_time + delta_tp).

        Lazily advances through repeated cycles of the base (rotated) trace
        queue, offsetting each cycle's timestamps by tile_idx * scaled_span,
        so coverage of an arbitrarily long simulation costs O(arrivals this
        tick) rather than O(total tiles) memory.

        Args:
            sim_time: Current simulation time (seconds).
            delta_tp: Tick duration (seconds).

        Returns:
            List of TaskArrival due this tick (may be empty).
        """
        tick_end = sim_time + delta_tp
        arrivals: list[TaskArrival] = []
        n = len(self._base_queue)
        while True:
            if self._base_cursor >= n:
                self._base_cursor = 0
                self._tile_idx += 1
                continue
            task, home_idx = self._base_queue[self._base_cursor]
            abs_created_at = task.created_at + self._tile_idx * self._scaled_span
            if abs_created_at >= tick_end:
                break
            self._base_cursor += 1
            arrivals.append(
                TaskArrival(
                    task=replace(
                        task,
                        task_id=f"{task.task_id}_t{self._tile_idx}",
                        created_at=abs_created_at,
                    ),
                    home_node_idx=home_idx,
                )
            )
        return arrivals


