"""Phase A3 gate: scheduler selection, predictor selection, and admission-gate wiring.

Coverage:
- arrivals.mode == "poisson" (default) builds a PoissonScheduler.
- MetricsCollector's burst window defaults to steady-state (no burst window) and
  can be set via an explicit config["metrics"] override.
- predictor.mode selects the correct load-predictor class, with a safe fallback.
- A high parasite_fraction with the admission gate enabled produces a nonzero
  rejection rate, unlike a clean run of the same config.
"""

from __future__ import annotations

from typing import Any

from hedge.simulation.engine import HEDGESimulationEngine
from hedge.simulation.scheduler import PoissonScheduler

_BASE_CFG: dict[str, Any] = {
    "label": "burst_test",
    "simulation": {
        "duration": 20.0,
        "random_seed": 0,
        "delta_tp": 0.1,
        "delta_q": 0.1,
        "warmup_duration": 0.0,
    },
    "topology": {
        "source": "synthetic",
        "N_nodes": 10,
        "tau_c_mean": 0.03,
        "tau_c_std": 0.005,
    },
    "arrivals": {
        "mode": "poisson",
        "lambda_bar": 0.5,
        "n_hotspots": 3,
        "hotspot_rate_multiplier": 2.0,
        "quiet_rate_multiplier": 0.3,
    },
    "task_model": {
        "w_min": 1.0e8,
        "w_max": 5.0e8,
        "s_min": 1.0e6,
        "s_max": 5.0e6,
        "d_min": 0.5,
        "d_max": 5.0,
        "a_buyer_min": 0.5,
        "a_buyer_max": 10.0,
    },
    "hedge": {
        "K_max": 4,
        "H_max": 2,
        "tau_lock": 0.02,
        "theta_commit": 0.02,
        "W_fail": 20,
        "theta_risk": 0.12,
        "epsilon_trans": 0.05,
        "x_weight": 0.5,
        "y_weight": 0.5,
        "alpha_u": 0.5,
        "gamma_u": 0.5,
    },
    "pricing": {
        "a_ref": 5.0e-10,
        "b_shape": 1.0,
        "eta": 1.0,
        "lambda_max": 0.5,
        "R_ref": 1.0,
        "alpha_R": 0.2,
        "pi_co2": 5.0e-5,
        "mu_c": 1.5,
    },
    "cloud": {
        "f_max": 1.0e12,
        "kappa": 1.0e-27,
        "rho": 1.0e-4,
        "beta": 80.0,
        "pi_E": 0.06,
        "P_idle": 0.0,
    },
    "predictor": {
        "mode": "instantaneous",
    },
    "market": {
        "enable_admission_gate": True,
    },
    "experiment": {"algorithm": "HEDGE"},
    "output": {
        "base_dir": "outputs",
        "log_level": "WARNING",
    },
}


def test_poisson_mode_uses_plain_poisson_scheduler() -> None:
    """Default/poisson mode builds a PoissonScheduler."""
    engine = HEDGESimulationEngine(config=_BASE_CFG, seed=0, output_dir=None)
    assert isinstance(engine.scheduler, PoissonScheduler)


def test_metrics_collector_burst_window_default_when_poisson() -> None:
    """Poisson-mode engines keep the steady-state default (M12 == 0)."""
    engine = HEDGESimulationEngine(config=_BASE_CFG, seed=0, output_dir=None)
    assert engine._mc._burst_start == -1.0
    assert engine._mc._burst_duration == 0.0


def test_metrics_collector_burst_window_explicit_override() -> None:
    """An explicit config["metrics"] override sets the MetricsCollector's burst window."""
    cfg: dict[str, Any] = {
        **_BASE_CFG,
        "metrics": {"burst_start": 9.0, "burst_duration": 1.0},
    }
    engine = HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
    assert engine._mc._burst_start == 9.0
    assert engine._mc._burst_duration == 1.0


def test_predictor_mode_selects_instantaneous_filter() -> None:
    """engine._init_predictors builds InstantaneousLoadFilter for mode='instantaneous'."""
    from hedge.predictor.instantaneous import InstantaneousLoadFilter

    cfg: dict[str, Any] = {
        **_BASE_CFG,
        "predictor": {"mode": "instantaneous"},
    }
    engine = HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
    any_node_uid = engine.nodes[0].unique_id
    assert isinstance(engine._load_predictors[any_node_uid], InstantaneousLoadFilter)


def test_predictor_mode_unknown_falls_back_to_instantaneous() -> None:
    """An unrecognised predictor.mode falls back to InstantaneousLoadFilter."""
    from hedge.predictor.instantaneous import InstantaneousLoadFilter

    cfg: dict[str, Any] = {
        **_BASE_CFG,
        "predictor": {"mode": "not_a_real_mode"},
    }
    engine = HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
    any_node_uid = engine.nodes[0].unique_id
    assert isinstance(engine._load_predictors[any_node_uid], InstantaneousLoadFilter)


def test_parasite_injection_activates_admission_gate() -> None:
    """Adv3: a high parasite_fraction with the admission gate enabled must
    produce a nonzero rejection rate (M6), unlike a clean run of the same
    config where the gate structurally never fires."""
    clean_cfg: dict[str, Any] = {
        **_BASE_CFG,
        "market": {"enable_admission_gate": True},
        "arrivals": {**_BASE_CFG["arrivals"], "parasite_fraction": 0.0},
    }
    parasite_cfg: dict[str, Any] = {
        **_BASE_CFG,
        "market": {"enable_admission_gate": True},
        "arrivals": {**_BASE_CFG["arrivals"], "parasite_fraction": 0.4},
    }

    clean_engine = HEDGESimulationEngine(config=clean_cfg, seed=0, output_dir=None)
    clean_metrics = clean_engine.run(n_tasks=1000)
    assert clean_metrics["M6_rejection_rate"] == 0.0

    parasite_engine = HEDGESimulationEngine(config=parasite_cfg, seed=0, output_dir=None)
    parasite_metrics = parasite_engine.run(n_tasks=1000)
    assert parasite_metrics["M6_rejection_rate"] > 0.0
