"""Phase 3 gate: smoke-run every ablation config to verify they load and run.

Parametrized over all *.yaml files in configs/ablations/.
Each config runs for 50 tasks and must produce a finite metrics dict without crashing.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from experiments.runner import load_config
from hedge.simulation.engine import HEDGESimulationEngine

# Locate configs/ablations/ relative to this test file
_REPO_ROOT = Path(__file__).parent.parent.parent
_ABLATION_DIR = _REPO_ROOT / "configs" / "ablations"

_ABLATION_YAMLS = sorted(_ABLATION_DIR.glob("*.yaml"))


@pytest.mark.parametrize(
    "config_path",
    _ABLATION_YAMLS,
    ids=[p.stem for p in _ABLATION_YAMLS],
)
def test_ablation_config_loads(config_path: Path) -> None:
    """Each ablation config must load without error via runner.load_config."""
    cfg = load_config(config_path)
    assert isinstance(cfg, dict), f"{config_path.name} did not load as a dict"
    assert (
        "label" in cfg or "simulation" in cfg
    ), f"{config_path.name} missing 'label' or 'simulation' key"


@pytest.mark.parametrize(
    "config_path",
    _ABLATION_YAMLS,
    ids=[p.stem for p in _ABLATION_YAMLS],
)
def test_ablation_config_smoke_run(config_path: Path) -> None:
    """Each ablation config runs 50 tasks without exception and returns finite metrics."""
    cfg = load_config(config_path)

    # Force synthetic topology (EUA Melbourne data may not be present in CI)
    cfg.setdefault("topology", {})["source"] = "synthetic"
    cfg.setdefault("topology", {})["N_nodes"] = 10

    # Disable warmup for speed
    cfg.setdefault("simulation", {})["warmup_duration"] = 0.0
    cfg.setdefault("simulation", {})["duration"] = 3600.0

    # For sweep configs, take the first variant only
    if "sweep" in cfg:
        sweep = cfg.pop("sweep")
        # Set each sweep param to its first value
        for param, values in sweep.items():
            if isinstance(values, list) and values:
                _set_nested(cfg, param, values[0])

    engine = HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
    metrics = engine.run(n_tasks=50)

    assert isinstance(metrics, dict)
    non_finite = {k: v for k, v in metrics.items() if not math.isfinite(v)}
    assert not non_finite, f"{config_path.name}: non-finite metrics: {non_finite}"
    assert len(engine._mc.events) == 50


def _set_nested(cfg: dict, param: str, value: object) -> None:
    """Set a config key (possibly in a nested section) to value.

    Supports dot-notation (e.g., 'pricing.eta') or flat keys (e.g., 'K_max').
    Also handles runner-specific sweep keys like 'algorithm', 'predictor_mode'.
    """
    # Special mappings for sweep keys that live in sub-sections
    _SWEEP_MAP = {
        "K_max": ("hedge", "K_max"),
        "theta_risk": ("hedge", "theta_risk"),
        "W_fail": ("hedge", "W_fail"),
        "algorithm": ("experiment", "algorithm"),
        "predictor_mode": ("predictor", "mode"),
    }
    if param in _SWEEP_MAP:
        section, key = _SWEEP_MAP[param]
        cfg.setdefault(section, {})[key] = value
    elif "." in param:
        parts = param.split(".", 1)
        cfg.setdefault(parts[0], {})[parts[1]] = value
    else:
        cfg[param] = value
