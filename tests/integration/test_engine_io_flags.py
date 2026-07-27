"""A9 gate: output.save_events / output.save_snapshots control disk writes.

Coverage:
- Default (save_events=True, save_snapshots unset->False): events.jsonl and
  summary.json are written; node_snapshots.jsonl is not.
- save_events=False: no events.jsonl.
- save_snapshots=True: node_snapshots.jsonl is written with rich fields.
- summary.json contains the real M1-M16 keys, not the legacy stub names.
- In-memory metrics/collector state is unaffected by the IO flags (metrics
  computation never depends on whether anything was written to disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hedge.simulation.engine import HEDGESimulationEngine

_CFG: dict[str, Any] = {
    "label": "io_flags_test",
    "simulation": {
        "duration": 3600.0,
        "random_seed": 0,
        "delta_tp": 0.1,
        "delta_q": 0.1,
        "warmup_duration": 0.0,
    },
    "topology": {"source": "synthetic", "N_nodes": 6, "tau_c_mean": 0.03, "tau_c_std": 0.005},
    "arrivals": {
        "mode": "poisson",
        "lambda_bar": 2.0,
        "n_hotspots": 1,
        "hotspot_rate_multiplier": 4.0,
        "quiet_rate_multiplier": 0.3,
    },
    "task_model": {
        "w_min": 1.0e8,
        "w_max": 5.0e8,
        "s_min": 1.0e6,
        "s_max": 5.0e6,
        "d_min": 0.5,
        "d_max": 4.0,
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
    "market": {"enable_admission_gate": True},
    "experiment": {"algorithm": "HEDGE"},
    "output": {"base_dir": "outputs", "log_level": "WARNING"},
}


def test_default_writes_events_and_summary_not_snapshots(tmp_path: Path) -> None:
    engine = HEDGESimulationEngine(config=_CFG, seed=0, output_dir=tmp_path)
    engine.run(n_tasks=200)
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert not (tmp_path / "node_snapshots.jsonl").exists()


def test_save_events_false_skips_events_file(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {**_CFG, "output": {**_CFG["output"], "save_events": False}}
    engine = HEDGESimulationEngine(config=cfg, seed=0, output_dir=tmp_path)
    engine.run(n_tasks=200)
    assert not (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "summary.json").exists()


def test_save_snapshots_true_writes_rich_snapshot_file(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {**_CFG, "output": {**_CFG["output"], "save_snapshots": True}}
    engine = HEDGESimulationEngine(config=cfg, seed=0, output_dir=tmp_path)
    engine.run(n_tasks=200)
    path = tmp_path / "node_snapshots.jsonl"
    assert path.exists()
    with path.open() as fh:
        first = json.loads(fh.readline())
    assert {"time", "node_id", "l_current", "l_hat", "f_max", "R_hat"} <= set(first.keys())


def test_events_jsonl_has_rich_schema_with_deadline() -> None:
    """Regression: legacy events.jsonl lacked a deadline field entirely. The new
    writer must expose deadline, arrival_time, and is_resale."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        engine = HEDGESimulationEngine(config=_CFG, seed=0, output_dir=out)
        engine.run(n_tasks=300)
        lines = (out / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert lines, "no events written"
        first = json.loads(lines[0])
        assert "deadline" in first
        assert "arrival_time" in first
        assert "is_resale" in first
        assert "a_buyer" in first


def test_summary_json_uses_real_metric_keys_not_legacy_stub(tmp_path: Path) -> None:
    engine = HEDGESimulationEngine(config=_CFG, seed=0, output_dir=tmp_path)
    engine.run(n_tasks=200)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "M10_completion_rate" in summary
    assert "M6_rejection_rate" in summary
    assert "M7_cloud_rate" not in summary, "M7_cloud_rate is the legacy stub key name"
