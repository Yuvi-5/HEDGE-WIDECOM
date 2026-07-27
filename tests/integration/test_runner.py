"""Phase 11 gate: integration tests for the experiment runner and visualization pipeline.

Gate criteria:
- A2 ablation (K_max sweep) runs without error for 2 seeds
- Output CSV has correct schema (K_max column, metric columns)
- Wilcoxon test runs between K_max=0 and K_max=4 columns
- Figure generation produces at least one PNG without error
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.runner import (
    _deep_merge,
    _set_nested_param,
    compute_and_write_wilcoxon,
    expand_sweep,
    load_config,
    run_experiment,
    write_metrics_csv,
)
from visualization.generate_all import generate_all_figures

# ---------------------------------------------------------------------------
# Minimal fast config for integration tests (synthetic topology, few tasks)
# ---------------------------------------------------------------------------

_FAST_CONFIG: dict[str, Any] = {
    "label": "test_runner",
    "simulation": {
        "duration": 3600.0,
        "random_seed": 0,
        "delta_tp": 0.1,
        "delta_q": 0.1,
        "warmup_duration": 0.0,
    },
    "topology": {
        "source": "synthetic",
        "N_nodes": 5,
        "tau_c_mean": 0.035,
        "tau_c_std": 0.008,
    },
    "arrivals": {
        "mode": "poisson",
        "lambda_bar": 5.0,
        "n_hotspots": 1,
        "hotspot_rate_multiplier": 3.0,
        "quiet_rate_multiplier": 0.5,
    },
    "task_model": {
        "w_min": 1.0e9,
        "w_max": 5.0e10,
        "s_min": 1.0e6,
        "s_max": 2.0e7,
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
    "output": {"log_level": "WARNING"},
}

_SWEEP_CONFIG: dict[str, Any] = {
    **_FAST_CONFIG,
    "sweep": {"K_max": [0, 4]},
}

# ---------------------------------------------------------------------------
# Unit tests for config utilities
# ---------------------------------------------------------------------------


def test_deep_merge_basic() -> None:
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    override = {"b": {"x": 99}, "c": 3}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["x"] == 99
    assert result["b"]["y"] == 20
    assert result["c"] == 3


def test_deep_merge_does_not_mutate_base() -> None:
    base = {"a": {"nested": 1}}
    override = {"a": {"nested": 2}}
    result = _deep_merge(base, override)
    assert base["a"]["nested"] == 1
    assert result["a"]["nested"] == 2


def test_set_nested_param_top_level() -> None:
    config: dict[str, Any] = {"K_max": 0}
    _set_nested_param(config, "K_max", 8)
    assert config["K_max"] == 8


def test_set_nested_param_nested() -> None:
    config: dict[str, Any] = {"hedge": {"K_max": 0, "H_max": 2}}
    found = _set_nested_param(config, "K_max", 4)
    assert found is True
    assert config["hedge"]["K_max"] == 4
    assert config["hedge"]["H_max"] == 2


def test_set_nested_param_missing() -> None:
    config: dict[str, Any] = {"hedge": {"H_max": 2}}
    found = _set_nested_param(config, "NONEXISTENT_PARAM", 99)
    assert found is False


def test_set_nested_param_dotted_path_resolves_exact_key() -> None:
    config: dict[str, Any] = {
        "arrivals": {"mode": "poisson"},
        "predictor": {"mode": "kalman"},
    }
    found = _set_nested_param(config, "predictor.mode", "exp_smooth")
    assert found is True
    assert config["predictor"]["mode"] == "exp_smooth"
    assert config["arrivals"]["mode"] == "poisson", "dotted path must not touch the sibling key"


def test_set_nested_param_bare_key_ambiguity_hits_first_match() -> None:
    """Documents the exact ambiguity a dotted path is meant to avoid: a bare leaf
    key search returns whichever section is visited first (insertion order)."""
    config: dict[str, Any] = {
        "arrivals": {"mode": "poisson"},
        "predictor": {"mode": "kalman"},
    }
    found = _set_nested_param(config, "mode", "exp_smooth")
    assert found is True
    assert config["arrivals"]["mode"] == "exp_smooth"
    assert config["predictor"]["mode"] == "kalman", "bare 'mode' matched the wrong section"


def test_set_nested_param_dotted_path_missing_returns_false() -> None:
    config: dict[str, Any] = {"predictor": {"mode": "kalman"}}
    found = _set_nested_param(config, "predictor.nonexistent_key", 1)
    assert found is False
    found2 = _set_nested_param(config, "nonexistent_section.mode", 1)
    assert found2 is False


def test_expand_sweep_raises_on_unresolved_axis() -> None:
    """A sweep axis that resolves nowhere must fail loudly, not silently no-op
    (this is the exact bug class that produced an all-identical algo sweep)."""
    config: dict[str, Any] = {
        "label": "bad_sweep",
        "sweep": {"predictor_mode": ["exp_smooth", "kalman"]},  # wrong key name
    }
    with pytest.raises(ValueError, match="predictor_mode"):
        expand_sweep(config)


def test_expand_sweep_dotted_axis_resolves() -> None:
    config: dict[str, Any] = {
        "label": "dotted_sweep",
        "arrivals": {"mode": "poisson"},
        "predictor": {"mode": "kalman"},
        "sweep": {"predictor.mode": ["instantaneous", "exp_smooth", "kalman"]},
    }
    variants, sweep_vals, params = expand_sweep(config)
    assert len(variants) == 3
    modes = {v["predictor"]["mode"] for v in variants}
    assert modes == {"instantaneous", "exp_smooth", "kalman"}
    for v in variants:
        assert v["arrivals"]["mode"] == "poisson", "dotted sweep must not perturb arrivals.mode"


def test_expand_sweep_no_sweep() -> None:
    import copy

    config = copy.deepcopy(_FAST_CONFIG)
    variants, sweep_vals, params = expand_sweep(config)
    assert len(variants) == 1
    assert params == []
    assert sweep_vals == [{}]


def test_expand_sweep_kmax() -> None:
    import copy

    config = copy.deepcopy(_SWEEP_CONFIG)
    variants, sweep_vals, params = expand_sweep(config)
    assert len(variants) == 2
    assert "K_max" in params
    kmax_values = [sv["K_max"] for sv in sweep_vals]
    assert set(kmax_values) == {0, 4}


def test_expand_sweep_sets_nested_kmax() -> None:
    import copy

    config = copy.deepcopy(_SWEEP_CONFIG)
    variants, sweep_vals, params = expand_sweep(config)
    for variant, sv in zip(variants, sweep_vals):
        assert variant["hedge"]["K_max"] == sv["K_max"]


def test_load_config_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(_FAST_CONFIG, fh)
    loaded = load_config(config_path)
    assert loaded["label"] == "test_runner"
    assert loaded["hedge"]["K_max"] == 4


def test_load_config_with_base_inheritance(tmp_path: Path) -> None:
    base_cfg = {"label": "base", "hedge": {"K_max": 4, "H_max": 2}}
    child_cfg = {"_base_": "base.yaml", "label": "child", "hedge": {"K_max": 8}}
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    with base_path.open("w") as f:
        yaml.dump(base_cfg, f)
    with child_path.open("w") as f:
        yaml.dump(child_cfg, f)
    loaded = load_config(child_path)
    assert loaded["label"] == "child"
    assert loaded["hedge"]["K_max"] == 8
    assert loaded["hedge"]["H_max"] == 2  # inherited from base


# ---------------------------------------------------------------------------
# CSV writer tests
# ---------------------------------------------------------------------------


def test_write_metrics_csv(tmp_path: Path) -> None:
    rows = [
        {"experiment": "test", "seed": 0, "K_max": 0, "M5_rejection_rate": 0.1},
        {"experiment": "test", "seed": 1, "K_max": 4, "M5_rejection_rate": 0.05},
    ]
    out = tmp_path / "metrics.csv"
    write_metrics_csv(rows, out)
    assert out.exists()
    with out.open() as f:
        reader = csv.DictReader(f)
        loaded = list(reader)
    assert len(loaded) == 2
    assert "K_max" in loaded[0]


def test_write_metrics_csv_empty(tmp_path: Path) -> None:
    out = tmp_path / "empty_metrics.csv"
    write_metrics_csv([], out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# Wilcoxon CSV test
# ---------------------------------------------------------------------------


def test_compute_and_write_wilcoxon(tmp_path: Path) -> None:
    rows = [
        {"K_max": 0, "M5_rejection_rate": 0.30},
        {"K_max": 0, "M5_rejection_rate": 0.25},
        {"K_max": 0, "M5_rejection_rate": 0.28},
        {"K_max": 4, "M5_rejection_rate": 0.10},
        {"K_max": 4, "M5_rejection_rate": 0.08},
        {"K_max": 4, "M5_rejection_rate": 0.12},
    ]
    out = tmp_path / "wilcoxon.csv"
    compute_and_write_wilcoxon(rows, out, sweep_param="K_max", reference_value=0, compare_value=4)
    assert out.exists()
    with out.open() as f:
        result = list(csv.DictReader(f))
    assert len(result) == 1
    assert "p_value" in result[0]
    assert "significant" in result[0]


def test_wilcoxon_insufficient_samples(tmp_path: Path) -> None:
    rows = [{"K_max": 0, "M5_rejection_rate": 0.2}]  # only 1 sample
    out = tmp_path / "wilcoxon.csv"
    compute_and_write_wilcoxon(rows, out, sweep_param="K_max", reference_value=0, compare_value=4)
    assert out.exists()
    with out.open() as f:
        result = list(csv.DictReader(f))
    assert result[0]["p_value"] == "1.0"


# ---------------------------------------------------------------------------
# Figure generation tests
# ---------------------------------------------------------------------------


def test_generate_all_figures_no_data(tmp_path: Path) -> None:
    fig_dir = tmp_path / "figures"
    generated = generate_all_figures(tmp_path, fig_dir)
    assert len(generated) >= 1
    assert all(p.suffix == ".png" for p in generated)
    assert all(p.exists() for p in generated)


def test_generate_all_figures_with_kmax_sweep_data(tmp_path: Path) -> None:
    rows = [
        {
            "experiment": "test",
            "algorithm": "HEDGE",
            "seed": s,
            "K_max": k,
            "M5_rejection_rate": 0.3 - 0.05 * k,
            "M6_completion_rate": 0.7 + 0.05 * k,
            "M1_mean_latency_s": 0.05 + 0.01 * k,
            "M2_mean_cost_usd": 0.1,
            "M3_p90_latency_s": 0.08,
            "M7_cloud_rate": 0.2,
        }
        for k in [0, 4]
        for s in range(3)
    ]
    csv_path = tmp_path / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig_dir = tmp_path / "figures"
    generated = generate_all_figures(tmp_path, fig_dir)
    assert len(generated) >= 1
    assert all(p.exists() for p in generated)
    assert any(p.suffix == ".png" for p in generated)


# ---------------------------------------------------------------------------
# End-to-end runner integration test (runs actual simulations, ~10-30s)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_experiment_sweep_end_to_end(tmp_path: Path) -> None:
    """Full pipeline: sweep K_max=[0,4], 2 seeds, 20 tasks each.

    Gate criteria verified:
    - CSV written with correct schema including K_max column
    - Wilcoxon CSV written
    - At least one PNG generated
    """
    import copy

    config = copy.deepcopy(_SWEEP_CONFIG)
    config_path = tmp_path / "test_sweep.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    metrics_path = run_experiment(
        config_path=config_path,
        n_seeds=2,
        seed_start=0,
        output_dir=tmp_path / "outputs",
        n_tasks=20,
    )

    # Gate criterion 1: CSV exists with correct schema
    assert metrics_path.exists(), "metrics.csv not written"
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4, f"Expected 4 rows (2 variants x 2 seeds), got {len(rows)}"
    assert "K_max" in rows[0], "K_max column missing from CSV"
    assert "seed" in rows[0], "seed column missing from CSV"
    assert "experiment" in rows[0], "experiment column missing from CSV"

    # Gate criterion 2: at least one metric column present
    metric_cols = [k for k in rows[0] if k.startswith("M")]
    assert len(metric_cols) >= 1, "No metric columns in CSV"

    # Gate criterion 3: Wilcoxon CSV exists
    wilcoxon_path = tmp_path / "outputs" / "wilcoxon.csv"
    assert wilcoxon_path.exists(), "wilcoxon.csv not written"
    with wilcoxon_path.open() as f:
        wrows = list(csv.DictReader(f))
    assert len(wrows) >= 1
    assert "p_value" in wrows[0]

    # Gate criterion 4: at least one PNG generated
    fig_dir = tmp_path / "outputs" / "figures"
    pngs = list(fig_dir.glob("*.png"))
    assert len(pngs) >= 1, f"No PNG files generated in {fig_dir}"
