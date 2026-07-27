"""Scope guard: verify runtime path never imports torch or tensorflow at module level."""

from __future__ import annotations

import importlib
import sys
import types


def _import_fresh(module_name: str) -> types.ModuleType:
    """Import a module, returning cached entry if already present."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    return importlib.import_module(module_name)


_RUNTIME_MODULES = [
    "hedge.core.constants",
    "hedge.core.task",
    "hedge.core.cloud",
    "hedge.core.node",
    "hedge.core.user",
    "hedge.core.topology",
    "hedge.pricing.layer1",
    "hedge.pricing.layer2",
    "hedge.pricing.layer2_5",
    "hedge.predictor.instantaneous",
    "hedge.predictor.ema",
    "hedge.market.phase0",
    "hedge.market.rfq",
    "hedge.market.afgm",
    "hedge.simulation.engine",
    "hedge.simulation.scheduler",
]

# Match on module root name only (avoids "tf" in "matfuncs" false positives)
_FORBIDDEN_ROOTS = {"torch", "tensorflow"}


def _is_forbidden(module_name: str) -> bool:
    """Return True if module_name's root package is torch or tensorflow."""
    root = module_name.split(".")[0]
    return root in _FORBIDDEN_ROOTS


def test_runtime_modules_do_not_import_torch_at_top_level() -> None:
    """No runtime module may pull in torch or tensorflow as a top-level side effect.

    Lazy imports inside function bodies are permitted (A10 ablation only).
    """
    before = set(sys.modules.keys())

    for mod in _RUNTIME_MODULES:
        _import_fresh(mod)

    after = set(sys.modules.keys())
    newly_imported = after - before

    forbidden_found = {m for m in newly_imported if _is_forbidden(m)}
    assert not forbidden_found, (
        f"Runtime import pulled in forbidden modules: {forbidden_found}. "
        "torch/tensorflow must only appear inside function bodies (lazy imports)."
    )


def test_baselines_have_no_torch() -> None:
    """Baselines B2/B4 are pure-numpy and must never import torch at module level."""
    b_modules = [
        "hedge.baselines.b2_ddps",
        "hedge.baselines.b4_cost_opd",
    ]
    before = set(sys.modules.keys())
    for mod in b_modules:
        _import_fresh(mod)
    after = set(sys.modules.keys())
    newly_imported = after - before
    forbidden_found = {m for m in newly_imported if _is_forbidden(m)}
    assert not forbidden_found, f"Baseline import pulled in forbidden modules: {forbidden_found}"
