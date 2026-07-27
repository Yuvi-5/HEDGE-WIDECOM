# HEDGE-WIDECOM

Reproduction package for:

> **HEDGE: A Decentralized Edge-to-Edge Framework for Efficient Resource
> Management**
> Yuvraj Singh Palh, A B M Bodrul Alam, Faria Khandaker
> School of Computer Science & Technology, Algoma University
> Submitted to WIDECOM 2026

## What this is

Vertical-only offloading escalates every overloaded edge task straight to
the cloud, leaving idle peer capacity a single hop away structurally
unused. HEDGE is a horizontal edge-to-edge market built on three
mechanisms: a per-task decentralized orchestrator election (one round trip,
no central coordinator), a single-pass affordable-first greedy matching
rule with a proven signaling budget, and a two-stage price — a physical
cost floor topped by a closed-form Stackelberg congestion markup — that
guarantees sellers are never priced below cost by construction. The paper
proves individual rationality and weak budget balance, and shows a
vertical-only configuration is an exact internal special case of the same
mechanism, isolating every measured gain to horizontal cooperation. It is
evaluated on the real EUA-Melbourne topology and the real Alibaba 2018
cluster trace against a vertical economic baseline and two published
edge-pricing schemes.

This repository contains the trimmed slice of the underlying simulator that
this paper's evaluation actually depends on — carbon-aware pricing, the
resale market, Kalman-filter prediction, the extra baselines/ablations, and
the burst-arrival generators from the broader research codebase have been
removed outright, not just disabled, since none of them is switched on for
this paper's configuration. Every number reported in the paper traces back
to a raw CSV or per-seed JSON under `results/`.

## Repository layout

```
src/hedge/           simulator engine: core entities, two-stage pricing,
                      orchestrator election + matching, baselines B2/B4/B6/B7
src/data/loaders/     real EUA-Melbourne topology + Alibaba trace loaders
src/experiments/      generic per-seed run + CSV-writing helpers
src/metrics/          the M1-M17 metric computations
src/visualization/    shared plot style + per-figure plotting code
configs/              the exact 3-file config chain used for every run
                      (hedge_c_base.yaml -> real_heavy.yaml -> default.yaml)
data/                 real EUA-Melbourne topology + Alibaba trace subset
scripts/              campaign runners, aggregation, and figure generation
tests/                unit / integration / invariant tests
results/              raw per-seed campaign output backing every paper number
```

## Evaluation arms

| Arm | Description |
|---|---|
| `HEDGE_C` | The full mechanism as described in the paper |
| `HEDGE_C_Kmax0` | Vertical-only ablation (`K_max=0`) — the internal special case proved in the paper |
| `B2_DDPS` | Published baseline |
| `B4_CostOPD` | Published baseline |
| `B6_GreedyNLF` | Published baseline |
| `B7_CloudOnly` | Cloud-only fallback baseline |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -v                 # full suite
pytest tests/unit -v             # component-level tests only
pytest tests/integration -v      # engine/RFQ/market wiring tests
pytest tests/invariants -v       # IR, WBB, Bertrand-floor property tests
```

## Commands

Every script below is run from the repo root with the venv active. Every
campaign runner is resumable: re-running with the same output directory
skips any seed that already has a summary file, unless `--force` is passed.
`--n-tasks` caps a run by task count instead of the full 3600s simulated
duration — useful for smoke tests, since real per-seed campaigns take
minutes each (task counts arrive far faster than simulated time advances,
so very small `--n-tasks` values may finish before the 300s warmup clears
and report all-zero metrics; `--n-tasks 5000` or higher reliably clears it).

```bash
# Inspect the exact resolved configuration used for every run
# (follows the hedge_c_base.yaml -> real_heavy.yaml -> default.yaml chain)
python scripts/dump_resolved_config.py

# Mean radio-coverage set size |C_u| over the real N=125 EUA-Melbourne
# topology (seed-invariant at full dataset size, no simulation loop needed)
python scripts/compute_coverage_stat.py

# Main comparison: 6 arms (HEDGE_C, HEDGE_C_Kmax0, B2_DDPS, B4_CostOPD,
# B6_GreedyNLF, B7_CloudOnly). Omit --arm to run all 6.
python scripts/run_hedge_c_comparison.py --arm HEDGE_C --n-tasks 5000 --n-seeds 1   # smoke test
python scripts/run_hedge_c_comparison.py                                            # full 30-seed x 6-arm run

# Phase-0 (proximity radio-coverage prefilter) on/off ablation, 2 arms
python scripts/run_phase0_ablation.py --arm Phase0_On --n-tasks 5000 --n-seeds 1
python scripts/run_phase0_ablation.py

# Supplementary campaign: edge/cloud profit, margin, buyer surplus per arm
# (fields the main 21-metric campaign above doesn't capture)
python scripts/run_economics_supplement.py --arm HEDGE_C
python scripts/run_economics_supplement.py

# Convenience wrapper: runs the main comparison then the Phase-0 ablation
# sequentially, passing --n-seeds/--n-tasks/--force through to both
python scripts/run_all_campaigns.py --n-tasks 5000 --n-seeds 1
python scripts/run_all_campaigns.py

# Consolidate the raw per-seed CSVs into a mean + 95% CI summary per arm
python scripts/build_final_summary.py

# Validate campaign integrity (Bertrand-floor violations, resale-leak check,
# config propagation, K_max=0 recovery) and log the computed headline
# numbers; also fills paper/numbers.tex macros if that file is present
python scripts/aggregate_and_write_numbers.py

# One-off diagnostic: edge-seller profit and buyer surplus per arm,
# computed directly rather than via the main campaign's metric columns
python scripts/diagnostic_edge_profit.py

# Figures (each writes into outputs/figures/)
python scripts/make_topology_figure.py       # network topology map
python scripts/make_widecom_figures.py       # figs 1-4 and 6, from the FINAL_SUMMARY CSVs
python scripts/make_floor_scatter.py         # fig 5: pricing-floor integrity scatter (re-runs one full-duration seed)
```

`data/README.md` documents dataset provenance and how to regenerate the
Alibaba trace subset from the full upstream archive if you want more than
the shipped 1-hour slice.

## Status

The paper is complete and submitted for review. `results/README.md`
documents the raw evidence backing every reported number. One reporting
quirk was found and worked around during evaluation: the DDPS baseline
intentionally prices off a narrower cost basis (DVFS only, no CAPEX/carbon
floor) than the C1_total figure its events report, so its raw revenue can
look higher than its true profit — this is accounted for wherever profit
is computed, not silently left in the headline cost numbers.

## License

MIT — see `LICENSE`.
