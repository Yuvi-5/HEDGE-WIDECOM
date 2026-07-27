# Results

Raw evidence backing every reported number in the paper. Every value in the
paper traces back to one of these files.

## Aggregated CSVs

- `hedge_c_comparison.csv` — 6 arms x 30 seeds, main comparison.
- `phase0_ablation.csv` — Phase-0 on/off, 2 arms x 30 seeds.
- `economics_supplement.csv` — edge profit, buyer surplus, cloud
  revenue/profit, total user spend, 6 arms x 30 seeds.
- `FINAL_SUMMARY_main_arms.csv` / `FINAL_SUMMARY_phase0_ablation.csv` —
  mean + 95% CI per arm, the "just show me the numbers" summary of the
  campaign.
- `coverage_stat.json` — mean coverage-set size over the real N=125
  topology (seed-invariant, no subsampling at full dataset size).

## Per-seed raw checkpoints

`hedge_c_comparison/`, `economics_supplement/`, `phase0_ablation/` each
contain one `<arm>/seed_NN/hedge_c_summary.json` per run (~1 KB each) — the
engine's native metrics plus floor-violation and resale-leak validation
counts, computed from `engine._mc.events` in memory. These are what the
aggregated CSVs above are built from; `scripts/run_hedge_c_comparison.py`
and friends are resumable off this exact file, so re-running the campaign
with these checkpoints present will skip completed seeds.

## Not included here

- `floor_scatter_seed0.csv` (114 MB) — the per-task `(price_paid,
  C1_winner, executor_id)` dump behind Figure 5's pricing-floor scatter.
  Too large to check into git as a one-off diagnostic; regenerate in
  ~10 minutes with `python scripts/make_floor_scatter.py`.
- `diagnostic_profit/` — an early, capped-duration probe superseded by the
  full 30-seed `economics_supplement/` above.
