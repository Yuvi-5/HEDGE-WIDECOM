"""Alibaba Cluster Trace v2018 loader.

Preprocesses raw batch-task traces into HEDGETask objects.
Raw dataset: https://github.com/alibaba/clusterdata (v2018, batch_task table).

For reviewer reproducibility, a pre-processed 1-hour anonymised subset at
data/alibaba_1h_subset.parquet is used when the full trace is unavailable.
Generate it via:
    python src/data/loaders/alibaba_loader.py \\
        --raw-path /path/to/alibaba --output data/alibaba_1h_subset.parquet
"""

from __future__ import annotations

import sys as _sys
from dataclasses import replace
from pathlib import Path as _Path

_src_dir = _Path(__file__).parent.parent.parent
if str(_src_dir) not in _sys.path:
    _sys.path.insert(0, str(_src_dir))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hedge.core.constants import (
    A_BUYER_MAX,
    A_BUYER_MIN,
    D_MAX,
    D_MIN,
    S_MAX,
    S_MIN,
    W_MAX,
    W_MIN,
)
from hedge.core.task import HEDGETask

# Reference machine: 32 cores at 3 GHz -> 96e9 cycles/second
_REF_CORES: int = 32
_REF_FREQ_HZ: float = 3e9
_REF_CYCLES_PER_S: float = _REF_CORES * _REF_FREQ_HZ  # 96e9

# Deadline slack factor: d = duration * 1.5
_DEADLINE_SLACK: float = 1.5

# Memory-to-size proxy: 1 GB -> 1e7 bits (rough scale to bits)
_MEM_TO_BITS: float = 1e7


def _task_from_row(
    row: pd.Series,
    task_idx: int,
    rng: np.random.Generator,
    sim_time_origin: float,
    sim_time_scale: float,
) -> HEDGETask:
    """Convert one Alibaba batch_task row to a HEDGETask."""
    cpu_avg = float(row.get("cpu_avg", row.get("plan_cpu", 0.1)))
    mem_avg = float(row.get("mem_avg", row.get("plan_mem", 0.01)))
    start_t = float(row.get("start_time", row.get("start", 0.0)))
    end_t = float(row.get("end_time", row.get("end", start_t + 1.0)))

    duration_s = max(0.01, end_t - start_t) * sim_time_scale
    w_raw = cpu_avg * _REF_CYCLES_PER_S * duration_s
    w = float(np.clip(w_raw, W_MIN, W_MAX))

    d_raw = duration_s * _DEADLINE_SLACK
    d = float(np.clip(d_raw, D_MIN, D_MAX))

    s_raw = mem_avg * _MEM_TO_BITS
    s = float(np.clip(s_raw, S_MIN, S_MAX))

    arrival_time = (start_t - sim_time_origin) * sim_time_scale
    a_buyer = float(rng.uniform(A_BUYER_MIN, A_BUYER_MAX))

    return HEDGETask(
        task_id=f"alibaba_{task_idx}",
        s=s,
        w=w,
        d=d,
        a_buyer=a_buyer,
        created_at=max(0.0, arrival_time),
    )


def load_alibaba_trace(
    data_path: str | Path,
    max_tasks: int = 100_000,
    seed: int = 42,
    sim_time_scale: float = 1.0,
) -> list[HEDGETask]:
    """Load and preprocess Alibaba 2018 batch task trace.

    Args:
        data_path: Path to batch_task CSV or parquet (raw or preprocessed subset).
        max_tasks: Maximum number of tasks to return (randomly subsampled if more).
        seed: Random seed for a_buyer sampling and subsampling.
        sim_time_scale: Scale raw timestamps to simulation seconds (default 1.0).

    Returns:
        List of HEDGETask objects sorted by created_at.

    Raises:
        FileNotFoundError: If data_path does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Alibaba trace not found at '{data_path}'. "
            "Download from https://github.com/alibaba/clusterdata (v2018) "
            "or regenerate the subset per data/README.md."
        )

    # Alibaba 2018 batch_task.csv ships without a header row.
    _ALIBABA_COLS = [
        "task_name",
        "instance_num",
        "job_name",
        "task_type",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
    ]

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        # Peek at first row to detect whether a header is present
        _peek = pd.read_csv(data_path, nrows=1, header=None)
        _first_val = str(_peek.iloc[0, 0])
        _has_header = _first_val.lower() == "task_name"
        if _has_header:
            df = pd.read_csv(data_path)
        else:
            df = pd.read_csv(data_path, header=None, names=_ALIBABA_COLS)

    rng = np.random.default_rng(seed)

    # Subsample if needed
    if len(df) > max_tasks:
        idx = rng.choice(len(df), size=max_tasks, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    # Preprocessed parquet already has HEDGE columns; load directly
    if "created_at" in df.columns and "w" in df.columns:
        df = df.dropna(subset=["w", "s", "d", "created_at"]).reset_index(drop=True)
        tasks: list[HEDGETask] = []
        for i, row in df.iterrows():
            tasks.append(
                HEDGETask(
                    task_id=str(row.get("task_id", f"alibaba_{i}")),
                    s=float(row["s"]),
                    w=float(row["w"]),
                    d=float(row["d"]),
                    a_buyer=float(row.get("a_buyer", rng.uniform(A_BUYER_MIN, A_BUYER_MAX))),
                    created_at=float(row["created_at"]),
                )
            )
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    # Raw trace: compute fields from plan_cpu / start_time columns
    time_col = next((c for c in df.columns if "start" in c.lower()), df.columns[0])
    sim_time_origin = float(df[time_col].min())

    tasks = []
    for i, row in df.iterrows():
        task = _task_from_row(
            row=row,
            task_idx=int(str(i)) + 1,
            rng=rng,
            sim_time_origin=sim_time_origin,
            sim_time_scale=sim_time_scale,
        )
        tasks.append(task)

    tasks.sort(key=lambda t: t.created_at)
    return tasks


def remap_to_heavy_envelope(
    tasks: list[HEDGETask],
    w_range: tuple[float, float],
    d_range: tuple[float, float],
    s_range: tuple[float, float],
) -> list[HEDGETask]:
    """Rank-transform each task's (w, s, d) into an edge-native operating envelope.

    The default preprocessing (_task_from_row) maps real Alibaba cpu_avg/mem_avg/
    duration values through a fixed formula clipped to the LIGHT task_model
    range (W_MIN..W_MAX etc from hedge.core.constants). That range spans
    orders of magnitude wider than any edge-appropriate "heavy load" envelope
    (see configs/heavy_load.yaml), so a naive re-clip into a narrower window
    saturates almost every task to the clip boundary and destroys
    heterogeneity. Instead, this rank-transforms each task's PERCENTILE within
    the trace's own w/s/d distribution into the target range independently per
    field, preserving each task's relative standing (a job that needed more
    compute than 90% of the trace still needs more compute than 90% of the
    trace after remapping) while moving the absolute scale into a regime where
    edge nodes are compute-feasible and cloud is transfer-time-infeasible --
    the same design principle used to calibrate heavy_load.yaml for synthetic
    tasks. Buyer valuation (a_buyer) is left untouched (economic, not physical).

    Caveat: w/s/d are rank-transformed INDEPENDENTLY, so any real correlation
    between compute demand and data size in the original trace (e.g. bigger
    jobs also using more memory) is not preserved after remapping. This is a
    disclosed limitation, not a silent one.

    Args:
        tasks: Tasks loaded via load_alibaba_trace (light-scale w/s/d).
        w_range: (min, max) target workload range (cycles).
        d_range: (min, max) target deadline range (seconds).
        s_range: (min, max) target data-size range (bits).

    Returns:
        New list of HEDGETask with w/s/d remapped; a_buyer/created_at/task_id
        unchanged.
    """
    n = len(tasks)
    if n == 0:
        return []

    def _percentile_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        return order / max(n - 1, 1)

    w_pct = _percentile_ranks(np.array([t.w for t in tasks], dtype=np.float64))
    s_pct = _percentile_ranks(np.array([t.s for t in tasks], dtype=np.float64))
    d_pct = _percentile_ranks(np.array([t.d for t in tasks], dtype=np.float64))

    w_lo, w_hi = w_range
    s_lo, s_hi = s_range
    d_lo, d_hi = d_range

    remapped: list[HEDGETask] = []
    for i, t in enumerate(tasks):
        remapped.append(
            replace(
                t,
                w=float(w_lo + w_pct[i] * (w_hi - w_lo)),
                s=float(s_lo + s_pct[i] * (s_hi - s_lo)),
                d=float(d_lo + d_pct[i] * (d_hi - d_lo)),
            )
        )
    return remapped


def _cli_preprocess() -> None:
    """CLI entry point: preprocess raw Alibaba trace to a parquet subset.

    Reads the full raw CSV, filters to the first --duration seconds of the trace,
    then writes up to --max-tasks rows as a parquet subset for fast simulation loading.
    """
    parser = argparse.ArgumentParser(description="Preprocess Alibaba 2018 batch trace")
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--max-tasks", type=int, default=100_000)
    args = parser.parse_args()

    _COLS = [
        "task_name",
        "instance_num",
        "job_name",
        "task_type",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
    ]
    raw = Path(args.raw_path)
    if raw.suffix == ".parquet":
        df_raw = pd.read_parquet(raw)
    else:
        _peek = pd.read_csv(raw, nrows=1, header=None)
        _has_hdr = str(_peek.iloc[0, 0]).lower() == "task_name"
        df_raw = pd.read_csv(raw) if _has_hdr else pd.read_csv(raw, header=None, names=_COLS)

    time_col = next((c for c in df_raw.columns if "start" in c.lower()), df_raw.columns[0])
    t_min = float(df_raw[time_col].min())
    df_window = df_raw[(df_raw[time_col] - t_min) <= args.duration].reset_index(drop=True)

    rng = np.random.default_rng(42)
    if len(df_window) > args.max_tasks:
        idx = rng.choice(len(df_window), size=args.max_tasks, replace=False)
        df_window = df_window.iloc[sorted(idx)].reset_index(drop=True)

    records = []
    for i, row in df_window.iterrows():
        t = _task_from_row(
            row=row,
            task_idx=int(str(i)) + 1,
            rng=rng,
            sim_time_origin=t_min,
            sim_time_scale=1.0,
        )
        records.append(
            {
                "task_id": t.task_id,
                "s": t.s,
                "w": t.w,
                "d": t.d,
                "a_buyer": t.a_buyer,
                "created_at": t.created_at,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)


if __name__ == "__main__":
    _cli_preprocess()
