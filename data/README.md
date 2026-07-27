# Data

Both files here are the real, unmodified datasets the paper's evaluation
runs against.

## `eua_melbourne.csv` (14 KB)

The EUA (Edge User Allocation) Melbourne CBD base-station dataset —
real-world coordinates and coverage radii for edge deployment sites,
introduced in Lai et al., "Optimal Edge User Allocation in Edge Computing
with Variable Sized Vector Bin Packing," ICSOC 2018. `configs/default.yaml`
loads it at `topology.eua_data_path`. Every arm in this paper's evaluation
uses the full `N=125`-node dataset (`topology.N_nodes: 125`), no
subsampling.

## `alibaba_1h_subset.parquet` (400 KB)

A one-hour subset of the Alibaba Cluster Trace 2018 `batch_task` table
(source: <https://github.com/alibaba/clusterdata>, v2018), preprocessed
into the columns `src/hedge/simulation/scheduler.py` and
`src/data/loaders/alibaba_loader.py` consume directly. `configs/default.yaml`
loads it at `arrivals.alibaba_data_path` / `arrivals.trace_path`.

### Regenerating a different subset

The full upstream `batch_task` table is ~800 MB uncompressed and is **not**
shipped in this repo. To build a different subset (e.g. a longer window,
different task-count cap):

```bash
# Download batch_task.csv from https://github.com/alibaba/clusterdata (v2018)

python src/data/loaders/alibaba_loader.py \
    --raw-path /path/to/batch_task.csv \
    --output data/alibaba_1h_subset.parquet \
    --max-tasks <N>
```

See the docstring at the top of `src/data/loaders/alibaba_loader.py` for
the exact column mapping and preprocessing this applies.
