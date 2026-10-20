# BDRS for Discrete Optimal Transport

This is the code of the paper [*GPU-Accelerated Bregman Douglas–Rachford Splitting for Discrete Optimal Transport*](https://arxiv.org/abs/2610.04715). The environment of the paper's experiments is listed under [Environment of the paper](#environment-of-the-paper).

This repository contains GPU solvers for the **discrete optimal transport (OT)** linear program based on **Bregman Douglas–Rachford splitting (BDRS)**, and the benchmark that compares them with GPU and CPU OT solvers on the **DOTmark** images, point clouds and random costs.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart)
- [Reproducing the paper](#reproducing-the-paper)
- [Environment of the paper](#environment-of-the-paper)
- [Citation](#citation)

---

## Repository layout

```
src/
  methods/     the BDRS solvers (grid_cuda.py with torch reference grid_torch.py for Grid;
               factored_cuda.py with torch reference factored_torch.py for Point Cloud and General),
               the CUDA extension (cuda/, built by cuda_ext.py), the c-transform
               certificate (dual_cleanup.py), cost helpers, eta.py, params.py
  protocol/    the orchestrator-runner contract (instance_record.py), the instance
               classification (rung) rules (regimes.py), the reader for result rows
  instances/   DOTmark build and the upsampled n_g = 1024 size, the dataset registry and loaders,
               dense costs, the dense DOT LP operator
  scoring/     KKT components of a recovered (X, α, β), the per-run Pareto block
  runners/     one subprocess runner per solver (run_<solver>.py), shared runner code,
               the extension point for new baselines (baseline.py)
  report/      aggregation of result rows, SGM10 statistics, accuracy-time plots
  benchmark/   the orchestrator: scheduling, instance lifts, dispatch, the PDOT adapter
  studies/     the drivers of the dual-stall and robustness studies (Figs. 6 and 7)
  figures/     the paper's figures, drawn from the aggregated runs and the studies' rows
               (python -m src.figures.make)
tests/         the solver's integrity tests (tests/methods): the CUDA kernels against the torch
               reference, certificates against exact optima
scripts/       build_all_baselines.sh, capture_software_versions.py
vendor/        third-party baselines; provenance and pins in vendor/README.md
requirements.txt, constraints.txt, requirements-ott.txt
               the Python dependencies, the paper's versions of them, and the
               separate environment of the SinkhornGrid baseline
DOTmark_1.0/   the downloaded DOTmark CSVs (gitignored; see Data)
ModelNet10/    the downloaded ModelNet10 meshes (gitignored; see Data)
data/          generated input data (gitignored)
```

---

## Installation

```bash
git clone https://github.com/mrhoggboss/BDRS-for-DOT.git && cd BDRS-for-DOT
pip install torch -c constraints.txt --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -c constraints.txt
```

Use Python 3.11. `constraints.txt` holds the versions of the paper's environment. The CUDA extension of our solvers compiles on first use and needs `nvcc` and a host C++ compiler.

**SinkhornGrid** (OTT-JAX) runs in its own environment, which the orchestrator calls through `$BDRS_OTT_PYTHON`:

```bash
python -m venv ~/ott-env && ~/ott-env/bin/pip install -r requirements-ott.txt
export BDRS_OTT_PYTHON=~/ott-env/bin/python
```

**Other baselines** are built by `bash scripts/build_all_baselines.sh`, which prints an OK/FAIL line per baseline (pins and provenance in `vendor/README.md`). It expects:
- `julia` to start Julia 1.11, for PDOT and LAMP (their CUDA.jl rejects 1.12);
- GCC as the host compiler of M3S (set `CXX` and `CUDAHOSTCXX` to a `g++` if the default is Clang);
- no CUDA toolkit `lib64` on `LD_LIBRARY_PATH` in the shells that build or run PDOT and LAMP, so that CUDA.jl loads its own libraries:
  ```bash
  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr : '\n' | grep -vi 'cuda.*/lib64' | paste -sd: -)
  ```

Importing HALO prints a warning that its optional HPR-LP backend is missing; the benchmark does not use it.

---

## Data

Run the commands below from the repository root: DOTmark and ModelNet10 are downloaded and unpacked there, and the builders write the instances under `data/`.

**DOTmark** comes from the [DOTmark page](https://www.stochastik.math.uni-goettingen.de/index.php?id=215/) (about 110 MB):

```bash
curl -LO https://www.stochastik.math.uni-goettingen.de/DOTmark_1.0.zip
echo "11156bef4a1e1710f5f394271ea24d49e95a7b341d9ee0fa0feda50b43bc8201  DOTmark_1.0.zip" | sha256sum -c
unzip -q DOTmark_1.0.zip 'Data/*' -d DOTmark_1.0
```

On macOS, use `shasum -a 256 -c` in place of `sha256sum -c`. The images are then at `DOTmark_1.0/Data/<Class>/data<n_g>_<id>.csv`, and the instances are built with

```bash
python -m src.instances.dotmark_build                                     # n_g = 32, 64, 128
python -m src.instances.dotmark_build --resolutions 32 64 128 256 512 --skip-costs
python -m src.instances.dotmark_upsample_1024 --norms l1 sql2             # n_g = 1024 tier
```

This writes unit-mass marginals under `data/marginals/`, optional max-1-normalized cost matrices under `data/cost_matrices/` (every dense runner builds the lattice cost when the file is missing; one file is 34 GB at `n_g = 256`, so build at your own discretion) and the index `data/dotmark_instances_index.json`.

**ModelNet10** comes from the [ModelNet page](https://modelnet.cs.princeton.edu/) (about 473 MB); only its test meshes are unpacked:

```bash
curl -LO https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
echo "9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51  ModelNet10.zip" | sha256sum -c
unzip -q ModelNet10.zip 'ModelNet10/*/test/*'
```

The meshes are then at `ModelNet10/<class>/test/`. The registry builds the paper's instances (10 sizes, 3 shapes per class) into `data/modelnet10/`:

```bash
python -m src.instances.registry --datasets modelnet10 --n-pairs 3 --seed 20260904 \
    --scales 128 256 512 1024 2048 4096 8192 16384 32768 65536
```

**`random_dense`** needs no download, because its costs are regenerated from seeds; the registry builds the paper's instances into `data/random_dense/` with

```bash
python -m src.instances.registry --datasets random_dense --n-pairs 10 --seed 20260907
```

---

## Quickstart

Run everything from the repository root.

### Solve one instance on the GPU (Grid)

```python
import torch
from src.methods.params import BregmanDRParams
from src.methods.eta import eta_from_mean, grid_cost_mean
from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
from src.methods.dual_cleanup import feasible_dual_and_gap, make_gap_stop_fn

ng, norm, tol = 64, "sql2", 1e-6                       # m = n = ng * ng = 4096
r = torch.rand(ng * ng, dtype=torch.float64, device="cuda"); r /= r.sum()
c = torch.rand(ng * ng, dtype=torch.float64, device="cuda"); c /= c.sum()

eta = eta_from_mean(grid_cost_mean(ng, norm))          # 0.7 * mean(C)
params = BregmanDRParams(eta=eta, max_iters=100_000, epsilon_primal=tol, check_every=200)

def certify(info, objective):                          # c-transform of (η log u, η log v)
    return feasible_dual_and_gap(info["log_u"], info["log_v"], eta, r, c, ng, norm, objective)

_, info = bregman_dr_solve_grid_cuda(
    None, r, c, params, cost_type=norm, grid_size=ng, return_plan=False,
    gap_stop_fn=make_gap_stop_fn(certify), gap_stop_tol=tol)

alpha, beta, g_feas = feasible_dual_and_gap(
    info["log_u"], info["log_v"], eta, r, c, ng, norm, info["objective"])
print(info["native_termination"], info["iters"], info["primal_res_L2"], g_feas)
```

### Solve one instance on the CPU (General)

```python
import torch
from src.methods.params import BregmanDRParams
from src.methods.eta import eta_from_mean
from src.methods.factored_torch import bregman_dr_solve_general_factored
from src.methods.dual_cleanup import make_dense_gap_stop_fn, feasible_dual_and_gap_storedC

g = torch.Generator().manual_seed(0)
C = torch.rand(300, 200, generator=g, dtype=torch.float64); C /= C.max()
r = torch.rand(300, generator=g, dtype=torch.float64); r /= r.sum()
c = torch.rand(200, generator=g, dtype=torch.float64); c /= c.sum()

tol = 1e-6
eta = eta_from_mean(C.mean())
params = BregmanDRParams(eta=eta, max_iters=200_000, epsilon_primal=tol, check_every=50)
X, info = bregman_dr_solve_general_factored(
    C, r, c, params, gap_stop_fn=make_dense_gap_stop_fn(C, r, c), gap_stop_tol=tol)

alpha, beta, g_feas = feasible_dual_and_gap_storedC(
    info["log_u"], info["log_v"], eta, r, c, C, info["objective"])
print(info["native_termination"], int(info["iters"]), float(info["primal_res_L2"]), g_feas)
```

The solvers assume, as the paper does (Section 4.2), that `C ≥ 0` and that `C` is not the zero matrix. With `C = 0` the step-size rule gives `η = 0.7·mean(C) = 0`, and the solver functions raise `ZeroDivisionError` instead of returning. No benchmark instance has a zero cost: each is normalized to `max C = 1`.

---

## Reproducing the paper

Every figure of the paper ([arXiv:2610.04715](https://arxiv.org/abs/2610.04715)) is regenerated from this repository in three steps: run the benchmark on the figure's instances, merge the runs, then aggregate them and draw the figure.

> **This takes a long time.** The commands below run about 98,600 solves (Fig. 1 repeats 5,400 runs of Figs. 4b and 5), which add up to about 930 GPU-hours of solve time on an A100-SXM4-80GB, plus a few seconds of start-up per run and about a minute per LAMP shard. Split each figure into shards that run in parallel, one GPU per shard, as described below. A single figure is between 0.2 and about 350 GPU-hours. To draw the figures from the paper's own runs without a GPU, see [Redrawing the figures from the paper's data](#redrawing-the-figures-from-the-papers-data); for a check of a few hours on one GPU, see [A reduced reproduction](#a-reduced-reproduction-fig-1-on-50-instances).

**Protocol.** The runs follow the paper's protocol (Section 5.1). Give every shard its resources, one A100-SXM4-80GB, 16 CPU cores and 93 GB of host memory, and its time limit of 1800 s (`--time-limit 1800`, the default); do not raise the limit. Each row records the limit of its run (`algo_extras["time_limit_s"]`); a row that records none, such as the paper's, is held to 1800 s. `src.report.aggregate` and `src.figures.make` count as a timeout (reason `time_limit`) any run whose `t_solve` exceeds its limit, whatever its row says, and plot the paper's time to certificate. In it, DROT's time leaves out two clocks that time our harness, not DROT: its shim writing the plan to disk and the runner reading it back. A baseline run without `t_total` has no time to certificate and is left out (reason `no_t_total`).

**Before you start.**
- Build the data (see [Data](#data)): DOTmark at every resolution, including the `n_g = 1024` tier, and, for the figures that use them, `random_dense` and ModelNet10.
- Build every baseline and set `BDRS_OTT_PYTHON` (see [Installation](#installation)); its Julia and `LD_LIBRARY_PATH` settings apply to the shells of the PDOT and LAMP shards.
- LaTeX is optional: with `latex` on `PATH` the figures are typeset as in the paper, and without it in matplotlib's bundled Computer Modern (`BDRS_FIG_TEX=0` or `1` forces either).

**LAMP** (`LAMP` on the General format, `LAMPPointCloud` on the Point Cloud format) runs only in an orchestrator process without other algorithms (the two names may share one); the orchestrator refuses any other mix. Its runs are solved by a Julia server, which the first LAMP run of a job starts and later runs reuse: about a minute of start-up and warm-up, on no clock. A run that ends at LAMP's own 600 s limit (Section 5.1) is an `ok` row with `native_termination` `tmax`. The server keeps a CUDA context on the GPU until it has been idle for 300 s (`BDRS_LAMP_IDLE_TIMEOUT`), so it must not overlap a timed shard of another solver on that GPU. When a LAMP shard ends, wait 300 s before starting the next shard on its GPU, or stop the server from the same job, with the same `CUDA_VISIBLE_DEVICES`:

```bash
python -m src.runners.lamp_client shutdown
```

### Redrawing the figures from the paper's data

The rows behind the paper's figures are attached to the `v2.0` release as `bdrs-v2.0-paper-data.tar.gz` (29 MB): one row per run of the paper's benchmark (96,361 runs, Figs. 1 to 5) and the rows of the two studies (Figs. 6 and 7). Drawing all nine figures from them takes seconds and needs no GPU:

```bash
curl -LO https://github.com/mrhoggboss/BDRS-for-DOT/releases/download/v2.0/bdrs-v2.0-paper-data.tar.gz
echo "b7a93c230587a1cb5fb67ad765792fd1574b19bfe6d3ef9a84de9214db829141  bdrs-v2.0-paper-data.tar.gz" | sha256sum -c
tar -xzf bdrs-v2.0-paper-data.tar.gz
python -m src.figures.make --figure 1 2 3 4a 4b 5 --runs bdrs-v2.0-paper-data/figs1-5/pareto.csv.gz --out results/paper_figures
python -m src.figures.make --figure 6 --runs bdrs-v2.0-paper-data/fig6 --out results/paper_figures
python -m src.figures.make --figure 7a 7b --runs bdrs-v2.0-paper-data/fig7 --out results/paper_figures
```

`bdrs-v2.0-paper-data/README.md` describes the files and the columns of the per-run table.

### A reduced reproduction: Fig. 1 on 50 instances

Fig. 1's cell (DOTmark, `sql2`, `n_g = 64`) is the one cell in which all three formats and every solver run. The commands below run it on pairs (0,1) to (0,5) of each of the ten classes, 50 of its 450 instances, with every solver and tolerance of the figure: 1,150 runs, one after another on one GPU. They need DOTmark at `n_g = 64` (`python -m src.instances.dotmark_build --resolutions 64`, after the download under [Data](#data)) and every baseline built.

```bash
for TOL in 1e-4 1e-6 1e-8; do
  python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
      --pairs 0,1 0,2 0,3 0,4 0,5 \
      --algos Grid PointCloud General --policy-exception General:sql2 \
      --tolerance $TOL --output-dir results/reduced
done
python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
    --pairs 0,1 0,2 0,3 0,4 0,5 \
    --algos HOT HALO M3S DROT CUPDLPX PDOT Sinkhorn SinkhornGrid --policy-exception General:sql2 \
    --sinkhorn-eps 0.01 0.001 0.0001 --ott-eps 0.01 0.001 0.0001 \
    --output-dir results/reduced
python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
    --pairs 0,1 0,2 0,3 0,4 0,5 \
    --algos LAMP LAMPPointCloud --policy-exception General:sql2 \
    --output-dir results/reduced
python -m src.runners.lamp_client shutdown
python -m src.report.aggregate --runs-dir results/reduced/runs --output-dir results/reduced/summaries
python -m src.figures.make --figure 1 --runs results/reduced/summaries/pareto.csv \
    --out results/reduced/figures --cells-out results/reduced/figures
```

In the paper, these 1,150 runs took 3.6 hours of solve time on an A100-SXM4-80GB, 1.6 h of it PDOT's (one PDOT run reached the 1800 s limit). The largest GPU memory a run of these recorded is at most 4.2 GiB for every solver except the two that run in Julia: PDOT recorded up to 41 GiB, and LAMP up to 42 GiB on the General format and 5.2 GiB on the Point Cloud format. No GPU smaller than the A100-SXM4-80GB has been tested.

The paper's data above hold the rows of exactly these runs in `reduced_fig1/pareto.csv.gz`, with Fig. 1 drawn from them and its cells (`reduced_fig1/pareto_exp1_sql2_n64.pdf` and `.csv`). Compare `results/reduced/figures/` with them, and each run's `status` and `iters`, matched by `(instance_id, algorithm, tolerance)`; times depend on the GPU. On these 50 instances, the paper's Fig. 1 is close to the full figure: each arm's time to certificate is between 0.97 and 1.09 times its value over the 450 instances, except HOT (1.26) and PDOT (1.18), and each arm's KKT residual between 0.90 and 1.12 times.

### Sharding

The orchestrator runs one (solver, instance) pair at a time and appends one row per run to `<output-dir>/runs/<algorithm>.jsonl`. To shard a figure, run several orchestrator processes at once, one per GPU. Give each process:
- a disjoint part of the work, chosen with `--algos`, `--tolerance`, `--classes` (DOTmark), `--resolutions` (DOTmark) or `--sizes` (ModelNet10, `random_dense`);
- its own `--output-dir`. Two processes must never write to the same directory.

A shard that stops early (a node failure, a job time limit) can be resubmitted with the same command: runs that already finished are skipped, except runs recorded with status `error`, which run again. A few runs at the 1800 s limit dominate the time of several cells, so size the shards for them, not for typical run times.

When every shard has finished, merge them, aggregate them and draw the figure. For Fig. 1:

```bash
mkdir -p results/fig1/runs
for f in results/fig1_shards/*/runs/*.jsonl; do cat "$f" >> "results/fig1/runs/$(basename "$f")"; done
python -m src.report.aggregate --runs-dir results/fig1/runs --output-dir results/fig1/summaries
python -m src.figures.make --figure 1 --runs results/fig1/summaries/pareto.csv --out results/figures
```

Each figure is written to `--out` as a PDF named as in the paper's LaTeX source (here `pareto_exp1_sql2_n64.pdf`); `python -m src.figures.make --help` lists all nine.

### Fig. 1: every format on DOTmark, `n_g = 64`

The General-format runs on `sql2` need `--policy-exception General:sql2`.

```bash
for TOL in 1e-4 1e-6 1e-8; do
  python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
      --algos Grid PointCloud General --policy-exception General:sql2 \
      --tolerance $TOL --output-dir results/fig1_shards/ours_$TOL
done
python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
    --algos HOT HALO M3S DROT CUPDLPX Sinkhorn SinkhornGrid --policy-exception General:sql2 \
    --sinkhorn-eps 0.01 0.001 0.0001 --ott-eps 0.01 0.001 0.0001 \
    --output-dir results/fig1_shards/baselines
for C in "CauchyDensity ClassicImages" "GRFmoderate GRFrough" "GRFsmooth LogGRF" \
         "LogitGRF MicroscopyImages" "Shapes WhiteNoise"; do
  python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
      --algos PDOT --policy-exception General:sql2 --classes $C \
      --output-dir "results/fig1_shards/pdot_${C%% *}"
done
python -m src.benchmark.orchestrator --pareto --resolutions 64 --norms sql2 \
    --algos LAMP LAMPPointCloud --policy-exception General:sql2 \
    --output-dir results/fig1_shards/lamp
```

Each loop iteration and each command is one shard; PDOT is split into five shards by class because its runs dominate the figure, and LAMP runs in a shard of its own. Merge, aggregate and draw with the commands under [Sharding](#sharding). `src.figures.make` leaves the ε = 1e-4 SinkhornGrid arm out of the co-solved set itself, as the caption of Fig. 1 says, and `--cells-out DIR` writes the cells a figure draws, with their co-solved counts, to `DIR/<figure file>.csv`.

### Fig. 2: General format, lifted DOTmark

```bash
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 --norms l1 l2 inf \
    --algos General DROT Sinkhorn --sinkhorn-eps 0.01 0.001 0.0001 \
    --output-dir results/fig2_shards/main
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 --norms l1 l2 inf \
    --algos PDOT --output-dir results/fig2_shards/pdot
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 --norms l1 l2 inf \
    --algos LAMP --output-dir results/fig2_shards/lamp
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 --norms l1 l2 inf \
    --algos CUPDLPX --output-dir results/fig2_shards/cupdlpx
python -m src.benchmark.orchestrator --pareto --resolutions 128 --norms l1 l2 inf \
    --algos CUPDLPX --pairs 0,3 0,7 2,6 2,7 3,8 --output-dir results/fig2_shards/cupdlpx_128
python -m src.benchmark.orchestrator --pareto --resolutions 128 --norms inf --algos CUPDLPX \
    --classes CauchyDensity --pairs 0,1 --output-dir results/fig2_shards/cupdlpx_128_inf
```

Shard by `--algos` and `--resolutions`, and split PDOT, LAMP and DROT at `n_g = 128` further with `--classes`. cuPDLPx's host-memory failure at `n_g = 128` (the caption of Fig. 2) ends each runner by the out-of-memory killer; the orchestrator writes an `oom` row (`runner_killed_-9`) and goes on, and under Slurm such a job ends in state `OUT_OF_MEMORY` although every run has its row. The paper therefore ran cuPDLPx at `n_g = 128` on five pairs per class and one more `inf` pair only (150 of these 151 runs are `oom`, the last a timeout), and not at `n_g = 256`. LAMP's runs at `n_g = 256` are `oom` rows written without a solve. Merge the shards into `results/fig2/runs` as for Fig. 1, then aggregate and draw:

```bash
python -m src.report.aggregate --runs-dir results/fig2/runs --output-dir results/fig2/summaries
python -m src.figures.make --figure 2 --runs results/fig2/summaries/pareto.csv --out results/figures
```

### Fig. 3: General format, random cost matrices

```bash
python -m src.benchmark.orchestrator --pareto --datasets random_dense \
    --algos General DROT CUPDLPX Sinkhorn --sinkhorn-eps 0.01 0.001 0.0001 \
    --output-dir results/fig3_shards/tol1e-4
python -m src.benchmark.orchestrator --pareto --datasets random_dense --algos PDOT \
    --sizes 1024 4096 16384 --output-dir results/fig3_shards/pdot
python -m src.benchmark.orchestrator --pareto --datasets random_dense --algos General \
    --tolerance 1e-6 --output-dir results/fig3_shards/general_1e-6
python -m src.benchmark.orchestrator --pareto --datasets random_dense --algos LAMP \
    --output-dir results/fig3_shards/lamp
```

Shard by `--algos` and `--sizes`. The failures the caption of Fig. 3 lists appear as rows: cuPDLPx's runs at `N = 2^14` are `oom` rows (`runner_killed_-9`, as in Fig. 2), cuPDLPx and LAMP refuse `N = 2^16` at once with an `oom` row, PDOT is not run there, and the 10 `tol = 1e-6` runs of our General solver on the uniform cost at `N = 2^16` take about 5 GPU-h to reach the time limit. Merge the shards into `results/fig3/runs` as for Fig. 1, then aggregate and draw:

```bash
python -m src.report.aggregate --runs-dir results/fig3/runs --output-dir results/fig3/summaries
python -m src.figures.make --figure 3 --runs results/fig3/summaries/pareto.csv --out results/figures
```

### Fig. 4: Point Cloud format

Two figures with separate runs, (a) on ModelNet10 and (b) on lifted DOTmark; LAMP runs in shards of its own.

**(a) ModelNet10.** The `tol = 1e-6` runs go up to `N = 2^15`. HALO's rejection of `N = 2^7` and `2^8` (the caption of Fig. 4) is an `error` row on each of those 60 runs (`ValueError: Mode 'tree' does not support exact-flat solve.`): its tree mode sends a problem whose two clouds have at most 256 points each to a single-level exact solve, which the pinned upstream implements in grid mode only (`vendor/halo/upstream/src/hierarchical_ot/core/mode_dispatch.py`, after `fetch.sh`). A resubmitted shard runs them again, and they fail the same way. HALO's untimed warm-up in tree mode therefore runs on dummy clouds of 1024 points per side (`TREE_WARMUP_POINTS` in `src/runners/run_halo.py`), so that the first call of its multilevel solve (about 0.6 s on an A100) is not timed.

```bash
python -m src.benchmark.orchestrator --pareto --datasets modelnet10 --norms sql2 \
    --sizes 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
    --algos PointCloud HALO --output-dir results/fig4a_shards/tol1e-4
python -m src.benchmark.orchestrator --pareto --datasets modelnet10 --norms sql2 \
    --sizes 128 256 512 1024 2048 4096 8192 16384 32768 \
    --algos PointCloud --tolerance 1e-6 --output-dir results/fig4a_shards/tol1e-6
python -m src.benchmark.orchestrator --pareto --datasets modelnet10 --norms sql2 \
    --sizes 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
    --algos LAMPPointCloud --output-dir results/fig4a_shards/lamp
```

Shard by `--sizes`. Merge the shards into `results/fig4a/runs` as for Fig. 1, then aggregate and draw:

```bash
python -m src.report.aggregate --runs-dir results/fig4a/runs --output-dir results/fig4a/summaries
python -m src.figures.make --figure 4a --runs results/fig4a/summaries/pareto.csv --out results/figures
```

**(b) lifted DOTmark.** The `tol = 1e-6` runs cover every pair from `n_g = 32` to 128 and two pairs at `n_g = 256`, both of which reach the limit.

```bash
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 --norms sql2 \
    --algos PointCloud HALO M3S --output-dir results/fig4b_shards/tol1e-4
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 --norms sql2 \
    --algos PointCloud --tolerance 1e-6 --output-dir results/fig4b_shards/tol1e-6
python -m src.benchmark.orchestrator --pareto --resolutions 256 --norms sql2 \
    --classes CauchyDensity --pairs 0,1 0,2 \
    --algos PointCloud --tolerance 1e-6 --output-dir results/fig4b_shards/tol1e-6_n256
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 --norms sql2 \
    --algos LAMPPointCloud --output-dir results/fig4b_shards/lamp
```

Shard by `--resolutions`. Merge the shards into `results/fig4b/runs` as for Fig. 1, then aggregate and draw:

```bash
python -m src.report.aggregate --runs-dir results/fig4b/runs --output-dir results/fig4b/summaries
python -m src.figures.make --figure 4b --runs results/fig4b/summaries/pareto.csv --out results/figures
```

### Fig. 5: Grid format, DOTmark

What the figure draws at each size, and which of its crosses are projected rather than run, is stated in the caption of Fig. 5. The commands below make every run it draws. With the `l1` cost, `tol = 1e-6` at `n_g = 1024` runs every pair except the 32 WhiteNoise pairs that already reach the limit at `n_g = 512`, and `tol = 1e-8` at `n_g = 512` runs five pairs per class. The paper's data also hold `tol = 1e-6` runs at `n_g = 1024` with `sql2`, made under an earlier stopping rule and step size; the figure leaves them out.

```bash
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 512 1024 --norms l1 sql2 \
    --algos Grid HOT SinkhornGrid --ott-eps 0.01 0.001 0.0001 \
    --output-dir results/fig5_shards/tol1e-4
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 512 --norms l1 sql2 \
    --algos Grid --tolerance 1e-6 --output-dir results/fig5_shards/tol1e-6
python -m src.benchmark.orchestrator --pareto --resolutions 1024 --norms l1 \
    --classes CauchyDensity ClassicImages GRFmoderate GRFrough GRFsmooth \
              LogGRF LogitGRF MicroscopyImages Shapes \
    --algos Grid --tolerance 1e-6 --output-dir results/fig5_shards/tol1e-6_n1024
python -m src.benchmark.orchestrator --pareto --resolutions 1024 --norms l1 --classes WhiteNoise \
    --pairs 0,3 1,2 1,4 1,5 1,6 2,4 3,4 3,7 4,6 5,6 5,7 6,9 7,9 \
    --algos Grid --tolerance 1e-6 --output-dir results/fig5_shards/tol1e-6_n1024_wn
python -m src.benchmark.orchestrator --pareto --resolutions 32 64 128 256 --norms l1 sql2 \
    --algos Grid --tolerance 1e-8 --output-dir results/fig5_shards/tol1e-8
python -m src.benchmark.orchestrator --pareto --resolutions 512 --norms l1 \
    --pairs 0,3 0,7 2,6 2,7 3,8 \
    --algos Grid --tolerance 1e-8 --output-dir results/fig5_shards/tol1e-8_n512
```

Shard by `--resolutions`, and split `n_g = 512` and `1024` further with `--classes`. Merge the shards into `results/fig5/runs` as for Fig. 1, then aggregate and draw:

```bash
python -m src.report.aggregate --runs-dir results/fig5/runs --output-dir results/fig5/summaries
python -m src.figures.make --figure 5 --runs results/fig5/summaries/pareto.csv --out results/figures
```

### Fig. 6: the dual stall of BDRS, DROT and HOT

These runs use `src.studies.dual_stall`, not the orchestrator, with the iteration counts and checks of the caption of Fig. 6, on pairs (0,1) to (0,5) of each class. DROT and HOT are rerun from scratch to every check iteration, which is where the figure's 14 GPU-hours go. DROT reads the stored `sql2` cost matrices at `n_g` 32 and 64 (do not pass `--skip-costs` for these sizes) and runs on the default GPU, so select its GPU with `CUDA_VISIBLE_DEVICES`.

```bash
out=results/ctransform_ablation
run="python -m src.studies.dual_stall"
for ng in 32 64; do
  $run --solver BDRS --resolution $ng --output-dir $out/shards/n$ng/BDRS
done
$run --solver DROT --resolution 32 --output-dir $out/shards/n32/DROT
$run --solver HOT --resolution 32 --output-dir $out/shards/n32/HOT
for solver in DROT HOT; do
  for cls in CauchyDensity ClassicImages GRFmoderate GRFrough GRFsmooth \
             LogGRF LogitGRF MicroscopyImages Shapes WhiteNoise; do
    $run --solver $solver --resolution 64 --classes $cls --output-dir $out/shards/n64/${solver}_$cls
  done
done
```

Each command is one shard with its own `--output-dir`; run them in parallel, one GPU each. A BDRS shard takes a few minutes, DROT at `n_g = 32` under an hour, HOT at `n_g = 32` about 2 hours, and each `n_g = 64` class about 20 minutes (DROT) or 50 minutes (HOT). A shard run again skips the instances it has finished. Merge the shards and draw the figure with

```bash
out=results/ctransform_ablation
for ng in 32 64; do
  mkdir -p $out/n$ng/runs
  for solver in BDRS DROT HOT; do
    cat $out/shards/n$ng/$solver*/runs/$solver.jsonl > $out/n$ng/runs/$solver.jsonl
  done
done
python -m src.figures.make --figure 6 --runs $out --out results/figures
```

Each instance contributes an `instance` row, one `check` row per check iteration and a `result` row. The `native_*` fields of a check row belong to the solver's own triple, the `repaired_*` fields to the triple with the c-transformed dual.

### Fig. 7: robustness to the step size and the check period

These runs use `src.studies.robustness`, which calls the Grid and General runners directly at `tol = 1e-4` on the sweeps of the caption of Fig. 7. It also runs two arms the figure does not draw: the General solver on the same sweeps, and a wider η ladder (0.01 to 32 times the deployed η on `sql2`, with an iteration cap of 10^6). The 3,960 runs take about 15 minutes of solve time on an A100.

```bash
python -m src.studies.robustness --out results/fig7 --arms 1 2 3   # eta_sweep: the η ladder, Grid and General
python -m src.studies.robustness --out results/fig7 --arms 4       # eta_sweep_wide: the wide η ladder
python -m src.studies.robustness --out results/fig7 --arms 5 6     # tchk_sweep: the T_chk ladder, Grid and General
```

The three commands may run at once into the same `--out`, one GPU each (`CUDA_VISIBLE_DEVICES`); never run two on one GPU, because Fig. 7b plots solve times. A command run again skips the runs it has finished. `--smoke` runs a subset of 120 runs to check the setup. The rows are in `results/fig7/<tree>/runs/sweep.jsonl`. Fig. 7a is drawn from `eta_sweep` and Fig. 7b from `tchk_sweep`:

```bash
python -m src.figures.make --figure 7a 7b --runs results/fig7 --out results/figures
```

---

## Environment of the paper

The experiments ran on one NVIDIA A100-SXM4-80GB per solve, with 16 CPU cores and 93 GB of host memory.

| Component | Version or pinned commit | Source |
|-----------|--------------------------|--------|
| NVIDIA driver | 595.71.05 | the runs' `header.json`, written by `src/benchmark/env_capture.py` (the runs' records are not part of this release) |
| Operating system | Linux 5.14.0-570.58.1.el9_6.x86_64 (glibc 2.34) | the runs' `header.json` |
| CUDA toolkit (`nvcc`) | 13.0 (V13.0.88, build cuda_13.0.r13.0/compiler.36424714_0) | the runs' `header.json` |
| Host C++ compiler | GCC 12.3.0 | the runs' build environment |
| Python | 3.11.16 | the runs' `header.json` |
| PyTorch | 2.13.0+cu130 (CUDA runtime 13.0) | the runs' `header.json`; `constraints.txt` |
| NumPy, SciPy | 2.4.6, 1.17.1 | `constraints.txt` |
| trimesh (ModelNet10 point sampling) | 4.12.2 | `constraints.txt` |
| CuPy, Numba (HALO) | cupy-cuda12x 14.1.1, 0.66.0 | `constraints.txt` (not recorded by the runs) |
| matplotlib, pandas (figures) | 3.10.1, 2.2.3 | `constraints.txt` |
| POT (Sinkhorn) | 0.9.7 | `constraints.txt` |
| OTT-JAX (Grid Sinkhorn) | 0.6.0 | `requirements-ott.txt` |
| JAX | 0.10.2 (`jax[cuda12]`) | `requirements-ott.txt` |
| Julia (PDOT, LAMP) | 1.11.9 | `julia_version` in `vendor/PDOT.jl/Manifest.toml` and `vendor/lamp/Manifest.toml` |
| CUDA.jl (PDOT, LAMP) | 5.11.3 | the same two `Manifest.toml` files; `CUDA = "=5.11.3"` in both `Project.toml` files |
| DROT | [vienmai/drot](https://github.com/vienmai/drot) `a5ffa08d6911c7245984d3b0e5835f90767801f4` (2021-10-04) | committed in `vendor/drot/`, run through `vendor/drot/shim/` |
| HOT | [PolyU-IOR/HOT](https://github.com/PolyU-IOR/HOT) `4b0de44ed28622ece1ff63371828f0081cbb1580` (2025-07-04) | committed in `vendor/HOT/` |
| PDOT | [MIT-Lu-Lab/PDOT.jl](https://github.com/MIT-Lu-Lab/PDOT.jl) `7e1d762854dda732c78c02b66fe6a300a469486e` (2024-08-01) | committed in `vendor/PDOT.jl/`, run through `vendor/PDOT.jl/bench_driver.jl` |
| cuPDLPx | [MIT-Lu-Lab/cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) `931c94c0c3767d38b3df71514c84ccd314aa2ac7` (tag v0.2.9, 2026-03-31) | `vendor/cupdlpx/PIN`, committed in `vendor/cupdlpx/upstream/` |
| HALO | [WenzhouXia/HALO](https://github.com/WenzhouXia/HALO) `1671ae690597deea1f17e3928d5ac282f8760d23` (2026-04-24) | `vendor/halo/PIN`, fetched and repaired by `vendor/halo/fetch.sh` |
| M3S | [ucascnic/CudaOT](https://github.com/ucascnic/CudaOT) `ec100cb67e445d400cb61064e6eb6924debb5d80` (2023-11-12) | `vendor/m3s/PIN`, fetched by `vendor/m3s/fetch.sh`, run through `vendor/m3s/shim/` |
| LAMP | [mxburns2022/CuLAMP.jl](https://github.com/mxburns2022/CuLAMP.jl) `f3d32f56572dabd3e534040dab37742fcb18687e` (2026-05-07) | `vendor/lamp/PIN`, fetched by `vendor/lamp/fetch.sh`, loaded by `vendor/lamp/lamp_shim.jl` |

---

## Citation

If you find this code helpful, please cite:

```bibtex
@article{xu2026bdrs,
  title   = {{GPU}-Accelerated {B}regman {D}ouglas--{R}achford Splitting for Discrete Optimal Transport},
  author  = {Xu, Yifan and Ma, Shiqian},
  journal = {arXiv preprint arXiv:2610.04715},
  year    = {2026}
}
```
