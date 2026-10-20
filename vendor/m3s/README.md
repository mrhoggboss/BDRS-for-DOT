# M3S (fetched at a pinned commit, with an extraction shim)

Upstream: https://github.com/ucascnic/CudaOT, which contains the official GPU implementation
of M3S, the multiscale sparse Sinkhorn solver of "Computing Wasserstein-p Distance Between
Images with Linear Cost" (Chen, Li, Lu; CVPR 2022): ε-scaling Sinkhorn on a grid hierarchy
with sparse kernels. Pinned commit (`PIN`): `ec100cb67e445d400cb61064e6eb6924debb5d80`.

## Fetching

The upstream tree is not committed here. `fetch.sh` clones it at `PIN`, checks the SHA, removes `.git` and syncs it
into `upstream/` (gitignored). It applies no changes. The files committed in this directory
(`PIN`, `fetch.sh`, `build.sh`, `shim/` and this file) are first-party, with one
qualification: `shim/m3s_shim.cu` contains modified copies of two functions of upstream's
`SparseSinkhorn/solver.cu`, the body of `cuda_image_ot_solver` (`solver.cu:1241-1631`) and
`solveSingle_` (`solver.cu:181-367`), which the extraction of the plan and the duals requires;
the comments of that file mark both copies.

## Build

```bash
bash vendor/m3s/fetch.sh    # clone upstream at PIN into upstream/ (needs network)
bash vendor/m3s/build.sh    # CMake build of shim/ -> _build/m3s_shim (gitignored)
```

Requirements: the CUDA toolkit (nvcc, cuBLAS, cuSPARSE), CMake ≥ 3.18 and GCC (any version)
as the host compiler (see below; `build.sh` stops with a message when `CXX`, by default
`c++`, or `CUDAHOSTCXX`, by default `$CXX` or else `g++`, is not GCC). Eigen, OpenMP, CPLEX and Lemon are not needed: `shim/CMakeLists.txt` compiles only
the shim, `SparseSinkhorn/kernel_manage.cu` and upstream's `Common` library sources, and
drops the `Sinkhorn` library that upstream links into its `SparseSinkhorn` target without
using it. `build.sh` takes the CUDA architecture from `nvidia-smi` (override with
`M3S_CUDA_ARCH`; 80 when the query fails or returns no number). Do not run upstream's
top-level CMake: the tree contains committed CMake caches and prebuilt `.a` archives,
requires Eigen and OpenMP for other targets, and sets `LINK_DIRECTORIES("/home")`.

`shim/CMakeLists.txt` compiles host code at `-O0` with `-fno-unreachable-traps`, as
upstream's own build effectively does: two upstream functions
(`Familys::check_for_layers`, `CuEspsScalingHandler_::init_eps`) can reach the end of an
`int`-returning body without a `return`. gcc exploits this undefined behaviour at `-O2` and
above by deleting the loop-exit test, and from gcc 13 on turns it into a trap instruction at
`-O0`; `-fno-unreachable-traps` restores the fall-through. The flag exists only from GCC 13
on and is passed only to a GCC that has it: older GCC inserts no trap and builds the same
host code without it (the shim behind the paper's rows was built with GCC 12.3). The host
compiler, also nvcc's, must be GCC; select it with `CXX` and `CUDAHOSTCXX`. Device code
keeps the default `-O3`.

## The extraction shim

The published API `image_ot_solver(image_size, muX, muY)` returns the OT distance only, with
no plan and no duals. `shim/m3s_shim.cu` therefore includes upstream's
`SparseSinkhorn/solver.cu` (include-guarded by `MY_SK_H`; upstream's `image_ot_solver.cu`
includes it the same way) and repeats the body of `cuda_image_ot_solver<image_size>` with
these additions:

1. reading the marginals from binary blobs and writing the results to blobs;
2. copying the final coupling and duals to the host at the end of the solve;
3. per-layer iteration counters. These need a copy of upstream's `solveSingle_` with one
   added counter line; the copy also casts the arguments of three `%ld` `printf` calls to
   `long`, which changes no output;
4. device synchronizations that bound the timed solve, and early returns that record a
   nonzero `msg`.

The upstream sources are not modified, and the solver's parameters are upstream's hard-wired
values (an ε schedule ending at 0.1, `maxError = 1e-4`, 1000 inner iterations between error
checks). Upstream's unconditional dump of the coupling to `output.csv` in the working
directory is replaced by the blob outputs. `shim/cuda12_compat.cuh`, force-included into every
CUDA translation unit, supplies two APIs that upstream uses and current CUDA lacks, with the
same results: `__shfl_down` on sm_70 and newer (as `__shfl_down_sync` over the full warp, where
upstream calls it from a converged warp) and `cusparseDgthr`, removed in CUDA 12 (as a gather
kernel).

Details for a reader of the outputs:

- The coupling entry is `π_k = val[k]·u[row_k]·v[col_k]`, the expression upstream uses to
  score its own plan. The shim copies it before upstream's `calculate_result`, which
  overwrites `kernel->val`.
- The native cost is the squared Euclidean distance between integer pixel coordinates
  (`p_index = 2`; a `static_assert` stops the build if upstream changes it). The runner
  divides the objective by `2(n_g − 1)²` to obtain the max-1 `sql2` cost.
- `alpha.bin` and `beta.bin` are the finest layer's absorbed potentials, in the native units.
  In this stabilized-scaling form the kernel carries the reference measure,
  `K_ij = exp((α_i + β_j − c_ij)/ε)·μX_i·μY_j`, and `u = v = 1` at delivery, so the
  potentials of the entropic problem are `α_i + ε·log μX_i` and `β_j + ε·log μY_j`, with `ε`
  the final value `eps_final` in `meta.json`. The runner applies this correction.
- `--warmup` runs one full, untimed solve on a fixed pair of smooth dummy images of the same
  `n_g` before the measured solve; the runner passes it when the spec asks for a warm-up.

## Shim interface

```
m3s_shim <n_g> <r.bin> <c.bin> <outdir> [--warmup]
```

Inputs: fp64 vectors of length `n_g²` (row-major images, unit mass, strictly positive:
upstream's `TMultiScaleSetupSingleBase::BasicMeasureChecks` rejects a measure whose minimum is
≤ 0). The runner passes the marginals floored with `floor_marginal` at `δ_m = 1e-12` and
scores the result against the original marginals.

Outputs in `<outdir>`: `plan_row.bin` and `plan_col.bin` (int32), `plan_val.bin` (fp64 π
values), `alpha.bin` and `beta.bin` (fp64), and `meta.json` (`msg`, `distance_native`,
`eps_final`, `shim_solve_seconds`, `shim_extract_seconds`, `nnz`, `iters_per_layer`, …;
written on every completed solve, also when upstream returns a nonzero `msg`).

Exit codes: 0 success; 3 nonzero upstream solver `msg` (recorded in `meta.json`); 1 for
usage errors, blob I/O errors and failures of upstream's `CHECK` macro, which exits 1 itself
and prints "out of memory" on an allocation failure; 2 when one of the shim's own CUDA calls
(device synchronizations and copies) reports an allocation failure. The shim allocates no
device memory itself, so an out-of-memory solve normally exits 1 with "out of memory" on
stderr, and the runner records any exit whose stderr reports an OOM as an `oom` row.

## `n_g = 1024` is refused

The size macros `GETLOG2` and `ALLOCALTE_MEMORY` extend to 1024, but upstream's structs are
sized by `MAXSIZE 512` (`Common/basic_settings.h`): `XData::data[MAXSIZE·MAXSIZE·3]` and
`COO_Sparese_Matrix::crs_row_ptrl[MAXSIZE·MAXSIZE + 1]` (`Common/data_struct.h`). The
hierarchy builder always keeps the original pixels as the finest layer, so at `n_g = 1024`
the position hierarchy overflows `XData::data` on the host and `cusparseXcoo2csr` overflows
`crs_row_ptrl` on the device. Running it would need upstream edits, so the shim accepts
`n_g ∈ {32, 64, 128, 256, 512}` and exits 1 for 1024, and the runner records an `n_g = 1024`
instance as an `oom` row without starting the shim.

## Use in the benchmark

Algorithm `M3S`, in the PointCloud rung (`src/protocol/regimes.py`), `sql2` only: the cost
kernel computes the Euclidean distance to the power `p_index = 2`, fixed in upstream's
headers. It solves the entropic problem at the final ε of its schedule (0.1 in pixel² units),
so its plan is not an exact LP solution. `spec.tolerance` and `spec.iter_cap` are recorded but
not passed, since the shim has no stopping parameter. The stop is upstream's: each ε stage ends
when the L1 error of the first marginal is at most `maxError = 1e-4`. Runner:
`src/runners/run_m3s.py`.

M3S runs on grid instances only; on a point-cloud instance the orchestrator skips it with
reason `m3s-image-grid-only`. Its only GPU entry point, `cuda_image_ot_solver<image_size>`,
is templated on a power-of-two grid side, and its hierarchy, kernel generation and ε schedule
are built for image grids. Upstream's multiscale code for arbitrary point positions runs on
the CPU only.
