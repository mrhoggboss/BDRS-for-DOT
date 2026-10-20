# HALO (fetched at a pinned commit)

Upstream: https://github.com/WenzhouXia/HALO, the official implementation of "A
Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems" (Xia,
Zhu, Liang, Zhang; ICLR 2026). Pinned commit (`PIN`):
`1671ae690597deea1f17e3928d5ac282f8760d23`.

## Fetching

The upstream tree is not committed here. `fetch.sh` clones it at `PIN`, checks the SHA, removes `.git`,
syncs it into `upstream/` (gitignored) and applies the repairs below. This directory commits
only `PIN`, `fetch.sh`, `build.sh` and this file.

## Build

```bash
bash vendor/halo/fetch.sh     # needs network: the upstream clone and a pybind11 release tarball
bash vendor/halo/build.sh     # the pycupdlpx CUDA extension -> _build/ (gitignored)
```

`build.sh` replaces upstream's micromamba-based
`lp_solvers/build_solvers/build_cupdlpx_micromamba.sh`. It configures
`…/lp_solvers/cuPDLPx/pycupdlpx` with CMake out of tree (`_cmake/`, gitignored) and copies
`pycupdlpx*.so` and `libcupdlpx*.so` into `_build/`. It reads `PYTHON` (default `python3`)
and `CUDA_HOME` (default `/usr/local/cuda`), takes the CUDA architecture from the compute
capability `nvidia-smi` reports (80 when that query fails), and links with an `$ORIGIN`
RPATH so that `pycupdlpx` finds `libcupdlpx` beside it (if the RPATH is missing it sets it
with `patchelf`, and fails when `patchelf` is not installed). It ends by importing
`pycupdlpx` and `hierarchical_ot`. Nothing is installed into site-packages:
`src/runners/run_halo.py` puts `upstream/src` and `_build` on `sys.path`.

Runtime dependencies, all in the repository's `requirements.txt`: `cupy-cuda12x` (grid-mode
GPU kernels), `numba` (imported at the top of `core/solver_utils.py`; upstream's
`pyproject.toml` declares no dependencies) and `nvidia-ml-py` (the `pynvml` module, for the
runner's peak-memory sampler).

## Repairs applied by `fetch.sh`

Paths are relative to `upstream/src/hierarchical_ot/`. Repair 1 is skipped when the pybind11
tooling is already present and checked afterwards. Repairs 2 to 5 each replace an exact text
anchor of the upstream file: a file that already holds the repaired text is left unchanged,
and an anchor that is missing or occurs more than once stops the script, so a new pin cannot
silently drop a repair. Repairs 2 to 4 add a comment that points to this file.

1. **pybind11 tooling.** Upstream's `lp_solvers/cuPDLPx/.gitignore` ignores `*.cmake`, so
   the published tree lacks pybind11's `tools/*.cmake` and `add_subdirectory(pybind11)`
   fails. `fetch.sh` overlays the pybind11 v2.12.0 release, the version upstream vendors,
   onto `lp_solvers/cuPDLPx/pycupdlpx/pybind11/`.
2. **numpy ≥ 2.** In `modes/grid/violation_kernels.py`, `find_violations_local_dt_gpu`
   reads a shape-(1,) cupy counter with `int(ctr1.get())`, which numpy ≥ 2 rejects; it
   becomes `int(ctr1.get()[0])`. This function handles hierarchy levels of side
   `r ≥ 512`, so smaller grids never reach it.
3. **Empty violation set.** The same function reads `slk_u[-1]` even when a round finds no
   violation, which raises `IndexError` on the empty array. The repair adds the guard of its
   sibling `find_violations_global_gpu`: an empty index set and criterion 0.0.
4. **Host allocations above 2 GiB.** cuPDLPx's `safe_malloc(int)` and
   `safe_calloc(int, int)` (`lp_solvers/cuPDLPx/cupdlpx/utils.cu` and `utils.h`) truncate a
   size above 2³¹ − 1 bytes; `malloc` then fails and the library calls
   `exit(EXIT_FAILURE)`, which ends the Python process. The parameters become `size_t`
   (with `<stddef.h>` in the header); the call sites already compute `size_t` products.
5. **C++17.** `lp_solvers/cuPDLPx/pycupdlpx/CMakeLists.txt` sets the C++ and CUDA standards
   to 14; CUDA 13's headers (CCCL) require C++17, so both become 17.

## Use in the benchmark

Algorithm `HALO`, in the PointCloud rung (`src/protocol/regimes.py`), `sql2` only: the
upstream code is written for the squared Euclidean cost (`l2^2`). Grid instances run grid
mode (`emd2_grid` on the `n_g × n_g` mass arrays); point-cloud instances run tree mode
(`emd2` on the coordinates scaled by `1/√c_max`, so HALO solves the max-1 cost). Grid mode's
native cost is `(Δi² + Δj²)/n_g²`; the runner recomputes the objective in fp64 on the max-1
cost from the returned sparse coupling and rescales HALO's dual by `n_g²/(2(n_g − 1)²)`.
Under `--pareto` neither mode receives `spec.tolerance`, so upstream's default tolerance
`1e-6` applies.
