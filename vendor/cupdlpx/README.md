# cuPDLPx (committed at a pinned release)

Upstream: https://github.com/MIT-Lu-Lab/cuPDLPx, a general GPU LP solver in fp64: restarted
Halpern PDHG with reflection, adaptive restarts and a PID-controlled primal weight (Lu, Peng,
Yang 2025). Pinned commit (`PIN`): `931c94c0c3767d38b3df71514c84ccd314aa2ac7`, tag v0.2.9.
The tree is committed unchanged under `upstream/`. The first-party files are `PIN`, `fetch.sh`, `build.sh` and this README.

This is upstream cuPDLPx, not the older copy embedded in HALO
(`vendor/halo/upstream/src/hierarchical_ot/lp_solvers/cuPDLPx`), which has a different layout
and its own `pycupdlpx` wrapper instead of the official binding. The two builds coexist:
`src/runners/run_halo.py` imports `pycupdlpx` from `vendor/halo/_build`, and
`src/runners/run_cupdlpx.py` imports the official `cupdlpx` package that `build.sh` installs.

## Build

```bash
bash vendor/cupdlpx/build.sh     # pip install --no-build-isolation --no-deps --force-reinstall ./upstream
python3 -c "import cupdlpx"
```

`build.sh` follows upstream's documented route, `pip install .` (scikit-build-core, pybind11,
CMake ≥ 3.21; upstream asks for CUDA ≥ 12.4), with two changes:

- `--no-build-isolation`, so the extension builds against the active environment's numpy and
  scipy. Install `scikit-build-core` and `"pybind11>=2.11"` first if they are missing.
- `CMAKE_CUDA_ARCHITECTURES` set to the compute capability of the visible GPU instead of
  upstream's `"all"` (override with `CUPDLPX_CUDA_ARCH`; 80 when the query fails).

The build needs network access: upstream's CMake fetches the PSLP presolver (tag v0.0.8)
and, when no system zlib is found, zlib v1.3. An offline build needs the CMake cache
variable `FETCHCONTENT_SOURCE_DIR_PSLP` set to a local PSLP checkout, which `build.sh` does
not pass.

## Refreshing `upstream/`

A checkout already contains `upstream/`; `fetch.sh` is needed only to re-fetch it. It clones
upstream at `PIN`, checks the SHA, removes `.git` and syncs the tree into `upstream/`
(keeping a local `upstream/build/`). Upstream ships its own `upstream/.gitignore`, whose
`*.txt`, `*.sh` and `test/*` patterns would hide its `CMakeLists.txt` files and its test suite
from `git add` and leave an unbuildable tree. `fetch.sh` therefore runs `git add -f upstream/`,
unstages `upstream/build/`, and fails if any upstream file is still ignored. It changes the
git index.

## Upstream behaviour the runner relies on

- `cupdlpx.Model(objective_vector, constraint_matrix, constraint_lower_bound,
  constraint_upper_bound, variable_lower_bound, variable_upper_bound)` accepts a SciPy CSR
  matrix.
- A missing variable lower bound defaults to −∞ (`fill_or_copy(..., var_lb, -INFINITY)` in
  `src/cupdlpx.c`), although `python/README.md` says it defaults to 0. The transport LP
  therefore passes an explicit zero lower bound.
- The binding converts the CSR indices to int32 without a range check
  (`python/cupdlpx/model.py`). The dense transport LP has `2·n_g⁴` nonzeros, which
  exceeds 2³¹ − 1 from `n_g = 256` on; the runner records such an instance as an `oom` row
  without calling the solver.
- Defaults (`set_default_parameters` in `src/utils.cu`): `eps_optimal_relative =
  eps_feasible_relative = 1e-4`, termination checked every 200 iterations, presolve on,
  verbose on. Parameters are changed with `Model.setParams(...)`.
- A CUDA allocation failure inside cuPDLPx exits the process. The runner therefore refuses an
  instance whose estimated device memory, about 36 bytes per nonzero plus 104 bytes per
  variable (counted from upstream's allocations), exceeds the device memory; at `n_g = 128`
  the estimate is about 47 GB. In the paper's runs (A100-SXM4-80GB, 93 GB of host memory)
  cuPDLPx ran out of host memory at `n_g = 128`, so it reaches `n_g = 64` on DOTmark.

## Use in the benchmark

Algorithm `CUPDLPX`, in the General rung (`src/protocol/regimes.py`), on the costs of the
General norm policy. `src/runners/run_cupdlpx.py` builds the vectorized transport LP
(`build_dot_lp`: equality rows `X1 = r`, `Xᵀ1 = c` from `src/instances/dot_lp.py`, `X ≥ 0`)
and solves it in-process. Under `--pareto` it keeps the library's default tolerances;
otherwise `spec.tolerance` sets `eps_optimal_relative` and `eps_feasible_relative`. The time
and iteration limits come from the spec, and the dual is read from `Model.Pi`.
