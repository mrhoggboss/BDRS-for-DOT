# Vendored baselines

The vendored GPU baselines of the benchmark, each at a pinned upstream commit. DROT, PDOT.jl, HOT and
cuPDLPx are committed as copies of their upstream trees with `.git` removed. HALO, M3S and
LAMP are not committed: `halo/fetch.sh`, `m3s/fetch.sh` and `lamp/fetch.sh` clone them at their
pin into the gitignored `upstream/` directories. `m3s/shim/m3s_shim.cu` contains modified copies
of two functions of M3S's `SparseSinkhorn/solver.cu` (see `m3s/README.md`).

Upstream files are not edited, with one exception: PDOT.jl's `Project.toml` adds a
`[compat]` pin `CUDA = "=5.11.3"`, its `Manifest.toml` is resolved under Julia 1.11, and its
`README.md` ends with a "Local pin" section that explains why. PDOT's source (`src/*.jl`,
`scripts/solve.jl`) is upstream's, byte for byte.

## Pinned commits

| Directory | Upstream | Pinned commit | Commit date and subject | In this repository |
|---|---|---|---|---|
| `drot/` | https://github.com/vienmai/drot | `a5ffa08d6911c7245984d3b0e5835f90767801f4` | 2021-10-04, "set eps" | committed |
| `PDOT.jl/` | https://github.com/MIT-Lu-Lab/PDOT.jl | `7e1d762854dda732c78c02b66fe6a300a469486e` | 2024-08-01, "Update README" | committed |
| `HOT/` | https://github.com/PolyU-IOR/HOT | `4b0de44ed28622ece1ff63371828f0081cbb1580` | 2025-07-04, "add star history" | committed |
| `cupdlpx/upstream/` | https://github.com/MIT-Lu-Lab/cuPDLPx | `931c94c0c3767d38b3df71514c84ccd314aa2ac7` (tag v0.2.9) | 2026-03-31, "fix: objective constant bug with presolve (#90)" | committed |
| `halo/upstream/` | https://github.com/WenzhouXia/HALO | `1671ae690597deea1f17e3928d5ac282f8760d23` | 2026-04-24, "simplify the codebase" | fetched by `halo/fetch.sh` |
| `m3s/upstream/` | https://github.com/ucascnic/CudaOT | `ec100cb67e445d400cb61064e6eb6924debb5d80` | 2023-11-12, "fix bugs and update python examples" | fetched by `m3s/fetch.sh` |
| `lamp/upstream/` | https://github.com/mxburns2022/CuLAMP.jl | `f3d32f56572dabd3e534040dab37742fcb18687e` | 2026-05-07, "final update" | fetched by `lamp/fetch.sh` (a subset) |

Dates are the committers' local dates. The `PIN` files of `cupdlpx/`, `halo/`, `m3s/` and
`lamp/` hold the SHA their `fetch.sh` checks out; each script fails if the checked-out SHA
differs.

LAMP's upstream has moved past the pin: its later commit `d966d306` changes all three source
files that `lamp/lamp_shim.jl` reads. The pin is kept, because the shim's line ranges, patch
anchors and sha256 checks are those of the pinned files, and the paper's LAMP results were
computed at it.

## First-party files

- `drot/shim/bdrs_shim.cu`, `drot/shim/Makefile`: a command-line wrapper around DROT's
  `step()` iteration. It reads the cost (column-major) and the marginals from fp32 binary
  blobs, runs DROT's initialization and `step()` loop on a plan buffer it allocates itself,
  and writes the plan (`--plan-out`, fp32, column-major), DROT's multipliers φ₁, φ₂ on
  request (`--alpha-out`, `--beta-out`) and `info.json` (iterations, residuals, objective,
  step size σ, clocks). It stops when DROT's residual is at most `--tol` (default `1e-4`),
  checked every `--check-freq` iterations (default 100), or at `--max-iters`. `--help` lists
  the other options, which the runner does not use. Exit codes: 2 on a CUDA allocation
  failure, 3 on another CUDA error, 4 when an input blob cannot be read, 5 when an output
  cannot be written, 1 on bad arguments.
- `PDOT.jl/bench_driver.jl`: a long-lived Julia process. It reads one instance spec per line
  of JSON on stdin and writes one result record per line on stdout; it signals readiness by
  writing `READY` to stderr after two warm-up solves on a 32 × 32 dummy. It calls
  `PDOT.optimize` directly, with the restart, step-size and termination parameters of
  upstream's `scripts/solve.jl` (`eps_optimal_absolute = eps_optimal_relative` = the spec's
  tolerance; verbosity 0; time and iteration limits from the spec), so PDOT's `rounding` of
  the plan, which only `scripts/solve.jl` calls, never runs. Before each measured solve it
  runs `GC.gc(true)` and `CUDA.reclaim()`, then a 100-iteration warm-up on the instance, as
  upstream's `warm_up` does.
- `halo/{PIN,fetch.sh,build.sh,README.md}`: fetch, repair and build HALO (see
  `halo/README.md`).
- `m3s/{PIN,fetch.sh,build.sh,README.md}` and `m3s/shim/{m3s_shim.cu,CMakeLists.txt,cuda12_compat.cuh}`:
  fetch M3S and build an extraction shim that returns the plan and the duals; the shim
  contains modified copies of two functions of upstream's `solver.cu` (see `m3s/README.md`).
- `cupdlpx/{PIN,fetch.sh,build.sh,README.md}`: refresh the committed tree and install the
  official Python binding (see `cupdlpx/README.md`).
- `lamp/{PIN,fetch.sh,build.sh,README.md,Project.toml,Manifest.toml,lamp_shim.jl,lamp_server.jl}`:
  fetch LAMP, its Julia environment (CUDA.jl 5.11.3, as PDOT's), a loader that evaluates
  sha256-checked line ranges of the pinned sources with patches that return the solver's
  exit and state (no file on disk is edited), and a persistent solve server (see
  `lamp/README.md`).

## Building

```bash
make -C vendor/drot/shim          # DROT shim, for the visible GPU (NVCC_ARCH="-gencode arch=compute_90,code=sm_90" overrides)
julia --project=vendor/PDOT.jl -e 'import Pkg; Pkg.instantiate()'   # Julia 1.11
bash vendor/cupdlpx/build.sh      # pip-installs the cupdlpx binding
bash vendor/halo/fetch.sh && bash vendor/halo/build.sh
bash vendor/m3s/fetch.sh && bash vendor/m3s/build.sh       # GCC host compiler (CXX, CUDAHOSTCXX)
bash vendor/lamp/fetch.sh && bash vendor/lamp/build.sh   # Julia 1.11
```

HOT needs no build. `scripts/build_all_baselines.sh` runs all of these steps. PDOT and LAMP run on CUDA.jl's own CUDA libraries, so their Julia processes need the CUDA toolkit's `lib64` off `LD_LIBRARY_PATH` (the top-level README gives a one-line way).

## Notes on each baseline

### DROT (`drot/`)

- Header-only CUDA C++ (`src/*.hpp`). The kernels run in fp32 only: `step()` in
  `src/drot.hpp` is templated on `T` but launches `update_x_even_<float>` and
  `update_x_odd_<float>`. The runner records `algo_extras.drot_dtype = "fp32"`.
- Upstream's `drot()` allocates the plan, frees it before returning and returns only the
  objective, and its example programs hard-code the problem size. The shim therefore repeats
  `drot()`'s initialization and calls `step()` itself, with sizes and limits from the
  command line.
- After an odd-indexed step, DROT's plan buffer holds `X − σC` rather than the plan `X`
  (`update_x_odd_` in `src/kernels.hpp`). The shim adds `σC` back before writing the plan and
  reports `recovered_from_splitting` in `info.json`.
- The dual of the transport LP is `(α, β) = (φ₁/σ, φ₂/σ)` with `σ = 2/(m + n)`; the runner
  divides.
- Upstream's `Makefile` builds the examples for `sm_35`, which CUDA 12 and later no longer
  support; the benchmark uses only `shim/Makefile`.

### PDOT.jl (`PDOT.jl/`)

- Install with `julia --project=vendor/PDOT.jl -e 'import Pkg; Pkg.instantiate()'` under
  Julia 1.11, the version `Manifest.toml` is resolved under.
- Upstream's `scripts/solve.jl` does not run as shipped: `solve_instance_and_output` uses
  `output_dir` and `time_sec_limit`, and `warm_up` uses `tolerance` and
  `warmup_termination_params`, none of which is defined in its scope. `bench_driver.jl`
  builds `PdhgParameters` and the termination criteria itself and calls `PDOT.optimize`,
  which returns the unrounded plan.

### LAMP (`lamp/`)

- Julia on CUDA.jl, in its own environment (`lamp/Project.toml`, `lamp/Manifest.toml`,
  Julia 1.11). `using CuLAMP` would load PythonOT, PyCall, Plots and Images, and upstream
  ships no Manifest, so `lamp/lamp_shim.jl` loads only the solver's own definitions from
  `upstream/`.
- The runner sends each solve to `lamp/lamp_server.jl`, one process per job, which compiles
  once and frees the device and the heap before every solve (see `lamp/README.md`).

### HOT (`HOT/`)

- Pure Python on PyTorch. `HOT_Solver.py` imports only `math`, `torch` and `tqdm`, and the
  repository's `requirements.txt` provides the last two. `HOT/requirements.txt` is upstream's
  and pins `torch==2.6.0`, `numpy==2.3.1` and other packages; installing it would replace the
  benchmark's torch.
- `HOTSolver.optimize()` returns its log only when constructed with `logging=True`; the log
  has `iter`, `kkt_error`, `pr_obj`, `dual_obj` and `gap`, and `compute_feaserr()` adds
  `pr_feaserr`, which is measured on HOT's reduced LP, not on the transport plan.
- `HOTSolver.recover_transport_plan()` builds a dense `(mn, mn)` plan
  (`torch.zeros((self.m, self.m, self.n, self.n))`): in fp64, 2.1 GB at `n_g = 128`, 34 GB
  at 256 and 550 GB at 512. It also rescales and clips `solver.x` in place, so it may be
  called at most once, after the solve. The runner never calls it: at every `n_g`,
  `src/scoring/hot_plan.py` evaluates in O(n_g³) the row sums, column sums and ⟨C, X⟩ of the
  plan it would return (the same scaling, clamp and north-west-corner rule), reading
  `solver.x` without changing it. The dual-stall study (`src/studies/dual_stall.py`, `n_g` 32
  and 64) calls it once per run, after reading the duals. `recover_transport_map_gpu_sparse()`
  returns a sparse plan and is not used.
