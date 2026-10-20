# LAMP (fetched at a pinned commit)

Upstream: https://github.com/mxburns2022/CuLAMP.jl, the authors' Julia/CUDA.jl
implementation of LAMP, "Log-Averaged Mirror Prox for Fast, Large-Scale Optimal Transport
in Linear Space" (Burns and Liang, arXiv 2511.11359). Pinned commit (`PIN`):
`f3d32f56572dabd3e534040dab37742fcb18687e`.

Upstream's default branch has moved past the pin: its later commit `d966d306` changes all
three source files the shim reads (`src/Utilities.jl`, `src/solvers/MirrorProx.jl`,
`src/solvers/MirrorProxKernel.jl`). The pin is kept, because the shim's line ranges, patch
anchors and sha256 values are those of the pinned files, and the paper's LAMP results were
computed at it.

## Fetching

The upstream tree is not committed here. `fetch.sh`
clones it at `PIN`, checks the SHA and syncs `src/`, `configurations/`, `Project.toml` and
`README.md` into `upstream/` (gitignored); the rest of the tree is figure data. It writes
`upstream/.bdrs_pin` last, and `build.sh`, the shim and the runner refuse an `upstream/`
whose stamp differs from `PIN`. Every committed file in this directory is first-party:

| File | Contents |
|---|---|
| `PIN` | the upstream commit |
| `fetch.sh` | the clone and sync described above |
| `Project.toml`, `Manifest.toml` | the Julia environment of the shim and server (CUDA.jl, JSON3 and standard libraries) |
| `build.sh` | instantiates and precompiles that environment, then loads the shim as a check; fails if instantiation rewrote `Manifest.toml` |
| `lamp_shim.jl` | evaluates the solver's source lines from `upstream/` |
| `lamp_server.jl` | the persistent solve server the runner talks to |

## Build

```bash
bash vendor/lamp/fetch.sh     # needs network
bash vendor/lamp/build.sh     # needs Julia 1.11
```

Both scripts and the runner run `julia`, or the command in `BDRS_LAMP_JULIA` (for example
`"julia +1.11"`) when `julia` is another version. `Manifest.toml` is resolved under Julia
1.11 with the CUDA.jl stack of `vendor/PDOT.jl` (CUDA.jl 5.11.3, which does not support
Julia 1.12).

## The shim

`using CuLAMP` would load PythonOT, PyCall, Plots and Images, which the solver never uses,
and upstream ships no Manifest and no `[compat]` bounds. `lamp_shim.jl` instead reads exact
line ranges of three pinned sources (`src/Utilities.jl`, `src/solvers/MirrorProx.jl`,
`src/solvers/MirrorProxKernel.jl`), each range one or more whole top-level definitions,
after checking each file's sha256 against its value at the pin, and evaluates them into a
module. Nothing on disk is edited.

The two solver functions are evaluated under new names with exact-string patches that
record which exit fired (`epsilon`, `tmax` or `itermax`) and how many iterations
completed, and return that state together with ν, θ and η_t, which upstream does not return
from both paths. The iteration is untouched. The matrix-free variant also skips two
colour-transfer outputs (barycentric maps, and ψ, which is infinite or NaN at η_p = 0).
Each patch anchor must match exactly once, or loading fails.

## The server

A fresh Julia process pays start-up, the CUDA.jl load and kernel compilation on every
call, tens of seconds per instance. `lamp_server.jl` compiles once per job and serves
solves over a Unix socket. Its warm-up solves seeded synthetic problems, never instance
data, at every size for which CUDA.jl's `mapreduce` compiles a different kernel: N = 1024
and 4096 on both GPU paths, and a cloud just past CUDA.jl's serial threshold (maximum
threads per block times the SM count) on the matrix-free path. A failed warm-up exits the
server. Each timed solve reports the host-side Julia compile seconds and the number of new
GPU kernels inside its clock, so every row shows whether compilation reached the timing.
Before every solve the server runs `GC.gc(true)` and `CUDA.reclaim()`, so no instance data,
heap or device-pool state carries from one solve to the next. A CUDA error other than an
out-of-memory one makes it answer and exit, since the context may be unusable.

The client, `src/runners/lamp_client.py`, starts the server before the solve's time budget
is computed, so start-up is on no clock. Later runners reuse it; a server whose code
fingerprint (shim, server, `PIN`, `Project.toml`, `Manifest.toml`) differs is restarted.
The socket is keyed on the Slurm job and step, `CUDA_VISIBLE_DEVICES`, the device index and
the checkout. The server exits after `BDRS_LAMP_IDLE_TIMEOUT` seconds idle (default 300).
It holds a CUDA context while idle, and is left running an orphaned solve when a runner is
killed, so the orchestrator runs `LAMP` and `LAMPPointCloud` only in a job of their own;
`python -m src.runners.lamp_client shutdown` stops it.

## Upstream facts the runner relies on (at the pin)

- Dense GPU entry `LAMP(r::CuArray, c::CuArray, W::CuArray, args, frequency)`
  (`MirrorProx.jl:397`): returns the unrounded plan `r .* p` and θ. Its stopping test is
  ‖Xᵀ1 − c‖₁ < ε/2 and primal − dual < ε/2; the CPU method tests both at ε/6.
- Matrix-free entry `extragradient_color_transfer(img1, img2, r, c, args, frequency, p)`
  (`MirrorProxKernel.jl:1019`): the coordinates are 3 × N; the distance is Σ|d|, Σd² or
  max|d| for p = 1, 2, ∞ (no unsquared ℓ2); the instance must be square; the coordinate
  arrays are rescaled in place. Its stopping test (at ε/6) runs only when `verbose` is true.
  For p = ∞ it never rescales, and the test compares W_∞⟨d, X⟩ + 2W_∞‖Xᵀ1 − c‖₁ with a dual
  bound on the unscaled d: with W_∞ > 1 it cannot fire, and with W_∞ < 1 it fires on the
  column residual alone. Dividing the coordinates by the exact C_max, as the runner does,
  makes W_∞ = 1.
- The plan on both paths is log X_ij = (log r_i − log Z_i) − (st·d_ij/(2W_∞) + ν_j)/η_t,
  with st = 1 and, at η_p = 0, η_t = 1/K after K iterations. d is the cost LAMP sees: C on
  the dense paths; on the matrix-free path with p ≠ ∞, C divided by `W∞_scaling`, because
  upstream divides the coordinates by `W∞_scaling`^(1/p) and resets W_∞ to 1. The runner
  therefore uses W = W_∞·`W∞_scaling` for both the plan and the dual.
- Settings: `configurations/default_0.json` (η_p = η_μ = 0, ε = 10⁻¹⁰, 10⁶ iterations,
  t_max = 600 s), and a check every 100 iterations, the command line's default.
- Marginals: the DOTmark entry point `run_dot` (`src/CmdLineUtils.jl:176`) solves with
  `normalize(r .+ 1e-6, 1)` and likewise for c. The runner does the same on every instance
  and both paths; it also keeps log r finite, since a zero in r makes the dense stopping
  test NaN.
