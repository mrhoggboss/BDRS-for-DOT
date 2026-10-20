# Loads the LAMP solver from the pinned upstream (mxburns2022/CuLAMP.jl, see PIN).
#
# `using CuLAMP` would also load PythonOT, PyCall, Plots and Images, which the
# solver does not use, and upstream ships no Manifest. So this file reads exact
# line ranges of three upstream sources, each range one or more whole top-level
# definitions, and evaluates them into the module `LampCore`. Every source file is
# sha256-checked first, so a changed upstream refuses to load instead of running
# under a stale line map. Nothing on disk is edited.
#
# Neither solver function returns all the state the runner needs: the dense
# `LAMP` returns the plan and theta but not nu, eta_t, the iteration count or the
# exit taken; the matrix-free `extragradient_color_transfer` returns nu and phi but
# not theta, eta_t or the count. Each is evaluated under a new name with
# exact-string patches that record which exit fired and how many iterations
# completed, and return that state; the iteration is untouched. The matrix-free
# variant also skips two colour-transfer outputs (barycentric maps, and
# psi = -2W_inf*nu/eta_p, which is Inf or NaN at eta_p = 0) that play no part in
# the transport solve. Every patch anchor must occur exactly once in its range, or
# loading fails.

module LampShim

using SHA

export load!

const PIN_FILE = joinpath(@__DIR__, "PIN")

const UPSTREAM_SHA256 = Dict(
    "src/Utilities.jl" =>
        "c440a2d8f3a7700f61009df7f8a14961a567d72a6ff9c4bc5c0817d86f06fdfd",
    "src/solvers/MirrorProx.jl" =>
        "66f817cf0068bef3dc4bc855adbb688d01774415968bc7ab1c47c056051a0aa0",
    "src/solvers/MirrorProxKernel.jl" =>
        "a4a4393a10108a4f3b0df826669e1bc687a0ef68499ff4d1a2836e85fe3ee7b1",
    "configurations/default_0.json" =>
        "a986c84362f122b641bc44c554a1bfbdf49edbba0f6610be6f111739bb227566",
)

# (file, first line, last line, label); each range is one or more whole
# top-level definitions at the PIN.
const RANGES = [
    ("src/Utilities.jl", 18, 31, "EOTArgs"),
    ("src/Utilities.jl", 34, 41, "logsumexp"),
    ("src/Utilities.jl", 93, 105, "softmax"),
    ("src/Utilities.jl", 136, 142, "neg_entropy"),
    ("src/Utilities.jl", 156, 163, "round"),
    ("src/solvers/MirrorProx.jl", 7, 32, "dualv_primalv"),
    ("src/solvers/MirrorProx.jl", 159, 180, "update_theta_residual"),
    ("src/solvers/MirrorProx.jl", 229, 235, "lse_pair_combine"),
    ("src/solvers/MirrorProx.jl", 237, 296, "warp_logsumexp_fused"),
    ("src/solvers/MirrorProx.jl", 299, 333, "residual_c"),
    ("src/solvers/MirrorProx.jl", 397, 503, "lamp_gpu"),
    ("src/solvers/MirrorProx.jl", 506, 575, "lamp_cpu"),
    ("src/solvers/MirrorProxKernel.jl", 7, 94, "residual_spp_c"),
    ("src/solvers/MirrorProxKernel.jl", 280, 326, "max_logsumexp_spp_ct"),
    ("src/solvers/MirrorProxKernel.jl", 702, 829, "warp_logsumexp_spp_ct_opt"),
    ("src/solvers/MirrorProxKernel.jl", 833, 938, "warp_logsumexp_spp_ct_fused"),
    ("src/solvers/MirrorProxKernel.jl", 941, 1016, "warp_min_reduce"),
    ("src/solvers/MirrorProxKernel.jl", 1019, 1140, "lamp_kernel"),
]

lines(xs...) = join(xs, "\n") * "\n"

# The tmax exit, identical in all three solver bodies at the PIN.
const TMAX_OLD = lines(
    "    for i in 1:args.itermax",
    "        elapsed_time = (time_ns() - time_start) / 1e9",
    "        if elapsed_time > args.tmax",
    "            break",
    "        end")
const TMAX_NEW = lines(
    "    lampcap_reason = \"itermax\"",
    "    lampcap_iters = args.itermax",
    "    for i in 1:args.itermax",
    "        elapsed_time = (time_ns() - time_start) / 1e9",
    "        if elapsed_time > args.tmax",
    "            lampcap_reason = \"tmax\"",
    "            lampcap_iters = i - 1",
    "            break",
    "        end")

# The epsilon exit: the check runs at the top of iteration i, before its update,
# so i - 1 iterations are complete when it fires.
eps_patch(test_line) = (
    lines(test_line, "                break", "            end"),
    lines(test_line,
          "                lampcap_reason = \"epsilon\"",
          "                lampcap_iters = i - 1",
          "                break",
          "            end"))

const PATCHES = Dict(
    "lamp_gpu" => [
        ("function LAMP(r::CuArray{R},", "function lamp_gpu_capture(r::CuArray{R},"),
        (TMAX_OLD, TMAX_NEW),
        eps_patch("            if feas < args.epsilon / 2 && primal_value - dual_value < args.epsilon / 2"),
        ("    return r .* p, θ\n\nend",
         "    return r .* p, θ, ν, ηt, W∞, lampcap_iters, lampcap_reason, (time_ns() - time_start) / 1e9\n\nend"),
    ],
    "lamp_cpu" => [
        ("function LAMP(r::AbstractArray{R},", "function lamp_cpu_capture(r::AbstractArray{R},"),
        (TMAX_OLD, TMAX_NEW),
        eps_patch("            if (pobj - dobj) < args.epsilon / 6 && feas < args.epsilon / 6"),
        ("    return r .* p, θ\nend",
         "    return r .* p, θ, ν, ηt, W∞, lampcap_iters, lampcap_reason, (time_ns() - time_start) / 1e9\nend"),
    ],
    "lamp_kernel" => [
        ("function extragradient_color_transfer(img1::CuArray{T},",
         "function lamp_kernel_capture(img1::CuArray{T},"),
        (TMAX_OLD, TMAX_NEW),
        eps_patch("            if primal_value - dual_value < args.epsilon / 6 && residual_value < args.epsilon / 6"),
        (lines("    output_img1 = CUDA.zeros(T, 3, N)", "    output_img2 = CUDA.zeros(T, 3, N)"),
         lines("    # [bdrs shim] color-transfer outputs (output_img1/2, psi) are not computed")),
        (lines("    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img1, img1, img2, ν, marginal1, sumvals, ηt, st, W∞, p)"), ""),
        (lines("    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct_t!(output_img2, img1, img2, marginal1, ν, sumvals, ηt, st, W∞, p)"), ""),
        (lines("    ψ = -2W∞ * ν ./ args.eta_p"), ""),
        ("    return Array(ν), Array(φ), Array(ψ), Array(output_img1), Array(output_img2)",
         "    CUDA.synchronize()\n    return Array(ν), Array(φ), Array(θ), ηt, st, W∞, W∞_scaling, lampcap_iters, lampcap_reason, (time_ns() - time_start) / 1e9"),
    ],
)

# The extracted upstream code lives here.
module LampCore
using CUDA, LinearAlgebra, Printf
end

const _STATE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)

count_occ(hay::AbstractString, needle::AbstractString) = length(findall(needle, hay))

"""
    load!(upstream_dir) -> Dict

Verify the fetched tree (PIN stamp and per-file sha256), extract the pinned line
ranges, apply the capture patches, and evaluate everything into `LampCore`.
Idempotent within a process.
"""
function load!(upstream_dir::AbstractString)
    _STATE[] === nothing || return _STATE[]
    pin = strip(read(PIN_FILE, String))
    stamp_path = joinpath(upstream_dir, ".bdrs_pin")
    isfile(stamp_path) ||
        error("LampShim: $(stamp_path) missing; run vendor/lamp/fetch.sh")
    stamp = strip(read(stamp_path, String))
    stamp == pin ||
        error("LampShim: upstream fetched at $(stamp) but PIN is $(pin); re-run vendor/lamp/fetch.sh")
    text = Dict{String,Vector{String}}()
    for (f, want) in UPSTREAM_SHA256
        bytes = read(joinpath(upstream_dir, f))
        got = bytes2hex(sha256(bytes))
        got == want ||
            error("LampShim: sha256 of upstream/$(f) is $(got), expected $(want) at the PIN")
        text[f] = split(String(bytes), "\n")
    end
    applied = Dict{String,Int}()
    for (f, a, b, label) in RANGES
        code = join(text[f][a:b], "\n")
        for (old, new) in get(PATCHES, label, Tuple{String,String}[])
            n = count_occ(code, old)
            n == 1 || error("LampShim: patch anchor in $(label) occurs $(n) times, expected 1: " *
                            repr(first(old, 120)))
            code = replace(code, old => new)
            applied[label] = get(applied, label, 0) + 1
        end
        Base.include_string(LampCore, code, "upstream/$(f):$(a)-$(b)")
    end
    for (label, pats) in PATCHES
        get(applied, label, 0) == length(pats) ||
            error("LampShim: $(label) applied $(get(applied, label, 0)) of $(length(pats)) patches")
    end
    _STATE[] = Dict{String,Any}("pin" => pin, "ranges" => length(RANGES),
                                "patches" => sum(values(applied)))
    return _STATE[]
end

end # module LampShim
