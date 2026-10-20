# bench_driver.jl
#
# Long-lived Julia driver of the PDOT baseline. It reads one InstanceSpec per
# line from stdin, calls PDOT.optimize directly, scores the returned
# (X, α, β) in Julia and writes one ResultRecord per spec. Upstream's
# scripts/solve.jl cannot be run as shipped: it references undefined names
# (`tolerance`, `warmup_termination_params`, `output_dir`, `time_sec_limit`,
# `GZip`, `rounding`). PDOT's src/ is upstream's, unedited. The plan is scored
# as PDOT.optimize returns it; scripts/solve.jl also writes a copy rounded
# onto the marginals (`rounding`), which this driver never computes.
#
# Wire protocol:
#   stdin:  one InstanceSpec JSON object per line.
#   stdout: one ResultRecord JSON object per spec, one line each, in order.
#           PDOT.optimize also prints its iteration-table heading to stdout at
#           every verbosity (solver_gpu.jl), so the reader
#           (src/runners/run_pdot.py) skips lines that do not start with '{'.
#   stderr: log messages, and "READY" once the two startup warm-up solves
#           are done.
#
# Run:
#   julia --project=vendor/PDOT.jl vendor/PDOT.jl/bench_driver.jl

using Dates
using JSON3
using Printf
using Random
using UUIDs

# Bound explicitly, as PDOT's own module does (src/PDOT.jl), so a bare `dot`
# or `norm` here is LinearAlgebra's and never a name from `using CUDA`.
import LinearAlgebra
const dot  = LinearAlgebra.dot
const norm = LinearAlgebra.norm

import PDOT
using CUDA

# PDOT calls `CUDA.dot` and `CUDA.norm` (saddle_point_gpu.jl, solver_gpu.jl,
# utils_gpu.jl). A CUDA.jl that does not provide them gets LinearAlgebra's,
# so PDOT runs without patching its source. The pinned CUDA.jl 5.11.3 provides
# both (it does `using LinearAlgebra`), so these lines do nothing under the pin.
if !isdefined(CUDA, :dot)
    @eval CUDA const dot = $LinearAlgebra.dot
end
if !isdefined(CUDA, :norm)
    @eval CUDA const norm = $LinearAlgebra.norm
end

# --------------------------------------------------------------------------
# .npy reader (format versions 1 to 3, fp64 little-endian), so the project
# needs no NPZ.jl. A C-order array comes back correctly only when it is 1-D
# or square (every array read here is); a non-square one comes back with the
# wrong shape and order.
# --------------------------------------------------------------------------

function read_npy_fp64(path::AbstractString)
    open(path) do io
        magic = read(io, 6)
        @assert magic == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] "Not a .npy file: $path"
        v_major = read(io, UInt8)
        _v_minor = read(io, UInt8)
        if v_major == 0x01
            header_len = Int(ltoh(read(io, UInt16)))
        elseif v_major in (0x02, 0x03)
            header_len = Int(ltoh(read(io, UInt32)))
        else
            error("Unsupported NPY major version $(v_major) in $(path)")
        end
        header = String(read(io, header_len))

        m_descr = match(r"'descr'\s*:\s*'([^']+)'", header)
        m_descr === nothing && error("Missing descr in NPY header: $header")
        descr = m_descr.captures[1]
        @assert descr == "<f8" "Only fp64 little-endian supported, got '$descr' in $(path)"

        m_fort = match(r"'fortran_order'\s*:\s*(True|False)", header)
        fortran = m_fort !== nothing && m_fort.captures[1] == "True"

        m_shape = match(r"'shape'\s*:\s*\(([^)]*)\)", header)
        m_shape === nothing && error("Missing shape in NPY header: $header")
        dims = Int[]
        for piece in split(m_shape.captures[1], ",")
            s = strip(String(piece))
            isempty(s) && continue
            push!(dims, parse(Int, s))
        end
        n = prod(dims)
        data = Vector{Float64}(undef, n)
        read!(io, data)
        if length(dims) <= 1
            return data
        end
        arr = reshape(data, Tuple(dims))
        return fortran ? arr : permutedims(arr, reverse(1:length(dims)))
    end
end

# --------------------------------------------------------------------------
# The max-1 lattice cost, for a lattice spec whose cost file does not exist:
# src/instances/dotmark_build.py writes none for `inf`, and none at a
# resolution built with --skip-costs.
# --------------------------------------------------------------------------

function build_cost_matrix_on_grid(n::Int, norm_str::AbstractString)
    nn = n * n
    if norm_str == "sql2"
        c_max = 2.0 * (n - 1)^2
    elseif norm_str == "l1"
        c_max = 2.0 * (n - 1)
    elseif norm_str == "l2"
        c_max = sqrt(2.0) * (n - 1)
    elseif norm_str == "inf"
        c_max = Float64(n - 1)
    else
        error("Unknown norm: $(norm_str)")
    end
    C = Matrix{Float64}(undef, nn, nn)
    @inbounds for j in 0:nn-1
        bj_r, bj_c = divrem(j, n)
        for i in 0:nn-1
            ai_r, ai_c = divrem(i, n)
            dr = ai_r - bj_r
            dc = ai_c - bj_c
            v = norm_str == "sql2" ? Float64(dr * dr + dc * dc) :
                norm_str == "l2" ? sqrt(Float64(dr * dr + dc * dc)) :
                norm_str == "inf" ? Float64(max(abs(dr), abs(dc))) :
                Float64(abs(dr) + abs(dc))
            C[i + 1, j + 1] = v / c_max
        end
    end
    return C
end

# --------------------------------------------------------------------------
# The marginals reach PDOT as read from disk, neither floored nor
# renormalised: an exact zero (DOTmark's Shapes and MicroscopyImages have
# many) is a valid right-hand side of the LP, and PDOT never takes a log of
# the marginals. Every metric in the record is scored against these vectors.
# --------------------------------------------------------------------------

const MARGINAL_FLOOR = 0.0

# --------------------------------------------------------------------------
# Clocks. One PDOT.optimize call runs cached_ot_info (host norms of C, r and
# c), ot_cpu_to_gpu (the H2D copy of the dense nn x nn cost, 2.1 GB at
# n_g = 128), the device state allocation, the PDHG iterations with their
# termination checks, gpu_to_cpu! (the D2H copy of the returned plan and both
# dual vectors) and pdhg_final_log. The driver cannot split the call without
# editing upstream, so t_solve is the whole call, and T_SOLVE_INCLUDES lists
# the transfers, allocation and logging inside it. PDOT starts its own clock
# after the H2D copy and the state allocation (`start_time` in solver_gpu.jl)
# and stores the elapsed time as `cumulative_time_sec` at every recorded
# check. The value at the terminating check is recorded as
# t_solver_internal_s, and the rest of t_solve as t_solve_unseparated_s;
# t_solver_internal_s is the time upstream's scripts/solve.jl reports as
# `solve_time_sec`.
const T_SOLVE_BOUNDARY  = "pdot_optimize_call"
const T_SOLVE_INCLUDES  = ["cost_h2d", "device_state_alloc", "plan_dual_d2h",
                           "final_log"]
const T_SOLVE_EXCLUDES  = ["julia_process_startup", "startup_warmup",
                           "gc_reclaim_reset", "npy_marginal_read",
                           "cost_matrix_load_or_build",
                           "per_instance_warm_up"]
const T_TOTAL_BOUNDARY  = "pdot_optimize_call_returns_triple"
# t_total = t_solve: PDOT.optimize returns the host (X, α, β) itself, so
# there is no separate recovery.
const T_TOTAL_INCLUDES  = ["cost_h2d"]

"""
PDOT's own post-H2D clock at the terminating check (seconds), or `nothing`
when no iteration stats were recorded. This is the one component of the
optimize call the driver can separate without touching upstream.
"""
function solver_internal_seconds(output)
    stats = output.iteration_stats
    isempty(stats) && return nothing
    return Float64(stats[end].cumulative_time_sec)
end

# GPU peak memory, read from NVML: PDOT allocates through CUDA.jl, which
# torch's allocator statistics cannot see. NVML reports device-wide used
# bytes (this process's CUDA context and the CUDA.jl pool included), so the
# peak is the solve's own only when no other process holds memory on the
# device. The poller is an `@async` task, which runs only when the solving
# task yields, so samples are at least 50 ms apart and a peak between two
# samples is missed. The label `nvml_poll` is that of
# src/runners/nvml_sampler.py.
mutable struct MemPoll
    peak::Int
    stop::Bool
    task::Union{Task,Nothing}
    ok::Bool
end

function _mem_poll_start(interval_s::Float64 = 0.05)
    p = MemPoll(0, false, nothing, false)
    try
        dev = CUDA.NVML.Device(CUDA.uuid(CUDA.device()))
        # Probe once here: a failure inside the task would go unseen.
        CUDA.NVML.memory_info(dev)
        p.ok = true
        p.task = @async begin
            while !p.stop
                try
                    used = Int(CUDA.NVML.memory_info(dev).used)
                    if used > p.peak
                        p.peak = used
                    end
                catch
                    # A transient NVML read failure must not kill the poller.
                end
                sleep(interval_s)
            end
        end
    catch e
        @warn "bench_driver.jl: NVML peak-memory poller unavailable" exception=e
    end
    return p
end

# The peak in bytes, or `nothing` (not 0) when NVML was unavailable or took
# no sample.
function _mem_poll_stop(p::MemPoll)
    p.stop = true
    if p.task !== nothing
        try
            wait(p.task)
        catch
        end
    end
    return (p.ok && p.peak > 0) ? p.peak : nothing
end

# The parameters of upstream's scripts/solve.jl, except verbosity (0 here;
# upstream uses 2 for the measured solve) and the time and iteration limits,
# which are arguments. With eps_optimal_absolute = eps_optimal_relative = tol,
# PDOT declares OPTIMAL when, in the L2 norm, the primal residual is below
# tol (1 + ‖(r; c)‖), the dual residual below tol (1 + ‖C‖_F) and
# |p_obj - d_obj| below tol (1 + |p_obj| + |d_obj|) (termination.jl).
function pdot_parameters(tolerance::Float64, time_limit_s::Real, iter_cap::Integer)
    restart = PDOT.construct_restart_parameters(
        PDOT.ADAPTIVE_KKT,
        PDOT.KKT_GREEDY,
        1000,           # restart_frequency_if_fixed
        0.36,           # artificial_restart_threshold
        0.1,            # sufficient_reduction_for_restart
        0.9,            # necessary_reduction_for_restart
        0.5,            # primal_weight_update_smoothing
    )
    term = PDOT.construct_termination_criteria(
        optimality_norm = PDOT.L_2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        time_sec_limit = Float64(time_limit_s),
        iteration_limit = Int32(min(iter_cap, typemax(Int32))),
        kkt_matrix_pass_limit = Inf,
    )
    return PDOT.PdhgParameters(
        1.0,                            # primal_importance
        false,                          # scale_invariant_initial_primal_weight
        0,                              # verbosity
        true,                           # record_iteration_stats
        128,                            # termination_evaluation_frequency
        term,
        restart,
        PDOT.AdaptiveStepsizeParams(2, 1),
    )
end

function _now_iso()
    return Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS.sssZ")
end

function _failure_record(spec, status, kind, msg, started_utc, run_uid)
    return Dict(
        "schema_version" => "v1",
        "instance_id" => spec["instance_id"],
        "algorithm" => "PDOT",
        "regime" => "General",
        "norm" => spec["norm"],
        "resolution" => spec["resolution"],
        "n" => dense_side(spec),
        "run_uid" => run_uid,
        "started_utc" => started_utc,
        "ended_utc" => _now_iso(),
        "status" => status,
        "oom_flag" => status == "oom",
        "timeout_flag" => false,
        "error_kind" => kind,
        "error_msg" => first(msg, 4096),
        "wall_time_s" => nothing,
        "wall_time_solve_s" => nothing,
        "wall_time_recovery_s" => nothing,
        "iters" => nothing,
        "iter_cap_hit" => false,
        "native_termination" => nothing,
        "tolerance" => spec["tolerance"],
        "feaserr" => nothing,
        "native_duality_gap" => nothing,
        "objective" => nothing,
        "peak_gpu_mem_bytes" => nothing,
        "peak_gpu_mem_source" => nothing,
        "device_name" => nothing,
        "device_uuid" => nothing,
        "algo_extras" => Dict{String, Any}(),
    )
end

"""
    dense_side(spec) -> Int

Marginal length of the instance: n_g^2 for a lattice instance, else the
explicit row count `m`. A procedural (random_dense) instance has no lattice,
so `resolution` is null and the shape arrives in `m`/`n_cols`; it always
ships a materialised cost file, so nothing here reconstructs a grid cost.
"""
function dense_side(spec)
    res = get(spec, "resolution", nothing)
    if res !== nothing && !isnothing(res)
        return Int(res)^2
    end
    m = get(spec, "m", nothing)
    (m === nothing || isnothing(m)) &&
        error("spec has neither a grid resolution nor m: cannot size the instance")
    return Int(m)
end

function solve_one(spec)
    started_utc = _now_iso()
    run_uid = string(UUIDs.uuid4())
    nn = dense_side(spec)

    # Start each instance from the state of a fresh process: a collected Julia
    # heap and an empty CUDA.jl pool. The process stays alive only to pay the
    # Julia/CUDA JIT once; GC and pool state carried over from earlier
    # instances otherwise slows the measured solve. GC.gc(true) runs the
    # finalizers of dead CuArrays, so that CUDA.reclaim() can return their
    # blocks.
    GC.gc(true)
    CUDA.reclaim()

    r = read_npy_fp64(spec["marginal_i_path"])
    c = read_npy_fp64(spec["marginal_j_path"])
    @assert length(r) == nn && length(c) == nn

    C = nothing
    cost_path = get(spec, "cost_matrix_path", nothing)
    try
        if cost_path !== nothing && !isnothing(cost_path) && cost_path != "" && isfile(String(cost_path))
            C = read_npy_fp64(String(cost_path))
            if size(C) != (nn, nn)
                C = reshape(C, nn, nn)
            end
        else
            # Only a lattice cost can be rebuilt; any other cost arrives as a
            # file (a procedural cost is written to one before dispatch, by
            # src/instances/dense_cost.materialized_cost_path).
            res = get(spec, "resolution", nothing)
            (res === nothing || isnothing(res)) && error(
                "no cached cost matrix and no grid resolution: nothing to " *
                "reconstruct a dense cost from")
            C = build_cost_matrix_on_grid(Int(res), String(spec["norm"]))
        end
    catch e
        if e isa OutOfMemoryError
            return _failure_record(spec, "oom", "cpu_oom_cost_matrix", repr(e), started_utc, run_uid)
        end
        rethrow(e)
    end

    problem = PDOT.OptimalTransportProblem(C, r, c)
    params = pdot_parameters(
        Float64(spec["tolerance"]),
        Float64(spec["time_limit_s"]),
        Int(spec["iter_cap"]),
    )

    # Upstream's per-instance warm-up (`warm_up` in scripts/solve.jl, same
    # parameters): 100 iterations on this instance, result discarded, before
    # the measured solve. It moves this instance's first-use costs, such as
    # regrowing the CUDA.jl pool emptied by the reset above, out of the timed
    # window. A failed warm-up is not fatal.
    warmup_params = pdot_parameters(Float64(spec["tolerance"]), Inf, 100)
    try
        CUDA.synchronize()
        PDOT.optimize(warmup_params, problem)
        CUDA.synchronize()
    catch e
        @warn "bench_driver.jl: per-instance warm_up failed; proceeding" exception=e
    end

    output = nothing
    wall_solve = NaN
    # Brackets the timed solve only, but NVML also counts the memory the
    # CUDA.jl pool kept from the warm-up.
    mem_poll = _mem_poll_start()
    peak_gpu_mem = nothing
    try
        CUDA.synchronize()
        t0 = time_ns()
        output = PDOT.optimize(params, problem)
        CUDA.synchronize()
        t1 = time_ns()
        wall_solve = (t1 - t0) / 1e9
    catch e
        peak_gpu_mem = _mem_poll_stop(mem_poll)
        if e isa OutOfMemoryError
            return _failure_record(spec, "oom", "cuda_oom_solve", repr(e), started_utc, run_uid)
        end
        if isa(e, CUDA.OutOfGPUMemoryError)
            return _failure_record(spec, "oom", "cuda_oom_solve", repr(e), started_utc, run_uid)
        end
        return _failure_record(spec, "error", string(typeof(e)), sprint(showerror, e), started_utc, run_uid)
    end
    peak_gpu_mem = _mem_poll_stop(mem_poll)

    # Host arrays: X is nn x nn, α and β have length nn.
    X = output.primal_solution
    α = output.dual_source_solution
    β = output.dual_target_solution

    t_internal = solver_internal_seconds(output)
    t_unseparated = t_internal === nothing ? nothing : (wall_solve - t_internal)

    t2 = time_ns()
    row_sums = vec(sum(X, dims=2))
    col_sums = vec(sum(X, dims=1))
    err_r = norm(row_sums .- r)
    err_c = norm(col_sums .- c)
    # r_P of the returned plan; the negative part is summed without an
    # nn x nn temporary.
    neg = sqrt(sum(x -> x < 0.0 ? x * x : 0.0, X))
    r_P = err_r + err_c + neg
    objective = dot(vec(C), vec(X))   # no nn x nn temporary
    dual_obj = dot(r, α) + dot(c, β)
    native_gap = abs(objective - dual_obj)
    t3 = time_ns()
    wall_eval = (t3 - t2) / 1e9

    # The KKT components, computed here where X and the dense C are resident.
    # PDOT's duals are the potentials of the original LP (its dual objective
    # is rᵀα + cᵀβ, utils_gpu.jl), so they are scored as returned. The loop
    # gets r_Dinf and r_DF in one column-major pass with no nn x nn temporary.
    # t_eval times the negative-part norm and that loop.
    paretoblk = nothing
    if get(spec, "pareto", false) === true
        fin(x) = (isa(x, AbstractFloat) && !isfinite(x)) ? nothing : x
        tol_used = Float64(spec["tolerance"])
        tp0 = time_ns()
        rP_neg = norm(min.(X, 0.0))
        maxviol = -Inf
        frosq = 0.0
        @inbounds for j in 1:size(C, 2)
            bj = β[j]
            for i in 1:size(C, 1)
                v = α[i] + bj - C[i, j]
                if v > maxviol
                    maxviol = v
                end
                if v > 0.0
                    frosq += v * v
                end
            end
        end
        rDinf = max(0.0, maxviol)
        tp1 = time_ns()
        paretoblk = Dict{String, Any}(
            "r_P" => fin(err_r + err_c + rP_neg),
            "r_P_row" => fin(err_r), "r_P_col" => fin(err_c), "r_P_neg" => fin(rP_neg),
            "r_Dinf" => fin(rDinf), "r_DF" => fin(sqrt(frosq)),
            "p_obj" => fin(objective), "d_obj" => fin(dual_obj),
            "gap" => fin(objective - dual_obj),
            "C_inf" => fin(maximum(abs.(C))),
            "t_solve" => wall_solve, "t_total" => wall_solve,
            "t_eval" => (tp1 - tp0) / 1e9,
            "t_solve_boundary" => T_SOLVE_BOUNDARY,
            "t_solve_includes" => T_SOLVE_INCLUDES,
            "t_solve_excludes" => T_SOLVE_EXCLUDES,
            "t_total_boundary" => T_TOTAL_BOUNDARY,
            "t_total_includes" => T_TOTAL_INCLUDES,
            "t_solver_internal_s" => t_internal,
            "t_solve_unseparated_s" => t_unseparated,
            "pareto_partial" => false, "precision" => "fp64",
            # Labelled from the tolerance the solve ran with. 1e-4 is the
            # default of upstream's scripts/solve.jl and of the orchestrator's
            # --tolerance; --pareto does not force it.
            "native_stop" => "pdot_eps_opt_abs_rel_" * @sprintf("%g", tol_used),
            "native_stop_is_upstream_default" => (tol_used == 1.0e-4),
        )
    end

    iters_run = Int(output.iteration_count)
    iter_cap_hit = iters_run >= Int(spec["iter_cap"])
    term_string = String(output.termination_string)

    # feaserr is PDOT's own primal-feasibility measure
    #   relative_l2_primal_residual
    #     = ‖(r - X1; c - Xᵀ1)‖₂ / (eps_abs/eps_rel + ‖(r; c)‖₂),
    # which with eps_abs = eps_rel is the quantity PDOT's L2 primal test
    # compares with tol. It is read from the check PDOT stopped at, which
    # measures the returned plan, except after a numerical error, when that
    # check measures the current iterate; the same formula is then applied to
    # the returned plan.
    tc = params.termination_criteria
    eps_ratio = tc.eps_optimal_absolute / tc.eps_optimal_relative
    feaserr_recomputed = norm(vcat(r .- row_sums, c .- col_sums)) /
                         (eps_ratio + norm(vcat(r, c)))
    stats = output.iteration_stats
    ci_end = (isempty(stats) || isempty(stats[end].convergence_information)) ?
             nothing : stats[end].convergence_information[1]
    if ci_end !== nothing && !occursin("NUMERICAL", uppercase(term_string))
        feaserr = Float64(ci_end.relative_l2_primal_residual)
        feaserr_source = "pdot_iteration_stats"
    else
        feaserr = feaserr_recomputed
        feaserr_source = "recomputed_on_returned_plan"
    end
    # A stop on PDOT's own time limit (tested against its internal clock at
    # each check) did not reach the native stop, so the row is a timeout, not
    # a Pareto point: its time is the limit and its KKT residual is wherever
    # the iteration was.
    timed_out = occursin("TIME", uppercase(term_string))
    # An iteration-capped run stays `ok` and is scored: the cap is a
    # reproducible truncation point, so its time and KKT residual measure that
    # point, whereas a time-limit stop falls at an instant that moves with
    # machine load. `iter_cap_hit` marks such rows.

    _extras = Dict{String, Any}(
        "row_sum_err_l2" => err_r,
        "col_sum_err_l2" => err_c,
        "wall_time_eval_s" => wall_eval,
        "feaserr_measure" => "pdot_relative_l2_primal_residual",
        "feaserr_source" => feaserr_source,
        "feaserr_recomputed" => feaserr_recomputed,
        "r_P" => r_P,
        "r_P_negpart_assumed_zero" => false,
        "marginal_floor" => MARGINAL_FLOOR,
        "pdot_rounding_disabled_via" => "called PDOT.optimize directly, never invoked rounding()",
        "pdot_termination_reason" => term_string,
        "t_solve_boundary" => T_SOLVE_BOUNDARY,
        "t_solve_includes" => T_SOLVE_INCLUDES,
        "t_solve_excludes" => T_SOLVE_EXCLUDES,
        "t_total_boundary" => T_TOTAL_BOUNDARY,
        "t_total_includes" => T_TOTAL_INCLUDES,
        "t_solver_internal_s" => t_internal,
        "t_solve_unseparated_s" => t_unseparated,
    )
    if paretoblk !== nothing
        _extras["pareto"] = paretoblk
    end

    return Dict(
        "schema_version" => "v1",
        "instance_id" => spec["instance_id"],
        "algorithm" => "PDOT",
        "regime" => "General",
        "norm" => spec["norm"],
        "resolution" => get(spec, "resolution", nothing),
        "n" => nn,
        "run_uid" => run_uid,
        "started_utc" => started_utc,
        "ended_utc" => _now_iso(),
        "status" => timed_out ? "timeout" : "ok",
        "oom_flag" => false,
        "timeout_flag" => timed_out,
        "error_kind" => nothing,
        "error_msg" => nothing,
        "wall_time_s" => wall_solve + wall_eval,
        "wall_time_solve_s" => wall_solve,
        "wall_time_recovery_s" => nothing,
        "iters" => iters_run,
        "iter_cap_hit" => iter_cap_hit,
        "native_termination" => term_string,
        "tolerance" => Float64(spec["tolerance"]),
        "feaserr" => feaserr,
        "native_duality_gap" => native_gap,
        "objective" => objective,
        "peak_gpu_mem_bytes" => peak_gpu_mem,
        "peak_gpu_mem_source" => peak_gpu_mem === nothing ? "unavailable" : "nvml_poll",
        "device_name" => nothing,
        "device_uuid" => nothing,
        "algo_extras" => _extras,
    )
end

# --------------------------------------------------------------------------
# Startup warm-up: two solves of a random 32 x 32 sql2 instance, which
# compile PDOT's methods and CUDA.jl's kernels before READY.
# --------------------------------------------------------------------------

function warmup_pdot()
    n = 32
    nn = n * n
    Random.seed!(20260427)
    r = abs.(randn(nn)) .+ 1e-3
    c = abs.(randn(nn)) .+ 1e-3
    r ./= sum(r)
    c ./= sum(c)
    C = build_cost_matrix_on_grid(n, "sql2")
    problem = PDOT.OptimalTransportProblem(C, r, c)
    params = pdot_parameters(1.0e-4, 60.0, 1000)
    for _ in 1:2
        try
            CUDA.synchronize()
            PDOT.optimize(params, problem)
            CUDA.synchronize()
        catch e
            @warn "Warmup solve failed; proceeding anyway" exception=e
        end
    end
end

function main()
    @info "bench_driver.jl: warming up..."
    flush(stdout); flush(stderr)
    warmup_pdot()
    @info "bench_driver.jl: warmup complete; entering spec loop."
    println(stderr, "READY"); flush(stderr)

    for line in eachline(stdin)
        line = strip(line)
        isempty(line) && continue
        spec = nothing
        rec = nothing
        try
            spec = JSON3.read(String(line), Dict{String, Any})
            rec = solve_one(spec)
        catch e
            kind = string(typeof(e))
            msg = sprint(showerror, e)
            spec_id = spec === nothing ? "unknown" : get(spec, "instance_id", "unknown")
            rec = Dict(
                "schema_version" => "v1",
                "instance_id" => spec_id,
                "algorithm" => "PDOT",
                "regime" => "General",
                "status" => "error",
                "error_kind" => kind,
                "error_msg" => first(msg, 4096),
                "started_utc" => _now_iso(),
                "ended_utc" => _now_iso(),
                "run_uid" => string(UUIDs.uuid4()),
                "norm" => spec === nothing ? nothing : get(spec, "norm", nothing),
                "resolution" => spec === nothing ? nothing : get(spec, "resolution", nothing),
                "n" => spec === nothing ? nothing : dense_side(spec),
                "tolerance" => spec === nothing ? nothing : get(spec, "tolerance", nothing),
                "feaserr" => nothing,
                "native_duality_gap" => nothing,
                "objective" => nothing,
                "wall_time_s" => nothing,
                "iters" => nothing,
                "algo_extras" => Dict{String, Any}(),
            )
        end
        println(stdout, JSON3.write(rec; allow_inf=true)); flush(stdout)
        # Release the pool again once the record is out: this process lives
        # for the whole orchestrator run, and the other solvers run as
        # subprocesses on the same device, so memory held here between PDOT
        # solves is memory they cannot allocate. Outside every clock.
        GC.gc(true)
        CUDA.reclaim()
    end
end

main()
