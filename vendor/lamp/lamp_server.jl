# Persistent LAMP solve server. One process serves every solve of a job, so the
# Julia start-up, package load and CUDA kernel compilation that a fresh process
# would pay on every instance are paid once. It keeps compiled code only: before
# every solve it runs GC.gc(true) and CUDA.reclaim(), so no instance data, heap or
# device-pool state carries from one solve to the next. The Python client
# src/runners/lamp_client.py starts it; it exits after --idle-timeout seconds idle.
#
# Protocol: Unix-domain socket, one JSON request line, one JSON response line,
# then the connection closes. Requests carry the paths of raw little-endian
# float64 blobs written by the runner, and state comes back the same way. Logging
# goes to stderr (the server log); LAMP's own per-check rows, which it prints to
# stdout, are redirected per solve into the request's log_path.
#
#   julia --project=vendor/lamp vendor/lamp/lamp_server.jl --socket PATH \
#         [--idle-timeout S] [--upstream DIR] [--fingerprint HEX] [--warmup 0|1] [--device I]
#
# A failed warm-up exits the process, since a cold server would compile inside
# timed solves. So does a CUDA error other than an out-of-memory one, which can
# leave the context unusable: the client relaunches a dead server, not a live one.

using Sockets, JSON3, LinearAlgebra, Printf, Random
using CUDA

include(joinpath(@__DIR__, "lamp_shim.jl"))
using .LampShim
const LC = LampShim.LampCore

function parse_cli(argv)
    opts = Dict{String,String}("socket" => "", "idle-timeout" => "300",
                               "upstream" => joinpath(@__DIR__, "upstream"),
                               "fingerprint" => "", "warmup" => "1", "device" => "0")
    i = 1
    while i <= length(argv)
        k = argv[i]
        startswith(k, "--") && i < length(argv) || error("bad argument $(k)")
        opts[k[3:end]] = argv[i+1]
        i += 2
    end
    isempty(opts["socket"]) && error("--socket is required")
    return opts
end

logmsg(msg) = (println(stderr, "[lamp_server ", round(Int, time()), "] ", msg); flush(stderr))

function read_f64(path::AbstractString, n::Integer)
    filesize(path) == 8 * n || error("$(path): $(filesize(path)) bytes, expected $(8n)")
    v = Vector{Float64}(undef, n)
    open(io -> read!(io, v), path, "r")
    return v
end

write_f64(path::AbstractString, v) = open(io -> write(io, Vector{Float64}(v)), path, "w")

# Nested JSON objects may arrive String- or Symbol-keyed depending on JSON3's parse.
function getk(d, k::String, default = nothing)
    haskey(d, k) && return d[k]
    haskey(d, Symbol(k)) && return d[Symbol(k)]
    default === nothing && error("missing setting $(k)")
    return default
end

function make_args(s)
    LC.EOTArgs(; eta_p = Float64(getk(s, "eta_p")), eta_mu = Float64(getk(s, "eta_mu")),
               tau_p = Float64(getk(s, "tau_p")), tau_mu = Float64(getk(s, "tau_mu")),
               alpha = Float64(getk(s, "alpha")), B = Float64(getk(s, "B")),
               epsilon = Float64(getk(s, "epsilon")), anneal_mult = 0.95,
               itermax = Int(getk(s, "itermax")), inner_iter = Int(getk(s, "inner_iter", 10)),
               tmax = Float64(getk(s, "tmax")), verbose = Bool(getk(s, "verbose")))
end

parse_p(s::AbstractString) = s == "Inf" ? Inf : parse(Float64, s)

function free_device!()
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()
    return nothing
end

# GPU kernels compiled so far: CUDA.jl keeps one HostKernel per compiled kernel
# (src/compiler/execution.jl, pinned 5.11.3), so a change across a timed solve means
# JIT landed inside it. -1 if the internal is absent.
n_kernels() = isdefined(CUDA, :_kernel_instances) ? length(CUDA._kernel_instances) : -1

# Run f with LAMP's stdout rows captured into log_path; returns (result, seconds,
# jit), the clock bracketing exactly the solver call between two device syncs, and
# jit the host-side Julia compile seconds and new GPU kernels inside that window.
function timed_capture(f, log_path::AbstractString)
    result = nothing
    secs = 0.0
    k0 = 0; c0 = UInt64(0); k1 = 0; c1 = UInt64(0)
    open(log_path, "w") do io
        redirect_stdout(io) do
            CUDA.functional() && CUDA.synchronize()
            k0 = n_kernels(); c0 = Base.cumulative_compile_time_ns()[1]
            t0 = time_ns()
            result = f()
            CUDA.functional() && CUDA.synchronize()
            secs = (time_ns() - t0) / 1e9
            k1 = n_kernels(); c1 = Base.cumulative_compile_time_ns()[1]
        end
    end
    jit = Dict("julia_compile_s" => (c1 - c0) / 1e9,
               "gpu_kernels_compiled" => (k0 < 0 ? -1 : k1 - k0))
    return result, secs, jit
end

function solve(req)
    mode = String(req["mode"])
    m = Int(req["m"]); n = Int(req["n"])
    out_dir = String(req["out_dir"]); log_path = String(req["log_path"])
    freq = Int(req["frequency"])
    args = make_args(req["settings"])
    free_device!()
    r = read_f64(String(req["r_path"]), m)
    c = read_f64(String(req["c_path"]), n)
    resp = Dict{String,Any}("status" => "ok", "mode" => mode)

    if mode == "gpu_dense" || mode == "cpu_dense"
        W = reshape(read_f64(String(req["cost_path"]), m * n), m, n)   # column-major C
        if mode == "gpu_dense"
            CUDA.functional() || return Dict("status" => "error", "error_kind" => "no_cuda",
                                             "error_msg" => "CUDA.functional() is false")
            rd = CuArray(r); cd = CuArray(c); Wd = CuArray(W); W = nothing
            CUDA.synchronize()
            GC.gc(true)                       # the N^2 host blob, swept before the clock
            res, t_solve, jit = timed_capture(() -> LC.lamp_gpu_capture(rd, cd, Wd, args, freq), log_path)
            X, θ, ν, ηt, Winf, iters, reason, t_loop = res
            # Untimed cross-check on the plan LAMP itself returned.
            resp["xc_row_l1"] = Float64(sum(abs.(vec(sum(X, dims=2)) .- rd)))
            resp["xc_col_l1"] = Float64(sum(abs.(vec(sum(X, dims=1)) .- cd)))
            resp["xc_obj"] = Float64(dot(Wd, X))
            X = nothing; rd = nothing; cd = nothing; Wd = nothing
        else
            res, t_solve, jit = timed_capture(() -> LC.lamp_cpu_capture(r, c, W, args, freq), log_path)
            X, θ, ν, ηt, Winf, iters, reason, t_loop = res
            resp["xc_row_l1"] = Float64(sum(abs.(vec(sum(X, dims=2)) .- r)))
            resp["xc_col_l1"] = Float64(sum(abs.(vec(sum(X, dims=1)) .- c)))
            resp["xc_obj"] = Float64(dot(W, X))
            X = nothing; W = nothing
        end
        write_f64(joinpath(out_dir, "theta.f64"), Array(θ))
        write_f64(joinpath(out_dir, "nu.f64"), Array(ν))
        merge!(resp, Dict("st" => 1.0, "W_inf" => Float64(Winf), "W_inf_scaling" => 1.0))
    elseif mode == "kernel"
        CUDA.functional() || return Dict("status" => "error", "error_kind" => "no_cuda",
                                         "error_msg" => "the matrix-free LAMP path is CUDA-only")
        P = reshape(read_f64(String(req["P_path"]), 3 * m), 3, m)
        Q = reshape(read_f64(String(req["Q_path"]), 3 * n), 3, n)
        p = parse_p(String(req["p"]))
        Pd = CuArray(P); Qd = CuArray(Q); rd = CuArray(r); cd = CuArray(c)   # fresh: the kernel rescales in place
        CUDA.synchronize()
        GC.gc(true)
        res, t_solve, jit = timed_capture(() -> LC.lamp_kernel_capture(Pd, Qd, rd, cd, args, freq, p), log_path)
        ν, φ, θ, ηt, st, Winf, Winfs, iters, reason, t_loop = res
        write_f64(joinpath(out_dir, "theta.f64"), θ)
        write_f64(joinpath(out_dir, "nu.f64"), ν)
        write_f64(joinpath(out_dir, "phi.f64"), φ)
        merge!(resp, Dict("st" => Float64(st), "W_inf" => Float64(Winf),
                          "W_inf_scaling" => Float64(Winfs)))
        Pd = nothing; Qd = nothing; rd = nothing; cd = nothing
    else
        return Dict("status" => "error", "error_kind" => "bad_mode", "error_msg" => mode)
    end
    merge!(resp, Dict("t_solve" => t_solve, "t_loop" => Float64(t_loop),
                      "iters" => Int(iters), "reason" => String(reason),
                      "eta_t" => Float64(ηt), "jit_in_solve" => jit))
    free_device!()
    return resp
end

function oom_kind(e)
    e isa OutOfMemoryError && return true
    (CUDA.functional() && e isa CUDA.OutOfGPUMemoryError) && return true
    occursin("out of memory", lowercase(sprint(showerror, e)))
end

const WARM = Ref(false)
const EXIT_AFTER_REPLY = Ref(false)

function handle(req, fingerprint)
    op = String(get(req, "op", "solve"))
    op == "ping" && return Dict("status" => "pong", "fingerprint" => fingerprint,
                                "has_gpu" => CUDA.functional(), "warm" => WARM[],
                                "device" => (CUDA.functional() ? CUDA.deviceid(CUDA.device()) : -1),
                                "pid" => getpid())
    op == "solve" || return Dict("status" => "error", "error_kind" => "bad_op", "error_msg" => op)
    try
        return solve(req)
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        oom = oom_kind(e)
        # A non-OOM CUDA error may have left the context unusable: answer, then exit
        # so the next runner relaunches instead of meeting a broken live server.
        if !oom && e isa CUDA.CuError
            EXIT_AFTER_REPLY[] = true
        end
        try
            free_device!()
        catch e2
            EXIT_AFTER_REPLY[] = true
            msg *= "\nfree_device! failed: " * sprint(showerror, e2)
        end
        return Dict("status" => oom ? "oom" : "error",
                    "error_kind" => oom ? "cuda_oom_solve" : string(typeof(e)),
                    "error_msg" => first(msg, 4000))
    end
end

# Generic solves (seeded random marginals and coordinates, never instance data)
# compile every kernel and solver body before the server reports ready. Kernel
# argument types do not depend on N or p (p is a runtime Float64), but CUDA.jl's
# mapreduce picks its kernel from the input length: one block up to 1024, a
# multi-block kernel plus a second stage above it, and a serial kernel once the
# non-reduced length reaches 1024 x the SM count. So the warm-up covers every
# case: N = 1024 and 4096 on both GPU paths, and a cloud just past the serial
# threshold on the kernel path (the only path that reaches it: N^2 dense costs stop
# at N = 16384). A few iterations with a check at each suffice at the large size.
function warm_args(itermax, tmax)
    LC.EOTArgs(; eta_p = 0.0, eta_mu = 0.0, tau_p = 1.0, tau_mu = 1.0, alpha = 0.01,
               B = 1.1, epsilon = 1e-10, anneal_mult = 0.95, itermax = itermax,
               inner_iter = 10, tmax = tmax, verbose = true)
end

function warm_grid(ng)
    N = ng * ng
    rows = [(k - 1) ÷ ng for k in 1:N]; cols = [(k - 1) % ng for k in 1:N]
    W = [(abs(rows[a] - rows[b]) + abs(cols[a] - cols[b])) / (2 * (ng - 1)) for a in 1:N, b in 1:N]
    P = zeros(3, N); P[1, :] .= rows ./ (2 * (ng - 1)); P[2, :] .= cols ./ (2 * (ng - 1))
    r = rand(N) .+ 0.1; r ./= sum(r)
    c = rand(N) .+ 0.1; c ./= sum(c)
    return W, P, r, c
end

function warmup()
    Random.seed!(20260427)
    gpu = CUDA.functional()
    log = tempname()
    try
        # The CPU dense path is warmed on every node: a GPU node can still be asked
        # for cpu_dense (a spec with device "cpu"), and an unwarmed CPU path
        # compiles inside each such timed solve.
        W, _, r, c = warm_grid(8)
        for _ in 1:2
            timed_capture(() -> LC.lamp_cpu_capture(r, c, W, warm_args(401, 120.0), 200), log)
        end
        gpu || return nothing
        for ng in (32, 64), _ in 1:2
            W, P, r, c = warm_grid(ng)
            timed_capture(() -> LC.lamp_gpu_capture(CuArray(r), CuArray(c), CuArray(W),
                                                    warm_args(401, 120.0), 200), log)
            timed_capture(() -> LC.lamp_kernel_capture(CuArray(P), CuArray(copy(P)), CuArray(r),
                                                       CuArray(c), warm_args(401, 120.0), 200, 1.0), log)
            W = nothing
            free_device!()
        end
        # CUDA.jl's own threshold (max threads per block x SMs), then one block past it.
        Nbig = CUDA.serial_mapreduce_threshold(CUDA.device()) + 1024
        P = zeros(3, Nbig); P[1:2, :] .= rand(2, Nbig)
        r = rand(Nbig) .+ 0.1; r ./= sum(r)
        c = rand(Nbig) .+ 0.1; c ./= sum(c)
        for _ in 1:2
            timed_capture(() -> LC.lamp_kernel_capture(CuArray(P), CuArray(copy(P)), CuArray(r),
                                                       CuArray(c), warm_args(3, 600.0), 1, 1.0), log)
        end
        logmsg("warmup sizes: CPU dense n_g 8; grid n_g 32, 64 (GPU dense and kernel); kernel cloud N=$(Nbig); " *
               "kernels cached=$(n_kernels())")
    finally
        rm(log; force = true)
        free_device!()
    end
    return nothing
end

function serve(opts)
    sock = opts["socket"]
    idle = parse(Float64, opts["idle-timeout"])
    fp = opts["fingerprint"]
    ispath(sock) && rm(sock)
    server = listen(sock)
    last_seen = Ref(time())
    busy = Ref(false)
    @async while true
        sleep(5)
        if !busy[] && time() - last_seen[] > idle
            logmsg("idle for $(round(Int, time() - last_seen[])) s; exiting")
            try close(server) catch end
            rm(sock; force = true)
            exit(0)
        end
    end
    logmsg("READY socket=$(sock) gpu=$(CUDA.functional()) fingerprint=$(fp)")
    # One write per response; close() flushes it. The client closes as soon as it
    # has the newline, so nothing after the write is allowed to fail the request.
    send_line(conn, d) = write(conn, JSON3.write(d; allow_inf = true) * "\n")
    while true
        conn = try accept(server) catch; break end
        busy[] = true
        responded = false
        try
            line = readline(conn)
            req = JSON3.read(line, Dict{String,Any})
            op = String(get(req, "op", "solve"))
            if op == "shutdown"
                send_line(conn, Dict("status" => "bye"))
                try close(conn) catch end
                logmsg("shutdown requested")
                rm(sock; force = true)
                exit(0)
            end
            op == "solve" && logmsg("solve $(get(req, "instance_id", "?")) mode=$(get(req, "mode", "?"))")
            resp = handle(req, fp)
            send_line(conn, resp)
            responded = true
            op == "solve" && logmsg("done status=$(get(resp, "status", "?")) " *
                                    "reason=$(get(resp, "reason", "-")) iters=$(get(resp, "iters", "-"))")
            if EXIT_AFTER_REPLY[]
                try close(conn) catch end
                logmsg("exiting after a CUDA error that may have left the context unusable")
                rm(sock; force = true)
                exit(1)
            end
        catch e
            if !responded
                logmsg("request failed: " * first(sprint(showerror, e), 2000))
                try
                    send_line(conn, Dict("status" => "error", "error_kind" => string(typeof(e)),
                                         "error_msg" => first(sprint(showerror, e), 4000)))
                catch end
            end
        finally
            try close(conn) catch end
            busy[] = false
            last_seen[] = time()
        end
    end
end

function main(opts, info)
    logmsg("shim loaded: pin=$(info["pin"]) ranges=$(info["ranges"]) patches=$(info["patches"])")
    Base.cumulative_compile_timing(true)
    CUDA.functional() && CUDA.device!(parse(Int, opts["device"]))
    if opts["warmup"] == "1"
        try
            warmup()
            WARM[] = true
            logmsg("warmup done")
        catch e
            # Fatal: a cold server would put JIT inside every timed solve.
            logmsg("warmup FAILED; exiting: " * first(sprint(showerror, e, catch_backtrace()), 4000))
            exit(1)
        end
    end
    serve(opts)
end

# Load the shim at TOP LEVEL, before main() runs: it evaluates the upstream methods
# into LampCore, and Julia's world-age rule hides methods defined while a function
# is already executing from that function. Calling main() afterwards, as a fresh
# top-level statement, runs in the newer world where they are visible.
const OPTS = parse_cli(ARGS)
const SHIM_INFO = LampShim.load!(OPTS["upstream"])
main(OPTS, SHIM_INFO)
