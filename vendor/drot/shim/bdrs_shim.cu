// Command-line wrapper around upstream DROT's step() loop. Upstream drot()
// frees the plan before returning, so the shim repeats drot()'s
// initialization, owns the plan buffer, and copies the plan back to the host.
//
// fp32 throughout: DROT's kernels are templated on T, but step() launches
// update_x_even_<float> / update_x_odd_<float> by name, so fp32 is the only
// precision without forking upstream. Matrices are column-major, as in DROT.
//
// Exit codes: 1 bad arguments, 2 CUDA allocation failure, 3 other CUDA
// error, 4 input blob unreadable, 5 output unwritable.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "drot.hpp"


// update_x_odd_ (src/kernels.hpp) computes the plan entry
//     xval = max(X + phi1 + phi2 - stepsize*C, 0)
// but stores the splitting variable
//     X = xval - stepsize*C,
// so after an odd-indexed step X is not the plan. Adding stepsize*C back
// recovers it, non-negative up to fp32 rounding because xval is.
// update_x_even_ stores max(X + phi1 + phi2, 0), the plan itself.

template<typename T>
__global__ void recover_feasible_from_splitting(
    T *X, const T *C, T stepsize, int nrows, int ncols)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= nrows || tidy >= ncols) return;
    int idx = tidx + tidy * nrows;
    X[idx] = X[idx] + stepsize * C[idx];
}


namespace {

struct ShimArgs {
    std::string cost_path;
    std::string r_path;
    std::string c_path;
    std::string plan_out_path;
    std::string info_out_path;
    // Outputs for DROT's multipliers φ₁ = dphi1, φ₂ = dphi2. The LP dual is
    // (α, β) = (φ₁/σ, φ₂/σ) with σ = stepsize; the runner divides. da2/db2
    // are row- and column-sum accumulators that update_auxs_* zeroes every
    // step, not duals.
    std::string alpha_out_path;
    std::string beta_out_path;
    int n         = 0;
    float tol     = 1e-4f;
    int max_iters = 1'000'000;
    int check_freq = 100;
    double budget_seconds = 0.0;
    bool emit_plan = true;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --cost C.bin --r r.bin --c c.bin --n N\n"
        "    [--tol 1e-4] [--max-iters 1000000] [--check-freq 100]\n"
        "    [--budget-seconds 0.0]\n"
        "    [--alpha-out alpha.bin] [--beta-out beta.bin]\n"
        "    [--no-plan] --plan-out X.bin --info-out info.json\n"
        "All binaries are float32, host-byte-order, contiguous.\n",
        prog);
}

bool parse_args(int argc, char** argv, ShimArgs& a) {
    auto need_val = [&](int i) -> const char* {
        if (i + 1 >= argc) return nullptr;
        return argv[i + 1];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--cost") { a.cost_path = need_val(i); ++i; }
        else if (k == "--r") { a.r_path = need_val(i); ++i; }
        else if (k == "--c") { a.c_path = need_val(i); ++i; }
        else if (k == "--n") { a.n = std::atoi(need_val(i)); ++i; }
        else if (k == "--tol") { a.tol = std::strtof(need_val(i), nullptr); ++i; }
        else if (k == "--max-iters") { a.max_iters = std::atoi(need_val(i)); ++i; }
        else if (k == "--check-freq") { a.check_freq = std::atoi(need_val(i)); ++i; }
        else if (k == "--budget-seconds") { a.budget_seconds = std::strtod(need_val(i), nullptr); ++i; }
        else if (k == "--alpha-out") { a.alpha_out_path = need_val(i); ++i; }
        else if (k == "--beta-out") { a.beta_out_path = need_val(i); ++i; }
        else if (k == "--no-plan") { a.emit_plan = false; }
        else if (k == "--plan-out") { a.plan_out_path = need_val(i); ++i; }
        else if (k == "--info-out") { a.info_out_path = need_val(i); ++i; }
        else if (k == "--help" || k == "-h") { usage(argv[0]); return false; }
        else { std::fprintf(stderr, "Unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.n <= 0 || a.cost_path.empty() || a.r_path.empty() || a.c_path.empty() ||
        a.info_out_path.empty()) {
        std::fprintf(stderr, "Missing required arguments.\n");
        usage(argv[0]);
        return false;
    }
    if (a.emit_plan && a.plan_out_path.empty()) {
        std::fprintf(stderr, "--plan-out required unless --no-plan.\n");
        return false;
    }
    return true;
}

bool read_binary_floats(const std::string& path, std::vector<float>& out, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for reading\n", path.c_str());
        return false;
    }
    out.resize(expected);
    f.read(reinterpret_cast<char*>(out.data()), expected * sizeof(float));
    if (!f || (static_cast<size_t>(f.gcount()) != expected * sizeof(float))) {
        std::fprintf(stderr, "Short read from %s: expected %zu floats\n",
                     path.c_str(), expected);
        return false;
    }
    return true;
}

bool write_binary_floats(const std::string& path, const float* data, size_t count) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    return static_cast<bool>(f);
}

#define CUDA_CHECK(stmt) do {                                                  \
    cudaError_t _e = (stmt);                                                   \
    if (_e == cudaErrorMemoryAllocation) {                                     \
        std::fprintf(stderr, "CUDA OOM at " #stmt ": %s\n",                    \
                     cudaGetErrorString(_e));                                  \
        std::exit(2);                                                          \
    }                                                                          \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error at " #stmt ": %s\n",                  \
                     cudaGetErrorString(_e));                                  \
        std::exit(3);                                                          \
    }                                                                          \
} while(0)

}  // namespace


int main(int argc, char** argv) {
    ShimArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    const int nrows = args.n;
    const int ncols = args.n;
    const size_t matsize = static_cast<size_t>(nrows) * ncols * sizeof(float);
    const size_t rowsize = static_cast<size_t>(nrows) * sizeof(float);
    const size_t colsize = static_cast<size_t>(ncols) * sizeof(float);

    std::vector<float> C_host;
    std::vector<float> p_host;
    std::vector<float> q_host;
    if (!read_binary_floats(args.cost_path, C_host, static_cast<size_t>(nrows) * ncols)) return 4;
    if (!read_binary_floats(args.r_path,    p_host, nrows)) return 4;
    if (!read_binary_floats(args.c_path,    q_host, ncols)) return 4;

    // Allocation and initialization as in upstream drot().
    float* dC = nullptr;
    float* dp = nullptr;
    float* dq = nullptr;
    float* dX = nullptr;
    float* da1 = nullptr; float* da2 = nullptr;
    float* db1 = nullptr; float* db2 = nullptr;
    float* dphi1 = nullptr; float* dphi2 = nullptr;
    float* dcommon = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&dC,      matsize));
    CUDA_CHECK(cudaMalloc((void**)&dp,      rowsize));
    CUDA_CHECK(cudaMalloc((void**)&dq,      colsize));
    CUDA_CHECK(cudaMemcpy(dC, C_host.data(), matsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp, p_host.data(), rowsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq, q_host.data(), colsize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&dX,      matsize));
    CUDA_CHECK(cudaMalloc((void**)&da1,     rowsize));
    CUDA_CHECK(cudaMalloc((void**)&da2,     rowsize));
    CUDA_CHECK(cudaMalloc((void**)&db1,     colsize));
    CUDA_CHECK(cudaMalloc((void**)&db2,     colsize));
    CUDA_CHECK(cudaMalloc((void**)&dphi1,   rowsize));
    CUDA_CHECK(cudaMalloc((void**)&dphi2,   colsize));
    CUDA_CHECK(cudaMalloc((void**)&dcommon, 16 * sizeof(float)));

    dim3 grid_auxs((std::max(nrows, ncols) + BLOCK_SIZE - 1) / BLOCK_SIZE, 2);
    dim3 block_auxs(BLOCK_SIZE);
    zeros<float><<<1024, 256>>>(dX, nrows * ncols);
    init_auxs<float><<<grid_auxs, block_auxs>>>(
        da1, db1, da2, db2, dphi1, dphi2, dcommon, nrows, ncols);

    dim3 grid_x((ncols + BLOCK_SIZE * WORK_SIZE - 1) / (BLOCK_SIZE * WORK_SIZE),
                (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_x(1, BLOCK_SIZE);
    const float stepsize = 2.0f / static_cast<float>(nrows + ncols);
    // Grid and block are swapped relative to step()'s update_x_* launches.
    // This is upstream drot()'s launch, kept so the initial iterate matches.
    init_x<float><<<block_x, grid_x>>>(dX, dC, dp, dq, stepsize, nrows, ncols);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_start = std::chrono::steady_clock::now();
    int iters_run = 0;
    int term_reason = 1;
    float loginfo[5] = {0, 0, 0, 0, 0};
    float res = 0.0f;
    float fval = 0.0f;

    int k = 0;
    while (k < args.max_iters) {
        step(dX, da1, db1, da2, db2,
             &dcommon[14], &dcommon[15], dphi1, dphi2,
             &dcommon[0], dC, dp, dq, stepsize, nrows, ncols, k);

        const bool do_check = ((k + 1) % args.check_freq == 0) || ((k + 1) == args.max_iters);
        if (do_check) {
            CUDA_CHECK(cudaMemcpy(loginfo, &dcommon[0], 5 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            // Upstream's residual: loginfo[0] = ‖X1 − r‖², loginfo[1] = ‖Xᵀ1 − c‖².
            res  = std::sqrt(loginfo[0] + loginfo[1]);
            fval = (k % 2 == 0) ? fval : loginfo[2];
            if (res <= args.tol) {
                term_reason = 0;
                ++k;
                iters_run = k;
                break;
            }
            // The cudaMemcpy above synchronizes, so the clock covers the
            // completed steps.
            if (args.budget_seconds > 0.0) {
                auto t_now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                if (elapsed >= args.budget_seconds) {
                    term_reason = 2;
                    ++k;
                    iters_run = k;
                    break;
                }
            }
        }
        ++k;
        iters_run = k;
    }

    auto t_end = std::chrono::steady_clock::now();
    const double solve_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    CUDA_CHECK(cudaMemcpy(loginfo, &dcommon[0], 5 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    res  = std::sqrt(loginfo[0] + loginfo[1]);
    if (iters_run % 2 != 0) fval = loginfo[2];

    auto t_rec0 = std::chrono::steady_clock::now();

    // The last step's index is iters_run - 1, so an even iters_run means the
    // last step was odd and X holds the splitting variable.
    bool recovered_from_splitting = false;
    if ((iters_run % 2) == 0 && iters_run > 0) {
        dim3 block(16, 16);
        dim3 grid((nrows + block.x - 1) / block.x,
                  (ncols + block.y - 1) / block.y);
        recover_feasible_from_splitting<float><<<grid, block>>>(
            dX, dC, stepsize, nrows, ncols);
        CUDA_CHECK(cudaDeviceSynchronize());
        recovered_from_splitting = true;
    }

    std::vector<float> X_host;
    if (args.emit_plan) {
        X_host.resize(static_cast<size_t>(nrows) * ncols);
        CUDA_CHECK(cudaMemcpy(X_host.data(), dX, matsize, cudaMemcpyDeviceToHost));
    }

    std::vector<float> alpha_host, beta_host;
    if (!args.alpha_out_path.empty()) {
        alpha_host.resize(static_cast<size_t>(nrows));
        CUDA_CHECK(cudaMemcpy(alpha_host.data(), dphi1, rowsize, cudaMemcpyDeviceToHost));
    }
    if (!args.beta_out_path.empty()) {
        beta_host.resize(static_cast<size_t>(ncols));
        CUDA_CHECK(cudaMemcpy(beta_host.data(), dphi2, colsize, cudaMemcpyDeviceToHost));
    }

    const double recovery_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_rec0).count();

    size_t mem_free = 0, mem_total = 0;
    cudaMemGetInfo(&mem_free, &mem_total);

    cudaFree(dC); cudaFree(dp); cudaFree(dq); cudaFree(dX);
    cudaFree(da1); cudaFree(da2); cudaFree(db1); cudaFree(db2);
    cudaFree(dphi1); cudaFree(dphi2); cudaFree(dcommon);

    auto t_wr0 = std::chrono::steady_clock::now();
    if (args.emit_plan && !write_binary_floats(args.plan_out_path, X_host.data(),
                                                static_cast<size_t>(nrows) * ncols)) {
        return 5;
    }
    if (!args.alpha_out_path.empty() &&
        !write_binary_floats(args.alpha_out_path, alpha_host.data(),
                              static_cast<size_t>(nrows))) {
        return 5;
    }
    if (!args.beta_out_path.empty() &&
        !write_binary_floats(args.beta_out_path, beta_host.data(),
                              static_cast<size_t>(ncols))) {
        return 5;
    }
    const double write_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wr0).count();

    {
        std::ofstream info(args.info_out_path);
        info << std::setprecision(17);
        if (!info) {
            std::fprintf(stderr, "Failed to open %s for writing\n",
                         args.info_out_path.c_str());
            return 5;
        }
        info << "{\n";
        info << "  \"iters\": " << iters_run << ",\n";
        info << "  \"max_iters\": " << args.max_iters << ",\n";
        info << "  \"check_freq\": " << args.check_freq << ",\n";
        info << "  \"residual\": " << res << ",\n";
        // Both terms are primal despite the key names: primal_res_sq is
        // ‖X1 − r‖² and dual_res_sq is ‖Xᵀ1 − c‖².
        info << "  \"primal_res_sq\": " << loginfo[0] << ",\n";
        info << "  \"dual_res_sq\": " << loginfo[1] << ",\n";
        info << "  \"objective\": " << fval << ",\n";
        info << "  \"loginfo3\": " << loginfo[3] << ",\n";
        info << "  \"loginfo4\": " << loginfo[4] << ",\n";
        info << "  \"stepsize\": " << stepsize << ",\n";
        info << "  \"tol\": " << args.tol << ",\n";
        info << "  \"solve_seconds\": " << solve_seconds << ",\n";
        info << "  \"recovery_seconds\": " << recovery_seconds << ",\n";
        info << "  \"write_seconds\": " << write_seconds << ",\n";
        const char* term_str = "max_iters";
        if (term_reason == 0)      term_str = "converged";
        else if (term_reason == 2) term_str = "budget_exceeded";
        info << "  \"term_reason\": \"" << term_str << "\",\n";
        info << "  \"budget_seconds\": " << args.budget_seconds << ",\n";
        info << "  \"alpha_out\": \"" << args.alpha_out_path << "\",\n";
        info << "  \"beta_out\": \"" << args.beta_out_path << "\",\n";
        info << "  \"alpha_size\": " << (args.alpha_out_path.empty() ? 0 : nrows) << ",\n";
        info << "  \"beta_size\": " << (args.beta_out_path.empty() ? 0 : ncols) << ",\n";
        info << "  \"gpu_mem_free_after_bytes\": " << mem_free << ",\n";
        info << "  \"gpu_mem_total_bytes\": " << mem_total << ",\n";
        info << "  \"recovered_from_splitting\": "
             << (recovered_from_splitting ? "true" : "false") << ",\n";
        info << "  \"plan_layout\": \"column_major\",\n";
        info << "  \"dtype\": \"fp32\"\n";
        info << "}\n";
    }

    return 0;
}
