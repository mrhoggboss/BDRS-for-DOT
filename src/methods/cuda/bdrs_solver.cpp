// Host side of the BDRS CUDA extension and its pybind11 module. Only the
// launchers of bdrs_kernels.cu are declared here (extern "C"), so this file
// compiles with the host C++ compiler.
//
// A session iterates the factored state
//     log X^k_ij = ξ_i + k·L_ij + ζ_j,    L = −C/η,
// with ξ (`A`) and ζ (`B`) vectors. Its constructor captures one iteration as
// a CUDA graph whose last node increments k in device memory, so step(n) is n
// replays that take no arguments and no host-to-device copy.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <memory>
#include <vector>

namespace py = pybind11;


extern "C" {
    void launch_advance(
        cudaStream_t stream,
        int *d_k, double *kL, const double *base_ell,
        int ng);

    void launch_fused_u(
        cudaStream_t stream,
        double *A, double *log_u,
        const double *beta_tilde, const double *kL, const double *log_r,
        int ng);

    void launch_fused_v(
        cudaStream_t stream,
        double *B, double *log_v, double *beta_tilde_next,
        const double *A, const double *kL, const double *log_c,
        int ng);

    void launch_materialize_X(
        cudaStream_t stream,
        double *X, const double *A, const double *B, const double *kL,
        int ng);

    void launch_ctransform_separable(
        cudaStream_t stream,
        double *q, const double *p, const double *ax,
        int ng);

    void launch_dense_row_lse_scaled(
        cudaStream_t stream,
        double *out, const double *C, const double *w, double neg_scale,
        int m, int n);

    void launch_dense_col_lse_scaled(
        cudaStream_t stream,
        double *out, double *scratch_m, double *scratch_s,
        const double *C, const double *w, double neg_scale,
        int m, int n, int rows_per_chunk, int n_chunks);

    void launch_pointcloud_advance(cudaStream_t stream, int *d_k);

    void launch_pointcloud_fused_u(
        cudaStream_t stream,
        double *A, double *log_u, const double *beta_tilde,
        const double *P, const double *Q, const double *log_r,
        const int *d_k, double eta, double c_max,
        int cost_type, int d, int m, int n);

    void launch_pointcloud_fused_v(
        cudaStream_t stream,
        double *B, double *log_v, double *beta_tilde,
        const double *A, const double *P, const double *Q, const double *log_c,
        const int *d_k, double eta, double c_max,
        int cost_type, int d, int m, int n);

    void launch_dense_factored_u(
        cudaStream_t stream,
        double *A, double *log_u, const double *beta_tilde,
        const double *C, const double *log_r, const int *d_k,
        double eta, int m, int n);

    void launch_dense_factored_v_stage1(
        cudaStream_t stream,
        double *scratch_m, double *scratch_s,
        const double *C, const double *A, const int *d_k,
        double eta, int m, int n, int rows_per_chunk, int n_chunks);

    void launch_dense_factored_v_stage2(
        cudaStream_t stream,
        double *B, double *log_v, double *beta_tilde,
        const double *scratch_m, const double *scratch_s, const double *log_c,
        int n, int n_chunks);

}


#define CUDA_OK(stmt) do { \
    cudaError_t _err = (stmt); \
    TORCH_CHECK(_err == cudaSuccess, \
        "CUDA error at " #stmt ": ", cudaGetErrorString(_err)); \
} while (0)


// ---- Grid cost table -------------------------------------------------------
// ℓ[δ] = −φ(δ)/(C_max·η) (`base_ell`), with C_max = 2φ(n_g − 1) the largest
// grid cost, so L_ij = ℓ[|Δrow|] + ℓ[|Δcol|] = −C_ij/η for the cost normalized
// to max C = 1. The same table as the torch grid_cost._base_ell.

static std::vector<double> build_base_ell(int ng, int cost_type, double eta) {
    std::vector<double> v(ng);
    if (cost_type == 0) {  // L1
        double c_max = 2.0 * static_cast<double>(ng - 1);
        for (int d = 0; d < ng; ++d) {
            v[d] = -static_cast<double>(d) / (c_max * eta);
        }
    } else if (cost_type == 1) {  // squared L2
        double c_max = 2.0 * static_cast<double>(ng - 1) * static_cast<double>(ng - 1);
        for (int d = 0; d < ng; ++d) {
            double dd = static_cast<double>(d) * static_cast<double>(d);
            v[d] = -dd / (c_max * eta);
        }
    } else {
        TORCH_CHECK(false, "Unknown cost_type: ", cost_type,
                    " (expected 0=l1 or 1=sql2)");
    }
    return v;
}


// ---- Grid session ----------------------------------------------------------
// One iteration is fused_u (ξ, log u) → fused_v (ζ, log v, the next β̃) →
// advance (k, kL = k·ℓ).

class BdrsSession {
public:
    BdrsSession(
        torch::Tensor r,            // positive: the floored r̃
        torch::Tensor c,            // positive: the floored c̃
        int64_t ng,
        double eta,
        int64_t cost_type,          // 0=l1, 1=sql2
        torch::Tensor init_A,       // (ξ^0, ζ^0, log v^0), each [N] fp64 CUDA,
        torch::Tensor init_B,       //   or empty for the uniform start
        torch::Tensor init_log_v
    )
        : ng_(ng), N_(ng * ng), eta_(eta),
          cost_type_(static_cast<int>(cost_type)),
          iters_abs_(0),
          stream_(at::cuda::getStreamFromPool(/*is_high_priority=*/false,
                                              r.device().index()))
    {
        TORCH_CHECK(r.is_cuda() && c.is_cuda(), "r, c must be CUDA tensors");
        TORCH_CHECK(r.dtype() == torch::kFloat64 && c.dtype() == torch::kFloat64,
                    "r, c must be float64");
        TORCH_CHECK(r.numel() == N_ && c.numel() == N_,
                    "marginal length must be ng*ng=", N_);
        // The fused kernels exist only for n_g = 4, 8, ..., 1024; for any
        // other n_g the launchers would silently launch nothing.
        TORCH_CHECK(ng_ >= 4 && ng_ <= 1024 && (ng_ & (ng_ - 1)) == 0,
                    "ng must be a power of 2 in [4, 1024]; got ", ng_);

        device_ = r.device();
        at::cuda::CUDAGuard device_guard(device_);

        r = r.contiguous();
        c = c.contiguous();

        // Setup runs on the current stream; the graph is captured on stream_,
        // a pool stream, because the current stream is normally the NULL
        // stream (cudaStreamLegacy), on which capture cannot begin.
        auto setup_stream = at::cuda::getCurrentCUDAStream();

        auto f64_opts = torch::TensorOptions().dtype(torch::kFloat64).device(device_);
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

        // The init tensors give the k = 0 state: the first replay always runs
        // k = 1, so they cannot resume a solve. ξ^0 enters only
        // log u^1 = ξ^1 − ξ^0.
        double logN = std::log(static_cast<double>(N_));
        auto _init = [&](const torch::Tensor& src, double default_val) {
            if (src.defined() && src.numel() == N_) {
                TORCH_CHECK(src.is_cuda(),                  "init tensor must be CUDA");
                TORCH_CHECK(src.dtype() == torch::kFloat64, "init tensor must be fp64");
                return src.contiguous().clone();   // the kernels update the state in place
            }
            TORCH_CHECK(!src.defined() || src.numel() == 0,
                        "init tensor must have numel = 0 (default-init) or N=ng*ng; got ",
                        src.numel());
            return torch::full({N_}, default_val, f64_opts);
        };
        A_     = _init(init_A,     -logN);
        B_     = _init(init_B,     -logN);
        log_v_ = _init(init_log_v, 0.0);

        // Zero, not empty: with K_max = 0 the delivered gap reads log u
        // before any step has written it.
        log_u_ = torch::zeros({N_}, f64_opts);

        // β̃^0 = ζ^0 + log v^0; from then on fused_v writes β̃ for the next
        // iteration.
        beta_tilde_ = B_ + log_v_;

        log_r_ = torch::log(r);
        log_c_ = torch::log(c);

        // base_ell_ must outlive the graph: the advance node reads it at every
        // replay.
        std::vector<double> base_ell_cpu = build_base_ell(
            static_cast<int>(ng_), cost_type_, eta_);
        auto base_ell_host = torch::from_blob(
            base_ell_cpu.data(), {ng_}, torch::kFloat64).clone();
        base_ell_ = base_ell_host.to(device_);

        // d_k is the k of the next replay, and kL = k·ℓ for that k.
        d_k_ = torch::full({1}, 1, i32_opts);
        kL_  = base_ell_.clone();

        CUDA_OK(cudaEventCreateWithFlags(&handoff_event_, cudaEventDisableTiming));
        {
            // stream_ waits for the setup work enqueued on the current stream.
            cudaEvent_t setup_done;
            CUDA_OK(cudaEventCreateWithFlags(&setup_done, cudaEventDisableTiming));
            CUDA_OK(cudaEventRecord(setup_done, setup_stream.stream()));
            CUDA_OK(cudaStreamWaitEvent(stream_.stream(), setup_done, 0));
            CUDA_OK(cudaEventDestroy(setup_done));
        }

        at::cuda::CUDAStreamGuard stream_guard(stream_);
        cudaStream_t stream = stream_.stream();

        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

        launch_fused_u(
            stream,
            A_.data_ptr<double>(), log_u_.data_ptr<double>(),
            beta_tilde_.data_ptr<double>(), kL_.data_ptr<double>(),
            log_r_.data_ptr<double>(),
            static_cast<int>(ng_));

        launch_fused_v(
            stream,
            B_.data_ptr<double>(), log_v_.data_ptr<double>(),
            beta_tilde_.data_ptr<double>(),       // written: β̃ for the next iteration
            A_.data_ptr<double>(), kL_.data_ptr<double>(),
            log_c_.data_ptr<double>(),
            static_cast<int>(ng_));

        launch_advance(
            stream,
            d_k_.data_ptr<int>(), kL_.data_ptr<double>(),
            base_ell_.data_ptr<double>(),
            static_cast<int>(ng_));

        CUDA_OK(cudaStreamEndCapture(stream, &graph_));
        CUDA_OK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    }

    ~BdrsSession() {
        // Return codes are ignored: a destructor must not throw.
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_)      cudaGraphDestroy(graph_);
        if (handoff_event_) cudaEventDestroy(handoff_event_);
    }

    BdrsSession(const BdrsSession&) = delete;
    BdrsSession& operator=(const BdrsSession&) = delete;

    // Returns the k of the last enqueued iteration. The host does not wait;
    // the caller's current stream waits for the batch. The ordering is
    // one-way: stream_ does not wait for the caller's stream, so the caller's
    // reads of the state must have completed before the next step().
    int64_t step(int64_t n_iters) {
        TORCH_CHECK(n_iters > 0, "n_iters must be positive");
        at::cuda::CUDAGuard device_guard(device_);
        cudaStream_t stream = stream_.stream();

        for (int64_t i = 0; i < n_iters; ++i) {
            CUDA_OK(cudaGraphLaunch(graph_exec_, stream));
        }
        iters_abs_ += n_iters;

        CUDA_OK(cudaEventRecord(handoff_event_, stream));
        CUDA_OK(cudaStreamWaitEvent(
            at::cuda::getCurrentCUDAStream(device_.index()).stream(),
            handoff_event_, 0));
        return iters_abs_;
    }

    void synchronize() {
        at::cuda::CUDAGuard device_guard(device_);
        CUDA_OK(cudaStreamSynchronize(stream_.stream()));
    }

    // The session's own state buffers, not copies: later step() calls
    // overwrite them in place.
    torch::Tensor A()     const { return A_; }
    torch::Tensor B()     const { return B_; }
    torch::Tensor log_u() const { return log_u_; }
    torch::Tensor log_v() const { return log_v_; }
    int64_t iters()       const { return iters_abs_; }

private:
    int64_t ng_, N_;
    double eta_;
    int cost_type_;
    int64_t iters_abs_;
    torch::Device device_ = torch::kCUDA;

    torch::Tensor A_, B_, log_v_, log_u_, beta_tilde_, kL_;
    torch::Tensor base_ell_, log_r_, log_c_, d_k_;

    at::cuda::CUDAStream stream_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaEvent_t handoff_event_ = nullptr;
};


// ---- Point Cloud session ---------------------------------------------------
// L_ij = −κ(p_i, q_j)/(C_max·η), with κ recomputed from P and Q at every pass.
// The kernels read k from d_k; the advance node only increments it.

class PointCloudBdrsSession {
public:
    PointCloudBdrsSession(
        torch::Tensor P,
        torch::Tensor Q,
        torch::Tensor r,            // positive: the floored r̃
        torch::Tensor c,            // positive: the floored c̃
        double eta,
        int64_t cost_type,          // 0=l1, 1=sql2, 2=l2, 3=linf; the kernel computes l1
                                    //   for any other value
        double c_max                // C_max > 0, not checked here
    )
        : eta_(eta), cost_type_(static_cast<int>(cost_type)), c_max_(c_max),
          iters_abs_(0),
          stream_(at::cuda::getStreamFromPool(/*is_high_priority=*/false,
                                              r.device().index()))
    {
        TORCH_CHECK(P.is_cuda() && Q.is_cuda() && r.is_cuda() && c.is_cuda(),
                    "P, Q, r, c must be CUDA tensors");
        TORCH_CHECK(P.dtype() == torch::kFloat64 && Q.dtype() == torch::kFloat64
                    && r.dtype() == torch::kFloat64 && c.dtype() == torch::kFloat64,
                    "P, Q, r, c must be float64");
        TORCH_CHECK(P.dim() == 2 && Q.dim() == 2, "P, Q must be 2D [N, d]");
        m_ = P.size(0);
        n_ = Q.size(0);
        d_ = P.size(1);
        TORCH_CHECK(Q.size(1) == d_, "P and Q must share the coordinate dim d");
        // The kernels copy a point into a per-thread array of
        // POINTCLOUD_MAXD = 16 doubles.
        TORCH_CHECK(d_ >= 1 && d_ <= 16, "coordinate dim d must be in [1, 16]; got ", d_);
        TORCH_CHECK(r.numel() == m_, "r length must equal P rows m=", m_);
        TORCH_CHECK(c.numel() == n_, "c length must equal Q rows n=", n_);
        TORCH_CHECK(m_ >= 1 && n_ >= 1, "m, n must be positive");

        device_ = r.device();
        at::cuda::CUDAGuard device_guard(device_);

        P_ = P.contiguous();
        Q_ = Q.contiguous();
        r = r.contiguous();
        c = c.contiguous();

        auto setup_stream = at::cuda::getCurrentCUDAStream();
        auto f64_opts = torch::TensorOptions().dtype(torch::kFloat64).device(device_);
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

        // X^0 = 1/(mn), split as in factored_torch.bregman_dr_solve_pointcloud.
        // In exact arithmetic log u, log v and X do not depend on the split
        // into ξ^0 and ζ^0.
        A_     = torch::full({m_}, -std::log(static_cast<double>(m_)), f64_opts);
        B_     = torch::full({n_}, -std::log(static_cast<double>(n_)), f64_opts);
        log_v_ = torch::zeros({n_}, f64_opts);
        // Zero, not empty: with K_max = 0 the delivered gap reads log u.
        log_u_ = torch::zeros({m_}, f64_opts);
        beta_tilde_ = B_ + log_v_;        // β̃^0; fused_v writes β̃ thereafter

        log_r_ = torch::log(r);
        log_c_ = torch::log(c);

        d_k_ = torch::full({1}, 1, i32_opts);   // the first replay runs k = 1

        CUDA_OK(cudaEventCreateWithFlags(&handoff_event_, cudaEventDisableTiming));
        {
            cudaEvent_t setup_done;
            CUDA_OK(cudaEventCreateWithFlags(&setup_done, cudaEventDisableTiming));
            CUDA_OK(cudaEventRecord(setup_done, setup_stream.stream()));
            CUDA_OK(cudaStreamWaitEvent(stream_.stream(), setup_done, 0));
            CUDA_OK(cudaEventDestroy(setup_done));
        }

        at::cuda::CUDAStreamGuard stream_guard(stream_);
        cudaStream_t stream = stream_.stream();

        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

        launch_pointcloud_fused_u(
            stream,
            A_.data_ptr<double>(), log_u_.data_ptr<double>(),
            beta_tilde_.data_ptr<double>(),
            P_.data_ptr<double>(), Q_.data_ptr<double>(),
            log_r_.data_ptr<double>(),
            d_k_.data_ptr<int>(), eta_, c_max_, cost_type_,
            static_cast<int>(d_), static_cast<int>(m_), static_cast<int>(n_));

        launch_pointcloud_fused_v(
            stream,
            B_.data_ptr<double>(), log_v_.data_ptr<double>(),
            beta_tilde_.data_ptr<double>(),
            A_.data_ptr<double>(),
            P_.data_ptr<double>(), Q_.data_ptr<double>(),
            log_c_.data_ptr<double>(),
            d_k_.data_ptr<int>(), eta_, c_max_, cost_type_,
            static_cast<int>(d_), static_cast<int>(m_), static_cast<int>(n_));

        launch_pointcloud_advance(stream, d_k_.data_ptr<int>());

        CUDA_OK(cudaStreamEndCapture(stream, &graph_));
        CUDA_OK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    }

    ~PointCloudBdrsSession() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_)      cudaGraphDestroy(graph_);
        if (handoff_event_) cudaEventDestroy(handoff_event_);
    }

    PointCloudBdrsSession(const PointCloudBdrsSession&) = delete;
    PointCloudBdrsSession& operator=(const PointCloudBdrsSession&) = delete;

    int64_t step(int64_t n_iters) {
        TORCH_CHECK(n_iters > 0, "n_iters must be positive");
        at::cuda::CUDAGuard device_guard(device_);
        cudaStream_t stream = stream_.stream();
        for (int64_t i = 0; i < n_iters; ++i) {
            CUDA_OK(cudaGraphLaunch(graph_exec_, stream));
        }
        iters_abs_ += n_iters;
        CUDA_OK(cudaEventRecord(handoff_event_, stream));
        CUDA_OK(cudaStreamWaitEvent(
            at::cuda::getCurrentCUDAStream(device_.index()).stream(),
            handoff_event_, 0));
        return iters_abs_;
    }

    void synchronize() {
        at::cuda::CUDAGuard device_guard(device_);
        CUDA_OK(cudaStreamSynchronize(stream_.stream()));
    }

    torch::Tensor A()     const { return A_; }
    torch::Tensor B()     const { return B_; }
    torch::Tensor log_u() const { return log_u_; }
    torch::Tensor log_v() const { return log_v_; }
    int64_t iters()       const { return iters_abs_; }

private:
    int64_t m_, n_, d_;
    double eta_;
    int cost_type_;
    double c_max_;
    int64_t iters_abs_;
    torch::Device device_ = torch::kCUDA;

    torch::Tensor P_, Q_, A_, B_, log_v_, log_u_, beta_tilde_;
    torch::Tensor log_r_, log_c_, d_k_;

    at::cuda::CUDAStream stream_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaEvent_t handoff_event_ = nullptr;
};


// ---- General session -------------------------------------------------------
// L = −C/η with C as given; the session does not normalize C. The column pass
// keeps the two-stage shape of dense_col_lse_scaled (partial LSEs over chunks
// of rows_per_chunk rows, then a merge per column), so with the same
// rows_per_chunk the iterates are bit-identical to the eager loop over
// dense_row_lse_scaled and dense_col_lse_scaled.
class DenseFactoredSession {
public:
    DenseFactoredSession(
        torch::Tensor C,
        torch::Tensor r,            // positive: the floored r̃
        torch::Tensor c,            // positive: the floored c̃
        double eta,
        int64_t rows_per_chunk
    )
        : eta_(eta), iters_abs_(0),
          stream_(at::cuda::getStreamFromPool(/*is_high_priority=*/false,
                                              r.device().index()))
    {
        TORCH_CHECK(C.is_cuda() && r.is_cuda() && c.is_cuda(),
                    "C, r, c must be CUDA tensors");
        TORCH_CHECK(C.dtype() == torch::kFloat64 && r.dtype() == torch::kFloat64
                    && c.dtype() == torch::kFloat64,
                    "C, r, c must be float64");
        TORCH_CHECK(C.dim() == 2, "C must be 2D [m, n]");
        TORCH_CHECK(C.is_contiguous(),
                    "C must be contiguous: the kernels index it as C[i*n + j]");
        m_ = C.size(0);
        n_ = C.size(1);
        TORCH_CHECK(r.numel() == m_, "r length must equal C rows m=", m_);
        TORCH_CHECK(c.numel() == n_, "c length must equal C cols n=", n_);
        TORCH_CHECK(m_ >= 1 && n_ >= 1, "m, n must be positive");
        TORCH_CHECK(rows_per_chunk >= 1, "rows_per_chunk must be positive");
        rows_per_chunk_ = static_cast<int>(rows_per_chunk);
        n_chunks_ = static_cast<int>((m_ + rows_per_chunk_ - 1) / rows_per_chunk_);

        device_ = r.device();
        at::cuda::CUDAGuard device_guard(device_);

        // Borrowed, not copied (C is 32 GiB at m = n = 65536). The graph reads
        // C at every replay, so the caller must not modify C while the session
        // is in use.
        C_ = C;
        r = r.contiguous();
        c = c.contiguous();

        auto setup_stream = at::cuda::getCurrentCUDAStream();
        auto f64_opts = torch::TensorOptions().dtype(torch::kFloat64).device(device_);
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

        // X^0 = 1/(mn), split as in factored_torch.bregman_dr_solve_pointcloud.
        A_     = torch::full({m_}, -std::log(static_cast<double>(m_)), f64_opts);
        B_     = torch::full({n_}, -std::log(static_cast<double>(n_)), f64_opts);
        log_v_ = torch::zeros({n_}, f64_opts);
        // Zero, not empty: with K_max = 0 the delivered gap reads log u.
        log_u_ = torch::zeros({m_}, f64_opts);
        beta_tilde_ = B_ + log_v_;        // β̃^0; v_stage2 writes β̃ thereafter

        log_r_ = torch::log(r);
        log_c_ = torch::log(c);

        scratch_m_ = torch::empty({(int64_t)n_chunks_ * n_}, f64_opts);
        scratch_s_ = torch::empty({(int64_t)n_chunks_ * n_}, f64_opts);

        d_k_ = torch::full({1}, 1, i32_opts);   // the first replay runs k = 1

        CUDA_OK(cudaEventCreateWithFlags(&handoff_event_, cudaEventDisableTiming));
        {
            cudaEvent_t setup_done;
            CUDA_OK(cudaEventCreateWithFlags(&setup_done, cudaEventDisableTiming));
            CUDA_OK(cudaEventRecord(setup_done, setup_stream.stream()));
            CUDA_OK(cudaStreamWaitEvent(stream_.stream(), setup_done, 0));
            CUDA_OK(cudaEventDestroy(setup_done));
        }

        at::cuda::CUDAStreamGuard stream_guard(stream_);
        cudaStream_t stream = stream_.stream();

        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

        launch_dense_factored_u(
            stream,
            A_.data_ptr<double>(), log_u_.data_ptr<double>(),
            beta_tilde_.data_ptr<double>(),
            C_.data_ptr<double>(), log_r_.data_ptr<double>(),
            d_k_.data_ptr<int>(), eta_,
            static_cast<int>(m_), static_cast<int>(n_));

        launch_dense_factored_v_stage1(
            stream,
            scratch_m_.data_ptr<double>(), scratch_s_.data_ptr<double>(),
            C_.data_ptr<double>(), A_.data_ptr<double>(),
            d_k_.data_ptr<int>(), eta_,
            static_cast<int>(m_), static_cast<int>(n_),
            rows_per_chunk_, n_chunks_);

        launch_dense_factored_v_stage2(
            stream,
            B_.data_ptr<double>(), log_v_.data_ptr<double>(),
            beta_tilde_.data_ptr<double>(),
            scratch_m_.data_ptr<double>(), scratch_s_.data_ptr<double>(),
            log_c_.data_ptr<double>(),
            static_cast<int>(n_), n_chunks_);

        launch_pointcloud_advance(stream, d_k_.data_ptr<int>());

        CUDA_OK(cudaStreamEndCapture(stream, &graph_));
        CUDA_OK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    }

    ~DenseFactoredSession() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_)      cudaGraphDestroy(graph_);
        if (handoff_event_) cudaEventDestroy(handoff_event_);
    }

    DenseFactoredSession(const DenseFactoredSession&) = delete;
    DenseFactoredSession& operator=(const DenseFactoredSession&) = delete;

    int64_t step(int64_t n_iters) {
        TORCH_CHECK(n_iters > 0, "n_iters must be positive");
        at::cuda::CUDAGuard device_guard(device_);
        cudaStream_t stream = stream_.stream();
        for (int64_t i = 0; i < n_iters; ++i) {
            CUDA_OK(cudaGraphLaunch(graph_exec_, stream));
        }
        iters_abs_ += n_iters;
        CUDA_OK(cudaEventRecord(handoff_event_, stream));
        CUDA_OK(cudaStreamWaitEvent(
            at::cuda::getCurrentCUDAStream(device_.index()).stream(),
            handoff_event_, 0));
        return iters_abs_;
    }

    void synchronize() {
        at::cuda::CUDAGuard device_guard(device_);
        CUDA_OK(cudaStreamSynchronize(stream_.stream()));
    }

    torch::Tensor A()     const { return A_; }
    torch::Tensor B()     const { return B_; }
    torch::Tensor log_u() const { return log_u_; }
    torch::Tensor log_v() const { return log_v_; }
    int64_t iters()       const { return iters_abs_; }
    int64_t rows_per_chunk() const { return rows_per_chunk_; }
    int64_t n_chunks()       const { return n_chunks_; }
    // d_k, the k of the next iteration (n + 1 after n replays). Exposed for
    // tests: a graph that replayed a fixed k would still give finite iterates.
    torch::Tensor k_device() const { return d_k_; }

private:
    int64_t m_, n_;
    double eta_;
    int rows_per_chunk_ = 1;
    int n_chunks_ = 1;
    int64_t iters_abs_;
    torch::Device device_ = torch::kCUDA;

    torch::Tensor C_, A_, B_, log_v_, log_u_, beta_tilde_;
    torch::Tensor log_r_, log_c_, d_k_;
    torch::Tensor scratch_m_, scratch_s_;

    at::cuda::CUDAStream stream_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaEvent_t handoff_event_ = nullptr;
};



// ---- Grid plan materializer ------------------------------------------------
// X_ij = exp(ξ_i + k·L_ij + ζ_j) at k = k_iter, row i the source pixel (row
// sums ≈ r). kL = k·ℓ is the product advance_kernel forms, so X matches the
// session's state at that k.

torch::Tensor materialize_X_impl(
    torch::Tensor A,
    torch::Tensor B,
    int64_t ng,
    double eta,
    int64_t cost_type,
    int64_t k_iter
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),                "A, B must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat64 && B.dtype() == torch::kFloat64,
                "A, B must be fp64");
    int64_t N = ng * ng;
    TORCH_CHECK(A.numel() == N && B.numel() == N,
                "A, B must have length ng*ng = ", N);
    TORCH_CHECK(ng > 0 && (ng & (ng - 1)) == 0,
                "ng must be a positive power of 2; got ", ng);

    at::cuda::CUDAGuard device_guard(A.device());
    auto f64_opts = torch::TensorOptions().dtype(torch::kFloat64).device(A.device());

    std::vector<double> base_ell_cpu = build_base_ell(
        static_cast<int>(ng), static_cast<int>(cost_type), eta);
    auto base_ell_host = torch::from_blob(
        base_ell_cpu.data(), {ng}, torch::kFloat64).clone();
    auto base_ell = base_ell_host.to(A.device());
    auto kL = base_ell * static_cast<double>(k_iter);

    auto X = torch::empty({N, N}, f64_opts);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_materialize_X(
        stream,
        X.data_ptr<double>(),
        A.contiguous().data_ptr<double>(),
        B.contiguous().data_ptr<double>(),
        kL.data_ptr<double>(),
        static_cast<int>(ng));
    CUDA_OK(cudaStreamSynchronize(stream));
    return X;
}


// ---- Separable min-plus c-transform ----------------------------------------
// q_m = min_k (C_mk − p_k) for the Grid cost C_mk = ax[|m_a − k_a|] +
// ax[|m_b − k_b|], with ax[δ] = φ(δ)/C_max (no η), as two 1-D min passes in
// O(n_g³); each block runs n_g threads, hence n_g ≤ 1024. q is bit-identical to
// dual_cleanup._ctransform_separable, NaN included: a min does not round, and
// each term is the same single subtraction or addition.

torch::Tensor ctransform_separable_impl(
    torch::Tensor p,    // row-major (k_a, k_b)
    torch::Tensor ax,
    int64_t ng
) {
    TORCH_CHECK(p.is_cuda() && ax.is_cuda(), "p, ax must be CUDA");
    TORCH_CHECK(p.dtype() == torch::kFloat64 && ax.dtype() == torch::kFloat64,
                "p, ax must be fp64");
    TORCH_CHECK(p.numel() == ng * ng, "p must have ng*ng elements");
    TORCH_CHECK(ax.numel() == ng, "ax must have ng elements");
    TORCH_CHECK(ng >= 1 && ng <= 1024, "ng must be in [1, 1024]");

    at::cuda::CUDAGuard device_guard(p.device());
    auto pc = p.contiguous();
    auto axc = ax.contiguous();
    auto q = torch::empty_like(pc);
    launch_ctransform_separable(
        at::cuda::getCurrentCUDAStream().stream(),
        q.data_ptr<double>(), pc.data_ptr<double>(), axc.data_ptr<double>(),
        static_cast<int>(ng));
    return q.view_as(p);
}


// ---- Scaled log-sum-exp passes over a stored C (General, eager form) ------
//   dense_row_lse_scaled:  out_i = LSE_j(s·C_ij + w_j),
//   dense_col_lse_scaled:  out_j = LSE_i(s·C_ij + w_i), in two stages over
//                          chunks of rows_per_chunk rows,
// with s (`neg_scale`) = −k/η. scratch_m and scratch_s must be contiguous fp64
// CUDA tensors; only their size is checked.
static void _check_dense_mat(const torch::Tensor& A, int64_t m, int64_t n,
                             const char* name) {
    TORCH_CHECK(A.is_cuda(), name, ": matrix must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat64, name, ": matrix must be fp64");
    TORCH_CHECK(A.is_contiguous(), name, ": matrix must be contiguous");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == m && A.size(1) == n,
                name, ": matrix must be (", m, ", ", n, ")");
}

static void _check_dense_vec(const torch::Tensor& v, int64_t len,
                             const char* name) {
    TORCH_CHECK(v.is_cuda(), name, ": vector must be CUDA");
    TORCH_CHECK(v.dtype() == torch::kFloat64, name, ": vector must be fp64");
    TORCH_CHECK(v.is_contiguous(), name, ": vector must be contiguous");
    TORCH_CHECK(v.numel() == len, name, ": vector must have length ", len);
}

void dense_row_lse_scaled_impl(torch::Tensor out, torch::Tensor C,
                               torch::Tensor w, double neg_scale) {
    TORCH_CHECK(C.dim() == 2, "dense_row_lse_scaled: C must be 2-D");
    int64_t m = C.size(0), n = C.size(1);
    _check_dense_mat(C, m, n, "dense_row_lse_scaled");
    _check_dense_vec(w, n, "dense_row_lse_scaled");
    _check_dense_vec(out, m, "dense_row_lse_scaled");
    at::cuda::CUDAGuard device_guard(C.device());
    launch_dense_row_lse_scaled(
        at::cuda::getCurrentCUDAStream().stream(),
        out.data_ptr<double>(), C.data_ptr<double>(), w.data_ptr<double>(),
        neg_scale, static_cast<int>(m), static_cast<int>(n));
}

void dense_col_lse_scaled_impl(torch::Tensor out, torch::Tensor C,
                               torch::Tensor w, double neg_scale,
                               torch::Tensor scratch_m, torch::Tensor scratch_s,
                               int64_t rows_per_chunk) {
    TORCH_CHECK(C.dim() == 2, "dense_col_lse_scaled: C must be 2-D");
    int64_t m = C.size(0), n = C.size(1);
    _check_dense_mat(C, m, n, "dense_col_lse_scaled");
    _check_dense_vec(w, m, "dense_col_lse_scaled");
    _check_dense_vec(out, n, "dense_col_lse_scaled");
    TORCH_CHECK(rows_per_chunk > 0, "rows_per_chunk must be positive");
    int64_t n_chunks = (m + rows_per_chunk - 1) / rows_per_chunk;
    TORCH_CHECK(scratch_m.numel() >= n_chunks * n && scratch_s.numel() >= n_chunks * n,
                "dense_col_lse_scaled: scratch too small (need ", n_chunks * n, ")");
    at::cuda::CUDAGuard device_guard(C.device());
    launch_dense_col_lse_scaled(
        at::cuda::getCurrentCUDAStream().stream(),
        out.data_ptr<double>(),
        scratch_m.data_ptr<double>(), scratch_s.data_ptr<double>(),
        C.data_ptr<double>(), w.data_ptr<double>(), neg_scale,
        static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(rows_per_chunk), static_cast<int>(n_chunks));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BdrsSession, std::shared_ptr<BdrsSession>>(m, "BdrsSession")
        .def(py::init<torch::Tensor, torch::Tensor, int64_t, double, int64_t,
                      torch::Tensor, torch::Tensor, torch::Tensor>(),
             py::arg("r"), py::arg("c"), py::arg("ng"), py::arg("eta"),
             py::arg("cost_type"),
             py::arg("init_A") = torch::Tensor(),
             py::arg("init_B") = torch::Tensor(),
             py::arg("init_log_v") = torch::Tensor())
        .def("step", &BdrsSession::step, py::arg("n_iters"),
             "Run n_iters graph replays (fused_u, fused_v, advance); returns the "
             "total iteration count after them.")
        .def("synchronize", &BdrsSession::synchronize)
        .def_property_readonly("A", &BdrsSession::A)
        .def_property_readonly("B", &BdrsSession::B)
        .def_property_readonly("log_u", &BdrsSession::log_u)
        .def_property_readonly("log_v", &BdrsSession::log_v)
        .def_property_readonly("iters", &BdrsSession::iters);

    py::class_<PointCloudBdrsSession, std::shared_ptr<PointCloudBdrsSession>>(m, "PointCloudBdrsSession")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                      double, int64_t, double>(),
             py::arg("P"), py::arg("Q"), py::arg("r"), py::arg("c"),
             py::arg("eta"), py::arg("cost_type"), py::arg("c_max"))
        .def("step", &PointCloudBdrsSession::step, py::arg("n_iters"),
             "Run n_iters graph replays (pointcloud_fused_u, pointcloud_fused_v, "
             "advance); returns the total iteration count after them.")
        .def("synchronize", &PointCloudBdrsSession::synchronize)
        .def_property_readonly("A", &PointCloudBdrsSession::A)
        .def_property_readonly("B", &PointCloudBdrsSession::B)
        .def_property_readonly("log_u", &PointCloudBdrsSession::log_u)
        .def_property_readonly("log_v", &PointCloudBdrsSession::log_v)
        .def_property_readonly("iters", &PointCloudBdrsSession::iters);

    py::class_<DenseFactoredSession, std::shared_ptr<DenseFactoredSession>>(m, "DenseFactoredSession")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
                      double, int64_t>(),
             py::arg("C"), py::arg("r"), py::arg("c"),
             py::arg("eta"), py::arg("rows_per_chunk"))
        .def("step", &DenseFactoredSession::step, py::arg("n_iters"),
             "Run n_iters graph replays (dense_factored_u, v_stage1, v_stage2, "
             "advance); returns the total iteration count after them.")
        .def("synchronize", &DenseFactoredSession::synchronize)
        .def_property_readonly("A", &DenseFactoredSession::A)
        .def_property_readonly("B", &DenseFactoredSession::B)
        .def_property_readonly("log_u", &DenseFactoredSession::log_u)
        .def_property_readonly("log_v", &DenseFactoredSession::log_v)
        .def_property_readonly("iters", &DenseFactoredSession::iters)
        .def_property_readonly("rows_per_chunk", &DenseFactoredSession::rows_per_chunk)
        .def_property_readonly("n_chunks", &DenseFactoredSession::n_chunks)
        .def_property_readonly("k_device", &DenseFactoredSession::k_device);


    m.def("materialize_X",
          &materialize_X_impl,
          "Materialize the dense N×N plan from a final (A, B, k_iter) factored state.",
          py::arg("A"),
          py::arg("B"),
          py::arg("ng"),
          py::arg("eta"),
          py::arg("cost_type"),
          py::arg("k_iter"));

    m.def("ctransform_separable",
          &ctransform_separable_impl,
          "Separable min-plus c-transform q[m] = min_k(C[m,k] - p[k]), bit-identical "
          "to the torch reference; a NaN in p gives NaN.",
          py::arg("p"), py::arg("ax"), py::arg("ng"));

    m.def("dense_row_lse_scaled", &dense_row_lse_scaled_impl,
          "out[i] = LSE_j(neg_scale*C[i,j] + w[j]), the u-pass of the General "
          "solver. neg_scale = -k/eta is negated on the host (the torch helpers "
          "take +k/eta) and passed by value, so the call must not be graph-"
          "captured. Each term is rounded as torch rounds it (no fused multiply-"
          "add); the reduction order differs from torch's.",
          py::arg("out"), py::arg("C"), py::arg("w"), py::arg("neg_scale"));

    m.def("dense_col_lse_scaled", &dense_col_lse_scaled_impl,
          "out[j] = LSE_i(neg_scale*C[i,j] + w[i]) in two stages, the v-pass of the "
          "General solver. neg_scale = -k/eta is negated on the host and passed "
          "by value, so the call must not be graph-captured. scratch_m and "
          "scratch_s hold the per-chunk partial states.",
          py::arg("out"), py::arg("C"), py::arg("w"), py::arg("neg_scale"),
          py::arg("scratch_m"), py::arg("scratch_s"),
          py::arg("rows_per_chunk") = static_cast<int64_t>(1024));
}
