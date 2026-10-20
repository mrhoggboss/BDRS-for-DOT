// bdrs_kernels.cu
//
// CUDA kernels of the BDRS solver for discrete optimal transport, in the Grid,
// General and Point Cloud formats, and their extern "C" launchers.
//
// Every format iterates the factored state
//
//     log X^k_ab = ξ^k_a + k·L_ab + ζ^k_b,     L = −C/η,
//
// with ξ (`A`) of length m and ζ (`B`) of length n, so X is never stored. One
// iteration is a u-update then a v-update:
//
//     ξ^k     = log r̃ − LSE_b(k·L_ab + β̃_b),     β̃ = ζ^{k−1} + log v^{k−1},
//     log u^k = ξ^k − ξ^{k−1},
//     ζ^k     = log c̃ − LSE_a(k·L_ab + ξ^k_a),
//     log v^k = ζ^k − ζ^{k−1},
//
// where LSE is log-sum-exp, r̃, c̃ are the floored marginals (`log_r`, `log_c`
// hold their logs) and β̃ is `beta_tilde`. The formats differ in how k·L_ab is
// formed:
//
//   Grid: an n_g × n_g lattice, N = n_g², with the separable cost
//       L_ab = ℓ[|a_r − b_r|] + ℓ[|a_c − b_c|],   ℓ[δ] = −φ(δ)/(C_max·η),
//       φ(δ) = δ (l1) or δ² (sql2), C_max the largest lattice cost. One table
//       kL = k·ℓ of length n_g serves both axes, and each LSE is two 1-D
//       passes: O(n_g³) work per iteration.
//       Every length-N array is row-major on the lattice, a = a_r·n_g + a_c.
//   General: −(k/η)·C_ab from a stored C, used as given (the solvers scale it
//       to max C = 1 by default).
//   Point Cloud: κ(p_a, q_b)/C_max recomputed from the clouds P [m, d] and
//       Q [n, d] at every term; nothing of size m × n is stored.
//
// All arithmetic is fp64. nvcc's default -fmad=true may contract a*b + c into
// one FMA, which rounds once instead of twice; where a term must match a torch
// reference bit for bit, it is written with __dmul_rn/__dadd_rn, which are
// never contracted.

#include <cuda_runtime.h>
#include <math.h>
#include <cstdlib>
#include <cstdint>


// Online log-sum-exp (Milakov and Gimelshein, 2018). The state (m, s) holds the
// running maximum m and s = Σ exp(val − m); start from (−inf, 0) and read the
// result as m + log(s). No exp argument is positive. A −inf value is skipped:
// seen while m = −inf it would give exp(−inf − (−inf)) = NaN. If every value
// is −inf the result is −inf.

__device__ __forceinline__
void online_lse_update(double &m, double &s, double val) {
    if (val == -INFINITY) return;
    if (val > m) {
        s = s * exp(m - val) + 1.0;
        m = val;
    } else {
        s += exp(val - m);
    }
}

// (m1, s1) ← (m1, s1) ⊕ (m2, s2). A merge sums the same terms as a serial
// accumulation in another order, so the last bits can differ. An empty state
// (m2 = −inf after the swap) is skipped to avoid the NaN of −inf − (−inf).
__device__ __forceinline__
void lse_merge(double &m1, double &s1, double m2, double s2) {
    if (m2 > m1) {
        double t = m1; m1 = m2; m2 = t;
        t = s1; s1 = s2; s2 = t;
    }
    if (m2 != -INFINITY) {
        s1 += s2 * exp(m2 - m1);
    }
}


// ----------------------------------------------------------------------------
// advance_kernel (Grid): the last node of the per-iteration CUDA graph.
// Increments *d_k and rebuilds the table kL[δ] = k·base_ell[δ]
// for the next replay. BdrsSession seeds d_k = 1 and kL = base_ell once,
// before the first replay.
//
// Launch: 1 block, n_g threads (n_g ≤ 1024).
// ----------------------------------------------------------------------------

__global__
void advance_kernel(
    int * __restrict__          d_k,
    double * __restrict__       kL,
    const double * __restrict__ base_ell,
    int ng
) {
    __shared__ int k_new;
    if (threadIdx.x == 0) {
        k_new = *d_k + 1;
        *d_k = k_new;
    }
    __syncthreads();
    int idx = (int)threadIdx.x;
    if (idx < ng) {
        kL[idx] = (double)k_new * base_ell[idx];
    }
}


// Fused Grid u-update (NG = n_g, split width W), one block per lattice row a_r:
//
//     Pass 1: logS[b_c]   = LSE_{b_r}(kL[|a_r − b_r|] + beta_tilde[b_r, b_c])
//     Pass 2: A[a_r, a_c] = log_r − LSE_{b_c}(kL[|a_c − b_c|] + logS[b_c])
//
// Thread tid has output index o = tid % NG (b_c in pass 1, a_c in pass 2) and
// lane = tid / NG. A warp covers 32 consecutive o (NG ≥ 32), so every global
// access arr[i·NG + o] is coalesced.
//
// W > 1 splits each reduction over W lanes (lane takes every W-th term) and
// merges the W states in shared memory in log2(W) rounds. This reassociates
// the sum, so the last bits can differ from W = 1; it raises the thread count
// where NG blocks of NG threads cannot fill the GPU (lse_w_for).

template<int NG, int W>
__global__ __launch_bounds__(NG * W)
void fused_u_update_t(
    double * __restrict__       A,           // [N]
    double * __restrict__       log_u,       // [N]
    const double * __restrict__ beta_tilde,  // [N]
    const double * __restrict__ kL,          // [n_g]
    const double * __restrict__ log_r        // [N]
) {
    extern __shared__ double smem_u[];
    double * smem_kL   = smem_u;
    double * smem_logS = smem_u + NG;
    double * smem_m    = smem_u + 2 * NG;
    double * smem_s    = smem_u + 2 * NG + NG * W;

    const int a_r  = blockIdx.x;
    const int tid  = (int)threadIdx.x;
    const int o    = tid % NG;
    const int lane = tid / NG;

    if (tid < NG) {
        smem_kL[tid] = kL[tid];
    }
    __syncthreads();

    // Pass 1
    {
        const int b_c = o;
        double m = -INFINITY;
        double s = 0.0;
        #pragma unroll 4
        for (int b_r = lane; b_r < NG; b_r += W) {
            double val = smem_kL[abs(a_r - b_r)] + beta_tilde[b_r * NG + b_c];
            online_lse_update(m, s, val);
        }
        if (W > 1) {
            smem_m[lane * NG + o] = m;
            smem_s[lane * NG + o] = s;
            __syncthreads();
            #pragma unroll
            for (int off = W >> 1; off > 0; off >>= 1) {
                if (lane < off) {
                    double m1 = smem_m[lane * NG + o];
                    double s1 = smem_s[lane * NG + o];
                    lse_merge(m1, s1,
                              smem_m[(lane + off) * NG + o],
                              smem_s[(lane + off) * NG + o]);
                    smem_m[lane * NG + o] = m1;
                    smem_s[lane * NG + o] = s1;
                }
                __syncthreads();
            }
            if (lane == 0) {
                smem_logS[o] = smem_m[o] + log(smem_s[o]);
            }
        } else {
            smem_logS[o] = m + log(s);
        }
    }
    __syncthreads();

    // Pass 2
    {
        const int a_c = o;
        double m = -INFINITY;
        double s = 0.0;
        #pragma unroll 4
        for (int b_c = lane; b_c < NG; b_c += W) {
            double val = smem_kL[abs(a_c - b_c)] + smem_logS[b_c];
            online_lse_update(m, s, val);
        }
        if (W > 1) {
            __syncthreads();
            smem_m[lane * NG + o] = m;
            smem_s[lane * NG + o] = s;
            __syncthreads();
            #pragma unroll
            for (int off = W >> 1; off > 0; off >>= 1) {
                if (lane < off) {
                    double m1 = smem_m[lane * NG + o];
                    double s1 = smem_s[lane * NG + o];
                    lse_merge(m1, s1,
                              smem_m[(lane + off) * NG + o],
                              smem_s[(lane + off) * NG + o]);
                    smem_m[lane * NG + o] = m1;
                    smem_s[lane * NG + o] = s1;
                }
                __syncthreads();
            }
            if (lane == 0) {
                int a = a_r * NG + o;
                double lse = smem_m[o] + log(smem_s[o]);
                double A_old = A[a];
                double A_new = log_r[a] - lse;
                A[a] = A_new;
                log_u[a] = A_new - A_old;
            }
        } else {
            int a = a_r * NG + o;
            double lse = m + log(s);
            double A_old = A[a];
            double A_new = log_r[a] - lse;
            A[a] = A_new;
            log_u[a] = A_new - A_old;
        }
    }
}


// Fused Grid v-update: the u-update with A and B swapped, one block per b_r:
//
//     Pass 1: logT[a_c]   = LSE_{a_r}(kL[|a_r − b_r|] + A[a_r, a_c])
//     Pass 2: B[b_r, b_c] = log_c − LSE_{a_c}(kL[|a_c − b_c|] + logT[a_c])
//
// The thread that owns b also writes the next β̃[b] = B_new + log_v_new from
// registers, the same fp64 addition as on the stored values, so no extra
// kernel is needed. BdrsSession computes β̃^0.

template<int NG, int W>
__global__ __launch_bounds__(NG * W)
void fused_v_update_t(
    double * __restrict__       B,                // [N]
    double * __restrict__       log_v,            // [N]
    double * __restrict__       beta_tilde_next,  // [N]
    const double * __restrict__ A,                // [N]
    const double * __restrict__ kL,               // [n_g]
    const double * __restrict__ log_c             // [N]
) {
    extern __shared__ double smem_v[];
    double * smem_kL   = smem_v;
    double * smem_logT = smem_v + NG;
    double * smem_m    = smem_v + 2 * NG;
    double * smem_s    = smem_v + 2 * NG + NG * W;

    const int b_r  = blockIdx.x;
    const int tid  = (int)threadIdx.x;
    const int o    = tid % NG;
    const int lane = tid / NG;

    if (tid < NG) {
        smem_kL[tid] = kL[tid];
    }
    __syncthreads();

    // Pass 1
    {
        const int a_c = o;
        double m = -INFINITY;
        double s = 0.0;
        #pragma unroll 4
        for (int a_r = lane; a_r < NG; a_r += W) {
            double val = smem_kL[abs(a_r - b_r)] + A[a_r * NG + a_c];
            online_lse_update(m, s, val);
        }
        if (W > 1) {
            smem_m[lane * NG + o] = m;
            smem_s[lane * NG + o] = s;
            __syncthreads();
            #pragma unroll
            for (int off = W >> 1; off > 0; off >>= 1) {
                if (lane < off) {
                    double m1 = smem_m[lane * NG + o];
                    double s1 = smem_s[lane * NG + o];
                    lse_merge(m1, s1,
                              smem_m[(lane + off) * NG + o],
                              smem_s[(lane + off) * NG + o]);
                    smem_m[lane * NG + o] = m1;
                    smem_s[lane * NG + o] = s1;
                }
                __syncthreads();
            }
            if (lane == 0) {
                smem_logT[o] = smem_m[o] + log(smem_s[o]);
            }
        } else {
            smem_logT[o] = m + log(s);
        }
    }
    __syncthreads();

    // Pass 2
    {
        const int b_c = o;
        double m = -INFINITY;
        double s = 0.0;
        #pragma unroll 4
        for (int a_c = lane; a_c < NG; a_c += W) {
            double val = smem_kL[abs(a_c - b_c)] + smem_logT[a_c];
            online_lse_update(m, s, val);
        }
        if (W > 1) {
            __syncthreads();
            smem_m[lane * NG + o] = m;
            smem_s[lane * NG + o] = s;
            __syncthreads();
            #pragma unroll
            for (int off = W >> 1; off > 0; off >>= 1) {
                if (lane < off) {
                    double m1 = smem_m[lane * NG + o];
                    double s1 = smem_s[lane * NG + o];
                    lse_merge(m1, s1,
                              smem_m[(lane + off) * NG + o],
                              smem_s[(lane + off) * NG + o]);
                    smem_m[lane * NG + o] = m1;
                    smem_s[lane * NG + o] = s1;
                }
                __syncthreads();
            }
            if (lane == 0) {
                int b = b_r * NG + o;
                double lse = smem_m[o] + log(smem_s[o]);
                double B_old     = B[b];
                double B_new     = log_c[b] - lse;
                double log_v_new = B_new - B_old;
                B[b] = B_new;
                log_v[b] = log_v_new;
                beta_tilde_next[b] = B_new + log_v_new;
            }
        } else {
            int b = b_r * NG + o;
            double lse = m + log(s);
            double B_old     = B[b];
            double B_new     = log_c[b] - lse;
            double log_v_new = B_new - B_old;
            B[b] = B_new;
            log_v[b] = log_v_new;
            beta_tilde_next[b] = B_new + log_v_new;
        }
    }
}


// Grid: the dense plan X[a, b] = exp(ξ_a + k·L_ab + ζ_b), one block per row a;
// n_g must be a power of 2. The exponential is not split into a product of
// factors: ξ, ζ and k·L grow in magnitude with k and cancel in the sum, so the
// factors can overflow or underflow where X itself is representable.

__global__
void materialize_X_kernel(
    double * __restrict__       X,         // [N * N], row-major
    const double * __restrict__ A,         // [N]
    const double * __restrict__ B,         // [N]
    const double * __restrict__ kL,        // [n_g]
    int ng,
    int log2_ng
) {
    extern __shared__ double smem_kL_m[];
    const int a = blockIdx.x;
    const int N = ng * ng;
    if (a >= N) return;

    for (int d = (int)threadIdx.x; d < ng; d += blockDim.x) {
        smem_kL_m[d] = kL[d];
    }
    __syncthreads();

    const int a_r = a >> log2_ng;
    const int a_c = a & (ng - 1);
    const double A_a = A[a];

    for (int b_r = 0; b_r < ng; ++b_r) {
        const double row_term = A_a + smem_kL_m[abs(a_r - b_r)];
        const int64_t base = (int64_t)a * N + (int64_t)b_r * ng;
        for (int b_c = (int)threadIdx.x; b_c < ng; b_c += blockDim.x) {
            X[base + b_c] = exp(row_term + smem_kL_m[abs(a_c - b_c)]
                                + B[b_r * ng + b_c]);
        }
    }
}


// Grid: the c-transform q_m = min_k (C_mk − p_k) as two 1-D min-plus passes,
//
//     H[k_b]      = min_{k_a} ( ax[|m_a − k_a|] − p[k_a, k_b] )   (pass 1)
//     q[m_a, m_b] = min_{k_b} ( ax[|m_b − k_b|] + H[k_b] )        (pass 2)
//
// with ax[δ] = φ(δ)/C_max in cost units (no η). One block per m_a, launched
// with exactly n_g threads (so n_g ≤ 1024). A min does not round, and each
// term is the same single subtraction or addition as in
// dual_cleanup._ctransform_separable, so the output equals the torch path bit
// for bit. A NaN term makes the min NaN, as torch's min does, so a NaN
// potential gives a NaN certificate. The Grid cost is symmetric, so p = α
// gives β′ and p = β′ gives α′.

__global__
void ctransform_separable_kernel(
    double * __restrict__       q,    // [N], row-major (m_a, m_b)
    const double * __restrict__ p,    // [N], row-major (k_a, k_b)
    const double * __restrict__ ax,   // [n_g]
    int ng
) {
    extern __shared__ double smem_ct[];
    double * smem_ax = smem_ct;
    double * smem_H  = smem_ct + ng;

    const int m_a = blockIdx.x;
    const int tid = (int)threadIdx.x;

    if (tid < ng) {
        smem_ax[tid] = ax[tid];
    }
    __syncthreads();

    // Pass 1
    {
        const int k_b = tid;
        double best = INFINITY;
        for (int k_a = 0; k_a < ng; ++k_a) {
            double val = smem_ax[abs(m_a - k_a)] - p[k_a * ng + k_b];
            if (val < best || isnan(val)) best = val;
        }
        smem_H[k_b] = best;
    }
    __syncthreads();

    // Pass 2
    {
        const int m_b = tid;
        double best = INFINITY;
        for (int k_b = 0; k_b < ng; ++k_b) {
            double val = smem_ax[abs(m_b - k_b)] + smem_H[k_b];
            if (val < best || isnan(val)) best = val;
        }
        q[m_a * ng + m_b] = best;
    }
}


// ---- General format: eager kernels over the stored C ----------------------
// The LSE argument is neg_scale·C_ij + w with neg_scale = −k/η, negated by the
// caller (the torch helpers _pointcloud_row_lse and _pointcloud_col_lse take
// +k/η instead; a wrong sign raises no error). __dmul_rn/__dadd_rn keep each
// term bit-identical to torch's separately rounded product and sum; only the
// LSE evaluation differs.
// neg_scale is passed by value, so these kernels must not be captured in a
// CUDA graph: every replay would silently reuse the capture-time k.

// out[i] = LSE_j(neg_scale·C_ij + w_j), one block per row i.
__global__ __launch_bounds__(256)
void dense_row_lse_scaled_kernel(
    double * __restrict__       out,  // [m]
    const double * __restrict__ C,    // [m*n]
    const double * __restrict__ w,    // [n]
    double neg_scale,           // −k/η
    int m, int n
) {
    __shared__ double sm[256];
    __shared__ double ss[256];
    const int i = blockIdx.x;
    const int t = (int)threadIdx.x;
    if (i >= m) return;
    const int64_t base = (int64_t)i * n;

    double mm = -INFINITY;
    double s = 0.0;
    for (int j = t; j < n; j += 256) {
        online_lse_update(mm, s, __dadd_rn(__dmul_rn(neg_scale, C[base + j]), w[j]));
    }
    sm[t] = mm;
    ss[t] = s;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (t < off) {
            double m1 = sm[t], s1 = ss[t];
            lse_merge(m1, s1, sm[t + off], ss[t + off]);
            sm[t] = m1; ss[t] = s1;
        }
        __syncthreads();
    }
    if (t == 0) {
        out[i] = sm[0] + log(ss[0]);
    }
}

// Column LSE out[j] = LSE_i(neg_scale·C_ij + w_i), stage 1: the partial state
// of column j over the rows of chunk blockIdx.y.
__global__
void dense_col_lse_scaled_stage1_kernel(
    double * __restrict__       scratch_m,  // [n_chunks * n]
    double * __restrict__       scratch_s,  // [n_chunks * n]
    const double * __restrict__ C,          // [m*n]
    const double * __restrict__ w,          // [m]
    double neg_scale,           // −k/η
    int m, int n,
    int rows_per_chunk
) {
    const int j = blockIdx.x * blockDim.x + (int)threadIdx.x;
    const int chunk = blockIdx.y;
    if (j >= n) return;
    const int i0 = chunk * rows_per_chunk;
    int i1 = i0 + rows_per_chunk;
    if (i1 > m) i1 = m;

    double mm = -INFINITY;
    double s = 0.0;
    for (int i = i0; i < i1; ++i) {
        online_lse_update(mm, s, __dadd_rn(__dmul_rn(neg_scale, C[(int64_t)i * n + j]), w[i]));
    }
    scratch_m[(int64_t)chunk * n + j] = mm;
    scratch_s[(int64_t)chunk * n + j] = s;
}

// Column LSE, stage 2: merges the n_chunks partial states of column j from
// dense_col_lse_scaled_stage1_kernel, in chunk order.
__global__
void dense_col_lse_stage2_kernel(
    double * __restrict__       out,        // [n]
    const double * __restrict__ scratch_m,  // [n_chunks * n]
    const double * __restrict__ scratch_s,  // [n_chunks * n]
    int n,
    int n_chunks
) {
    const int j = blockIdx.x * blockDim.x + (int)threadIdx.x;
    if (j >= n) return;
    double mm = scratch_m[j];
    double s = scratch_s[j];
    for (int c = 1; c < n_chunks; ++c) {
        lse_merge(mm, s, scratch_m[(int64_t)c * n + j], scratch_s[(int64_t)c * n + j]);
    }
    out[j] = mm + log(s);
}

// ---- General format: captured-session kernels -----------------------------
// DenseFactoredSession's kernels. The u-pass and v-pass stage 1 read k from the
// device counter d_k and form neg_scale = −(k/η) with the same division and
// negation as the eager caller, so one iteration can be captured as the graph
// u → v_stage1 → v_stage2 → advance. The iterates are bit-identical to the
// eager path: the terms keep the __dmul_rn/__dadd_rn form, the column pass
// keeps the eager two-stage shape so the reduction order matches, and the
// updates after each LSE, β̃ = ζ + log v included, are those of
// factored_torch.bregman_dr_solve_pointcloud in the same order.

// u-pass: ξ_a = log r̃_a − LSE_j(neg_scale·C_aj + β̃_j), one block per row a.
__global__ __launch_bounds__(256)
void dense_factored_u_kernel(
    double * __restrict__       A,            // [m]
    double * __restrict__       log_u,        // [m]
    const double * __restrict__ beta_tilde,   // [n]
    const double * __restrict__ C,            // [m*n]
    const double * __restrict__ log_r,        // [m]
    const int * __restrict__    d_k,
    double eta, int m, int n
) {
    __shared__ double sm[256];
    __shared__ double ss[256];
    const int a = blockIdx.x;
    if (a >= m) return;
    const int t = (int)threadIdx.x;
    const double neg_scale = -((double)(*d_k) / eta);
    const int64_t base = (int64_t)a * n;

    double mm = -INFINITY;
    double s = 0.0;
    for (int j = t; j < n; j += 256) {
        online_lse_update(mm, s, __dadd_rn(__dmul_rn(neg_scale, C[base + j]), beta_tilde[j]));
    }
    sm[t] = mm;
    ss[t] = s;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (t < off) {
            double m1 = sm[t], s1 = ss[t];
            lse_merge(m1, s1, sm[t + off], ss[t + off]);
            sm[t] = m1; ss[t] = s1;
        }
        __syncthreads();
    }
    if (t == 0) {
        double lse   = sm[0] + log(ss[0]);
        double A_old = A[a];
        double A_new = log_r[a] - lse;
        A[a]     = A_new;
        log_u[a] = A_new - A_old;
    }
}

// v-pass, stage 1: dense_col_lse_scaled_stage1_kernel with weights ξ.
__global__
void dense_factored_v_stage1_kernel(
    double * __restrict__       scratch_m,  // [n_chunks * n]
    double * __restrict__       scratch_s,  // [n_chunks * n]
    const double * __restrict__ C,          // [m*n]
    const double * __restrict__ A,          // [m]
    const int * __restrict__    d_k,
    double eta,
    int m, int n,
    int rows_per_chunk
) {
    const int j = blockIdx.x * blockDim.x + (int)threadIdx.x;
    const int chunk = blockIdx.y;
    if (j >= n) return;
    const double neg_scale = -((double)(*d_k) / eta);
    const int i0 = chunk * rows_per_chunk;
    int i1 = i0 + rows_per_chunk;
    if (i1 > m) i1 = m;

    double mm = -INFINITY;
    double s = 0.0;
    for (int i = i0; i < i1; ++i) {
        online_lse_update(mm, s, __dadd_rn(__dmul_rn(neg_scale, C[(int64_t)i * n + j]), A[i]));
    }
    scratch_m[(int64_t)chunk * n + j] = mm;
    scratch_s[(int64_t)chunk * n + j] = s;
}

// v-pass, stage 2: the merge of dense_col_lse_stage2_kernel, then ζ, log v and
// the next β̃.
__global__
void dense_factored_v_stage2_kernel(
    double * __restrict__       B,           // [n]
    double * __restrict__       log_v,       // [n]
    double * __restrict__       beta_tilde,  // [n]
    const double * __restrict__ scratch_m,   // [n_chunks * n]
    const double * __restrict__ scratch_s,   // [n_chunks * n]
    const double * __restrict__ log_c,       // [n]
    int n, int n_chunks
) {
    const int b = blockIdx.x * blockDim.x + (int)threadIdx.x;
    if (b >= n) return;
    double mm = scratch_m[b];
    double s  = scratch_s[b];
    for (int c = 1; c < n_chunks; ++c) {
        lse_merge(mm, s, scratch_m[(int64_t)c * n + b], scratch_s[(int64_t)c * n + b]);
    }
    double lse   = mm + log(s);
    double B_old = B[b];
    double B_new = log_c[b] - lse;
    double lv    = B_new - B_old;
    B[b]          = B_new;
    log_v[b]      = lv;
    beta_tilde[b] = B_new + lv;
}



// ---- Point Cloud format ----------------------------------------------------
// L_ab = −κ(p_a, q_b)/(C_max·η) with κ recomputed from the coordinates at every
// term, so there is no m × n array and no kL table; the advance node only
// increments k. P [m, d] and Q [n, d] are row-major; cost_type is 0: l1,
// 1: sql2, 2: l2, 3: linf (factored_cuda._POINTCLOUD_COST_TYPES). Each term
// (κ/C_max)·(−k/η) + weight is a plain multiply-add that nvcc may contract into
// an FMA (as it may the sums of squares of sql2 and l2), and the LSE order
// differs, so the iterates agree with the torch reference
// (factored_torch.bregman_dr_solve_pointcloud) to rounding, not bit for bit.

// pa/qb below are per-thread arrays of this length; PointCloudBdrsSession
// rejects d > 16.
#define POINTCLOUD_MAXD 16

// κ(p_a, q_b) before division by C_max. A cost_type outside {1, 2, 3} gives l1.
__device__ __forceinline__
double pointcloud_cost_raw(const double *pa, const double *qb, int cost_type, int d) {
    if (cost_type == 1) {            // sql2
        double acc = 0.0;
        for (int dd = 0; dd < d; ++dd) { double df = pa[dd] - qb[dd]; acc += df * df; }
        return acc;
    } else if (cost_type == 2) {     // l2
        double acc = 0.0;
        for (int dd = 0; dd < d; ++dd) { double df = pa[dd] - qb[dd]; acc += df * df; }
        return sqrt(acc);
    } else if (cost_type == 3) {     // linf
        double acc = 0.0;
        for (int dd = 0; dd < d; ++dd) { double df = fabs(pa[dd] - qb[dd]); if (df > acc) acc = df; }
        return acc;
    }                                // l1 (0, default)
    double acc = 0.0;
    for (int dd = 0; dd < d; ++dd) acc += fabs(pa[dd] - qb[dd]);
    return acc;
}

// x / c from rc = RN(1/c), computed once per thread (RN: round to nearest). An
// in-loop x / c compiles to div.rn.f64, which recomputes at every term a
// reciprocal that depends on c only. By Markstein's theorem, with q0 = RN(x·rc)
// within one ulp of x/c, the remainder r = x − c·q0 is exact in one FMA and
// RN(q0 + r·rc) = RN(x/c), the bits of the IEEE division, unless an
// intermediate underflows or overflows. The callers use it only for
// 1e-60 ≤ c ≤ 1e60, and x outside {0} ∪ [1e-180, 1e180] takes the plain
// division; these ranges keep every intermediate clear of both.
__device__ __forceinline__
double pointcloud_div_hoisted(double x, double c, double rc) {
    if (x == 0.0 || (x >= 1e-180 && x <= 1e180)) {
        double q0 = __dmul_rn(x, rc);
        double r  = __fma_rn(-c, q0, x);
        return __fma_rn(r, rc, q0);
    }
    return x / c;
}

// The last graph node of PointCloudBdrsSession and DenseFactoredSession.
__global__
void pointcloud_advance_kernel(int * __restrict__ d_k) {
    if (threadIdx.x == 0) *d_k += 1;
}

__global__ __launch_bounds__(256)
void pointcloud_fused_u_kernel(
    double * __restrict__       A,            // [m]
    double * __restrict__       log_u,        // [m]
    const double * __restrict__ beta_tilde,   // [n]
    const double * __restrict__ P,            // [m*d]
    const double * __restrict__ Q,            // [n*d]
    const double * __restrict__ log_r,        // [m]
    const int * __restrict__    d_k,
    double eta, double c_max, int cost_type, int d, int m, int n, int rcp_hoist
) {
    __shared__ double sm[256];
    __shared__ double ss[256];
    const int a = blockIdx.x;
    if (a >= m) return;
    const int t = (int)threadIdx.x;
    const double neg_scale = -((double)(*d_k) / eta);
    const double rc = 1.0 / c_max;
    const bool hoist = rcp_hoist && c_max >= 1e-60 && c_max <= 1e60;

    double pa[POINTCLOUD_MAXD];
    for (int dd = 0; dd < d; ++dd) pa[dd] = P[(int64_t)a * d + dd];

    double mloc = -INFINITY, sloc = 0.0;
    for (int b = t; b < n; b += 256) {
        double raw = pointcloud_cost_raw(pa, Q + (int64_t)b * d, cost_type, d);
        double cost_norm = hoist ? pointcloud_div_hoisted(raw, c_max, rc) : raw / c_max;
        double val = cost_norm * neg_scale + beta_tilde[b];
        online_lse_update(mloc, sloc, val);
    }
    sm[t] = mloc; ss[t] = sloc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (t < off) {
            double m1 = sm[t], s1 = ss[t];
            lse_merge(m1, s1, sm[t + off], ss[t + off]);
            sm[t] = m1; ss[t] = s1;
        }
        __syncthreads();
    }
    if (t == 0) {
        double lse = sm[0] + log(ss[0]);
        double A_old = A[a];
        double A_new = log_r[a] - lse;
        A[a] = A_new;
        log_u[a] = A_new - A_old;
    }
}

__global__ __launch_bounds__(256)
void pointcloud_fused_v_kernel(
    double * __restrict__       B,            // [n]
    double * __restrict__       log_v,        // [n]
    double * __restrict__       beta_tilde,   // [n]
    const double * __restrict__ A,            // [m]
    const double * __restrict__ P,            // [m*d]
    const double * __restrict__ Q,            // [n*d]
    const double * __restrict__ log_c,        // [n]
    const int * __restrict__    d_k,
    double eta, double c_max, int cost_type, int d, int m, int n, int rcp_hoist
) {
    __shared__ double sm[256];
    __shared__ double ss[256];
    const int b = blockIdx.x;
    if (b >= n) return;
    const int t = (int)threadIdx.x;
    const double neg_scale = -((double)(*d_k) / eta);
    const double rc = 1.0 / c_max;
    const bool hoist = rcp_hoist && c_max >= 1e-60 && c_max <= 1e60;

    double qb[POINTCLOUD_MAXD];
    for (int dd = 0; dd < d; ++dd) qb[dd] = Q[(int64_t)b * d + dd];

    double mloc = -INFINITY, sloc = 0.0;
    for (int a = t; a < m; a += 256) {
        double raw = pointcloud_cost_raw(P + (int64_t)a * d, qb, cost_type, d);
        double cost_norm = hoist ? pointcloud_div_hoisted(raw, c_max, rc) : raw / c_max;
        double val = cost_norm * neg_scale + A[a];
        online_lse_update(mloc, sloc, val);
    }
    sm[t] = mloc; ss[t] = sloc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (t < off) {
            double m1 = sm[t], s1 = ss[t];
            lse_merge(m1, s1, sm[t + off], ss[t + off]);
            sm[t] = m1; ss[t] = s1;
        }
        __syncthreads();
    }
    if (t == 0) {
        double lse = sm[0] + log(ss[0]);
        double B_old = B[b];
        double B_new = log_c[b] - lse;
        double lv = B_new - B_old;
        B[b] = B_new;
        log_v[b] = lv;
        beta_tilde[b] = B_new + lv;
    }
}


// ---- Launchers -------------------------------------------------------------
// bdrs_solver.cpp is compiled by the host C++ compiler, which does not accept
// the <<<...>>> syntax, so it launches the kernels through these.

// BDRS_LSE_SPLIT=0 forces W = 1 at every n_g; read once per process.
static int lse_split_enabled() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("BDRS_LSE_SPLIT");
        v = (e != nullptr && e[0] == '0') ? 0 : 1;
    }
    return v;
}

// Dynamic shared memory of the fused Grid kernels: kL [NG], logS or logT [NG],
// and the lane states m, s [NG·W] each.
static size_t fused_shmem_bytes(int ng, int w) {
    return (size_t)(2 * ng + 2 * ng * w) * sizeof(double);
}


extern "C"
void launch_advance(
    cudaStream_t stream,
    int *d_k, double *kL, const double *base_ell,
    int ng
) {
    advance_kernel<<<1, ng, 0, stream>>>(d_k, kL, base_ell, ng);
}


#define BDRS_DISPATCH_FUSED_U(NG_, W_)                                         \
    fused_u_update_t<NG_, W_>                                                  \
        <<<NG_, NG_ * W_, fused_shmem_bytes(NG_, W_), stream>>>(               \
            A, log_u, beta_tilde, kL, log_r)

#define BDRS_DISPATCH_FUSED_V(NG_, W_)                                         \
    fused_v_update_t<NG_, W_>                                                  \
        <<<NG_, NG_ * W_, fused_shmem_bytes(NG_, W_), stream>>>(               \
            B, log_v, beta_tilde_next, A, kL, log_c)

// W = 1024/n_g at n_g ∈ {32, 64, 128}, where n_g blocks of n_g threads cannot
// fill the GPU; at n_g = 256 a launch already has 2048 warps.
static int lse_w_for(int ng) {
    if (!lse_split_enabled()) return 1;
    switch (ng) {
        case 32:  return 32;
        case 64:  return 16;
        case 128: return 8;
        default:  return 1;
    }
}


extern "C"
void launch_fused_u(
    cudaStream_t stream,
    double *A, double *log_u,
    const double *beta_tilde, const double *kL, const double *log_r,
    int ng
) {
    const int w = lse_w_for(ng);
    switch (ng) {
        case 4:    BDRS_DISPATCH_FUSED_U(4, 1);    break;
        case 8:    BDRS_DISPATCH_FUSED_U(8, 1);    break;
        case 16:   BDRS_DISPATCH_FUSED_U(16, 1);   break;
        case 32:   if (w == 32) { BDRS_DISPATCH_FUSED_U(32, 32); }
                   else         { BDRS_DISPATCH_FUSED_U(32, 1);  } break;
        case 64:   if (w == 16) { BDRS_DISPATCH_FUSED_U(64, 16); }
                   else         { BDRS_DISPATCH_FUSED_U(64, 1);  } break;
        case 128:  if (w == 8)  { BDRS_DISPATCH_FUSED_U(128, 8); }
                   else         { BDRS_DISPATCH_FUSED_U(128, 1); } break;
        case 256:  BDRS_DISPATCH_FUSED_U(256, 1);  break;
        case 512:  BDRS_DISPATCH_FUSED_U(512, 1);  break;
        case 1024: BDRS_DISPATCH_FUSED_U(1024, 1); break;
        default:   break;   // BdrsSession admits only powers of 2 in [4, 1024]
    }
}


extern "C"
void launch_fused_v(
    cudaStream_t stream,
    double *B, double *log_v, double *beta_tilde_next,
    const double *A, const double *kL, const double *log_c,
    int ng
) {
    const int w = lse_w_for(ng);
    switch (ng) {
        case 4:    BDRS_DISPATCH_FUSED_V(4, 1);    break;
        case 8:    BDRS_DISPATCH_FUSED_V(8, 1);    break;
        case 16:   BDRS_DISPATCH_FUSED_V(16, 1);   break;
        case 32:   if (w == 32) { BDRS_DISPATCH_FUSED_V(32, 32); }
                   else         { BDRS_DISPATCH_FUSED_V(32, 1);  } break;
        case 64:   if (w == 16) { BDRS_DISPATCH_FUSED_V(64, 16); }
                   else         { BDRS_DISPATCH_FUSED_V(64, 1);  } break;
        case 128:  if (w == 8)  { BDRS_DISPATCH_FUSED_V(128, 8); }
                   else         { BDRS_DISPATCH_FUSED_V(128, 1); } break;
        case 256:  BDRS_DISPATCH_FUSED_V(256, 1);  break;
        case 512:  BDRS_DISPATCH_FUSED_V(512, 1);  break;
        case 1024: BDRS_DISPATCH_FUSED_V(1024, 1); break;
        default:   break;
    }
}


extern "C"
void launch_materialize_X(
    cudaStream_t stream,
    double *X, const double *A, const double *B, const double *kL,
    int ng
) {
    int N = ng * ng;
    int threads = (N < 256) ? N : 256;
    int log2_ng = 0;
    while ((1 << log2_ng) < ng) ++log2_ng;
    size_t shmem = (size_t)ng * sizeof(double);
    materialize_X_kernel<<<N, threads, shmem, stream>>>(X, A, B, kL, ng, log2_ng);
}


extern "C"
void launch_ctransform_separable(
    cudaStream_t stream,
    double *q, const double *p, const double *ax,
    int ng
) {
    size_t shmem = (size_t)(2 * ng) * sizeof(double);
    ctransform_separable_kernel<<<ng, ng, shmem, stream>>>(q, p, ax, ng);
}


extern "C"
void launch_dense_row_lse_scaled(
    cudaStream_t stream,
    double *out, const double *C, const double *w, double neg_scale, int m, int n
) {
    dense_row_lse_scaled_kernel<<<m, 256, 0, stream>>>(out, C, w, neg_scale, m, n);
}


extern "C"
void launch_dense_col_lse_scaled(
    cudaStream_t stream,
    double *out, double *scratch_m, double *scratch_s,
    const double *C, const double *w, double neg_scale,
    int m, int n, int rows_per_chunk, int n_chunks
) {
    dim3 block(256);
    dim3 grid1((n + 255) / 256, n_chunks);
    dense_col_lse_scaled_stage1_kernel<<<grid1, block, 0, stream>>>(
        scratch_m, scratch_s, C, w, neg_scale, m, n, rows_per_chunk);
    dim3 grid2((n + 255) / 256);
    dense_col_lse_stage2_kernel<<<grid2, block, 0, stream>>>(
        out, scratch_m, scratch_s, n, n_chunks);
}


// ---- General format: captured-session launchers ---------------------------

extern "C"
void launch_dense_factored_u(
    cudaStream_t stream,
    double *A, double *log_u, const double *beta_tilde,
    const double *C, const double *log_r, const int *d_k,
    double eta, int m, int n
) {
    dense_factored_u_kernel<<<m, 256, 0, stream>>>(
        A, log_u, beta_tilde, C, log_r, d_k, eta, m, n);
}

extern "C"
void launch_dense_factored_v_stage1(
    cudaStream_t stream,
    double *scratch_m, double *scratch_s,
    const double *C, const double *A, const int *d_k,
    double eta, int m, int n, int rows_per_chunk, int n_chunks
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256, n_chunks);
    dense_factored_v_stage1_kernel<<<grid, block, 0, stream>>>(
        scratch_m, scratch_s, C, A, d_k, eta, m, n, rows_per_chunk);
}

extern "C"
void launch_dense_factored_v_stage2(
    cudaStream_t stream,
    double *B, double *log_v, double *beta_tilde,
    const double *scratch_m, const double *scratch_s, const double *log_c,
    int n, int n_chunks
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    dense_factored_v_stage2_kernel<<<grid, block, 0, stream>>>(
        B, log_v, beta_tilde, scratch_m, scratch_s, log_c, n, n_chunks);
}


// ---- Point Cloud launchers -------------------------------------------------

extern "C"
void launch_pointcloud_advance(cudaStream_t stream, int *d_k) {
    pointcloud_advance_kernel<<<1, 1, 0, stream>>>(d_k);
}

// BDRS_PC_RCP_HOIST=0 selects the in-loop division, with bit-identical results.
// Read at every launch, which in a session means once, at graph capture.
static int pc_rcp_hoist_enabled() {
    const char *e = getenv("BDRS_PC_RCP_HOIST");
    return (e != nullptr && e[0] == '0') ? 0 : 1;
}

extern "C"
void launch_pointcloud_fused_u(
    cudaStream_t stream,
    double *A, double *log_u, const double *beta_tilde,
    const double *P, const double *Q, const double *log_r,
    const int *d_k, double eta, double c_max,
    int cost_type, int d, int m, int n
) {
    pointcloud_fused_u_kernel<<<m, 256, 0, stream>>>(
        A, log_u, beta_tilde, P, Q, log_r, d_k, eta, c_max, cost_type, d, m, n,
        pc_rcp_hoist_enabled());
}

extern "C"
void launch_pointcloud_fused_v(
    cudaStream_t stream,
    double *B, double *log_v, double *beta_tilde,
    const double *A, const double *P, const double *Q, const double *log_c,
    const int *d_k, double eta, double c_max,
    int cost_type, int d, int m, int n
) {
    pointcloud_fused_v_kernel<<<n, 256, 0, stream>>>(
        B, log_v, beta_tilde, A, P, Q, log_c, d_k, eta, c_max, cost_type, d, m, n,
        pc_rcp_hoist_enabled());
}
