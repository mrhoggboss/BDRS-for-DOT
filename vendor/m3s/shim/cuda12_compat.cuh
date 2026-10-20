// Polyfills for two CUDA APIs that the pinned M3S upstream uses and that a
// CUDA 12 build for sm_70+ lacks. CMakeLists.txt force-includes this header
// into every CUDA translation unit (nvcc --pre-include), so the upstream
// sources stay unedited. Neither polyfill changes any arithmetic.
//
// 1. __shfl_down, the pre-CUDA-9 warp shuffle, which device code for sm_70+
//    cannot use. Its only caller is upstream's __shfl_down_(double) helper
//    (kernel_manage.h), used only in the warp reduction of
//    kernal_mat_u_32_wrap (solver.cu). That kernel gives each warp one row
//    behind a warp-uniform branch, so all 32 lanes reach the shuffle and the
//    full-mask __shfl_down_sync is equivalent.
//
// 2. cusparseDgthr, the cuSPARSE gather xVal[i] = y[xInd[i]], removed in
//    CUDA 12. Upstream calls it twice in solver.cu, with
//    CUSPARSE_INDEX_BASE_ZERO, to permute COO values into sorted order; a
//    gather kernel replaces it.

#ifndef M3S_SHIM_CUDA12_COMPAT_CUH
#define M3S_SHIM_CUDA12_COMPAT_CUH

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>

// The CUDA headers still declare __shfl_down for the host pass and for
// __CUDA_ARCH__ < 700, so only the sm_70+ device pass is polyfilled.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__device__ __forceinline__ int __shfl_down(int var, unsigned int delta,
                                           int width = warpSize) {
    return __shfl_down_sync(0xffffffffu, var, delta, width);
}
#endif

__global__ inline void m3s_shim_dgthr_kernel_(int nnz, const double *y,
                                              double *xVal, const int *xInd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) xVal[i] = y[xInd[i]];
}

inline cusparseStatus_t cusparseDgthr(cusparseHandle_t /*handle*/, int nnz,
                                      const double *y, double *xVal,
                                      const int *xInd,
                                      cusparseIndexBase_t /*idxBase: always ZERO upstream*/) {
    if (nnz > 0)
        m3s_shim_dgthr_kernel_<<<(nnz + 255) / 256, 256>>>(nnz, y, xVal, xInd);
    return cudaPeekAtLastError() == cudaSuccess ? CUSPARSE_STATUS_SUCCESS
                                                : CUSPARSE_STATUS_EXECUTION_FAILED;
}

#endif  // M3S_SHIM_CUDA12_COMPAT_CUH
