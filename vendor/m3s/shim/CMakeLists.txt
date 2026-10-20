# vendor/m3s/shim/CMakeLists.txt — first-party build of the M3S extraction
# shim against the PINNED upstream tree in ../upstream (fetched by
# ../fetch.sh; NEVER build upstream's top-level CMake — it carries committed
# caches, requires Eigen/OpenMP for unrelated targets, and hard-codes
# LINK_DIRECTORIES("/home")).
#
# Source list = the shim TU + exactly the upstream TUs the M3S chain needs:
#   - SparseSinkhorn/kernel_manage.cu      (the second TU of upstream's
#     SparseSinkhorn target; solver.cu is textually #included by the shim,
#     exactly as upstream's image_ot_solver.cu does)
#   - the Common library source list verbatim (Common/CMakeLists.txt:3-12)
#     incl. the Models OBJECT sources (Common/Models/CMakeLists.txt).
# Upstream's `Sinkhorn` link is vestigial for this chain (no SparseSinkhorn
# TU includes a Sinkhorn/ header — verified at the PIN), so it is dropped;
# with it goes the only Eigen dependency. OpenMP is not used by any compiled
# TU (upstream's top-level find_package(OpenMP) serves other targets).
#
# CUDA separable compilation mirrors upstream's effective nvcc flags
# (--relocatable-device-code=true; the gencode/-O3 line in upstream's
# top-level CMake is dead — overwritten two lines later). getCostEff is a
# cross-TU __device__ function, so RDC is required.
cmake_minimum_required(VERSION 3.18)
project(m3s_shim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)            # upstream: add_definitions(-std=c++11)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# HOST code MUST stay at -O0 with pre-GCC-13 fallthrough semantics —
# upstream's effective build (no CMAKE_BUILD_TYPE; its -O3 nvcc-flag line is
# overwritten two lines later). Two upstream functions fall off the end of
# int-returning bodies (Familys::check_for_layers, family.h:97-121;
# CuEspsScalingHandler_::init_eps, eps_handler.cpp:4-78) — UB that
#   (a) gcc at -O2+ exploits by deleting the loop-exit test (the only
#       defined exits are the exit(0) calls) -> verified segfault at -O3
#       (check_for_layers ran i past nlayers);
#   (b) gcc >= 13 at -O0 turns into a ud2 trap (-funreachable-traps is
#       default-on at -O0 since GCC 13) -> verified SIGILL.
# -fno-unreachable-traps restores the silent-fallthrough behavior upstream
# was built and published under; it is passed only to a compiler that has it
# (GCC >= 13), since older GCC inserts no such trap and rejects the flag. So
# GCC 12 and GCC 13+ build the same host code. Device code keeps ptxas'
# default -O3, identical to upstream's effective build.
# (no -DNDEBUG: upstream's effective build keeps assert() ACTIVE; the
# cusparse-status asserts in sortcoo_and_to_csr* are load-bearing failure
# detection there.)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fno-unreachable-traps M3S_HAS_NO_UNREACHABLE_TRAPS)
if(M3S_HAS_NO_UNREACHABLE_TRAPS)
  set(M3S_HOST_FLAGS "-O0 -fno-unreachable-traps")
  set(M3S_XCOMPILER "-O0,-fno-unreachable-traps")
else()
  set(M3S_HOST_FLAGS "-O0")
  set(M3S_XCOMPILER "-O0")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${M3S_HOST_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=${M3S_XCOMPILER}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${M3S_HOST_FLAGS} -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-Xcompiler=${M3S_XCOMPILER} -g")

set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/../upstream")
if(NOT EXISTS "${UPSTREAM}/SparseSinkhorn/solver.cu")
  message(FATAL_ERROR
    "vendor/m3s/upstream is missing or incomplete — run vendor/m3s/fetch.sh first")
endif()

add_executable(m3s_shim
  m3s_shim.cu
  "${UPSTREAM}/SparseSinkhorn/kernel_manage.cu"
  # Common library sources, verbatim from upstream Common/CMakeLists.txt
  "${UPSTREAM}/Common/eps_handler.cpp"
  "${UPSTREAM}/Common/family.cpp"
  "${UPSTREAM}/Common/GridTools.cpp"
  "${UPSTREAM}/Common/Tools.cpp"
  "${UPSTREAM}/Common/TVarListHandler.cpp"
  "${UPSTREAM}/Common/TCouplingHandler.cpp"
  "${UPSTREAM}/Common/THierarchicalPartition.cpp"
  "${UPSTREAM}/Common/THierarchyBuilder.cpp"
  "${UPSTREAM}/Common/MultiScaleTools.cpp"
  "${UPSTREAM}/Common/Verbose.cpp"
  "${UPSTREAM}/Common/TEpsScaling.cpp"
  "${UPSTREAM}/Common/THierarchicalCostFunctionProvider.cpp"
  # Models OBJECT sources (Common/Models/CMakeLists.txt)
  "${UPSTREAM}/Common/Models/OT.cpp"
  "${UPSTREAM}/Common/Models/WFR.cpp"
  "${UPSTREAM}/Common/Models/Geometry_Sphere.cpp"
  "${UPSTREAM}/Common/Models/Geometry_Reflector.cpp"
)

target_include_directories(m3s_shim PRIVATE
  "${UPSTREAM}"                      # <SparseSinkhorn/solver.cu>, "Common/..."
  "${UPSTREAM}/Common"               # <THierarchicalPartition.h>, "basic_settings.h"
  "${UPSTREAM}/SparseSinkhorn"       # "kernel_manage.h", "solver.h"
)

# Upstream defines VERBOSE for C++ TUs only (the nvcc flag line carrying it
# is overwritten), so .cu TUs see the no-op eprintf — mirrored here.
target_compile_definitions(m3s_shim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:VERBOSE>)

# CUDA-12 compat polyfills (__shfl_down, cusparseDgthr — removed APIs the
# pinned upstream still uses), force-included into every CUDA TU so the
# upstream sources stay byte-untouched. See cuda12_compat.cuh.
target_compile_options(m3s_shim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${CMAKE_CURRENT_SOURCE_DIR}/cuda12_compat.cuh>)

set_target_properties(m3s_shim PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(m3s_shim PRIVATE
  CUDA::cusparse CUDA::cublas Threads::Threads)
