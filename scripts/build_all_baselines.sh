#!/bin/bash
# Best-effort build of every baseline. Each step ends with a probe, and a failed
# step does not stop the others: read the OK/FAIL summary at the end.
#
# Usage: bash scripts/build_all_baselines.sh   (bash >= 4, for the associative array)
#
# What it changes:
# - the active Python environment: pip installs cupy-cuda12x, numba,
#   nvidia-ml-py and tqdm (also in requirements.txt), the build tools
#   scikit-build-core, pybind11 and cmake, and the cupdlpx binding built from
#   vendor/cupdlpx/upstream (whose --force-reinstall also reinstalls numpy and
#   scipy), all under constraints.txt, the paper's versions;
# - build outputs inside vendor/ (gitignored): the DROT shim, HALO's _build/,
#   M3S's _build/, and the fetched upstream/ trees of HALO, M3S and LAMP;
# - the Julia depot ($JULIA_DEPOT_PATH, by default ~/.julia): the packages and
#   CUDA artifacts of vendor/PDOT.jl and vendor/lamp.
# It installs no Julia and changes no juliaup setting, and it never installs
# JAX into this environment: SinkhornGrid's environment is found through
# $BDRS_OTT_PYTHON and only probed.
#
# Requirements it checks rather than installs: `julia` must start Julia 1.11
# (PDOT's and LAMP's CUDA.jl 5.11.3 rejects 1.12; under juliaup, run
# `juliaup add 1.11` and `export JULIAUP_CHANNEL=1.11` first), and M3S needs
# GCC (any version) as its host compiler (set CXX and CUDAHOSTCXX when the
# default is not GCC).
set +e
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
export PIP_CONSTRAINT="$REPO/constraints.txt"

declare -A STATUS
mark() { STATUS["$1"]="$2"; }
hr() { printf '%.0s-' {1..70}; echo; }

# CUDA.jl (PDOT, LAMP) uses its own CUDA libraries, so the Julia steps run
# without the CUDA toolkit's lib64 on LD_LIBRARY_PATH; the runs need the same.
JULIA_LD=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr : '\n' | grep -vi 'cuda.*/lib64' | paste -sd: -)
jl() { LD_LIBRARY_PATH="$JULIA_LD" "$@"; }

echo "### build_all_baselines: repo=$REPO python=$(command -v python)"; hr

echo "## [1] python deps"
# tqdm: vendor/HOT/HOT_Solver.py imports it at module level.
PYDEPS=(cupy-cuda12x numba nvidia-ml-py tqdm scikit-build-core "pybind11>=2.11" cmake)
echo "pip install into $(python -c 'import sys; print(sys.prefix)'): ${PYDEPS[*]}"
python -m pip install --quiet "${PYDEPS[@]}" 2>&1 | tail -5
python -c "import cupy, numba, pynvml, ot, tqdm" 2>/dev/null \
  && mark pydeps OK || mark pydeps FAIL
# OTT-JAX lives in its own environment (README, Installation).
if [ -z "${BDRS_OTT_PYTHON:-}" ]; then
  echo "BDRS_OTT_PYTHON is not set: create the OTT environment and point it at its python"
  mark ott "FAIL (BDRS_OTT_PYTHON not set)"
elif "$BDRS_OTT_PYTHON" -c "import ott, jax" 2>/dev/null; then
  mark ott OK
  "$BDRS_OTT_PYTHON" -c "import jax; print('jax devices:', jax.devices())" 2>&1 | tail -1
else
  mark ott "FAIL (no ott/jax under $BDRS_OTT_PYTHON)"
fi
hr

echo "## [2] DROT shim"
[ -x vendor/drot/shim/bdrs_shim ] && mark drot OK || { make -C vendor/drot/shim 2>&1 | tail -5; [ -x vendor/drot/shim/bdrs_shim ] && mark drot OK || mark drot FAIL; }
hr

echo "## [3] cuPDLPx"
bash vendor/cupdlpx/build.sh 2>&1 | tail -8
python -c "import cupdlpx" 2>/dev/null && mark cupdlpx OK || mark cupdlpx FAIL
hr

echo "## [4] HALO"
# HALO is not committed; fetch.sh clones it at PIN.
[ -d vendor/halo/upstream/src ] || bash vendor/halo/fetch.sh 2>&1 | tail -4
bash vendor/halo/build.sh 2>&1 | tail -8
PYTHONPATH="vendor/halo/_build:vendor/halo/upstream/src" \
  python -c "import pycupdlpx, hierarchical_ot" 2>/dev/null && mark halo OK || mark halo FAIL
hr

echo "## [5] M3S"
bash vendor/m3s/fetch.sh 2>&1 | tail -4
M3S_LOG=$(mktemp)
bash vendor/m3s/build.sh > "$M3S_LOG" 2>&1
tail -8 "$M3S_LOG"
if [ -x vendor/m3s/_build/m3s_shim ]; then
  mark m3s OK
elif grep -q "M3S needs GCC" "$M3S_LOG"; then
  mark m3s "FAIL (host compiler is not GCC: set CXX and CUDAHOSTCXX)"
else
  mark m3s FAIL
fi
rm -f "$M3S_LOG"
hr

echo "## [6] Julia 1.11 + PDOT"
JVER=$(jl julia --version 2>/dev/null)
case "$JVER" in
  *" 1.11."*)
    echo "$JVER"
    jl julia --project=vendor/PDOT.jl -e 'import Pkg; Pkg.instantiate()' 2>&1 | tail -8
    jl julia --project=vendor/PDOT.jl -e 'using PDOT; println("PDOT loaded")' 2>/dev/null \
      && mark pdot OK || mark pdot FAIL ;;
  *)
    echo "julia is '${JVER:-not on PATH}'; PDOT and LAMP need Julia 1.11"
    mark pdot "FAIL (julia is not 1.11)" ;;
esac
hr

echo "## [7] LAMP"
# LAMP is not committed; fetch.sh clones it at PIN.
# build.sh runs ${BDRS_LAMP_JULIA:-julia}, which must be Julia 1.11, and ends
# by loading the shim, which checks the fetched sources' sha256 and patch anchors.
bash vendor/lamp/fetch.sh 2>&1 | tail -3
LAMP_LOG=$(mktemp)
jl bash vendor/lamp/build.sh > "$LAMP_LOG" 2>&1
tail -4 "$LAMP_LOG"
if grep -q "shim OK" "$LAMP_LOG"; then
  mark lamp OK
elif grep -q "need Julia 1.11" "$LAMP_LOG"; then
  mark lamp "FAIL (julia is not 1.11; or set BDRS_LAMP_JULIA)"
else
  mark lamp FAIL
fi
rm -f "$LAMP_LOG"
hr

echo "### BUILD STATUS SUMMARY"
for k in pydeps ott drot cupdlpx halo m3s pdot lamp; do
  printf '  %-10s %s\n' "$k" "${STATUS[$k]:-UNKNOWN}"
done
hr
echo "### done"
