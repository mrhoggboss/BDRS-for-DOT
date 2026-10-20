"""Dense cost matrices C for the General format, normalized to max C = 1.

A lattice cost (``l1``, ``sql2``, ``l2``, ``inf``) is a closed form on the
n_g × n_g grid; a procedural cost (``rand_uniform``, ``rand_absgauss``) is
regenerated from a seed. Nothing is cached: the benchmark never reuses a cost
matrix from one solve in the next.
"""

import contextlib
import dataclasses
import shutil
import tempfile

from pathlib import Path

import numpy as np
import torch

from src.methods.grid_cmax import grid_cmax
from src.protocol.regimes import (ALL_NORMS, PROCEDURAL_NORMS,
                                           SEPARABLE_NORMS)

def is_procedural(spec) -> bool:
    """True when ``resolve_dense_cost`` regenerates the spec's cost from its seed."""
    path = getattr(spec, "cost_matrix_path", None)
    if path and Path(path).exists():
        return False
    return getattr(spec, "cost_seed", None) is not None


def lattice_cost_rows(ng: int, norm: str, i0: int, i1: int, *, device,
                      dtype=torch.float64) -> torch.Tensor:
    """Rows ``i0..i1-1`` of the max-1 lattice cost on an n_g × n_g grid
    (row-major flat index).

    Every entry is computed elementwise, so any blocking of the rows gives the
    same values."""
    if norm not in ALL_NORMS or norm in PROCEDURAL_NORMS:
        raise ValueError(f"Unknown lattice norm: {norm!r}; valid: {ALL_NORMS}")
    idx = torch.arange(ng * ng, device=device)
    a_r = (idx // ng).to(dtype)
    a_c = (idx % ng).to(dtype)
    dr = (a_r[i0:i1, None] - a_r[None, :]).abs_()
    dc = (a_c[i0:i1, None] - a_c[None, :]).abs_()
    c_max = grid_cmax(norm, ng)
    if norm == "inf":
        return torch.maximum(dr, dc).div_(c_max)
    if norm in ("sql2", "l2"):
        dr.mul_(dr)
        dc.mul_(dc)
    C = dr.add_(dc)
    if norm == "l2":
        C.sqrt_()
    return C.div_(c_max)


def build_separable_cost(ng: int, norm: str, *, device, dtype=torch.float64,
                         row_chunk: int = 2048) -> torch.Tensor:
    """``build_dense_cost`` restricted to the separable norms l1 and sql2."""
    if norm not in SEPARABLE_NORMS:
        raise ValueError(f"Not a separable norm: {norm!r}; valid: {SEPARABLE_NORMS}")
    return build_dense_cost(ng, norm, device=device, dtype=dtype, row_chunk=row_chunk)


# Procedural costs are regenerated from (m, n, cost_seed, cost_family) instead
# of stored: an fp64 C at m = n = 65536 is 34 GB.

COST_FAMILIES = ("uniform", "absgauss")
_PROC_ROW_BLOCK = 4096


def procedural_dense_cost(m: int, n: int, seed: int, family: str = "uniform",
                          *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Max-1 dense m × n cost determined by ``(m, n, seed, family)``.

    The entries are drawn from one PCG64 stream in row order, so they do not
    depend on ``_PROC_ROW_BLOCK``. They are reproducible for a fixed numpy
    version: numpy fixes the PCG64 bit stream across versions, but not the
    draws of ``Generator.uniform`` and ``Generator.standard_normal``.
    """
    if family not in COST_FAMILIES:
        raise ValueError(f"unknown cost family {family!r}; valid: {COST_FAMILIES}")
    m, n = int(m), int(n)
    if m <= 0 or n <= 0:
        raise ValueError(f"procedural_dense_cost needs positive shape, got {(m, n)}")
    rng = np.random.default_rng(int(seed))
    out = torch.empty((m, n), dtype=dtype, device=device)
    running_max = 0.0
    for i0 in range(0, m, _PROC_ROW_BLOCK):
        i1 = min(m, i0 + _PROC_ROW_BLOCK)
        if family == "uniform":
            blk = rng.uniform(0.0, 1.0, size=(i1 - i0, n))
        else:                                    # "absgauss"
            blk = np.abs(rng.standard_normal((i1 - i0, n)))
        running_max = max(running_max, float(blk.max()))
        out[i0:i1] = torch.as_tensor(blk, dtype=dtype, device=device)
    if running_max <= 0.0:
        raise ValueError("degenerate procedural cost: max <= 0")
    out /= running_max
    return out


def materialize_procedural_cost(spec, out_dir) -> str:
    """Write the procedural cost of ``spec`` to ``out_dir`` as .npy and return
    its path; the caller deletes the file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C = resolve_dense_cost(spec, device="cpu", dtype=torch.float64)
    path = out_dir / f"cost_seed{int(spec.cost_seed)}_{C.shape[0]}x{C.shape[1]}.npy"
    np.save(path, C.numpy())
    return str(path)


@contextlib.contextmanager
def materialized_cost_path(spec):
    """Yield ``spec`` with its procedural cost written to a temporary file, and
    delete the file on exit; a spec with no ``cost_seed`` or with a
    ``cost_matrix_path`` is yielded unchanged.

    DROT and PDOT take a non-lattice C only as a file. ``mkdtemp`` never
    deletes its directory, and PDOT's process lives for the whole benchmark
    run, so the cleanup must happen here.
    """
    if getattr(spec, "cost_seed", None) is None or spec.cost_matrix_path:
        yield spec
        return
    tmp = tempfile.mkdtemp(prefix="proc_cost_")
    try:
        yield dataclasses.replace(
            spec, cost_matrix_path=materialize_procedural_cost(spec, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def resolve_dense_cost(spec, device=None, dtype=torch.float64) -> torch.Tensor:
    """The cost matrix of ``spec`` in ``dtype``: its stored file, else its
    procedural cost, else its lattice cost."""
    path = getattr(spec, "cost_matrix_path", None)
    if path and Path(path).exists():
        return torch.as_tensor(np.load(path), dtype=dtype, device=device)
    seed = getattr(spec, "cost_seed", None)
    if seed is not None:
        m = int(getattr(spec, "m", 0) or 0)
        n = int(getattr(spec, "n", 0) or 0) or m
        if m <= 0:
            raise ValueError("procedural cost needs spec.m (and spec.n)")
        return procedural_dense_cost(
            m, n, int(seed), str(getattr(spec, "cost_family", "uniform") or "uniform"),
            device=device, dtype=dtype)
    return build_dense_cost(spec.resolution, spec.norm, device=device, dtype=dtype)


def build_dense_cost(ng: int, norm: str, *, device, dtype=torch.float64,
                     row_chunk: int = 2048) -> torch.Tensor:
    """Dense N × N lattice cost (N = n_g²), built ``row_chunk`` rows at a time."""
    if norm in PROCEDURAL_NORMS:
        raise ValueError(
            f"{norm!r} has no closed-form cost: a procedural instance must be "
            "resolved through resolve_dense_cost (from its cost_seed) or from "
            "a stored cost_matrix_path")
    if norm not in ALL_NORMS:
        raise ValueError(f"Unknown norm: {norm!r}; valid: {ALL_NORMS}")
    N = ng * ng
    C = torch.empty((N, N), dtype=dtype, device=device)
    for i0 in range(0, N, row_chunk):
        i1 = min(N, i0 + row_chunk)
        C[i0:i1] = lattice_cost_rows(ng, norm, i0, i1, device=device, dtype=dtype)
    return C


def dense_trace_CX(C: torch.Tensor, X: torch.Tensor, *,
                   row_chunk: int = 0) -> torch.Tensor:
    """⟨C, X⟩ as a 0-dim tensor, summed one row block at a time: ``(C * X).sum()``
    would allocate a third N × N tensor (34 GB in fp64 at n_g = 256)."""
    N = C.shape[0]
    if row_chunk <= 0:
        row_chunk = max(1, min(N, 32 * 1024 * 1024 // max(1, N)))
    obj = torch.zeros((), dtype=C.dtype, device=C.device)
    for i0 in range(0, N, row_chunk):
        i1 = min(N, i0 + row_chunk)
        obj = obj + (C[i0:i1] * X[i0:i1]).sum()
    return obj
