"""The Grid CUDA solver against the General solver on one DOTmark instance.

Both run a fixed number of iterations (every ε is 0, so no stop fires). Pass:
equal iteration counts and W = ⟨C, X⟩ within 1e-6 relative; the dense KKT residuals of the
CUDA triple are printed only. The solvers form the cost differently (a stored
dense C against per-axis tables) and sum the LSEs in a different order, hence
the loose bar.

Run on a CUDA machine:
    python -m tests.methods.test_bit_closeness \
        --instance CauchyDensity_n64_i0_j1_sql2 \
        --iters 200 --eta 0.3
"""

from typing import Dict
import argparse
import json
import os
import sys

import numpy as np
import torch

from tests.helpers import requires_cuda


def compute_dense_kkt(
    C: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    X: torch.Tensor,
    log_u: torch.Tensor,
    log_v: torch.Tensor,
    eta: float,
) -> Dict[str, float]:
    """Dense r_P, r_D and |g| of (X, α, β) = (X, η·log u, η·log v).

    X and C are (N, N), rows the source pixel a = a_r·n_g + a_c, as
    ``materialize_X`` in bdrs_kernels.cu lays X out.
    """
    N = r.numel()
    if X.shape != (N, N):
        raise ValueError(f"X shape mismatch: {tuple(X.shape)} vs ({N}, {N})")
    if C.shape != (N, N):
        raise ValueError(f"C shape mismatch: {tuple(C.shape)} vs ({N}, {N})")
    if log_u.numel() != N or log_v.numel() != N:
        raise ValueError(
            f"log_u, log_v lengths must equal N={N}; got "
            f"{log_u.numel()}, {log_v.numel()}"
        )

    r = r.to(torch.float64)
    c = c.to(torch.float64)
    X = X.to(torch.float64)
    C = C.to(torch.float64)

    row_sum = X.sum(dim=1)
    col_sum = X.sum(dim=0)
    primal_res = (row_sum - r).norm(2).item() + (col_sum - c).norm(2).item()

    # r_D = η·‖(log u ⊕ log v − C/η)₊‖_F
    logK = -C / float(eta)
    viol = (log_u.unsqueeze(1) + log_v.unsqueeze(0) + logK).clamp_min_(0.0)
    dual_res = (float(eta) * viol.norm()).item()

    obj = (C * X).sum().item()
    gap = abs(obj - float(eta) * (log_u.dot(r).item() + log_v.dot(c).item()))

    return {
        "primal_res_L2": float(primal_res),
        "dual_res_F": float(dual_res),
        "gap_abs": float(gap),
        "objective": float(obj),
    }


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INDEX_PATH = os.path.join(REPO_ROOT, "data", "dotmark_instances_index.json")


def load_instance(instance_id: str, device: torch.device):
    with open(INDEX_PATH, "r") as f:
        idx = {inst["instance_id"]: inst for inst in json.load(f)}
    if instance_id not in idx:
        raise KeyError(f"Instance {instance_id!r} not in {INDEX_PATH}")
    inst = idx[instance_id]
    ng = int(inst["resolution"])
    norm = inst["norm"]

    C = np.asarray(np.load(inst["cost_matrix_path"], mmap_mode="r"), dtype=np.float64)
    r = np.load(inst["marginal_i_path"]).astype(np.float64)
    c = np.load(inst["marginal_j_path"]).astype(np.float64)
    r = r / max(r.sum(), 1e-300)
    c = c / max(c.sum(), 1e-300)

    Ct = torch.as_tensor(C, dtype=torch.float64, device=device)
    rt = torch.as_tensor(r, dtype=torch.float64, device=device)
    ct = torch.as_tensor(c, dtype=torch.float64, device=device)
    return ng, norm, Ct, rt, ct


def run(instance_id: str, iters: int, eta: float) -> int:
    if not torch.cuda.is_available():
        print("[test_bit_closeness] SKIP: CUDA not available")
        return 0
    device = torch.device("cuda")

    ng, norm, C, r, c = load_instance(instance_id, device)
    print(f"[test] instance={instance_id} ng={ng} norm={norm} eta={eta} iters={iters}")

    from src.methods.factored_torch import bregman_dr_solve_general_factored
    from src.methods.params import BregmanDRParams
    params_pt = BregmanDRParams(
        eta=eta, max_iters=iters, check_every=iters + 1,  # never check
        epsilon_primal=0.0, epsilon_dual=0.0, epsilon_gap=0.0,
    )
    X_pt, info_pt = bregman_dr_solve_general_factored(C, r, c, params_pt)
    obj_pt = float((C * X_pt).sum().item())
    iters_pt = int(info_pt["iters"])
    print(f"  General: iters={iters_pt}, W={obj_pt:.10e}")

    from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
    params_cuda = BregmanDRParams(
        eta=eta, max_iters=iters, check_every=iters + 1,
        epsilon_primal=0.0, epsilon_dual=0.0, epsilon_gap=0.0,
    )
    X_cuda, info_cuda = bregman_dr_solve_grid_cuda(
        None, r, c, params_cuda,
        cost_type=norm, grid_size=ng,
    )
    obj_cuda = float((C * X_cuda).sum().item())
    iters_cuda = int(info_cuda["iters"])
    print(f"  Grid:    iters={iters_cuda}, W={obj_cuda:.10e}")

    w_rel = abs(obj_cuda - obj_pt) / max(abs(obj_pt), 1e-300)
    print(f"  |ΔW|/W = {w_rel:.3e}")

    log_u_c = info_cuda["log_u"]
    log_v_c = info_cuda["log_v"]
    kkt = compute_dense_kkt(C, r, c, X_cuda, log_u_c, log_v_c, eta)
    print(f"  Grid KKT: rP={kkt['primal_res_L2']:.3e}  "
          f"rD={kkt['dual_res_F']:.3e}  gap={kkt['gap_abs']:.3e}")

    ok_iters = (iters_pt == iters_cuda)
    ok_w = (w_rel < 1e-6)
    kkt_all = (kkt["primal_res_L2"] <= 1e-4 and
               kkt["dual_res_F"]    <= 1e-4 and
               kkt["gap_abs"]       <= 1e-4)
    ok_kkt = kkt_all  # informational: expected only after convergence

    print(f"  iters match: {ok_iters}")
    print(f"  W within rtol 1e-6: {ok_w}")
    print(f"  dense KKT all ≤ 1e-4 (converged?): {ok_kkt}")

    if ok_iters and ok_w:
        print("[test_bit_closeness] PASS (bit-closeness)")
        return 0
    else:
        print("[test_bit_closeness] FAIL")
        return 1


@requires_cuda
def test_bit_closeness_default_instance():
    """``run()`` on its default instance; skips without the DOTmark data."""
    import pytest
    if not os.path.exists(INDEX_PATH):
        pytest.skip(f"DOTmark index absent: {INDEX_PATH}")
    try:
        rc = run("CauchyDensity_n64_i0_j1_sql2", 200, 0.3)
    except (FileNotFoundError, KeyError, OSError) as exc:
        pytest.skip(f"DOTmark instance data absent: {exc}")
    assert rc == 0, "see the printed iteration counts and |dW|/W above"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="CauchyDensity_n64_i0_j1_sql2")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--eta", type=float, default=0.3)
    args = p.parse_args()
    sys.exit(run(args.instance, args.iters, args.eta))


if __name__ == "__main__":
    main()
