"""The ``random_dense`` loader: General instances generated from a seed, with
an i.i.d. random cost and Dirichlet(1) marginals."""

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.instances import instance_io as io

# N = n_g² for n_g ∈ {32, 64, 128, 256}, so the General runner's T_chk schedule
# applies at n_g = √N.
DENSE_N = (1024, 4096, 16384, 65536)
DENSE_FAMILIES = ("uniform", "absgauss")


def build_random_dense(out_dir: Path, scales: Sequence[int], n_pairs: int, seed: int) -> Dict:
    """General instances with an i.i.d. cost, which only General solvers can take.

    Only the marginals are written: every dense runner regenerates C from
    ``(N, N, cost_seed, family)``. ``scales`` is not read.
    """
    all_entries: List[Dict] = []
    for N in DENSE_N:
        for family in DENSE_FAMILIES:
            for p in range(n_pairs):
                # Additive in seed and p: pair p + 1 at seed s is pair p at
                # seed s + 101, under another id.
                mseed = seed + 4242 + 1_000_003 * DENSE_N.index(N) + 101 * p \
                    + 17 * DENSE_FAMILIES.index(family)
                rng = np.random.default_rng(mseed)
                r = rng.dirichlet(np.ones(N)); c = rng.dirichlet(np.ones(N))
                cost_seed = mseed + 777
                # The seed is in the id, so builds at different seeds never share
                # a resume key.
                ents = io.write_procedural_dense_instance(
                    "random_dense", f"N{N}_{family}_p{p}_s{seed}", N, N,
                    cost_seed, family, r, c, out_root=out_dir)
                all_entries.extend(ents)
    io.write_index("random_dense", all_entries, out_root=out_dir)
    return {"name": "random_dense", "rung": "General", "n_instances": len(all_entries),
            "data_present": True, "reason": None,
            "instance_ids": [e["instance_id"] for e in all_entries]}
