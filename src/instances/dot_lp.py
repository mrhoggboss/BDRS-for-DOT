"""The marginal constraint operator of the dense DOT linear program, used by the
cuPDLPx runner.

With the plan flattened row-major, ``x[i*nn + j] = X[i, j]``, the marginal
constraints ``X 1 = r`` and ``Xᵀ 1 = c`` are ``A x = (r; c)`` with the
``(2nn × nn²)`` 0/1 operator built here.
"""

import numpy as np
import scipy.sparse as sp


def marginal_csr(nn: int, signed: bool = False) -> "sp.csr_matrix":
    """The marginal operator A as a CSR matrix. With ``signed=True`` the row
    block is negated, giving the node-arc incidence form (one −1 and one +1 per
    column), and the caller must pass the right-hand side (−r; c).
    """
    n2 = nn * nn
    dt = np.int64 if 2 * n2 > 2 ** 31 - 1 else np.int32
    indices = np.empty(2 * n2, dtype=dt)
    indices[:n2] = np.arange(n2, dtype=dt)                       # row block
    indices[n2:] = ((np.arange(nn, dtype=dt) * nn)[None, :]
                    + np.arange(nn, dtype=dt)[:, None]).ravel()  # column block
    indptr = np.arange(2 * nn + 1, dtype=dt) * nn
    data = np.ones(2 * n2)
    if signed:
        data[:n2] = -1.0
    return sp.csr_matrix((data, indices, indptr), shape=(2 * nn, n2))
