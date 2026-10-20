"""Building the benchmark instances, and the arrays a solver reads from a spec.

DOTmark is built from its CSVs by ``dotmark_build`` (and ``dotmark_upsample_1024``
for n_g = 1024). ModelNet10 and ``random_dense`` are built by the loaders in
``registry``, each into ``data/<dataset>/`` with its own ``index.json``.
``dense_cost`` gives the General runners their cost, and ``dot_lp`` gives the
cuPDLPx runner its constraint matrix.
"""
