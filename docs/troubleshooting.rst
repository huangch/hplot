Troubleshooting
===============

AnnData interface
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Likely cause / fix
   * - ``ImportError: ... pip install 'hplot[...]'``
     - Install the optional extra: ``pip install "hplot[anndata]"``.
   * - All ``hplot_layer`` are NaN
     - No base cells matched — check ``cluster_key`` values and
       ``base_categories`` spelling/case.
   * - Only NaN for one sample (plus a warning)
     - That sample had < 4 cells or a degenerate (collinear) layout; other
       samples are unaffected.
   * - Border sits in the wrong place
     - Tune ``n_min`` / ``ratio`` (region call) and ``max_edge`` (Delaunay
       pruning), or supply a squidpy graph.
   * - Layers cross between tissues
     - Pass ``sample_key=`` so the graph is computed per sample.
   * - ``KeyError: adata.obs['hplot_layer']``
     - Run ``pp.border_layers`` before ``tl.hplot``.
   * - ``write_h5ad`` fails on ``.uns``
     - Use ``tl.hplot`` to populate ``.uns["hplot"]`` (it is h5ad-safe); do not
       stash raw ``HPlot`` objects there.
   * - Wrong quantity plotted
     - ``value_kind="expression"`` needs a gene in ``.X``;
       ``value_kind="proportion"`` needs an ``.obs`` categorical column.
   * - ``ValueError: var_names are not unique``
     - Call ``adata.var_names_make_unique()`` before selecting a gene.

Engine / CLI
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Likely cause / fix
   * - ``RuntimeError: Call fit() before plot()``
     - ``HPlot.plot`` was called before ``HPlot.fit``.
   * - ``plot(pvalue_show=True)`` errors
     - Requires ``fit(..., pvalue=True)`` first.
   * - ``plot(band='auto')`` errors
     - Requires ``fit(..., pvalue=True)`` first.
   * - GAM functions raise about ``pygam``
     - Install the hard dependency: ``pip install pygam``.
   * - Effect size looks inflated
     - Do not fit the GAM only on the Stage-1 window — pass the full layer
       range (double-dipping guard).

