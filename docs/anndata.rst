AnnData interface (scanpy / squidpy)
====================================

If your data already lives in an :class:`~anndata.AnnData`, you do **not** need
to build a tidy DataFrame by hand. ``hplot`` ships a scanpy-style API that
mirrors the ``pp`` → ``tl`` → ``pl`` workflow you know from ``scanpy`` /
``squidpy``.

.. code-block:: python

   import scanpy as sc
   import squidpy as sq
   import hplot

   # 0) (optional) build a spatial neighbour graph the squidpy way
   sq.gr.spatial_neighbors(adata)          # -> adata.obsp["spatial_connectivities"]

   # 1) pp: assign every cell a signed border layer + micron distance
   hplot.pp.border_layers(adata, cluster_key="cell_type",
                          base_categories=["tumour"], sample_key="sample_id")
   #    -> adata.obs["hplot_layer"], adata.obs["hplot_distance_um"]

   # 2) tl: fit the H-Plot and stash the result in adata.uns
   hplot.tl.hplot(adata, target="CD8A", groupby="cell_subtype",
                  value_kind="expression", sample_key="sample_id")
   #    -> adata.uns["hplot"]  (h5ad-safe; survives adata.write_h5ad)

   # 3) pl: draw it (returns a matplotlib Axes)
   hplot.pl.hplot(adata)

A complete, runnable version is in ``examples/anndata_quickstart.py`` (synthetic
2-sample dataset, no real data or squidpy graph required).

Namespace mapping
-----------------

============================  ======================  =========================  ==================================
hplot call                    scanpy analogue         squidpy analogue           writes
============================  ======================  =========================  ==================================
``hplot.pp.border_layers``    ``sc.pp.neighbors``     ``sq.gr.spatial_neighbors`` ``.obs`` + ``.uns["hplot_border"]``
``hplot.tl.hplot``            ``sc.tl.umap``          ``sq.tl.var_by_distance``   ``.uns["hplot"]``
``hplot.pl.hplot``            ``sc.pl.umap``          ``sq.pl.var_by_distance``   *(draws)*
============================  ======================  =========================  ==================================

``border_layers`` lives under ``pp`` (scanpy idiom); the fit/plot live in
``tl`` / ``pl``, matching squidpy's own ``var_by_distance``.

Graph source (both, with fallback)
----------------------------------

The border layer of a cell is its signed shortest-hop distance to the
tumour/base boundary over a **spatial neighbour graph**, obtained as follows:

1. **Reuse** ``adata.obsp[connectivity_key]`` (default
   ``"spatial_connectivities"``) if it exists — whatever
   ``sq.gr.spatial_neighbors`` produced.
2. **Fallback**: otherwise build a Delaunay graph from
   ``adata.obsm[spatial_key]`` (default ``"spatial"``), pruned at ``max_edge`` µm.
3. If no graph exists **and** ``build_graph_if_missing=False``, raise instead of
   guessing.

With ``sample_key`` set, the graph is sliced per sample so hops never cross
tissues. The source actually used is recorded in
``adata.uns["hplot_border"]["graph_source"]`` (``"precomputed"`` or
``"delaunay"``).

What gets profiled
------------------

===================  ===========================  ============================  ==================
``value_kind``       ``target`` is                one curve per                 y-axis
===================  ===========================  ============================  ==================
``"expression"``     a ``var_name`` (gene in .X)  ``groupby`` category / all    mean expression
``"proportion"``     an ``.obs`` categorical      each category of ``target``   cell-type fraction
===================  ===========================  ============================  ==================

``sample_key`` marks the unit of replication: per-layer curves are averaged
across samples, so the confidence band reflects between-sample variability.
Pass ``zscore=True`` to z-score a gene per sample before aggregating.

``adata.uns["hplot"]`` layout
-----------------------------

The result is stored as a flat, **h5ad-safe** dict — no cell-type label is ever
used as a dict key, so labels containing ``/`` (e.g. ``"T/NK cells"``) round-trip
through :meth:`~anndata.AnnData.write_h5ad` cleanly::

   adata.uns["hplot"] = {
       "stats":        {group_index, layer, distance, mean, ci_lower, ci_upper, n},
       "group_order":  [labels...],       # group_index -> label
       "colors":       [hex per group],   # "" when unset
       "unit", "value_kind", "display_base_type", "display_target_type",
       "target", "legend_title",
   }

``hplot.pl.hplot(adata, key="hplot", ...)`` reconstructs the per-group curves
and forwards any extra keyword to :func:`hplot.plotting.plot_hplot`.

CSV bridge (no AnnData needed)
------------------------------

To re-plot a saved ``hplot-outputs.csv`` without any AnnData:

.. code-block:: python

   import hplot
   hplot.pl.hplot_from_csv("hplot-outputs.csv")          # returns an Axes
   stats = hplot.io.read_hplot_csv("hplot-outputs.csv")  # -> {group: DataFrame}

Column names are auto-detected (case-insensitive): ``layer``,
``distance`` / ``distance_um``, ``mean`` / ``target_type_prop`` / ``value``,
optional ``ci_lower`` / ``ci_upper`` and ``n`` / ``all_count``. Pass
``group_col=`` to split one file into multiple curves.

See :doc:`api/index` for the full parameter reference of every function, and
:doc:`troubleshooting` for common AnnData issues.
