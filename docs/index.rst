hplot documentation
===================

**H-Plot: A graph-geodesic framework for distance-stratified spatial profiling
at tissue boundaries.**

.. image:: _static/hplot_cartoon_hires.png
   :alt: H-Plot illustration
   :width: 640px

``hplot`` converts per-cell spatial transcriptomics or digital-pathology data
into a Kaplan–Meier-style curve family that shows how tissue composition changes
with signed distance from a tissue boundary (e.g. the tumour–stroma interface).

Two ways to use hplot
---------------------

.. rubric:: 1. You already have an :class:`~anndata.AnnData` (scanpy / squidpy)

hplot assigns border layers and fits the curve directly on ``adata`` — no manual
table needed. Start with :doc:`anndata`.

.. code-block:: python

   import hplot
   hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample_id")
   hplot.tl.hplot(adata, target="CD8A", groupby="cell_subtype")
   hplot.pl.hplot(adata)

.. rubric:: 2. You have a tidy table / CSV (one row per case × layer)

Use the :class:`~hplot.HPlot` engine and CLI. Start with :doc:`engine`.

Both paths share the same statistics and plotting core; the AnnData layer is a
thin adapter on top of it.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   installation
   anndata
   engine
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
