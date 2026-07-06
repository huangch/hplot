Engine and CLI (tidy-table workflow)
====================================

When your input is a tidy table (one row per case × layer), use the
:class:`~hplot.HPlot` class and the ``hplot`` command-line interface. This is
also the reproducibility path for the paper.

The analysis is structured in **three stages** of increasing specificity:

===== ============================================================ =========================================
Stage What                                                         Function / CLI
===== ============================================================ =========================================
0     Per-layer mean ± CI curve                                    :meth:`hplot.HPlot.fit` / ``hplot plot``
1     Cluster-mass permutation test (which layer window matters?)  :func:`hplot.stats.compute_layer_pvalues` / ``hplot test``
2     H-Plot–GAM effect size + ΔH-Plot–GAM difference              :func:`hplot.stats.gam_group_curves`, :func:`hplot.stats.gam_delta_curve` / ``hplot gam``
===== ============================================================ =========================================

Stage 0 — per-layer mean ± CI
-----------------------------

.. code-block:: python

   import hplot
   hp = hplot.HPlot().fit(df, "immune_fraction", layer="layer",
                          group="hpv_status")
   ax = hp.plot(value_kind="proportion", display_target_type="immune cells")

For each group and layer the per-case fractions are summarised as mean ±
:math:`t_{\alpha/2,n-1}\cdot \mathrm{SE}` (t for :math:`n\le 30`, z otherwise).

Stage 1 — cluster-mass permutation test
---------------------------------------

Spatial biology is autocorrelated along the layer axis, so a per-layer test with
independent multiple-testing correction loses power. The cluster-mass test
treats a *contiguous run* of significant layers as the statistic and calibrates
it against a label-permutation null.

.. code-block:: python

   from hplot.stats import compute_layer_pvalues
   pvals = compute_layer_pvalues(df, prop="immune_fraction", layer_col="layer",
                                 group_col="hpv_status", groups=("HPV-", "HPV+"),
                                 correction="fdr_bh", min_n=3)

Stage 2 — GAM effect size and confounder adjustment
---------------------------------------------------

A penalised-spline GAM fits a smooth curve over the whole layer range and yields
an interpretable effect size, optionally adjusted for demographic confounders.

.. code-block:: python

   from hplot.stats import gam_group_curves, gam_delta_curve, gam_pooled_effect
   import numpy as np

   grid = np.arange(df["layer"].min(), df["layer"].max() + 1)
   curves = gam_group_curves(df, "immune_fraction", "layer", "hpv_status", grid,
                             groups=("HPV-", "HPV+"))
   diff, lo, hi, sig_pos, sig_neg = gam_delta_curve(curves, groups=("HPV-", "HPV+"))
   effect, pval, n = gam_pooled_effect(df, "immune_fraction", "layer",
                                       "hpv_status", at_layer=0,
                                       groups=("HPV-", "HPV+"))

.. note::

   Always pass the **full** layer range to the GAM functions. Fitting only on
   the Stage-1 significant window is double-dipping and inflates the effect.

Command-line interface
----------------------

.. code-block:: bash

   hplot plot -i data.csv --targets immune_fraction [--group hpv_status] [--ci]
   hplot test -i data.csv --target immune_fraction --group hpv_status \
              --permutations 999 --correction fdr_bh -o pvalues.csv
   hplot gam  -i data.csv --target immune_fraction --group hpv_status \
              --at-layer 0 [--covariates AGE late_stage is_female]

Input data format
-----------------

============  ==============  =====================================================
Column        Required        Description
============  ==============  =====================================================
``layer``     yes             Signed integer layer index; 0 = boundary, <0 outside.
target prop   yes             Fraction of the target cell type (any column name).
``group``     Stage 1/2       Binary group label (e.g. ``"HPV+"`` / ``"HPV-"``).
``distance``  no              Mean physical distance (µm) for the secondary axis.
``case_id``   no              Patient/slide id; one row per case per layer.
confounders   Stage 2 only    Columns passed to ``covariate_cols=``.
============  ==============  =====================================================

The project ``README.md`` contains the full mathematical derivation of each
stage (spline penalty, GCV smoothing, cluster-mass statistic, error
propagation) and the complete ``plot_hplot`` / GAM plotting parameter tables.
See :doc:`api/index` for the generated API reference.
