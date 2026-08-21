# hplot

**H-Plot: A graph-geodesic framework for distance-stratified spatial profiling
at tissue boundaries**

![H-Plot illustration](docs/_static/hplot_cartoon_hires.png)

`hplot` converts per-cell spatial transcriptomics or digital-pathology data into
a Kaplan–Meier-style curve family that shows how tissue composition changes with
signed distance from a tissue boundary (e.g. the tumour–stroma interface).

> **Three ways to use hplot — pick one:**
>
> - **You have an `AnnData`** (scanpy/squidpy user) → jump to
>   [AnnData interface](#anndata-interface-scanpy--squidpy-users). hplot assigns
>   border layers and fits the curve directly on `adata`, no manual table needed.
> - **You have a tidy table / CSV** (one row per case × layer) → use the
>   [`HPlot` engine and CLI](#conceptual-background) documented below. This is
>   also the reproducibility path for the paper.
> - **You want pathway/signature-level profiling** → use the
>   [H-Pathway Summary](#h-pathway-summary) workflow with UCell scoring and
>   signature catalogs (MSigDB, GO).
>
> All three share the same statistics and plotting core; the AnnData and
> pathway layers are thin adapters on top of it.

The analysis is structured in **three stages** with increasing specificity:

| Stage | What | Function / CLI |
|-------|------|----------------|
| 0 | Per-layer mean ± CI curve | `HPlot.fit()` / `hplot plot` |
| 1 | Cluster-mass permutation test — which layer window is significant? | `compute_layer_pvalues()` / `hplot test --permutations` |
| 2 | H-Plot–GAM effect size (smooth per-group curves + ΔH-Plot–GAM difference) — how large is the effect, and is it a demographic confound? | `gam_group_curves()`, `gam_delta_curve()`, `gam_pooled_effect()` / `hplot gam` |

---

## Installation

```bash
pip install -e .
```

**Hard dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`, **`pygam`**

**Optional extra** — `squidpy` is never imported by hplot; install it only if you
want to build spatial graphs yourself before calling `pp.border_layers`
(§ *AnnData interface* below). `anndata` is a core dependency and needs no extra:

```bash
pip install -e ".[squidpy]"    # adds squidpy (>=1.2) for spatial graphs
```

```bash
# Docker (for paper reproducibility — no local Python setup needed)
docker build -t hplot .
docker run --rm -v "$PWD":/data hplot test -i /data/data.csv \
    --target immune_fraction --group hpv_status --permutations 999
```

---

## AnnData interface (scanpy / squidpy users)

If your data already lives in an `AnnData`, you do **not** need to build a tidy
DataFrame by hand. `hplot` ships a scanpy-style API that mirrors the
`pp` → `tl` → `pl` workflow you already know from `scanpy`/`squidpy`, so the
learning curve is essentially zero:

```python
import scanpy as sc
import squidpy as sq
import hplot

# 0) (optional) build a spatial neighbour graph the squidpy way
sq.gr.spatial_neighbors(adata)                 # -> adata.obsp["spatial_connectivities"]

# 1) pp: assign every cell a signed border layer + micron distance
hplot.pp.border_layers(adata, cluster_key="cell_type", base_categories=["tumour"],
                       sample_key="sample_id")
#    -> adata.obs["hplot_layer"], adata.obs["hplot_distance_um"]

# 2) tl: fit the H-Plot and stash the result in adata.uns
hplot.tl.hplot(adata, target="CD8A", groupby="cell_subtype",
               value_kind="expression", sample_key="sample_id")
#    -> adata.uns["hplot"]  (h5ad-safe; survives adata.write_h5ad)

# 3) pl: draw it (returns a matplotlib Axes)
hplot.pl.hplot(adata)
```

### Namespace mapping

| hplot call | scanpy analogue | squidpy analogue | writes |
|------------|-----------------|------------------|--------|
| `hplot.pp.border_layers` | `sc.pp.neighbors` | `sq.gr.spatial_neighbors` | `.obs` + `.uns["hplot_border"]` |
| `hplot.tl.hplot` | `sc.tl.umap` | `sq.tl.var_by_distance` | `.uns["hplot"]` |
| `hplot.pl.hplot` | `sc.pl.umap` | `sq.pl.var_by_distance` | *(draws)* |

`border_layers` lives under `pp` (scanpy idiom: "preprocess my cells"). The
fit/plot live in `tl`/`pl`, matching squidpy's own `var_by_distance`.

> **Runnable example:** [`examples/anndata_quickstart.py`](examples/anndata_quickstart.py)
> is a self-contained script that builds a synthetic 2-sample `AnnData`, runs the
> full `pp` → `tl` → `pl` workflow, saves a two-panel figure, and demonstrates
> the `write_h5ad` round-trip — no real data or squidpy graph required. Run it
> with `python examples/anndata_quickstart.py`.

### `pp.border_layers` — graph source (both, with fallback)

The border layer of a cell is its signed shortest-hop distance to the
tumour/base boundary over a **spatial neighbour graph**. That graph is obtained
as follows:

1. **Reuse** `adata.obsp[connectivity_key]` (default `"spatial_connectivities"`)
   if it exists — i.e. whatever `sq.gr.spatial_neighbors` produced.
2. **Fallback**: otherwise build a Delaunay graph from
   `adata.obsm[spatial_key]` (default `"spatial"`), pruned at `max_edge` µm.
3. If no graph exists **and** `build_graph_if_missing=False`, raise instead of
   guessing.

With `sample_key` set, the graph is sliced per sample so hops never cross
tissues. The source actually used is recorded in
`adata.uns["hplot_border"]["graph_source"]` (`"precomputed"` or `"delaunay"`).

### `tl.hplot` — what gets profiled

| `value_kind` | `target` is | one curve per | y-axis |
|--------------|-------------|---------------|--------|
| `"expression"` (default) | a `var_name` (gene in `.X`) | `groupby` category (or all cells) | mean expression |
| `"proportion"` | an `.obs` categorical column | each category of `target` | cell-type fraction |

`sample_key` marks the unit of replication: per-layer curves are averaged
across samples, so the confidence band reflects between-sample variability
(single-sample data gives the raw per-layer curve with a degenerate band).
Pass `zscore=True` to z-score a gene per sample before aggregating.

### `adata.uns["hplot"]` layout

The result is stored as a flat, **h5ad-safe** dict — no cell-type label is ever
used as a dict key, so labels containing `/` (e.g. `"T/NK cells"`) round-trip
through `write_h5ad` cleanly:

```text
adata.uns["hplot"] = {
    "stats":        {group_index, layer, distance, mean, ci_lower, ci_upper, n},  # flat arrays
    "group_order":  [labels...],          # group_index -> label
    "colors":       [hex per group],      # "" when unset
    "unit", "value_kind", "display_base_type", "display_target_type",
    "target", "legend_title",
}
```

`hplot.pl.hplot(adata, key="hplot", ...)` reconstructs the per-group curves from
this and forwards any extra keyword to `plot_hplot` (e.g. `ci_show=False`).

### CSV bridge (no AnnData needed)

To re-plot a saved `hplot-outputs.csv` without any AnnData:

```python
import hplot
hplot.pl.hplot_from_csv("hplot-outputs.csv")           # returns an Axes
stats = hplot.io.read_hplot_csv("hplot-outputs.csv")   # -> {group: DataFrame}
```

Column names are auto-detected (case-insensitive): `layer`, `distance`/
`distance_um`, `mean`/`target_type_prop`/`value`, optional `ci_lower`/`ci_upper`
and `n`/`all_count`. Pass `group_col=` to split one file into multiple curves.

### `pp.border_layers()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adata` | — | The `AnnData` (modified in place unless `copy=True`). |
| `cluster_key` | — | `.obs` column with cell compartments. |
| `base_categories` | — | Value(s) of `cluster_key` forming the base/tumour region (str or list). |
| `spatial_key` | `"spatial"` | `.obsm` key with cell coordinates (µm). |
| `connectivity_key` | `"spatial_connectivities"` | `.obsp` key of a precomputed graph to reuse (squidpy). |
| `sample_key` | `None` | `.obs` column of independent tissues; graph is computed per sample. |
| `k` | `2` | Neighbourhood radius (hops) used to call the base *region*. |
| `n_min` | `10` | Minimum k-hop neighbourhood size for a cell to seed a region. |
| `ratio` | `0.2` | Minimum base fraction within the neighbourhood to be "region". |
| `max_edge` | `25.0` | Delaunay edge-length cap (µm); ignored when a graph is reused. |
| `build_graph_if_missing` | `True` | Build Delaunay when no `.obsp` graph exists; else raise. |
| `layer_key` / `distance_key` | `"hplot_layer"` / `"hplot_distance_um"` | `.obs` columns written. |
| `copy` | `False` | Return a modified copy instead of writing in place. |

Writes `.obs[layer_key]` (signed hops, NaN where unreachable),
`.obs[distance_key]` (signed µm), and `.uns["hplot_border"]` (run parameters +
`graph_source`).

### `tl.hplot()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adata` | — | Must already carry `layer_key` from `pp.border_layers`. |
| `target` | — | A `var_name` (gene) for expression; an `.obs` column for proportion. |
| `value_kind` | `"expression"` | `"expression"` / `"interaction"` (gene in `.X`) or `"proportion"` / `"fraction"` (`.obs` category). |
| `groupby` | `None` | `.obs` column → one curve per category (expression modes). |
| `sample_key` | `None` | Replicate column; curves averaged across samples per layer. |
| `exclude_base` | `False` | Proportion modes only: divide the target count by **non-base** cells (`all − base`) instead of all cells, using the `cluster_key`/`base_categories` recorded by `pp.border_layers`. Base categories are dropped from the curve set. |
| `min_base_excluded_count` | `1` | With `exclude_base`, layers with fewer non-base cells yield `NaN`. |
| `zscore` | `False` | Z-score the gene per sample before aggregating (expression modes). |
| `smoother` | `"mean"` | `"mean"` (per-layer average) or `"gam"` (penalised smooth). |
| `layer_key` / `distance_key` | `"hplot_layer"` / `"hplot_distance_um"` | `.obs` columns read. |
| `color_map` | `None` | `{group: colour}`; stored as hex, reused at plot time. |
| `display_base_type` / `display_target_type` | `"tumour border"` / `target` | Title / y-axis phrasing. |
| `legend_title` | auto | Legend heading (defaults to `target` or `groupby`). |
| `key_added` | `"hplot"` | `.uns` key to write the serialised result to. |
| `copy` | `False` | Operate on and return a copy. |

Extra keyword arguments are forwarded to `HPlot.fit()`.

### `pl.hplot()` / `pl.hplot_from_csv()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adata` / `path` | — | Source: the `AnnData` (reads `.uns[key]`) or a CSV path. |
| `key` | `"hplot"` | (`pl.hplot`) `.uns` key produced by `tl.hplot`. |
| `group_col` | `None` | (`hplot_from_csv`) column that splits the CSV into curves. |
| `ax` | `None` | Existing axis to draw into; a new one is created if omitted. |
| `ci_show` | `True` | Draw the confidence band. |
| `unit` / `value_kind` / `display_*` | from `.uns` (or args for CSV) | Override stored plot settings. |

Both return a matplotlib `Axes`. Any extra keyword is forwarded to
`plot_hplot()` (e.g. `pvalue_show`, `band`, `legend_kwargs`).

### AnnData troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `ImportError: ... needs anndata` | `anndata` is a core dependency, so this means the install is incomplete: `pip install "anndata>=0.11,<0.13"`. |
| All `hplot_layer` are NaN | No base cells matched — check `cluster_key` values and `base_categories` spelling/case. |
| Only NaN for one sample + a warning | That sample had `< 4` cells or a degenerate (collinear) layout; other samples are fine. |
| Border sits in the wrong place | Tune `n_min` / `ratio` (region call) and `max_edge` (Delaunay pruning), or supply a squidpy graph via `sq.gr.spatial_neighbors`. |
| Layers cross between tissues | Pass `sample_key=` so the graph is computed per sample. |
| `KeyError: adata.obs['hplot_layer'] missing` | Run `pp.border_layers` before `tl.hplot`. |
| `write_h5ad` fails on `.uns` | Use `tl.hplot` to populate `.uns["hplot"]` (it is h5ad-safe); don't stash raw `HPlot` objects there. |
| Wrong quantity plotted | `value_kind="expression"` needs a gene in `.X`; `value_kind="proportion"` needs an `.obs` categorical column. |

> The `pp`/`tl`/`pl`/`gr` modules import `anndata` **lazily** (inside the
> functions), so `import hplot` and `import hplot.core` stay cheap even though
> `anndata` ships as a core dependency.

---

## Conceptual background

### Graph-geodesic layer definition

Each cell in a tissue is assigned a signed integer *layer index* l ∈ ℤ based on
the shortest graph-geodesic path to the boundary of interest:

- **l = 0**: cells at the boundary itself (e.g. first tumour-adjacent layer)
- **l > 0**: cells *inside* the tumour, in ascending hops
- **l < 0**: cells *outside* the tumour (stromal side)

Within a layer, the fraction of a target cell type is averaged across all cells
belonging to that patient/slide.  Stacking these per-patient fractions across a
cohort and computing the mean ± CI at each layer produces the H-Plot curve.

### Stage 0 — per-layer mean ± CI

For each group g and layer l, the aggregated fraction y_{ig,l} over patients
i = 1..n_{g,l} is summarised as:

- **Mean**: μ_{g,l} = (1/n) Σ_i y_{il}
- **Standard error**: SE_{g,l} = s_{g,l} / √n
- **CI bounds**: μ ± t_{α/2, n-1} · SE (t-distribution for n ≤ 30,
  z-distribution for n > 30)

---

## Stage 1 — Cluster-mass permutation test

### Motivation

A naïve approach would apply a Mann-Whitney U test at each layer and correct for
multiple testing with Bonferroni or FDR.  But spatial biology is *autocorrelated*
along the layer axis: if layer 3 is significantly enriched for immune cells,
layers 2 and 4 usually are too.  Correcting independently ignores this structure
and loses power.

The cluster-mass (or "cluster-forming") test accounts for spatial autocorrelation
by treating a *contiguous run* of significant layers as the test statistic.

### Algorithm

**Per-layer test statistic**

At each layer l with at least `min_n` cases per group, compute the Mann-Whitney U
rank sum:

```
W_l = U(y_{1,l}, y_{2,l})
```

The p-value p_l is derived from the asymptotic normal approximation of U.

**Cluster-mass statistic T_obs**

A layer is "cluster-forming" if p_l < α_cluster (default 0.05).  Maximal runs of
consecutive cluster-forming layers are identified.  The cluster-mass of a run
[l_start, l_end] is:

```
T = Σ_{l=l_start}^{l_end}  (α_cluster - p_l)
```

(Equivalently, the sum of "excess significance" over the cluster.)  Runs shorter
than `band_min_width` are discarded as noise spikes.

**Permutation null distribution**

Under H₀ (no spatial group difference), group labels are permuted n_perm times.
For each permutation the whole per-layer testing + cluster-mass computation is
repeated, yielding T_{null}^{(b)}, b = 1..n_perm.

**Permutation p-value**

```
p_perm = #{b : T_{null}^{(b)} >= T_obs} / n_perm
```

A conventional threshold of p_perm < 0.05 is used.  Because the null is
constructed empirically, no parametric assumption about the T distribution is
needed.

### Usage

```python
from hplot.stats import compute_layer_pvalues

pvals = compute_layer_pvalues(
    df,
    prop="immune_fraction",
    layer_col="layer",
    group_col="hpv_status",
    groups=("HPV-", "HPV+"),   # (low, high) — order matters for effect sign
    test="mannwhitney",
    correction="fdr_bh",       # optional FDR correction across layers
    min_n=3,
)
# columns: layer, distance, p_value, p_adj, stat, n1, n2
```

CLI:

```bash
hplot test -i data.csv --target immune_fraction --group hpv_status \
    --groups "HPV-" "HPV+" --permutations 999 --correction fdr_bh \
    -o pvalues.csv
```

---

## Stage 2 — GAM effect size and confounder adjustment

### Why GAM after Stage 1?

Stage 1 answers **"is there a significant spatial difference?"** but not:

- **How large is it?** (Mann-Whitney U gives a rank-based test statistic, not
  an effect size in natural units)
- **Is it caused by demographic differences?** (e.g. HPV+ patients are younger
  — is the immune enrichment actually driven by age, not HPV status?)

A Generalised Additive Model (GAM) fits a smooth non-parametric curve over the
whole layer range and provides an interpretable marginal effect size, with or
without adjusting for confounders.

### Mathematical model

The pooled model is:

```
y_{il} = f(l) + β_g · g_i + Σ_k β_k · x_{ik} + ε_{il}
```

where:

- `y_{il}` is the target fraction for patient i at layer l
- `f(l) = B(l)^T α` is a penalised B-spline smooth:
  - **B(l)** is the K-dimensional B-spline basis at layer l (K = n_splines = 10 by default)
  - **α** are the spline coefficients estimated by penalised least squares
- `β_g` is the linear group effect (g_i ∈ {0, 1})
- `β_k` are linear effects of optional confounders x_{ik}
  (z-scored internally: μ=0, σ=1)
- `ε_{il} ~ N(0, σ²)` is residual error

### Penalised estimation

The spline coefficients are estimated by:

```
α* = argmin_α  ||y - B α||² + λ · ||D² α||²
```

where **D²** is the second-difference matrix that penalises curvature, and λ is
the smoothing parameter.

### Smoothing parameter selection (GCV)

λ is chosen by Generalised Cross-Validation (GCV):

```
λ* = argmin_λ  RSS(λ) / [n · (1 - trace(H_λ)/n)]²
```

where H_λ = B (B^T B + λ D^T D)^{-1} B^T is the hat matrix.  GCV avoids
over-smoothing (underfitting biology) and under-smoothing (fitting noise).

### Effect size

The high-minus-low group effect at layer l₀ (the Stage-1 peak) is:

```
Δ = f̂(l₀, g=1, x=x̄) - f̂(l₀, g=0, x=x̄)
```

Covariates are evaluated at their mean (0 after z-scoring), so Δ is the marginal
group contrast at a "typical" patient.  If Δ is essentially the same with and
without covariate adjustment, the Stage-1 signal is not a demographic confound.

### Confidence intervals (per-group curves)

`gam_group_curves()` fits separate `target ~ s(layer)` models per group and
returns 95 % pointwise CIs:

```
CI(l) = f̂(l) ± 1.96 · SE_f̂(l)
```

where `SE_f̂(l) = sqrt[B(l)^T (B^T B + λ D^T D)^{-1} B(l) · σ̂²]`.

### Differential curve — ΔH-Plot–GAM

The per-group smooths from `gam_group_curves()` are the **H-Plot–GAM** view
(one penalised curve per group, drawn on top of the raw layer means). Their
layer-wise difference is the **differential H-Plot–GAM** (ΔH-Plot–GAM):

```
Δ(l) = f̂_high(l) − f̂_low(l)
```

`gam_delta_curve()` computes this together with a CI propagated in quadrature
(Gaussian error propagation, assuming the two group models are independent):

```
σ_Δ(l) = sqrt[ σ_high(l)² + σ_low(l)² ]      CI(l) = Δ(l) ± σ_Δ(l)
```

and two pointwise significance masks: `sig_pos` where the CI excludes 0 from
below (high group larger) and `sig_neg` where it excludes 0 from above (low
group larger).

> **Pointwise, not corrected.** `sig_pos` / `sig_neg` reflect layer-by-layer CI
> exclusion of zero and are **not** corrected for multiple comparisons across
> layers. Use the ΔH-Plot–GAM colouring for visualisation and localisation
> only; use the Stage-1 cluster-mass permutation test (with FDR) for
> confirmatory inference.

### Stage-1 double-dipping guard

**Always pass the full layer range to GAM functions.**  Fitting the GAM only on
layers selected by Stage 1 (the cluster-mass band) constitutes double-dipping:
the outcome and the analysis window are no longer independent, inflating the
apparent effect.  The correct workflow is:

1. Stage 1 identifies *which* layers are significant (whole-range test).
2. Stage 2 fits the model on the *whole* range and only *reads* the effect at
   the Stage-1 peak layer.

### Python API

```python
from hplot.stats import gam_group_curves, gam_pooled_effect, gam_delta_curve
import numpy as np

grid = np.arange(df["layer"].min(), df["layer"].max() + 1)

# Per-group smooth curves — the H-Plot–GAM view (for plotting)
curves = gam_group_curves(
    long_df=df,
    target_col="immune_fraction",
    layer_col="layer",
    group_col="hpv_status",
    grid=grid,
    groups=("HPV-", "HPV+"),
)
# curves["HPV+"] -> (pred_array shape=(G,), ci_array shape=(G,2))

# Difference curve — the ΔH-Plot–GAM view
diff, ci_lo, ci_hi, sig_pos, sig_neg = gam_delta_curve(
    curves, groups=("HPV-", "HPV+"),   # (low, high) -> Δ = high − low
)

# Pooled effect at the tumour border (layer 0), unadjusted
effect, pval, n = gam_pooled_effect(
    long_df=df, target_col="immune_fraction", layer_col="layer",
    group_col="hpv_status", at_layer=0,
    groups=("HPV-", "HPV+"),
)

# Confounder-adjusted (age, clinical stage, sex)
effect_adj, pval_adj, n_adj = gam_pooled_effect(
    long_df=df, target_col="immune_fraction", layer_col="layer",
    group_col="hpv_status", at_layer=0,
    groups=("HPV-", "HPV+"),
    covariate_cols=["AGE", "late_stage", "is_female"],
)

print(f"Unadjusted : Δ={effect:+.3f}  p={pval:.2e}  n={n}")
print(f"Adjusted   : Δ={effect_adj:+.3f}  p={pval_adj:.2e}  n={n_adj}")
```

CLI:

```bash
# Unadjusted effect
hplot gam -i data.csv --target immune_fraction --group hpv_status \
    --groups "HPV-" "HPV+" --at-layer 0 \
    --curves-output gam_curves.csv

# Confounder-adjusted
hplot gam -i data.csv --target immune_fraction --group hpv_status \
    --groups "HPV-" "HPV+" --at-layer 0 \
    --covariates AGE late_stage is_female
```

---

## Putting it all together

### One-stop H-Plot–GAM via the `HPlot` class

`HPlot.fit(smoother="gam")` fits the per-group GAM smooths and renders them as
the H-Plot–GAM top panel; `HPlot.plot_delta()` draws the matching ΔH-Plot–GAM
difference panel below it:

```python
from hplot import HPlot
import matplotlib.pyplot as plt

fig, (ax_top, ax_delta) = plt.subplots(
    2, 1, figsize=(6, 5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.4]},
)

hp = (
    HPlot()
    .fit(df, "immune_fraction", layer="layer", group="hpv_status",
         smoother="gam", gam_group_order=["HPV-", "HPV+"],
         color_map={"HPV+": "#d62728", "HPV-": "#1f77b4"})
)

# Top: H-Plot–GAM (per-group smooths ± 95 % CI)
hp.plot(ax=ax_top, value_kind="proportion", display_target_type="immune cells")

# Bottom: ΔH-Plot–GAM (high − low, coloured where the pointwise CI excludes 0)
hp.plot_delta(ax=ax_delta, ref_band=(band_lo, band_hi), ref_peak=peak_layer)
```

`ref_band` / `ref_peak` are optional reference markers (e.g. the Stage-1
cluster-mass band and peak); they come from a *different* method and are drawn
lightly, not implied to coincide with the coloured region.

For a bare axis without the `HPlot` wrapper, use the two functional plotters
directly with the `gam_group_curves()` / `gam_delta_curve()` output — this is
the grid-friendly path used for dense multi-panel figures:

```python
from hplot import plot_hplot_gam, plot_hplot_gam_delta
from hplot.stats import gam_group_curves, gam_delta_curve
import numpy as np

grid = np.linspace(-7, 14, 200)
curves = gam_group_curves(df, "immune_fraction", "layer", "hpv_status", grid,
                          groups=("HPV-", "HPV+"))

# Top: H-Plot–GAM (per-group smooths ± CI)
plot_hplot_gam(
    grid, curves, ax=ax_top, group_labels=["HPV-", "HPV+"],
    color_map={"HPV+": "#d62728", "HPV-": "#1f77b4"},
    ref_band=(band_lo, band_hi), ylabel="immune fraction",
)

# Bottom: ΔH-Plot–GAM (high − low)
plot_hplot_gam_delta(
    grid, *gam_delta_curve(curves, groups=("HPV-", "HPV+")),
    ax=ax_delta, group_labels=("HPV-", "HPV+"),
    high_color="#d62728", low_color="#1f77b4",
)
```

### Single-panel overlay via `plot_hplot()`

All three stages can also be layered into one panel:

```python
import hplot
from hplot.stats import compute_layer_pvalues, gam_group_curves
from hplot.plotting import plot_hplot
import numpy as np, matplotlib.pyplot as plt

# Stage 0
hp = hplot.HPlot().fit(df, "immune_fraction", layer="layer", group="hpv_status")

# Stage 1
pvals = compute_layer_pvalues(df, prop="immune_fraction",
                               layer_col="layer", group_col="hpv_status",
                               correction="fdr_bh")

# Stage 2 — curves for overlay
grid = np.arange(df["layer"].min(), df["layer"].max() + 1)
gam_curves = gam_group_curves(df, "immune_fraction", "layer", "hpv_status", grid)

# Single figure with all three layers
fig, ax = plt.subplots(figsize=(9, 4))
plot_hplot(
    hp.target_grouped_stats_,
    ax=ax,
    pvalue_stats=pvals,            # Stage 1: p-value track (right y-axis)
    pvalue_show=True,
    pvalue_use_adjusted=True,
    band=(band_lo, band_hi),       # Stage 1: cluster-mass significant band
    band_label="cluster-mass p<0.05",
    gam_curves=gam_curves,         # Stage 2: GAM smooth overlay (dashed)
    gam_curves_label_suffix=" (GAM)",
)
```

### `plot_hplot()` GAM parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gam_curves` | `None` | Dict from `gam_group_curves()`: `{group: (pred, ci)}`. |
| `gam_curves_ci_show` | `True` | Shade the GAM 95 % pointwise CI band. |
| `gam_curves_linestyle` | `"--"` | Line style for the GAM smooth. |
| `gam_curves_linewidth` | `1.8` | Line width. |
| `gam_curves_ci_alpha` | `0.10` | Opacity of the GAM CI shading. |
| `gam_curves_grid` | `None` | X-coordinates aligned with the prediction arrays. |
| `gam_curves_label_suffix` | `" (GAM)"` | Appended to the group label in the legend. |

### `plot_hplot_gam()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid` | — | Layer coordinates the smooths are evaluated on. |
| `curves` | — | `{group: (pred, ci)}` from `gam_group_curves()`. |
| `group_labels` | `None` | Order/subset of `curves` keys to draw. |
| `color_map` | `None` | `{group: colour}`; falls back to `palette`. |
| `ci_show` / `ci_alpha` | `True` / `0.18` | Show / opacity of the pointwise CI band. |
| `zero_line` | `True` | Dashed vertical reference at layer 0. |
| `ref_band` / `ref_peak` | `None` | Reference span / x-position (e.g. Stage-1 band / peak). |

### `plot_hplot_gam_delta()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid` | — | Layer coordinates aligned with the difference arrays. |
| `diff_pred`, `ci_lower`, `ci_upper` | — | Δ estimate and propagated CI from `gam_delta_curve()`. |
| `sig_pos`, `sig_neg` | — | Pointwise significance masks (high>low / low>high). |
| `group_labels` | `("low", "high")` | `(low, high)` labels used in the legend. |
| `high_color` / `low_color` | red / blue | Fill colours for the significant regions. |
| `ref_band` | `None` | `(lo, hi)` reference span (e.g. Stage-1 band). |
| `ref_peak` | `None` | Reference x-position (e.g. Stage-1 peak). |
| `show_sig_pct` | `True` | Annotate the pointwise-significant fraction of the range. |

---

## H-Loci Summary panels

A Stage-1 screen run across **many features** (genes, ligand→receptor pairs,
cell-type fractions) yields, per feature, a scored *cluster band* — the run of
border layers where the value departs from baseline — with a direction
(up / down), a centre (peak layer), and a magnitude (cluster mass). The
**H-Loci Summary** family renders those bands as compact, stackable panels so a
whole screen is readable at a glance. Feed them the output columns of
`gradient_cluster_mass_screen()` / `directional_cluster_bands()`.

Three renderings share the same signed border-layer x-axis (0 = boundary,
`< 0` inside the tissue, `> 0` in the stroma):

| Function | One row shows | Best for |
|----------|---------------|----------|
| `plot_hloci_strip()` | a **strip** planted at the band position; thickness = cluster mass, ▲ / ▼ = direction | dense overviews of hundreds of features |
| `plot_hloci_bands()` | a **horizontal bar** spanning `[band_start_layer, band_end_layer]`, filled by direction, with a peak tick | showing *where* and *how wide* each band sits |
| `plot_hloci_bands_bidir()` | **two bars** per row (elevated + depressed), each with a centre tick | features with a band on both sides of the front |
| `plot_hloci_fdr()` | a two-panel figure: band spans on the left, `-log10(FDR)` on the right | ranking a whole screen by significance |
| `plot_hloci_dotplot()` | a (feature × layer) dotplot; size = score, colour = direction | per-layer detail for a feature subset |

Direction is colour-coded by `up_color` (default red `#d62728`) and `down_color`
(default blue `#1f77b4`). These names are deliberately **direction-neutral** —
"up" = above baseline, "down" = below — so the same panel serves an *intensity*
readout (elevated / depressed expression or interaction score) **and** a
*compositional* one (enriched / depleted cell fraction); the caption supplies
the domain wording. Row ordering is controlled by `sort=` (keyed on the peak
centre of cluster mass); pass `sort=None` to keep the caller's order.

```python
import hplot

# `screen_df` is gradient_cluster_mass_screen(...)["long"]
rank = screen_df[screen_df["peak_layer"].notna()].sort_values("peak_layer")

ax = hplot.plot_hloci_bands(
    rank["band_start_layer"], rank["band_end_layer"], rank["direction"],
    peak=rank["peak_layer"], labels=rank["gene"],
    sort=None,                       # keep caller order (already sorted)
    xlabel="border layer L",
)
```

### Physical-distance x-axis

The band x-axis is in integer **layer** units `L`. To also show the physical
distance from the border (µm), rescale the axis with the two helpers instead of
hand-rolling a twin axis:

```python
# map each layer L -> mean distance (µm), pooled across slides
layer2um = hplot.build_layer_distance_map(
    [(res[sid]["layers"], res[sid]["distances"]) for sid in sample_ids])

# bottom axis -> µm, twin top axis kept in layer L (x-limits preserved)
hplot.add_border_distance_axis(ax, layer2um)
```

`build_layer_distance_map()` also accepts two flat aligned arrays
(`layers`, `distances`). `add_border_distance_axis()` returns the twin top axis
(or `None` when `add_top_axis=False`) and leaves the band bars aligned because
it never changes the x-limits.

| Helper | Purpose |
|--------|---------|
| `build_layer_distance_map(layers, distances=None)` | Average physical distance (µm) per signed layer `L`. Accepts two flat arrays, or one iterable of per-slide `(layers, distances)` pairs. |
| `add_border_distance_axis(ax, layer_to_distance, *, max_ticks=9, add_top_axis=True, …)` | Re-label a band panel's bottom axis in µm and add a twin top axis in layer `L`. |

---

## H-Pathway Summary

The H-Pathway Summary extends the single-target H-Plot to **pathway/signature-level
profiling**. Instead of tracking one gene or cell type across layers, it profiles
an entire gene-set catalog (e.g. MSigDB Hallmark, GO Biological Process) and
renders the results as a compact dotplot or signpost panel.

### Workflow overview

```python
import hplot

# 1) Load a signature catalog (MSigDB Hallmark, GO BP, or custom GMT).
#    `cache_dir` is required: catalogs are materialised there as .gmt and reused.
signatures = hplot.load_catalog("msigdb", cache_dir="pathway_catalogs")

# 2) Gate signatures to genes present on your assay panel.
#    Returns (present, coverage); coverage is {name: (n_present, n_total, fraction)}.
panel_genes = set(adata.var_names)
sig_filtered, coverage = hplot.select_signatures_on_panel(
    signatures, panel_genes, mode="discovery", min_panel_genes=5)

# 3) Test each signature against the gene-level screen, layer by layer.
#    `gene_bands` is the per-gene band table from gradient_cluster_mass_screen():
#    gene, band_lo, band_hi and an FDR column. The universe is its rows, so the
#    background is the rate at which the measured panel itself is banded there.
grid_df, summary = hplot.hpathway_layer_ora(
    gene_bands, sig_filtered, grid=range(-6, 13),
    fdr_col="fdr_global", alpha=0.05, min_genes=5, min_run=2,
)

# 4) Render the dotplot. Over-representation counts genes and has no direction,
#    so no direction column is passed.
out = hplot.plot_hpathway_dotplot(
    grid_df, score_col="enrichment", fdr_col="q",
    select_fdr_below=None, max_rows=30,
)
```

**There is no pathway-score channel, by design.** Scoring each signature per cell,
averaging per layer and testing that average against its own baseline is a
*self-contained* test: on a targeted panel it calls almost every signature
significant, including random level-matched gene sets, because a tissue-wide
gradient acts on every gene. `hpathway_layer_ora()` instead counts genes that the
per-gene screen already tested against a patient-level permutation null, which is
the standard over-representation design. `pathway_competitive_test()` is the
pooled counterpart.


### Signature catalogs

| Source | Description | Usage |
|--------|-------------|-------|
| `"msigdb"` | MSigDB Hallmark (50 gene sets via Enrichr) | `load_catalog("msigdb", cache_dir=...)` |
| `"go_bp"` | GO Biological Process (via Enrichr) | `load_catalog("go_bp", cache_dir=...)` |
| `"go_goatools"` | GO with DAG propagation (via goatools + NCBI gene2go) | `load_catalog("go_goatools", cache_dir=..., obo_path=..., gene2go_path=...)` |
| Custom `.gmt` | Any GMT file | `hplot.read_gmt("path/to/file.gmt")` |

```python
# Write your own GMT (path first, then the {name: genes} mapping)
hplot.write_gmt("custom.gmt", {"MySignature": ["GENE1", "GENE2", "GENE3"]})

# Filter signatures to panel genes. Returns a 2-tuple; `min_genes` applies to
# mode="user" only, `min_panel_genes` / `max_panel_genes` to mode="discovery".
filtered, coverage = hplot.select_signatures_on_panel(
    signatures, panel_genes, mode="discovery", min_panel_genes=5)
```

### `plot_hpathway_dotplot()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_df` | — | Tidy (pathway x layer) DataFrame with score and FDR columns. |
| `score_col` | `"score"` | Column for the row-relative score (dot size). |
| `fdr_col` | `"fdr_dev"` | Column for FDR (dot opacity + significance ring). Use `"q"` with `hpathway_layer_ora()` output. |
| `path_col` | `"pathway"` | Column identifying each pathway/signature. |
| `layer_col` | `"layer"` | Column for signed border layer. |
| `fdr_threshold` | `0.05` | FDR below which a black ring marks the dot. |
| `select_fdr_below` | `None` | Keep only pathways with ≥1 layer below this FDR. |
| `max_rows` | `40` | Maximum pathways to show (best min-FDR first). |
| `direction_col` | `None` | Signed column for dot colour (elevated/depressed). |
| `direction_labels` | `None` | `(depressed_label, elevated_label)` for legend. |
| `elevated_color` | `"#d62728"` | Colour for positive direction. |
| `depressed_color` | `"#1f77b4"` | Colour for negative direction. |
| `nodir_color` | `"0.7"` | Colour when no direction column. |
| `size_range` | `(12, 400)` | Marker size range (points²). |
| `alpha_range` | `(0.25, 1.0)` | Opacity range mapped to −log₁₀(FDR). |
| `neglog_fdr_cap` | `3.0` | Cap for −log₁₀(FDR) (opacity saturates at 10⁻³). |
| `cell_in` | `0.30` | Physical size (inches) of one grid cell. |
| `layer_to_distance` | `None` | Optional µm axis from `build_layer_distance_map()`. |
| `order_by_peak` | `True` | Order rows by cluster-mass peak layer. |
| `savepath` | `None` | Save figure (PNG + SVG) to this path. |

---

## Input data format

| Column | Required | Description |
|--------|----------|-------------|
| `layer` | Yes | Signed integer layer index. 0 = boundary; negative = outside. |
| `target_prop` (or any name) | Yes | Fraction of the target cell type at this layer for this patient. |
| `group` (or any name) | For Stage 1/2 | Binary group label (e.g. `"HPV+"` / `"HPV-"`). |
| `distance` | No | Mean physical distance (µm) for this layer — secondary x-axis labels. |
| `case_id` | No | Patient/slide identifier; one row per patient per layer. |
| confounders | Stage 2 only | Continuous or binary columns passed to `covariate_cols=`. |

---

## CLI reference

### `hplot plot`

```
hplot plot -i data.csv --targets immune_fraction [--group hpv_status]
           [--layer layer] [--distance distance] [-u um]
           [-o out/] [-f svg|pdf|png] [--dpi 300] [--ci]
           [--exclude-base] [--min-base-excluded-count 1]
```

`--exclude-base` derives the target curve from count columns as
`target_count / (all_count − base_count)` (accepts `*_type_count` / `n_cells`
aliases) instead of using a precomputed proportion, so the fraction is taken
among non-base (e.g. non-tumour) cells only. `--min-base-excluded-count` drops
layers with too few non-base cells (`NaN`).

### `hplot test`

```
hplot test -i data.csv [--target immune_fraction] --group hpv_status
           [--groups "HPV-" "HPV+"] [--test mannwhitney|ttest|welch]
           [--correction fdr_bh|bonferroni] [--min-n 3]
           [--permutations 999] [--threshold 0.05] [--seed 42]
           [--exclude-base] [--min-base-excluded-count 1]
           [-o pvalues.csv]
```

Outputs: per-layer p-value table (CSV) + cluster-mass permutation result.
`--target` is required unless `--exclude-base` supplies the numerator/denominator
from count columns.

### `hplot gam`

```
hplot gam -i data.csv [--target immune_fraction] --group hpv_status
          [--groups "HPV-" "HPV+"] --at-layer 0
          [--covariates AGE late_stage is_female]
          [--n-splines 10] [--curves-output gam_curves.csv]
          [--exclude-base] [--min-base-excluded-count 1]
```

Outputs: effect size (Δ), Wald p-value, n; optionally per-group curve CSV.
`--target` is required unless `--exclude-base` is given.

### `hplot screen`

```
hplot screen -i long.csv --sample sample --layer layer --unit unit --value value
             [--distance dist] [--grid LO HI] [--baseline window|far_stroma|far_tumor|"a,b"]
             [--band-mode dominant|bidirectional] [--min-per-group 10]
             [--permutations 1000] [--min-w 1] [--cluster-alpha 0.05] [--seed 0]
             [-o ranking.csv] [--wide-output wide.csv]
```

Runs `gradient_cluster_mass_screen()` across **every feature** (`unit`) in a
long `sample × layer × unit × value` CSV and writes the banded **ranking table**
(one row per feature with a scored band: `direction`, `band_start_layer`,
`band_end_layer`, `peak_layer`, `cluster_mass`, `fdr`, …). This is the slow
permutation step — run it once. `--baseline` selects the deviation reference
(`window` = per-slide window mean; `far_stroma` / `far_tumor` = tissue beyond
the grid; `"a,b"` = an explicit layer range).

### `hplot loci`

```
hplot loci -i ranking.csv [--kind bands|summary|bidirectional]
           [--sort outer_to_inner|inner_to_outer|none] [--top-n N]
           [--fdr-col fdr --fdr-max 0.1] [--title "..."] [-o hloci.svg]
           [--label-col gene --lo-col band_start_layer --hi-col band_end_layer ...]
           [--screen --sample ... --layer ... --unit ... --value ...]
```

Renders an **H-Loci Summary** panel from a ranking table (e.g. `hplot screen`
output). The canonical view is `bands` (default): each feature is a horizontal
band bar spanning its cluster extent, filled by direction (`up_color` for
elevated, `down_color` for depressed), with a short vertical tick at the
cluster-mass **peak** layer. `bidirectional` draws separate elevated + depressed
bars per row (from a wide table); `summary` is the older strip+triangle
rendering. When the ranking table carries `*_um` distance columns (from
`hplot screen --distance ...`), `loci` reconstructs the layer→µm map and draws
the same **dual x-axis as the H-Plot curves** (bottom = physical distance in µm,
top = border layer L). Rendering is cheap and meant to be iterated (`--sort`,
`--top-n`, `--fdr-max`, colours). Pass `--screen` to chain a screen first from a
raw long CSV instead of a precomputed ranking table. Column-name flags
(`--label-col`, `--lo-col`, …) let you point it at any ranking schema.

---

## Project structure

```
hplot/
  core.py      — HPlot class (fit / plot / plot_delta / gam_delta / savefig)
  plotting.py  — plot_hplot(), plot_hplot_gam(), plot_hplot_gam_delta();
                 H-Loci Summary panels (plot_hloci_strip / plot_hloci_bands /
                 plot_hloci_bands_bidir / plot_hloci_fdr / plot_hloci_dotplot);
                 plot_hpathway_dotplot();
                 border distance-axis helpers (build_layer_distance_map /
                 add_border_distance_axis)
  stats.py     — compute_layer_stats(), compute_layer_pvalues(),
                 gam_group_curves(), gam_delta_curve(), gam_pooled_effect(),
                 cluster_mass_screen(), gradient_cluster_mass_screen(),
                 directional_cluster_bands(), hpathway_layer_ora(),
                 deviation_tensor(), benjamini_hochberg()
  catalogs.py  — load_catalog(), read_gmt(), write_gmt(),
                 select_signatures_on_panel() — signature/pathway catalogs
                 (MSigDB, GO BP, GO goatools, custom GMT)
  runners.py   — run_hplot_batch() batch helper
  cli.py       — argparse CLI (hplot plot / test / gam / screen / loci)
  pp.py        — AnnData preprocessing: border_layers()
  tl.py        — AnnData tool: hplot()
  pl.py        — AnnData plotting: hplot(), hplot_from_csv()
  io.py        — read_hplot_csv() CSV bridge
  _geometry.py — pure numpy/scipy border-layer geometry (Delaunay, k-hop, BFS);
                 border_layers_from_coords() for non-AnnData workflows
  _anndata.py  — lazy anndata guard + adata -> tidy-DataFrame extraction
  _serial.py   — h5ad-safe (de)serialisation of a fitted HPlot
run_hplot.py   — legacy convenience script
Dockerfile     — analysis container for paper reproducibility
```

---

## Citation

If you use H-Plot in your research, please cite:

> Huang, C.-H. et al. *H-Plot: A graph-geodesic framework for
> distance-stratified spatial profiling at tissue boundaries.*
> bioRxiv (2025). https://www.biorxiv.org/content/10.1101/2025.12.07.692260v1
