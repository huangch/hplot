---
name: hplot
description: Install and operate hplot for spatial heterogeneity visualization — H-Plots, H-Loci Summaries, and H-Pathway Summaries for cell-type proportions, gene expression, and pathway activity across tissue distance layers
---

# hplot — Agentic AI Skill File

> **Purpose**: Enable an agentic AI (Claude, OpenClaw, Hermes, or any
> tool-using LLM agent) to autonomously install and operate hplot for
> generating spatial heterogeneity visualizations — H-Plots for single targets,
> H-Loci Summaries for multi-feature screens, and H-Pathway Summaries for
> pathway/signature-level profiling across tissue distance layers.

---

## 1. What Is hplot?

hplot is a Python package that produces **H-Plots**, a visualization
introduced by Huang et al. as a Kaplan-Meier-inspired curve showing how
cell-type proportions change across concentric distance layers measured from a
tissue boundary (e.g., a tumor border).  Layer index replaces time on the
x-axis; cell proportion is on the y-axis; per-layer
confidence intervals capture across-case variability.

- **Repository**: Part of the WSInsight project ecosystem
- **License**: Apache 2.0
- **Python**: ≥ 3.8
- **Entry points**: `python run_hplot.py` (batch CLI), `python -m hplot.cli` (full CLI), or `from hplot import HPlot` (Python API)

### Key capabilities

| Visualization | Purpose | Function |
|--------------|---------|----------|
| **H-Plot** | Single-target curve across layers | `HPlot.fit()` / `plot_hplot()` |
| **H-Loci Summary** | Multi-feature screen overview | `plot_hloci_bands()` / `plot_hloci_summary()` |
| **H-Pathway Summary** | Pathway/signature dotplot | `plot_hpathway_summary()` |

---

## 2. Install

### 2.1 Dependencies

| Package    | Version  | Why                         |
| ---------- | -------- | --------------------------- |
| matplotlib | ≥ 3.0    | Plotting engine             |
| pandas     | ≥ 1.0    | Data handling               |
| scipy      | ≥ 1.6    | Confidence interval stats   |
| numpy      | ≥ 1.18   | Numerical computation       |
| pygam      | ≥ 0.8    | GAM smoothing (Stage 2)     |

Optional extras (only for the AnnData / scanpy / squidpy interface, § 5.x):

| Extra              | Adds                       | Install                        |
| ------------------ | -------------------------- | ------------------------------ |
| `hplot[anndata]`   | `anndata` ≥ 0.8            | `pip install "hplot[anndata]"` |
| `hplot[squidpy]`   | `anndata` + `squidpy` ≥1.2 | `pip install "hplot[squidpy]"` |

### 2.2 Editable Install (Recommended)

```bash
cd hplot
pip install -e .
```

### 2.3 Direct from Source

```bash
pip install matplotlib pandas scipy numpy pygam
# then use the package directly from the repository root
```

---

## 3. Input Data Format

hplot expects a **CSV file** (or a pandas DataFrame) where each row represents
one tissue region (case) at one spatial layer.

### Required columns

| Column        | Type    | Description                                                      |
| ------------- | ------- | ---------------------------------------------------------------- |
| `layer`       | int     | Layer index. `0` = tissue boundary; negative = outside; positive = inside. |
| `target_prop` | float   | Proportion of the target cell type in that layer for that case.  |

### Optional columns

| Column        | Type    | Description                                                      |
| ------------- | ------- | ---------------------------------------------------------------- |
| `base_prop`   | float   | Proportion of a reference cell type (e.g., epithelial) to overlay. |
| `case_id`     | str     | Identifies individual cases for confidence interval computation. |
| `group`       | str     | Groups rows into separate H-Plot lines (e.g., tumor subtype).   |
| `distance`    | float   | Mean physical distance (µm) for each layer; enables dual x-axis. |

### Minimal example CSV

```csv
case_id,layer,target_prop,base_prop,subtype,distance
C1,-2,0.05,0.40,hot,210.3
C1,-1,0.08,0.38,hot,105.1
C1,0,0.15,0.35,hot,0.0
C1,1,0.20,0.30,hot,98.7
C2,-1,0.03,0.50,cold,112.0
C2,0,0.10,0.45,cold,0.0
C2,1,0.12,0.42,cold,95.5
```

---

## 4. CLI Usage

The batch CLI reads a CSV, optionally groups by a column, and saves one plot
per group.

### 4.1 Command

```bash
python run_hplot.py \
  --input input.csv \
  --targets target_prop base_prop \
  --layer layer \
  --group subtype \
  --distance distance \
  --unit µm \
  --output hplots \
  --prefix case \
  --format svg \
  --dpi 300 \
  --ci
```

Alternatively, invoke via the module:

```bash
python -m hplot.cli \
  --input input.csv \
  --targets target_prop \
  --layer layer \
  --output hplots \
  --ci
```

### 4.2 CLI Arguments

| Argument     | Short | Default        | Description                                                     |
| ------------ | ----- | -------------- | --------------------------------------------------------------- |
| `--input`    | `-i`  | *(required)*   | Path to input CSV file.                                         |
| `--targets`  |       | `target_prop`  | One or more column names for cell proportions. Each becomes a separate line. |
| `--layer`    |       | `layer`        | Column for the layer index.                                     |
| `--group`    |       | `None`         | Column to split into separate output files.                     |
| `--distance` |       | `None`         | Column for physical distance per layer (enables dual x-axis).   |
| `--unit`     | `-u`  | `None`         | Physical distance unit label (e.g., `µm`).                      |
| `--output`   | `-o`  | `hplots`       | Directory for output files.                                     |
| `--prefix`   | `-p`  | `hplot`        | Prefix for output filenames.                                    |
| `--format`   | `-f`  | `svg`          | Output format: `svg`, `pdf`, or `png`.                          |
| `--dpi`      |       | `300`          | DPI for raster output (PNG).                                    |
| `--ci`       |       | flag (off)     | Show confidence interval bands.                                 |

### 4.3 CLI Output

Files are written to `<output>/<prefix>_<group>.{svg,pdf,png}`.  If `--group`
is not specified, a single file `<prefix>_all.<format>` is produced.

### 4.4 Multi-feature screen + H-Loci Summary (`screen` / `loci`)

Beyond single-target plotting, the `hplot` sub-command CLI can screen **many
features** at once and render the result as an H-Loci Summary. These are two
separate steps on purpose: `screen` is the slow permutation stage (run once →
ranking CSV); `loci` is the fast render (iterate freely).

```bash
# 1) slow: cluster-mass screen over every `unit` in a long CSV -> ranking table
python -m hplot.cli screen -i long.csv \
  --sample sample --layer layer --unit unit --value value \
  --distance dist --grid -6 12 --baseline far_stroma \
  --band-mode dominant --min-per-group 3 --permutations 1000 \
  -o ranking.csv

# 2) fast: render an H-Loci Summary panel from the ranking table
python -m hplot.cli loci -i ranking.csv --kind bands \
  --sort outer_to_inner --top-n 24 --fdr-col fdr --fdr-max 0.1 \
  -o hloci.svg

# one-shot: chain the screen inside loci from a raw long CSV
python -m hplot.cli loci -i long.csv --screen --kind bands \
  --sample sample --layer layer --unit unit --value value \
  --grid -6 12 --min-per-group 3 -o hloci.png
```

| Command | Key arguments | Purpose |
| ------- | ------------- | ------- |
| `screen` | `--sample --layer --unit --value`, `--grid LO HI`, `--baseline window\|far_stroma\|far_tumor\|"a,b"`, `--band-mode dominant\|bidirectional`, `--permutations`, `-o ranking.csv` | Screen every feature → banded ranking table. |
| `loci` | `--kind bands\|summary\|bidirectional`, `--sort`, `--top-n`, `--fdr-col/--fdr-max`, column-name flags, `--screen`, `-o fig.svg` | Render an H-Loci Summary from a ranking CSV. |

---

## 5. Python API Usage

### 5.1 Basic Single-Plot

```python
import pandas as pd
from hplot.core import HPlot

df = pd.read_csv("input.csv")

hplot = HPlot()
hplot.fit(
    df,
    targets="target_prop",   # single target column
    layer="layer",
)
hplot.savefig("hplot.svg", format="svg")
```

### 5.2 Multi-Target with Groups and Confidence Intervals

```python
import pandas as pd
from hplot.core import HPlot

df = pd.read_csv("input.csv")

hplot = HPlot()
hplot.fit(
    df,
    targets=["target_prop", "base_prop"],  # two lines per group
    layer="layer",
    group="subtype",           # one line set per group value
    distance="distance",       # enables physical distance on x-axis
    unit="µm",
    ci=0.95,                   # 95% confidence intervals
)
hplot.savefig("hplot_grouped.svg", format="svg")
```

### 5.3 Plotting into an Existing Matplotlib Axis

```python
import matplotlib.pyplot as plt
from hplot.core import HPlot

fig, ax = plt.subplots(figsize=(8, 5))

hplot = HPlot()
hplot.fit(df, targets="target_prop", layer="layer", group="subtype")
hplot.plot(
    ci_show=True,
    ax=ax,
    display_base_type="tumor",
    display_target_type="lymphocytes",
)
plt.tight_layout()
plt.savefig("hplot_custom.png", dpi=300)
```

### 5.4 Custom Colors and Legend

```python
hplot = HPlot()
hplot.fit(
    df,
    targets="target_prop",
    layer="layer",
    group="subtype",
    color_map={"hot": "red", "cold": "blue"},
    legend_order=["hot", "cold"],
    legend_title="Tumor Subtype",
    legend_kwargs={"loc": "upper right", "fontsize": 10},
)
hplot.savefig("hplot_styled.pdf", format="pdf")
```

### 5.5 Batch Generation (Python)

```python
import pandas as pd
from hplot.runners import run_hplot_batch

df = pd.read_csv("input.csv")

run_hplot_batch(
    df=df,
    targets=["target_prop", "base_prop"],
    layer="layer",
    group="subtype",
    distance="distance",
    unit="µm",
    output="hplots",
    prefix="case",
    ci_show=True,
    format="svg",
    dpi=300,
)
```

### 5.6 AnnData interface (scanpy / squidpy)

When the input is an `AnnData` rather than a tidy CSV, use the scanpy-style
`pp`/`tl`/`pl` API (requires `pip install "hplot[anndata]"`, or
`"hplot[squidpy]"` for spatial graphs):

```python
import hplot

# 1) assign a signed border layer to every cell (Delaunay fallback if no graph)
hplot.pp.border_layers(adata, cluster_key="cell_type",
                       base_categories=["tumour"], sample_key="sample_id")

# 2) fit and stash the H-Plot in adata.uns["hplot"] (survives write_h5ad)
hplot.tl.hplot(adata, target="CD8A", groupby="cell_subtype",
               value_kind="expression", sample_key="sample_id")

# 3) draw (returns a matplotlib Axes)
hplot.pl.hplot(adata)
```

### 5.7 H-Loci Summary — visualizing a multi-feature screen

The engine above profiles **one** target across layers. To summarize a
**screen over many features** (genes, ligand→receptor pairs, or cell-type
fractions), each feature is first reduced to a scored *cluster band* — the run
of border layers where it departs from baseline — with a direction, a centre
(peak layer), and a magnitude (cluster mass).

| Function | One row per feature shows | Use when |
| -------- | ------------------------- | -------- |
| `plot_hloci_summary()` | a strip at the band position; thickness = cluster mass, ▲/▼ = direction | overview of hundreds of features |
| `plot_hloci_bands()` | a horizontal bar over `[band_lo, band_hi]`, filled by direction, peak tick | where + how wide each band sits |
| `plot_hloci_bands_bidirectional()` | two bars (elevated + depressed) per row | features banded on both sides |

```python
import hplot

rank = screen_df[screen_df["peak_layer"].notna()].sort_values("peak_layer")
ax = hplot.plot_hloci_bands(
    rank["band_lo"], rank["band_hi"], rank["direction"],
    peak=rank["peak_layer"], labels=rank["gene"],
    sort=None,                        # keep caller order
)

# optional: add a physical-distance (µm) axis under the layer axis
layer2um = hplot.build_layer_distance_map(
    [(res[sid]["layers"], res[sid]["distances"]) for sid in sample_ids])
hplot.add_border_distance_axis(ax, layer2um)
```

### 5.8 H-Pathway Summary — pathway/signature-level profiling

For **pathway/signature-level profiling**, hplot provides UCell scoring,
signature catalogs, and the H-Pathway Summary dotplot.

```python
import hplot

# 1) Load a signature catalog (MSigDB Hallmark, GO BP, or custom GMT)
signatures = hplot.load_catalog("msigdb")  # or "go_bp", "go_goatools"

# 2) Gate signatures to genes present on your assay panel
panel_genes = adata.var_names.tolist()
sig_filtered = hplot.select_signatures_on_panel(signatures, panel_genes, min_genes=5)

# 3) Compute per-cell UCell scores for each signature
scores = hplot.ucell_scores(adata.X, sig_filtered, max_rank=1500)

# 4) Profile signatures across border layers
profiles = hplot.pathway_layer_profile(
    adata.X, layers=adata.obs["hplot_layer"].values,
    signatures=sig_filtered, var_names=adata.var_names.tolist(),
    sample=adata.obs["sample_id"].values,
)

# 5) Build the (pathway x layer) grid with FDR columns
grid_df = hplot.hpathway_summary_grid(
    profiles, path_names=list(sig_filtered.keys()),
    grid=range(-6, 13),
)

# 6) Render the H-Pathway Summary dotplot
ax = hplot.plot_hpathway_summary(
    grid_df,
    direction_col="dir_contrast",      # signed column for colour
    fdr_col="fdr_contrast",            # FDR column for opacity + significance ring
    select_fdr_below=0.1,              # only show significant pathways
    max_rows=30,
)
```

#### Signature catalogs

| Source | Description | Usage |
|--------|-------------|-------|
| `"msigdb"` | MSigDB Hallmark (50 gene sets) | `hplot.load_catalog("msigdb")` |
| `"go_bp"` | GO Biological Process | `hplot.load_catalog("go_bp")` |
| `"go_goatools"` | GO with DAG propagation | `hplot.load_catalog("go_goatools")` |
| Custom `.gmt` | Any GMT file | `hplot.read_gmt("path/to/file.gmt")` |

---

## 6. `HPlot.fit()` Parameters

| Parameter       | Type               | Default | Description                                              |
| --------------- | ------------------ | ------- | -------------------------------------------------------- |
| `df`            | `pd.DataFrame`     | —       | Input data frame.                                        |
| `targets`       | `str \| list[str]` | —       | Column name(s) for cell proportions.                     |
| `layer`         | `str`              | —       | Column for the layer index.                              |
| `group`         | `str \| None`      | `None`  | Column to split into separate lines.                     |
| `distance`      | `str \| None`      | `None`  | Column for mean physical distance per layer.             |
| `unit`          | `str \| None`      | `None`  | Unit label for the x-axis (e.g., `"µm"`).               |
| `ci`            | `float`            | `0.95`  | Confidence level. t-distribution for n ≤ 30, z for n > 30. |
| `smoother`      | `str`              | `"mean"` | `"mean"` (per-layer average) or `"gam"` (penalised smooth). |
| `color_map`     | `dict \| None`     | `None`  | Explicit `{label: color}` mapping. Overrides `palette`.  |
| `palette`       | `sequence \| None` | `None`  | Color sequence. Defaults to `plt.cm.tab10.colors`.       |
| `legend_order`  | `list \| None`     | `None`  | Order of legend entries.                                 |
| `legend_title`  | `str \| None`      | `None`  | Title for the legend box.                                |
| `legend_kwargs` | `dict \| None`     | `None`  | Extra kwargs forwarded to `ax.legend()`.                 |

---

## 7. `HPlot.plot()` Parameters

| Parameter            | Type            | Default           | Description                                         |
| -------------------- | --------------- | ----------------- | --------------------------------------------------- |
| `ci_show`            | `bool`          | `True`            | Draw shaded confidence interval bands.              |
| `ax`                 | `Axes \| None`  | `None`            | Existing matplotlib axis. Creates a new figure if `None`. |
| `display_base_type`  | `str`           | `"tumor"`         | Reference tissue type (used in title, x-axis).      |
| `display_target_type`| `str`           | `"immune cells"`  | Target quantity name, interpolated into the y-axis label. |
| `value_kind`         | `str`           | `"proportion"`    | Y-axis label phrasing: `"proportion"` / `"fraction"` / `"expression"` / `"interaction"`. |
| `ylabel`             | `str \| None`   | `None`            | Explicit y-axis label; overrides `value_kind`.      |

---

## 8. `plot_hpathway_summary()` Parameters

| Parameter       | Type               | Default           | Description                                         |
| --------------- | ------------------ | ----------------- | --------------------------------------------------- |
| `grid_df`       | `pd.DataFrame`     | —                 | Tidy (pathway x layer) DataFrame with score and FDR columns. |
| `score_col`     | `str`              | `"score"`         | Column for the row-relative score (dot size).       |
| `fdr_col`       | `str`              | `"fdr_dev"`       | Column for FDR (dot opacity + significance ring).   |
| `path_col`      | `str`              | `"pathway"`       | Column identifying each pathway/signature.          |
| `layer_col`     | `str`              | `"layer"`         | Column for signed border layer.                     |
| `fdr_threshold` | `float`            | `0.05`            | FDR below which a black ring marks the dot.         |
| `select_fdr_below` | `float \| None` | `None`            | Keep only pathways with ≥1 layer below this FDR.   |
| `max_rows`      | `int \| None`      | `40`              | Maximum pathways to show (best min-FDR first).      |
| `direction_col` | `str \| None`      | `None`            | Signed column for dot colour (elevated/depressed).  |
| `direction_labels` | `tuple \| None` | `None`            | `(depressed_label, elevated_label)` for legend.     |
| `elevated_color`| `str`              | `"#d62728"`       | Colour for positive direction.                      |
| `depressed_color`| `str`             | `"#1f77b4"`       | Colour for negative direction.                      |
| `nodir_color`   | `str`              | `"0.7"`           | Colour when no direction column.                    |
| `size_range`    | `tuple`            | `(12, 400)`       | Marker size range (points²).                        |
| `alpha_range`   | `tuple`            | `(0.12, 1.0)`     | Opacity range mapped to −log₁₀(FDR).               |
| `neglog_fdr_cap`| `float`            | `3.0`             | Cap for −log₁₀(FDR) (opacity saturates at 10⁻³).   |
| `cell_in`       | `float`            | `0.30`            | Physical size (inches) of one grid cell.            |
| `layer_to_distance` | `Mapping \| None` | `None`         | Optional µm axis from `build_layer_distance_map()`. |
| `order_by_peak` | `bool`             | `True`            | Order rows by cluster-mass peak layer.              |
| `savepath`      | `str \| None`      | `None`            | Save figure (PNG + SVG) to this path.               |

---

## 9. Agentic Workflow Examples

### 9.1 End-to-End: WSInsight → hplot

A typical agent-driven workflow pairs WSInsight inference with hplot
visualization:

```bash
# Step 1-3: WSInsight inference and spatial analytics (produces layers.csv)
wsinsight run --wsi-dir slides/ --results-dir results/ --model CellViT-SAM-H-x40

# Step 4: Generate H-Plots
python run_hplot.py \
  --input results/layers.csv \
  --targets immune_prop epithelial_prop \
  --layer layer \
  --group tumor_subtype \
  --distance distance \
  --unit µm \
  --output figures/ \
  --format svg \
  --ci
```

### 9.2 Agent Decision Guide

| Agent Goal                                          | Action                                                   |
| --------------------------------------------------- | -------------------------------------------------------- |
| Visualize immune infiltration gradient               | `fit(df, targets="immune_prop", layer="layer")`          |
| Compare subtypes                                     | Add `group="subtype"` to `fit()`                         |
| Show physical distances instead of layer indices     | Add `distance="distance", unit="µm"` to `fit()`         |
| Overlay multiple cell types                          | Pass `targets=["immune_prop", "epithelial_prop"]`        |
| Generate batch plots for a full cohort               | Use `run_hplot_batch()` or the CLI with `--group`        |
| Customize colors to match publication style          | Use `color_map={"hot": "red", "cold": "blue"}`           |
| Embed plot in a larger multi-panel figure            | Pass an existing `ax` to `plot()`                        |
| Summarize a many-feature screen                      | `plot_hloci_bands()` / `plot_hloci_summary()`            |
| Profile pathway activity across layers               | `plot_hpathway_summary()` with UCell scores              |
| Add physical-distance (µm) axis to a panel           | `add_border_distance_axis(ax, build_layer_distance_map(...))` |
| Load MSigDB or GO signatures                         | `hplot.load_catalog("msigdb")` / `"go_bp"`               |

---

## 10. Complete API Reference

### Core classes

| Class | Purpose |
|-------|---------|
| `HPlot` | Main engine: `fit()` / `plot()` / `plot_delta()` / `savefig()` |

### Plotting functions

| Function | Purpose |
|----------|---------|
| `plot_hplot()` | Single H-Plot with optional GAM overlay |
| `plot_hplot_gam()` | H-Plot–GAM (per-group smooths) |
| `plot_delta_hplot_gam()` | ΔH-Plot–GAM (difference curve) |
| `plot_hloci_summary()` | H-Loci Summary (strip+triangle) |
| `plot_hloci_bands()` | H-Loci Summary (horizontal bars) |
| `plot_hloci_bands_bidirectional()` | H-Loci Summary (two bars per row) |
| `plot_hloci_fdr_summary()` | H-Loci FDR Summary |
| `plot_hpathway_summary()` | H-Pathway Summary dotplot |
| `build_layer_distance_map()` | Layer → µm mapping |
| `add_border_distance_axis()` | Add physical distance axis |

### Statistics functions

| Function | Purpose |
|----------|---------|
| `compute_layer_stats()` | Per-layer mean ± CI |
| `compute_layer_pvalues()` | Mann-Whitney / t-test per layer |
| `compute_layer_kruskal_pvalues()` | Kruskal-Wallis per layer |
| `gam_group_curves()` | Per-group GAM smooths |
| `gam_delta_curve()` | Difference curve between groups |
| `gam_pooled_effect()` | Pooled GAM effect size ± p-value |
| `cluster_mass_screen()` | Cluster-mass permutation screen |
| `gradient_cluster_mass_screen()` | Multi-feature gradient screen |
| `directional_cluster_bands()` | Extract directional cluster bands |
| `deviation_tensor()` | Per-layer deviation tensor |
| `hpathway_summary_grid()` | Build (pathway x layer) grid |
| `benjamini_hochberg()` | FDR correction |
| `binarize()` | Binarize continuous variable |

### Signature/catalog functions

| Function | Purpose |
|----------|---------|
| `load_catalog()` | Load MSigDB / GO catalog |
| `read_gmt()` | Read GMT file |
| `write_gmt()` | Write GMT file |
| `select_signatures_on_panel()` | Filter signatures to panel genes |

### UCell/pathway functions

| Function | Purpose |
|----------|---------|
| `ucell_scores()` | Per-cell UCell signature scores |
| `pathway_layer_profile()` | Per-layer pathway profile |
| `pathway_layer_profile_adata()` | Profile from AnnData |
| `pathway_layer_profile_h5ad()` | Profile from H5AD file |

### Geometry functions

| Function | Purpose |
|----------|---------|
| `border_layers_from_coords()` | Compute border layers from coordinates |

### AnnData API

| Module | Functions |
|--------|-----------|
| `hplot.pp` | `border_layers()` |
| `hplot.tl` | `hplot()`, `ucell_scores()`, `pathway_layer_profile()` |
| `hplot.pl` | `hplot()`, `hplot_from_csv()` |
| `hplot.io` | `read_hplot_csv()` |

---

## 11. Troubleshooting

| Symptom                              | Cause                                    | Fix                                              |
| ------------------------------------ | ---------------------------------------- | ------------------------------------------------ |
| `RuntimeError: Call fit() before plot()` | `plot()` called without `fit()`       | Call `hplot.fit(df, ...)` first.                 |
| `ValueError: missing ci_lower/ci_upper` | `ci_show=True` but layer has n=1 case | Ensure each layer has ≥ 2 cases, or set `ci_show=False`. |
| Empty plot                           | All rows have NaN in target column       | Check CSV for missing values in target columns.  |
| No dual x-axis                       | `distance` or `unit` not provided        | Pass both `distance=` and `unit=` to `fit()`.    |
| Colors don't match expectation       | `color_map` missing a group label        | Ensure every group value has a key in `color_map`. |
| `ImportError: ... pip install 'hplot[anndata]'` | AnnData extra not installed | `pip install "hplot[anndata]"` |
| All `hplot_layer` are NaN            | No base cells matched                    | Check `cluster_key` values and `base_categories` spelling. |
