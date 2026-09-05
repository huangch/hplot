---
name: hplot
description: Operate hplot for signed-distance spatial profiling at tissue boundaries — H-Plot curves, cluster-mass screens, H-Loci summaries, GAM effect sizes, and H-Pathway dotplots
---

# hplot — Agentic AI Skill File

> **Purpose**: Enable an agentic AI (Claude, OpenClaw, Hermes, or any
> tool-using LLM agent) to autonomously operate hplot — drawing H-Plot curves
> for single targets, running cluster-mass screens over many features, and
> rendering H-Loci and H-Pathway summaries across signed tissue distance
> layers.

---

## 1. What Is hplot?

hplot produces **H-Plots**: Kaplan-Meier-inspired curves showing how a quantity
(cell-type proportion, gene expression, ligand-receptor score) changes across
concentric layers measured from a tissue boundary, such as a tumour-stroma
interface. Layer index replaces time on the x-axis; per-layer confidence
intervals capture across-case variability.

- **Layer semantics**: layer `0` = the boundary, `> 0` = inside the base
  region, `< 0` = outside. The index is a **graph hop count**, not microns;
  physical distance is a separate optional column.
- **Repository**: part of the WSInsight project ecosystem
- **License**: Apache-2.0
- **Python**: ≥ 3.11
- **Entry points**: `hplot` (CLI), `hplot-mcp` (MCP server),
  `import hplot` (Python API)

### 1.1 Where hplot sits

hplot is the **downstream, user-facing analysis layer**. The pipelines write
tables; hplot reads them.

```
wsinsight / sptxinsight  ->  CSV / h5ad / GeoJSON on disk
                                     |
                                     v
                          hplot  (usually in JupyterLab)
```

The pipelines do **not** import hplot, and their own `hplot` / `hplot-finalize`
subcommands are a **name collision only** — those run each pipeline's internal
`insightlib.insight_helpers.compute_hplot`, a separate implementation from the
`HPlot` class here. If a user says "the hplot output from wsinsight", they mean
the pipeline's CSV, which is then *input* to this package.

### 1.2 The three stages — do not conflate them

| Stage | What it answers | Entry point |
| ----- | --------------- | ----------- |
| **Stage 0** | What is the per-layer mean ± CI? | `hplot plot` / `HPlot.fit()` |
| **Stage 1** | Is a run of layers significantly different, given spatial autocorrelation? | `hplot test` / `hplot screen` (cluster-mass permutation) |
| **Stage 2** | How large is the effect, adjusted for confounders? | `hplot gam` (penalised B-spline, GCV smoothing) |

Stage 0 is descriptive and must never be reported as evidence of a difference.
Stage 1 supplies the p-value/FDR. Stage 2 supplies the effect size.

### 1.3 How it differs from squidpy `var_by_distance`

squidpy measures **unsigned** distance from anchor points and fits a polynomial
— descriptive only. hplot uses **signed** border layers and does **cohort-level
inference** (confidence intervals, permutation tests, GAM). If asked "why not
just use squidpy", that is the answer.

---

## 2. Assumed Install

Assume hplot is already installed, and verify with `hplot --help` before
anything else. Installing is the fallback, not the normal path.

| Situation | Command |
| --------- | ------- |
| Standalone environment | `sh ./conda-setup.sh hplot [-m\|--mcp]` |
| Existing environment | `pip install -e .` from the repository root |

Optional extras that change what is available:

| Extra | Adds | Effect if absent |
| ----- | ---- | ---------------- |
| `hplot[mcp]` | `fastmcp` | `hplot-mcp` is unavailable. |
| `hplot[squidpy]` | `squidpy` ≥ 1.2 | Nothing in hplot imports squidpy; this is a convenience extra for building spatial graphs yourself. |

`anndata` is a **core** dependency — the `pp` / `tl` / `pl` API needs no extra.
Core stack: matplotlib, pandas, scipy, numpy, pygam, anndata. No GPU or CUDA
stack is needed; hplot is pure CPU.

---

## 3. Input Data Format

hplot expects a **tidy CSV** (or a pandas DataFrame) where each row is one case
at one layer.

### 3.1 Single-target format (`plot`, `test`, `gam`)

| Column | Type | Required | Description |
| ------ | ---- | -------- | ----------- |
| `layer` | int | yes | Layer index. `0` = boundary, negative = outside, positive = inside. |
| `target_prop` | float | yes | The value being profiled in that layer for that case. |
| `case_id` | str | for CIs | Identifies cases; needed to compute across-case confidence intervals. |
| `group` | str | for tests | Splits rows into separate lines / comparison arms. |
| `base_prop` | float | no | A reference quantity to overlay as a second line. |
| `distance` | float | no | Mean physical distance (µm) per layer; enables the dual x-axis. |

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

### 3.2 Long/screen format (`screen`, `loci --screen`)

One row per (sample, layer, feature):

| Column | Default flag | Description |
| ------ | ------------ | ----------- |
| `sample` | `--sample` | Slide / sample id. |
| `layer` | `--layer` | Signed layer index. |
| `unit` | `--unit` | The feature — gene, ligand→receptor pair, or cell type. |
| `value` | `--value` | The per-layer value for that feature in that sample. |
| *(optional)* | `--distance` | Physical distance (µm); enables `*_um` outputs. |

---

## 4. CLI Reference

The CLI is **sub-command based**:

```bash
hplot --help
hplot <sub-command> --help
```

```
usage: hplot [-h] {plot,test,gam,screen,loci,schema} ...
```

`python -m hplot.cli <sub-command> ...` and `python run_hplot.py
<sub-command> ...` are equivalent; `run_hplot.py` is only a convenience wrapper
around the same entry point. **All three forms require a sub-command** — there
is no bare top-level `--input` form.

| Sub-command | Stage | Purpose |
| ----------- | ----- | ------- |
| `plot` | 0 | Draw H-Plot curves and save as SVG/PNG/PDF. |
| `test` | 1 | Per-layer Mann-Whitney (or t/Welch) + optional cluster-mass permutation. |
| `screen` | 1 | Multi-feature cluster-mass border-gradient screen → ranking CSV. |
| `loci` | — | Render an H-Loci Summary panel from a ranking CSV. |
| `gam` | 2 | GAM effect size with optional confounder adjustment. |
| `schema` | — | Emit a machine-readable JSON schema of every sub-command. |

`screen` and `loci` are deliberately separate: `screen` is the slow permutation
stage (run once → ranking CSV); `loci` is the fast render (iterate freely on
styling without repaying the permutation cost).

### 4.1 `hplot plot` — Stage 0 curves

```bash
hplot plot -i input.csv \
  --targets target_prop base_prop \
  --layer layer --group subtype \
  --distance distance -u um \
  -o hplots -p case -f svg --dpi 300 --ci
```

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `--input / -i` | path | *required* | Input CSV. |
| `--targets` | str (1+) | `target_prop` | Column name(s) for the target quantity. Each becomes a separate line. |
| `--layer` | str | `layer` | Layer index column. |
| `--group` | str | none | Group label column; splits into separate output files. |
| `--distance` | str | none | Physical distance column (enables the dual x-axis). |
| `--unit / -u` | str | none | Distance unit label, e.g. `um`. |
| `--output / -o` | path | `hplots` | Output directory. |
| `--prefix / -p` | str | `hplot` | Output filename prefix. |
| `--format / -f` | choice | `svg` | `svg`, `pdf`, `png`. |
| `--dpi` | int | `300` | DPI for PNG output. |
| `--ci` | flag | off | Draw confidence interval bands. |
| `--exclude-base` | flag | off | Use `target_count / (all_count - base_count)` as the denominator. |
| `--min-base-excluded-count` | int | `1` | Drop layers where `all_count - base_count` is below this. Only with `--exclude-base`. |

Files are written to `<output>/<prefix>_<group>.<format>`, or
`<prefix>_all.<format>` when `--group` is omitted.

The dual x-axis needs **both** `--distance` and `--unit`; passing only one
silently gives you a layer-index axis.

### 4.2 `hplot test` — Stage 1 per-layer test + cluster mass

```bash
hplot test -i input.csv --target target_prop --group subtype \
  --groups cold hot --test mannwhitney --correction fdr_bh \
  --permutations 1000 -o pvalues.csv
```

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `--input / -i` | path | *required* | Input CSV. |
| `--group` | str | *required* | Group label column. |
| `--target` | str | none | Target column. Required unless `--exclude-base` is given. |
| `--groups` | str str | auto | Explicit group pair `LOW HIGH`. **Required when the group column has more than two unique values.** |
| `--layer` | str | `layer` | Layer index column. |
| `--distance` | str | none | Physical distance column. |
| `--test` | choice | `mannwhitney` | `mannwhitney`, `ttest`, `welch`. |
| `--correction` | choice | none | Across-layer correction: `bonferroni`, `fdr_bh`. |
| `--min-n` | int | `3` | Minimum cases per group required to test a layer. |
| `--permutations` | int | `0` | Label permutations for the cluster-mass test. `0` skips it. |
| `--threshold` | float | `0.05` | Per-layer significance threshold used to form clusters. |
| `--seed` | int | `42` | Random seed. |
| `--exclude-base` | flag | off | Derive the target from counts before testing. |
| `--min-base-excluded-count` | int | `1` | Only with `--exclude-base`. |
| `--output / -o` | path | stdout | Output CSV for the p-value table. |

**`--permutations 0` is the default, and it means no cluster-mass test.** The
bare per-layer p-values it emits are not corrected for spatial autocorrelation
across adjacent layers. If the user asks whether a region differs, pass
`--permutations 1000` (or more) — do not report raw per-layer p-values as the
answer.

### 4.3 `hplot gam` — Stage 2 effect size

```bash
hplot gam -i input.csv --target target_prop --group subtype \
  --groups cold hot --at-layer 0 \
  --covariates AGE late_stage is_female \
  --n-splines 10 --curves-output curves.csv
```

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `--input / -i` | path | *required* | Input CSV. |
| `--target` | str | none | Response column. Required unless `--exclude-base` is given. |
| `--group` | str | none | Group label column. |
| `--groups` | str str | auto | Explicit `(low, high)` group pair. |
| `--layer` | str | `layer` | Layer index column. |
| `--at-layer` | float | none | Layer at which to evaluate the group effect. |
| `--covariates` | str (1+) | none | Columns included as linear confounders. |
| `--n-splines` | int | `10` | Number of B-spline basis functions. |
| `--exclude-base` | flag | off | Derive the response from counts before fitting. |
| `--min-base-excluded-count` | int | `1` | Only with `--exclude-base`. |
| `--curves-output` | path | none | CSV of per-group GAM predictions + 95% CI. |

Fits `target ~ s(layer) + group [+ covariates]` with a penalised spline and GCV
smoothing.

### 4.4 `hplot screen` — Stage 1 multi-feature screen

```bash
hplot screen -i long.csv \
  --sample sample --layer layer --unit unit --value value \
  --distance dist --grid -6 12 --baseline far \
  --band-mode dominant --min-per-group 3 --permutations 1000 \
  -o ranking.csv
```

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `--input / -i` | path | *required* | Long CSV with sample / layer / unit / value columns. |
| `--sample` | str | `sample` | Slide or sample id column. |
| `--layer` | str | `layer` | Signed layer index column. |
| `--unit` | str | `unit` | Feature column (gene / LR pair / cell type). |
| `--value` | str | `value` | Per-layer value column. |
| `--distance` | str | none | Physical distance (µm) column; enables `*_um` outputs. |
| `--grid` | int int | data range | Analysis-window layer range `LO HI`. |
| `--baseline` | str | `window` | Baseline region: `window`, `far`, `core`, or an explicit `a,b` layer range. |
| `--min-baseline-layers` | int | `3` | Minimum baseline-region layers required per slide. |
| `--band-mode` | choice | `dominant` | `dominant` (winner-take-all) or `bidirectional` (per-direction bands). |
| `--cluster-alpha` | float | `0.05` | Cluster-forming alpha (chi² threshold). |
| `--min-w` | int | `1` | Minimum contiguous band width in layers. |
| `--min-per-group` | int | `10` | Minimum contributing slides per layer. |
| `--permutations` | int | `1000` | Layer-shuffle permutations. |
| `--seed` | int | `0` | Random seed. |
| `--progress` | flag | off | Show a tqdm bar over permutations. |
| `--output / -o` | path | `ranking.csv` | Output ranking CSV (one row per banded feature). |
| `--wide-output` | path | none | Optional CSV for the per-feature wide table. |

`--min-per-group 10` is the default and it **drops layers** that fewer than ten
slides contribute to. On a small cohort this can empty the analysis window and
produce a ranking CSV with no rows — lower it deliberately (and say so) rather
than being surprised by an empty result.

The ranking CSV is the contract for `loci`: it carries `gene`,
`band_start_layer`, `band_end_layer`, `direction`, `peak_layer`,
`cluster_mass`, and `fdr`.

### 4.5 `hplot loci` — render the H-Loci Summary

```bash
# fast path: render from an existing ranking table
hplot loci -i ranking.csv --kind bands \
  --sort outer_to_inner --top-n 24 --fdr-col fdr --fdr-max 0.1 \
  -o hloci.svg

# one-shot: run the screen inside loci from a raw long CSV
hplot loci -i long.csv --screen --kind bands \
  --sample sample --layer layer --unit unit --value value \
  --grid -6 12 --min-per-group 3 -o hloci.png
```

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `--input / -i` | path | *required* | Ranking CSV, or a raw long CSV when `--screen` is set. |
| `--output / -o` | path | `hloci.svg` | Output figure path (`.svg` / `.pdf` / `.png`). |
| `--kind` | choice | `bands` | `bands` (canonical band view), `bidirectional`, `summary` (legacy strip). |
| `--sort` | choice | `outer_to_inner` | Row ordering by band centre: `outer_to_inner`, `inner_to_outer`, `none`. |
| `--top-n` | int | all | Keep the top-N rows by cluster mass before drawing. |
| `--width` | float | `6.4` | Figure width in inches. |
| `--dpi` | int | `300` | Raster DPI. |
| `--title` | str | none | Panel title. |
| `--fdr-col` | str | `fdr` | FDR column. |
| `--fdr-max` | float | none | Drop rows with FDR above this before drawing. |
| `--label-col` | str | `gene` | Column holding feature labels. |
| `--lo-col` | str | `band_start_layer` | Band start-layer column. |
| `--hi-col` | str | `band_end_layer` | Band end-layer column. |
| `--dir-col` | str | `direction` | Direction column. |
| `--peak-col` | str | `peak_layer` | Peak-layer column. |
| `--mass-col` | str | `cluster_mass` | Cluster-mass column. |
| `--screen` | flag | off | Run the screen first; input is then a raw long CSV. |

With `--screen`, `loci` additionally accepts **every** `screen` option
(`--sample`, `--layer`, `--unit`, `--value`, `--distance`, `--grid`,
`--baseline`, `--min-baseline-layers`, `--band-mode`, `--cluster-alpha`,
`--min-w`, `--min-per-group`, `--permutations`, `--seed`, `--progress`) with
the same defaults. Without `--screen` those options are ignored.

### 4.6 `hplot schema` — machine-readable command surface

```bash
hplot schema                    # JSON to stdout
hplot schema --output cli.json  # JSON to a file
```

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `--output` | path | stdout | Write the schema JSON here instead of stdout. |

Emits `{"schema_version": 1, "commands": {...}}`. This is the same table the
MCP server registers from, so the CLI and MCP surfaces cannot drift. Prefer
reading it over guessing when a flag name or default is unclear.

---

## 5. Python API Usage

The CLI covers the common paths. Use the Python API when you need custom
figures, an existing matplotlib axis, AnnData input, or the pathway layer.

### 5.1 Single plot

```python
import pandas as pd
from hplot.core import HPlot

df = pd.read_csv("input.csv")
h = HPlot()
h.fit(df, targets="target_prop", layer="layer")
h.savefig("hplot.svg", format="svg")
```

### 5.2 Multiple targets, groups, confidence intervals

```python
h = HPlot()
h.fit(
    df,
    targets=["target_prop", "base_prop"],
    layer="layer",
    group="subtype",
    distance="distance",
    unit="um",
    ci=0.95,
)
h.savefig("hplot_grouped.svg", format="svg")
```

### 5.3 Into an existing axis, with custom styling

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
h = HPlot()
h.fit(
    df, targets="target_prop", layer="layer", group="subtype",
    color_map={"hot": "red", "cold": "blue"},
    legend_order=["hot", "cold"],
    legend_title="Tumor Subtype",
)
h.plot(ci_show=True, ax=ax,
       display_base_type="tumor", display_target_type="lymphocytes")
plt.tight_layout()
plt.savefig("hplot_custom.png", dpi=300)
```

### 5.4 Batch generation

```python
from hplot.runners import run_hplot_batch

run_hplot_batch(
    df=df, targets=["target_prop", "base_prop"], layer="layer",
    group="subtype", distance="distance", unit="um",
    output="hplots", prefix="case", ci_show=True, format="svg", dpi=300,
)
```

### 5.5 AnnData interface (scanpy-style)

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

`anndata` is imported lazily inside these functions, so `import hplot.core`
stays cheap.

### 5.6 `HPlot.fit()` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `df` | `pd.DataFrame` | — | Input data frame. |
| `targets` | `str \| list[str]` | — | Column name(s) for the profiled quantity. |
| `layer` | `str` | — | Layer index column. |
| `group` | `str \| None` | `None` | Column to split into separate lines. |
| `distance` | `str \| None` | `None` | Mean physical distance per layer. |
| `unit` | `str \| None` | `None` | Unit label for the x-axis. |
| `ci` | `float` | `0.95` | Confidence level. t-distribution for n ≤ 30, z for n > 30. |
| `smoother` | `str` | `"mean"` | `"mean"` (per-layer average) or `"gam"` (penalised smooth). |
| `color_map` | `dict \| None` | `None` | Explicit `{label: color}`. Overrides `palette`. |
| `palette` | `sequence \| None` | `None` | Colour sequence. Defaults to `plt.cm.tab10.colors`. |
| `legend_order` | `list \| None` | `None` | Order of legend entries. |
| `legend_title` | `str \| None` | `None` | Legend box title. |
| `legend_kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to `ax.legend()`. |

### 5.7 `HPlot.plot()` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `ci_show` | `bool` | `True` | Draw shaded confidence interval bands. |
| `ax` | `Axes \| None` | `None` | Existing axis; creates a new figure if `None`. |
| `display_base_type` | `str` | `"tumor"` | Reference tissue type used in the title and x-axis. |
| `display_target_type` | `str` | `"immune cells"` | Target name interpolated into the y-axis label. |
| `value_kind` | `str` | `"proportion"` | Y-axis phrasing: `"proportion"`, `"fraction"`, `"expression"`, `"interaction"`. |
| `ylabel` | `str \| None` | `None` | Explicit y-axis label; overrides `value_kind`. |

### 5.8 H-Loci Summary from Python

Each feature in a screen is reduced to a scored *cluster band* — the run of
layers where it departs from baseline — with a direction, a centre (peak
layer), and a magnitude (cluster mass).

| Function | One row per feature shows | Use when |
| -------- | ------------------------- | -------- |
| `plot_hloci_strip()` | strip at the band position; thickness = cluster mass, ▲/▼ = direction | overview of hundreds of features |
| `plot_hloci_bands()` | horizontal bar over `[band_start_layer, band_end_layer]`, peak tick | where and how wide each band sits |
| `plot_hloci_bands_bidir()` | two bars (elevated + depressed) per row | features banded on both sides |
| `plot_hloci_fdr()` | band spans + `-log10(FDR)` in two panels | ranking a whole screen by significance |
| `plot_hloci_dotplot()` | (feature × layer) dotplot; size = score, colour = direction | per-layer detail for a subset |

```python
import hplot

# `screen_df` is gradient_cluster_mass_screen(...)["long"]
rank = screen_df[screen_df["peak_layer"].notna()].sort_values("peak_layer")
ax = hplot.plot_hloci_bands(
    rank["band_start_layer"], rank["band_end_layer"], rank["direction"],
    peak=rank["peak_layer"], labels=rank["gene"],
    sort=None,                        # keep caller order
)

# optional: add a physical-distance (um) axis under the layer axis
layer2um = hplot.build_layer_distance_map(
    [(res[sid]["layers"], res[sid]["distances"]) for sid in sample_ids])
hplot.add_border_distance_axis(ax, layer2um)
```

### 5.9 H-Pathway Summary

```python
import hplot

# 1) load a signature catalog (cache_dir is required; catalogs cache as .gmt)
signatures = hplot.load_catalog("msigdb", cache_dir="pathway_catalogs")

# 2) gate signatures to genes present on the assay panel
panel_genes = set(adata.var_names)
sig_filtered, coverage = hplot.select_signatures_on_panel(
    signatures, panel_genes, mode="discovery", min_panel_genes=5)

# 3) test each signature against the gene-level screen, layer by layer.
#    `gene_bands` is the per-gene band table from gradient_cluster_mass_screen().
grid_df, summary = hplot.hpathway_layer_ora(
    gene_bands, sig_filtered, grid=range(-6, 13),
    fdr_col="fdr_global", alpha=0.05, min_genes=5, min_run=2,
)

# 4) render. ORA counts genes and has no direction, so no direction column.
out = hplot.plot_hpathway_dotplot(
    grid_df, score_col="enrichment", fdr_col="q",
    select_fdr_below=None, max_rows=30,
)
```

**There is no per-cell pathway-score test, by design.** Scoring a signature per
cell, averaging per layer, and testing that average against its own baseline is
a *self-contained* test; on a targeted panel it calls almost every signature
significant — random level-matched gene sets included — because a tissue-wide
gradient acts on every gene. Use `hpathway_layer_ora()`, which counts genes the
per-gene screen already tested against a patient-level permutation null, or its
pooled counterpart `pathway_competitive_test()`. Do not reintroduce a
UCell-average test; `ucell_scores()` exists for scoring only, and that
separation is deliberate.

#### Signature catalogs

| Source | Description | Usage |
| ------ | ----------- | ----- |
| `"msigdb"` | MSigDB Hallmark (50 gene sets) | `hplot.load_catalog("msigdb", cache_dir=...)` |
| `"go_bp"` | GO Biological Process | `hplot.load_catalog("go_bp", cache_dir=...)` |
| `"go_goatools"` | GO with DAG propagation | `hplot.load_catalog("go_goatools", cache_dir=..., obo_path=..., gene2go_path=...)` |
| Custom `.gmt` | Any GMT file | `hplot.read_gmt("path/to/file.gmt")` |

### 5.10 `plot_hpathway_dotplot()` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `grid_df` | `pd.DataFrame` | — | Tidy (pathway × layer) frame with score and FDR columns. |
| `score_col` | `str` | `"score"` | Row-relative score column (dot size). |
| `fdr_col` | `str` | `"fdr_dev"` | FDR column (dot opacity + significance ring). |
| `path_col` | `str` | `"pathway"` | Column identifying each pathway/signature. |
| `layer_col` | `str` | `"layer"` | Signed border-layer column. |
| `fdr_threshold` | `float` | `0.05` | FDR below which a black ring marks the dot. |
| `select_fdr_below` | `float \| None` | `None` | Keep only pathways with ≥ 1 layer below this FDR. |
| `max_rows` | `int \| None` | `40` | Maximum pathways shown (best min-FDR first). |
| `direction_col` | `str \| None` | `None` | Signed column for dot colour. |
| `direction_labels` | `tuple \| None` | `None` | `(depressed_label, elevated_label)` for the legend. |
| `elevated_color` | `str` | `"#d62728"` | Colour for the positive direction. |
| `depressed_color` | `str` | `"#1f77b4"` | Colour for the negative direction. |
| `nodir_color` | `str` | `"0.7"` | Colour when no direction column is given. |
| `size_range` | `tuple` | `(12, 400)` | Marker size range (points²). |
| `alpha_range` | `tuple` | `(0.25, 1.0)` | Opacity range mapped to −log₁₀(FDR). |
| `neglog_fdr_cap` | `float` | `3.0` | Cap for −log₁₀(FDR). |
| `cell_in` | `float` | `0.30` | Physical size (inches) of one grid cell. |
| `layer_to_distance` | `Mapping \| None` | `None` | Optional µm axis from `build_layer_distance_map()`. |
| `order_by_peak` | `bool` | `True` | Order rows by cluster-mass peak layer. |
| `savepath` | `str \| None` | `None` | Save the figure (PNG + SVG) to this path. |

### 5.11 Public API index

**Core class** — `HPlot`: `fit()`, `plot()`, `plot_delta()`, `savefig()`.

| Plotting | Purpose |
| -------- | ------- |
| `plot_hplot()` | Single H-Plot with optional GAM overlay |
| `plot_hplot_gam()` | H-Plot-GAM (per-group smooths) |
| `plot_hplot_gam_delta()` | ΔH-Plot-GAM (difference curve) |
| `plot_hloci_strip()` | H-Loci Summary (strip + triangle) |
| `plot_hloci_bands()` | H-Loci Summary (horizontal bars) |
| `plot_hloci_bands_bidir()` | H-Loci Summary (two bars per row) |
| `plot_hloci_fdr()` | H-Loci band + FDR two-panel summary |
| `plot_hloci_dotplot()` | H-Loci (feature × layer) dotplot |
| `plot_hpathway_dotplot()` | H-Pathway Summary dotplot |
| `build_layer_distance_map()` | Layer → µm mapping |
| `add_border_distance_axis()` | Add a physical-distance axis |

| Statistics | Purpose |
| ---------- | ------- |
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
| `hpathway_layer_ora()` | Per-layer over-representation vs the gene screen |
| `pathway_competitive_test()` | Pooled competitive pathway test |
| `benjamini_hochberg()` | FDR correction |
| `binarize()` | Binarise a continuous variable |

| Signatures / scoring | Purpose |
| -------------------- | ------- |
| `load_catalog()` | Load an MSigDB / GO catalog |
| `read_gmt()` / `write_gmt()` | Read / write GMT files |
| `select_signatures_on_panel()` | Filter signatures to panel genes |
| `ucell_scores()` | Rank-based per-cell signature scores (scoring only, not a test) |
| `pathway_layer_profile()` | Per-layer signature profile from scores |
| `pathway_layer_profile_h5ad()` | Same, from an AnnData file |

| Geometry / AnnData | Purpose |
| ------------------ | ------- |
| `border_layers_from_coords()` | Compute border layers from coordinates |
| `hplot.pp` | `border_layers()` |
| `hplot.tl` | `hplot()` |
| `hplot.pl` | `hplot()`, `hplot_from_csv()` |
| `hplot.io` | `read_hplot_csv()` |

---

## 6. Common Workflows

### 6.1 Pipeline output → single H-Plot

```bash
# wsinsight (or sptxinsight) has already written the layer table
hplot plot -i results/hplot-outputs.csv \
  --targets immune_prop epithelial_prop \
  --layer layer --group tumor_subtype \
  --distance distance -u um \
  -o figures/ -f svg --ci
```

### 6.2 Full three-stage answer for one target

```bash
hplot plot -i layers.csv --targets immune_prop --group subtype --ci -o fig/
hplot test -i layers.csv --target immune_prop --group subtype \
  --permutations 1000 --correction fdr_bh -o pvalues.csv
hplot gam  -i layers.csv --target immune_prop --group subtype \
  --at-layer 0 --covariates AGE is_female --curves-output curves.csv
```

### 6.3 Many-feature screen → H-Loci panel

```bash
hplot screen -i long.csv --grid -6 12 --baseline far \
  --permutations 1000 --progress -o ranking.csv
hplot loci -i ranking.csv --kind bands --sort outer_to_inner \
  --top-n 24 --fdr-max 0.1 -o hloci.svg
```

Re-render freely from `ranking.csv`; only re-run `screen` when the data,
`--grid`, `--baseline`, or `--band-mode` change.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `usage: hplot [-h] {plot,test,gam,...}` then an error | Called without a sub-command | Every invocation needs one, including `python run_hplot.py` |
| `RuntimeError: Call fit() before plot()` | `plot()` used without `fit()` | Call `fit(df, ...)` first |
| `ValueError: missing ci_lower/ci_upper` | `ci_show=True` but a layer has n = 1 case | Ensure ≥ 2 cases per layer, or set `ci_show=False` |
| Empty plot | All rows NaN in the target column | Check the CSV for missing values |
| No dual x-axis | Only one of `--distance` / `--unit` given | Pass both |
| Colours not as expected | `color_map` missing a group label | Give every group value a key |
| Ranking CSV has no rows | `--min-per-group 10` dropped every layer on a small cohort | Lower `--min-per-group` deliberately |
| `test` reports significance that vanishes later | `--permutations 0` (default) → no cluster-mass correction | Re-run with `--permutations 1000` |
| `--groups` demanded | Group column has > 2 unique values | Pass `--groups LOW HIGH` explicitly |
| `ImportError: ... needs anndata` | Incomplete install (`anndata` is core) | `pip install "anndata>=0.11,<0.13"` |
| All `hplot_layer` are NaN | No base cells matched | Check `cluster_key` values and `base_categories` spelling |

---

## 8. Agent Decision Guide

```text
Is hplot installed?  (hplot --help)
├─ Yes → continue
└─ No  → fallback install (§2), then continue

Is the input a tidy per-layer table?
├─ Yes, one target → §8.1 below
├─ Yes, long (sample/layer/unit/value) over many features → screen → loci
├─ It is an AnnData → hplot.pp.border_layers() → hplot.tl.hplot()
└─ It is raw pipeline output → the pipeline writes the layer table; run
        wsinsight / sptxinsight first, then point hplot at its CSV

8.1 What is the question?
├─ "show me the gradient"        → hplot plot   (Stage 0, descriptive only)
├─ "is it significantly different?" → hplot test --permutations 1000  (Stage 1)
├─ "how big is the effect, adjusted?" → hplot gam --covariates ...    (Stage 2)
├─ "which of these 2000 genes are banded?" → hplot screen → hplot loci
└─ "which pathways?"             → hpathway_layer_ora() + plot_hpathway_dotplot()
```

### Key constraints for agents

1. **Never answer a significance question with Stage 0.** `plot` and
   `HPlot.fit()` produce means and confidence intervals, not a test. Use
   `test`/`screen` for p-values and `gam` for effect size.
2. **`test --permutations` defaults to 0**, which skips the cluster-mass
   correction entirely. Pass an explicit count when spatial autocorrelation
   matters — which is almost always, since adjacent layers share cells.
3. **`screen --min-per-group` defaults to 10.** On cohorts smaller than that it
   silently empties the analysis window. Check the row count of the ranking CSV
   before rendering.
4. **`screen` is slow, `loci` is fast.** Run `screen` once, then iterate on
   `loci`. Do not re-run `screen` for a styling change.
5. **Layer is a hop count, not microns.** Only pass `--distance` (plus `--unit`)
   when a real physical-distance column exists; do not fabricate one.
6. **`--baseline` accepts `window`, `far`, `core`, or `a,b`.** There are no
   `far_stroma` / `far_tumor` values.
7. **The `hplot` sub-command in wsinsight/sptxinsight is a different program.**
   Do not mix their flags with this CLI's.
8. **Read `hplot schema`** rather than guessing a flag name or default.

---

## 9. MCP Server (`hplot-mcp`)

Requires the `mcp` extra.

```bash
hplot-mcp                              # stdio (default)
hplot-mcp --http 127.0.0.1:8767        # streamable HTTP, loopback only
hplot-mcp --max-concurrent 2           # run up to 2 jobs in parallel
```

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--http HOST:PORT` | stdio | Serve over streamable HTTP. Suggested port 8767 (wsinsight 8765, sptxinsight 8766). |
| `--max-concurrent N` | `1` | Cap on simultaneous jobs. hplot is pure CPU, hence the low default. |

One tool per sub-command, each a faithful mirror of `hplot <sub> --help`.

**Job model.** `test`, `screen` and `loci` are permutation-heavy and therefore
long-running: they return a `job_id`. Poll `job_status`, stream `job_logs`,
stop with `cancel_job`, enumerate with `list_jobs`. `plot` and `gam` run
synchronously (600 s timeout) and return
`{status, returncode, argv, duration_s, log_tail}` directly.

Also exposed: resource `hplot://schema` (the command table) and prompt
`hplot_workflow`.

The adapter translates snake_case arguments to kebab-case flags
(`min_per_group` → `--min-per-group`); `nargs` arguments are repeated. **No
positional arguments are supported.**
