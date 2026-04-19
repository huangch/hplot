---
name: hplot
description: Install and operate hplot for spatial heterogeneity visualization of cell-type proportions across tissue distance layers
---

# hplot — Agentic AI Skill File

> **Purpose**: Enable an agentic AI (Claude, OpenClaw, Hermes, or any
> tool-using LLM agent) to autonomously install and operate hplot for
> generating H-Plot spatial heterogeneity visualizations from cell-proportion
> data across tissue distance layers.

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
- **Python**: ≥ 3.7
- **Entry points**: `python run_hplot.py` (batch CLI) or `from hplot.core import HPlot` (Python API)

---

## 2. Install

### 2.1 Dependencies

| Package    | Version  | Why                         |
| ---------- | -------- | --------------------------- |
| matplotlib | ≥ 3.0    | Plotting engine             |
| pandas     | ≥ 1.0    | Data handling               |
| scipy      | ≥ 1.6    | Confidence interval stats   |
| numpy      | ≥ 1.18   | Numerical computation       |

### 2.2 Editable Install (Recommended)

```bash
cd hplot
pip install -e .
```

### 2.3 Direct from Source

```bash
pip install matplotlib pandas scipy numpy
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

This writes one file per unique value in the `group` column to the `output`
directory.

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
| `display_target_type`| `str`           | `"immune cells"`  | Target cell type (used in y-axis label).            |

---

## 8. Agentic Workflow Examples

### 8.1 End-to-End: WSInsight → hplot

A typical agent-driven workflow pairs WSInsight inference with hplot
visualization:

1. Run WSInsight cell-level inference on a cohort of WSIs.
2. Use WSInsight spatial analytics (`ncomp`) to compute per-layer cell-type
   proportions relative to a tissue boundary.
3. Export the layer-proportion CSV.
4. Feed the CSV to hplot to produce publication-ready H-Plot figures.

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

### 8.2 Agent Decision Guide

| Agent Goal                                          | Action                                                   |
| --------------------------------------------------- | -------------------------------------------------------- |
| Visualize immune infiltration gradient               | `fit(df, targets="immune_prop", layer="layer")`          |
| Compare subtypes                                     | Add `group="subtype"` to `fit()`                         |
| Show physical distances instead of layer indices     | Add `distance="distance", unit="µm"` to `fit()`         |
| Overlay multiple cell types                          | Pass `targets=["immune_prop", "epithelial_prop"]`        |
| Generate batch plots for a full cohort               | Use `run_hplot_batch()` or the CLI with `--group`        |
| Customize colors to match publication style          | Use `color_map={"hot": "red", "cold": "blue"}`           |
| Embed plot in a larger multi-panel figure            | Pass an existing `ax` to `plot()`                        |

---

## 9. Troubleshooting

| Symptom                              | Cause                                    | Fix                                              |
| ------------------------------------ | ---------------------------------------- | ------------------------------------------------ |
| `RuntimeError: Call fit() before plot()` | `plot()` called without `fit()`       | Call `hplot.fit(df, ...)` first.                 |
| `ValueError: missing ci_lower/ci_upper` | `ci_show=True` but layer has n=1 case | Ensure each layer has ≥ 2 cases, or set `ci_show=False`. |
| Empty plot                           | All rows have NaN in target column       | Check CSV for missing values in target columns.  |
| No dual x-axis                       | `distance` or `unit` not provided        | Pass both `distance=` and `unit=` to `fit()`.    |
| Colors don't match expectation       | `color_map` missing a group label        | Ensure every group value has a key in `color_map`. |
