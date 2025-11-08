# h-plot

**H-Plot: A spatial heterogeneity plot for tissue-based distance layers**

`h-plot` is a Python package for visualizing spatial heterogeneity across concentric distance layers in tissue regions, particularly tumor borders. Inspired by Kaplan-Meier survival curves, the H-Plot tracks the distribution of a target cell type (e.g., lymphocytes) across spatial layers instead of time.

## 🚀 Installation

```bash
pip install -e .
```

## 📁 Input Data Format (CSV)

| case_id | layer | value  | subtype |
|---------|-------|--------|---------|
| C1      | -2    | 0.05   | hot     |
| C1      | -1    | 0.08   | hot     |
| C1      | 0     | 0.10   | hot     |
| ...     | ...   | ...    | ...     |

- `case_id`: Unique ID per tissue region or patient case (optional)
- `layer`: Distance from the tumor boundary (0 = boundary, positive = inside, negative = outside)
- `value`: Proportion of a specific cell type in that layer
- `subtype`: Used to draw multiple lines within each plot (optional)

## 🧭 CLI Usage

```bash
python run_hplot.py \
  input.csv \
  --value-col value \
  --layer-col layer \
  --case-col case_id \
  --group-col subtype \
  --file-format svg
```

The CLI reads a CSV file, groups the data by `--case-col` (if provided), and writes one H-Plot per case into the output directory (`hplots` by default).

## 📘 Python API Example

```python
from hplot.core import HPlot

hplot = HPlot()
hplot.fit(
    df,
    value_col="value",
    layer_col="layer",
    group_col="subtype",
    distance_col="distance",
    distance_unit="µm",
)
hplot.savefig("hplot_case.svg", format="svg")
```

## 🔍 Features

- Handles variable tumor region sizes and uneven layer counts
- Groups by label and subgroup
- Automatically computes confidence intervals using z- or t-distribution
- Output formats: SVG / PDF / PNG, styled similar to Kaplan-Meier plots

## 📄 License

MIT License
