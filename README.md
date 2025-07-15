# h-plot

**H-Plot: A spatial heterogeneity plot for tissue-based distance layers (e.g., tumor regions)**

`h-plot` is a Python package for visualizing spatial heterogeneity across concentric distance layers in tissue regions, particularly tumor borders. Inspired by Kaplan-Meier survival curves, the H-Plot tracks the distribution of a target cell type (e.g., lymphocytes) across spatial layers instead of time.

## 🚀 Installation

```bash
pip install -e .
```

## 📁 Input Data Format (CSV)

| region_id | layer | value  | tumor_type | subtype |
|-----------|-------|--------|------------|---------|
| R1        | -2    | 0.05   | lumA       | hot     |
| R1        | -1    | 0.08   | lumA       | hot     |
| R1        | 0     | 0.10   | lumA       | hot     |
| ...       | ...   | ...    | ...        | ...     |

- `region_id`: Unique ID per tumor region
- `layer`: Distance from the tumor boundary (0 = boundary, positive = inside, negative = outside)
- `value`: Proportion of a specific cell type in that layer
- `tumor_type`: Used to split plots (one plot per type)
- `subtype`: Used to draw multiple lines within each plot

## 🧭 CLI Usage

```bash
python run_hplot.py \
  --input mydata.csv \
  --value_col value \
  --layer_col layer \
  --region_col region_id \
  --label_col tumor_type \
  --group_col subtype \
  --ci \
  --file_format svg
```

## 📘 Python API Example

```python
from hplot.core import HPlot

h = HPlot()
h.fit(
    df,
    value_col="value",
    layer_col="layer",
    region_col="region_id",
    group_col="subtype"
)
h.plot(ci_show=True)
h.savefig("hplot_lumA.svg")
```

## 🔍 Features

- Handles variable tumor region sizes and uneven layer counts
- Groups by label and subgroup
- Automatically computes confidence intervals using z- or t-distribution
- Output formats: SVG / PDF / PNG, styled similar to Kaplan-Meier plots

## 📄 License

MIT License