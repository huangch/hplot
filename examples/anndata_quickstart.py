"""Self-contained AnnData quickstart for hplot (scanpy / squidpy style).

Run:  python examples/anndata_quickstart.py

Builds a synthetic 2-sample AnnData (a tumour disc in stroma, plus one gene that
peaks just outside the tumour border), then walks the pp -> tl -> pl workflow:

    pp.border_layers   assign each cell a signed border layer + micron distance
    tl.hplot           fit the H-Plot, store it in adata.uns['hplot']
    pl.hplot           draw it

No real data or squidpy graph is required: pp.border_layers falls back to a
Delaunay graph built from adata.obsm['spatial']. anndata is a core dependency,
so a plain `pip install hplot` is enough.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless; drop this line to view interactively
import matplotlib.pyplot as plt

import anndata as ad
import hplot


def make_sample(seed, n=1500):
    """One synthetic tissue: a tumour disc in stroma + a border-ring gene."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 200, size=(n, 2))
    d = np.linalg.norm(xy - np.array([100.0, 100.0]), axis=1)
    cell_type = np.where(d < 60, "tumour", "stroma").astype(object)
    ring = np.exp(-((d - 70) ** 2) / (2 * 12.0 ** 2))       # peaks ~10 um outside
    expr = rng.poisson(1.0 + 6.0 * ring).astype(np.float32)
    return xy, cell_type, expr


# --- build a 2-sample AnnData -------------------------------------------------
X, coords, ctype, sample = [], [], [], []
for s in range(2):
    xy, ct, expr = make_sample(seed=s)
    X.append(expr[:, None])
    coords.append(xy)
    ctype.append(ct)
    sample.append(np.full(len(xy), f"patient_{s}"))

adata = ad.AnnData(
    X=np.vstack(X),
    obs=pd.DataFrame({
        "cell_type": pd.Categorical(np.concatenate(ctype)),
        "sample_id": np.concatenate(sample),
    }),
    var=pd.DataFrame(index=["RINGGENE"]),
)
adata.obsm["spatial"] = np.vstack(coords)
print(adata)

# --- 1) pp: assign border layers ---------------------------------------------
# (a squidpy user would first run sq.gr.spatial_neighbors(adata); here we let
#  hplot fall back to a Delaunay graph from adata.obsm['spatial'].)
hplot.pp.border_layers(
    adata, cluster_key="cell_type", base_categories=["tumour"],
    sample_key="sample_id",
)
print("graph source :", adata.uns["hplot_border"]["graph_source"])
print("layer range  :", np.nanmin(adata.obs['hplot_layer']),
      "..", np.nanmax(adata.obs['hplot_layer']))

# --- 2) tl: fit two H-Plots ---------------------------------------------------
# expression of the ring gene across layers (one overall curve)
hplot.tl.hplot(
    adata, target="RINGGENE", value_kind="expression",
    sample_key="sample_id", display_target_type="RINGGENE",
    key_added="hplot_expr",
)
# cell-type proportions across layers (one curve per cell type)
hplot.tl.hplot(
    adata, target="cell_type", value_kind="proportion",
    sample_key="sample_id", key_added="hplot_prop",
)

# --- 3) pl: draw both panels --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
hplot.pl.hplot(adata, key="hplot_expr", ax=ax1)
ax1.set_title("Ring gene expression vs border layer")
hplot.pl.hplot(adata, key="hplot_prop", ax=ax2)
ax2.set_title("Cell-type proportion vs border layer")
fig.tight_layout()
out = "anndata_quickstart.png"
fig.savefig(out, dpi=150)
print("saved figure ->", out)

# --- 4) persistence: uns['hplot_*'] survives write_h5ad ----------------------
adata.write_h5ad("anndata_quickstart.h5ad")
reloaded = ad.read_h5ad("anndata_quickstart.h5ad")
ax = hplot.pl.hplot(reloaded, key="hplot_expr")   # re-plot from the reloaded file
print("re-plotted from h5ad OK; expression peak near layer",
      int(np.asarray(reloaded.uns["hplot_expr"]["stats"]["layer"])[
          int(np.nanargmax(np.asarray(reloaded.uns["hplot_expr"]["stats"]["mean"])))]))
print("done.")
