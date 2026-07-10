"""Tests for the AnnData / scanpy / squidpy interface (pp / tl / pl / io).

Skipped automatically when ``anndata`` is not installed, so the core test suite
still runs with only the hard dependencies present.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the package can be imported without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hplot  # noqa: E402

ad = pytest.importorskip("anndata")

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _make_sample(seed, n=1200):
    r = np.random.default_rng(seed)
    xy = r.uniform(0, 200, size=(n, 2))
    d = np.linalg.norm(xy - np.array([100.0, 100.0]), axis=1)
    is_tum = d < 60
    cell_type = np.where(is_tum, "tumour", "stroma").astype(object)
    ring = np.exp(-((d - 70) ** 2) / (2 * 12.0 ** 2))          # peaks ~just outside
    expr = r.poisson(1.0 + 6.0 * ring).astype(np.float32)
    return xy, cell_type, expr


@pytest.fixture
def adata():
    X, obsm, ct, samp = [], [], [], []
    for s in range(2):
        xy, cell_type, expr = _make_sample(seed=s)
        X.append(expr[:, None])
        obsm.append(xy)
        ct.append(cell_type)
        samp.append(np.full(len(xy), f"S{s}"))
    a = ad.AnnData(
        X=np.vstack(X).astype(np.float32),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(np.concatenate(ct)),
            "sample": np.concatenate(samp),
        }),
    )
    a.var_names = ["RINGGENE"]
    a.obsm["spatial"] = np.vstack(obsm)
    return a


# --------------------------------------------------------------------------- #
# lazy-import invariant
# --------------------------------------------------------------------------- #

def test_core_imports_without_anndata_reference():
    import importlib
    import hplot.core as core
    import hplot.stats as stats
    import hplot.plotting as plotting
    for mod in (core, stats, plotting):
        src = importlib.util.find_spec(mod.__name__).origin
        with open(src) as fh:
            text = fh.read()
        assert "import anndata" not in text
        assert "import squidpy" not in text


def test_namespaces_present():
    for ns in ("pp", "tl", "pl", "io"):
        assert hasattr(hplot, ns)


# --------------------------------------------------------------------------- #
# pp.border_layers
# --------------------------------------------------------------------------- #

def test_border_layers_delaunay_fallback(adata):
    hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample")
    lay = adata.obs["hplot_layer"].to_numpy()
    assert np.isfinite(lay).all()
    assert np.nanmin(lay) < 0 < np.nanmax(lay)          # inside and outside
    assert (lay == 0).any()                             # a border layer exists
    assert adata.uns["hplot_border"]["graph_source"] == "delaunay"
    # signed micron distance sign convention matches the hop layer
    um = adata.obs["hplot_distance_um"].to_numpy()
    assert np.all(np.sign(um[np.isfinite(lay) & (lay != 0)])
                  == np.sign(lay[np.isfinite(lay) & (lay != 0)]))


def test_border_layers_reuses_precomputed_graph(adata):
    from scipy.sparse import lil_matrix
    import hplot._geometry as geom

    n = adata.n_obs
    conn = lil_matrix((n, n), dtype=np.uint8)
    for s in np.unique(adata.obs["sample"]):
        idx = np.where(adata.obs["sample"].to_numpy() == s)[0]
        e = geom.delaunay_edges(adata.obsm["spatial"][idx], max_edge=30.0)
        for a_, b_ in zip(e["source"].to_numpy(), e["target"].to_numpy()):
            conn[idx[a_], idx[b_]] = 1
            conn[idx[b_], idx[a_]] = 1
    adata.obsp["spatial_connectivities"] = conn.tocsr()

    hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample")
    assert adata.uns["hplot_border"]["graph_source"] == "precomputed"
    assert np.isfinite(adata.obs["hplot_layer"].to_numpy()).any()


def test_border_layers_no_graph_and_disallowed_raises(adata):
    with pytest.raises(KeyError):
        hplot.pp.border_layers(adata, "cell_type", ["tumour"],
                               build_graph_if_missing=False)


# --------------------------------------------------------------------------- #
# tl.hplot + serialisation
# --------------------------------------------------------------------------- #

def test_tl_expression_peaks_just_outside_border(adata):
    hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample")
    hplot.tl.hplot(adata, target="RINGGENE", value_kind="expression",
                   sample_key="sample")
    from hplot._serial import deserialize
    stats, _ = deserialize(adata.uns["hplot"])
    s = stats["overall"]
    peak = s["layer"].to_numpy()[int(np.nanargmax(s["mean"].to_numpy()))]
    assert 0 <= peak <= 3          # ring gene peaks just outside the border


def test_tl_proportion_groups(adata):
    hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample")
    hplot.tl.hplot(adata, target="cell_type", value_kind="proportion",
                   sample_key="sample", key_added="hplot_prop")
    groups = list(np.asarray(adata.uns["hplot_prop"]["group_order"]).tolist())
    assert set(groups) == {"stroma", "tumour"}


def test_tl_requires_border_layers(adata):
    with pytest.raises(KeyError):
        hplot.tl.hplot(adata, target="RINGGENE")


def test_add_base_excluded_proportion_arithmetic():
    from hplot.runners import add_base_excluded_proportion
    df = pd.DataFrame({
        "target_count": [10, 20],
        "base_count": [40, 50],
        "all_count": [100, 100],
        "layer": [0, 1],
    })
    out, col = add_base_excluded_proportion(df)
    assert col == "target_prop_base_excluded"
    # target / (all - base)
    assert out[col].tolist() == pytest.approx([10 / 60, 20 / 50])


def test_add_base_excluded_proportion_min_count_nan():
    from hplot.runners import add_base_excluded_proportion
    df = pd.DataFrame({
        "target_count": [5, 3],
        "base_count": [99, 10],
        "all_count": [100, 100],
    })
    out, col = add_base_excluded_proportion(df, min_base_excluded_count=5)
    # row 0: all - base = 1 < 5  -> NaN ; row 1: 90 >= 5 -> 3/90
    assert np.isnan(out[col].iloc[0])
    assert out[col].iloc[1] == pytest.approx(3 / 90)


def test_tl_proportion_exclude_base(adata):
    hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample")
    hplot.tl.hplot(adata, target="cell_type", value_kind="proportion",
                   sample_key="sample", exclude_base=True,
                   key_added="hplot_excl")
    groups = list(np.asarray(adata.uns["hplot_excl"]["group_order"]).tolist())
    # the base category (tumour) is dropped once excluded from the denominator
    assert "tumour" not in groups
    assert "stroma" in groups


def test_tl_exclude_base_without_border_raises(adata):
    adata.obs["hplot_layer"] = 0.0     # satisfy the layer_key check only
    with pytest.raises(KeyError):
        hplot.tl.hplot(adata, target="cell_type", value_kind="proportion",
                       sample_key="sample", exclude_base=True)


# --------------------------------------------------------------------------- #
# h5ad round-trip incl. '/' in a label (HDF5 separator)
# --------------------------------------------------------------------------- #

def test_h5ad_roundtrip_with_slash_label(adata, tmp_path):
    ct2 = np.where(adata.obs["cell_type"].astype(str).to_numpy() == "tumour",
                   "tumour", "T/NK cells").astype(object)
    adata.obs["ct_slash"] = pd.Categorical(ct2)
    hplot.pp.border_layers(adata, "ct_slash", ["tumour"], sample_key="sample")
    hplot.tl.hplot(adata, target="ct_slash", value_kind="proportion",
                   sample_key="sample")
    assert "T/NK cells" in np.asarray(adata.uns["hplot"]["group_order"]).tolist()

    p = tmp_path / "hplot_slash.h5ad"
    adata.write_h5ad(p)                        # would corrupt if labels were keys
    a2 = ad.read_h5ad(p)
    groups = np.asarray(a2.uns["hplot"]["group_order"]).tolist()
    assert "T/NK cells" in groups
    ax = hplot.pl.hplot(a2)
    assert ax.__class__.__name__ == "Axes"


# --------------------------------------------------------------------------- #
# pl.hplot / CSV bridge
# --------------------------------------------------------------------------- #

def test_pl_hplot_returns_axes(adata):
    hplot.pp.border_layers(adata, "cell_type", ["tumour"], sample_key="sample")
    hplot.tl.hplot(adata, target="RINGGENE", sample_key="sample")
    ax = hplot.pl.hplot(adata)
    assert ax.__class__.__name__ == "Axes"


def test_read_hplot_csv_and_plot(tmp_path):
    csv = tmp_path / "hplot-outputs.csv"
    pd.DataFrame({
        "layer": [-2, -1, 0, 1, 2],
        "distance_um": [-20, -10, 0, 10, 20],
        "target_type_prop": [0.1, 0.2, 0.5, 0.3, 0.1],
        "all_count": [100, 120, 130, 110, 90],
    }).to_csv(csv, index=False)

    d = hplot.io.read_hplot_csv(csv)
    assert set(d) == {"target"}
    cols = set(d["target"].columns)
    assert {"layer", "mean", "distance", "ci_lower", "ci_upper", "n"} <= cols
    ax = hplot.pl.hplot_from_csv(csv)
    assert ax.__class__.__name__ == "Axes"
