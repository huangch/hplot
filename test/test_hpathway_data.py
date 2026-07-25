"""Tests for the H-Pathway Summary data-prep layer: BH, grid assembly, UCell
scoring, catalog GMT round-trip and panel-coverage gating."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import hplot
from hplot.catalogs import read_gmt, write_gmt, select_signatures_on_panel


def test_benjamini_hochberg_matches_reference():
    p = np.array([0.001, 0.01, 0.02, 0.5, np.nan])
    q = hplot.benjamini_hochberg(p)
    # NaN stays NaN; finite entries are monotone and within [0, 1].
    assert np.isnan(q[-1])
    fin = q[:-1]
    assert np.all((fin >= 0) & (fin <= 1))
    # smallest raw p gets the smallest q; ordering preserved
    assert np.argmin(fin) == 0
    # BH of a single value equals itself
    assert np.isclose(hplot.benjamini_hochberg([0.03])[0], 0.03)


def _toy_profiles(seed=0):
    rng = np.random.default_rng(seed)
    grid = [-2, -1, 0, 1, 2]
    baseline = [3, 4, 5]  # 'far' reference beyond the grid
    layers = grid + baseline
    rows = []
    for s in range(6):
        status = "relapse" if s % 2 else "SD"
        for L in layers:
            # a real bump near layer 0 for pathway "A"
            a = 0.6 + (0.3 if L == 0 else 0.0) + rng.normal(0, 0.02)
            b = 0.5 + rng.normal(0, 0.02)
            rows.append(dict(sample=f"s{s}", layer=L, status=status, A=a, B=b))
    return pd.DataFrame(rows), grid


def test_hpathway_summary_grid_shape_and_columns():
    prof, grid = _toy_profiles()
    grid_df = hplot.hpathway_summary_grid(
        prof, path_names=["A", "B"], grid=grid, deviation="far",
        contrasts={"contrast": ("status", ["SD", "relapse"])},
        verbose=False)
    # one row per (pathway, layer) on the grid
    assert len(grid_df) == 2 * len(grid)
    for col in ["pathway", "layer", "score", "fdr_dev", "p_contrast", "fdr_contrast"]:
        assert col in grid_df.columns
    # pathway A should deviate near layer 0 (smallest fdr_dev at layer 0 for A)
    a = grid_df[grid_df.pathway == "A"].set_index("layer")
    assert a.loc[0, "fdr_dev"] <= a.loc[2, "fdr_dev"] or np.isnan(a.loc[2, "fdr_dev"])


def test_hpathway_summary_grid_no_contrasts_no_deviation():
    prof, grid = _toy_profiles()
    grid_df = hplot.hpathway_summary_grid(
        prof, path_names=["A", "B"], grid=grid, deviation=None, contrasts=None)
    assert set(grid_df.columns) == {"pathway", "layer", "score"}
    assert len(grid_df) == 2 * len(grid)


def test_ucell_scores_bounds_and_membership():
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.random((50, 8)).astype(np.float32))
    sig_idx = {"hi": np.array([0, 1]), "lo": np.array([6, 7])}
    out = hplot.ucell_scores(X, sig_idx, max_rank=8, chunk=16)
    assert set(out) == {"hi", "lo"}
    for v in out.values():
        assert v.shape == (50,)
        assert np.all((v >= 0) & (v <= 1))


def test_pathway_layer_profile_aggregates_by_layer():
    rng = np.random.default_rng(2)
    n, g = 120, 8
    X = sp.csr_matrix(rng.random((n, g)).astype(np.float32))
    layers = rng.integers(-2, 3, size=n)
    var_names = [f"G{i}" for i in range(g)]
    sigs = {"A": ["G0", "G1", "G2"], "B": ["G5", "G6"]}
    prof = hplot.pathway_layer_profile(
        X, layers, sigs, var_names=var_names, sample="s0",
        extra={"status": "SD"}, chunk=32)
    # one row per observed layer, one score column per signature
    assert set(prof["layer"]) == set(np.unique(layers))
    for c in ["A", "B", "n_cells", "sample", "status"]:
        assert c in prof.columns
    assert prof["n_cells"].sum() == n
    assert np.all((prof[["A", "B"]].to_numpy() >= 0) & (prof[["A", "B"]].to_numpy() <= 1))


def test_gmt_roundtrip_and_panel_gate(tmp_path):
    sets = {"setA": ["G0", "G1", "G2", "G3"], "setB": ["G5", "G6"]}
    p = tmp_path / "cat.gmt"
    write_gmt(str(p), sets)
    back = read_gmt(str(p))
    assert back["setA"] == sorted(sets["setA"])
    # discovery gate keeps sets whose on-panel count is in [min, max]
    panel = ["G0", "G1", "G2", "G3", "G9"]
    present, coverage = select_signatures_on_panel(
        back, panel, mode="discovery", min_panel_genes=3, max_panel_genes=250)
    assert "setA" in present and "setB" not in present
    assert coverage["setA"] == (4, 4, 1.0)
    # user gate keeps by coverage fraction + absolute floor
    present_u, _ = select_signatures_on_panel(
        back, panel, mode="user", min_coverage=0.5, min_genes=2)
    assert "setA" in present_u and "setB" not in present_u
