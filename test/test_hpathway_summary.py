import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ensure the package can be imported without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep <text> in the SVG

from hplot import plot_hpathway_dotplot, hpathway_score_grid
from hplot import pl as hpl


def _synthetic_grid(n_pathways=6, layers=range(-5, 6), seed=0):
    """Tidy (pathway x layer) grid with score + all four FDR columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for pi in range(n_pathways):
        peak = rng.integers(-4, 5)
        for L in layers:
            score = np.exp(-0.15 * (L - peak) ** 2) + 0.05 * rng.standard_normal()
            fdr = float(np.clip(rng.beta(0.5, 4.0), 1e-6, 1.0))
            rows.append(dict(
                pathway=f"PATHWAY_{pi}", layer=int(L), score=float(score),
                fdr_dev=fdr, fdr_contrast=fdr, fdr_treatment=fdr, fdr_strata4=fdr,
            ))
    return pd.DataFrame(rows)


class TestHPathwaySummary(unittest.TestCase):
    def setUp(self):
        self.grid = _synthetic_grid()

    def test_returns_axes_dict(self):
        out = plot_hpathway_dotplot(self.grid, fdr_col="fdr_dev",
                                    select_fdr_below=None, max_rows=40)
        self.assertIsInstance(out, dict)
        for key in ("figure", "ax", "colorbar_axis", "selected"):
            self.assertIn(key, out)
        self.assertGreater(len(out["selected"]), 0)

    def test_empty_selection_returns_none(self):
        # An impossible FDR gate keeps no pathway.
        out = plot_hpathway_dotplot(self.grid, fdr_col="fdr_dev",
                                    select_fdr_below=0.0, max_rows=40)
        self.assertIsNone(out)

    def test_max_rows_caps_pathways(self):
        out = plot_hpathway_dotplot(self.grid, fdr_col="fdr_dev",
                                    select_fdr_below=None, max_rows=3)
        self.assertLessEqual(len(out["selected"]), 3)

    def test_savepath_writes_png_and_vector_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "hpathway.png")
            out = plot_hpathway_dotplot(self.grid, fdr_col="fdr_contrast",
                                        select_fdr_below=None, max_rows=40,
                                        savepath=png)
            svg = png.replace(".png", ".svg")
            self.assertTrue(os.path.exists(png))
            self.assertTrue(os.path.exists(svg))
            self.assertIsNotNone(out)
            with open(svg, "r", encoding="utf-8") as handle:
                body = handle.read()
            # dot labels/ticks stay as <text> (svg.fonttype='none'). The alpha-
            # ramp colorbars legitimately rasterize to <image>, so only the
            # text-as-text invariant is asserted here.
            self.assertIn("<text", body)

    def test_csv_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = os.path.join(tmp, "grid.csv")
            self.grid.to_csv(csv, index=False)
            out = hpl.hpathway_summary_from_csv(csv, fdr_col="fdr_treatment",
                                                select_fdr_below=None, max_rows=40)
            self.assertIsInstance(out, dict)
            self.assertIn("ax", out)


def _synthetic_profiles(n_units=6, n_sets=4, layers=range(-5, 6), seed=0):
    """Per-(unit, layer) activity table, the input a fixed signature list produces."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for L in layers:
            rec = {"patient": u, "layer": int(L), "n_cells": 500}
            for j in range(n_sets):
                rec[f"SET{j}"] = 0.1 * j + 0.02 * L + 0.005 * rng.standard_normal()
            rows.append(rec)
    return pd.DataFrame(rows)


class TestHPathwayScoreGrid(unittest.TestCase):
    """The fixed-list channel: position only, no direction, no self-contained p."""

    def setUp(self):
        self.names = [f"SET{j}" for j in range(4)]
        self.layers = list(range(-5, 6))
        self.profiles = _synthetic_profiles()

    def test_grid_shape_and_columns(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", count_col="n_cells",
            verbose=False)
        self.assertEqual(len(grid), len(self.names) * len(self.layers))
        self.assertEqual(set(grid.columns),
                         {"pathway", "layer", "score", "n_units", "deviation"})

    def test_carries_no_direction_or_significance_channel(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", count_col="n_cells",
            verbose=False)
        for banned in ("fdr_dev", "dir_dev", "p_dev"):
            self.assertNotIn(banned, grid.columns)

    def test_window_centring_makes_deviation_sum_to_zero(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", baseline="window",
            count_col="n_cells", verbose=False)
        # the sign flip is forced by centring, which is why only its POSITION is read
        for nm in self.names:
            total = grid.loc[grid.pathway == nm, "deviation"].sum()
            self.assertAlmostEqual(float(total), 0.0, places=6)

    def test_skip_baseline_drops_deviation(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", baseline="skip",
            verbose=False)
        self.assertNotIn("deviation", grid.columns)

    def test_dotplot_without_fdr_or_direction(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", count_col="n_cells",
            verbose=False)
        out = plot_hpathway_dotplot(grid, score_col="score", fdr_col=None,
                                    direction_col=None, max_rows=None)
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["selected"]), len(self.names))

    def test_max_rows_without_fdr_ranks_by_score(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", count_col="n_cells",
            verbose=False)
        out = plot_hpathway_dotplot(grid, score_col="score", fdr_col=None,
                                    direction_col=None, max_rows=2)
        self.assertEqual(len(out["selected"]), 2)
        # SET3 has the largest offset, so it must survive a score-ranked cap
        self.assertIn("SET3", list(out["selected"]))

    def test_select_fdr_below_requires_an_fdr_column(self):
        grid = hpathway_score_grid(
            self.profiles, path_names=self.names, grid=self.layers,
            sample_col="patient", layer_col="layer", count_col="n_cells",
            verbose=False)
        with self.assertRaises(ValueError):
            plot_hpathway_dotplot(grid, score_col="score", fdr_col=None,
                                  select_fdr_below=0.05)


class TestRowOrdering(unittest.TestCase):
    """A row with no signal must not be able to claim an end of the panel."""

    layers = list(range(-5, 6))

    def _grid(self):
        """Five rows with real, evenly spaced mass, plus one that is flat noise
        tilted to the far right. The spread matters: with too few rows the median
        centroid coincides with a real row and there is no middle to shrink to."""
        rows = []
        centres = {"C-4": -4, "C-2": -2, "C0": 0, "C2": 2, "C4": 4}
        for L in self.layers:
            for name, c in centres.items():
                rows.append(dict(pathway=name, layer=L,
                                 score=1.0 if abs(L - c) <= 1 else 0.01))
            rows.append(dict(pathway="EMPTY", layer=L,
                             score=0.001 if L == 5 else 0.0))
        return pd.DataFrame(rows)

    def test_unshrunk_centroid_lets_an_empty_row_reach_the_edge(self):
        out = plot_hpathway_dotplot(
            self._grid(), score_col="score", fdr_col=None, max_rows=None,
            order_by_peak=True, order_by="centroid", order_shrink=0.0)
        self.assertEqual(list(out["selected"])[-1], "EMPTY")

    def test_shrinkage_pulls_the_empty_row_off_the_edge(self):
        out = plot_hpathway_dotplot(
            self._grid(), score_col="score", fdr_col=None, max_rows=None,
            order_by_peak=True, order_by="centroid", order_shrink=1.0)
        order = list(out["selected"])
        self.assertNotEqual(order[0], "EMPTY")
        self.assertNotEqual(order[-1], "EMPTY")
        # the rows carrying real mass keep their spatial ordering
        real = [p for p in order if p != "EMPTY"]
        self.assertEqual(real, ["C-4", "C-2", "C0", "C2", "C4"])

    def test_centroid_is_tie_free_where_peak_is_not(self):
        # two rows peaking on the same layer but with different mass distributions
        rows = []
        for L in self.layers:
            rows.append(dict(pathway="A", layer=L, score=1.0 if L == 0 else 0.5))
            rows.append(dict(pathway="B", layer=L, score=1.0 if L == 0 else
                             (0.9 if L > 0 else 0.1)))
        df = pd.DataFrame(rows)
        peak = plot_hpathway_dotplot(df, score_col="score", fdr_col=None,
                                     max_rows=None, order_by_peak=True,
                                     order_by="peak")
        cent = plot_hpathway_dotplot(df, score_col="score", fdr_col=None,
                                     max_rows=None, order_by_peak=True,
                                     order_by="centroid", order_shrink=0.0)
        self.assertEqual(len(peak["selected"]), 2)
        # B leans outward, so the centroid separates them; the shared peak cannot
        self.assertEqual(list(cent["selected"]), ["A", "B"])

    def test_unknown_order_by_rejected(self):
        with self.assertRaises(ValueError):
            plot_hpathway_dotplot(self._grid(), score_col="score", fdr_col=None,
                                  order_by_peak=True, order_by="nonsense")


if __name__ == "__main__":
    unittest.main()
