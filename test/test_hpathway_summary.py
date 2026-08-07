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

from hplot import plot_hpathway_dotplot
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


if __name__ == "__main__":
    unittest.main()
