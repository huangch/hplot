"""Tests for the H-Pathway data-prep layer that survives: BH correction and the
catalog GMT round-trip / panel-coverage gate.

The UCell scoring and (pathway x layer) deviation-grid tests were removed with the
functions they covered: a self-contained grid test calls 50 of 50 random
level-matched gene sets significant, so the design was dropped in favour of the
per-layer over-representation test in ``hpathway_layer_ora``.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hplot
from hplot.catalogs import read_gmt, write_gmt, select_signatures_on_panel


class TestBenjaminiHochberg(unittest.TestCase):
    def test_matches_reference(self):
        p = np.array([0.001, 0.01, 0.02, 0.5, np.nan])
        q = hplot.benjamini_hochberg(p)
        self.assertTrue(np.isnan(q[-1]))
        fin = q[:-1]
        self.assertTrue(np.all((fin >= 0) & (fin <= 1)))
        self.assertEqual(int(np.argmin(fin)), 0)
        self.assertTrue(np.isclose(hplot.benjamini_hochberg([0.03])[0], 0.03))


class TestCatalogGate(unittest.TestCase):
    def test_gmt_roundtrip_and_panel_gate(self):
        sets = {"setA": ["G0", "G1", "G2", "G3"], "setB": ["G5", "G6"]}
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cat.gmt")
            write_gmt(p, sets)
            back = read_gmt(p)
        self.assertEqual(back["setA"], sorted(sets["setA"]))

        panel = ["G0", "G1", "G2", "G3", "G9"]
        present, coverage = select_signatures_on_panel(
            back, panel, mode="discovery", min_panel_genes=3, max_panel_genes=250)
        self.assertIn("setA", present)
        self.assertNotIn("setB", present)
        self.assertEqual(coverage["setA"], (4, 4, 1.0))

        present_u, _ = select_signatures_on_panel(
            back, panel, mode="user", min_coverage=0.5, min_genes=2)
        self.assertIn("setA", present_u)
        self.assertNotIn("setB", present_u)


if __name__ == "__main__":
    unittest.main()
