"""Control tests for the per-layer over-representation test.

These exist because the self-contained ``fdr_dev`` channel of
the removed deviation grid was found to call 50 of 50 *random* level-matched gene
sets significant on a real 5046-gene panel -- the same rate it gave real Hallmark
sets -- and because a first replacement that averaged member *profiles* failed
its controls in both directions (a biologically empty set survived, a real one
did not) once inter-gene correlation was accounted for. Calibration is therefore
asserted here rather than assumed.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hplot import hpathway_layer_ora

N_GENES = 2000
GRID = np.arange(-5, 16)
GENES = [f"g{i:04d}" for i in range(N_GENES)]
PLANTED = GENES[:30]          # bands sit over L0..L5, where the background is thin


def _gene_bands(seed=0):
    """Genes whose bands concentrate on the stromal side, as in real data."""
    rng = np.random.default_rng(seed)
    lo = np.full(N_GENES, np.nan)
    hi = np.full(N_GENES, np.nan)
    fdr = np.ones(N_GENES)
    carries = rng.random(N_GENES) < 0.45
    for i in np.where(carries)[0]:
        start = int(rng.integers(7, 13))              # background lives at L+7..+15
        lo[i], hi[i] = start, int(min(15, start + rng.integers(2, 8)))
        fdr[i] = float(rng.uniform(0, 0.04))
    for i, _g in enumerate(PLANTED):                  # planted band over L0..L5
        lo[i], hi[i] = 0, 5
        fdr[i] = 0.001
    return pd.DataFrame({"gene": GENES, "band_lo": lo, "band_hi": hi,
                         "fdr_global": fdr})


def _sets(seed=1, n_random=10, k=30):
    rng = np.random.default_rng(seed)
    out = {"planted": list(PLANTED)}
    for i in range(n_random):
        out[f"random_{i}"] = [GENES[j] for j in
                              rng.choice(N_GENES, size=k, replace=False)]
    return out


class TestLayerORA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bands = _gene_bands()
        cls.sets = _sets()
        cls.grid_df, cls.summary = hpathway_layer_ora(
            cls.bands, cls.sets, grid=GRID, verbose=False)
        cls.by = cls.summary.set_index("pathway")

    def test_planted_set_is_recovered(self):
        self.assertTrue(bool(self.by.loc["planted", "significant"]))

    def test_planted_band_is_located_correctly(self):
        sub = self.grid_df[(self.grid_df.pathway == "planted")
                           & (self.grid_df.q < 0.05)]
        self.assertTrue(set(sub["layer"]).issubset(set(range(0, 6))))

    def test_random_sets_are_not_significant(self):
        rand = [n for n in self.sets if n.startswith("random_")]
        n_sig = int(self.by.loc[rand, "significant"].sum())
        self.assertLessEqual(n_sig, 1, f"{n_sig}/{len(rand)} random sets significant")

    def test_background_rate_varies_by_layer(self):
        """A pooled test cannot see this; it is why the test is per layer."""
        bg = (self.grid_df.groupby("layer")["background_frac"].first())
        self.assertGreater(bg.loc[10] / max(bg.loc[-5], 1e-9), 3.0)

    def test_min_run_rejects_isolated_layers(self):
        _g, strict = hpathway_layer_ora(self.bands, self.sets, grid=GRID,
                                        min_run=99, verbose=False)
        self.assertFalse(strict["significant"].any())

    def test_set_with_no_banded_genes_is_rejected(self):
        quiet = [g for g in GENES if g not in PLANTED][-40:]
        flat = self.bands.copy()
        flat.loc[flat["gene"].isin(quiet), "fdr_global"] = 1.0
        _g, summ = hpathway_layer_ora(flat, {"quiet": quiet}, grid=GRID,
                                      verbose=False)
        self.assertFalse(bool(summ.set_index("pathway").loc["quiet", "significant"]))

    def test_small_sets_are_skipped(self):
        _g, summ = hpathway_layer_ora(self.bands, {"tiny": PLANTED[:3]},
                                      grid=GRID, min_genes=5, verbose=False)
        self.assertTrue(summ.empty)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            hpathway_layer_ora(self.bands.drop(columns=["band_hi"]), self.sets,
                               grid=GRID, verbose=False)


if __name__ == "__main__":
    unittest.main()
