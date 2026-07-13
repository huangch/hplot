import os
import sys
import unittest

import pandas as pd

# Ensure the package can be imported without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hplot.stats import compute_layer_stats, compute_layer_pvalues


class TestLayerStats(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "layer": [1, 1, 1, 1, 1],
                "value": [0.1, 0.3, 0.2, 0.1, 0.1],
            }
        )

    def test_ci_large_sample(self):
        df_large = pd.concat([self.df] * 7, ignore_index=True)  # 35 samples
        stats = compute_layer_stats(df_large, prop="value", layer_col="layer", distance_col=None)
        self.assertIn("ci_lower", stats.columns)
        self.assertIn("ci_upper", stats.columns)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats["ci_upper"].iloc[0], stats["mean"].iloc[0])
        self.assertLess(stats["ci_lower"].iloc[0], stats["mean"].iloc[0])

    def test_ci_small_sample(self):
        stats = compute_layer_stats(self.df, prop="value", layer_col="layer", distance_col=None)
        self.assertIn("ci_lower", stats.columns)
        self.assertIn("ci_upper", stats.columns)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats["ci_upper"].iloc[0], stats["mean"].iloc[0])
        self.assertLess(stats["ci_lower"].iloc[0], stats["mean"].iloc[0])

    def test_requires_columns(self):
        with self.assertRaises((ValueError, KeyError)):
            compute_layer_stats(self.df, prop="missing", layer_col="layer", distance_col=None)
        with self.assertRaises((ValueError, KeyError)):
            compute_layer_stats(self.df, prop="value", layer_col="missing", distance_col=None)


class TestLayerPValues(unittest.TestCase):
    def _make_df(self, a_vals, b_vals, layer=1):
        rows = [{"layer": layer, "grp": "A", "value": v} for v in a_vals]
        rows += [{"layer": layer, "grp": "B", "value": v} for v in b_vals]
        return pd.DataFrame(rows)

    def test_separated_groups_significant(self):
        df = self._make_df([0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                           [0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
        out = compute_layer_pvalues(df, prop="value", layer_col="layer", group_col="grp")
        self.assertEqual(len(out), 1)
        self.assertLess(out["p_value"].iloc[0], 0.05)
        self.assertEqual(out["n1"].iloc[0], 6)
        self.assertEqual(out["n2"].iloc[0], 6)

    def test_identical_groups_no_crash(self):
        df = self._make_df([0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2])
        out = compute_layer_pvalues(df, prop="value", layer_col="layer", group_col="grp")
        # Identical groups must not crash and must not be significant
        # (scipy returns a high p-value or NaN depending on version).
        self.assertEqual(len(out), 1)
        p = out["p_value"].iloc[0]
        self.assertTrue(pd.isna(p) or p > 0.05)

    def test_low_n_layer_is_nan_but_present(self):
        df = self._make_df([0.1], [0.9, 0.8, 0.7], layer=2)
        out = compute_layer_pvalues(df, prop="value", layer_col="layer",
                                    group_col="grp", min_n=3)
        self.assertEqual(len(out), 1)
        self.assertTrue(pd.isna(out["p_value"].iloc[0]))
        self.assertEqual(out["n1"].iloc[0], 1)

    def test_requires_two_groups(self):
        df = pd.DataFrame({
            "layer": [1, 1, 1],
            "grp": ["A", "B", "C"],
            "value": [0.1, 0.2, 0.3],
        })
        with self.assertRaises(ValueError):
            compute_layer_pvalues(df, prop="value", layer_col="layer", group_col="grp")

    def test_explicit_group_pair(self):
        df = pd.DataFrame({
            "layer": [1, 1, 1, 1, 1, 1, 1, 1],
            "grp": ["A", "A", "B", "B", "C", "C", "C", "C"],
            "value": [0.0, 0.1, 0.9, 0.8, 0.5, 0.5, 0.5, 0.5],
        })
        out = compute_layer_pvalues(df, prop="value", layer_col="layer",
                                    group_col="grp", groups=("A", "B"), min_n=2)
        self.assertEqual(out["n1"].iloc[0], 2)
        self.assertEqual(out["n2"].iloc[0], 2)

    def test_fdr_correction_column(self):
        df = self._make_df([0.0, 0.01, 0.02, 0.03], [0.9, 0.91, 0.92, 0.93])
        out = compute_layer_pvalues(df, prop="value", layer_col="layer",
                                    group_col="grp", correction="fdr_bh")
        self.assertIn("p_adj", out.columns)


import numpy as np
from scipy.stats import chi2

from hplot.stats import (
    directional_cluster_bands,
    gradient_cluster_mass_screen,
    deviation_tensor,
    _best_band_combined,
    _dominance_score,
)

# Cluster-forming threshold on the squared statistic at alpha=0.05, df=1.
# |z| > 1.96  <=>  h = z**2 > THR.
THR = float(chi2.ppf(0.95, df=1))


class TestDirectionalBands(unittest.TestCase):
    """Directional band detection — synthetic cases A–F."""

    def _bands(self, z):
        return directional_cluster_bands(np.asarray(z, dtype=float), thr=THR, min_w=1)

    def test_A_only_elevated(self):
        b = self._bands([0, 0, 3, 3, 3, 0, 0])
        self.assertIsNotNone(b["elevated"])
        self.assertIsNone(b["depressed"])
        self.assertEqual((b["elevated"]["start_idx"], b["elevated"]["end_idx"]), (2, 4))

    def test_B_only_depressed(self):
        b = self._bands([0, 0, -3, -3, -3, 0, 0])
        self.assertIsNone(b["elevated"])
        self.assertIsNotNone(b["depressed"])
        self.assertEqual((b["depressed"]["start_idx"], b["depressed"]["end_idx"]), (2, 4))

    def test_C_monotonic_returns_both(self):
        # depressed on the left, elevated on the right of a single gradient
        b = self._bands([-3, -3, 0, 3, 3])
        self.assertIsNotNone(b["elevated"])
        self.assertIsNotNone(b["depressed"])
        self.assertEqual((b["depressed"]["start_idx"], b["depressed"]["end_idx"]), (0, 1))
        self.assertEqual((b["elevated"]["start_idx"], b["elevated"]["end_idx"]), (3, 4))

    def test_D_two_positive_keep_larger(self):
        # two positive runs; the wider/heavier one (idx 3–5) must win
        b = self._bands([3, 3, 0, 4, 4, 4, 0])
        self.assertIsNotNone(b["elevated"])
        self.assertEqual((b["elevated"]["start_idx"], b["elevated"]["end_idx"]), (3, 5))
        self.assertIsNone(b["depressed"])

    def test_E_positive_and_negative(self):
        b = self._bands([4, 4, 0, -3, -3])
        self.assertEqual((b["elevated"]["start_idx"], b["elevated"]["end_idx"]), (0, 1))
        self.assertEqual((b["depressed"]["start_idx"], b["depressed"]["end_idx"]), (3, 4))

    def test_F_noisy_interruption_splits(self):
        # a single sub-threshold layer splits the run under strict contiguity
        b = self._bands([3, 3, 0, 3, 3])
        self.assertIsNotNone(b["elevated"])
        self.assertEqual(b["elevated"]["width_layers"], 2)  # not 4

    def test_center_of_mass_uses_mass_weights(self):
        # asymmetric heights pull the centre toward the heavier layer
        b = self._bands([2, 5, 0, 0, 0])
        com = b["elevated"]["center_of_mass"]
        self.assertGreater(com, 0.5)      # pulled toward layer 1 (heavier)
        self.assertEqual(b["elevated"]["peak_idx"], 1)


class TestBestBandCombined(unittest.TestCase):
    def test_combined_spans_sign_change(self):
        # combined (direction-agnostic) run stays contiguous across a sign flip
        h = np.array([9.0, 9.0, 9.0, 9.0])
        supra = h > THR
        mass, bs, be = _best_band_combined(h, supra, 1)
        self.assertEqual((bs, be), (0, 3))
        self.assertAlmostEqual(mass, 36.0)


class TestDominanceScore(unittest.TestCase):
    def test_one_direction(self):
        self.assertEqual(_dominance_score(10.0, 0.0), 1.0)

    def test_equal_masses(self):
        self.assertAlmostEqual(_dominance_score(10.0, 10.0), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(_dominance_score(10.0, 4.0), 0.6)

    def test_none(self):
        self.assertTrue(np.isnan(_dominance_score(0.0, 0.0)))


class TestGradientScreen(unittest.TestCase):
    """Screen-level behaviour — cases G (null) and H (backward compat)."""

    def _tensor(self, mu_by_layer, n_slides=40, noise=1.0, seed=1):
        """Build a (slides, layers, 1) deviation tensor with per-layer mean."""
        rng = np.random.default_rng(seed)
        mu = np.asarray(mu_by_layer, dtype=float)
        D = rng.normal(0.0, noise, size=(n_slides, mu.size, 1)) + mu[None, :, None]
        return D

    def test_H_dominant_matches_combined_band(self):
        # contiguous supra run crossing zero: dominant = ONE wide band
        grid = np.arange(4)
        D = self._tensor([6, 6, -6, -6], n_slides=40, noise=1.0)
        res = gradient_cluster_mass_screen(
            D, grid, unit_names=["g"], band_mode="dominant", n_perm=200, seed=3)
        long = res["long"]
        self.assertEqual(len(long), 1)
        self.assertEqual(int(long["width_layers"].iloc[0]), 4)  # spans sign change

    def test_bidirectional_splits_monotonic(self):
        # same profile → bidirectional yields TWO bands of width 2
        grid = np.arange(4)
        D = self._tensor([6, 6, -6, -6], n_slides=40, noise=1.0)
        res = gradient_cluster_mass_screen(
            D, grid, unit_names=["g"], band_mode="bidirectional", n_perm=200, seed=3)
        long = res["long"].sort_values("direction").reset_index(drop=True)
        self.assertEqual(len(long), 2)
        self.assertEqual(set(long["direction"]), {"elevated", "depressed"})
        self.assertTrue((long["width_layers"] == 2).all())

    def test_bidirectional_both_significant(self):
        # a well-separated elevated block and depressed block in a longer,
        # mostly-flat profile: the layer-shuffle null rarely reconstructs a
        # 3-wide same-sign run, so both directions pass FDR<0.05.
        grid = np.arange(12)
        D = self._tensor([0, 0, 6, 6, 6, 0, 0, 0, -6, -6, -6, 0],
                         n_slides=40, noise=1.0)
        res = gradient_cluster_mass_screen(
            D, grid, unit_names=["g"], band_mode="bidirectional", n_perm=400, seed=3)
        long = res["long"].sort_values("direction").reset_index(drop=True)
        self.assertEqual(set(long["direction"]), {"elevated", "depressed"})
        self.assertTrue((long["fdr"] < 0.05).all())

    def test_wide_table_shape(self):
        grid = np.arange(4)
        D = self._tensor([6, 6, -6, -6], n_slides=40, noise=1.0)
        res = gradient_cluster_mass_screen(
            D, grid, unit_names=["g"], band_mode="bidirectional", n_perm=100, seed=3)
        wide = res["wide"]
        self.assertEqual(len(wide), 1)
        for col in ("elevated_mass", "depressed_mass", "dominance_score",
                    "elevated_center", "depressed_center"):
            self.assertIn(col, wide.columns)

    def test_G_null_profile_not_significant(self):
        # pure noise across many units: essentially no direction passes FDR<0.05
        grid = np.arange(10)
        rng = np.random.default_rng(7)
        n_units = 30
        D = rng.normal(0.0, 1.0, size=(40, 10, n_units))
        res = gradient_cluster_mass_screen(
            D, grid, unit_names=[f"g{i}" for i in range(n_units)],
            band_mode="bidirectional", n_perm=300, seed=5)
        long = res["long"]
        if len(long):
            n_sig = int((long["fdr"] < 0.05).sum())
            self.assertLessEqual(n_sig, 1)  # controlled false-positive rate

    def test_dominant_backward_compat_columns(self):
        grid = np.arange(6)
        D = self._tensor([0, 4, 4, 4, 0, 0], n_slides=40, noise=1.0)
        res = gradient_cluster_mass_screen(
            D, grid, unit_names=["g"], band_mode="dominant", n_perm=150, seed=2)
        long = res["long"]
        self.assertEqual(len(long), 1)
        self.assertEqual(long["direction"].iloc[0], "elevated")
        for col in ("cluster_mass", "peak_layer", "fdr", "permutation_p"):
            self.assertIn(col, long.columns)


class TestDeviationTensor(unittest.TestCase):
    def setUp(self):
        # grid = analysis window -2..2 (5 layers); two slides, one unit.
        self.grid = np.arange(-2, 3)  # [-2,-1,0,1,2]
        # slide 0 has far-stroma layers 3,4 and far-tumor layers -4,-3;
        # slide 1 has only in-window layers.
        self.layers = [np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
                       np.array([-2, -1, 0, 1, 2])]
        self.values = [np.array([[0.], [0.], [10.], [10.], [10.],
                                 [10.], [10.], [100.], [100.]]),
                       np.array([[1.], [2.], [3.], [4.], [5.]])]

    def test_window_baseline_default(self):
        D = deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window="window", min_baseline_layers=1,
                             verbose=False)
        # slide0 in-window values are all 10 -> baseline 10 -> deviations 0.
        self.assertTrue(np.allclose(D[0, :, 0], 0.0))
        # slide1 baseline = mean(1..5)=3 -> deviations -2,-1,0,1,2.
        self.assertTrue(np.allclose(D[1, :, 0], [-2, -1, 0, 1, 2]))

    def test_far_stroma_baseline(self):
        D = deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window="far_stroma", min_baseline_layers=1,
                             verbose=False)
        # slide0 far-stroma layers (3,4) values = 100 -> baseline 100.
        self.assertTrue(np.allclose(D[0, :, 0], 10.0 - 100.0))
        # slide1 has no far-stroma layers -> skipped (NaN).
        self.assertTrue(np.all(np.isnan(D[1, :, 0])))

    def test_far_tumor_baseline(self):
        D = deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window="far_tumor", min_baseline_layers=1,
                             verbose=False)
        # slide0 far-tumor layers (-4,-3) values = 0 -> baseline 0 -> dev = 10.
        self.assertTrue(np.allclose(D[0, :, 0], 10.0))
        self.assertTrue(np.all(np.isnan(D[1, :, 0])))

    def test_explicit_range_baseline(self):
        # baseline = layers in [1,2] on each slide.
        D = deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window=(1, 2), min_baseline_layers=1,
                             verbose=False)
        # slide1 baseline = mean(values at L=1,2) = mean(4,5)=4.5.
        self.assertTrue(np.allclose(D[1, :, 0], np.array([1, 2, 3, 4, 5]) - 4.5))

    def test_min_baseline_layers_skips_slide(self):
        # far_stroma slide0 has exactly 2 far layers; require 3 -> both skipped.
        D = deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window="far_stroma", min_baseline_layers=3,
                             verbose=False)
        self.assertTrue(np.all(np.isnan(D)))

    def test_verbose_warns_on_skip(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window="far_stroma", min_baseline_layers=1,
                             verbose=True)
        self.assertIn("skipped", buf.getvalue())

    def test_bad_baseline_window_raises(self):
        with self.assertRaises(ValueError):
            deviation_tensor(self.values, self.layers, self.grid,
                             baseline_window="nonsense", verbose=False)


class TestBidirectionalPlot(unittest.TestCase):
    def test_plot_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        from hplot.plotting import plot_hloci_bands_bidirectional
        ax = plot_hloci_bands_bidirectional(
            labels=["A", "B", "C"],
            elev_lo=[3, np.nan, 1], elev_hi=[6, np.nan, 4],
            depr_lo=[-4, -2, np.nan], depr_hi=[-1, 0, np.nan],
            elev_mass=[10, np.nan, 5], depr_mass=[8, 3, np.nan],
            sort_by="dominant_center",
        )
        self.assertEqual(len(ax.get_yticklabels()), 3)


if __name__ == "__main__":
    unittest.main()
