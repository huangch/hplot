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


if __name__ == "__main__":
    unittest.main()
