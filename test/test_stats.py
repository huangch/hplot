import os
import sys
import unittest

import pandas as pd

# Ensure the package can be imported without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hplot.stats import compute_layer_stats


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
        stats = compute_layer_stats(df_large, value_col="value", layer_col="layer")
        self.assertIn("ci_lower", stats.columns)
        self.assertIn("ci_upper", stats.columns)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats["ci_upper"].iloc[0], stats["mean"].iloc[0])
        self.assertLess(stats["ci_lower"].iloc[0], stats["mean"].iloc[0])

    def test_ci_small_sample(self):
        stats = compute_layer_stats(self.df, value_col="value", layer_col="layer")
        self.assertIn("ci_lower", stats.columns)
        self.assertIn("ci_upper", stats.columns)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats["ci_upper"].iloc[0], stats["mean"].iloc[0])
        self.assertLess(stats["ci_lower"].iloc[0], stats["mean"].iloc[0])

    def test_requires_columns(self):
        with self.assertRaises(ValueError):
            compute_layer_stats(self.df, value_col="missing", layer_col="layer")
        with self.assertRaises(ValueError):
            compute_layer_stats(self.df, value_col="value", layer_col="missing")


if __name__ == "__main__":
    unittest.main()
