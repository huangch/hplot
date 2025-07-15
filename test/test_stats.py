import unittest
import pandas as pd
from hplot.stats import compute_layer_stats

class TestLayerStats(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "region_id": ["R1", "R2", "R3", "R4", "R5"],
            "layer": [1, 1, 1, 1, 1],
            "value": [0.1, 0.3, 0.2, 0.1, 0.1]
        })

    def test_ci_large_sample(self):
        # Test with sample size > 30 using normal distribution
        df_large = pd.concat([self.df] * 7, ignore_index=True)  # 35 samples
        stats = compute_layer_stats(df_large, value_col="value", layer_col="layer", region_col="region_id")
        self.assertIn("ci_lower", stats.columns)
        self.assertIn("ci_upper", stats.columns)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats["ci_upper"].iloc[0], stats["mean"].iloc[0])
        self.assertLess(stats["ci_lower"].iloc[0], stats["mean"].iloc[0])

    def test_ci_small_sample(self):
        # Test with small sample size using t-distribution
        stats = compute_layer_stats(self.df, value_col="value", layer_col="layer", region_col="region_id")
        self.assertIn("ci_lower", stats.columns)
        self.assertIn("ci_upper", stats.columns)
        self.assertEqual(len(stats), 1)
        self.assertGreater(stats["ci_upper"].iloc[0], stats["mean"].iloc[0])
        self.assertLess(stats["ci_lower"].iloc[0], stats["mean"].iloc[0])

if __name__ == '__main__':
    unittest.main()