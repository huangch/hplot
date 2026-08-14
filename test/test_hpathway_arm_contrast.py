import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

from hplot import hpathway_arm_contrast


def _profiles(n_per_arm=5, layers=range(-5, 6), effect=0.0, seed=0,
              names=("SIGNAL", "FLAT")):
    """Two arms of patients; `effect` shifts arm B's SIGNAL profile at positive layers."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm, tag in ((0, "A"), (1, "B")):
        for r in range(n_per_arm):
            for L in layers:
                sig = 0.02 * L + rng.normal(0, 0.002)
                if arm == 1 and L > 0:
                    sig += effect
                rows.append({"patient": f"{tag}{r}", "layer": int(L), "n_cells": 400,
                             names[0]: sig, names[1]: rng.normal(0, 0.002)})
    return pd.DataFrame(rows)


class TestArmContrast(unittest.TestCase):
    names = ["SIGNAL", "FLAT"]
    layers = list(range(-5, 6))

    def _arm_of(self, prof):
        return {u: ("B" if str(u).startswith("B") else "A")
                for u in prof["patient"].unique()}

    def test_exhaustive_null_when_feasible(self):
        prof = _profiles(n_per_arm=4)
        _, summ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col="n_cells", verbose=False)
        # C(8,4) = 70 assignments enumerated exhaustively -> p floor 1/71
        self.assertTrue(bool(summ["exact"].all()))
        self.assertEqual(int(summ["n_assignments"].iloc[0]), 70)
        self.assertAlmostEqual(float(summ["p_floor"].iloc[0]), 1.0 / 71.0, places=9)
        self.assertTrue((summ["p_exact"] >= 1.0 / 71.0 - 1e-9).all())

    def test_planted_difference_is_detected(self):
        prof = _profiles(n_per_arm=5, effect=0.05, seed=1)
        grid, summ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col="n_cells", verbose=False)
        top = summ.iloc[0]
        self.assertEqual(top["pathway"], "SIGNAL")
        self.assertLess(float(top["p_exact"]), 0.05)
        self.assertGreater(float(top["peak_layer"]), 0)   # planted on the stroma side
        # the flat signature must not be dragged along
        flat = summ[summ.pathway == "FLAT"].iloc[0]
        self.assertGreater(float(flat["p_exact"]), 0.05)

    def test_null_data_is_not_significant(self):
        prof = _profiles(n_per_arm=5, effect=0.0, seed=2)
        _, summ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col="n_cells", verbose=False)
        self.assertTrue((summ["p_exact"] > 0.05).all())

    def test_gap_sign_names_the_higher_arm(self):
        prof = _profiles(n_per_arm=5, effect=0.05, seed=3)
        grid, _ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col="n_cells", verbose=False)
        stroma = grid[(grid.pathway == "SIGNAL") & (grid.layer > 0)]
        self.assertGreater(float(stroma["gap"].mean()), 0.0)   # arm B is higher

    def test_paired_null_swaps_within_pair_only(self):
        prof = _profiles(n_per_arm=4, seed=4)
        arm = self._arm_of(prof)
        pair = {u: str(u)[1:] for u in prof["patient"].unique()}   # A0<->B0, A1<->B1, ...
        _, summ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers, arm_of=arm, pair_of=pair,
            count_col="n_cells", verbose=False)
        # 2**4 = 16 sign flips, not C(8,4) = 70
        self.assertEqual(int(summ["n_assignments"].iloc[0]), 16)
        self.assertTrue(bool(summ["exact"].all()))

    def test_paired_requires_two_units_per_pair(self):
        prof = _profiles(n_per_arm=3, seed=5)
        arm = self._arm_of(prof)
        bad = {u: "onepair" for u in prof["patient"].unique()}
        with self.assertRaises(ValueError):
            hpathway_arm_contrast(prof, path_names=self.names, grid=self.layers,
                                  arm_of=arm, pair_of=bad, count_col="n_cells",
                                  verbose=False)

    def test_three_arms_rejected(self):
        prof = _profiles(n_per_arm=3, seed=6)
        arm = self._arm_of(prof)
        arm[list(arm)[0]] = "C"
        with self.assertRaises(ValueError):
            hpathway_arm_contrast(prof, path_names=self.names, grid=self.layers,
                                  arm_of=arm, count_col="n_cells", verbose=False)

    def test_units_without_an_arm_are_dropped(self):
        prof = _profiles(n_per_arm=4, seed=7)
        arm = self._arm_of(prof)
        arm[list(arm)[0]] = None
        _, summ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers, arm_of=arm,
            count_col="n_cells", verbose=False)
        # 7 units left, 3 in one arm -> C(7,3) = 35
        self.assertEqual(int(summ["n_assignments"].iloc[0]), 35)

    def test_monte_carlo_when_capped(self):
        prof = _profiles(n_per_arm=6, seed=8)
        _, summ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers, arm_of=self._arm_of(prof),
            count_col="n_cells", n_perm=50, verbose=False)
        self.assertFalse(bool(summ["exact"].any()))
        self.assertEqual(int(summ["n_assignments"].iloc[0]), 50)

    def test_resolution_warning_is_printed(self, ):
        import io
        import contextlib
        prof = _profiles(n_per_arm=3, seed=9)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hpathway_arm_contrast(prof, path_names=self.names, grid=self.layers,
                                  arm_of=self._arm_of(prof), count_col="n_cells",
                                  verbose=True)
        out = buf.getvalue()
        # the report must say how many tests at the p floor BH would need, not quote
        # p_floor x m as if a single extreme test were the only route to alpha
        self.assertIn("smallest attainable p", out)
        self.assertIn("sit at that floor", out)
        self.assertIn("q_row", out)

    def test_null_reference_columns(self):
        prof = _profiles(n_per_arm=4, seed=10)
        grid, _ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col="n_cells", verbose=False)
        for col in ("null_ref", "ratio_vs_null"):
            self.assertIn(col, grid.columns)
        ok = grid["null_ref"].notna()
        self.assertTrue((grid.loc[ok, "null_ref"] >= 0).all())
        # ratio is the effect expressed in units of chance, so 1.0 is the reference
        ratio = (grid.loc[ok, "abs_gap"] / grid.loc[ok, "null_ref"]).to_numpy()
        self.assertTrue(np.allclose(ratio, grid.loc[ok, "ratio_vs_null"].to_numpy(),
                                    equal_nan=True))

    def test_null_data_rarely_exceeds_the_chance_reference(self):
        prof = _profiles(n_per_arm=5, effect=0.0, seed=11)
        grid, _ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col="n_cells", verbose=False)
        frac = float((grid["ratio_vs_null"] > 1.0).mean())
        # a 95th-percentile reference is exceeded ~5% of the time under the null;
        # allow slack for 11 layers x 2 sets
        self.assertLess(frac, 0.30)

    def test_peak_is_read_off_the_standardised_profile(self):
        """A noisy layer can carry the largest raw gap without being the peak."""
        rng = np.random.default_rng(0)
        layers = list(range(6))
        rows = []
        for tag in ("A", "B"):
            for r in range(5):
                # layer 0 swings wildly between units -> wide null, weak evidence.
                # layer 3 is tight and carries a small consistent arm offset.
                for L in layers:
                    if L == 0:
                        v = rng.normal(0, 0.5)
                    elif L == 3:
                        v = rng.normal(0, 0.001) + (0.02 if tag == "B" else 0.0)
                    else:
                        v = rng.normal(0, 0.001)
                    rows.append({"patient": f"{tag}{r}", "layer": L,
                                 "n_cells": 500, "SET": v})
        prof = pd.DataFrame(rows)
        arm = {u: ("B" if str(u).startswith("B") else "A")
               for u in prof["patient"].unique()}
        grid, summ = hpathway_arm_contrast(
            prof, path_names=["SET"], grid=layers, arm_of=arm,
            sample_col="patient", count_col="n_cells", verbose=False)
        raw = int(grid.loc[grid["abs_gap"].idxmax(), "layer"])
        std = int(grid.loc[grid["ratio_vs_null"].idxmax(), "layer"])
        self.assertIn("max_ratio_vs_null", summ.columns)
        self.assertEqual(int(summ["peak_layer"].iloc[0]), std)
        self.assertEqual(raw, 0)     # the noisy layer wins on the raw scale
        self.assertEqual(std, 3)     # the informative layer wins once standardised

    def test_min_cells_drops_sparse_unit_layers(self):
        prof = _profiles(n_per_arm=5, seed=12)
        prof.loc[prof["layer"] == -5, "n_cells"] = 3      # sparse inner edge
        arm = self._arm_of(prof)
        wide, _ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers, arm_of=arm,
            count_col="n_cells", min_cells=0, verbose=False)
        gated, _ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers, arm_of=arm,
            count_col="n_cells", min_cells=50, verbose=False)
        self.assertTrue(wide.loc[wide.layer == -5, "abs_gap"].notna().any())
        # every unit is below the gate at that layer, so the layer cannot be tested
        self.assertTrue(gated.loc[gated.layer == -5, "abs_gap"].isna().all())
        self.assertTrue(gated.loc[gated.layer > -5, "abs_gap"].notna().any())

    def test_min_cells_needs_counts_to_do_anything(self):
        prof = _profiles(n_per_arm=4, seed=13)
        grid, _ = hpathway_arm_contrast(
            prof, path_names=self.names, grid=self.layers,
            arm_of=self._arm_of(prof), count_col=None, min_cells=10_000,
            verbose=False)
        self.assertTrue(grid["abs_gap"].notna().any())


if __name__ == "__main__":
    unittest.main()
