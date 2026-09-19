"""Small numerical checks of normalization, history, and diagnostic output."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.prime_frame_normalized import model


class NormalizedFrameTests(unittest.TestCase):
    def test_general_and_rh_inherited_coordinates(self):
        zeros = np.array([0.5 + 14.1347251417347j, 0.5 + 21.022j, 0.4 + 14j, 0.7 - 30j])
        for p in model.FRAMES:
            state = model.normalized_frame(p, zeros, 2)
            np.testing.assert_allclose(state["inherited"].real, 2 * zeros.real, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(state["inherited"].imag, 2 * zeros.imag, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(state["inherited"][:2].real, 1, atol=2e-14)
            self.assertAlmostEqual(state["inherited"][2].real, 0.8)
            self.assertAlmostEqual((state["raw"]["inherited"][2].real - 1/p) / (1/p), -0.2)

    def test_comb_equations_and_normalized_heights(self):
        for p in model.FRAMES:
            state = model.normalized_frame(p, [0.5 + 14j], 3)
            for q, comb in state["combs"].items():
                np.testing.assert_array_equal(comb.real, 0)
                np.testing.assert_allclose(comb.imag, 4 * np.pi * np.arange(-3, 4) / np.log(q), atol=2e-14)
                np.testing.assert_allclose(1 - np.exp(-comb / 2 * np.log(q)), 0, atol=2e-14)
                self.assertEqual(comb[3], 0j)

    def test_cross_frame_collapse_and_history(self):
        states = [model.normalized_frame(p, [0.5 + 14j, 0.4 + 21j], 3) for p in model.FRAMES]
        self.assertEqual(states[0]["combs"], {})
        for previous, current in zip(states, states[1:]):
            np.testing.assert_allclose(previous["inherited"], current["inherited"], rtol=2e-14)
            self.assertEqual(set(current["combs"]) - set(previous["combs"]), {previous["frame_prime"]})
            for q, comb in previous["combs"].items():
                np.testing.assert_allclose(comb, current["combs"][q], rtol=2e-14)
            self.assertTrue(np.all(np.abs(current["raw"]["inherited"]) < np.abs(previous["raw"]["inherited"])))

    def test_computed_first_zero_and_diagnostics(self):
        zeros = model.first_zeta_zeros(3)
        self.assertAlmostEqual(2 * zeros[0].imag, 28.2694502834694, places=11)
        states = [model.normalized_frame(p, zeros, 2) for p in model.FRAMES]
        rows = list(model.diagnostic_rows(states, zeros, 0.4 + zeros[0].imag * 1j))
        self.assertLess(model.verify_invariants(rows), 1e-12)
        self.assertEqual(len(rows), 8 * (6 + 1) + sum(len(s["combs"]) * 5 for s in states))
        origins = [r for r in rows if r["frame_prime"] == 19 and r["k"] == 0]
        self.assertEqual(len(origins), 7)
        rows[0]["normalized_X"] += 0.01
        with self.assertRaises(AssertionError):
            model.verify_invariants(rows)

    def test_zero_only_combs_and_invalid_inputs(self):
        for comb in model.normalized_frame(7, [0.5 + 14j], 0)["combs"].values():
            np.testing.assert_array_equal(comb, [0j])
        for p, points in [(4, [1j]), (True, [1j]), (3, [np.nan]), (3, [[1j]])]:
            with self.assertRaises(ValueError):
                model.normalize_points(p, points)


if __name__ == "__main__":
    unittest.main()
