"""Lightweight coordinate and factor-equation checks, not scientific evidence."""

import unittest
import numpy as np

try:
    from . import model
except ImportError:
    import model


class GeometryTests(unittest.TestCase):
    def test_sample_map(self):
        rho = 0.5 + 14.1347251417347j
        self.assertAlmostEqual(model.frame_map_zero(rho, 5), 0.2 + 5.65389005669388j)
        self.assertAlmostEqual(model.frame_map_zero(rho, 2), rho)

    def test_inherited_line_and_general_invariant(self):
        zeros = np.array([0.5 + 14.13j, 0.5 + 21.02j])
        general = np.array([0.3 + 14j, 0.8 - 21j])
        for p in [2, 3, 5, 7, 11]:
            points = model.inherited_zero_points(p, zeros)
            np.testing.assert_allclose(points.real, 1 / p)
            np.testing.assert_allclose(p * points / 2, zeros)
            np.testing.assert_allclose(p * model.inherited_zero_points(p, general).real, 2 * general.real)

    def test_comb_axis_and_factor_equation(self):
        for p in [3, 5, 7, 11]:
            for q in model.passed_prime_list(p):
                points = model.passed_prime_comb_points(p, q, 5)
                np.testing.assert_array_equal(points.real, np.zeros(11))
                np.testing.assert_allclose(1 - np.exp(-(p / 2) * points * np.log(q)), 0, atol=2e-14)
                self.assertEqual(points[5], 0)
                np.testing.assert_allclose(points, np.conjugate(points[::-1]))
                np.testing.assert_allclose(np.diff(points.imag), 4 * np.pi / (p * np.log(q)))

    def test_gap(self):
        for p in [2, 3, 5, 7, 11]:
            self.assertEqual(model.frame_gap(p), 1 / p)

    def test_passed_primes_and_recursive_extension(self):
        np.testing.assert_array_equal(model.passed_prime_list(11), [2, 3, 5, 7])
        self.assertEqual(model.passed_prime_list(2).size, 0)
        frames = [2, 3, 5, 7, 11, 13]
        for p, next_p in zip(frames, frames[1:]):
            np.testing.assert_array_equal(model.passed_prime_list(next_p), np.r_[model.passed_prime_list(p), p])

    def test_frame_state_and_inward_motion(self):
        zeros = [0.5 + 14j, 0.5 + 21j]
        states = [model.frame_geometry(p, zeros, 0) for p in [2, 3, 5, 7, 11]]
        self.assertEqual(states[0]["combs"], {})
        for comb in states[-1]["combs"].values():
            np.testing.assert_array_equal(comb, [0j])
        for a, b in zip(states, states[1:]):
            self.assertTrue(np.all(np.abs(b["inherited"]) < np.abs(a["inherited"])))

    def test_first_computed_zero(self):
        zero = model.first_zeta_zeros(1)[0]
        self.assertAlmostEqual(zero.real, 0.5)
        self.assertAlmostEqual(zero.imag, 14.1347251417347, places=11)

    def test_invalid_inputs(self):
        for p in [0, 1, 4, 9, 2.5, True]:
            with self.assertRaises(ValueError):
                model.frame_gap(p)
        for q, k in [(5, 2), (9, 2), (2, -1), (2, 1.5)]:
            with self.assertRaises(ValueError):
                model.passed_prime_comb_points(5, q, k)
        with self.assertRaises(ValueError):
            model.first_zeta_zeros(0)


if __name__ == "__main__":
    unittest.main()
