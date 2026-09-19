"""Small numerical checks; no research-scale parameter sweep."""

import unittest
import numpy as np

try:
    from . import model
except ImportError:
    import model


class HistoryTests(unittest.TestCase):
    def test_primes(self):
        np.testing.assert_array_equal(model.primes_below(12), [2, 3, 5, 7, 11])
        np.testing.assert_array_equal(model.primes_below(11), [2, 3, 5, 7])
        self.assertEqual(len(model.primes_below(2)), 0)
        with self.assertRaises(ValueError):
            model.chi(9)

    def test_recursive_identity(self):
        z = np.array([0.5, 0.5 + 1.2j, 2.3 + 7j])
        frames = [2, 3, 5, 7, 11, 13]
        for p, next_p in zip(frames, frames[1:]):
            np.testing.assert_allclose(model.history_factor(z, next_p),
                                       (1 - float(p)**(-z)) * model.history_factor(z, p), rtol=2e-14)

    def test_cosine_complex_and_log_products(self):
        tau = np.array([-12, -0.3, 0, 0.01, 2.5, 14])
        for p in (2, 3, 7, 29, 101):
            q = model.primes_below(p).astype(float)
            direct = np.prod(1 + 1/q[:, None] - 2/np.sqrt(q[:, None])
                             * np.cos(tau * np.log(q[:, None])), axis=0)
            complex_power = np.abs(model.history_factor(0.5 + 1j*tau, p))**2
            np.testing.assert_allclose(direct, complex_power, rtol=3e-13, atol=0)
            np.testing.assert_allclose(np.exp(model.log_history_power(tau, p)), direct, rtol=3e-13, atol=0)

    def test_central_product(self):
        for p in (2, 5, 11, 29, 101):
            direct = np.prod((1 - 1/np.sqrt(model.primes_below(p)))**2)
            np.testing.assert_allclose(model.history_power(0, p), direct, rtol=2e-14, atol=0)

    def test_curvature(self):
        for p in (3, 7, 29, 101):
            for step in (1e-3, 1e-4):
                np.testing.assert_allclose(model.numerical_chi(p, step), model.chi(p), rtol=6e-6)

    def test_coordinate_mapping_and_widths(self):
        for p in (3, 5, 29):
            t = np.array([-2.0, 0, 3.0])
            z = model.to_common_coordinate(1/p + 1j*t, p)
            np.testing.assert_allclose(z.real, 0.5)
            np.testing.assert_allclose(z.imag, p*t/2)
            self.assertAlmostEqual(p * model.quadratic_width_frame(p)/2,
                                   model.quadratic_width_common(p))

    def test_empty_product(self):
        self.assertEqual(model.history_power(0, 2), 1)
        self.assertEqual(model.baseline(2), 1)
        self.assertEqual(model.chi(2), 0)
        self.assertTrue(np.isinf(model.quadratic_width_common(2)))

    def test_sweep_matches_independent_products(self):
        tau = np.linspace(-3, 3, 61)
        rows, log_profiles = model.run_frame_sweep(101, tau)
        for row, profile in zip(rows, log_profiles):
            p = row['frame_prime']
            np.testing.assert_allclose(np.exp(profile), model.normalized_history_power(tau, p), rtol=2e-13)
            self.assertAlmostEqual(row['chi'], model.chi(p))
            self.assertAlmostEqual(row['log_W0'], float(model.log_history_power(0, p)))
            self.assertAlmostEqual(row['B'], np.prod(1 + 1/model.primes_below(p)))


if __name__ == '__main__':
    unittest.main()
