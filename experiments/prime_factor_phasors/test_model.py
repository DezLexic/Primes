"""Lightweight numerical checks, including all seven requested identities."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.prime_factor_phasors import model
from experiments.prime_frame import model as previous


class PhasorTests(unittest.TestCase):
    def test_complex_cosine_and_stable_power_agree(self):
        tau = np.linspace(-20, 20, 121)
        for q in model.PRIMES:
            for sigma in (-0.5, 0, 0.5):
                a1, a2, total = model.phasors(q, sigma, tau)
                cosine = 1 + q**(-2*sigma) - 2*q**(-sigma)*np.cos(tau*np.log(q))
                np.testing.assert_allclose(total, a1+a2, atol=3e-15)
                np.testing.assert_allclose(abs(total)**2, cosine, atol=1e-14)
                np.testing.assert_allclose(abs(total)**2, model.factor_power(q, sigma, tau), atol=1e-14)

    def test_analytic_factor_zeros(self):
        for q in model.PRIMES:
            _, tau = model.zero_heights(q, 50)
            z = 1j*tau
            np.testing.assert_array_equal(z.real, 0)
            np.testing.assert_allclose(model.phasors(q, 0, tau)[2], 0, atol=2e-14)

    def test_positive_minimum(self):
        for q in model.PRIMES:
            _, tau = model.zero_heights(q, 50)
            expected = (1-1/np.sqrt(q))**2
            self.assertGreater(expected, 0)
            np.testing.assert_allclose(model.factor_power(q, 0.5, tau), expected, atol=2e-15)
            power = model.factor_power(q, 0.5, np.linspace(-20, 20, 501))
            self.assertTrue(np.all(power >= expected-2e-15))
            for sigma in (-0.5, 0.5):
                self.assertGreater(float(model.minimum_power(q, sigma)), 0)

    def test_frame_real_parts_and_normalized_heights(self):
        rows = list(model.analytic_rows(model.TRACK_FRAMES, 30))
        for row in rows:
            q, k, p = row["source_prime_q"], row["k"], row["frame_prime"]
            self.assertEqual(row["z_real"], 0)
            self.assertEqual(row["frame_real"], 0)
            self.assertEqual(row["normalized_X"], 0)
            self.assertAlmostEqual(row["normalized_Y"], 4*np.pi*k/np.log(q), places=12)
            self.assertAlmostEqual(row["frame_imag"]*p/2, row["z_imag"], places=12)
        for q in model.PRIMES:
            copies = [r["normalized_Y"] for r in rows if r["source_prime_q"] == q and r["k"] == 1]
            np.testing.assert_allclose(copies, 4*np.pi/np.log(q), atol=2e-14)

    def test_any_factor_zero_annihilates_product(self):
        for p in model.HISTORY_FRAMES:
            for q in model.passed_prime_list(p):
                _, tau = model.zero_heights(int(q), 30)
                np.testing.assert_allclose(model.history_power(p, 0, tau), 0, atol=1e-22)
                np.testing.assert_allclose(model.history_factor(1j*tau, p), 0, atol=1e-11)

    def test_previous_history_and_recursion(self):
        tau = np.linspace(-20, 20, 101)
        for p in model.HISTORY_FRAMES:
            np.testing.assert_allclose(model.history_power(p, 0.5, tau), previous.history_power(tau, p), rtol=2e-14)
            for sigma in (0, 0.5):
                np.testing.assert_allclose(model.history_power(p, sigma, tau),
                                           abs(previous.history_factor(sigma+1j*tau, p))**2, atol=1e-12, rtol=2e-14)
        for p, following in zip(model.HISTORY_FRAMES, model.HISTORY_FRAMES[1:]):
            np.testing.assert_allclose(model.history_power(following, 0.5, tau),
                                       model.history_power(p, 0.5, tau)*model.factor_power(p, 0.5, tau), rtol=2e-14)
        np.testing.assert_array_equal(model.history_power(2, 0, tau), np.ones_like(tau))

    def test_grid_and_product_are_not_a_sum(self):
        grid = model.sampling_grid(20, 101, model.PRIMES)
        for q in model.PRIMES:
            self.assertTrue(np.all(np.isin(model.zero_heights(q, 20)[1], grid)))
        tau = 2*np.pi/np.log(2)
        self.assertLess(float(model.history_power(5, 0, tau)), 1e-25)
        summed = sum(model.phasors(q, 0, tau)[2] for q in (2, 3))
        self.assertGreater(float(abs(summed)**2), 0.1)

    def test_invalid_inputs(self):
        for q in (1, 4, True):
            with self.assertRaises(ValueError):
                model.factor_power(q, 0, 1)
        with self.assertRaises(ValueError):
            model.zero_heights(2, -1)
        with self.assertRaises(ValueError):
            model.phasors(2, 0, np.nan)


if __name__ == "__main__":
    unittest.main()
