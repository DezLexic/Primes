"""Small deterministic numerical tests; no large scientific sweep."""

import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.prime_zero_features import model
from experiments.prime_zero_features.run import summarize
from experiments.prime_factor_phasors.model import factor_power
from experiments.prime_frame.model import log_history_power


class FeatureTests(unittest.TestCase):
    def test_log_matches_direct_product_and_previous_experiment(self):
        tau = np.linspace(0, 35, 301)
        signal = model.make_signal(29)
        direct = np.prod(1-np.exp(-(0.5+1j*tau[:, None])*np.log(model.primes_through(29))), axis=1)
        result = model.evaluate(signal, tau)
        np.testing.assert_allclose(result["L"], np.log(abs(direct)**2), atol=2e-13)
        np.testing.assert_allclose(np.exp(1j*result["phase"]), direct/abs(direct), atol=2e-13)
        np.testing.assert_allclose(result["L"], log_history_power(tau, 31), atol=2e-13)
        np.testing.assert_allclose(model.safe_power(result["L"]), abs(direct)**2, rtol=2e-13)

    def test_factor_formula(self):
        tau = np.linspace(-20, 20, 101)
        for q in (2, 3, 11, 29):
            power = 1+1/q-2/np.sqrt(q)*np.cos(tau*np.log(q))
            np.testing.assert_allclose(factor_power(q, 0.5, tau), power, atol=3e-15)
            np.testing.assert_allclose(abs(1-q**(-0.5-1j*tau))**2, power, atol=3e-15)

    def test_inclusive_prime_cutoffs(self):
        np.testing.assert_array_equal(model.primes_through(2), [2])
        np.testing.assert_array_equal(model.primes_through(11), [2, 3, 5, 7, 11])
        np.testing.assert_array_equal(model.primes_through(12), [2, 3, 5, 7, 11])
        for bad in (True, 1, 3.5):
            with self.assertRaises(ValueError):
                model.primes_through(bad)

    def test_first_twenty_zeros(self):
        zeros = model.first_zeta_zeros(20, 30)
        self.assertEqual(len(zeros), 20)
        self.assertTrue(np.all(np.diff(zeros.imag) > 0))
        self.assertAlmostEqual(zeros[0].imag, 14.134725141734695, places=12)
        np.testing.assert_array_equal(zeros.real, np.full(20, 0.5))

    def test_refinement_brackets_and_resolution(self):
        signal = model.make_signal(29)
        found = []
        for points in (2001, 4001):
            tau = np.linspace(0, 35, points)
            features = model.detect_extrema(tau, model.evaluate(signal, tau)["L"], signal)
            self.assertGreater(len(features), 10)
            for f in features:
                self.assertLess(f["bracket_left"], f["tau"])
                self.assertLess(f["tau"], f["bracket_right"])
                self.assertTrue(f["refinement_ok"])
                self.assertGreaterEqual(f["prominence"], 0.1)
                self.assertGreater(f["width_tau"], 0)
            found.append(features)
        self.assertTrue(all(row["passed"] for row in model.resolution_comparison(*found, 1e-7)))
        # Missing a feature cannot pass on the strength of the others' alignment.
        self.assertFalse(all(row["passed"] for row in model.resolution_comparison(found[0][1:], found[1], 1e-7)))

    def test_scramble_is_deterministic_and_nested(self):
        a = model.make_signal(101, "phase_scrambled", 123, 2)
        b = model.make_signal(101, "phase_scrambled", 123, 2)
        small = model.make_signal(11, "phase_scrambled", 123, 2)
        np.testing.assert_array_equal(a.phases, b.phases)
        np.testing.assert_array_equal(small.phases, a.phases[:len(small.phases)])
        np.testing.assert_array_equal(model.evaluate(a, [0, 14, 20])["L"], model.evaluate(b, [0, 14, 20])["L"])
        self.assertFalse(np.array_equal(a.phases, model.make_signal(101, "phase_scrambled", 123, 3).phases))

    def test_shuffle_preserves_frequency_and_amplitude_sets(self):
        actual = model.make_signal(101)
        shuffled = model.make_signal(101, "shuffled_rates", 123, 2)
        np.testing.assert_array_equal(actual.amplitudes, shuffled.amplitudes)
        np.testing.assert_array_equal(actual.rates, np.sort(shuffled.rates))
        np.testing.assert_array_equal(shuffled.rates, model.make_signal(101, "shuffled_rates", 123, 2).rates)
        self.assertFalse(np.array_equal(actual.rates, shuffled.rates))

    def test_detection_is_zero_blind(self):
        self.assertNotIn("gamma", inspect.signature(model.detect_extrema).parameters)
        signal = model.make_signal(29)
        tau = np.linspace(0, 35, 1001)
        values = model.evaluate(signal, tau)["L"]
        with patch.object(model, "first_zeta_zeros", side_effect=AssertionError("detector must not request zeros")):
            before = model.detect_extrema(tau, values, signal)
            model.compare_zeros(np.array([0.5+14j]), signal, before, 0, 35)
            model.compare_zeros(np.array([0.5+18j, 0.5+100j]), signal, before, 0, 35)
            after = model.detect_extrema(tau, values, signal)
        self.assertEqual(before, after)

    def test_analytic_derivatives_for_real_and_controls(self):
        tau = np.array([0.1, 2.3, 14.13, 21.0])
        h = 1e-5
        for kind in ("real", "phase_scrambled", "shuffled_rates", "raised_cosine"):
            signal = model.make_signal(29, kind)
            center, left, right = [model.evaluate(signal, t) for t in (tau, tau-h, tau+h)]
            np.testing.assert_allclose(center["slope"], (right["L"]-left["L"])/(2*h), atol=1e-7)
            np.testing.assert_allclose(center["curvature"], (right["slope"]-left["slope"])/(2*h), atol=3e-6)
            np.testing.assert_allclose(center["phase_slope"], (right["phase"]-left["phase"])/(2*h), atol=1e-7)

    def test_missing_and_outside_features_are_not_zero_offsets(self):
        signal = model.make_signal(5)
        zeros = np.array([0.5+14j, 0.5+40j])
        rows = model.compare_zeros(zeros, signal, [], 0, 35)
        self.assertTrue(rows[0]["in_window"])
        self.assertFalse(rows[1]["in_window"])
        self.assertEqual(rows[0]["peak_offset"], "")
        self.assertEqual(rows[1]["min_offset"], "")
        self.assertTrue(np.isfinite(rows[1]["L_at_gamma"]))
        tau = np.linspace(0, 35, 1001)
        found = model.detect_extrema(tau, model.evaluate(signal, tau)["L"], signal, prominence=1e6)
        self.assertEqual(found, [])

    def test_uniform_grid_validation(self):
        signal = model.make_signal(5)
        for grid in ([0, 1, 3], [0, 0, 0], [0, 1, np.nan]):
            with self.assertRaises(ValueError):
                model.detect_extrema(np.array(grid), np.array([0, 1, 0]), signal)

    def test_safe_power_marks_unrepresentable_values(self):
        result = model.safe_power(np.array([-1000, 0, 1000]))
        self.assertTrue(np.isnan(result[0]) and np.isnan(result[2]))
        self.assertEqual(result[1], 1)

    def test_common_control_cohort(self):
        comparisons = []
        for kind in ("real", "phase_scrambled"):
            for index in (1, 2):
                comparisons.append(dict(dataset=kind, replicate=0, prime_cutoff=5, zero_index=index,
                                        in_window=True, peak_bracketed=(kind == "real" or index == 1),
                                        min_bracketed=True, peak_offset=0.2*index, min_offset=-0.3*index))
        rows = summarize(comparisons, [], [5])
        peaks = [r for r in rows if r["cohort"] == "common_bracketed" and r["kind"] == "peak"]
        self.assertTrue(all(r["zero_indices"] == "1" and r["matched_zeros"] == 1 for r in peaks))

    def test_tracking_does_not_merge_two_features_into_one(self):
        old = [dict(kind="peak", tau=10, track_id=1)]
        new = [dict(kind="peak", tau=10.1), dict(kind="peak", tau=10.2), dict(kind="min", tau=10.1)]
        model.assign_tracks(old, new, 2, 0.5)
        self.assertEqual(new[0]["track_id"], 1)
        self.assertEqual(len(set(r["track_id"] for r in new)), 3)


if __name__ == "__main__":
    unittest.main()
