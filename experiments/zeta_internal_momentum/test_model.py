"""Small numerical identity and CSV-integration tests; no expensive sweeps."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.zeta_internal_momentum import model


class InternalMomentumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gamma, cls.momenta = model.internal_modes(5, kappa=2.3)

    def test_zeros_order_mapping_and_toy_gap(self):
        self.assertTrue(np.all(np.diff(self.gamma) > 0))
        self.assertAlmostEqual(self.gamma[0], 14.134725141734695, places=12)
        np.testing.assert_array_equal(self.momenta, 2.3*self.gamma)
        self.assertGreater(self.momenta[0], 0)
        gap = np.diff(np.r_[0., self.momenta])[0]
        self.assertEqual(gap, 2.3*self.gamma[0])
        self.assertFalse(np.allclose(np.diff(self.gamma), np.diff(self.gamma)[0]))

    def test_kg_equivalence_and_external_rest_in_non_natural_units(self):
        for c in (0.7, 1.0, 2.998):
            p = np.array([-50., -1., 0., 3., 50.])
            P = self.momenta[:, None]
            internal = model.internal_energy(p, P, c)
            standard = model.standard_kg_energy(p, P/c, c)
            np.testing.assert_allclose(internal, standard, rtol=3e-15, atol=3e-14)
            np.testing.assert_allclose(internal[:, 2], c*self.momenta, rtol=2e-15)
            self.assertTrue(np.all(internal[:, 2] > 0))
            self.assertTrue(np.all(internal >= internal[:, 2, None]))
            np.testing.assert_array_equal(internal[:, 0], internal[:, -1])

    def test_dirac_hermiticity_square_and_double_degeneracy(self):
        for c in (1., 2.7):
            for P in (0., *self.momenta[::2]):
                for p in (-13., 0., 8.):
                    H = model.dirac_hamiltonian(p, P, c)
                    E = c*np.sqrt(p*p + P*P)
                    np.testing.assert_array_equal(H, H.conj().T)
                    np.testing.assert_allclose(H@H, E**2*np.eye(4), atol=1e-10, rtol=2e-15)
                    np.testing.assert_allclose(np.linalg.eigvalsh(H), [-E, -E, E, E], atol=5e-13, rtol=2e-15)

    def test_directional_invariants_and_scaling(self):
        for shape in model.directional_shapes():
            self.assertTrue(np.all(shape.weights >= 0))
            A, B, square, P = model.directional_moments(shape)
            self.assertAlmostEqual(A*A - B@B, 1., places=13)
            self.assertAlmostEqual(square, 1., places=13)
            self.assertAlmostEqual(P, 1., places=13)
            for scale in (0.3, self.momenta[0], self.momenta[-1]):
                a, b, inv, p = model.directional_moments(shape.scaled(scale))
                np.testing.assert_allclose([a, *b], scale*np.array([A, *B]), atol=5e-14, rtol=2e-14)
                self.assertAlmostEqual(inv/scale**2, 1., places=13)
                self.assertAlmostEqual(p/scale, 1., places=13)

    def test_smooth_quadrature_matches_analytic_moments(self):
        sphere, two, six, dipole = model.directional_shapes()
        for shape in (sphere, two, six):
            A, B, _, _ = model.directional_moments(shape)
            self.assertAlmostEqual(A, 1., places=14)
            np.testing.assert_allclose(B, 0, atol=2e-16)
        A, B, _, _ = model.directional_moments(dipole)
        # rho=1+0.6*n_x gives |B|/A=0.2; exact normalized moments.
        self.assertAlmostEqual(A, 1/np.sqrt(1-0.2**2), places=14)
        np.testing.assert_allclose(B, [0.2/np.sqrt(1-0.2**2), 0, 0], atol=2e-16)

    def test_diagonal_operator_and_product_space_eigenstates(self):
        operator = model.spectral_operator(self.momenta)
        square = operator @ operator
        for i, P in enumerate(self.momenta):
            chi = np.eye(len(self.momenta))[:, i]
            np.testing.assert_array_equal(operator @ chi, P*chi)
            np.testing.assert_array_equal(square @ chi, P**2*chi)
        np.testing.assert_allclose(np.linalg.eigvalsh(square), self.momenta**2)

    def test_invalid_inputs_and_null_shape(self):
        for kappa in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.internal_modes(1, kappa)
        for weights, directions in (([-1], [[1, 0, 0]]), ([1], [[2, 0, 0]])):
            with self.assertRaises(ValueError):
                model.directional_moments(model.DirectionalState("invalid", np.array(directions), np.array(weights)))
        beam = model.DirectionalState("single null beam", np.array([[1., 0, 0]]), np.ones(1))
        self.assertEqual(model.directional_moments(beam)[3], 0.)
        with self.assertRaises(ValueError):
            model.unit_invariant(beam)
        with self.assertRaises(ValueError):
            model.internal_energy(0, -1)

    def test_optional_missing_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, status = model.load_prime_features(Path(temp)/"absent.csv", self.gamma)
        self.assertEqual(rows, [])
        self.assertIn("absent", status)

    def test_peak_mapping_filters_controls_preserves_missing_and_scales(self):
        gamma = self.gamma[0]
        base = dict(dataset="real", replicate="0", prime_cutoff=11, zero_index=1,
                    gamma=gamma, in_window="True", nearest_peak_tau=gamma+0.1,
                    peak_bracketed="True", nearest_peak_track_id="9", peak_track_switched="False")
        inputs = [base, dict(base, dataset="phase_scrambled"), dict(base, replicate="1"),
                  dict(base, prime_cutoff=29, nearest_peak_tau=""),
                  dict(base, prime_cutoff=101, in_window="False")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"nearest_features.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(base))
                writer.writeheader()
                writer.writerows(inputs)
            rows, _ = model.load_prime_features(path, self.gamma, 3.)
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(rows[0]["momentum_error"], 0.3, places=12)
            self.assertAlmostEqual(rows[0]["P_peak"], 3*(gamma+0.1), places=12)
            self.assertEqual(rows[0]["P_true"], 3*gamma)
            self.assertEqual(rows[0]["peak_track_id"], "9")
            self.assertEqual(rows[1]["P_peak"], "")
            self.assertEqual(rows[2]["momentum_error"], "")
            path.write_text("wrong,columns\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                model.load_prime_features(path, self.gamma)


if __name__ == "__main__":
    unittest.main()
