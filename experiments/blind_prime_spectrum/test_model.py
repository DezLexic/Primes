"""Small numerical and architectural checks; never a normal/deep sweep."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.blind_prime_spectrum import artifacts, blind_model as model, evaluate, generate, run


def peak(tau, cutoff=5, tid=1):
    return dict(tau=tau, prime_cutoff=cutoff, track_id=tid, prominence=2., width=0.3,
                value=3., curvature=-2., refinement_ok=True, feature_id="peak_1", refinement="derivative_root")


def metric(tau, tid=1):
    return dict(track_id=tid, observations=10, tail_count=3, persistence=1., tail_drift=0.1,
                tail_slope=0.05, tail_tau_median=tau, median_prominence=1.)


class BlindTests(unittest.TestCase):
    def test_blind_import_graph_has_no_zero_module(self):
        code = "from experiments.blind_prime_spectrum import generate, run; import sys; assert 'experiments.prime_frame_geometry.model' not in sys.modules; assert 'experiments.blind_prime_spectrum.evaluate' not in sys.modules; assert 'mpmath' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True)

    def test_zero_blind_api_signatures(self):
        for fn in (model.detect_peaks, model.detect_extrema, model.Tracker.step, model.track_metrics,
                   model.stable_track, model.consensus_spectrum, generate.generate):
            self.assertFalse(any("gamma" in p or "zero" in p for p in inspect.signature(fn).parameters))

    def test_schedule_and_presets(self):
        normal = run.parse_args([])
        self.assertGreaterEqual(normal.cutoffs[-1], 10000)
        self.assertTrue(40 <= len(normal.cutoffs) <= 80)
        self.assertEqual(tuple(run.parse_args(["--mode", "smoke"]).cutoffs), model.SMOKE_CUTOFFS)
        self.assertEqual(tuple(normal.cutoffs), tuple(sorted(set(normal.cutoffs))))

    def test_signal_reuses_stable_formula_and_taper(self):
        cutoffs = (5, 11, 29, 101)
        primes, bank = model.signal_bank(cutoffs, "real", 1729, 0)
        tau = np.linspace(0, 15, 101)
        for mode in ("hard", "taper"):
            signal = model.cutoff_signal(primes, bank, 29, mode)
            q = model.primes_through(29)
            weights = np.ones(len(q)) if mode == "hard" else (1+np.cos(np.pi*q/29))/2
            direct = np.sum(weights[:, None]*np.log(abs(1-q[:, None]**(-0.5-1j*tau))**2), axis=0)
            np.testing.assert_allclose(model.evaluate(signal, tau)["L"], direct, atol=1e-13)

    def test_controls_are_reproducible_nested_and_preserve_rates(self):
        cutoffs = (5, 11, 29, 101)
        for dataset in ("phase_scrambled", "shuffled_rates"):
            primes, first = model.signal_bank(cutoffs, dataset, 1729, 1)
            _, second = model.signal_bank(cutoffs, dataset, 1729, 1)
            _, other = model.signal_bank(cutoffs, dataset, 1729, 2)
            for name in ("amplitudes", "rates", "phases"):
                np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
            key = "phases" if dataset == "phase_scrambled" else "rates"
            self.assertFalse(np.array_equal(getattr(first, key), getattr(other, key)))
            for cutoff in cutoffs:
                small = model.cutoff_signal(primes, first, cutoff, "hard")
                np.testing.assert_array_equal(np.sort(small.rates), np.log(model.primes_through(cutoff)))
                np.testing.assert_array_equal(small.rates, first.rates[:len(small.rates)])
                np.testing.assert_array_equal(small.amplitudes, 1/np.sqrt(model.primes_through(cutoff)))
                _, short_bank = model.signal_bank(tuple(p for p in cutoffs if p <= cutoff), dataset, 1729, 1)
                np.testing.assert_array_equal(small.rates, short_bank.rates)
                np.testing.assert_array_equal(small.phases, short_bank.phases)

    def test_tracking_missing_reconnect_death_and_birth(self):
        tracker = model.Tracker(max_gap=1)
        a = [peak(5.)]
        tracker.step(a, 0)
        first_id = a[0]["track_id"]
        tracker.step([], 1)
        b = [peak(5.1)]
        diagnostic = tracker.step(b, 2)
        self.assertEqual(b[0]["track_id"], first_id)
        self.assertEqual(diagnostic[0]["event"], "reconnect")
        tracker.step([], 3)
        tracker.step([], 4)
        c = [peak(5.1)]
        diagnostic = tracker.step(c, 5)
        self.assertNotEqual(c[0]["track_id"], first_id)
        self.assertIn("death", [r["event"] for r in diagnostic])

    def test_split_ambiguity_does_not_force_continuation(self):
        tracker = model.Tracker()
        a = [peak(5.)]
        tracker.step(a, 0)
        b = [peak(4.98), peak(5.02)]
        events = tracker.step(b, 1)
        self.assertEqual(len({p["track_id"] for p in b}), 2)
        self.assertTrue(all(p["track_id"] != a[0]["track_id"] for p in b))
        self.assertEqual(sum(r["event"] == "ambiguous_birth" for r in events), 2)

    def test_empty_and_failed_peaks(self):
        tracker = model.Tracker()
        self.assertEqual(tracker.step([], 0), [])
        bad = {**peak(3), "refinement_ok": False, "track_id": ""}
        events = tracker.step([bad], 1)
        self.assertEqual(bad["track_id"], "")
        self.assertEqual(events[0]["event"], "refinement_failed")
        self.assertEqual(model.track_metrics([], (5, 7, 11, 17), model.Rules()), [])

    def test_metrics_use_global_tail_denominator(self):
        cutoffs = (5, 7, 11, 17, 29, 47, 71, 101)
        rules = model.Rules(tail_fraction=0.5, min_tail_count=3)
        rows = [peak(10., 5), peak(10.1, 11), peak(10.2, 29), peak(10.3, 71), peak(10.25, 101)]
        result = model.track_metrics(rows, cutoffs, rules)[0]
        self.assertEqual(result["eligible_cutoffs"], 4)
        self.assertEqual(result["tail_count"], 3)
        self.assertEqual(result["persistence"], 0.75)
        self.assertAlmostEqual(result["tail_drift"], 0.1)
        self.assertAlmostEqual(result["tail_tau_mean"], 10.25)
        self.assertAlmostEqual(result["tail_tau_median"], 10.25)
        self.assertAlmostEqual(result["tail_tau_std"], np.std([10.2, 10.3, 10.25]))
        expected = abs(np.polyfit(np.log([29, 71, 101]), [10.2, 10.3, 10.25], 1)[0])
        self.assertAlmostEqual(result["tail_slope"], expected)
        late = model.track_metrics([peak(10., 101)], cutoffs, rules)[0]
        self.assertEqual(late["persistence"], 0.25)
        self.assertFalse(late["stable"])

    def test_converging_stable_and_wandering_unstable(self):
        cutoffs = tuple(range(10, 110, 10))
        convergent = [peak(10+1/p, p) for p in cutoffs]
        wandering = [peak(10+(-1)**i, p) for i, p in enumerate(cutoffs)]
        self.assertTrue(model.track_metrics(convergent, cutoffs, model.Rules())[0]["stable"])
        self.assertFalse(model.track_metrics(wandering, cutoffs, model.Rules())[0]["stable"])

    def test_consensus_uses_only_blind_metrics_and_one_to_one(self):
        rules = model.Rules()
        hard = [metric(10., 1), metric(10.1, 2), metric(20., 3)]
        taper = [metric(10.02, 10), metric(20.2, 11)]
        found = model.consensus_spectrum(hard, taper, rules)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["hard_track_id"], 1)
        self.assertAlmostEqual(found[0]["consensus_tau"], 10.01)
        self.assertEqual(len({r["taper_track_id"] for r in found}), len(found))
        taper[0]["tail_drift"] = 1.
        self.assertEqual(len(model.consensus_spectrum(hard, taper, rules)), 1)

    def test_optional_momentum(self):
        result = model.internal_momentum([dict(predicted_index=1, consensus_tau=12.3)], 2.)
        self.assertEqual(result[0]["predicted_P_int"], 24.6)

    def test_evaluation_ordered_nearest_missing_extra_duplicates(self):
        spectrum = [dict(consensus_tau=t) for t in [10., 4.1, 4.05]]
        rows = evaluate.compare_spectrum(spectrum, [4., 8., 12., 20.])
        self.assertEqual([r["predicted_tau"] for r in rows], [4.05, 4.1, 10.])
        self.assertEqual(rows[1]["zero_index"], 2)
        self.assertEqual(rows[1]["nearest_zero_index"], 1)
        self.assertAlmostEqual(rows[1]["signed_error"], -3.9)
        self.assertAlmostEqual(rows[1]["relative_error"], 3.9/8)
        failures = evaluate.failure_diagnostics(rows, [4., 8., 12., 20.], 16., 0.5)
        self.assertEqual(sum(r["kind"] == "missing" for r in failures), 2)
        self.assertEqual(sum(r["kind"] == "extra" for r in failures), 2)
        self.assertIn("duplicate_near_reference", [r["reason"] for r in failures])

    def test_no_freeze_no_loader(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(evaluate, "load_reference") as loader:
            with self.assertRaises(FileNotFoundError):
                evaluate.evaluate_frozen(folder)
            loader.assert_not_called()

    def test_complete_blind_phase_forbids_zeros_and_freezes_before_evaluation(self):
        import mpmath
        from experiments.prime_frame_geometry import model as zero_model
        from experiments.prime_zero_features import model as old_model
        from experiments.zeta_internal_momentum import model as momentum_model
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder) / "run"
            for target, name in ((mpmath, "zetazero"), (mpmath.mp, "zetazero"), (zero_model, "first_zeta_zeros"),
                                 (zero_model, "_zeta_zeros"), (old_model, "first_zeta_zeros"), (momentum_model, "first_zeta_zeros")):
                stack.enter_context(patch.object(target, name, side_effect=AssertionError("zero loader called during blind phase")))
            detector = stack.enter_context(patch.object(generate, "detect_peaks", wraps=model.detect_peaks))
            classifier = stack.enter_context(patch.object(generate, "track_metrics", wraps=model.track_metrics))
            rules = model.Rules(tail_fraction=0.5, min_tail_count=2, min_observations=2)
            generate.generate(output, (5, 7, 11, 17), rules, tau_max=16., points=401,
                              control_replicates=2, kappa=2., progress=lambda _: None)
            self.assertEqual(len(classifier.call_args_list), 10)  # same path for 5 groups x 2 cutoff modes
            self.assertTrue(all(call.args[2] == rules for call in classifier.call_args_list))
            self.assertTrue(all(call.args[2] == 0.1 for call in detector.call_args_list))
            for path in output.glob("blind_*.csv"):
                with path.open(encoding="utf-8") as handle:
                    self.assertNotIn("gamma", handle.readline().lower())
            frozen = artifacts.verify_freeze(output)
            before = {name: (output / name).read_bytes() for name in frozen["files"]}

            def reference(tau_max, minimum_count, dps):
                artifacts.verify_freeze(output)
                self.assertTrue((output / "blind_consensus_spectrum.png").is_file())
                return np.array([4., 8., 12., 20.])

            with patch.object(evaluate, "load_reference", side_effect=reference) as loader:
                evaluate.evaluate_frozen(output)
                loader.assert_called_once()
            for name, content in before.items():
                self.assertEqual((output / name).read_bytes(), content)
            # Both mutation detection and protection against reusing output dirs.
            with self.assertRaises(ValueError):
                generate.generate(output, (5, 7, 11, 17), rules)
            (output / "blind_consensus_spectrum.csv").write_text("modified", encoding="utf-8")
            with patch.object(evaluate, "load_reference") as loader:
                with self.assertRaises(ValueError):
                    evaluate.evaluate_frozen(output)
                loader.assert_not_called()

    def test_cli_imports_evaluation_after_generate_returns(self):
        events = []
        with patch.object(run, "generate", side_effect=lambda *a, **k: events.append("freeze")), patch.object(evaluate, "evaluate_frozen", side_effect=lambda *a, **k: events.append("evaluate")):
            run.main(["--mode", "smoke"])
        self.assertEqual(events, ["freeze", "evaluate"])

    def test_evaluate_known_frozen_prediction_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            spectrum = model.consensus_spectrum([metric(4.), metric(8., 2)], [metric(4.2), metric(8.2, 2)], model.Rules())
            artifacts.write_csv(output / "blind_consensus_spectrum.csv", spectrum, artifacts.CONSENSUS_FIELDS)
            artifacts.write_csv(output / "blind_control_consensus.csv", [], ["dataset", "replicate"]+artifacts.CONSENSUS_FIELDS)
            artifacts.write_csv(output / "blind_sensitivity_spectra.csv", [], ["variant", "dataset", "replicate"]+artifacts.CONSENSUS_FIELDS)
            artifacts.write_csv(output / "blind_tracks.csv", [], artifacts.TRACK_FIELDS)
            (output / "blind_consensus_spectrum.png").write_bytes(b"fixture image, not rendered")
            artifacts.write_json(output / "blind_parameters.json", dict(tau_max=16., groups=[dict(dataset="real", replicate=0)], sensitivity={"baseline": {}}))
            artifacts.freeze(output)
            before = (output / "blind_consensus_spectrum.csv").read_bytes()
            with patch.object(evaluate, "load_reference", return_value=np.array([4., 8., 12., 20.])):
                summary = evaluate.evaluate_frozen(output)
            rows = artifacts.read_csv(output / "evaluation_zeta_comparison.csv")
            self.assertAlmostEqual(float(rows[0]["signed_error"]), 0.1)
            self.assertAlmostEqual(float(rows[1]["signed_error"]), 0.1)
            self.assertEqual(summary[0]["missing_modes"], 1)
            self.assertEqual((output / "blind_consensus_spectrum.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
