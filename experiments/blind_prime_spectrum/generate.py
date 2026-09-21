"""Generate and freeze all predictions before importing evaluation code."""
from __future__ import annotations

from dataclasses import asdict
import platform
from pathlib import Path

import numpy as np
import scipy
import matplotlib

from .artifacts import (CONSENSUS_FIELDS, CONTEXT, GUARDRAILS, PEAK_FIELDS, TRACK_FIELDS,
                        freeze, sha256, write_csv, write_json)
from .blind_model import (Tracker, consensus_spectrum, cutoff_signal, detect_peaks,
                          internal_momentum, signal_bank, sensitivity_rules, track_metrics)
from .plotting import save_blind_plots


def generate(output, cutoffs, rules, *, tau_max=60., points=6001, prominence=0.1,
             seed=1729, control_replicates=3, max_jump=0.5, max_gap=1,
             ambiguity_margin=0.05, kappa=None, progress=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty: use a new directory to preserve frozen runs")
    if len(cutoffs) < 4 or tuple(cutoffs) != tuple(sorted(set(cutoffs))) or any(isinstance(p, bool) or not isinstance(p, int) or p < 2 for p in cutoffs):
        raise ValueError("need >=4 strictly increasing integer cutoffs >=2")
    if not np.isfinite(tau_max) or tau_max <= 0 or points < 101 or not isinstance(points, int):
        raise ValueError("positive finite tau_max and integer points>=101 required")
    if not np.isfinite(prominence) or prominence < 0 or seed < 0 or control_replicates < 2:
        raise ValueError("finite prominence>=0, seed>=0 and at least two control replicates required")
    if int(np.ceil(len(cutoffs)*rules.tail_fraction)) < rules.min_tail_count:
        raise ValueError("tail window too short for min_tail_count; increase cutoffs or tail fraction")
    Tracker(max_jump, max_gap, ambiguity_margin)
    if kappa is not None:
        internal_momentum([], kappa)
    output.mkdir(parents=True, exist_ok=True)
    groups = [("real", 0)] + [(d, r) for d in ("phase_scrambled", "shuffled_rates") for r in range(control_replicates)]
    variants = sensitivity_rules(rules)
    parameters = dict(cutoffs=list(cutoffs), tau_min=0., tau_max=tau_max, points=points, prominence=prominence,
                      seed=seed, control_replicates=control_replicates, max_jump=max_jump, max_gap=max_gap,
                      ambiguity_margin=ambiguity_margin, kappa=kappa, rules=asdict(rules),
                      sensitivity={name: asdict(r) for name, r in variants.items()},
                      groups=[dict(dataset=d, replicate=r) for d, r in groups],
                      tail_definition="final ceil(tail_fraction * number_of_cutoffs) scheduled cutoffs, common to all tracks",
                      shuffled_rates="fixed permutations within entering prime batches; exact frequency set at each scheduled cutoff",
                      taper="(1+cos(pi*q/P))/2 multiplying each log term; weights follow amplitude prime q",
                      resolution="double grid at first and final cutoff, every dataset and replicate; tolerance=1e-6",
                      versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                      source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py") if p.name != "test_model.py"})
    # Include the reused numerical implementation in the provenance.
    shared = Path(__file__).parents[1] / "prime_zero_features" / "prime_only.py"
    parameters["source_sha256"]["prime_zero_features/prime_only.py"] = sha256(shared)
    write_json(output / "blind_parameters.json", parameters)
    tau = np.linspace(0, tau_max, points)
    peaks, tracks, diagnostics, counts, checks, factors = [], [], [], [], [], []
    by_group, spectra, summaries, sensitivity, sensitivity_summary = {}, {}, [], [], []
    for dataset, replicate in groups:
        primes, bank = signal_bank(cutoffs, dataset, seed, replicate)
        for q, a, rate, phase in zip(primes, bank.amplitudes, bank.rates, bank.phases):
            factors.append(dict(dataset=dataset, replicate=replicate, prime=int(q), amplitude=float(a), rate=float(rate), phase=float(phase)))
        for mode in ("hard", "taper"):
            context = dict(cutoff_mode=mode, dataset=dataset, replicate=replicate)
            tracker, group_peaks = Tracker(max_jump, max_gap, ambiguity_margin), []
            for step, cutoff in enumerate(cutoffs):
                progress(f"BLIND {dataset} r{replicate} {mode} P={cutoff}")
                signal = cutoff_signal(primes, bank, cutoff, mode)
                found = detect_peaks(tau, signal, prominence)
                diagnostic = tracker.step(found, step)
                at = {**context, "prime_cutoff": cutoff}
                rows = [{**at, **r} for r in found]
                group_peaks.extend(rows)
                diagnostics.extend({**at, **r} for r in diagnostic)
                counts.append({**at, "feature_count": len(found), "refinement_failures": sum(not r["refinement_ok"] for r in found),
                               "density_per_tau": len(found)/tau_max, "prime_count": int(np.searchsorted(primes, cutoff, side="right"))})
                if step in (0, len(cutoffs)-1):
                    fine = detect_peaks(np.linspace(0, tau_max, 2*(points-1)+1), signal, prominence)
                    error = max((abs(a["tau"]-b["tau"]) for a, b in zip(found, fine)), default=0.) if len(found) == len(fine) else None
                    checks.append({**at, "coarse_count": len(found), "fine_count": len(fine), "max_position_change": error,
                                   "passed": error is not None and error <= 1e-6 and all(r["refinement_ok"] for r in found+fine)})
            metrics = [{**context, **r} for r in track_metrics(group_peaks, cutoffs, rules)]
            peaks.extend(group_peaks)
            tracks.extend(metrics)
            by_group[(dataset, replicate, mode)] = metrics
        h, t = [by_group[(dataset, replicate, mode)] for mode in ("hard", "taper")]
        spectrum = consensus_spectrum(h, t, rules)
        spectra[(dataset, replicate)] = spectrum
        for mode, metrics in (("hard", h), ("taper", t)):
            stable = [r for r in metrics if r["stable"]]
            tail = [r for r in metrics if r["tail_count"] >= rules.min_tail_count]
            diag = [r for r in diagnostics if (r["dataset"], r["replicate"], r["cutoff_mode"]) == (dataset, replicate, mode)]
            median = lambda rows, key: float(np.median([r[key] for r in rows])) if rows else ""
            summaries.append(dict(dataset=dataset, replicate=replicate, cutoff_mode=mode, total_tracks=len(metrics),
                                  tail_eligible_tracks=len(tail), stable_modes=len(stable), consensus_modes=len(spectrum),
                                  stable_fraction=len(stable)/len(metrics) if metrics else 0.,
                                  median_persistence=median(tail, "persistence"), median_tail_drift=median(tail, "tail_drift"),
                                  stable_median_persistence=median(stable, "persistence"), stable_median_tail_drift=median(stable, "tail_drift"),
                                  stable_median_prominence=median(stable, "median_prominence"),
                                  ambiguous_births=sum(r["event"] == "ambiguous_birth" for r in diag),
                                  reconnects=sum(r["event"] == "reconnect" for r in diag)))
        baseline_ids = {(r["hard_track_id"], r["taper_track_id"]) for r in spectrum}
        for name, variant in variants.items():
            predicted = consensus_spectrum(h, t, variant)
            sensitivity.extend(dict(variant=name, dataset=dataset, replicate=replicate, **r) for r in predicted)
            ids = {(r["hard_track_id"], r["taper_track_id"]) for r in predicted}
            overlap = ids & baseline_ids
            sensitivity_summary.append(dict(variant=name, dataset=dataset, replicate=replicate, consensus_modes=len(predicted),
                                             shared_with_baseline=len(overlap), added_vs_baseline=len(ids-baseline_ids),
                                             lost_vs_baseline=len(baseline_ids-ids),
                                             jaccard_vs_baseline=len(overlap)/len(ids | baseline_ids) if ids | baseline_ids else 1.))
    write_csv(output / "blind_peaks.csv", peaks, PEAK_FIELDS)
    write_csv(output / "blind_tracks.csv", tracks, TRACK_FIELDS)
    for mode in ("hard", "taper"):
        write_csv(output / f"blind_stable_modes_{mode}.csv", [r for r in tracks if r["cutoff_mode"] == mode and r["stable"]], TRACK_FIELDS)
    real = spectra[("real", 0)]
    write_csv(output / "blind_consensus_spectrum.csv", real, CONSENSUS_FIELDS)
    control = [dict(dataset=d, replicate=r, **row) for (d, r), rows in spectra.items() if d != "real" for row in rows]
    write_csv(output / "blind_control_consensus.csv", control, ["dataset", "replicate"]+CONSENSUS_FIELDS)
    write_csv(output / "blind_control_summary.csv", summaries, list(summaries[0]))
    write_csv(output / "blind_sensitivity_spectra.csv", sensitivity, ["variant", "dataset", "replicate"]+CONSENSUS_FIELDS)
    write_csv(output / "blind_sensitivity_summary.csv", sensitivity_summary, list(sensitivity_summary[0]))
    write_csv(output / "blind_tracking_diagnostics.csv", diagnostics, CONTEXT+["prime_cutoff", "event", "track_id", "feature_id", "distance", "missing_steps"])
    write_csv(output / "blind_feature_counts.csv", counts, list(counts[0]))
    write_csv(output / "blind_resolution_checks.csv", checks, list(checks[0]))
    write_csv(output / "blind_factor_assignments.csv", factors, list(factors[0]))
    if kappa is not None:
        write_csv(output / "blind_internal_momentum.csv", internal_momentum(real, kappa),
                  ["predicted_index", "consensus_tau", "kappa", "predicted_P_int"])
    names = save_blind_plots(output, peaks, tracks, spectra, groups, rules)
    notes = [GUARDRAILS, f"Real-prime consensus modes: {len(real)}.",
             f"Resolution failures: {sum(not r['passed'] for r in checks)} / {len(checks)}.",
             f"Refinement failures: {sum(not r['refinement_ok'] for r in peaks)}.",
             "Failed numerical checks require a new, finer blind run before interpretation; thresholds are not retuned here.",
             "Stability is finite-window behavior, not certified limiting eigenvalues.",
             "The batch-shuffled control is nested but constrained; small batches can leave rates unchanged.",
             "Inspect factor assignments and sensitivity results before interpreting differences from controls.", ""]
    for row in summaries:
        notes.append(f"{row['dataset']} r{row['replicate']} {row['cutoff_mode']}: stable={row['stable_modes']}, consensus={row['consensus_modes']}, ambiguous_births={row['ambiguous_births']}, reconnects={row['reconnects']}")
    (output / "blind_summary.txt").write_text("\n".join(notes)+"\n", encoding="utf-8")
    gallery = ["# Frozen blind prime-spectrum results", "", "Read [blind notes](blind_summary.txt), [parameters](blind_parameters.json), and [integrity manifest](blind_freeze.json).",
               "", "Smoke runs verify plumbing only. Evaluation files, if present, were made after the freeze.", ""]
    gallery += [f"![{name}]({name})\n" for name in names]
    (output / "README.md").write_text("\n".join(gallery), encoding="utf-8")
    manifest = freeze(output)
    progress(f"FROZEN {output / 'blind_consensus_spectrum.csv'} SHA-256={manifest['files']['blind_consensus_spectrum.csv']}")
    return manifest
