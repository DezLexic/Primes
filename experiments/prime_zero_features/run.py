"""Manual finite-product sweeps. Defaults are not executed during development."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import matplotlib
import mpmath
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.signal import find_peaks

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.prime_factor_phasors.model import factor_power
from experiments.prime_zero_features.model import (
    DEFAULT_CUTOFFS, assign_tracks, compare_zeros, detect_extrema, evaluate,
    first_zeta_zeros, make_signal, primes_through, resolution_comparison, safe_power,
)
from experiments.prime_zero_features.plotting import save_plots

GUARDRAILS = """This measures finite products, not a convergent critical-line Euler product.
The ordinary product does NOT naively converge to 1/zeta on Re(z)=1/2.
Zeta zeros are collective global features; individual primes do not generate individual zeros.
Elementary factors have interference-like phasor algebra; their product is multiplicative,
not a standard many-slit amplitude sum. No quantum mechanics, mass gap, or RH claim follows.
Resemblance must be measured, not assumed. A feature near gamma_1 is interesting only if
stable across cutoffs AND better aligned than reasonable controls.
Finite sweeps cannot establish asymptotic convergence, even if offsets decrease.
Nearest extrema may switch identity. Track IDs use mutual-nearest matching with a fixed
maximum jump, independent of gamma; they are heuristic, not certified continuation.
Shrinking nearest distances can result from increasing feature density. Counts are saved.
Missing features are blank, never zero distances. Endpoints are not extrema.
Out-of-window zeros have exact signal/phase evaluations but no feature offsets.
Unbracketed offsets may be boundary censored. Common-bracketed summaries use the same
zero cohort across ALL datasets/replicates per cutoff and feature kind. Cohorts may
change with cutoff; inspect IDs/counts before treating averages as trajectories.
Control bands show replicate ranges, not confidence intervals. No p-values are fitted.
Phase-scrambled primes keep fixed seeded phases across cutoffs. Shuffled rates permute
the exact frequency set separately at each cutoff and are not nested trajectories.
Prominence and half-prominence widths are grid estimates; locations/heights are refined.
No zero ordinates enter the detector or the refinement. No gamma points are inserted
into the uniform detection grid. Thresholds are shared across every dataset.
"""


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_events(tau, profile, signal):
    """Zero-blind stationary-phase roots and sampled local rotation maxima."""
    derivative = profile["phase_slope"]
    roots = list(tau[1:-1][derivative[1:-1] == 0])
    for i in np.flatnonzero(derivative[:-1]*derivative[1:] < 0):
        roots.append(brentq(lambda t: float(evaluate(signal, t)["phase_slope"]), tau[i], tau[i+1], xtol=1e-11))
    rows = [dict(kind="stationary_phase", tau=float(t), phase_slope=0.0) for t in sorted(set(roots))]
    peaks, _ = find_peaks(abs(derivative))
    rows += [dict(kind="rotation_max_grid", tau=float(tau[i]), phase_slope=float(derivative[i])) for i in peaks]
    return rows


def summarize(comparisons, extrema, cutoffs):
    rows = []
    datasets = sorted({(r["dataset"], r["replicate"]) for r in comparisons})
    for p in cutoffs:
        at_p = [r for r in comparisons if r["prime_cutoff"] == p]
        for kind in ("peak", "min"):
            common_ids = set.intersection(*[
                {r["zero_index"] for r in at_p if (r["dataset"], r["replicate"]) == group and r[f"{kind}_bracketed"]}
                for group in datasets])
            for r in at_p:
                r[f"{kind}_common_cohort"] = r["zero_index"] in common_ids
            for dataset, replicate in datasets:
                subset = [r for r in at_p if (r["dataset"], r["replicate"]) == (dataset, replicate)]
                count = sum(r["prime_cutoff"] == p and r["dataset"] == dataset and r["replicate"] == replicate and r["kind"] == kind for r in extrema)
                for cohort in ("all_in_window", "common_bracketed"):
                    selected = [r for r in subset if r["in_window"] and (cohort == "all_in_window" or r["zero_index"] in common_ids)]
                    distances = [abs(r[f"{kind}_offset"]) for r in selected if r[f"{kind}_offset"] != ""]
                    rows.append(dict(dataset=dataset, replicate=replicate, prime_cutoff=p, kind=kind,
                                     cohort=cohort, zero_indices=";".join(str(r["zero_index"]) for r in selected),
                                     eligible_zeros=len(selected), matched_zeros=len(distances),
                                     feature_count=count, mean_distance=float(np.mean(distances)) if distances else np.nan,
                                     median_distance=float(np.median(distances)) if distances else np.nan))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--tau-max", type=float, default=60)
    parser.add_argument("--points", type=int, default=12001, help="uniform samples including endpoints")
    parser.add_argument("--zeros", type=int, default=20)
    parser.add_argument("--dps", type=int, default=30)
    parser.add_argument("--prominence", type=float, default=0.1, help="minimum prominence in L units; fixed for all datasets")
    parser.add_argument("--min-width", type=float, default=0, help="minimum half-prominence width in tau units")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--control-replicates", type=int, default=5)
    parser.add_argument("--resolution-factor", type=int, choices=(2, 4), default=2)
    parser.add_argument("--resolution-tolerance", type=float, default=1e-5, help="absolute refined-position tolerance in tau")
    parser.add_argument("--track-max-jump", type=float, default=0.5)
    parser.add_argument("--trajectory-zeros", type=int, default=6)
    parser.add_argument("--buildup-cutoff", type=int, default=997)
    parser.add_argument("--buildup-points", type=int, default=801)
    parser.add_argument("--buildup-half-window", type=float, default=2)
    parser.add_argument("--buildup-zeros", type=int, nargs="+", default=[1])
    parser.add_argument("--taper", action="store_true", help="also evaluate weights (1+cos(pi*q/P))/2 for q<=P")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "normal")
    args = parser.parse_args()
    if args.cutoffs != sorted(set(args.cutoffs)) or min(args.cutoffs) < 2:
        parser.error("--cutoffs must be strictly increasing integers >=2")
    if args.points < 101 or args.zeros < 1 or args.dps < 15 or args.control_replicates < 1 or args.seed < 0:
        parser.error("points>=101, zeros>=1, dps>=15, control-replicates>=1, seed>=0 required")
    if args.buildup_cutoff < 2 or args.buildup_points < 101 or args.trajectory_zeros < 1:
        parser.error("buildup-cutoff>=2, buildup-points>=101 and trajectory-zeros>=1 required")
    if len(set(args.buildup_zeros)) != len(args.buildup_zeros) or not all(1 <= n <= args.zeros for n in args.buildup_zeros):
        parser.error("--buildup-zeros must be unique indices among the generated zeros")
    for name in ("tau_max", "resolution_tolerance", "track_max_jump", "buildup_half_window"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("prominence", "min_width"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tau = np.linspace(0, args.tau_max, args.points)
    zeros = first_zeta_zeros(args.zeros, args.dps)
    write_csv(output / "zeta_zeros.csv", [dict(zero_index=i, beta=z.real, gamma=z.imag,
              in_window=bool(0 < z.imag < args.tau_max)) for i, z in enumerate(zeros, 1)])
    datasets = [("real", 0)] + [(kind, rep) for kind in ("phase_scrambled", "shuffled_rates") for rep in range(args.control_replicates)]
    if args.taper:
        datasets += [("raised_cosine", 0)]
    profiles, extrema, comparisons, checks, events, phase_rows = {}, [], [], [], [], []
    check_cutoffs = {args.cutoffs[i] for i in np.unique(np.linspace(0, len(args.cutoffs)-1, min(3, len(args.cutoffs)), dtype=int))}
    for dataset, replicate in datasets:
        previous, next_track, previous_comparisons = [], 1, None
        for p in args.cutoffs:
            print(f"{dataset} replicate={replicate} P={p} ({args.points} samples)", flush=True)
            signal = make_signal(p, dataset, args.seed, replicate)
            profile = evaluate(signal, tau)
            profiles[(dataset, replicate, p)] = profile
            found = detect_extrema(tau, profile["L"], signal, args.prominence, args.min_width)
            # A rate permutation is rebuilt each cutoff, so never link those features.
            next_track = assign_tracks(previous if dataset != "shuffled_rates" else [], found, next_track, args.track_max_jump)
            context = dict(dataset=dataset, replicate=replicate, prime_cutoff=p)
            extrema.extend([{**context, **r} for r in found])
            compared = compare_zeros(zeros, signal, found, tau[0], tau[-1])
            phase_found = phase_events(tau, profile, signal)
            events.extend([{**context, **r} for r in phase_found])
            for i, r in enumerate(compared):
                for kind in ("peak", "min"):
                    key = f"nearest_{kind}_track_id"
                    r[f"{kind}_track_switched"] = bool(previous_comparisons is not None and r[key] != ""
                        and previous_comparisons[i][key] != "" and r[key] != previous_comparisons[i][key])
                phase_row = {**context, **{key: r[key] for key in ("zero_index", "gamma", "in_window", "phase_at_gamma", "phase_slope_at_gamma")}}
                phase_row["abs_rotation_window_percentile"] = float(np.mean(abs(profile["phase_slope"]) <= abs(r["phase_slope_at_gamma"]))) if r["in_window"] else ""
                for kind in ("stationary_phase", "rotation_max_grid"):
                    candidates = [e for e in phase_found if e["kind"] == kind] if r["in_window"] else []
                    nearest = min(candidates, key=lambda e: abs(e["tau"]-r["gamma"])) if candidates else None
                    phase_row[f"nearest_{kind}_offset"] = nearest["tau"]-r["gamma"] if nearest else ""
                phase_rows.append(phase_row)
            comparisons.extend([{**context, **r} for r in compared])
            previous, previous_comparisons = found, compared
            if p in check_cutoffs and replicate == 0:
                fine_tau = np.linspace(0, args.tau_max, (args.points-1)*args.resolution_factor+1)
                fine = detect_extrema(fine_tau, evaluate(signal, fine_tau)["L"], signal, args.prominence, args.min_width)
                checks.extend([{**context, **r} for r in resolution_comparison(found, fine, args.resolution_tolerance)])

    summaries = summarize(comparisons, extrema, args.cutoffs)
    # Write headers even when no features survive the configured threshold.
    extrema_fields = ["dataset", "replicate", "prime_cutoff", "kind", "feature_id", "tau", "value", "grid_tau",
                      "grid_value", "prominence", "width_tau", "bracket_left", "bracket_right", "refinement",
                      "refined_slope", "refined_curvature", "refinement_ok", "track_id"]
    write_csv(output / "extrema.csv", extrema, extrema_fields)
    write_csv(output / "nearest_features.csv", [r for r in comparisons if r["dataset"] == "real"])
    write_csv(output / "control_results.csv", [r for r in comparisons if r["dataset"] != "real"])
    write_csv(output / "control_summary.csv", summaries)
    write_csv(output / "phase_summary.csv", phase_rows)
    write_csv(output / "phase_events.csv", events, ["dataset", "replicate", "prime_cutoff", "kind", "tau", "phase_slope"])
    write_csv(output / "resolution_checks.csv", checks)

    arrays = {"tau": tau, "cutoffs": np.array(args.cutoffs), "gammas": zeros.imag}
    for (kind, rep, p), profile in profiles.items():
        arrays.update({f"{kind}_r{rep}_P{p}_{key}": value for key, value in profile.items()})
        arrays[f"{kind}_r{rep}_P{p}_power"] = safe_power(profile["L"])
    np.savez_compressed(output / "profiles.npz", **arrays)
    buildup = []
    build_arrays = {}
    build_primes = primes_through(min(args.buildup_cutoff, args.cutoffs[-1]))
    for index in args.buildup_zeros:
        gamma = zeros[index-1].imag
        local_tau = np.linspace(max(0, gamma-args.buildup_half_window), gamma+args.buildup_half_window, args.buildup_points)
        values = np.cumsum([np.log(factor_power(int(q), 0.5, local_tau)) for q in build_primes], axis=0)
        buildup.append(dict(zero_index=index, gamma=gamma, tau=local_tau, primes=build_primes, L=values))
        build_arrays.update({f"zero_{index}_tau": local_tau, f"zero_{index}_L": values, f"zero_{index}_gamma": gamma})
    np.savez_compressed(output / "buildup.npz", primes=build_primes, **build_arrays)
    names = save_plots(output, tau, args.cutoffs, zeros, profiles, comparisons, summaries, buildup, args.trajectory_zeros)
    settings = {**vars(args), "output": str(output), "created_utc": datetime.now(timezone.utc).isoformat(),
                "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                "mpmath": mpmath.__version__, "python": platform.python_version(),
                "zero_source": "mpmath.zetazero; generated at dps, exported as float64",
                "phase_branch": "sum of continuous factor arguments; no wrapping or recentering",
                "resolution_cutoffs": sorted(check_cutoffs), "resolution_replicates": [0],
                "actual_buildup_cutoff": int(build_primes[-1]), "plots": names,
                "profile_key_format": "{dataset}_r{replicate}_P{cutoff}_{L|phase|slope|curvature|phase_slope|power}",
                "taper_definition": "w(q/P)=(1+cos(pi*q/P))/2 for q<=P, zero beyond P" if args.taper else None}
    (output / "parameters.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    failed = [r for r in checks if not r["passed"]]
    failures = sum(not r["refinement_ok"] for r in extrema)
    notes = [GUARDRAILS, f"\nComputed {args.zeros} zeros; {sum(0 < z.imag < args.tau_max for z in zeros)} inside the main grid.",
             f"Resolution checks: {len(checks)-len(failed)}/{len(checks)} passed; {failures} refinement warnings.",
             "Resolution check includes real/taper and replicate 0 of each control at up to three cutoffs.",
             "A failed check means detection counts or refined locations changed; increase resolution before interpretation.",
             "Small/smoke runs verify plumbing only. Even the default run does not establish convergence.",
             "\nFirst-zero signed offsets (blank means unavailable):"]
    for r in comparisons:
        if r["dataset"] == "real" and r["zero_index"] == 1:
            notes.append(f"P={r['prime_cutoff']}: peak={r['peak_offset']}, min={r['min_offset']}")
    notes += ["\nPhase: inspect phase_summary.csv for stationary-phase offsets and rotation percentiles.",
              "A percentile describes this finite window, not a probability or evidence of a universal phase crossing.",
              "Raw log heatmap and mean-centered heatmap use shared color scales; no row variance rescaling.",
              "Full profiles use common y limits, including the deep tau=0 minimum; this may compress visible peaks.",
              "Optional taper results are sensitivity diagnostics, not a regularization or a zeta approximation.",
              f"Output: {output}"]
    (output / "summary.txt").write_text("\n".join(notes)+"\n", encoding="utf-8")
    gallery = ["# Finite prime-product feature results", "", "[Settings](parameters.json) | [Notes and guardrails](summary.txt)", "",
               "Finite exploratory data; no convergence conclusion. Check resolution_checks.csv before interpretation.", "",
               "[Nearest real features](nearest_features.csv) | [Control summaries](control_summary.csv) | [All extrema](extrema.csv)", ""]
    gallery += [f"![{name.removesuffix('.png').replace('_', ' ')}]({name})\n" for name in names]
    (output / "README.md").write_text("\n".join(gallery), encoding="utf-8")
    print(f"Saved {len(names)} PNGs, 8 CSVs, 2 NPZs, settings, notes and gallery: {output}", flush=True)
    if failed or failures:
        print(f"WARNING: {len(failed)} resolution checks failed; {failures} refinement warnings. Read resolution_checks.csv and extrema.csv.", flush=True)


if __name__ == "__main__":
    main()
