"""After-the-fact evaluation. Reads frozen predictions; never selects modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.blind_prime_spectrum.artifacts import GUARDRAILS, read_csv, verify_freeze, write_csv, write_json

COMPARISON_FIELDS = ["predicted_index", "predicted_tau", "zero_index", "gamma", "signed_error", "absolute_error", "relative_error", "nearest_zero_index", "nearest_zero_gamma", "nearest_zero_distance"]


def load_reference(tau_max, minimum_count, dps):
    # This is the ONLY zero-loader import in the new experiment. It is called
    # only after evaluate_frozen has verified all frozen artifact hashes.
    from experiments.prime_frame_geometry.model import first_zeta_zeros
    count = max(8, minimum_count)
    while True:
        values = first_zeta_zeros(count, dps).imag
        if values[-1] > tau_max:
            return values
        count *= 2


def compare_spectrum(spectrum, gammas):
    """Ordered comparison begins at the first positive zero; nearest is separate."""
    gammas = np.asarray(gammas, dtype=float)
    if gammas.ndim != 1 or not len(gammas) or not np.all(np.isfinite(gammas)) or np.any(gammas <= 0) or np.any(np.diff(gammas) <= 0):
        raise ValueError("reference ordinates must be finite, positive, strictly increasing")
    rows = []
    for index, row in enumerate(sorted(spectrum, key=lambda r: float(r["consensus_tau"]))):
        tau = float(row["consensus_tau"])
        if not np.isfinite(tau):
            raise ValueError("prediction must be finite")
        nearest = int(np.argmin(abs(gammas-tau)))
        overlap = index < len(gammas)
        error = tau-gammas[index] if overlap else None
        rows.append(dict(predicted_index=index+1, predicted_tau=tau, zero_index=index+1 if overlap else "",
                         gamma=float(gammas[index]) if overlap else "", signed_error=float(error) if overlap else "",
                         absolute_error=float(abs(error)) if overlap else "",
                         relative_error=float(abs(error)/gammas[index]) if overlap else "",
                         nearest_zero_index=nearest+1, nearest_zero_gamma=float(gammas[nearest]),
                         nearest_zero_distance=float(abs(tau-gammas[nearest]))))
    return rows


def failure_diagnostics(comparison, gammas, tau_max, tolerance):
    """Maximum-cardinality, then minimum-distance one-to-one in-window match.

    Used only to count missing/extra/duplicate modes; never reorders prediction.
    """
    targets = [(i+1, float(g)) for i, g in enumerate(gammas) if 0 < g < tau_max]
    predicted = np.array([r["predicted_tau"] for r in comparison])
    matched_p, matched_z, accepted = set(), set(), []
    if len(predicted) and targets:
        distance = abs(predicted[:, None]-np.array([g for _, g in targets])[None, :])
        # Each invalid edge costs more than all possible valid edges together.
        penalty = (min(distance.shape)+1)*(tolerance+1)
        ii, jj = linear_sum_assignment(np.where(distance <= tolerance, distance, penalty))
        for i, j in zip(ii, jj):
            if distance[i, j] <= tolerance:
                matched_p.add(int(i))
                matched_z.add(int(j))
                accepted.append(dict(kind="matched", predicted_index=int(i)+1, predicted_tau=float(predicted[i]),
                                     zero_index=targets[j][0], gamma=targets[j][1], distance=float(distance[i, j]), reason="one_to_one_within_tolerance"))
    rows = accepted
    for i, r in enumerate(comparison):
        if i not in matched_p:
            nearby = any(abs(r["predicted_tau"]-g) <= tolerance for _, g in targets)
            rows.append(dict(kind="extra", predicted_index=i+1, predicted_tau=r["predicted_tau"], zero_index=r["nearest_zero_index"],
                             gamma=r["nearest_zero_gamma"], distance=r["nearest_zero_distance"],
                             reason="duplicate_near_reference" if nearby else "no_in_window_reference_within_tolerance"))
    for j, (index, g) in enumerate(targets):
        if j not in matched_z:
            rows.append(dict(kind="missing", predicted_index="", predicted_tau="", zero_index=index, gamma=g,
                             distance=float(np.min(abs(predicted-g))) if len(predicted) else "", reason="no_one_to_one_prediction"))
    return rows


def save_evaluation_plots(output, comparison, gammas, tau_max):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = [r["predicted_tau"] for r in comparison]
    zeros = [g for g in gammas if 0 < g < tau_max]
    fig, (ax, error_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    ax.scatter(predictions, np.ones(len(predictions)), marker="|", s=180, label="Frozen blind consensus")
    ax.scatter(zeros, np.zeros(len(zeros)), marker="x", s=45, label="Reference zeros (loaded afterward)")
    ax.set(yticks=[], ylim=(-0.5, 1.5), xlim=(0, tau_max), title="After-the-fact comparison — no modes selected using zeros")
    ax.legend(loc="upper left", fontsize=8)
    if not predictions:
        ax.text(0.5, 0.65, "No blind consensus modes", ha="center", transform=ax.transAxes)
    ordered = [r for r in comparison if r["zero_index"] != ""]
    error_ax.scatter([r["predicted_tau"] for r in ordered], [r["signed_error"] for r in ordered], label="Ordered signed error")
    error_ax.axhline(0, color="gray", lw=0.7)
    error_ax.set(xlabel="Frozen predicted tau", ylabel="Predicted tau minus ordered reference")
    fig.tight_layout()
    fig.savefig(output / "evaluation_zeta_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter([r["predicted_index"] for r in ordered], [r["absolute_error"] for r in ordered])
    axes[1].scatter([r["predicted_index"] for r in comparison], [r["nearest_zero_distance"] for r in comparison])
    for ax, title in zip(axes, ("Ordered absolute error", "Nearest-zero distance (diagnostic only)")):
        ax.set(xlabel="Frozen predicted index", ylabel="Absolute distance in tau", title=title)
    fig.tight_layout()
    fig.savefig(output / "evaluation_errors.png", dpi=150)
    plt.close(fig)


def evaluate_frozen(output, *, tolerance=0.5, dps=30):
    output = Path(output)
    if not np.isfinite(tolerance) or tolerance <= 0 or dps < 15:
        raise ValueError("finite tolerance>0 and dps>=15 required")
    manifest = verify_freeze(output)  # MUST precede every reference-loader call.
    parameters = json.loads((output / "blind_parameters.json").read_text(encoding="utf-8"))
    real = read_csv(output / "blind_consensus_spectrum.csv")
    controls = read_csv(output / "blind_control_consensus.csv")
    sensitivity = read_csv(output / "blind_sensitivity_spectra.csv")
    tau_max = parameters["tau_max"]
    groups = [(g["dataset"], g["replicate"]) for g in parameters["groups"]]
    select = lambda rows, d, r: [row for row in rows if row["dataset"] == d and int(row["replicate"]) == r]
    spectra = {("baseline", d, r): real if d == "real" else select(controls, d, r) for d, r in groups}
    for variant in parameters["sensitivity"]:
        if variant != "baseline":
            for d, r in groups:
                spectra[(variant, d, r)] = [row for row in select(sensitivity, d, r) if row["variant"] == variant]
    gammas = load_reference(tau_max, max(map(len, spectra.values()), default=0), dps)
    comparisons, failures, summaries = [], [], []
    real_comparison = []
    for (variant, dataset, replicate), spectrum in spectra.items():
        compared = compare_spectrum(spectrum, gammas)
        diagnostics = failure_diagnostics(compared, gammas, tau_max, tolerance)
        context = dict(variant=variant, dataset=dataset, replicate=replicate)
        comparisons.extend({**context, **r} for r in compared)
        failures.extend({**context, **r} for r in diagnostics)
        if (variant, dataset) == ("baseline", "real"):
            real_comparison = compared
        matched = sum(r["kind"] == "matched" for r in diagnostics)
        expected = sum(0 < g < tau_max for g in gammas)
        nearest = [r["nearest_zero_distance"] for r in compared]
        ordered = [r["absolute_error"] for r in compared if r["zero_index"] != ""]
        summaries.append({**context, "predicted_modes": len(spectrum), "in_window_zeros": int(expected),
                          "one_to_one_matches": matched, "missing_modes": int(expected)-matched,
                          "extra_modes": len(spectrum)-matched, "precision": matched/len(spectrum) if spectrum else "",
                          "recall": matched/int(expected) if expected else "",
                          "mean_nearest_distance": float(np.mean(nearest)) if nearest else "",
                          "mean_ordered_absolute_error": float(np.mean(ordered)) if ordered else ""})
    write_csv(output / "evaluation_zeta_comparison.csv", real_comparison, COMPARISON_FIELDS)
    write_csv(output / "evaluation_all_comparisons.csv", comparisons, ["variant", "dataset", "replicate"]+COMPARISON_FIELDS)
    write_csv(output / "evaluation_failure_modes.csv", failures, ["variant", "dataset", "replicate", "kind", "predicted_index", "predicted_tau", "zero_index", "gamma", "distance", "reason"])
    write_csv(output / "evaluation_control_summary.csv", summaries, list(summaries[0]))
    write_csv(output / "evaluation_zeta_zeros.csv", [dict(zero_index=i+1, gamma=float(g), in_window=bool(0 < g < tau_max)) for i, g in enumerate(gammas)], ["zero_index", "gamma", "in_window"])
    write_json(output / "evaluation_parameters.json", dict(tolerance=tolerance, dps=dps,
               zero_source="existing first_zeta_zeros / mpmath.zetazero, exported as float64",
               consensus_sha256=manifest["files"]["blind_consensus_spectrum.csv"], frozen_utc=manifest["frozen_utc"],
               ordered_definition="jth prediction vs jth positive zero, starting at one; no nearest-based reordering"))
    save_evaluation_plots(output, real_comparison, gammas, tau_max)
    notes = [GUARDRAILS, "All comparisons use files verified against the pre-evaluation freeze manifest.",
             "The reference table covers the entire tau window and includes at least one zero above it.",
             "Missing/extra counts use one-to-one in-window matching; nearest distances can count the same zero repeatedly.",
             f"Evaluation-only matching tolerance: {tolerance}. It does not change the predictions.",
             "No sensitivity variant is selected using its evaluation score.",
             "Control means depend on mode count and density. These descriptive numbers are not significance tests.", ""]
    for r in summaries:
        if r["variant"] == "baseline":
            notes.append(f"{r['dataset']} r{r['replicate']}: predicted={r['predicted_modes']}, matched={r['one_to_one_matches']}, missing={r['missing_modes']}, extra={r['extra_modes']}, mean nearest distance={r['mean_nearest_distance']}")
    notes.append("\nInspect blind_tracking_diagnostics.csv for possible switching and blind_feature_counts.csv for density effects.")
    (output / "evaluation_summary.txt").write_text("\n".join(notes)+"\n", encoding="utf-8")
    (output / "evaluation_README.md").write_text("# After-the-fact evaluation\n\nRead [evaluation notes](evaluation_summary.txt) and [control comparison](evaluation_control_summary.csv).\n\n![Frozen spectrum and references](evaluation_zeta_comparison.png)\n\n![Errors](evaluation_errors.png)\n", encoding="utf-8")
    verify_freeze(output)  # Evaluation must not modify any blind artifact.
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory containing an already frozen blind run")
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--dps", type=int, default=30)
    args = parser.parse_args()
    evaluate_frozen(args.output, tolerance=args.tolerance, dps=args.dps)
    print(f"Evaluated frozen files in {args.output.resolve()}")


if __name__ == "__main__":
    main()
