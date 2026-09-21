"""Blind plots only. This module cannot load a reference spectrum."""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_blind_plots(output, peaks, tracks, spectra, groups, rules):
    names = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        names.append(name)

    for mode in ("hard", "taper"):
        stable = {r["track_id"] for r in tracks if r["dataset"] == "real" and r["cutoff_mode"] == mode and r["stable"]}
        grouped = defaultdict(list)
        for p in peaks:
            if p["dataset"] == "real" and p["cutoff_mode"] == mode and p["track_id"] != "":
                grouped[p["track_id"]].append(p)
        fig, ax = plt.subplots(figsize=(10, 6))
        for tid, observations in grouped.items():
            selected = tid in stable
            ax.plot([p["prime_cutoff"] for p in observations], [p["tau"] for p in observations],
                    ".-", ms=2, lw=1.3 if selected else 0.5, alpha=0.9 if selected else 0.35,
                    color="tab:blue" if selected else "gray")
        ax.set(xscale="log", xlabel="Prime cutoff P (log scale)", ylabel="Peak location tau",
               title=f"zero-blind prime peak trajectories — {mode}\nBlue: stable by predeclared rules; gray: other tracks")
        save(fig, f"blind_peak_trajectories_{mode}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mode in zip(axes, ("hard", "taper")):
        for dataset, color in (("real", "tab:blue"), ("phase_scrambled", "tab:orange"), ("shuffled_rates", "tab:green")):
            rows = [r for r in tracks if r["cutoff_mode"] == mode and r["dataset"] == dataset and r["tail_count"] >= 2]
            ax.scatter([r["persistence"] for r in rows], [r["tail_drift"] for r in rows],
                       s=[8+6*min(r["median_prominence"], 10) for r in rows], alpha=0.5, label=dataset, color=color)
        ax.axvline(rules.persistence, ls=":", color="black")
        ax.axhline(rules.max_drift, ls=":", color="black")
        ax.set(xlabel="Persistence in global tail window", ylabel="Tail drift (tau)", title=f"Blind stability — {mode}")
        ax.legend(fontsize=8)
    fig.suptitle("Size: tail prominence; slope and observation-count gates also apply")
    save(fig, "blind_stability_metrics.png")

    fig, ax = plt.subplots(figsize=(10, 3))
    real = spectra[("real", 0)]
    ax.vlines([r["consensus_tau"] for r in real], 0, 1, color="tab:blue")
    if not real:
        ax.text(0.5, 0.5, "No consensus modes passed the blind rules", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Consensus tau", yticks=[], title="Blind consensus spectrum: what did the primes predict?")
    save(fig, "blind_consensus_spectrum.png")

    fig, ax = plt.subplots(figsize=(11, max(4, 0.65*len(groups))))
    for i, group in enumerate(groups):
        for mode, delta, color in (("hard", -0.18, "tab:blue"), ("taper", 0, "tab:orange")):
            values = [r["tail_tau_median"] for r in tracks if (r["dataset"], r["replicate"]) == group and r["cutoff_mode"] == mode and r["stable"]]
            ax.scatter(values, np.full(len(values), i+delta), marker="|", s=75, color=color, label=mode if i == 0 else None)
        values = [r["consensus_tau"] for r in spectra[group]]
        ax.scatter(values, np.full(len(values), i+0.18), marker="|", s=95, color="black", label="consensus" if i == 0 else None)
    ax.set(yticks=range(len(groups)), yticklabels=[f"{d} r{r}" for d, r in groups], xlabel="Blind spectral location tau",
           title="Blind control spectra — identical selection rules")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "blind_control_spectra.png")
    return names
