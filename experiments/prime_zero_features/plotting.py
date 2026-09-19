"""Static, shared-scale figures; finite products are never labeled zeta."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"real": "#176b87", "phase_scrambled": "#b35d18", "shuffled_rates": "#8159a4",
          "raised_cosine": "#4b8a54"}
CAUTION = "Finite products on Re(z)=1/2; no naive Euler-product convergence or RH/physics claim."


def save_plots(output, tau, cutoffs, zeros, profiles, comparisons, summaries, buildup, trajectory_zeros=6):
    names = []
    gammas = zeros.imag
    visible = gammas[(gammas > tau[0]) & (gammas < tau[-1])]
    picked = np.unique(np.linspace(0, len(cutoffs)-1, min(6, len(cutoffs)), dtype=int))

    def finish(fig, name, title):
        fig.suptitle(title, fontsize=13)
        fig.text(0.5, 0.012, CAUTION, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        names.append(name)

    def markers(ax):
        for gamma in visible:
            ax.axvline(gamma, color="black", alpha=0.3, lw=0.7)
        ax.set_xlim(tau[0], tau[-1])
        ax.grid(alpha=0.16)

    fig, axes = plt.subplots(len(picked), 1, figsize=(12, 2.1*len(picked)), sharex=True, sharey=True, squeeze=False)
    for ax, index in zip(axes[:, 0], picked):
        cutoff = cutoffs[index]
        ax.plot(tau, profiles[("real", 0, cutoff)]["L"], lw=0.8, color=COLORS["real"])
        markers(ax)
        ax.set_ylabel(f"P={cutoff}\nL")
    axes[-1, 0].set_xlabel("tau (vertical markers: actual zeta-zero ordinates)")
    finish(fig, "profiles.png", "Log magnitude squared — identical x and y scales; no clipping")

    rows = [r for r in comparisons if r["dataset"] == "real"]
    inside_indices = [i+1 for i, g in enumerate(gammas) if tau[0] < g < tau[-1]]
    selected = inside_indices[:trajectory_zeros]
    for offset_mode in (False, True):
        displayed = inside_indices if offset_mode else selected
        n = max(1, len(displayed))
        fig, axes = plt.subplots(int(np.ceil(n/2)), 2, figsize=(12, 2.7*int(np.ceil(n/2))), squeeze=False)
        for ax, zero_index in zip(axes.flat, displayed):
            entries = [r for r in rows if r["zero_index"] == zero_index]
            gamma = gammas[zero_index-1]
            for kind, color in (("peak", "#b35d18"), ("min", "#176b87")):
                key = f"{kind}_offset" if offset_mode else f"nearest_{kind}_tau"
                y = [abs(r[key]) if offset_mode and r[key] != "" else r[key] if r[key] != "" else np.nan for r in entries]
                ax.plot(cutoffs, y, "o-", color=color, ms=3, lw=1, label=kind)
                # A cross marks a change in heuristic track ID, not a proven bifurcation.
                if not offset_mode:
                    switches = [j for j in range(1, len(entries)) if entries[j][f"{kind}_track_switched"]]
                    ax.scatter([cutoffs[j] for j in switches], [y[j] for j in switches], marker="x", color="black", s=35, zorder=5)
            ax.axhline(0 if offset_mode else gamma, color="black", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"gamma_{zero_index} = {gamma:.6f}", fontsize=10)
            ax.set_xlabel("prime cutoff P")
            ax.set_ylabel("absolute offset (linear)" if offset_mode else "nearest feature tau")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(displayed):]:
            ax.axis("off")
        finish(fig, "offsets.png" if offset_mode else "trajectories.png",
               "Nearest-feature absolute offsets — every in-window zero" if offset_mode else
               "Nearest features, not guaranteed branches; crosses = heuristic track switches")

    matrix = np.array([profiles[("real", 0, p)]["L"] for p in cutoffs])
    for centered in (False, True):
        data = matrix-matrix.mean(axis=1, keepdims=True) if centered else matrix
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest",
                       extent=(tau[0], tau[-1], -0.5, len(cutoffs)-0.5), cmap="viridis")
        markers(ax)
        ax.set_yticks(range(len(cutoffs)), [str(p) for p in cutoffs])
        ax.set(xlabel="tau", ylabel="cutoff P (equally spaced cutoff index)")
        fig.colorbar(im, ax=ax, label="L - row mean" if centered else "L (raw log magnitude squared)")
        finish(fig, "heatmap_centered.png" if centered else "heatmap_raw.png",
               "Each row has its mean subtracted; no rescaling" if centered else "Raw log magnitude — shared color scale")

    fig, axes = plt.subplots(len(picked), 2, figsize=(13, 2.2*len(picked)), sharex=True, sharey="col", squeeze=False)
    for row, index in enumerate(picked):
        p = cutoffs[index]
        for col, (key, label) in enumerate((("phase", "unwrapped Phi"), ("phase_slope", "dPhi/dtau"))):
            axes[row, col].plot(tau, profiles[("real", 0, p)][key], lw=0.8)
            markers(axes[row, col])
            axes[row, col].set_ylabel(f"P={p}\n{label}")
    for ax in axes[-1]:
        ax.set_xlabel("tau")
    finish(fig, "phase.png", "Continuous phase lift and analytic derivative — common scale within each column")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, kind in enumerate(("peak", "min")):
        for row, statistic in enumerate(("mean_distance", "median_distance")):
            ax = axes[row, col]
            for dataset in COLORS:
                matching = [r for r in summaries if r["dataset"] == dataset and r["kind"] == kind and r["cohort"] == "common_bracketed"]
                if not matching:
                    continue
                groups = [[r[statistic] for r in matching if r["prime_cutoff"] == p and np.isfinite(r[statistic])] for p in cutoffs]
                mid = [np.median(g) if g else np.nan for g in groups]
                low = [min(g) if g else np.nan for g in groups]
                high = [max(g) if g else np.nan for g in groups]
                ax.plot(cutoffs, mid, "o-", ms=3, label=dataset, color=COLORS[dataset])
                ax.fill_between(cutoffs, low, high, alpha=0.15, color=COLORS[dataset])
            ax.set_xscale("log")
            ax.set_title(f"{kind}: {statistic.replace('_', ' ')}")
            ax.set(xlabel="prime cutoff P", ylabel="distance in tau")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    finish(fig, "controls.png", "Common bracketed zero cohort per cutoff; control bands = replicate range (not CI)")

    p = cutoffs[-1]
    datasets = [kind for kind in COLORS if (kind, 0, p) in profiles]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 2.5*len(datasets)), sharex=True, sharey=True, squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        ax.plot(tau, profiles[(dataset, 0, p)]["L"], color=COLORS[dataset], lw=0.8)
        markers(ax)
        ax.set_ylabel(f"{dataset}\nL", fontsize=9)
    axes[-1, 0].set_xlabel("tau")
    finish(fig, "control_profiles.png", f"P={p}; preselected replicate 0; identical x and y scales")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, kind in zip(axes, ("peak", "min")):
        for dataset in datasets:
            # Use the same gamma cohort as the aggregate plot, including all control replicates.
            ids = {r["zero_index"] for r in comparisons if r["dataset"] == "real" and r["prime_cutoff"] == p and r[f"{kind}_common_cohort"]}
            vals = sorted(r[f"{kind}_offset"] for r in comparisons if r["dataset"] == dataset and r["prime_cutoff"] == p
                          and r["zero_index"] in ids and r[f"{kind}_offset"] != "")
            if vals:
                ax.step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", color=COLORS[dataset], label=dataset)
        ax.set(title=kind, xlabel="signed offset", ylabel="empirical cumulative fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    finish(fig, "control_offsets.png", f"P={p}; pooled descriptive signed offsets, common cohort; no significance test")

    fig, axes = plt.subplots(len(buildup), 1, figsize=(11, 4.5*len(buildup)), squeeze=False)
    for ax, data in zip(axes[:, 0], buildup):
        im = ax.imshow(data["L"], origin="lower", aspect="auto", interpolation="nearest", cmap="viridis",
                       extent=(data["tau"][0], data["tau"][-1], -0.5, len(data["primes"])-0.5))
        ax.axvline(data["gamma"], color="white", ls="--", lw=1)
        ticks = np.unique(np.linspace(0, len(data["primes"])-1, min(8, len(data["primes"])), dtype=int))
        ax.set_yticks(ticks, data["primes"][ticks])
        ax.set(xlabel="tau", ylabel="last included prime (prime index spacing)", title=f"gamma_{data['zero_index']} neighborhood; every prime is included")
        fig.colorbar(im, ax=ax, label="L (raw, shared color scale within panel)")
    finish(fig, "buildup.png", "Prime-by-prime buildup — fixed zero neighborhoods, no alignment fitting")
    return names
