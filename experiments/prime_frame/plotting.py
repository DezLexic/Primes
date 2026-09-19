"""Saved matplotlib diagnostics for the finite prime-history experiment."""

import matplotlib.pyplot as plt
import numpy as np


def save_plots(rows, tau, log_profiles, representatives, output):
    frames = np.array([row["frame_prime"] for row in rows])
    selected = [(p, int(np.searchsorted(frames, p))) for p in representatives]
    paths = []

    def save(fig, name):
        path = output / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path.name)

    for logarithmic in (False, True):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for p, index in selected:
            values = log_profiles[index] / np.log(10) if logarithmic else np.exp(log_profiles[index])
            ax.plot(tau, values, label=f"p = {p}", lw=1.2)
        ax.set(xlabel=r"Common coordinate $\tau$",
               ylabel=r"$\log_{10} D_p(\tau)$" if logarithmic else r"$D_p(\tau)$",
               title="Normalized finite-history profiles")
        ax.legend(ncols=3)
        ax.grid(alpha=0.2)
        save(fig, "profiles_log" if logarithmic else "profiles")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for key, label in (("log_W0", r"$W_p(0)$"), ("log_D0", r"$D_p(0)$")):
        ax.plot(frames, [row[key] / np.log(10) for row in rows], label=label)
    ax.set(xlabel="Frame prime p", ylabel="log10 central power", title="Central suppression")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "central_suppression")

    nonempty = rows[1:]  # p=2 is the empty product; chi=0 and widths=infinity.
    p = frames[1:]
    for key, label, title in (
        ("chi", r"$\chi_p$", "Local log-power curvature"),
        ("tau_quad", r"$\tau_{quad}$", "Local width in common coordinate"),
        ("t_quad", r"$t_{quad}$", "Local width in prime-frame coordinate"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.loglog(p, [row[key] for row in nonempty])
        ax.set(xlabel="Frame prime p", ylabel=label, title=title)
        ax.grid(alpha=0.2)
        save(fig, key)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, key, reference, label in (
        (axes[0], "chi_pnt_ratio", 1, r"$\chi_p / (2\sqrt{p}\ln p)$"),
        (axes[1], "scaled_width", 1 / np.sqrt(2), r"$\tau_{quad}p^{1/4}\sqrt{\ln p}$"),
    ):
        ax.semilogx(p, [row[key] for row in nonempty])
        ax.axhline(reference, ls="--", color="gray", label="PNT leading-term reference")
        ax.set(xlabel="Frame prime p", ylabel=label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Asymptotic diagnostics (finite-range evidence only)")
    save(fig, "asymptotic_diagnostics")

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    heat = ax.pcolormesh(tau, np.arange(len(frames)), log_profiles / np.log(10),
                         shading="nearest", cmap="viridis", rasterized=True)
    ticks = np.unique(np.linspace(0, len(frames) - 1, min(10, len(frames))).astype(int))
    ax.set_yticks(ticks, [str(frames[i]) for i in ticks])
    ax.set(xlabel=r"Common coordinate $\tau$", ylabel="Successive prime frame p (equal row spacing)",
           title="Normalized history across successive frames")
    fig.colorbar(heat, ax=ax, label=r"$\log_{10} D_p(\tau)$")
    save(fig, "heatmap")
    return paths
