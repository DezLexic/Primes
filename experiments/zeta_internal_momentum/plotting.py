"""Static Matplotlib figures, matching the existing experiment galleries."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import directional_moments, internal_energy

COLORS = ("#1768ac", "#d66024", "#26834a", "#9146a3")


def save_plots(output, gamma, momenta, p_ext, energies, kg_errors, states,
               state_rows, dirac_rows, mapping, kappa, c):
    names = []
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(gamma)))

    def finish(fig, name, title, footer="Imposed toy spectrum; ordinary KG / Dirac dynamics. No physical mass-gap or RH claim."):
        fig.suptitle(title, fontsize=14)
        fig.text(0.5, 0.015, footer, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.045, 1, 0.94))
        for ax in fig.axes:
            if ax.name != "3d":
                ax.grid(alpha=0.18)
                ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        names.append(name)

    fig, (ax, gaps) = plt.subplots(1, 2, figsize=(12, 6))
    ax.hlines(0, 0.15, 0.75, color="gray", linestyle="--")
    ax.text(0.77, 0, "P₀ = 0 (reference)", va="center", fontsize=9)
    for n, (g, color) in enumerate(zip(gamma, colors), 1):
        ax.hlines(g, 0.15, 0.75, color=color, linewidth=2)
        ax.text(0.77, g, f"P{n} / κ = {g:.4f}", va="center", fontsize=9)
    ax.annotate("", xy=(0.08, gamma[0]), xytext=(0.08, 0),
                arrowprops=dict(arrowstyle="<->", color=COLORS[1], lw=2))
    ax.text(0.13, gamma[0]/2, f"Toy internal momentum gap\nΔP = κγ₁ = {momenta[0]:.6g}",
            color=COLORS[1], va="center", fontsize=10)
    ax.set(xlim=(0, 1.45), xticks=[], ylabel="Internal momentum P_int,n / κ = γ_n",
           title=f"Discrete allowed invariant magnitudes · κ={kappa:g}")
    gaps.bar(np.arange(1, len(gamma)+1), np.diff(np.r_[0, momenta]), color=colors)
    gaps.set(xlabel="Mode n (n=1 includes the reference gap)", ylabel="P_int,n − P_int,n−1",
             title="Nonuniform spacings", xticks=np.arange(1, len(gamma)+1))
    finish(fig, "zeta_internal_momentum_spectrum.png", "Zeta-selected internal momentum ladder")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(p_ext, c*abs(p_ext), "--", color="0.65", lw=1, label="P₀ = 0 reference only")
    for n, (P, E, color) in enumerate(zip(momenta, energies, colors), 1):
        ax.plot(p_ext, E, color=color, label=f"n={n}: P_int={P:.3f}")
        ax.scatter([0], [c*P], color=color, s=22, zorder=4)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set(xlabel="External momentum p_ext", ylabel="Energy E",
           title=f"E = c √(p_ext² + P_int,n²) · c={c:g}, κ={kappa:g}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    finish(fig, "dispersion_branches.png", "External rest can retain nonzero internal invariant momentum",
           "At p_ext = 0, E_n(0) = c P_int,n. External rest does not mean P_int = 0 in this toy model.")

    fig = plt.figure(figsize=(12, 11))
    theta = np.linspace(0, 2*np.pi, 70)
    for i, (state, row, color) in enumerate(zip(states, state_rows, COLORS), 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        for plane in (0, 1, 2):
            ring = np.zeros((3, len(theta)))
            ring[(plane+1) % 3], ring[(plane+2) % 3] = np.cos(theta), np.sin(theta)
            ax.plot(*ring, color="0.8", lw=0.5)
        xyz = state.directions
        sizes = 12 + 150*state.weights/state.weights.max()
        ax.scatter(*xyz.T, s=sizes, c=state.weights/state.weights.max(), cmap="Blues", vmin=0, vmax=1, alpha=0.85)
        if len(xyz) <= 6:
            for direction, weight in zip(xyz, state.weights/state.weights.max()):
                ax.quiver(0, 0, 0, *direction, length=float(weight), color=COLORS[0], arrow_length_ratio=0.12)
        A, B, _, _ = directional_moments(state)
        if np.linalg.norm(B) > 1e-10:
            ax.quiver(0, 0, 0, *(B/A), color=COLORS[1], linewidth=3, arrow_length_ratio=0.25)
            ax.text(*(1.25*B/A), "B/A", color=COLORS[1])
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2),
               xlabel="n_x", ylabel="n_y", zlabel="n_z")
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{state.name}\nA={A:.6f}   |B|={row['B_magnitude']:.6f}\n"
                     f"P_int={row['P_int']:.6f}   error={row['invariant_error']:+.2e}", fontsize=11)
    finish(fig, "directional_states_same_invariant.png",
           f"Different directional geometries · same P_int,{state_rows[0]['zero_index']} = {state_rows[0]['target_P_int']:.6f}",
           "Dots: integrated quadrature/beam weights, normalized within each panel. Orange vector: B/A; not an extra p_ext term.")

    fig, (ax, err) = plt.subplots(1, 2, figsize=(12, 5))
    for n, (E, error, color) in enumerate(zip(energies, kg_errors, colors), 1):
        ax.plot(p_ext, E, color=color, label=f"n={n}")
        ax.plot(p_ext, E+error, "--", color="black", alpha=0.4, lw=0.8)
        err.plot(p_ext, error, color=color)
    ax.set(xlabel="External momentum p_ext", ylabel="Energy E", title="Solid: internal momentum; dashed: ordinary KG")
    ax.legend(ncol=2, fontsize=8)
    err.set(xlabel="External momentum p_ext", ylabel="E_standard − E_internal",
            title=f"Max |difference| = {np.max(abs(kg_errors)):.3e}")
    err.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    finish(fig, "kg_equivalence.png", "KG equivalence with m_n = P_int,n / c")

    fig, (ax, err) = plt.subplots(1, 2, figsize=(12, 6))
    selected = sorted({r["zero_index"] for r in dirac_rows})
    for n in selected:
        subset = [r for r in dirac_rows if r["zero_index"] == n]
        color = colors[n-1]
        curve = internal_energy(p_ext, momenta[n-1], c)
        ax.plot(p_ext, curve, color=color, label=f"n={n}")
        ax.plot(p_ext, -curve, color=color)
        for row in subset:
            ax.scatter([row["p_ext"]]*4, row["computed_eigenvalues"], marker="x", color=color, s=36)
        err.plot([r["p_ext"] for r in subset], [r["max_error"] for r in subset], "o-", color=color, label=f"n={n}")
    ax.set(xlabel="External momentum p_ext", ylabel="Dirac eigenvalue E", title="Curves: ±E; crosses: numerical eigenvalues")
    ax.legend()
    err.set(xlabel="External momentum p_ext", ylabel="Maximum absolute eigenvalue error", title="Both signs, each with multiplicity two")
    err.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    finish(fig, "dirac_spectrum_check.png", "Standard free Dirac Hamiltonian with zeta-selected internal momentum")

    if mapping:
        selected = sorted({r["zero_index"] for r in mapping})[:6]
        fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.5*len(selected)), squeeze=False)
        for ax, n in zip(axes[:, 0], selected):
            subset = [r for r in mapping if r["zero_index"] == n]
            reference = momenta[n-1]
            ax.axhline(reference, color=COLORS[0], label=f"Defined mode: κγ{n} = {reference:.5f}")
            x = [r["prime_cutoff"] for r in subset]
            y = [r["P_peak"] if r["P_peak"] != "" else np.nan for r in subset]
            ax.plot(x, y, ":", color=COLORS[1], alpha=0.5)
            for row, yy in zip(subset, y):
                bracketed = str(row["peak_bracketed"]).lower() == "true"
                ax.scatter(row["prime_cutoff"], yy, color=COLORS[1], marker="o" if bracketed else "x")
            ax.set(xscale="log", ylabel="Internal momentum", title=f"Mode n={n}")
            ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel("Prime cutoff (log scale)")
        finish(fig, "prime_peak_to_internal_mode.png", "Finite-product peak locations compared with imposed modes",
               "Dots: bracketed; crosses: unbracketed. Nearest peaks can switch identity; dotted guides do not establish convergence or causation.")
    return names
