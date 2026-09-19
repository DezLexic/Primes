"""Static phasor sequence and finite-product plots. No physics is modeled."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.prime_frame_geometry.model import passed_prime_list, passed_prime_comb_points
from experiments.prime_frame_normalized.model import normalized_frame, normalize_points
from .model import (PRIMES, HISTORY_FRAMES, TRACK_FRAMES, phasors, factor_power,
                    minimum_power, zero_heights, history_power)

COLORS = ("#1768ac", "#d66024", "#26834a", "#9146a3")
MARKERS = ("o", "s", "^", "D")


def _save(fig, output, name):
    fig.savefig(output / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def _clean(ax):
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _arrow(ax, start, vector, color, style="-", width=2):
    ax.annotate("", xy=(start.real + vector.real, start.imag + vector.imag),
                xytext=(start.real, start.imag),
                arrowprops=dict(arrowstyle="->", color=color, lw=width, linestyle=style))


def save_plots(output, tau, tau_max, zeros):
    names = []
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), layout="constrained")
    for row, sigma in enumerate((0, 0.5)):
        for col, phase in enumerate(np.linspace(0, 2*np.pi, 5)):
            ax = axes[row, col]
            a1, a2, total = (complex(v) for v in phasors(2, sigma, phase/np.log(2)))
            _arrow(ax, 0j, a1, COLORS[0])
            _arrow(ax, 0j, a2, COLORS[1])
            _arrow(ax, a1, a2, COLORS[1], "--", 1.5)
            if abs(total) < 1e-12:
                ax.scatter([0], [0], marker="x", color=COLORS[2], s=70, zorder=5)
                ax.text(0.04, 0.04, "F = 0", transform=ax.transAxes, color=COLORS[2])
            else:
                _arrow(ax, 0j, total, COLORS[2], width=3)
            ax.set(xlim=(-1.3, 2.3), ylim=(-1.3, 1.3), aspect="equal",
                   title=f"σ={sigma:g} · phase={phase/np.pi:g}π\n|F|={abs(total):.3f}",
                   xlabel="Real", ylabel="Imaginary" if col == 0 else "")
            _clean(ax)
    handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in zip(COLORS,
               ("A₁ = 1", "A₂ = −q⁻ᶻ (dashed: translated to A₁ tip)", "F = A₁ + A₂"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle("q=2 · One full phase cycle: exact cancellation only at matched magnitudes\n"
                 "σ=0: |A₂|=1; σ=½: |A₂|=1/√2, so |F| ≥ 1−1/√2 > 0", fontsize=14)
    names.append(_save(fig, output, "phasor_panels.png"))

    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    theta = np.linspace(0, 2*np.pi, 501)
    for sigma, color in zip((0, 0.5, -0.5), COLORS):
        r = 2.0 ** (-sigma)
        ax.plot(r*np.cos(theta), -r*np.sin(theta), color=color, label=f"σ={sigma:g}: radius={r:.3f}")
    ax.scatter([1], [0], marker="*", s=180, c="black", zorder=5, label="Target 1 + 0i")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set(aspect="equal", xlabel="Real", ylabel="Imaginary",
           title="q=2 · The circle q⁻ᶻ reaches 1 only when its radius is 1")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.supxlabel("This circle is the unsigned q⁻ᶻ; the arrow panels use A₂ = −q⁻ᶻ.\n"
                  "Circles of radius <1 or >1 never pass through the target.", fontsize=10)
    _clean(ax)
    names.append(_save(fig, output, "phasor_circles.png"))

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, sharey=True, layout="constrained")
    for i, q in enumerate(PRIMES):
        heights = zero_heights(q, tau_max)[1]
        for j, sigma in enumerate((0, 0.5)):
            ax = axes[i, j]
            ax.plot(tau, factor_power(q, sigma, tau), color=COLORS[i])
            for t in heights:
                ax.axvline(t, color=COLORS[i], alpha=0.23, lw=0.8)
            floor = float(minimum_power(q, sigma))
            ax.scatter(heights, np.full(len(heights), floor), color=COLORS[i], marker=MARKERS[i], s=22)
            ax.axhline(floor, color="gray", ls="--", lw=0.8)
            ax.set(title=f"q={q}, σ={sigma:g} · minimum={floor:.6f}", ylabel="|Fq|²", ylim=(-0.15, 4.25))
            _clean(ax)
    for ax in axes[-1]:
        ax.set_xlabel("τ (common coordinate)")
    fig.suptitle("Single factors · Phase matching gives minima; magnitude matching makes them zeros")
    fig.supxlabel("Vertical markers: τ=2πk/ln(q). Right column: positive minima, not zeros.", fontsize=10)
    names.append(_save(fig, output, "single_prime_power.png"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
    for i, q in enumerate(PRIMES):
        # Offset rows are a labelled display convention, not modified coordinates.
        axes[0].plot(tau, factor_power(q, 0, tau) + 5*i, color=COLORS[i])
        heights = zero_heights(q, tau_max)[1]
        axes[0].scatter(heights, np.full(len(heights), 5*i), color=COLORS[i], marker=MARKERS[i])
    axes[0].set(yticks=[5*i for i in range(4)], yticklabels=[f"q={q}" for q in PRIMES],
                xlabel="τ", ylabel="|Fq|², rows offset by 5", title="σ=0 · Larger q gives a tighter comb")
    spacings = 2*np.pi/np.log(PRIMES)
    axes[1].bar([str(q) for q in PRIMES], spacings, color=COLORS)
    for i, spacing in enumerate(spacings):
        axes[1].text(i, spacing+0.12, f"{spacing:.4f}", ha="center")
    axes[1].set(xlabel="Prime q", ylabel="Δτ = 2π / ln(q)", ylim=(0, 10.5), title="Phase rate ln(q); comb spacing Δτ")
    for ax in axes:
        _clean(ax)
    fig.suptitle("Prime phase-rate comparison · Mathematical oscillations, no physical frequency")
    names.append(_save(fig, output, "prime_comparison.png"))

    fig, axes = plt.subplots(len(HISTORY_FRAMES), 2, figsize=(14, 15), sharex=True, layout="constrained")
    for i, p in enumerate(HISTORY_FRAMES):
        passed = passed_prime_list(p)
        for j, sigma in enumerate((0, 0.5)):
            ax = axes[i, j]
            ax.plot(tau, history_power(p, sigma, tau), color=COLORS[j], lw=1)
            if sigma == 0:
                for q in passed:
                    heights = zero_heights(int(q), tau_max)[1]
                    ax.scatter(heights, np.zeros_like(heights), s=7, c="black", zorder=3)
                ax.set_ylim(bottom=-0.03*max(history_power(p, sigma, tau)))
            else:
                ax.set_yscale("log")
            ax.set(title=f"p={p} · q<{p}: {', '.join(map(str, passed))} · σ={sigma:g}", ylabel="|Rp|²")
            _clean(ax)
    for ax in axes[-1]:
        ax.set_xlabel("τ")
    fig.suptitle("Accumulated finite history · |Rp|² = ∏q<p |Fq|²\n"
                 "Left: linear scale, analytic zero markers. Right: log scale reveals strictly positive modulation.")
    fig.supxlabel("Individual panels autoscale vertically. No zeta multiplier and no baseline normalization.", fontsize=10)
    names.append(_save(fig, output, "history_products.png"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
    for j, sigma in enumerate((0, 0.5)):
        for i, q in enumerate(PRIMES):
            axes[0, j].plot(tau, factor_power(q, sigma, tau), color=COLORS[i], label=f"q={q}", lw=1)
        product = history_power(11, sigma, tau)
        summed = sum(phasors(q, sigma, tau)[2] for q in PRIMES)
        axes[1, j].plot(tau, product, color=COLORS[2])
        axes[2, j].plot(tau, abs(summed)**2, color=COLORS[3])
        axes[0, j].set_title(f"σ={sigma:g}: elementary two-phasor factors")
        axes[0, j].legend(ncol=4, fontsize=8)
        axes[0, j].set_ylabel("|Fq|²")
        axes[1, j].set_ylabel("∏ |Fq|² = |R₁₁|²")
        axes[2, j].set(ylabel="|Σ Fq|² (different object)", xlabel="τ")
        for ax in axes[:, j]:
            _clean(ax)
    fig.suptitle("Multiplication versus addition · The history model is the middle row\n"
                 "Bottom row is a mathematical comparison only; it is not the history factor or a physical slit model.")
    names.append(_save(fig, output, "product_vs_sum.png"))

    fig, axes = plt.subplots(1, 5, figsize=(15, 9), sharey=True, layout="constrained")
    axes[0].axvline(0, color="gray", lw=0.7)
    for i, q in enumerate(PRIMES):
        heights = zero_heights(q, tau_max)[1]
        axes[0].scatter(np.zeros_like(heights), heights, marker=MARKERS[i], s=100-20*i,
                        facecolors="none", edgecolors=COLORS[i], label=f"q={q}")
        ax = axes[i+1]
        ax.plot(factor_power(q, 0, tau), tau, color=COLORS[i])
        ax.scatter(np.zeros_like(heights), heights, color=COLORS[i], marker=MARKERS[i], s=25)
        for height in heights:
            ax.axhline(height, color=COLORS[i], alpha=0.18, lw=0.7)
        ax.set(xlabel="|Fq|² at σ=0", title=f"q={q}", xlim=(-0.3, 4.3))
    axes[0].set(xlim=(-0.3, 0.3), ylim=(-tau_max, tau_max), xlabel="Re(z)", ylabel="τ = Im(z)", title="All factor zeros")
    axes[0].legend(fontsize=8, loc="upper left")
    for ax in axes:
        _clean(ax)
    fig.suptitle("p=11 · A zero dip and a point on Re(z)=0 share the same height\n"
                 "Common z coordinates; curves are rotated to align their τ axes with the zero plot.")
    fig.supxlabel("Distinct open markers retain q identity at the shared origin; no coordinate jitter.", fontsize=10)
    names.append(_save(fig, output, "zero_line_correspondence.png"))

    k_max = int(np.ceil(tau_max*np.log(7)/(2*np.pi)))
    state = normalized_frame(11, zeros, k_max)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, (q, points) in enumerate(state["combs"].items()):
        # Use the same common-coordinate window as the analytic-zero CSV.
        points = points[np.abs(points.imag) <= 2*tau_max]
        ax.scatter(points.real, points.imag, marker=MARKERS[i], s=100-20*i,
                   facecolors="none", edgecolors=COLORS[i], label=f"q={q}")
    inherited = np.r_[state["inherited"], state["inherited"].conjugate()]
    ax.scatter(inherited.real, inherited.imag, color="black", marker="x", label="Sampled inherited zeros")
    for x in (0, 1):
        ax.axvline(x, color="gray", lw=0.8, ls="--")
    ax.set(xlim=(-0.35, 1.4), xlabel="X = p Re(sₚ) = 2 Re(z)", ylabel="Y = p Im(sₚ) = 2 Im(z)",
           title="Normalized frame p=11: two distinct families")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
    _clean(ax)
    axes[1].axis("off")
    axes[1].text(0, 0.94,
        "X = 0 · Passed-prime factor zeros\n\n"
        "1 − q⁻ᶻ = 0 requires q⁻ᶻ = 1.\n"
        "Magnitude: q^(−σ) = 1  ⇒  σ = 0.\n"
        "Phase: τ ln(q) = 2πk.\n"
        "Thus Y = 4πk / ln(q), independent of p.\n\n"
        "The signed arrows 1 and −q⁻ᶻ cancel exactly.\n"
        "These zeros are produced by the factors;\n"
        "they are not inherited zeros moved to X=0.\n\n"
        "X = 1 · Sampled critical-line zeta zeros\n\n"
        "ρ = ½ + iγ maps to (X,Y) = (1,2γ).\n"
        "General β + iγ maps to X=2β.\n"
        "No identical two-phasor mechanism is established\n"
        "for inherited zeta zeros. This does not prove RH.\n\n"
        "Independent axis scales; no physical model.",
        va="top", fontsize=12, linespacing=1.45)
    fig.suptitle("Why the factor combs lie on the zero line", fontsize=16)
    names.append(_save(fig, output, "normalized_geometry_phasors.png"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    for i, p in enumerate(TRACK_FRAMES):
        raw = passed_prime_comb_points(p, 2, 3)
        normalized = normalize_points(p, raw)
        axes[0].scatter(np.full(raw.size, p), raw.imag, label=f"p={p}")
        axes[1].scatter(normalized.real, normalized.imag, s=220-32*i, facecolors="none",
                        edgecolors=[plt.get_cmap("tab10")(i)], marker=("o", "s", "^", "D", "v", "P")[i], label=f"p={p}")
    axes[0].set(xlabel="Frame prime p (comparison index)", ylabel="Im(sₚ)", title="q=2 raw heights shrink as 1/p", xticks=TRACK_FRAMES)
    axes[1].set(xlim=(-0.25, 0.25), xlabel="X", ylabel="Y=p Im(sₚ)", title="q=2 normalized combs coincide exactly")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    for ax in axes:
        _clean(ax)
    fig.suptitle("Tracking one passed prime · k=−3,…,3 retained in every frame")
    names.append(_save(fig, output, "frame_normalization.png"))
    return names
