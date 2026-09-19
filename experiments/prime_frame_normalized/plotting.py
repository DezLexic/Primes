"""Static plots at actual coordinates: shared limits and no jitter."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .model import COMPARISON_FRAMES, HISTORY_FRAMES, normalize_points, inherited_zero_points

BLUE = "#165d9c"
RED = "#b83548"
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def _save(fig, output, name):
    fig.savefig(output / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def _axes(ax, normalized, ymax, xmax=1.3):
    ax.set(xlim=(-0.22 * xmax / 1.3, xmax), ylim=(-ymax, ymax),
           xlabel=r"$X=p\,\mathrm{Re}(s_p)$" if normalized else r"$\mathrm{Re}(s_p)$",
           ylabel=r"$Y=p\,\mathrm{Im}(s_p)$" if normalized else r"$\mathrm{Im}(s_p)$")
    ax.axvline(0, color="#777777", lw=0.8)
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)


def _points(ax, points, **kwargs):
    ax.scatter(np.r_[points.real, points.real], np.r_[points.imag, -points.imag], **kwargs)


def _frame(ax, state, colors, ymax, normalized=True, xmax=1.3, new_q=None):
    data = state if normalized else state["raw"]
    gap = 1 if normalized else data["gap"]
    _axes(ax, normalized, ymax, xmax)
    ax.axvline(gap, color=BLUE, ls="--", lw=1)
    _points(ax, data["inherited"], s=18, color=BLUE, zorder=3)
    for i, (q, points) in enumerate(data["combs"].items()):
        ax.scatter(points.real, points.imag, s=30 + 20 * i,
                   facecolors="none", edgecolors=[colors[q]],
                   linewidths=2 if q == new_q else 1.1, zorder=4)
    history = ", ".join(str(q) for q in data["combs"]) or "none"
    ax.set_title(f"p = {state['frame_prime']} · passed q: {history}", fontsize=10)


def save_plots(states, test_point, output):
    by_p = {s["frame_prime"]: s for s in states}
    primes = sorted({q for s in states for q in s["combs"]})
    colors = {q: plt.get_cmap("tab10")((i + 1) % 10) for i, q in enumerate(primes)}
    frame_colors = plt.get_cmap("tab10")
    heights = [abs(z.imag) for s in states for z in s["inherited"]]
    heights += [abs(z.imag) for s in states for c in s["combs"].values() for z in c]
    ymax = 1.12 * max(heights)
    names = []
    handles = [Line2D([], [], marker="o", color=BLUE, ls="", label="Inherited zeros (+ conjugates)")]
    handles += [Line2D([], [], marker="o", markerfacecolor="none", color=colors[q],
                       ls="", label=f"q={q} factor zeros") for q in primes]

    # A: raw limits fixed across frames, normalized limits fixed across frames.
    for p in COMPARISON_FRAMES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
        _frame(axes[0], by_p[p], colors, ymax / 2, False, 0.65)
        _frame(axes[1], by_p[p], colors, ymax)
        axes[0].set_title(f"Raw frame p={p} · separation 1/{p}")
        axes[1].set_title(f"Normalized frame p={p} · separation 1")
        fig.suptitle("A · Original vs normalized frame geometry", fontsize=15)
        axes[1].legend(handles=handles[:1] + [h for q, h in zip(primes, handles[1:]) if q < p],
                       loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        fig.supxlabel("Shared raw limits across frames; shared normalized limits across frames. Independent x/y scales.", fontsize=8)
        names.append(_save(fig, output, f"original_vs_normalized_p{p}.png"))

    # B: descending nested marker sizes expose exact overlap, without offsets.
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    _axes(ax, True, ymax)
    ax.axvline(1, color=BLUE, ls="--", lw=1)
    for i, state in enumerate(states):
        _points(ax, state["inherited"], s=220 - i * 24, marker=MARKERS[i],
                facecolors="none", edgecolors=[frame_colors(i)], linewidths=1,
                label=f"p={state['frame_prime']}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Frame", fontsize=9)
    ax.set_title("B · All normalized inherited zeros coincide")
    fig.supxlabel("Marker centers remain at (1, 2γ), with conjugates below. No horizontal or vertical jitter.", fontsize=9)
    names.append(_save(fig, output, "normalized_inherited_overlay.png"))

    fig, axes = plt.subplots(2, 4, figsize=(15, 10), sharex=True, sharey=True, layout="constrained")
    for ax, q in zip(axes.flat, primes):
        _axes(ax, True, ymax)
        for i, state in enumerate(states):
            if q in state["combs"]:
                points = state["combs"][q]
                ax.scatter(points.real, points.imag, s=220 - i * 24, marker=MARKERS[i],
                           facecolors="none", edgecolors=[frame_colors(i)], linewidths=1)
        ax.set_title(f"q={q} factor · present for p>{q}")
        ax.set_xlim(-0.25, 0.25)
    for ax in list(axes.flat)[len(primes):]:
        ax.axis("off")
    overlay_handles = [Line2D([], [], marker=MARKERS[i], color=frame_colors(i),
                              markerfacecolor="none", ls="", label=f"p={s['frame_prime']}")
                       for i, s in enumerate(states)]
    axes.flat[-1].legend(handles=overlay_handles, loc="center", title="Frame", fontsize=10)
    fig.suptitle("B · Each passed-prime comb coincides across every frame containing it", fontsize=15)
    fig.supxlabel("Each panel retains X=0; splitting by q reveals overlaps. All combs include k=0.", fontsize=9)
    names.append(_save(fig, output, "normalized_comb_overlay.png"))

    fig, axes = plt.subplots(1, 5, figsize=(17, 6), sharex=True, sharey=True)
    for ax, p in zip(axes, HISTORY_FRAMES):
        new_q = max(by_p[p]["combs"])
        _frame(ax, by_p[p], colors, ymax, new_q=new_q)
        ax.text(0.5, 1.07, f"Added q={new_q}", transform=ax.transAxes, ha="center", color=colors[new_q])
    fig.suptitle("C · Geometry stays fixed; history accumulates", fontsize=16)
    fig.legend(handles=handles[:1] + handles[1:6], loc="lower center", ncol=3, fontsize=9)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.23, top=0.78, wspace=0.25)
    names.append(_save(fig, output, "progressive_history.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    diag_ymax = max(2 * abs(test_point.imag), max(abs(by_p[2]["inherited"][:5].imag))) * 1.25
    _axes(axes[0], False, diag_ymax / 2, 0.65)
    _axes(axes[1], True, diag_ymax)
    for i, state in enumerate(states):
        p = state["frame_prime"]
        raw = inherited_zero_points(p, [test_point])
        normalized = normalize_points(p, raw)
        axes[0].axvline(1 / p, color=BLUE, alpha=0.15, ls="--")
        axes[0].scatter(state["raw"]["inherited"][:5].real, state["raw"]["inherited"][:5].imag,
                        color=BLUE, s=14)
        axes[0].scatter(raw.real, raw.imag, color=RED, marker=MARKERS[i], s=50)
        if i < 4:
            axes[0].annotate(f"p={p}", (raw[0].real, raw[0].imag), xytext=(-4, -15),
                             textcoords="offset points", fontsize=8, ha="right")
        else:
            axes[0].annotate(f"p={p}", (raw[0].real, raw[0].imag),
                             xytext=(-0.02, -(0.12 + 0.14 * (i - 4)) * diag_ymax / 2),
                             fontsize=8, ha="right",
                             arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7})
        axes[1].scatter(normalized.real, normalized.imag, s=240 - i * 25,
                        marker=MARKERS[i], facecolors="none", edgecolors=[frame_colors(i)],
                        label=f"artificial, p={p}")
    axes[1].axvline(1, color=BLUE, ls="--", lw=1)
    axes[1].scatter(by_p[2]["inherited"][:5].real, by_p[2]["inherited"][:5].imag,
                    color=BLUE, s=24, label="Sampled inherited zeros")
    axes[1].axvline(0.8, color=RED, ls=":", lw=1)
    axes[1].annotate("Artificial X=0.8\nOffset / gap = −0.2", (0.8, 2 * test_point.imag),
                     xytext=(0.2, -0.35 * diag_ymax), arrowprops={"arrowstyle": "->"}, fontsize=10)
    axes[0].set_title("Raw: artificial and inherited points shrink")
    axes[1].set_title("Normalized: the off-line offset is preserved")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle("Diagnostic · Artificial β=0.4 is never forced onto X=1", fontsize=15)
    fig.supxlabel("Artificial point uses the first sampled zero's height; it is not a claimed zeta zero. Positive heights shown.", fontsize=9)
    names.append(_save(fig, output, "off_line_diagnostic.png"))
    return names
