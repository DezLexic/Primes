"""Static geometry plots; all points retain their actual coordinates (no jitter)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

INHERITED = "#165d9c"
AXIS = "#687582"


def _finish(fig, output, name):
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


def _style(ax):
    ax.set_xlabel(r"$\mathrm{Re}(s_p)$")
    ax.set_ylabel(r"$\mathrm{Im}(s_p)$")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def _draw_frame(ax, state, colors, xlim, ymax):
    p, gap = state["frame_prime"], state["gap"]
    ax.axvline(0, color=AXIS, lw=1.2, label="passed-prime comb axis")
    ax.axvline(gap, color=INHERITED, ls="--", lw=1.3, label="inherited-zero line")
    points = state["inherited"]
    # Display the conjugate zeros too, using the same symmetric height window
    # as the k=-K,...,K combs. Trajectories use positive heights only.
    ax.scatter(np.r_[points.real, points.real], np.r_[points.imag, -points.imag],
               s=22, color=INHERITED, zorder=4, label="inherited zeta zeros (+ conjugates)")
    for index, (q, comb) in enumerate(state["combs"].items()):
        # Nested open circles reveal shared k=0 without moving zeros off x=0.
        ax.scatter(comb.real, comb.imag, s=30 + 19 * index, facecolors="none",
                   edgecolors=[colors[q]], linewidths=1.3, zorder=5,
                   label=f"q={q} factor zeros")
    if not state["combs"]:
        ax.text(0.04, 0.96, "No passed primes", transform=ax.transAxes,
                va="top", fontsize=9, color=AXIS)
    y = -0.88 * ymax
    ax.annotate("", xy=(0, y), xytext=(gap, y),
                arrowprops={"arrowstyle": "<->", "color": "#222222"})
    ax.text(gap / 2, y + 0.055 * ymax, rf"$\Delta_p=1/{p}$", ha="center",
            fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.set(xlim=xlim, ylim=(-ymax, ymax), title=f"Frame p = {p}")
    _style(ax)


def plot_single_frame(state, colors, output):
    """Autoscale each individual view to include every requested comb point."""
    heights = [np.max(np.abs(state["inherited"].imag))]
    heights.extend(np.max(np.abs(comb.imag)) for comb in state["combs"].values())
    gap = state["gap"]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    _draw_frame(ax, state, colors, (-0.3 * gap, 1.45 * gap), 1.2 * max(heights))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.suptitle("Prime-frame geometry", fontsize=16)
    fig.supxlabel("Coordinate separation; independent x/y scales. Coincident comb zeros are not displaced.", fontsize=9)
    return _finish(fig, output, f"frame_{state['frame_prime']}.png")


def plot_frame_comparison(states, colors, output):
    columns = min(5, len(states))
    rows = (len(states) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 6 * rows + 1),
                             sharex=True, sharey=True, squeeze=False)
    gap = max(state["gap"] for state in states)
    heights = [abs(z.imag) for state in states for z in state["inherited"]]
    heights.extend(abs(z.imag) for state in states for comb in state["combs"].values() for z in comb)
    for ax, state in zip(axes.flat, states):
        _draw_frame(ax, state, colors, (-0.3 * gap, 1.45 * gap), 1.2 * max(heights))
    for ax in list(axes.flat)[len(states):]:
        ax.set_visible(False)
    handles, labels = axes.flat[len(states) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9)
    fig.suptitle("Inherited-zero lines and passed-prime zero combs\nShared coordinate limits across prime frames", fontsize=16)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.22, top=0.84, wspace=0.23, hspace=0.35)
    return _finish(fig, output, "frame_comparison.png")


def plot_zero_trajectories(states, track_count, output):
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    colors = plt.get_cmap("tab10")
    for state in states:
        x = state["gap"]
        ax.axvline(x, color=AXIS, alpha=0.35, ls="--", lw=1)
        ax.text(x, 1.015, f"p={state['frame_prime']}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, rotation=45)
    for index in range(track_count):
        points = np.array([state["inherited"][index] for state in states])
        color = colors(index % 10)
        ax.plot(points.real, points.imag, "o-", color=color, ms=5, lw=1.3,
                label=rf"$\rho_{{{index + 1}}}$")
        for start, end in zip(points[:-1], points[1:]):
            ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5,
                                    "shrinkA": 7, "shrinkB": 7})
    ax.plot(0, 0, "+", color="black", ms=9)
    ax.set_xlim(-0.025, max(s["gap"] for s in states) * 1.12)
    ax.set_ylim(bottom=-1)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=AXIS, ls="--", alpha=0.5))
    labels.append("inherited-zero lines")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("The same inherited zeros across prime frames", fontsize=16)
    fig.supxlabel("Arrows follow increasing p toward the origin; positive heights shown. Independent x/y scales.", fontsize=9)
    return _finish(fig, output, "zero_trajectories.png")


def plot_frame_gaps(states, output):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    primes = [s["frame_prime"] for s in states]
    gaps = [s["gap"] for s in states]
    ax.plot(primes, gaps, "o-", color=INHERITED)
    for p, gap in zip(primes, gaps):
        ax.annotate(f"1/{p}", (p, gap), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set(xlabel="Frame prime p", ylabel=r"Horizontal separation $\Delta_p=1/p$",
           title="Geometric frame separation", ylim=(0, max(gaps) * 1.2))
    if len(primes) <= 15:
        ax.set_xticks(primes)
    ax.grid(alpha=0.2)
    fig.supxlabel(r"From passed-prime comb axis Re$(s_p)=0$ to inherited-zero line Re$(s_p)=1/p$", fontsize=9)
    return _finish(fig, output, "frame_gap.png")


def save_plots(states, track_count, output):
    passed = sorted({q for state in states for q in state["combs"]})
    palette = plt.get_cmap("tab20")
    # Use saturated, distinct hues before their lighter partners; reserve blue
    # for inherited zeros for as long as the palette permits.
    order = list(range(2, 20, 2)) + [0] + list(range(3, 20, 2)) + [1]
    colors = {q: palette(order[i % 20]) for i, q in enumerate(passed)}
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12}):
        names = [plot_single_frame(state, colors, output) for state in states]
        names += [plot_frame_comparison(states, colors, output),
                  plot_zero_trajectories(states, track_count, output),
                  plot_frame_gaps(states, output)]
    return names
