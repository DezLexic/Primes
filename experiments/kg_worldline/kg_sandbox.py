"""Klein-Gordon worldline clock sandbox.

This version distinguishes two observations of the same KG plane wave:

1. At a fixed laboratory position, the phase frequency is E = gamma*m.
2. Along the particle's worldline x = vt, the phase frequency is m/gamma.

The second view is the particle's time-dilated internal clock. The script also
lets momentum increase under a constant force and numerically integrates

    d_tau/dt = 1/gamma(t)
    phase(t) = m * tau(t)

so the oscillations stretch continuously. Prime numbers are calculated only
after the KG crossings have been generated; they never enter the dynamics.

Natural units are used throughout: c = hbar = 1.

Install:
    .venv/Scripts/python -m pip install -r requirements.txt

Run:
    .venv/Scripts/python experiments/kg_worldline/kg_sandbox.py

Examples:
    .venv/Scripts/python experiments/kg_worldline/kg_sandbox.py --view-momentum 0
    .venv/Scripts/python experiments/kg_worldline/kg_sandbox.py --force 0.02 --duration 200
    .venv/Scripts/python experiments/kg_worldline/kg_sandbox.py --mass 2 --force 0.1
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MASS = 1.0
DEFAULT_DURATION = 100.0
DEFAULT_FORCE = 0.05
DEFAULT_VIEW_MOMENTUM = 2.0
DEFAULT_HIT_TOLERANCE = 0.10
COMPARISON_MOMENTA = (0.0, 1.0, 2.0, 5.0)
NUM_POINTS = 50_000


def lorentz_gamma(momentum: np.ndarray | float, mass: float) -> np.ndarray | float:
    """gamma = E/m = sqrt(1 + (p/m)^2) in natural units."""
    return np.sqrt(1.0 + (np.asarray(momentum) / mass) ** 2)


def cumulative_trapezoid(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration without an extra dependency."""
    integral = np.zeros_like(values)
    integral[1:] = np.cumsum(
        0.5 * (values[:-1] + values[1:]) * np.diff(coordinate)
    )
    return integral


def constant_momentum_worldline_phase(
    lab_time: np.ndarray,
    momentum: float,
    mass: float,
) -> np.ndarray:
    """KG phase sampled along x=vt: theta = (m/gamma)t."""
    gamma = float(lorentz_gamma(momentum, mass))
    return mass * lab_time / gamma


def accelerated_worldline(
    lab_time: np.ndarray,
    mass: float,
    force: float,
) -> dict[str, np.ndarray]:
    """A simple blind law p(t)=F*t, followed along the particle worldline."""
    momentum = force * lab_time
    gamma = np.asarray(lorentz_gamma(momentum, mass))
    proper_time_rate = 1.0 / gamma
    proper_time = cumulative_trapezoid(proper_time_rate, lab_time)
    phase = mass * proper_time
    return {
        "momentum": momentum,
        "gamma": gamma,
        "proper_time_rate": proper_time_rate,
        "proper_time": proper_time,
        "phase": phase,
    }


def phase_crossings(lab_time: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Return lab times where sin(phase) crosses zero, excluding t=0."""
    maximum_index = int(np.floor(phase[-1] / np.pi))
    if maximum_index < 1:
        return np.array([], dtype=float)
    target_phases = np.arange(1, maximum_index + 1) * np.pi
    return np.interp(target_phases, phase, lab_time)


def select_near_integer_events(
    crossings: np.ndarray,
    tolerance: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find KG crossings that naturally land close to integer lab times."""
    if crossings.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    nearest = np.rint(crossings).astype(int)
    distances = np.abs(crossings - nearest)
    valid = (
        (nearest >= 2)
        & (nearest <= int(np.floor(duration)))
        & (distances <= tolerance)
    )

    best: dict[int, float] = {}
    for integer, distance in zip(nearest[valid], distances[valid]):
        if integer not in best or distance < best[integer]:
            best[integer] = float(distance)

    selected = np.array(sorted(best), dtype=int)
    selected_distances = np.array([best[value] for value in selected])
    return selected, selected_distances


def prime_mask(limit: int) -> np.ndarray:
    """Diagnostic only: primality never enters the KG calculation."""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for value in range(2, int(np.sqrt(limit)) + 1):
        if sieve[value]:
            sieve[value * value : limit + 1 : value] = False
    return sieve


def print_diagnostic(
    crossings: np.ndarray,
    selected: np.ndarray,
    distances: np.ndarray,
    is_prime: np.ndarray,
    mass: float,
    force: float,
    duration: float,
    tolerance: float,
    final_gamma: float,
) -> None:
    all_primes = np.flatnonzero(is_prime)
    selected_prime_mask = is_prime[selected] if selected.size else np.array([], dtype=bool)
    true_positives = int(np.count_nonzero(selected_prime_mask))
    precision = true_positives / len(selected) if len(selected) else 0.0
    recall = true_positives / len(all_primes) if len(all_primes) else 0.0

    print()
    print("Time-dilated Klein-Gordon worldline experiment")
    print("=" * 68)
    print(f"mass m                  : {mass:g}")
    print(f"constant force F        : {force:g}")
    print(f"lab-time range          : 0 to {duration:g}")
    print(f"final gamma             : {final_gamma:.6f}")
    print(f"final clock rate 1/gamma: {1.0 / final_gamma:.6f}")
    print(f"zero crossings          : {len(crossings)}")
    print()
    print("Worldline clock:")
    print("  p(t) = F*t")
    print("  d_tau/dt = 1/sqrt(1 + (p(t)/m)^2)")
    print("  phase(t) = m * integral(d_tau/dt) dt")
    print()
    print("Near-integer event rule:")
    print(f"  a continuous KG crossing must be within {tolerance:g} of an integer t")
    print()

    if selected.size:
        print(f"{'integer':>9} {'distance':>12} {'diagnostic':>14}")
        print("-" * 39)
        for integer, distance in zip(selected, distances):
            label = "prime" if is_prime[integer] else "composite"
            print(f"{integer:9d} {distance:12.6f} {label:>14}")
    else:
        print("No crossings landed within the near-integer tolerance.")

    print()
    print("After-the-fact prime comparison (not used by the field)")
    print("-" * 68)
    print(f"selected integers      : {len(selected)}")
    print(f"selected primes        : {true_positives}")
    print(f"precision              : {precision:.3f}")
    print(f"prime recall           : {recall:.3f}")


def build_figure(
    mass: float,
    force: float,
    duration: float,
    view_momentum: float,
    tolerance: float,
) -> plt.Figure:
    lab_time = np.linspace(0.0, duration, NUM_POINTS)
    accelerated = accelerated_worldline(lab_time, mass, force)
    crossings = phase_crossings(lab_time, accelerated["phase"])
    selected, distances = select_near_integer_events(
        crossings, tolerance, duration
    )
    is_prime = prime_mask(int(np.floor(duration)))

    print_diagnostic(
        crossings=crossings,
        selected=selected,
        distances=distances,
        is_prime=is_prime,
        mass=mass,
        force=force,
        duration=duration,
        tolerance=tolerance,
        final_gamma=float(accelerated["gamma"][-1]),
    )

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 0.9])
    ax_clocks = fig.add_subplot(grid[0, 0])
    ax_complex = fig.add_subplot(grid[0, 1])
    ax_accelerated = fig.add_subplot(grid[1, :])
    ax_rate = fig.add_subplot(grid[2, 0])
    ax_events = fig.add_subplot(grid[2, 1])

    # Higher momentum gives a visibly longer worldline-clock period.
    for momentum in COMPARISON_MOMENTA:
        gamma = float(lorentz_gamma(momentum, mass))
        phase = constant_momentum_worldline_phase(lab_time, momentum, mass)
        ax_clocks.plot(
            lab_time,
            np.cos(phase),
            lw=1.35,
            label=fr"p={momentum:g}, $\gamma$={gamma:.2f}",
        )
    ax_clocks.axhline(0, color="black", lw=0.7, alpha=0.45)
    ax_clocks.set_title("Constant momentum: worldline clocks stretch by gamma")
    ax_clocks.set_xlabel("laboratory time t")
    ax_clocks.set_ylabel(r"Re $\psi[x(t),t]$")
    ax_clocks.set_ylim(-1.15, 1.15)
    ax_clocks.legend(loc="upper right", ncols=2, fontsize=9)
    ax_clocks.grid(alpha=0.2)

    # Two projections of one time-dilated complex phase along the trajectory.
    view_phase = constant_momentum_worldline_phase(
        lab_time, view_momentum, mass
    )
    view_gamma = float(lorentz_gamma(view_momentum, mass))
    ax_complex.plot(lab_time, np.cos(view_phase), label=r"Re $\psi$", lw=1.8)
    ax_complex.plot(lab_time, -np.sin(view_phase), label=r"Im $\psi$", lw=1.8)
    ax_complex.axhline(0, color="black", lw=0.7, alpha=0.45)
    ax_complex.set_title(
        f"One complex worldline clock: p={view_momentum:g}, gamma={view_gamma:.2f}"
    )
    ax_complex.set_xlabel("laboratory time t")
    ax_complex.set_ylabel(r"$\psi[x(t),t]$")
    ax_complex.set_ylim(-1.15, 1.15)
    ax_complex.legend(loc="upper right")
    ax_complex.grid(alpha=0.2)

    # With p(t)=Ft, integrated proper-time phase stretches continuously.
    rest_phase = mass * lab_time
    ax_accelerated.plot(
        lab_time,
        np.cos(rest_phase),
        lw=1.0,
        alpha=0.45,
        label="rest clock",
    )
    ax_accelerated.plot(
        lab_time,
        np.cos(accelerated["phase"]),
        lw=1.8,
        label="accelerating worldline clock",
    )
    ax_accelerated.axhline(0, color="black", lw=0.7, alpha=0.45)
    for crossing in crossings:
        ax_accelerated.axvline(crossing, color="black", lw=0.45, alpha=0.13)
    ax_accelerated.set_title(
        r"Continuous stretching under the blind law $p(t)=Ft$: "
        + fr"$F={force:g}$"
    )
    ax_accelerated.set_xlabel("laboratory time / number-line parameter t")
    ax_accelerated.set_ylabel(r"Re $\psi[x(t),t]$")
    ax_accelerated.set_ylim(-1.15, 1.15)
    ax_accelerated.legend(loc="upper right")
    ax_accelerated.grid(alpha=0.2)

    # Expose the cause of the stretching.
    ax_rate.plot(
        lab_time,
        accelerated["gamma"],
        color="#7570b3",
        lw=1.8,
    )
    ax_rate.set_xlabel("laboratory time t")
    ax_rate.set_ylabel(r"Lorentz factor $\gamma$", color="#7570b3")
    ax_rate.tick_params(axis="y", labelcolor="#7570b3")
    ax_rate.grid(alpha=0.2)
    ax_rate_clock = ax_rate.twinx()
    ax_rate_clock.plot(
        lab_time,
        accelerated["proper_time_rate"],
        color="#1b9e77",
        lw=1.8,
    )
    ax_rate_clock.set_ylabel("proper-time rate", color="#1b9e77")
    ax_rate_clock.tick_params(axis="y", labelcolor="#1b9e77")
    ax_rate.set_title("Why the wave stretches")

    # KG dynamics finish before this diagnostic sees any prime number.
    primes = np.flatnonzero(is_prime)
    ax_events.scatter(
        crossings,
        np.zeros_like(crossings),
        marker="|",
        s=150,
        color="#333333",
        label="continuous KG crossings",
    )
    ax_events.scatter(
        primes,
        np.ones_like(primes),
        marker="|",
        s=150,
        color="#377eb8",
        label="actual primes (diagnostic only)",
    )
    if selected.size:
        selected_prime_mask = is_prime[selected]
        ax_events.scatter(
            selected[~selected_prime_mask],
            np.full(np.count_nonzero(~selected_prime_mask), 0.45),
            marker="x",
            s=65,
            color="#d95f02",
            label="near-integer composite",
            zorder=4,
        )
        ax_events.scatter(
            selected[selected_prime_mask],
            np.full(np.count_nonzero(selected_prime_mask), 0.45),
            marker="o",
            s=65,
            facecolors="none",
            edgecolors="#1b9e77",
            linewidths=1.8,
            label="near-integer prime",
            zorder=5,
        )
    ax_events.set_xlim(0, duration)
    ax_events.set_ylim(-0.3, 1.3)
    ax_events.set_yticks([0, 0.45, 1])
    ax_events.set_yticklabels(["KG", "integer hits", "primes"])
    ax_events.set_xlabel("laboratory time / number-line parameter t")
    ax_events.set_title("Crossings first; number-theory labels afterward")
    ax_events.legend(loc="upper right", fontsize=8)
    ax_events.grid(axis="x", alpha=0.15)

    fig.suptitle(
        "Klein-Gordon Worldline Sandbox — Momentum Produces Time-Dilated Stretching",
        fontsize=16,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow KG phase along time-dilated particle worldlines."
    )
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS)
    parser.add_argument("--force", type=float, default=DEFAULT_FORCE)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument(
        "--view-momentum",
        type=float,
        default=DEFAULT_VIEW_MOMENTUM,
        help="fixed momentum for the complex worldline-clock panel",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_HIT_TOLERANCE,
        help="maximum distance for a continuous crossing to select an integer",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "figure.png",
        help="saved image path (default: this experiment/results/figure.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="build the figure without opening a window",
    )
    args = parser.parse_args()

    if args.mass <= 0:
        parser.error("--mass must be positive for a proper-time rest clock")
    if args.force < 0:
        parser.error("--force must be non-negative")
    if args.duration < 2:
        parser.error("--duration must be at least 2")
    if args.tolerance < 0 or args.tolerance > 0.5:
        parser.error("--tolerance must be between 0 and 0.5")
    return args


def run(args) -> plt.Figure:
    figure = build_figure(
        mass=args.mass,
        force=args.force,
        duration=args.duration,
        view_momentum=args.view_momentum,
        tolerance=args.tolerance,
    )
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"\nSaved figure to: {args.save.resolve()}")
    return figure


def main() -> None:
    args = parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    # Save the numerical summary and settings with every figure.
    capture = io.StringIO()
    with redirect_stdout(capture):
        figure = run(args)
    summary = capture.getvalue()
    print(summary, end="")
    args.save.with_suffix(".txt").write_text(summary, encoding="utf-8")
    args.save.with_suffix(".json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
