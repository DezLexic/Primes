"""
Ordered Clifford internal-momentum sandbox.

This is a speculative follow-up to the Klein-Gordon control experiment.  It is
not the free scalar KG equation.  It tests a specific proposed mechanism:

  * six equal, opposing momentum channels keep visible momentum equal to zero;
  * their nonzero sum Q is used as a mass-like internal momentum scale;
  * a hidden rotation plane B(x) moves between quaternion units J and K;
  * ordered, noncommuting rotations can generate an observed I component;
  * primes are consulted only after all I-channel events are generated.

Quaternion basis and orientation:

    I^2 = J^2 = K^2 = -1
    I J = K,  J K = I,  K I = J
    J K = -K J

Natural units are used.  The horizontal variable x can be read as lab time for
a driven system, or simply as the parameter indexing a family of states.  Since
Q grows, an isolated physical system would require an external energy source.

Run:
    .venv/Scripts/python experiments/clifford_momentum/clifford_momentum_sandbox.py

Examples:
    .venv/Scripts/python experiments/clifford_momentum/clifford_momentum_sandbox.py --orientation-rate 0
    .venv/Scripts/python experiments/clifford_momentum/clifford_momentum_sandbox.py --momentum-growth 0
    .venv/Scripts/python experiments/clifford_momentum/clifford_momentum_sandbox.py --orientation-rate 0.5 --duration 200
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DURATION = 120.0
DEFAULT_POINTS = 60_001
DEFAULT_Q0 = 0.65
DEFAULT_Q_GROWTH = 0.012
DEFAULT_ORIENTATION_RATE = 0.23
DEFAULT_INTEGER_TOLERANCE = 0.10


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for arrays [scalar, I, J, K]."""
    a, b, c, d = left
    e, f, g, h = right
    return np.array(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ],
        dtype=float,
    )


def unit_quaternion_step(axis: np.ndarray, angle: float) -> np.ndarray:
    """exp(B*angle) for a unit pure quaternion B with B^2=-1."""
    return np.concatenate(([np.cos(angle)], np.sin(angle) * axis))


def simulate_rotor(
    coordinate: np.ndarray,
    q0: float,
    q_growth: float,
    orientation_rate: float,
) -> dict[str, np.ndarray]:
    """Integrate R' = Q(x) B(x) R with midpoint, path-ordered steps."""
    rotor = np.zeros((coordinate.size, 4), dtype=float)
    rotor[0, 0] = 1.0

    q = q0 + q_growth * coordinate
    chi = orientation_rate * coordinate

    for index in range(1, coordinate.size):
        dx = coordinate[index] - coordinate[index - 1]
        x_mid = 0.5 * (coordinate[index] + coordinate[index - 1])
        q_mid = q0 + q_growth * x_mid
        chi_mid = orientation_rate * x_mid

        # B = J*cos(chi) + K*sin(chi).  Its coefficient order is I,J,K.
        axis = np.array([0.0, np.cos(chi_mid), np.sin(chi_mid)])
        step = unit_quaternion_step(axis, q_mid * dx)
        rotor[index] = quaternion_multiply(step, rotor[index - 1])

        # Suppress accumulated floating-point drift from repeated products.
        rotor[index] /= np.linalg.norm(rotor[index])

    return {"rotor": rotor, "q": q, "chi": chi}


def find_zero_crossings(coordinate: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Linearly interpolate strict sign-changing zero crossings."""
    indices = np.flatnonzero(signal[:-1] * signal[1:] < 0)
    crossings = []
    for index in indices:
        x0, x1 = coordinate[index], coordinate[index + 1]
        y0, y1 = signal[index], signal[index + 1]
        crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return np.asarray(crossings, dtype=float)


def nearest_integer_events(
    crossings: np.ndarray,
    duration: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select I-channel crossings that lie naturally near integer x values."""
    if crossings.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    nearest = np.rint(crossings).astype(int)
    distance = np.abs(crossings - nearest)
    valid = (
        (nearest >= 2)
        & (nearest <= int(np.floor(duration)))
        & (distance <= tolerance)
    )

    best: dict[int, float] = {}
    for integer, error in zip(nearest[valid], distance[valid]):
        if integer not in best or error < best[integer]:
            best[integer] = float(error)

    selected = np.asarray(sorted(best), dtype=int)
    errors = np.asarray([best[value] for value in selected], dtype=float)
    return selected, errors


def prime_mask(limit: int) -> np.ndarray:
    """After-the-fact diagnostic; never used by the rotor dynamics."""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for value in range(2, int(np.sqrt(limit)) + 1):
        if sieve[value]:
            sieve[value * value : limit + 1 : value] = False
    return sieve


def gap_variation(crossings: np.ndarray) -> tuple[float, float, float]:
    """Return mean gap, standard deviation, and coefficient of variation."""
    if crossings.size < 2:
        return float("nan"), float("nan"), float("nan")
    gaps = np.diff(crossings)
    mean = float(np.mean(gaps))
    standard_deviation = float(np.std(gaps))
    coefficient = standard_deviation / mean if mean else float("nan")
    return mean, standard_deviation, coefficient


def print_summary(
    crossings: np.ndarray,
    selected: np.ndarray,
    errors: np.ndarray,
    is_prime: np.ndarray,
    q0: float,
    q_growth: float,
    orientation_rate: float,
    duration: float,
    tolerance: float,
    final_rotor: np.ndarray,
) -> None:
    mean_gap, gap_std, gap_cv = gap_variation(crossings)
    selected_prime = is_prime[selected] if selected.size else np.array([], dtype=bool)

    print()
    print("Ordered Clifford internal-momentum experiment")
    print("=" * 70)
    print("Visible vector momentum : 0 (opposing channels are exactly balanced)")
    print(f"Initial internal Q      : {q0:g}")
    print(f"Q growth               : {q_growth:g} per x")
    print(f"Final internal Q        : {q0 + q_growth * duration:g}")
    print(f"J-K orientation rate   : {orientation_rate:g}")
    print(f"Observed I crossings   : {len(crossings)}")
    print(f"Mean crossing gap      : {mean_gap:.6f}")
    print(f"Gap std. deviation     : {gap_std:.6f}")
    print(f"Gap variation (std/avg): {gap_cv:.6f}")
    print()
    print("Final rotor [scalar, I, J, K]")
    print("  " + np.array2string(final_rotor, precision=6, suppress_small=True))
    print()
    print("Near-integer I events (primality is checked only afterward)")
    print(f"Tolerance: {tolerance:g}")

    if selected.size:
        print(f"{'integer':>9} {'distance':>12} {'diagnostic':>14}")
        print("-" * 39)
        for integer, error in zip(selected, errors):
            label = "prime" if is_prime[integer] else "composite"
            print(f"{integer:9d} {error:12.6f} {label:>14}")
    else:
        print("  No I crossings landed within tolerance of an integer.")

    prime_hits = int(np.count_nonzero(selected_prime))
    print()
    print(f"Selected integers      : {len(selected)}")
    print(f"Selected primes        : {prime_hits}")
    if selected.size:
        print(f"Prime share            : {prime_hits / len(selected):.3f}")


def build_figure(
    coordinate: np.ndarray,
    forward: dict[str, np.ndarray],
    reverse: dict[str, np.ndarray],
    fixed: dict[str, np.ndarray],
    crossings: np.ndarray,
    selected: np.ndarray,
    is_prime: np.ndarray,
) -> plt.Figure:
    rotor = forward["rotor"]
    reverse_rotor = reverse["rotor"]
    fixed_rotor = fixed["rotor"]

    observed_amplitude = np.sqrt(rotor[:, 0] ** 2 + rotor[:, 1] ** 2)
    observed_phase = np.unwrap(np.arctan2(rotor[:, 1], rotor[:, 0]))
    observed_phase_rate = np.gradient(observed_phase, coordinate)

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.25, 1.0])
    ax_momentum = fig.add_subplot(grid[0, 0])
    ax_orientation = fig.add_subplot(grid[0, 1])
    ax_components = fig.add_subplot(grid[1, :])
    ax_projection = fig.add_subplot(grid[2, 0])
    ax_events = fig.add_subplot(grid[2, 1])

    # Six equal directional channels: the total rises while the vector sum is 0.
    channel = forward["q"] / 6.0
    ax_momentum.plot(coordinate, forward["q"], lw=2.0, label=r"internal $Q=\sum\rho$")
    ax_momentum.plot(
        coordinate,
        channel,
        lw=1.2,
        alpha=0.75,
        label="each of six balanced channels",
    )
    ax_momentum.axhline(0, color="black", lw=0.8, ls="--", label=r"visible $|\vec p|=0$")
    ax_momentum.set_title("Mass-like momentum grows while the object stays at rest")
    ax_momentum.set_xlabel("evolution parameter x")
    ax_momentum.set_ylabel("momentum-like magnitude")
    ax_momentum.legend(loc="upper left", fontsize=9)
    ax_momentum.grid(alpha=0.2)

    ax_orientation.plot(
        coordinate,
        np.cos(forward["chi"]),
        lw=1.5,
        label=r"J driver $\cos\chi$",
    )
    ax_orientation.plot(
        coordinate,
        np.sin(forward["chi"]),
        lw=1.5,
        label=r"K driver $\sin\chi$",
    )
    ax_orientation.axhline(0, color="black", lw=0.7, alpha=0.45)
    ax_orientation.set_title("Hidden generator rotates through the J-K plane")
    ax_orientation.set_xlabel("evolution parameter x")
    ax_orientation.set_ylabel("driver coefficient")
    ax_orientation.set_ylim(-1.15, 1.15)
    ax_orientation.legend(loc="upper right")
    ax_orientation.grid(alpha=0.2)

    ax_components.plot(coordinate, rotor[:, 0], lw=1.2, label="scalar")
    ax_components.plot(coordinate, rotor[:, 1], lw=1.8, label="I (observed output)")
    ax_components.plot(coordinate, rotor[:, 2], lw=1.0, alpha=0.75, label="J (hidden)")
    ax_components.plot(coordinate, rotor[:, 3], lw=1.0, alpha=0.75, label="K (hidden)")
    ax_components.axhline(0, color="black", lw=0.7, alpha=0.45)
    ax_components.set_title("Path-ordered rotor: J and K dynamics generate I")
    ax_components.set_xlabel("evolution parameter x")
    ax_components.set_ylabel("rotor component")
    ax_components.set_ylim(-1.15, 1.15)
    ax_components.legend(loc="upper right", ncols=4)
    ax_components.grid(alpha=0.2)

    ax_projection.plot(
        coordinate,
        rotor[:, 1],
        lw=1.8,
        label="I output: forward J-K orientation",
    )
    ax_projection.plot(
        coordinate,
        reverse_rotor[:, 1],
        lw=1.0,
        alpha=0.75,
        label="I output: reversed orientation",
    )
    ax_projection.plot(
        coordinate,
        fixed_rotor[:, 1],
        lw=1.0,
        ls="--",
        label="I output: fixed J plane",
    )
    ax_projection.axhline(0, color="black", lw=0.7, alpha=0.45)
    ax_projection.set_title("Order sensitivity: reversing J-K motion changes I")
    ax_projection.set_xlabel("evolution parameter x")
    ax_projection.set_ylabel("observed I coefficient")
    ax_projection.set_ylim(-1.15, 1.15)
    ax_projection.legend(loc="upper right", fontsize=8)
    ax_projection.grid(alpha=0.2)

    # Keep projection diagnostics available without adding another permanent panel.
    amplitude_min = float(np.min(observed_amplitude))
    rate_min = float(np.min(observed_phase_rate))
    rate_max = float(np.max(observed_phase_rate))
    ax_projection.text(
        0.01,
        0.03,
        f"min scalar-I amplitude={amplitude_min:.3f}   "
        f"observed phase-rate range=[{rate_min:.3f}, {rate_max:.3f}]",
        transform=ax_projection.transAxes,
        fontsize=9,
        va="bottom",
    )

    primes = np.flatnonzero(is_prime)
    ax_events.scatter(
        crossings,
        np.zeros_like(crossings),
        marker="|",
        s=150,
        color="#333333",
        label="I-channel zero crossings",
    )
    ax_events.scatter(
        primes,
        np.ones_like(primes),
        marker="|",
        s=150,
        color="#377eb8",
        label="primes (afterward only)",
    )
    if selected.size:
        prime_selected = is_prime[selected]
        ax_events.scatter(
            selected[~prime_selected],
            np.full(np.count_nonzero(~prime_selected), 0.45),
            marker="x",
            s=65,
            color="#d95f02",
            label="near-integer composite",
            zorder=4,
        )
        ax_events.scatter(
            selected[prime_selected],
            np.full(np.count_nonzero(prime_selected), 0.45),
            marker="o",
            s=65,
            facecolors="none",
            edgecolors="#1b9e77",
            linewidths=1.8,
            label="near-integer prime",
            zorder=5,
        )
    ax_events.set_xlim(coordinate[0], coordinate[-1])
    ax_events.set_ylim(-0.3, 1.3)
    ax_events.set_yticks([0, 0.45, 1])
    ax_events.set_yticklabels(["I events", "integer hits", "primes"])
    ax_events.set_xlabel("evolution parameter x")
    ax_events.set_title("Observed grouping, with primes kept out of the dynamics")
    ax_events.legend(loc="upper right", fontsize=8)
    ax_events.grid(axis="x", alpha=0.15)

    fig.suptitle(
        "Clifford Momentum Sandbox — Hidden J/K Motion Producing an I Projection",
        fontsize=16,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate balanced internal momentum and ordered J/K Clifford rotations."
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--q0", type=float, default=DEFAULT_Q0)
    parser.add_argument("--momentum-growth", type=float, default=DEFAULT_Q_GROWTH)
    parser.add_argument(
        "--orientation-rate",
        type=float,
        default=DEFAULT_ORIENTATION_RATE,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_INTEGER_TOLERANCE,
    )
    parser.add_argument("--save", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "figure.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.duration < 2:
        parser.error("--duration must be at least 2")
    if args.points < 1_001:
        parser.error("--points must be at least 1001")
    if args.q0 <= 0:
        parser.error("--q0 must be positive")
    if args.momentum_growth < 0:
        parser.error("--momentum-growth must be non-negative")
    if args.tolerance < 0 or args.tolerance > 0.5:
        parser.error("--tolerance must be between 0 and 0.5")
    return args


def run(args) -> plt.Figure:
    coordinate = np.linspace(0.0, args.duration, args.points)

    forward = simulate_rotor(
        coordinate,
        q0=args.q0,
        q_growth=args.momentum_growth,
        orientation_rate=args.orientation_rate,
    )
    reverse = simulate_rotor(
        coordinate,
        q0=args.q0,
        q_growth=args.momentum_growth,
        orientation_rate=-args.orientation_rate,
    )
    fixed = simulate_rotor(
        coordinate,
        q0=args.q0,
        q_growth=args.momentum_growth,
        orientation_rate=0.0,
    )

    crossings = find_zero_crossings(coordinate, forward["rotor"][:, 1])
    selected, errors = nearest_integer_events(
        crossings,
        duration=args.duration,
        tolerance=args.tolerance,
    )
    is_prime = prime_mask(int(np.floor(args.duration)))

    print_summary(
        crossings=crossings,
        selected=selected,
        errors=errors,
        is_prime=is_prime,
        q0=args.q0,
        q_growth=args.momentum_growth,
        orientation_rate=args.orientation_rate,
        duration=args.duration,
        tolerance=args.tolerance,
        final_rotor=forward["rotor"][-1],
    )

    figure = build_figure(
        coordinate=coordinate,
        forward=forward,
        reverse=reverse,
        fixed=fixed,
        crossings=crossings,
        selected=selected,
        is_prime=is_prime,
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
