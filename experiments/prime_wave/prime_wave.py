import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

X_MIN = 2
X_MAX = 1000
NUM_POINTS = 200_000


# ============================================================
# PRIME SIEVE
# ============================================================

def prime_sieve(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False

    for n in range(2, int(np.sqrt(limit)) + 1):
        if sieve[n]:
            sieve[n * n : limit + 1 : n] = False

    return sieve


def run(args):
    sieve = prime_sieve(X_MAX)
    primes = np.nonzero(sieve)[0]


    # ============================================================
    # CONTINUOUS NUMBER LINE
    # ============================================================

    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)


    # ============================================================
    # MOMENTUM / TIME-DILATION MODEL
    #
    # gamma(x) = log(x) / log(2)
    #
    # Therefore:
    #
    # proper time rate = 1/gamma
    #
    # and:
    #
    # phase rate = pi/log(x)
    # ============================================================

    gamma = np.log(x) / np.log(2)

    q = np.sqrt(gamma**2 - 1)

    proper_time_rate = 1 / gamma

    phase_rate = np.pi / np.log(x)


    # ============================================================
    # INTEGRATE PHASE
    # ============================================================

    theta = np.zeros_like(x)

    dx = np.diff(x)

    theta[1:] = np.cumsum(
        0.5
        * (phase_rate[:-1] + phase_rate[1:])
        * dx
    )


    wave = np.sin(theta)


    # ============================================================
    # THE CLOCK'S IMPLIED PRIME COUNT
    #
    # Every pi radians = one crossing.
    #
    # Starting at x=2 counts the prime 2 itself.
    # ============================================================

    smooth_count = 1 + theta / np.pi


    # ============================================================
    # ACTUAL PRIME COUNT pi(x)
    # ============================================================

    integer_x = np.arange(X_MIN, X_MAX + 1)

    prime_count = np.cumsum(
        sieve[X_MIN : X_MAX + 1]
    )


    # Interpolate smooth prediction at integer locations
    smooth_at_integers = np.interp(
        integer_x,
        x,
        smooth_count
    )

    error = prime_count - smooth_at_integers


    # ============================================================
    # ZERO CROSSINGS
    # ============================================================

    max_n = int(theta[-1] / np.pi)

    target_phases = np.arange(max_n + 1) * np.pi

    predicted_crossings = np.interp(
        target_phases,
        theta,
        x
    )


    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print()
    print("Prime count comparison")
    print("=" * 55)

    for test_x in [10, 25, 50, 100, 200, 500, 1000]:

        actual = prime_count[test_x - X_MIN]

        smooth = smooth_at_integers[test_x - X_MIN]

        print(
            f"x = {test_x:4d}   "
            f"actual pi(x) = {actual:4d}   "
            f"smooth clock = {smooth:8.3f}   "
            f"difference = {actual - smooth:8.3f}"
        )


    # ============================================================
    # PLOT
    # ============================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
        height_ratios=[2, 2, 1.5]
    )

    ax1, ax2, ax3 = axes


    # ============================================================
    # PANEL 1 — THE CLOCK
    # ============================================================

    ax1.plot(
        x,
        wave,
        linewidth=1
    )

    ax1.axhline(
        0,
        linewidth=0.8,
        alpha=0.5
    )

    ax1.set_ylabel("oscillator")

    ax1.set_title(
        "Prime-wave experiment — smooth relativistic clock"
    )

    ax1.grid(alpha=0.2)


    # ============================================================
    # PANEL 2 — PRIME COUNT
    # ============================================================

    ax2.step(
        integer_x,
        prime_count,
        where="post",
        label="Actual prime count π(x)",
        linewidth=1.3
    )

    ax2.plot(
        x,
        smooth_count,
        label="Smooth time-dilated clock",
        linewidth=1.5
    )

    ax2.set_ylabel("number of events")

    ax2.legend()

    ax2.grid(alpha=0.2)


    # ============================================================
    # PANEL 3 — WHAT THE SMOOTH MODEL MISSES
    # ============================================================

    ax3.plot(
        integer_x,
        error,
        linewidth=1.2
    )

    ax3.axhline(
        0,
        linewidth=0.8,
        alpha=0.5
    )

    ax3.set_xlabel("number line x")

    ax3.set_ylabel(
        "π(x) - clock"
    )

    ax3.set_title(
        "Residual structure the smooth clock does not explain"
    )

    ax3.grid(alpha=0.2)


    plt.tight_layout()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=160, bbox_inches="tight")
    print(f"\nSaved figure to: {args.save.resolve()}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare a smooth logarithmic clock with prime counts.")
    parser.add_argument("--save", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "figure.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    capture = io.StringIO()
    with redirect_stdout(capture):
        figure = run(args)
    summary = capture.getvalue()
    print(summary, end="")
    args.save.with_suffix(".txt").write_text(summary, encoding="utf-8")
    settings = dict(x_min=X_MIN, x_max=X_MAX, num_points=NUM_POINTS, **vars(args))
    args.save.with_suffix(".json").write_text(
        json.dumps(settings, default=str, indent=2) + "\n", encoding="utf-8"
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
