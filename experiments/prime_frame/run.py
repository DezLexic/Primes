"""Run from the repository root: .venv/Scripts/python experiments/prime_frame/run.py."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

if __package__:
    from .model import numerical_chi, primes_below, run_frame_sweep
    from .plotting import save_plots
else:
    from model import numerical_chi, primes_below, run_frame_sweep
    from plotting import save_plots


def main():
    parser = argparse.ArgumentParser(description="Finite prime-history depletion and local widths; no mass model.")
    parser.add_argument("--max-prime", type=int, default=1000)
    parser.add_argument("--tau-max", type=float, default=60)
    parser.add_argument("--points", type=int, default=4001)
    parser.add_argument("--frames", nargs="+", type=int, help="representative prime frames for profile plots")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.max_prime < 3:
        parser.error("--max-prime must be >= 3")
    if not np.isfinite(args.tau_max) or args.tau_max <= 0:
        parser.error("--tau-max must be finite and positive")
    if args.points < 3 or args.points % 2 == 0:
        parser.error("--points must be odd and >= 3 so the symmetric grid includes zero")
    primes = primes_below(args.max_prime + 1)
    representatives = args.frames or [p for p in (3, 5, 7, 11, 13, 29, 101, 331) if p <= args.max_prime]
    if any(p not in primes for p in representatives):
        parser.error("--frames must contain primes <= --max-prime")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tau = np.linspace(-args.tau_max, args.tau_max, args.points)
    rows, profiles = run_frame_sweep(args.max_prime, tau)
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Persist profiles too: plots can be revisited without recomputing a sweep.
    np.savez_compressed(output / "profiles.npz", tau=tau, frames=primes, log_D=profiles)
    plot_names = save_plots(rows, tau, profiles, representatives, output)
    checks = []
    for p in representatives:
        row = rows[int(np.searchsorted(primes, p))]
        numeric = numerical_chi(p)
        checks.append(f"p={p}: W(0)={row['W0']:.9e}, chi={row['chi']:.9g}, numerical chi={numeric:.9g}")
    last = rows[-1]
    tail = rows[max(1, len(rows) * 3 // 4):]
    ratios = [row["chi_pnt_ratio"] for row in tail]
    summary = "\n".join([
        "Finite prime-history experiment", f"Frames: {len(rows)} (2 through {last['frame_prime']})",
        f"tau grid: [{-args.tau_max:g}, {args.tau_max:g}], {args.points} points",
        *checks,
        f"Last frame: tau_quad={last['tau_quad']:.9g}; t_quad={last['t_quad']:.9g}",
        f"Last chi/(2 sqrt(p) ln(p))={last['chi_pnt_ratio']:.9g}",
        f"Last scaled width={last['scaled_width']:.9g}; reference 1/sqrt(2)={1/np.sqrt(2):.9g}",
        f"Final quarter of sampled frames: PNT ratio range [{min(ratios):.6g}, {max(ratios):.6g}]",
        "The PNT reference is a leading-term diagnostic; a finite sweep does not establish convergence.",
        "p=2 is the empty product: W=B=D=1, chi=0, widths=infinity (omitted from width plots).",
        "Finite factors are positive: central suppression is not an interval of zero spectrum or a mass gap.",
        "The 1/p lines are imposed by z=(p/2)s_p; primes enter through factors, not as critical-line points.",
        "No vacuum-energy baseline or mass is introduced. Raw powers can underflow; use log columns.",
        f"Outputs: {output}",
    ])
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    max_prime=args.max_prime, tau_max=args.tau_max, points=args.points,
                    representative_frames=representatives, numpy=np.__version__,
                    matplotlib=matplotlib.__version__)
    (output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    index = ["# Saved prime-frame results", "", "Run settings: [parameters.json](parameters.json).",
             "", "[Summary](summary.txt) | [Frame data](frames.csv) | [Profile arrays](profiles.npz)", ""]
    index.extend(f"![{name.removesuffix('.png').replace('_', ' ')}]({name})\n" for name in plot_names)
    (output / "README.md").write_text("\n".join(index), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
