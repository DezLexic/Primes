"""Run from the repo root with .venv/Scripts/python experiments/prime_frame_geometry/run.py."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mpmath
import numpy as np

if __package__:
    from .model import first_zeta_zeros, frame_geometry, validate_frame
    from .plotting import save_plots
else:
    from model import first_zeta_zeros, frame_geometry, validate_frame
    from plotting import save_plots


def _csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Prime-frame zero geometry and horizontal coordinate separation.")
    parser.add_argument("--frames", nargs="+", type=int, default=[2, 3, 5, 7, 11],
                        help="strictly increasing prime frames (default: 2 3 5 7 11)")
    parser.add_argument("--zeros", type=int, default=20, help="positive-height zeta zeros (default: 20)")
    parser.add_argument("--track-zeros", type=int, default=5, help="first zeros to track, capped at --zeros")
    parser.add_argument("--k-max", type=int, default=8, help="comb indices -K,...,K (default: 8)")
    parser.add_argument("--dps", type=int, default=30, help="mpmath decimal precision; output uses float64")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    try:
        for p in args.frames:
            validate_frame(p)
    except ValueError as error:
        parser.error(str(error))
    if any(a >= b for a, b in zip(args.frames, args.frames[1:])):
        parser.error("--frames must be strictly increasing, without duplicates")
    if args.zeros < 1 or args.track_zeros < 1 or args.k_max < 0 or args.dps < 15:
        parser.error("--zeros and --track-zeros must be >=1; --k-max >=0; --dps >=15")
    track_count = min(args.track_zeros, args.zeros)
    print(f"Computing {args.zeros} positive-height zeta zeros at {args.dps} decimal digits...", flush=True)
    zeros = first_zeta_zeros(args.zeros, args.dps)
    states = [frame_geometry(p, zeros, args.k_max) for p in args.frames]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "zeta_zeros.csv", ["zero_index", "beta", "gamma"],
         ({"zero_index": n, "beta": z.real, "gamma": z.imag} for n, z in enumerate(zeros, 1)))
    _csv(output / "inherited_zeros.csv", ["frame_prime", "zero_index", "sign", "real", "imag", "p_times_real"],
         ({"frame_prime": state["frame_prime"], "zero_index": n, "sign": sign,
           "real": z.real, "imag": sign * z.imag, "p_times_real": state["frame_prime"] * z.real}
          for state in states for n, z in enumerate(state["inherited"], 1) for sign in (1, -1)))
    _csv(output / "passed_prime_combs.csv", ["frame_prime", "passed_prime", "k", "real", "imag"],
         ({"frame_prime": state["frame_prime"], "passed_prime": q, "k": k, "real": z.real, "imag": z.imag}
          for state in states for q, comb in state["combs"].items()
          for k, z in zip(range(-args.k_max, args.k_max + 1), comb)))
    _csv(output / "frame_gaps.csv", ["frame_prime", "gap", "passed_prime_count"],
         ({"frame_prime": s["frame_prime"], "gap": s["gap"], "passed_prime_count": len(s["combs"])} for s in states))
    plot_names = save_plots(states, track_count, output)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), frames=args.frames,
                    zeros=args.zeros, track_zeros=track_count, k_max=args.k_max, dps=args.dps,
                    zero_source="mpmath.zetazero; positive-height critical-line samples",
                    output_precision="float64 / complex128", include_conjugates=True,
                    numpy=np.__version__, matplotlib=matplotlib.__version__, mpmath=mpmath.__version__)
    (output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join([
        "Prime-frame geometry: z=(p/2)*s_p",
        f"Frames: {args.frames}; zeros: {args.zeros} (+ conjugates); comb k: {-args.k_max}..{args.k_max}",
        "Inherited-zero line: Re(s_p)=1/p for the sampled beta=1/2 zeros.",
        "Passed-prime factor zero combs: Re(s_p)=0. Horizontal separation: Delta_p=1/p.",
        "For a general rho=beta+i*gamma, p*Re(s_p)=2*beta is invariant.",
        "The shifted line follows from the coordinate map; this does not establish RH or a physical gap.",
        "Primes label factors, not points on the inherited-zero line.",
        "At p=2 the comb axis is a reference only: there are no passed-prime zeros.",
        "All combs include k=0; coincident points are retained. No horizontal jitter is used.",
        "Static plots use independent x/y scales; comparison panels share limits and include all requested points.",
        "First inherited zero (positive height):",
        *[f"  p={s['frame_prime']}: {s['inherited'][0].real:.10f} + {s['inherited'][0].imag:.10f}i" for s in states],
        f"Outputs: {output}",
    ])
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    gallery = ["# Prime-frame geometry results", "", "[Settings](parameters.json) | [Summary](summary.txt)", "",
               "[Source zeros](zeta_zeros.csv) | [Inherited coordinates](inherited_zeros.csv) | "
               "[Comb coordinates](passed_prime_combs.csv) | [Frame gaps](frame_gaps.csv)", "",
               "Critical-line samples; coordinate geometry does not establish RH or a physical gap.", ""]
    gallery.extend(f"![{name.removesuffix('.png').replace('_', ' ')}]({name})\n" for name in plot_names)
    (output / "README.md").write_text("\n".join(gallery), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
