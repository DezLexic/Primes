"""PowerShell entry point for the small normalized prime-frame experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import mpmath
import numpy as np

# Allow direct script execution and `python -m`, reusing the sibling model.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.prime_frame_normalized.model import (
    FRAMES, COMPARISON_FRAMES, HISTORY_FRAMES, first_zeta_zeros,
    normalized_frame, diagnostic_rows, verify_invariants,
)
from experiments.prime_frame_normalized.plotting import save_plots


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Normalize both prime-frame axes by the frame gap 1/p.")
    parser.add_argument("--zeros", type=int, default=10, help="positive-height samples (default: 10)")
    parser.add_argument("--k-max", type=int, default=5, help="comb indices -K..K (default: 5)")
    parser.add_argument("--dps", type=int, default=30, help="mpmath precision; saved values use float64")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.zeros < 1 or args.k_max < 0 or args.dps < 15:
        parser.error("--zeros must be >=1; --k-max >=0; --dps >=15")
    print(f"Computing {args.zeros} zeta-zero samples; normalizing {len(FRAMES)} frames...", flush=True)
    zeros = first_zeta_zeros(args.zeros, args.dps)
    test_point = complex(0.4, zeros[0].imag)
    states = [normalized_frame(p, zeros, args.k_max) for p in FRAMES]
    rows = list(diagnostic_rows(states, zeros, test_point))
    max_error = verify_invariants(rows)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "invariants.csv", rows)
    tracked = [r for r in rows if r["source_type"] == "inherited" and r["sign"] == 1 and r["zero_index"] <= 5]
    write_csv(output / "tracked_zeros.csv", tracked)
    write_csv(output / "zeta_zeros.csv", [dict(zero_index=n, beta=z.real, gamma=z.imag) for n, z in enumerate(zeros, 1)])
    names = save_plots(states, test_point, output)
    parameters = dict(created_utc=datetime.now(timezone.utc).isoformat(), frames=FRAMES,
                      comparison_frames=COMPARISON_FRAMES, history_frames=HISTORY_FRAMES,
                      zeros=args.zeros, k_max=args.k_max, dps=args.dps,
                      track_zeros=min(5, args.zeros), artificial_beta=test_point.real,
                      artificial_gamma=test_point.imag, include_conjugates=True,
                      zero_source="mpmath.zetazero; critical-line samples, not an RH proof",
                      output_precision="float64 / complex128", max_absolute_residual=max_error,
                      tolerance_rtol=2e-14, tolerance_atol=2e-14,
                      numpy=np.__version__, matplotlib=matplotlib.__version__, mpmath=mpmath.__version__)
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join([
        "Normalized prime frames: (X,Y)=p*(Re(s_p),Im(s_p))=2*(Re(z),Im(z)).",
        f"Frames: {FRAMES}; {args.zeros} positive-height zeros (+ conjugates); comb k=-{args.k_max}..{args.k_max}.",
        "Raw frames shrink with 1/p; normalized inherited geometry remains fixed.",
        "For successive prime frames, the only new comb is the previous frame prime's factor-zero comb.",
        "Inherited spectrum is collective; individual zeta zeros are not assigned to individual primes.",
        "Primes label factors 1-q^(-z); factor zeros lie on X=0. Primes themselves are not plotted as points.",
        "At p=2 there are no passed-prime combs. Each factor's coincident origin is retained.",
        "Collapse is expected from the coordinate map, not evidence for RH. No physical interpretation is made.",
        "The artificial beta=0.4 point stays at X=0.8, offset -0.2 in units of the frame gap.",
        "Normalized p=11 matches normalized p=2; it is twice raw p=2 (which equals the common coordinate).",
        "Plots use independent x/y scales, shared limits within each coordinate system, and no jitter.",
        f"Verified {len(rows)} diagnostic rows; max absolute coordinate residual: {max_error:.3e}.",
        "First five zeros (or all requested if fewer), positive heights:",
        "  p   n       raw_real       raw_imag    normalized_X    normalized_Y",
        *[f" {r['frame_prime']:2}  {r['zero_index']:2}  {r['raw_real']:13.9f}  {r['raw_imag']:13.9f}"
          f"  {r['normalized_X']:14.9f}  {r['normalized_Y']:14.9f}" for r in tracked],
        f"Outputs: {output}",
    ])
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    gallery = ["# Normalized prime-frame results", "", "[Settings](parameters.json) | [Summary and zero tracking](summary.txt)", "",
               "[Invariant checks](invariants.csv) | [Tracked zeros](tracked_zeros.csv) | [Source zeros](zeta_zeros.csv)", "",
               "The collapse follows from the coordinate transformation; it supplies no new evidence for RH.", ""]
    gallery += [f"![{name.removesuffix('.png').replace('_', ' ')}]({name})\n" for name in names]
    (output / "README.md").write_text("\n".join(gallery), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
