"""Manual PowerShell entry point; a small finite experiment, not a sweep."""

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

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.prime_frame_geometry.model import first_zeta_zeros, passed_prime_list
from experiments.prime_factor_phasors.model import (
    PRIMES, HISTORY_FRAMES, TRACK_FRAMES, analytic_rows, sampling_grid,
)
from experiments.prime_factor_phasors.plotting import save_plots

INTERPRETATION = """Each factor Fq=1-q^(-z) is the sum of signed phasors 1 and -q^(-z).
Writing r=q^(-sigma), |Fq|^2=(1-r)^2+4r sin^2(tau ln(q)/2).
Both terms are nonnegative: a zero requires r=1 AND tau ln(q)=2 pi k.
Since q>1, r=1 forces sigma=0, including when considering sigma<0.
Thus z=2 pi i k/ln(q), s_p=4 pi i k/(p ln(q)), (X,Y)=(0,4 pi k/ln(q)).
At sigma=1/2 each minimum is (1-1/sqrt(q))^2 > 0.
For every finite frame, |Rp|^2=product |Fq|^2, not |sum Fq|^2.
The sigma=1/2 product is exactly W_p from prime_frame; its D_p is W_p/B_p.
Products explain that finite history modulation algebraically, without explaining zeta itself.
Each factor zero is simple. The origin has multiplicity equal to the number of passed primes.
Distinct prime combs share only the origin: a nonzero coincidence would imply q^a=r^b
for distinct primes and positive integers a,b, contradicting unique factorization.
At sigma=1/2 the finite product has a positive global minimum, attained at tau=0,
equal to product (1-1/sqrt(q))^2. Small plotted values are not exact zeros.
Inherited critical-line samples map to X=1; general beta maps to X=2 beta.
These factor zeros are separate from inherited zeros, not inherited zeros moved to X=0.
No identical two-phasor explanation for inherited zeta zeros is established.
No claims about double-slit physics, quantum measurement, wave-particle duality,
physical prime waves, an RH proof, or a mass gap are made.
Float64 evaluation leaves tiny residuals at analytic zeros; markers use analytic locations.
All products are finite. No infinite Euler product is evaluated on the critical line.
"""


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Two-phasor prime factors and finite history products.")
    parser.add_argument("--tau-max", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=2001, help="odd uniform grid size; analytic zeros are added")
    parser.add_argument("--zeros", type=int, default=3, help="critical-line samples for the geometry figure only")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if not np.isfinite(args.tau_max) or args.tau_max <= 0 or args.points < 101 or args.points % 2 != 1 or args.zeros < 1:
        parser.error("--tau-max must be finite and >0; --points must be odd and >=101; --zeros must be >=1")
    primes = passed_prime_list(max(HISTORY_FRAMES))
    tau = sampling_grid(args.tau_max, args.points, primes)
    print(f"Small finite experiment: {len(HISTORY_FRAMES)} frames, {len(tau)} samples, {args.zeros} zeta zeros.", flush=True)
    zeros = first_zeta_zeros(args.zeros, 30)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "analytic_zeros.csv", list(analytic_rows(HISTORY_FRAMES, args.tau_max)))
    write_csv(output / "comb_spacings.csv", [dict(
        q=int(q), phase_rate=np.log(q), delta_tau=2*np.pi/np.log(q),
        delta_normalized_Y=4*np.pi/np.log(q), min_power_sigma_half=(1-1/np.sqrt(q))**2,
    ) for q in primes])
    write_csv(output / "zeta_zeros.csv", [dict(index=i, beta=z.real, gamma=z.imag) for i, z in enumerate(zeros, 1)])
    names = save_plots(output, tau, args.tau_max, zeros)
    parameters = dict(created_utc=datetime.now(timezone.utc).isoformat(), tau_max=args.tau_max,
                      uniform_points=args.points, actual_points=len(tau), analytic_zeros_inserted=True,
                      single_primes=PRIMES, history_frames=HISTORY_FRAMES, track_frames=TRACK_FRAMES,
                      track_k_max=3, selected_frame=11, sigmas=[0, 0.5], circle_sigmas=[0, 0.5, -0.5],
                      zeros=args.zeros, dps=30, zero_source="mpmath.zetazero (critical-line samples, not RH proof)",
                      output_precision="float64 / complex128", numpy=np.__version__,
                      matplotlib=matplotlib.__version__, mpmath=mpmath.__version__, plots=names)
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2)+"\n", encoding="utf-8")
    (output / "summary.txt").write_text(INTERPRETATION+f"\nOutput: {output}\n", encoding="utf-8")
    gallery = ["# Prime-factor phasor results", "", "[Settings](parameters.json) | [Interpretation](summary.txt)", "",
               "[Analytic zero coordinates](analytic_zeros.csv) | [Comb spacings](comb_spacings.csv) | [Sampled zeta zeros](zeta_zeros.csv)", "",
               "Exact finite-factor mathematics; no physical model and no claim about the mechanism of inherited zeta zeros.", ""]
    gallery += [f"![{name.removesuffix('.png').replace('_', ' ')}]({name})\n" for name in names]
    (output / "README.md").write_text("\n".join(gallery), encoding="utf-8")
    print(f"Saved {len(names)} PNGs, 3 CSVs, settings, summary, and gallery to {output}")


if __name__ == "__main__":
    main()
