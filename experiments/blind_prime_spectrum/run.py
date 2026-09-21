"""Manual blind generation, freeze, then optional after-the-fact evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.blind_prime_spectrum.blind_model import Rules, SMOKE_CUTOFFS, cutoff_schedule
from experiments.blind_prime_spectrum.generate import generate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "normal", "deep"), default="normal")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blind-only", action="store_true")
    parser.add_argument("--cutoffs", type=int, nargs="+")
    parser.add_argument("--max-cutoff", type=int)
    parser.add_argument("--cutoff-count", type=int)
    parser.add_argument("--tau-max", type=float)
    parser.add_argument("--points", type=int)
    parser.add_argument("--control-replicates", type=int)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--prominence", type=float, default=0.1)
    parser.add_argument("--track-max-jump", type=float, default=0.5)
    parser.add_argument("--track-max-gap", type=int, default=1)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--min-observations", type=int, default=4)
    parser.add_argument("--min-tail-count", type=int, default=3)
    parser.add_argument("--persistence", type=float, default=0.75)
    parser.add_argument("--max-drift", type=float, default=0.5)
    parser.add_argument("--max-slope", type=float, default=0.25)
    parser.add_argument("--agreement", type=float, default=0.35)
    parser.add_argument("--kappa", type=float, help="enable optional blind internal-momentum CSV")
    parser.add_argument("--evaluation-tolerance", type=float, default=0.5)
    parser.add_argument("--dps", type=int, default=30)
    args = parser.parse_args(argv)
    if args.cutoffs is not None and (args.max_cutoff is not None or args.cutoff_count is not None):
        parser.error("--cutoffs cannot be combined with --max-cutoff or --cutoff-count")
    defaults = dict(smoke=(211, 10, 35., 1401, 2), normal=(10007, 60, 60., 6001, 3), deep=(50021, 80, 100., 16001, 5))[args.mode]
    maximum, count, tau, points, replicates = defaults
    for name, default in (("tau_max", tau), ("points", points), ("control_replicates", replicates)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.cutoffs is None:
        args.cutoffs = SMOKE_CUTOFFS if args.mode == "smoke" and args.max_cutoff is None and args.cutoff_count is None else cutoff_schedule(
            args.max_cutoff if args.max_cutoff is not None else maximum, args.cutoff_count if args.cutoff_count is not None else count)
    args.output = args.output or Path(__file__).resolve().parent / "results" / args.mode
    return args


def main(argv=None):
    args = parse_args(argv)
    rules = Rules(tail_fraction=args.tail_fraction, min_observations=args.min_observations,
                  min_tail_count=args.min_tail_count, persistence=args.persistence,
                  max_drift=args.max_drift, max_slope=args.max_slope, agreement=args.agreement)
    generate(args.output, tuple(args.cutoffs), rules, tau_max=args.tau_max, points=args.points,
             prominence=args.prominence, seed=args.seed, control_replicates=args.control_replicates,
             max_jump=args.track_max_jump, max_gap=args.track_max_gap, ambiguity_margin=args.ambiguity_margin, kappa=args.kappa)
    if not args.blind_only:
        # The evaluation module itself is not imported until freeze completes.
        from experiments.blind_prime_spectrum.evaluate import evaluate_frozen
        print("EVALUATION: verifying frozen artifacts before loading reference zeros", flush=True)
        evaluate_frozen(args.output, tolerance=args.evaluation_tolerance, dps=args.dps)
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
