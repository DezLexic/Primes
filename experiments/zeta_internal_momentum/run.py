"""Manual zeta internal-momentum toy experiment; no prime sweep is performed."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib
import mpmath
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.zeta_internal_momentum.model import (
    directional_rows, directional_shapes, dirac_hamiltonian, internal_energy,
    internal_modes, load_prime_features, spectral_operator, standard_kg_energy,
)
from experiments.zeta_internal_momentum.plotting import save_plots

HERE = Path(__file__).resolve().parent
GUARDRAILS = """Mass is interpreted as invariant internal momentum P_int=m*c in this toy model.
External p_ext is separate; at external rest E=c*P_int can remain nonzero.
gamma_n is dimensionless; kappa supplies momentum units. Zeta quantization is imposed.
Directional distributions alone allow continuous invariant magnitudes.
KG/Dirac equivalence is expected algebraically, not new dynamics or a derivation of spin.
The diagonal operator is defined from the zeros; it is not a Hilbert-Pólya operator.
Prime-product peaks are optional comparisons, not the defined spectrum or a physical cause.
No RH, quantum-gravity, Yang-Mills, particle-generation, or proven physical mass-gap claim follows.
"""


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeros", type=int, default=8)
    parser.add_argument("--dps", type=int, default=30)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--hbar", type=float, default=1.0, help="sets KG coefficient P_int²/hbar²; energy dispersion is independent of hbar")
    parser.add_argument("--p-max", type=float, default=50.0, help="external momentum interval [-p-max, p-max]")
    parser.add_argument("--points", type=int, default=1001, help="odd sample count so external rest is included exactly")
    parser.add_argument("--directional-mode", type=int, default=1)
    parser.add_argument("--prime-features", type=Path, help="nearest_features.csv; otherwise try existing normal, then smoke output")
    parser.add_argument("--no-prime-features", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "normal")
    args = parser.parse_args(argv)
    if args.zeros < 1 or args.dps < 15 or args.points < 3 or args.points % 2 != 1:
        parser.error("zeros>=1, dps>=15, and odd points>=3 required")
    if not 1 <= args.directional_mode <= args.zeros:
        parser.error("directional-mode must be among the generated zeros")
    for name in ("kappa", "c", "hbar", "p_max"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.no_prime_features and args.prime_features:
        parser.error("choose --prime-features or --no-prime-features")
    return args


def run(args):
    gamma, momenta = internal_modes(args.zeros, args.kappa, args.dps)
    p_ext = np.linspace(-args.p_max, args.p_max, args.points)
    p_ext[args.points//2] = 0.0
    energies = internal_energy(p_ext[None, :], momenta[:, None], args.c)
    standard = standard_kg_energy(p_ext[None, :], momenta[:, None]/args.c, args.c)
    kg_errors = standard - energies
    modes = [dict(zero_index=0, gamma=0.0, kappa=args.kappa, P_int=0.0,
                  P_int_squared=0.0, gap_from_previous="", gap_from_zero=0.0, reference_only=True)]
    for i, (g, P) in enumerate(zip(gamma, momenta), 1):
        modes.append(dict(zero_index=i, gamma=float(g), kappa=args.kappa, P_int=float(P),
                          P_int_squared=float(P**2), gap_from_previous=float(P-modes[-1]["P_int"]),
                          gap_from_zero=float(P), reference_only=False))
    dispersion = [dict(zero_index=i+1, gamma=float(gamma[i]), p_ext=float(p), P_int=float(P),
                       E=float(energies[i, j]), standard_mass_equivalent=float(P/args.c),
                       equivalence_error=float(kg_errors[i, j]), standard_KG_E=float(standard[i, j]))
                  for i, P in enumerate(momenta) for j, p in enumerate(p_ext)]
    target = float(momenta[args.directional_mode-1])
    states = [state.scaled(target) for state in directional_shapes()]
    directions = directional_rows(states, args.directional_mode, target)
    dirac_rows = []
    for index in np.unique(np.linspace(0, args.zeros-1, min(3, args.zeros), dtype=int)):
        P = momenta[index]
        for p in np.linspace(-args.p_max, args.p_max, 5):
            expected = float(internal_energy(p, P, args.c))
            eigenvalues = np.linalg.eigvalsh(dirac_hamiltonian(p, P, args.c))
            error = float(np.max(abs(eigenvalues - np.array([-expected, -expected, expected, expected]))))
            dirac_rows.append(dict(zero_index=int(index+1), p_ext=float(p), expected_positive_E=expected,
                                   expected_negative_E=-expected, computed_eigenvalues=eigenvalues.tolist(), max_error=error))
    operator = spectral_operator(momenta)
    squared_operator = operator @ operator
    operator_error = float(np.max(abs(np.linalg.eigvalsh(squared_operator) - momenta**2)))
    mapping, source = [], None
    source_status = "Optional prime-feature mapping disabled."
    if not args.no_prime_features:
        candidates = [HERE.parent / "prime_zero_features" / "results" / folder / "nearest_features.csv"
                      for folder in ("normal", "smoke")]
        source = args.prime_features or next((p for p in candidates if p.is_file()), candidates[0])
        source = source.resolve()
        mapping, source_status = load_prime_features(source, gamma, args.kappa)

    # Fail instead of publishing diagnostics as successful if an identity breaks.
    np.testing.assert_allclose(standard, energies, rtol=1e-12, atol=1e-12)
    for row in directions:
        np.testing.assert_allclose(row["invariant_squared"], target**2, rtol=1e-12, atol=1e-12)
    for row in dirac_rows:
        if row["max_error"] > 1e-12 * max(1., row["expected_positive_E"]):
            raise ArithmeticError("Dirac dispersion check failed")
    np.testing.assert_allclose(np.linalg.eigvalsh(squared_operator), momenta**2, rtol=1e-12)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "zeta_internal_modes.csv", modes)
    write_csv(output / "dispersion.csv", dispersion)
    write_csv(output / "directional_states.csv", directions)
    write_csv(output / "dirac_eigenvalues.csv", [dict(row, computed_eigenvalues=json.dumps(row["computed_eigenvalues"])) for row in dirac_rows])
    write_csv(output / "internal_operator.csv", [dict(zero_index=i+1, eigenvalue=float(P),
              squared_eigenvalue=float(squared_operator[i, i]), kg_coefficient=float(P**2/args.hbar**2))
              for i, P in enumerate(momenta)])
    np.savez(output / "internal_operator.npz", P_int=operator, P_int_squared=squared_operator)
    # These two known optional artifacts must not survive a rerun without mapping.
    for name in ("prime_peak_mapping.csv", "prime_peak_to_internal_mode.png"):
        path = output / name
        if path.is_file():
            path.unlink()
    if mapping:
        write_csv(output / "prime_peak_mapping.csv", mapping)
    names = save_plots(output, gamma, momenta, p_ext, energies, kg_errors, states,
                       directions, dirac_rows, mapping, args.kappa, args.c)
    parameters = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameters.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                      mpmath=mpmath.__version__, resolved_prime_features=str(source) if source else None,
                      prime_features_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source and source.is_file() else None,
                      prime_feature_status=source_status)
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    summary = (f"Zeta internal momentum: {args.zeros} imposed modes; c={args.c:g}, hbar={args.hbar:g}, kappa={args.kappa:g}\n"
               f"Toy internal momentum gap: {momenta[0]:.15g}\n"
               f"Lowest external-rest energy: {args.c*momenta[0]:.15g}\n"
               f"Max KG energy difference: {np.max(abs(kg_errors)):.6e}\n"
               f"Max Dirac eigenvalue error: {max(r['max_error'] for r in dirac_rows):.6e}\n"
               f"Max directional P_int error: {max(abs(r['invariant_error']) for r in directions):.6e}\n"
               f"Max directional squared-invariant error: {max(abs(r['squared_invariant_error']) for r in directions):.6e}\n"
               f"Max operator squared-spectrum error: {operator_error:.6e}\n"
               f"{source_status}\n\n{GUARDRAILS}")
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    gallery = "# Zeta internal momentum results\n\n[Experiment interpretation and commands](../../README.md)\n\n"
    # Relative experiment link also works with custom nested output directories.
    gallery = gallery.replace("../../README.md", Path(os.path.relpath(HERE / "README.md", output)).as_posix())
    gallery += "```text\n" + summary + "```\n\n"
    gallery += "\n\n".join(f"![{name.removesuffix('.png').replace('_', ' ')}]({name})" for name in names) + "\n"
    (output / "README.md").write_text(gallery, encoding="utf-8")
    print(summary)
    print(f"Saved results: {output}")
    return output


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
