"""Imposed zeta internal-momentum spectrum in ordinary KG / Dirac dynamics.

P_int = m*c is a parameter reinterpretation, not a derived quantization rule.
Directional quadrature weights are integrated nonnegative momentum contents;
delta beams are measures on the sphere, not smooth densities.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.prime_frame_geometry.model import first_zeta_zeros


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def internal_modes(count=8, kappa=1.0, dps=30):
    """The baseline is a reference only, deliberately excluded from modes."""
    kappa = positive(kappa, "kappa")
    gamma = first_zeta_zeros(count, dps).imag
    return gamma, kappa * gamma


def internal_energy(p_ext, P_int, c=1.0):
    """E = c*sqrt(p_ext**2 + P_int**2), with no extra directional B term."""
    c = positive(c, "c")
    p_ext, P_int = np.asarray(p_ext, dtype=float), np.asarray(P_int, dtype=float)
    if not np.all(np.isfinite(p_ext)) or not np.all(np.isfinite(P_int)) or np.any(P_int < 0):
        raise ValueError("finite p_ext and nonnegative finite P_int required")
    return c * np.hypot(p_ext, P_int)


def standard_kg_energy(p_ext, mass, c=1.0):
    """Independent evaluation of ordinary E^2 = p^2*c^2 + m^2*c^4."""
    c = positive(c, "c")
    p_ext, mass = np.asarray(p_ext, dtype=float), np.asarray(mass, dtype=float)
    if not np.all(np.isfinite(p_ext)) or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("finite p_ext and nonnegative finite mass required")
    return np.sqrt(p_ext**2 * c**2 + mass**2 * c**4)


def dirac_hamiltonian(p_ext, P_int, c=1.0):
    """Standard 4x4 Dirac representation, momentum oriented along x.

    alpha_x = [[0, sigma_x], [sigma_x, 0]], beta = diag(I, -I).
    The normal spinor structure is supplied by Dirac, not by zeta zeros.
    """
    internal_energy(p_ext, P_int, c)  # validate the same physical parameters
    if np.ndim(p_ext) or np.ndim(P_int):
        raise ValueError("Dirac matrix requires scalar momenta")
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    zero = np.zeros((2, 2), dtype=complex)
    alpha_x = np.block([[zero, sigma_x], [sigma_x, zero]])
    beta = np.diag([1, 1, -1, -1])
    return c * (p_ext * alpha_x + P_int * beta)


def spectral_operator(momenta):
    """Finite diagonal definition; this is NOT a Hilbert-Pólya construction."""
    momenta = np.asarray(momenta, dtype=float)
    if momenta.ndim != 1 or momenta.size == 0 or not np.all(np.isfinite(momenta)) or np.any(momenta <= 0):
        raise ValueError("need a nonempty vector of positive finite modes")
    return np.diag(momenta)


@dataclass(frozen=True)
class DirectionalState:
    name: str
    directions: np.ndarray
    weights: np.ndarray

    def scaled(self, scale):
        scale = positive(scale, "scale")
        return DirectionalState(self.name, self.directions.copy(), self.weights * scale)


def directional_moments(state):
    """Return A, B, A^2-|B|^2, P_int from integrated directional weights."""
    directions = np.asarray(state.directions, dtype=float)
    weights = np.asarray(state.weights, dtype=float)
    if weights.ndim != 1 or directions.shape != (weights.size, 3) or weights.size == 0:
        raise ValueError("need N weights and N three-dimensional directions")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("finite unit directions and nonnegative finite weights required")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("directions must be unit vectors")
    A = float(weights.sum())
    B = weights @ directions
    B_mag = float(np.linalg.norm(B))
    # Factored expression reduces cancellation near the null boundary.
    invariant_squared = (A - B_mag) * (A + B_mag)
    if invariant_squared < -64 * np.finfo(float).eps * A**2:
        raise ValueError("negative invariant beyond roundoff")
    invariant_squared = max(0.0, invariant_squared)
    return A, B, invariant_squared, float(np.sqrt(invariant_squared))


def unit_invariant(state):
    P_int = directional_moments(state)[3]
    if P_int <= 0:
        raise ValueError("a zero/null directional state cannot be normalized to unit invariant")
    return state.scaled(1 / P_int)


def directional_shapes():
    """Four f with A_f^2-|B_f|^2=1, using exact low-order sphere quadrature.

    Smooth examples use Gauss-Legendre in z=cos(theta) and uniform azimuth.
    For rho=1+a*n_x, A=4*pi and B=(4*pi*a/3,0,0) before normalization.
    Beams use integrated weights directly; no solid-angle factor is added.
    """
    z, wz = np.polynomial.legendre.leggauss(8)
    phi = np.linspace(0, 2*np.pi, 16, endpoint=False)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    radius = np.sqrt(1-zz**2)
    directions = np.column_stack((radius.ravel()*np.cos(pp.ravel()),
                                  radius.ravel()*np.sin(pp.ravel()), zz.ravel()))
    domega = np.repeat(wz, len(phi)) * (2*np.pi/len(phi))
    axes = np.concatenate((np.eye(3), -np.eye(3)))
    raw = [
        DirectionalState("isotropic sphere", directions, domega),
        DirectionalState("balanced two beams", np.array([[1., 0, 0], [-1., 0, 0]]), np.ones(2)),
        DirectionalState("balanced six beams", axes, np.array([1., 2, 3, 1, 2, 3])),
        DirectionalState("smooth dipole (a=0.6)", directions, domega * (1 + 0.6*directions[:, 0])),
    ]
    return [unit_invariant(state) for state in raw]


def directional_rows(states, zero_index, target):
    rows = []
    for state in states:
        A, B, square, P_int = directional_moments(state)
        rows.append(dict(zero_index=zero_index, state_name=state.name, A=A,
                         B_x=float(B[0]), B_y=float(B[1]), B_z=float(B[2]),
                         B_magnitude=float(np.linalg.norm(B)), P_int=P_int,
                         target_P_int=target, invariant_error=P_int-target,
                         relative_invariant_error=(P_int-target)/target,
                         invariant_squared=square, squared_invariant_error=square-target**2))
    return rows


def load_prime_features(path, gammas, kappa=1.0):
    """Map only real, replicate-0 peaks; independently generated gamma is truth.

    Missing files are optional. Invalid schemas/data produce a clear ValueError.
    Preserve missing peaks as blanks and retain boundary/track metadata.
    """
    kappa = positive(kappa, "kappa")
    path = Path(path)
    if not path.exists():
        return [], f"Optional prime-feature file absent: {path}"
    rows = []
    required = {"dataset", "replicate", "prime_cutoff", "zero_index", "gamma", "in_window", "nearest_peak_tau"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns {sorted(required-set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            if row["dataset"] != "real" or row["replicate"] != "0":
                continue
            try:
                n, cutoff = int(row["zero_index"]), int(row["prime_cutoff"])
                if n < 1 or cutoff < 2:
                    raise ValueError("invalid mode index or cutoff")
                if n > len(gammas):
                    continue
                gamma = float(gammas[n-1])
                source_gamma = float(row["gamma"])
                if not np.isfinite(source_gamma) or not np.isclose(source_gamma, gamma, rtol=0, atol=1e-9):
                    raise ValueError("source gamma does not match independently generated zero")
                in_window = row["in_window"].strip().lower() == "true"
                peak = row["nearest_peak_tau"].strip()
                tau = float(peak) if peak and in_window else None
                if tau is not None and (not np.isfinite(tau) or tau < 0):
                    raise ValueError("invalid peak ordinate")
                rows.append(dict(zero_index=n, gamma_n=gamma, prime_cutoff=cutoff,
                                 tau_peak=tau if tau is not None else "",
                                 gamma_error=tau-gamma if tau is not None else "",
                                 P_true=kappa*gamma, P_peak=kappa*tau if tau is not None else "",
                                 momentum_error=kappa*(tau-gamma) if tau is not None else "",
                                 in_window=in_window, peak_bracketed=row.get("peak_bracketed", ""),
                                 peak_track_id=row.get("nearest_peak_track_id", ""),
                                 peak_track_switched=row.get("peak_track_switched", "")))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}, line {line}: {exc}") from exc
    return sorted(rows, key=lambda r: (r["zero_index"], r["prime_cutoff"])), f"Loaded {len(rows)} real peak comparisons from {path}"
