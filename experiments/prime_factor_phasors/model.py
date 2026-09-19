"""Finite two-phasor factors and their products; no physical interpretation."""

from __future__ import annotations

import numpy as np

from experiments.prime_frame.model import history_factor
from experiments.prime_frame_geometry.model import (
    passed_prime_list, passed_prime_comb_points, validate_frame,
)
from experiments.prime_frame_normalized.model import normalize_points

PRIMES = (2, 3, 5, 7)
HISTORY_FRAMES = (3, 5, 7, 11, 13, 17)
TRACK_FRAMES = (3, 5, 7, 11, 13, 17)


def phasors(q, sigma, tau):
    """Return A1=1, signed A2=-q**(-z), and their sum (array-friendly)."""
    q = validate_frame(q)
    z = np.asarray(sigma) + 1j * np.asarray(tau)
    if not np.all(np.isfinite(z)):
        raise ValueError("sigma and tau must be finite")
    rotating = np.exp(-z * np.log(q))
    # expm1 avoids subtractive loss near z=0; arrows retain the signed A2.
    return np.ones_like(z), -rotating, -np.expm1(-z * np.log(q))


def factor_power(q, sigma, tau):
    """Stable equivalent of 1+r*r-2*r*cos(tau*ln(q)), r=q**(-sigma)."""
    q = validate_frame(q)
    sigma, tau = np.asarray(sigma, dtype=float), np.asarray(tau, dtype=float)
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(tau)):
        raise ValueError("sigma and tau must be finite")
    r = np.exp(-sigma * np.log(q))
    return np.expm1(-sigma * np.log(q)) ** 2 + 4 * r * np.sin(tau * np.log(q) / 2) ** 2


def minimum_power(q, sigma):
    return factor_power(q, sigma, 0.0)


def zero_heights(q, tau_max):
    """All analytic factor zeros within the displayed common-coordinate window."""
    q = validate_frame(q)
    if not np.isfinite(tau_max) or tau_max <= 0:
        raise ValueError("tau_max must be finite and positive")
    k_max = int(np.floor(tau_max * np.log(q) / (2 * np.pi)))
    k = np.arange(-k_max, k_max + 1)
    return k, 2 * np.pi * k / np.log(q)


def history_power(frame_prime, sigma, tau):
    """Product of intensities, not intensity of a many-amplitude sum."""
    result = np.ones_like(np.asarray(tau, dtype=float))
    for q in passed_prime_list(frame_prime):
        result = result * factor_power(int(q), sigma, tau)
    return result


def sampling_grid(tau_max, points, primes):
    # Insert the analytic heights so a uniform grid cannot miss narrow zeros.
    heights = [zero_heights(int(q), tau_max)[1] for q in primes]
    return np.unique(np.concatenate([np.linspace(-tau_max, tau_max, points), *heights]))


def analytic_rows(frames, tau_max):
    """Retain each factor's origin separately; export all three coordinates."""
    for p in frames:
        for q in passed_prime_list(p):
            q = int(q)
            k_values, tau = zero_heights(q, tau_max)
            raw = passed_prime_comb_points(p, q, len(k_values) // 2)
            normalized = normalize_points(p, raw)
            for k, height, point, norm in zip(k_values, tau, raw, normalized):
                yield dict(frame_prime=p, source_prime_q=q, k=int(k),
                           z_real=0.0, z_imag=float(height),
                           frame_real=point.real, frame_imag=point.imag,
                           normalized_X=norm.real, normalized_Y=norm.imag,
                           phase_rate=np.log(q), delta_tau=2*np.pi/np.log(q),
                           delta_frame_imag=4*np.pi/(p*np.log(q)),
                           delta_normalized_Y=4*np.pi/np.log(q))
