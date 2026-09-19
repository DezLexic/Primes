"""Coordinate geometry of inherited zeta zeros and passed-prime factor zeros.

For rho = beta + i*gamma, s_p = 2*rho/p, so p*Re(s_p) = 2*beta
is invariant. For critical-line samples beta=1/2 this equals 1. The
coordinate transformation does not establish RH. Primes label factors;
they are not points on the inherited-zero line. No physical gap is modeled.
"""

from __future__ import annotations

from functools import lru_cache
from math import isqrt

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_frame(frame_prime: int) -> int:
    p = _integer(frame_prime, "frame_prime", 2)
    if any(p % d == 0 for d in range(2, isqrt(p) + 1)):
        raise ValueError("frame_prime must be prime")
    return p


def passed_prime_list(frame_prime: int) -> np.ndarray:
    """Primes strictly below p; successive frames add the previous frame prime."""
    p = validate_frame(frame_prime)
    # Same small Eratosthenes sieve convention as the prime_frame experiment.
    sieve = np.ones(p, dtype=bool)
    sieve[:2] = False
    for q in range(2, isqrt(p - 1) + 1):
        if sieve[q]:
            sieve[q * q :: q] = False
    return np.flatnonzero(sieve)


@lru_cache(maxsize=8)
def _zeta_zeros(count: int, dps: int) -> tuple[complex, ...]:
    from mpmath import mp

    # mpmath supplies critical-line samples, not a numerical proof of RH.
    # Compute with guard digits, then use complex128 for plotting/CSV output.
    with mp.workdps(dps):
        return tuple(complex(mp.zetazero(n)) for n in range(1, count + 1))


def first_zeta_zeros(count: int = 20, dps: int = 30) -> np.ndarray:
    """First count positive-height critical-line zeros from mpmath.zetazero."""
    count = _integer(count, "count", 1)
    dps = _integer(dps, "dps", 15)
    return np.array(_zeta_zeros(count, dps), dtype=complex)


def frame_map_zero(rho: complex, frame_prime: int) -> complex:
    """Map general beta+i*gamma without projecting beta onto the critical line."""
    p = validate_frame(frame_prime)
    if not np.isfinite(rho):
        raise ValueError("rho must be finite")
    return 2 * complex(rho) / p


def inherited_zero_points(frame_prime: int, zero_list) -> np.ndarray:
    p = validate_frame(frame_prime)
    zeros = np.asarray(zero_list, dtype=complex)
    if zeros.ndim != 1 or not np.all(np.isfinite(zeros)):
        raise ValueError("zero_list must be a finite one-dimensional sequence")
    return 2 * zeros / p


def passed_prime_comb_points(frame_prime: int, passed_prime: int, k_max: int = 8) -> np.ndarray:
    """Factor zeros i*4*pi*k/(p*ln(q)), in order k=-K,...,K, including k=0.

    All q share the origin; keep each factor's point (and its identity) rather
    than deduplicating. p=2 has no passed factors and hence no comb points.
    """
    p = validate_frame(frame_prime)
    q = validate_frame(passed_prime)
    if q >= p:
        raise ValueError("passed_prime must be strictly below frame_prime")
    k_max = _integer(k_max, "k_max", 0)
    return 1j * 4 * np.pi * np.arange(-k_max, k_max + 1) / (p * np.log(q))


def frame_gap(frame_prime: int) -> float:
    """Horizontal separation from Re(s_p)=0 to the mapped critical line."""
    return 1 / validate_frame(frame_prime)


def frame_geometry(frame_prime: int, zero_list, k_max: int = 8) -> dict:
    """Animation-ready frame state; inherited array preserves zero identities."""
    p = validate_frame(frame_prime)
    _integer(k_max, "k_max", 0)
    return {
        "frame_prime": p,
        "gap": frame_gap(p),
        "inherited": inherited_zero_points(p, zero_list),
        "combs": {int(q): passed_prime_comb_points(p, int(q), k_max)
                  for q in passed_prime_list(p)},
    }
