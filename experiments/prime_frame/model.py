"""Finite prime-history products; no critical-line Euler product or mass model."""

from __future__ import annotations

import numpy as np


def primes_below(p: int) -> np.ndarray:
    """Return primes strictly below integer p, using an Eratosthenes sieve."""
    if not isinstance(p, (int, np.integer)) or p < 0:
        raise ValueError("p must be a non-negative integer")
    sieve = np.ones(p, dtype=bool)
    sieve[:2] = False
    for q in range(2, int(np.sqrt(max(0, p - 1))) + 1):
        if sieve[q]:
            sieve[q * q :: q] = False
    return np.flatnonzero(sieve)


def _discarded(frame_prime: int) -> np.ndarray:
    primes = primes_below(frame_prime + 1)
    if primes.size == 0 or primes[-1] != frame_prime:
        raise ValueError("frame_prime must be a prime integer")
    return primes[:-1]


def history_factor(z, frame_prime: int):
    result = np.ones_like(np.asarray(z, dtype=complex))
    for q in _discarded(frame_prime):
        result *= 1 - np.exp(-np.asarray(z) * np.log(float(q)))
    return result


def _log_factor(tau, q: int):
    # Equivalent to 1 + 1/q - 2/sqrt(q)*cos(tau*log(q)), with less
    # cancellation near zero. Each factor is strictly positive on Re(z)=1/2.
    r = 1 / np.sqrt(q)
    return np.log((1 - r) ** 2 + 4 * r * np.sin(np.asarray(tau) * np.log(q) / 2) ** 2)


def log_history_power(tau, frame_prime: int):
    result = np.zeros_like(np.asarray(tau, dtype=float))
    for q in _discarded(frame_prime):
        result += _log_factor(tau, q)
    return result


def history_power(tau, frame_prime: int):
    return np.exp(log_history_power(tau, frame_prime))


def log_baseline(frame_prime: int) -> float:
    return float(np.log1p(1 / _discarded(frame_prime)).sum())


def baseline(frame_prime: int) -> float:
    return float(np.exp(log_baseline(frame_prime)))


def normalized_history_power(tau, frame_prime: int):
    return np.exp(log_history_power(tau, frame_prime) - log_baseline(frame_prime))


def _chi_term(q):
    r = 1 / np.sqrt(q)
    return r * np.log(q) ** 2 / (1 - r) ** 2


def chi(frame_prime: int) -> float:
    return float(_chi_term(_discarded(frame_prime)).sum())


def numerical_chi(frame_prime: int, step: float = 1e-4) -> float:
    """Half the centered second derivative of log W at zero."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    return float((log_history_power(step, frame_prime)
                  - 2 * log_history_power(0, frame_prime)
                  + log_history_power(-step, frame_prime)) / (2 * step**2))


def quadratic_width_common(frame_prime: int) -> float:
    curvature = chi(frame_prime)
    return 1 / np.sqrt(curvature) if curvature else float("inf")


def quadratic_width_frame(frame_prime: int) -> float:
    return 2 * quadratic_width_common(frame_prime) / frame_prime


def to_common_coordinate(s_p, frame_prime: int):
    _discarded(frame_prime)
    return frame_prime * np.asarray(s_p) / 2


def run_frame_sweep(max_prime: int, tau) -> tuple[list[dict], np.ndarray]:
    """Accumulate one factor per successive frame; rows align with log D profiles.

    Runtime and memory are O(pi(max_prime) * len(tau)). Raw powers may
    underflow for very large frames; log columns remain available.
    """
    if not isinstance(max_prime, (int, np.integer)) or max_prime < 2:
        raise ValueError("max_prime must be an integer >= 2")
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or tau.size < 2 or not np.all(np.isfinite(tau)):
        raise ValueError("tau must be a finite one-dimensional grid with >= 2 points")
    frames = primes_below(max_prime + 1)
    profiles = np.empty((len(frames), len(tau)))
    log_w = np.zeros_like(tau)
    log_w0 = log_b = curvature = 0.0
    rows = []
    for index, p in enumerate(frames):
        width = 1 / np.sqrt(curvature) if curvature else float("inf")
        estimate = 2 * np.sqrt(p) * np.log(p)
        rows.append(dict(
            frame_prime=int(p), discarded_primes=index,
            W0=float(np.exp(log_w0)), log_W0=float(log_w0),
            B=float(np.exp(log_b)), log_B=float(log_b),
            D0=float(np.exp(log_w0 - log_b)), log_D0=float(log_w0 - log_b),
            chi=float(curvature), tau_quad=float(width),
            t_quad=float(2 * width / p),
            chi_pnt_ratio=float(curvature / estimate),
            scaled_width=float(width * p**0.25 * np.sqrt(np.log(p))),
        ))
        profiles[index] = log_w - log_b
        # R_next(z) = (1 - p**(-z)) R_p(z): p joins history AFTER its row.
        log_w += _log_factor(tau, p)
        log_w0 += float(_log_factor(0, p))
        log_b += float(np.log1p(1 / p))
        curvature += float(_chi_term(p))
    return rows, profiles
