"""Measure raw prime frames in units of their own gap, without projecting zeros.

Collapse follows algebraically from p*s_p=2*z; it is not evidence for RH.
The inherited spectrum is collective. Primes label factor-zero combs, not
individual inherited zeros or points. This experiment makes no physical claims.
"""

from __future__ import annotations

import numpy as np

from experiments.prime_frame_geometry.model import (
    first_zeta_zeros, frame_geometry, inherited_zero_points, validate_frame,
)

FRAMES = (2, 3, 5, 7, 11, 13, 17, 19)
COMPARISON_FRAMES = (2, 3, 5, 7, 11)
HISTORY_FRAMES = (3, 5, 7, 11, 13)


def normalize_points(frame_prime, raw_points):
    """Multiply actual raw coordinates by p (divide both axes by Delta_p)."""
    p = validate_frame(frame_prime)
    points = np.asarray(raw_points, dtype=complex)
    if points.ndim != 1 or not np.all(np.isfinite(points)):
        raise ValueError("raw_points must be a finite one-dimensional sequence")
    return p * points


def normalized_frame(frame_prime, zeros, k_max=5):
    raw = frame_geometry(frame_prime, zeros, k_max)
    return {
        "frame_prime": raw["frame_prime"], "raw": raw,
        "inherited": normalize_points(frame_prime, raw["inherited"]),
        "combs": {q: normalize_points(frame_prime, points)
                  for q, points in raw["combs"].items()},
    }


def diagnostic_rows(states, zeros, test_point):
    """Keep identities, conjugates, and all coincident k=0 factor zeros.

Expected coordinates are calculated independently from common coordinates;
saved normalized values come from multiplication of the raw frame values.
"""
    def row(p, source, raw, normalized, expected, q="", n="", k="", sign=""):
        return dict(frame_prime=p, source_type=source, source_prime_q=q,
                    zero_index=n, k=k, sign=sign, raw_real=raw.real,
                    raw_imag=raw.imag, normalized_X=normalized.real,
                    normalized_Y=normalized.imag, expected_X=expected.real,
                    expected_Y=expected.imag,
                    residual_X=normalized.real - expected.real,
                    residual_Y=normalized.imag - expected.imag)

    for state in states:
        p = state["frame_prime"]
        for n, (rho, raw, point) in enumerate(zip(zeros, state["raw"]["inherited"], state["inherited"]), 1):
            for sign in (1, -1):
                transform = (lambda z: z) if sign == 1 else np.conjugate
                yield row(p, "inherited", transform(raw), transform(point),
                          2 * transform(rho), n=n, sign=sign)
        for q, points in state["combs"].items():
            k_max = (len(points) - 1) // 2
            for k, raw, point in zip(range(-k_max, k_max + 1), state["raw"]["combs"][q], points):
                expected = complex(0, 4 * np.pi * k / np.log(q))
                yield row(p, "passed_prime_factor", raw, point, expected, q=q, k=k)
        raw = inherited_zero_points(p, [test_point])[0]
        yield row(p, "artificial_off_line", raw, normalize_points(p, [raw])[0], 2 * test_point)


def verify_invariants(rows):
    """Fail the run on a coordinate mismatch; allow float64 rounding."""
    actual = np.array([(r["normalized_X"], r["normalized_Y"]) for r in rows])
    expected = np.array([(r["expected_X"], r["expected_Y"]) for r in rows])
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    return float(np.max(np.abs(actual - expected)))
