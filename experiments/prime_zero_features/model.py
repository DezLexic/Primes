"""Finite critical-line history products and zero-blind feature detection.

The ordinary Euler product does not converge naively to 1/zeta on this line.
No individual prime generates a zeta zero; no physical interpretation is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

from experiments.prime_frame.model import primes_below
from experiments.prime_frame_geometry.model import first_zeta_zeros

DEFAULT_CUTOFFS = (5, 11, 29, 101, 331, 997, 5003, 10007)


def primes_through(cutoff):
    if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer)) or cutoff < 2:
        raise ValueError("cutoff must be an integer >=2 (need not be prime)")
    return primes_below(int(cutoff) + 1)


@dataclass(frozen=True)
class Signal:
    amplitudes: np.ndarray
    rates: np.ndarray
    phases: np.ndarray
    weights: np.ndarray


def make_signal(cutoff, kind="real", seed=1729, replicate=0):
    primes = primes_through(cutoff)
    rates = np.log(primes.astype(float))
    phases = np.zeros(len(primes))
    weights = np.ones(len(primes))
    if kind == "phase_scrambled":
        # Same seeded prefix at every cutoff: each prime keeps its phase.
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 1]))
        phases = rng.uniform(-np.pi, np.pi, len(primes))
    elif kind == "shuffled_rates":
        # At each cutoff permute exactly its own frequency set. These are
        # independently defined null samples, NOT a nested product trajectory.
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 2, int(cutoff)]))
        rates = rng.permutation(rates)
    elif kind == "raised_cosine":
        weights = 0.5 * (1 + np.cos(np.pi * primes / cutoff))
    elif kind != "real":
        raise ValueError(f"unknown signal kind: {kind}")
    return Signal(1 / np.sqrt(primes), rates, phases, weights)


def evaluate(signal, tau):
    """L, continuously lifted phase, L', L'', Phi'; bounded temporary arrays.

    Every factor has positive real part because r<1. Its principal argument
    is continuous. Summing those arguments ALREADY gives an unwrapped phase;
    wrapping the sum and applying np.unwrap could alias coarse grids.
    For scrambled phases the same branch is used, without re-zeroing Phi(0).
    """
    tau = np.asarray(tau, dtype=float)
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must be finite")
    flat = tau.ravel()
    result = np.zeros((5, flat.size))
    for start in range(0, flat.size, 2048):
        t = flat[start:start + 2048]
        target = result[:, start:start + len(t)]
        for begin in range(0, len(signal.rates), 64):
            sl = slice(begin, begin + 64)
            r, w = signal.amplitudes[sl, None], signal.weights[sl, None]
            rate = signal.rates[sl, None]
            theta = rate * t + signal.phases[sl, None]
            u = r * np.exp(-1j * theta)
            factor = 1 - u
            power = (1-r)**2 + 4*r*np.sin(theta/2)**2
            first = 1j * rate * u / factor
            second = rate**2 * u / factor**2
            target[0] += np.sum(w * np.log(power), axis=0)
            target[1] += np.sum(w * np.angle(factor), axis=0)
            target[2] += np.sum(w * 2 * first.real, axis=0)
            target[3] += np.sum(w * 2 * second.real, axis=0)
            target[4] += np.sum(w * first.imag, axis=0)
    return {name: result[i].reshape(tau.shape) for i, name in enumerate(
        ("L", "phase", "slope", "curvature", "phase_slope"))}


def safe_power(log_power):
    """NaN marks overflow/underflow; do not silently clip deep minima to zero."""
    value = np.asarray(log_power)
    valid = (value >= np.log(np.nextafter(0.0, 1.0))) & (value <= np.log(np.finfo(float).max))
    result = np.full(value.shape, np.nan)
    with np.errstate(over="ignore", under="ignore"):
        np.exp(value, out=result, where=valid)
    return result


def detect_extrema(tau, values, signal, prominence=0.1, min_width=0.0):
    """Detect BEFORE comparing zeros; this API has no gamma input.

    Prominence and half-prominence widths come from the sampled profile.
    Width threshold is in tau units. Endpoints are never local extrema.
    Locations/values are refined within [tau[i-1],tau[i+1]].
    """
    tau, values = np.asarray(tau), np.asarray(values)
    if tau.ndim != 1 or len(tau) < 3 or values.shape != tau.shape:
        raise ValueError("need matching one-dimensional grids with at least 3 points")
    dt = np.diff(tau)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(tau)) or not np.all(dt > 0) or not np.allclose(dt, dt[0]):
        raise ValueError("need a finite, increasing uniform grid")
    if not np.isfinite(prominence) or not np.isfinite(min_width) or min(prominence, min_width) < 0:
        raise ValueError("feature thresholds must be finite and nonnegative")
    rows = []
    for kind, sign in (("peak", 1), ("min", -1)):
        indices, props = find_peaks(sign * values, prominence=(prominence, None),
                                    width=(min_width / dt[0], None), plateau_size=(None, None))
        for j, i in enumerate(indices):
            left, right = float(tau[i-1]), float(tau[i+1])
            slope = lambda t: float(evaluate(signal, t)["slope"])
            # Positive derivative of sign*L on left, negative on right.
            if sign*slope(left) >= 0 and sign*slope(right) <= 0:
                location = brentq(slope, left, right, xtol=1e-11)
                method = "derivative_root"
            else:
                fit = minimize_scalar(lambda t: -sign*float(evaluate(signal, t)["L"]),
                                      bounds=(left, right), method="bounded",
                                      options={"xatol": 1e-11})
                location = float(fit.x)
                method = "bounded_fallback" if fit.success else "failed_fallback"
            exact = evaluate(signal, location)
            rows.append(dict(kind=kind, feature_id=f"{kind}_{j+1}", tau=location,
                             value=float(exact["L"]), grid_tau=float(tau[i]),
                             grid_value=float(values[i]), prominence=float(props["prominences"][j]),
                             width_tau=float(props["widths"][j]*dt[0]), bracket_left=left,
                             bracket_right=right, refinement=method,
                             refined_slope=float(exact["slope"]), refined_curvature=float(exact["curvature"]),
                             refinement_ok=bool(left < location < right and sign*exact["curvature"] < 0
                                                and abs(exact["slope"]) < 1e-6)))
    return sorted(rows, key=lambda row: row["tau"])


def assign_tracks(previous, current, next_id, max_jump=0.5):
    """Conservative mutual-nearest matching by kind; independent of zeta zeros.

    Discrete cutoff matching is a heuristic, not certified branch continuation.
    New/unmatched features get new IDs; old tracks terminate.
    """
    for row in current:
        candidates = [p for p in previous if p["kind"] == row["kind"]]
        closest = min(candidates, key=lambda p: abs(p["tau"]-row["tau"])) if candidates else None
        peers = [p for p in current if p["kind"] == row["kind"]]
        reverse = min(peers, key=lambda p: abs(p["tau"]-closest["tau"])) if closest else None
        matched = closest is not None and reverse is row and abs(closest["tau"]-row["tau"]) <= max_jump
        row["track_id"] = closest["track_id"] if matched else next_id
        if not matched:
            next_id += 1
    return next_id


def compare_zeros(zeros, signal, features, tau_min, tau_max):
    """Out-of-window ordinates retain exact evaluations but NO nearest offsets.

    Bracketing flags identify boundary-censored nearest-feature comparisons.
    """
    values = evaluate(signal, np.imag(zeros))
    rows = []
    for i, gamma in enumerate(np.imag(zeros)):
        inside = bool(tau_min < gamma < tau_max)
        row = dict(zero_index=i+1, gamma=float(gamma), in_window=inside,
                   L_at_gamma=float(values["L"][i]), slope_at_gamma=float(values["slope"][i]),
                   curvature_at_gamma=float(values["curvature"][i]),
                   phase_at_gamma=float(values["phase"][i]), phase_slope_at_gamma=float(values["phase_slope"][i]))
        for kind in ("peak", "min"):
            candidates = [f for f in features if f["kind"] == kind] if inside else []
            nearest = min(candidates, key=lambda f: abs(f["tau"]-gamma)) if candidates else None
            row[f"{kind}_bracketed"] = bool(candidates and min(f["tau"] for f in candidates) <= gamma <= max(f["tau"] for f in candidates))
            for suffix, key in (("tau", "tau"), ("value", "value"), ("prominence", "prominence"),
                                ("width", "width_tau"), ("id", "feature_id"), ("track_id", "track_id")):
                row[f"nearest_{kind}_{suffix}"] = nearest.get(key, "") if nearest else ""
            row[f"{kind}_offset"] = float(nearest["tau"]-gamma) if nearest else ""
        rows.append(row)
    return rows


def resolution_comparison(coarse, fine, tolerance):
    """Count unmatched extrema; a tiny nearest distance alone cannot pass."""
    rows = []
    for kind in ("peak", "min"):
        a = np.array([f["tau"] for f in coarse if f["kind"] == kind])
        b = np.array([f["tau"] for f in fine if f["kind"] == kind])
        # In one dimension sorted one-to-one matching detects missing features.
        equal = len(a) == len(b)
        error = float(np.max(np.abs(a-b))) if equal and len(a) else np.nan
        rows.append(dict(kind=kind, coarse_count=len(a), fine_count=len(b),
                         max_position_change=error, tolerance=tolerance,
                         passed=bool(equal and len(a) and error <= tolerance
                                     and all(f["refinement_ok"] for f in coarse+fine if f["kind"] == kind))))
    return rows
