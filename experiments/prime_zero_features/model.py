"""Finite critical-line history products and zero-blind feature detection.

The ordinary Euler product does not converge naively to 1/zeta on this line.
No individual prime generates a zeta zero; no physical interpretation is made.
"""

from __future__ import annotations

import numpy as np

from experiments.prime_zero_features.prime_only import (
    Signal, primes_through, make_signal, evaluate, safe_power, detect_extrema,
)
from experiments.prime_frame_geometry.model import first_zeta_zeros

DEFAULT_CUTOFFS = (5, 11, 29, 101, 331, 997, 5003, 10007)


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
