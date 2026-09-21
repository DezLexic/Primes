"""Prime-only detection, tracking and frozen mode selection. No reference spectrum."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np

from experiments.prime_zero_features.prime_only import (
    Signal, detect_extrema, evaluate, make_signal, primes_through,
)

SMOKE_CUTOFFS = (5, 7, 11, 17, 29, 47, 71, 101, 151, 211)


@dataclass(frozen=True)
class Rules:
    tail_fraction: float = 0.25
    min_observations: int = 4
    min_tail_count: int = 3
    persistence: float = 0.75
    max_drift: float = 0.5
    max_slope: float = 0.25
    agreement: float = 0.35

    def __post_init__(self):
        if not 0 < self.tail_fraction <= 1 or not 0 < self.persistence <= 1:
            raise ValueError("tail fraction and persistence must be in (0,1]")
        for name in ("min_observations", "min_tail_count"):
            n = getattr(self, name)
            if isinstance(n, bool) or not isinstance(n, int) or n < 2:
                raise ValueError(f"{name} must be an integer >=2")
        for name in ("max_drift", "max_slope", "agreement"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def cutoff_schedule(max_cutoff=10007, count=60):
    if max_cutoff < 5 or count < 4:
        raise ValueError("max cutoff >=5 and count >=4 required")
    # Integer P need not be prime. Save this exact sequence, including repeats
    # of the prime set where a rounded integer cutoff adds no new prime.
    return tuple(int(p) for p in np.unique(np.rint(np.geomspace(5, max_cutoff, count)).astype(int)))


def signal_bank(cutoffs, dataset, seed, replicate):
    """Fixed factor identities across all cutoffs, including shuffled controls.

    Shuffle only within each entering batch of primes. At every scheduled P
    the exact rate set is log(primes<=P), and old amplitude/rate pairs persist.
    This null depends on the schedule and is weaker than a global permutation.
    """
    base = make_signal(cutoffs[-1], "phase_scrambled" if dataset == "phase_scrambled" else "real", seed, replicate)
    primes = primes_through(cutoffs[-1])
    rates = base.rates.copy()
    if dataset == "shuffled_rates":
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 2]))
        start = 0
        for cutoff in cutoffs:
            end = int(np.searchsorted(primes, cutoff, side="right"))
            rates[start:end] = rng.permutation(rates[start:end])
            start = end
    elif dataset not in ("real", "phase_scrambled"):
        raise ValueError(f"unknown dataset: {dataset}")
    return primes, Signal(base.amplitudes, rates, base.phases, base.weights)


def cutoff_signal(primes, bank, cutoff, cutoff_mode):
    n = int(np.searchsorted(primes, cutoff, side="right"))
    if cutoff_mode not in ("hard", "taper"):
        raise ValueError("cutoff mode must be hard or taper")
    weights = np.ones(n) if cutoff_mode == "hard" else (1 + np.cos(np.pi * primes[:n] / cutoff)) / 2
    return Signal(bank.amplitudes[:n], bank.rates[:n], bank.phases[:n], weights)


def detect_peaks(tau, signal, prominence=0.1):
    """All sampled maxima passing the inherited fixed prominence threshold."""
    found = detect_extrema(tau, evaluate(signal, tau)["L"], signal,
                           prominence=prominence, kinds=("peak",))
    return [dict(feature_id=r["feature_id"], tau=r["tau"], value=r["value"],
                 prominence=r["prominence"], width=r["width_tau"], curvature=r["refined_curvature"],
                 refinement_ok=r["refinement_ok"], refinement=r["refinement"], track_id="") for r in found]


class Tracker:
    """Gated mutual nearest matching, one missing cutoff allowed by default.

    Near ties on either side are not linked. An ambiguous detection is a birth;
    old tracks remain eligible through max_gap missing steps, then die.
    Distances always use last observed tau; gaps do not enlarge the gate.
    """
    def __init__(self, max_jump=0.5, max_gap=1, ambiguity_margin=0.05):
        if not np.isfinite(max_jump) or max_jump <= 0:
            raise ValueError("max_jump must be finite and positive")
        if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
            raise ValueError("max_gap must be a nonnegative integer")
        if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be finite and nonnegative")
        self.max_jump, self.max_gap, self.margin = max_jump, max_gap, ambiguity_margin
        self.active, self.next_id = {}, 1

    def step(self, peaks, step):
        diagnostics = []
        for tid, old in list(self.active.items()):
            if step - old["step"] > self.max_gap + 1:
                diagnostics.append(dict(event="death", track_id=tid, feature_id="", distance="", missing_steps=step-old["step"]-1))
                del self.active[tid]
        valid = [p for p in peaks if p["refinement_ok"]]
        old = sorted(self.active.items())
        distances = np.array([[abs(o["tau"]-p["tau"]) for p in valid] for _, o in old]).reshape(len(old), len(valid))
        matched = set()
        for j, peak in enumerate(valid):
            candidates = sorted((distances[i, j], i) for i in range(len(old)) if distances[i, j] <= self.max_jump)
            accepted, ambiguous, gap, distance = False, False, 0, ""
            if candidates:
                distance, i = candidates[0]
                peers = sorted((distances[i, k], k) for k in range(len(valid)) if distances[i, k] <= self.max_jump)
                ambiguous = ((len(candidates) > 1 and candidates[1][0]-distance <= self.margin)
                             or (len(peers) > 1 and peers[1][0]-peers[0][0] <= self.margin))
                accepted = peers[0][1] == j and not ambiguous
            if accepted:
                tid, last = old[i]
                gap = step-last["step"]-1
                matched.add(tid)
                event = "reconnect" if gap else "match"
            else:
                tid, self.next_id = self.next_id, self.next_id + 1
                event = "ambiguous_birth" if ambiguous else "birth"
            peak["track_id"] = tid
            self.active[tid] = dict(step=step, tau=peak["tau"])
            diagnostics.append(dict(event=event, track_id=tid, feature_id=peak["feature_id"],
                                    distance=float(distance) if distance != "" else "", missing_steps=gap))
        for tid, last in old:
            if tid not in matched:
                diagnostics.append(dict(event="missing", track_id=tid, feature_id="", distance="", missing_steps=step-last["step"]))
        for peak in peaks:
            if not peak["refinement_ok"]:
                diagnostics.append(dict(event="refinement_failed", track_id="", feature_id=peak["feature_id"], distance="", missing_steps=0))
        return diagnostics


def stable_track(row, rules):
    return bool(row["observations"] >= rules.min_observations
                and row["tail_count"] >= rules.min_tail_count
                and row["persistence"] >= rules.persistence
                and row["tail_drift"] <= rules.max_drift
                and row["tail_slope"] <= rules.max_slope)


def track_metrics(peaks, cutoffs, rules):
    # All tracks share a global late-cutoff window, never a window after birth.
    eligible = math.ceil(len(cutoffs) * rules.tail_fraction)
    first_tail = cutoffs[-eligible]
    groups = {}
    for peak in peaks:
        if peak["track_id"] != "":
            groups.setdefault(peak["track_id"], []).append(peak)
    rows = []
    for tid, observations in sorted(groups.items()):
        tail = [p for p in observations if p["prime_cutoff"] >= first_tail]
        t = np.array([p["tau"] for p in tail])
        prom = np.array([p["prominence"] for p in tail])
        slope = float(abs(np.polyfit(np.log([p["prime_cutoff"] for p in tail]), t, 1)[0])) if len(t) >= 2 else math.nan
        row = dict(track_id=tid, first_cutoff=observations[0]["prime_cutoff"],
                   last_cutoff=observations[-1]["prime_cutoff"], observations=len(observations),
                   eligible_cutoffs=eligible, persistence=len(tail)/eligible, tail_count=len(tail),
                   tail_tau_mean=float(t.mean()) if len(t) else math.nan,
                   tail_tau_median=float(np.median(t)) if len(t) else math.nan,
                   tail_tau_std=float(t.std()) if len(t) else math.nan,
                   tail_tau_min=float(t.min()) if len(t) else math.nan,
                   tail_tau_max=float(t.max()) if len(t) else math.nan,
                   tail_drift=float(np.ptp(t)) if len(t) else math.nan, tail_slope=slope,
                   median_prominence=float(np.median(prom)) if len(t) else math.nan,
                   prominence_std=float(prom.std()) if len(t) else math.nan,
                   prominence_cv=float(prom.std()/prom.mean()) if len(t) and prom.mean() > 0 else math.nan)
        row["stable"] = stable_track(row, rules)
        rows.append(row)
    return rows


def consensus_spectrum(hard, taper, rules):
    """Mutual nearest stable tail medians within a fixed absolute tau gate."""
    hard = sorted((r for r in hard if stable_track(r, rules)), key=lambda r: r["tail_tau_median"])
    taper = sorted((r for r in taper if stable_track(r, rules)), key=lambda r: r["tail_tau_median"])
    rows = []
    for h in hard:
        t = min(taper, key=lambda r: abs(r["tail_tau_median"]-h["tail_tau_median"])) if taper else None
        if t is None:
            continue
        reverse = min(hard, key=lambda r: abs(r["tail_tau_median"]-t["tail_tau_median"]))
        error = abs(h["tail_tau_median"]-t["tail_tau_median"])
        if reverse is h and error <= rules.agreement:
            rows.append(dict(predicted_index=len(rows)+1, hard_track_id=h["track_id"], taper_track_id=t["track_id"],
                             hard_tau=h["tail_tau_median"], taper_tau=t["tail_tau_median"],
                             consensus_tau=(h["tail_tau_median"]+t["tail_tau_median"])/2,
                             agreement_error=error, hard_persistence=h["persistence"], taper_persistence=t["persistence"],
                             hard_tail_drift=h["tail_drift"], taper_tail_drift=t["tail_drift"],
                             hard_prominence=h["median_prominence"], taper_prominence=t["median_prominence"]))
    return rows


def sensitivity_rules(rules):
    return {
        "strict": replace(rules, persistence=min(1., rules.persistence+0.1), max_drift=rules.max_drift*0.75,
                          max_slope=rules.max_slope*0.75, agreement=rules.agreement*0.75),
        "baseline": rules,
        "loose": replace(rules, persistence=max(0.1, rules.persistence-0.1), max_drift=rules.max_drift*1.25,
                         max_slope=rules.max_slope*1.25, agreement=rules.agreement*1.25),
    }


def internal_momentum(spectrum, kappa):
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    return [dict(predicted_index=r["predicted_index"], consensus_tau=r["consensus_tau"], kappa=kappa,
                 predicted_P_int=kappa*r["consensus_tau"]) for r in spectrum]
