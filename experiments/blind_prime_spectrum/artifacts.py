"""CSV schemas and an integrity boundary shared by generation and evaluation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CONTEXT = ["cutoff_mode", "dataset", "replicate"]
PEAK_FIELDS = CONTEXT + ["prime_cutoff", "feature_id", "track_id", "tau", "value", "prominence", "width", "curvature", "refinement_ok", "refinement"]
TRACK_FIELDS = CONTEXT + ["track_id", "first_cutoff", "last_cutoff", "observations", "eligible_cutoffs", "persistence", "tail_count", "tail_tau_mean", "tail_tau_median", "tail_tau_std", "tail_tau_min", "tail_tau_max", "tail_drift", "tail_slope", "median_prominence", "prominence_std", "prominence_cv", "stable"]
CONSENSUS_FIELDS = ["predicted_index", "hard_track_id", "taper_track_id", "hard_tau", "taper_tau", "consensus_tau", "agreement_error", "hard_persistence", "taper_persistence", "hard_tail_drift", "taper_tail_drift", "hard_prominence", "taper_prominence"]

GUARDRAILS = """This is a finite critical-line prime-product numerical experiment.
Finite ordinary Euler products do not naively converge to 1/zeta on Re(s)=1/2.
Zeta zeros are not used to generate, select, tune, or plot the blind predictions.
A stable finite-product peak spectrum would not prove RH or asymptotic convergence.
Controls may also produce apparently stable discrete modes; no special status is assumed.
Feature density can create accidental structure. Inspect counts and grid checks.
Hard/taper agreement is a robustness test, not proof.
Track continuation is heuristic: ambiguities and reconnects warn of possible switching.
No physical particle or mass spectrum is established; internal momentum is a toy mapping.
Even an excellent numerical match leaves the generating operator and mathematical explanation unknown.
Smoke runs check implementation only, not the scientific hypothesis.
"""


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(output):
    """Called last, after every blind CSV, parameter file and PNG is written."""
    output = Path(output)
    required = ["blind_consensus_spectrum.csv", "blind_parameters.json", "blind_tracks.csv",
                "blind_control_consensus.csv", "blind_sensitivity_spectra.csv",
                "blind_consensus_spectrum.png"]
    if any(not (output / name).is_file() for name in required):
        raise ValueError("blind artifacts are incomplete; cannot freeze")
    files = {p.name: sha256(p) for p in sorted(output.glob("blind_*")) if p.is_file() and p.name != "blind_freeze.json"}
    manifest = dict(schema_version=1, frozen_utc=datetime.now(timezone.utc).isoformat(), files=files)
    # Exclusive creation prevents accidentally replacing a frozen prediction.
    with (output / "blind_freeze.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def verify_freeze(output):
    output = Path(output)
    manifest = json.loads((output / "blind_freeze.json").read_text(encoding="utf-8"))
    for name in ("blind_consensus_spectrum.csv", "blind_parameters.json", "blind_control_consensus.csv", "blind_sensitivity_spectra.csv"):
        if name not in manifest["files"]:
            raise ValueError(f"freeze manifest missing {name}")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or not name.startswith("blind_"):
            raise ValueError("invalid manifest filename")
        if not (output / name).is_file() or sha256(output / name) != expected:
            raise ValueError(f"frozen artifact changed or missing: {name}")
    return manifest
