# Blind finite-prime spectrum

Can a finite prime construction produce a stable discrete set of peak locations
before receiving any Riemann-zero ordinates? This experiment first answers that
finite-window question, freezes its answer, and only then compares it with zeros.
An empty predicted spectrum is a valid outcome.

```text
primes -> detected peaks -> tracks -> blind stability -> hard/taper consensus
       -> save controls, sensitivity predictions and plots -> SHA-256 freeze
       -> load reference zeros -> evaluate the frozen files
```

## Run manually from PowerShell

From the repository root, using the existing environment:

```powershell
# Lightweight smoke; the checked-in example already occupies results/smoke.
.\.venv\Scripts\python.exe .\experiments\blind_prime_spectrum\run.py --mode smoke --kappa 1 --output .\experiments\blind_prime_spectrum\results\smoke_local

# Normal sweep: approximately 60 integer cutoffs through P=10007.
.\.venv\Scripts\python.exe .\experiments\blind_prime_spectrum\run.py --mode normal --kappa 1

# Optional deeper sweep: approximately 80 cutoffs through P=50021.
.\.venv\Scripts\python.exe .\experiments\blind_prime_spectrum\run.py --mode deep --kappa 1

# Lightweight unit tests only.
.\.venv\Scripts\python.exe -m unittest discover -s experiments/blind_prime_spectrum -p "test_*.py" -v
```

Only unit tests and the smoke preset were run during implementation. Normal and
deep sweeps are deliberately manual. Defaults are anchored to this experiment's
directory and write to `results/normal` and `results/deep`. `--output` overrides
the destination. **Use a new or empty output directory on every generation run**;
the program refuses to replace a frozen or partly written run. Choose another
name if `smoke_local`, `normal`, or `deep` already exists.

| Preset | Cutoffs | Tau window / uniform points | Replicates per control |
| --- | --- | --- | --- |
| smoke | 5, 7, 11, 17, 29, 47, 71, 101, 151, 211 | 0 to 35 / 1401 | 2 |
| normal | rounded geometric schedule, requested 60, 5 to 10007 | 0 to 60 / 6001 | 3 |
| deep | rounded geometric schedule, requested 80, 5 to 50021 | 0 to 100 / 16001 | 5 |

Rounded duplicate cutoffs are removed; the exact list is saved. A cutoff need
not be prime. Some early steps can retain the same prime set; counts record this.
Both hard and tapered signals run independently for every dataset and replicate.
Normal/deep runtime grows with cutoffs, primes, samples, peaks, and replicates.

For a stronger operational separation, run generation and evaluation in separate
Python processes:

```powershell
.\.venv\Scripts\python.exe .\experiments\blind_prime_spectrum\run.py --mode normal --blind-only --output .\experiments\blind_prime_spectrum\results\normal_blind
.\.venv\Scripts\python.exe .\experiments\blind_prime_spectrum\evaluate.py --output .\experiments\blind_prime_spectrum\results\normal_blind
```

You can inspect and commit the blind directory between these commands. Evaluation
can be repeated without regenerating any predictions. Its optional `--tolerance`
and `--dps` affect evaluation only. `run.py --help` lists all generation overrides,
including explicit `--cutoffs`, or `--max-cutoff` with `--cutoff-count`.

## Prime-only implementation

`prime_zero_features/prime_only.py` contains the existing stable log-power
implementation, analytic derivatives, sieve adapter, controls, and extremum
refinement. The previous `model.py` re-exports its public helpers for compatibility.
The only detector extension is an optional `kinds` argument: old callers still
detect peaks and minima; this experiment requests peaks only.

`blind_model.py`, `generate.py`, `plotting.py`, and their imported numerical
dependencies contain no zero loader. They do not import the old zero-aware model
or the internal-momentum model. They receive no reference arrays. `evaluate.py`
imports the existing `first_zeta_zeros` helper lazily, after verifying the freeze.
`run.py` does not even import evaluation until generation has returned a manifest.

The signal is the finite sum

```text
L_P(tau) = sum[q<=P] w(q/P) log((1-r_q)^2 + 4 r_q sin(theta_q/2)^2)
r_q = 1/sqrt(q), theta_q = tau*log(q) for real primes
hard:  w = 1
taper: w = (1 + cos(pi*q/P))/2
```

This is exactly the requested log of squared factor magnitude, with stable
evaluation near cancellation. Taper weights multiply log terms; they do not
replace amplitudes inside the logarithm. Taper weights follow the prime that
supplies the amplitude, including in shuffled controls.

Every sampled local maximum passing **prominence >= 0.1** is detected, using the
previous experiment's unchanged default. There is no minimum-width filter.
Analytic derivative roots refine locations; bounded optimization is the fallback.
Height and curvature are evaluated at the refined point; prominence and
half-prominence width remain grid measurements. Endpoints are excluded. Failed
refinements remain in `blind_peaks.csv` with blank track IDs and diagnostics,
but cannot become stable modes.

Detection is finite-resolution, not a proof that every continuous local maximum
was resolved. Double-resolution checks at the first and final cutoff cover
**every dataset and replicate**. Counts must agree and refined locations must
change by at most 1e-6. Failures are retained and reported; they do not trigger
automatic retuning. Inspect them before drawing scientific conclusions.

## Tracking and stability, fixed before evaluation

Tracking uses mutual nearest tau locations across eligible tracks and current
peaks, with a fixed maximum jump of 0.5 tau. There is no reference-based ranking.
Each track may miss one scheduled cutoff and reconnect using its last observed
location; the jump gate never expands across gaps. Unmatched detections are
births, missing tracks expire after the allowed gap. Surviving tracks at the end
are right-censored, not assumed immortal.

If the two closest eligible alternatives on either side differ in distance by
at most 0.05, the link is ambiguous and is not accepted. Detections become new
tracks; old ones can expire or reconnect. This conservatively records potential
splits/merges. It cannot certify peak identity: ambiguous births, reconnects, and
link distances in `blind_tracking_diagnostics.csv` are switching warnings.
Track IDs are local to `(cutoff_mode, dataset, replicate)`.

The common tail window is the final `ceil(0.25 * number_of_cutoffs)` scheduled
cutoffs. It is **not** the last quarter of each track's lifetime or the last
quarter of the numerical P interval. Persistence is tail observations divided by
the number of scheduled cutoffs in this global tail, even for late births.
`eligible_cutoffs` records that denominator; `observations` covers the full run.

The baseline requires all of:

- At least 4 observations over the complete schedule and at least 3 in the tail.
- Tail persistence >= 0.75.
- Tail drift `max(tau)-min(tau) <= 0.5`.
- Absolute least-squares slope of `tau` against `log(P) <= 0.25`.

These are broad numerical defaults, set before running the smoke comparison.
They are configurable and are not optimized against reference zeros. Standard
deviation, mean, median, extrema, tail prominence median, population standard
deviation, and coefficient of variation are also saved for every track. Undefined
metrics are NaN, not zero. No additional prominence gate is used beyond detection.

Each predicted location is simply its tail **median**. There is no fitted
asymptote. Hard and taper stable modes are matched by mutual nearest median tau
within 0.35. Each track participates in at most one match. The consensus is the
arithmetic mean of the two medians, sorted by tau. Both tracks must independently
pass the same stability rules. All candidates and unmatched stable modes remain
available in the CSVs.

The predeclared sensitivity sweep uses the same measured tracks:

- Strict: persistence +0.1 (capped at 1), and drift/slope/agreement limits x0.75.
- Baseline: the configured rules above.
- Loose: persistence -0.1 (floored at 0.1), and drift/slope/agreement limits x1.25.

Every variant's spectrum is frozen before evaluation. The summary records counts,
added/lost track pairs and Jaccard overlap with baseline. Locations of retained
pairs do not change because the same tail medians are used. This is a small
selection-threshold sweep, not a sweep of detector, tracker, or tail-window
settings. All metrics are saved so further predeclared filters need no signal
recalculation; after viewing evaluation, any such analysis is exploratory rather
than a new held-out prediction. No variant is chosen for its zeta score.

## Controls and their limitations

Every replicate uses the same detector, tracker, rules, sensitivity variants, and
hard/taper consensus code. Seed 1729 is the default.

- **Phase scrambled:** each amplitude/rate pair gets a fixed seeded random phase
  in [-pi, pi], retained at every cutoff and in both truncation modes.
- **Shuffled rates:** amplitudes are unchanged and the exact set of log-prime
  rates is preserved at every scheduled cutoff. Newly entering prime batches
  are independently permuted once; each amplitude/rate assignment then stays
  fixed as P grows. The rate attached to every amplitude is saved explicitly.

The older experiment reshuffled all rates separately at each cutoff and
deliberately did not track that control. Using that control as a trajectory would
confound instability with rebuilding the signal. Fixed batch permutations avoid
that issue but constitute a **constrained null**, not a uniformly random global
permutation: nearby rates remain in the same batch, singleton batches cannot
shuffle, and the null depends on the cutoff schedule. This can preserve much of
the real-prime structure. Both that limitation and controls producing stable
spectra must inform interpretation. No non-prime control was added.

`blind_control_summary.csv` reports per-mode track counts, stable counts, consensus
counts, persistence, drift, prominence, and ambiguity/reconnection counts. Its
unqualified median persistence/drift use tracks with enough tail observations;
the `stable_median_*` fields use selected tracks only. Stable fraction uses all
tracks as denominator. Inspect raw counts and density alongside these summaries.
Replicate ranges are descriptive, not confidence intervals or formal tests.

## Freeze and evaluation

All blind CSVs, parameters, summaries, and five PNGs are written before
`blind_freeze.json`. This manifest contains SHA-256 hashes of every `blind_*`
artifact and a UTC freeze timestamp. Parameters save exact schedules, software
versions, rules, seed, control definition, and source hashes. Evaluation verifies
the manifest **before calling the zero helper**, reads the frozen CSVs, and
verifies again after writing its separate outputs. Missing or modified artifacts
abort evaluation. Hashes detect later changes; they are not an external trusted
timestamp or proof against deliberate tampering with both files and manifest.

Evaluation computes enough positive zeros to cover the entire tau window,
including a zero above it, and enough for every ordered prediction. It compares:

1. Ordered predicted tau j against positive zero j, starting at one, for the
   overlapping count. Missing/extra predictions can shift all later ordered errors.
2. Each predicted tau against its independently nearest zero, without reordering
   the primary prediction. Relative error is absolute error divided by gamma.

Missing/extra diagnostics additionally use maximum-cardinality, minimum-distance
one-to-one matching inside the observation window, with evaluation-only tolerance
0.5 by default. Duplicate predictions near one zero cannot inflate coverage.
Unmatched predictions and zeros are saved explicitly. Nearest matching includes
the bracketing zero above the window; missing/extra coverage uses only in-window
zeros. Neither method feeds back into generation.

The same evaluation is applied to frozen control and sensitivity spectra.
`evaluation_control_summary.csv` reports precision, recall, counts, ordered errors,
and nearest distances. Compare these together: a sparse selected spectrum can
have excellent nearest distances but miss most reference modes. Increasing peak
density can reduce nearest distances by accident. There is no automated claim
that real primes outperform controls.

## Outputs to inspect and send back

Keep and push the **whole run directory** so the freeze can be verified. All result
files are intentionally eligible for version control. Include the JSON and text
sidecars as well as CSVs/PNGs; do not hand-edit a frozen file.

| Files | Purpose |
| --- | --- |
| `blind_peaks.csv` | Every detection, feature properties and track assignment; join by context and cutoff |
| `blind_tracks.csv` | All lifetime/tail metrics and baseline classification |
| `blind_stable_modes_hard.csv`, `blind_stable_modes_taper.csv` | Stable tracks for real primes and all control replicates |
| `blind_consensus_spectrum.csv` | Main real-prime prediction, one sorted row per consensus mode |
| `blind_control_consensus.csv`, `blind_control_summary.csv` | Control predictions and stability comparison, including empty spectra in summary |
| `blind_sensitivity_spectra.csv`, `blind_sensitivity_summary.csv` | Frozen strict/baseline/loose predictions and overlap |
| `blind_tracking_diagnostics.csv` | Birth, death, gap, ambiguity, reconnect, and refinement events |
| `blind_feature_counts.csv`, `blind_resolution_checks.csv` | Peak density and numerical-resolution checks |
| `blind_factor_assignments.csv` | Exact prime amplitudes, assigned rates and fixed phases |
| `blind_parameters.json`, `blind_freeze.json`, `blind_summary.txt` | Reproducibility, hashes and interpretation limits |
| `blind_internal_momentum.csv` | Optional `P_int=kappa*consensus_tau`, enabled by `--kappa` |
| `evaluation_zeta_comparison.csv` | Real-prime ordered and nearest comparisons |
| `evaluation_all_comparisons.csv`, `evaluation_control_summary.csv` | All frozen variants and controls |
| `evaluation_failure_modes.csv`, `evaluation_zeta_zeros.csv` | Missing/extra/duplicate diagnostics and reference data |
| `evaluation_parameters.json`, `evaluation_summary.txt` | Evaluation settings, source hash and conclusions limited to this run |

The seven figures are:

```text
blind_peak_trajectories_hard.png
blind_peak_trajectories_taper.png
blind_stability_metrics.png
blind_consensus_spectrum.png
blind_control_spectra.png
evaluation_zeta_comparison.png
evaluation_errors.png
```

The first five contain no reference-zero markers. The comparison figure uses
separate markers with an ordered-error panel below; it does not connect predicted
and reference modes as if they were equal. `README.md` in a run is its blind
gallery; `evaluation_README.md` is the evaluation gallery. Optional dispersion
plots were omitted to keep the experiment focused. The momentum CSV uses only
the frozen prime predictions and never substitutes reference ordinates.

## Interpretation boundaries

The checked-in smoke run (P through 211, tau through 35) produced six real-prime
consensus modes. Five lie within 0.5 of the five in-window reference zeros; an
extra mode near 12.5201 comes first, so the primary ordered comparison is poor.
Phase-scrambled replicates produced 11 and 9 consensus modes; batch-shuffled
replicates produced 8 and 6. One shuffled replicate matched all five reference
zeros with a mean nearest distance similar to real primes. Thus stable finite
spectra also occur in controls, and this smoke result does not establish special
real-prime behavior. Strict/baseline/loose real counts were 6/6/7; defaults were
not changed after evaluation. All 20 first/final grid checks passed with no
refinement failures. This is a plumbing check, not a normal or deep experiment.

Git attributes preserve frozen artifact bytes, including CSV line endings, so
the manifest remains verifiable after a fresh checkout on another platform.

- Finite critical-line Euler products do not naively converge to 1/zeta.
- A stable finite-product peak spectrum does not prove RH or asymptotic convergence.
- Reference zeros are not used to generate the blind prediction.
- Controls may also produce apparently stable modes; real primes are not special by assumption.
- Feature density and uncertain track identity can create accidental structure.
- Hard/taper agreement is a robustness test, not proof.
- No physical particle or mass spectrum is established.
- Mapping tau to internal momentum is a toy interpretation.
- Even an excellent numerical match leaves the generating operator and mathematical explanation unknown.
