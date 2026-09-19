# Finite prime-product features versus Riemann-zero heights

This follows [prime-factor phasors](../prime_factor_phasors/README.md) and
[prime frames](../prime_frame/README.md). It asks whether objectively detected
features of the finite history product stabilize near actual zeta-zero ordinates,
and whether scrambled controls show comparable alignments. It assumes neither result.

The cutoff here is inclusive: `q <= P`. Earlier frame helpers use `q < frame_prime`.
The experiment reuses the existing sieve, mpmath zero generator, and single-factor
power helper. No earlier experiment is changed.

## Run manually from PowerShell

From `C:\Users\Dez\Desktop\Workspace\Primes`, install the pinned dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Normal run: eight cutoffs through 10007, 12001 samples on `[0,60]`, 20 generated
zeros, five replicates per control, 2x resolution checks at three cutoffs:

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_zero_features\run.py --output .\experiments\prime_zero_features\results\normal
```

Deeper run (manual only; substantially more computation and output):

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_zero_features\run.py --cutoffs 5 11 29 101 331 997 5003 10007 30011 100003 --tau-max 85 --points 34001 --zeros 20 --control-replicates 10 --resolution-factor 4 --buildup-cutoff 10007 --buildup-points 1201 --buildup-zeros 1 2 --taper --output .\experiments\prime_zero_features\results\deep
```

The deeper window includes all 20 generated zeros. Increasing `--zeros` does not
automatically extend the main grid. Outside-window rows retain exact L, slope,
curvature and phase evaluations but leave nearest-feature fields blank.

Development smoke run (only small three-cutoff sweeps were executed while implementing this experiment):

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_zero_features\run.py --cutoffs 5 29 101 --points 3001 --zeros 20 --control-replicates 2 --buildup-cutoff 101 --buildup-points 301 --taper --output .\experiments\prime_zero_features\results\smoke
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_zero_features -p "test_*.py" -v
```

Paths default to this experiment's `results/normal` independent of the working
directory. Explicit relative `--output` paths resolve from your working directory.
Rerunning replaces files of the same name. Use a new directory to preserve a run.
There are no GUI windows or animations to keep open.

The initial 1501-point smoke grid failed two count checks for the shuffled-rate
control at P=29 (42 versus 43 peaks/minima). The saved smoke run uses 3001 points
and checks against 6001. This resolution change was prompted by a detection-count
mismatch, not by alignment with zeta zeros. Thresholds and seeds were unchanged.

## Model and feature rules

The main signal is

`L_P(tau) = sum_(q<=P) log((1-1/sqrt(q))^2 + 4/sqrt(q)*sin(tau*ln(q)/2)^2)`.

This equals `log|product(1-q^(-1/2-i*tau))|^2`. The positive sine-squared
form avoids subtractive cancellation. Evaluation uses bounded prime/time chunks.
The NPZ also saves `exp(L)` wherever float64 can represent it; unrepresentable
values are NaN, with the full log signal retained. Complex multiplication is used
only in small unit tests. `-L` can be read from the saved arrays as the log squared
magnitude of the **reciprocal finite history product**; it must not be called zeta.

`scipy.signal.find_peaks` is applied to L and -L before comparing any zero heights.
Default minimum prominence is **0.1 log units**, not chosen from observed alignments.
Set `--prominence 0` to retain all sampled extrema; `--min-width` is a lower bound
on half-prominence width in tau units (default zero). Every dataset has the same
thresholds. Prominence and width are sampled estimates, not refined analytic
quantities. Width conventions follow [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).
No zeros are inserted into the detection grid and endpoints are excluded.

Refinement solves the analytic L' root within the grid-neighbor bracket, using
bounded scalar optimization if the derivative does not bracket the right kind of
extremum. Each row includes the bracket, method, slope, curvature and success flag.
Refinement warnings remain visible in the tables; do not treat those runs as
validated. Peak/minimum heights are values of the original L, including for minima.

Nearest-feature trajectories can jump between distinct extrema. Extrema also get
heuristic track IDs from same-kind mutual-nearest matching between adjacent cutoffs,
with `--track-max-jump 0.5` in tau units. These IDs are independent of zero heights.
Crosses in the trajectory plot mark a change in nearest track. Sparse cutoff
matching does **not** establish continuous branch identity. Inspect denser cutoffs
before describing a persistent branch or monotone approach.

All sampled zeros get individual absolute-offset panels; the first six in-window
zeros get location trajectories (`--trajectory-zeros`). Missing features stay blank.
A `*_bracketed` flag indicates whether detected features of that kind exist on both
sides of gamma, avoiding an unacknowledged boundary-censoring effect.

## Controls and descriptive summaries

- **Phase scrambled:** keep each prime amplitude/rate, add one uniform phase in
  `[-pi,pi)` per prime. Seeded prefixes are fixed across cutoffs.
- **Shuffled rates:** keep amplitudes in prime order, permute exactly the available
  `ln(q)` values at each cutoff. A seed derived from the base seed, replicate and
  cutoff makes this reproducible and order-independent. These permutations change
  with cutoff; they are not trajectories of a nested control product.
- **Optional raised cosine:** `--taper` adds weights
  `w(q/P)=(1+cos(pi*q/P))/2` for `q<=P`, zero beyond P, to both log magnitude and
  summed phase. This attenuates all included primes and vanishes at the cutoff.
  It is a separate sensitivity diagnostic, not a regularization or zeta approximation.

Default seed is 1729 (`--seed`). Both controls have the same prime counts and
frequency sets as the real signal at every cutoff. Replicates are all reported;
replicate 0 is chosen in advance for profile illustrations and resolution checks.

`control_summary.csv` reports mean/median nearest distances and feature counts for
each replicate, cutoff and kind. It includes `all_in_window` and `common_bracketed`
cohorts. The latter uses the intersection of bracketed zero indices across all
datasets/replicates at that cutoff (including the optional taper if enabled).
Thus datasets in a panel use the same zero targets. Cohorts can change with cutoff;
their indices and matched/eligible counts are saved. Empty cohorts yield NaN,
never false zero distances. Plots show replicate medians with full replicate ranges,
not confidence intervals. The signed-offset ECDF pools descriptive measurements at
the final cutoff; they are not independent observations for hypothesis testing.
No null p-values, fitted thresholds, or selected best seeds are reported.

Higher feature density alone can shorten nearest distances; inspect counts and
the controls as well as absolute offsets. Smaller offsets across a finite set of
cutoffs do not prove an asymptotic limit.

## Phase and resolution

Every factor has positive real part on this line. Its principal argument is
continuous; summing those arguments supplies an **already unwrapped** phase.
Wrapping the sum then applying `unwrap` could introduce coarse-grid aliases.
The phase anchor is the actual sum (zero at tau=0 for real primes); controls are
not arbitrarily shifted. Analytic derivatives give L', L'' and Phi'.

Phase summaries record exact Phi and Phi' at every gamma, nearest stationary-phase
roots, nearest sampled maxima of absolute rotation rate, and the percentile of
`abs(Phi'(gamma))` among grid rotation rates. These are descriptive diagnostics,
not significance levels. Rotation maxima are grid locations, explicitly labeled.
No universal phase value/crossing is assumed; raw phase arrays support follow-up.

At up to three evenly selected cutoffs, real/taper and replicate 0 of both controls
are recomputed at `--resolution-factor 2` or `4` times as many intervals. Counts
must agree and all sorted refined locations must agree within
`--resolution-tolerance 1e-5`; refinements must also pass. A count mismatch fails
even if the surviving features align perfectly. Empty detections cannot pass.
Checks cover the whole profile and both feature kinds. Read failures before drawing
conclusions, and rerun at higher resolution if needed. This is a numerical check,
not a guarantee that all fine structure has been sampled.

## Outputs and what to send back

| File | Contents |
| --- | --- |
| `profiles.png` | Raw L profiles with zero markers and shared x/y limits, no clipping |
| `trajectories.png` | Nearest peak/minimum locations and heuristic track switches |
| `offsets.png` | Absolute offsets for every in-window zero, log cutoff axis |
| `heatmap_raw.png`, `heatmap_centered.png` | Raw L and L minus each row's mean; no row variance normalization |
| `phase.png` | Continuous phase and analytic phase derivative |
| `controls.png`, `control_profiles.png`, `control_offsets.png` | Descriptive distance summaries, replicate-0 profiles and signed-offset distributions |
| `buildup.png`, `buildup.npz` | Every prime added in selected zero neighborhoods; prime-index rows, no primes skipped |
| `zeta_zeros.csv` | All mpmath-generated ordinates and main-window flags |
| `extrema.csv` | All datasets' features, prominence, width, height, refinement diagnostics, track IDs |
| `nearest_features.csv` | Real-product nearest features, offsets, exact gamma evaluations and switches |
| `control_results.csv`, `control_summary.csv` | Control/taper comparisons and descriptive real/control aggregates |
| `phase_summary.csv`, `phase_events.csv` | Gamma phase diagnostics and zero-blind phase events |
| `resolution_checks.csv` | Counts, maximum refined-position shifts and pass/fail flags |
| `profiles.npz` | Tau/cutoff/gamma arrays and every dataset's L, power, phase and derivatives |
| `parameters.json`, `summary.txt`, `README.md` | Reproducibility settings, guardrails, warnings and gallery |

Buildup defaults to gamma_1 +/-2 through the smaller of 997 and the largest main
cutoff. Use `--buildup-zeros 1 2 3` for further neighborhoods. Those local windows
are explicitly separate from the main detection window. The NPZ has one row per
successive prime, suitable for animation. Raw shared color scales are retained.

**Send back:** `trajectories.png`, `offsets.png`, `controls.png`, `buildup.png`,
`nearest_features.csv`, `control_summary.csv`, `resolution_checks.csv`,
`parameters.json`, and `summary.txt`. Include `extrema.csv`, `control_results.csv`,
`phase.png` and `phase_summary.csv` for branch/phase investigation. The full results
folder is also suitable. The included smoke gallery is a plumbing check, not a result.

## Mathematical guardrails

The ordinary finite Euler products do **not** naively converge to `1/zeta` on
`Re(z)=1/2`. Any alignment must be measured. Zeta zeros are collective global
features; an individual prime does not generate an individual Riemann zero.
Factor phasors have interference-like algebra, but their product is multiplicative,
not a standard many-slit amplitude sum. This experiment is not evidence for quantum
mechanics, a mass gap, or the Riemann hypothesis. A feature near 14.1347 is interesting
only if stable across cutoffs and better aligned than reasonable controls.

The first 20 ordinates are generated using the existing `mpmath.zetazero` helper
at 30 decimal digits, then exported/evaluated in float64. A larger precision setting
does not change the float64 accuracy of the product calculations.
