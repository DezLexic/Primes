# Normalize prime frames by their own gap

This experiment builds on `prime_frame_geometry` and reuses its raw coordinate
maps, passed-prime sieve, and mpmath zero source. It measures both axes in units
of the frame gap `Delta_p=1/p`:

```text
s_p = 2z/p
(X, Y) = (p Re(s_p), p Im(s_p)) = (2 Re(z), 2 Im(z))
Inherited rho=beta+i*gamma: (X,Y) = (2 beta, 2 gamma)
Passed-prime factor 1-q^(-z), q<p: (X,Y) = (0, 4*pi*k/ln(q))
```

The sampled critical-line zeros have beta=1/2, so they remain at X=1 and
Y=2*gamma across all frames. Each already-passed factor's comb also stays fixed.
When moving between successive prime frames, the previous frame prime adds one
new factor-zero comb. Skipping intermediate prime frames would add multiple combs.
At p=2 the X=0 reference axis has no factor zeros.

The collapse is expected from the coordinate transformation and is **not new
evidence for RH**. These samples do not establish the location of all zeros.
The inherited nontrivial spectrum belongs to the collective zeta system; no
individual inherited zero is assigned to a prime. Primes label factors, rather
than literally moving to X=0. This is exploratory coordinate geometry with no
physical or mass-gap interpretation.

An artificial point with beta=0.4 and the first sampled zero's height is carried
through the same map. It stays at X=0.8, with relative offset
`(Re(s_p)-1/p)/(1/p) = 2 beta-1 = -0.2`. It is not a claimed zeta zero.
The transformation does not force off-line inputs onto the mapped critical line.

## Run from the repository root (PowerShell)

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_frame_normalized\run.py
```

The default is a small run: 10 positive-height zeta zeros, their conjugates, and
comb indices -5 through 5. Plot A uses p=2,3,5,7,11; the overlays and diagnostics
use p=2,3,5,7,11,13,17,19; history uses p=3,5,7,11,13. These representative frame
sets are fixed to keep the figures consistent. There is no long sweep.

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_frame_normalized\run.py --zeros 5 --k-max 3 --output .\experiments\prime_frame_normalized\results\small
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_frame_normalized -p "test_*.py" -v
```

`--zeros` must be at least 1; `--k-max 0` retains only the origin of each comb;
`--dps` defaults to 30 and must be at least 15. Samples are computed once using
the existing model, then converted to complex128. Output coordinates use float64.
The first five requested zeros (or fewer when `--zeros < 5`) are tracked.

## Outputs and plot conventions

Results default to `experiments/prime_frame_normalized/results/`, anchored to
the script's directory regardless of working directory. The Agg backend saves
PNGs without opening windows. Reruns replace matching files; use `--output` to
preserve another run. Unrelated files are not deleted. No new dependency is needed.

| File | Contents |
| --- | --- |
| `original_vs_normalized_p2.png`, p3, p5, p7, p11 | A: five pairs of raw and normalized frame panels |
| `normalized_inherited_overlay.png` | B: inherited zero centers coincide across eight frames |
| `normalized_comb_overlay.png` | B: a panel per q overlays all frames containing that comb |
| `progressive_history.png` | C: fixed inherited zeros and combs, with each new comb emphasized |
| `off_line_diagnostic.png` | Raw inward movement and normalized X=0.8 artificial point beside X=1 samples |
| `invariants.csv` | Raw and normalized coordinates, independent expectations, and residuals for all sources |
| `tracked_zeros.csv` | First five positive-height zeros through all eight frames |
| `zeta_zeros.csv` | Common-coordinate source zeros, indices, beta, gamma |
| `summary.txt`, `parameters.json`, `README.md` | Numerical sanity table, checks, settings/versions, and plot gallery |

Both axes are multiplied by p using the actual raw coordinates. Raw panels have
shared limits; normalized panels have shared limits. All selected heights fit
in the main geometry plots. Independent x/y display scales make horizontal
separation legible, so screen angles and distances should not be interpreted as
conformal geometry. Colors identify q in A/C and frames in B. Nested open markers
expose coincident centers without jitter. Each factor's k=0 row is retained.
The diagnostic shows positive-height samples; full geometry plots include
conjugates. Truncated combs are finite samples of infinite factor-zero combs.

For a zoom comparison, **normalized p=11 matches normalized p=2**, not raw p=2.
Raw p=2 equals z; normalized coordinates equal 2z. The optional proposed raw-p=2
alignment would therefore miss a factor of two. Plot A shows the correct scaling.

The runner checks every diagnostic row with `rtol=2e-14, atol=2e-14` and reports
the maximum absolute residual. Mathematical coincidence may differ by float64
rounding. Tests check general beta and both coordinate invariants, the RH sample
case, cross-frame agreement, comb equations/heights, unchanged history plus one
new comb per successive prime, the artificial offset, origin multiplicity, and
failure of the diagnostic check when an output is perturbed.
