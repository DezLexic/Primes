# Recursive prime-frame filtering

**Question:** Does recursive prime-frame filtering produce measurable, structured
low-spectrum depletion, and how does its characteristic scale transform as the
frame advances?

## Construction

For prime frame p, use the common coordinate `z=(p/2)s_p` and finite history
`R_p(z)=product_(prime q<p)(1-q^(-z))`. The recursion is
`R_next(z)=(1-p^(-z))*R_p(z)`. On `z=1/2+i*tau`, compute

```text
W_p(tau) = product_(q<p) [1 + 1/q - 2/sqrt(q)*cos(tau*ln(q))]
B_p      = product_(q<p) (1 + 1/q)
D_p(tau) = W_p(tau)/B_p
chi_p    = sum_(q<p) q^(-1/2)*ln(q)^2 / (1-q^(-1/2))^2
tau_quad = 1/sqrt(chi_p)
t_quad   = 2/(p*sqrt(chi_p))
```

`B_p` is the independent-phase average (also the infinite common-coordinate mean
for a fixed finite product); a finite plotting window need not have this average.
Products accumulate in log space. The equivalent positive factor
`(1-1/sqrt(q))^2 + 4/sqrt(q)*sin(tau*ln(q)/2)^2` reduces cancellation.
The quadratic width measures an order-one local change of log W, not the width of
an interval of zero spectrum. At p=2 the product is empty and both widths are infinite.

No zeta evaluation is needed. `H_p=R_p*zeta` is a formal motivation, and its remaining
Euler product is valid only for Re(z)>1, never used numerically on the critical line.

## Run (PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_frame\run.py --max-prime 1000 --tau-max 20 --points 4001
# Larger manual comparison; preserve the first run:
.\.venv\Scripts\python.exe .\experiments\prime_frame\run.py --max-prime 10000 --tau-max 30 --points 6001 --output .\experiments\prime_frame\results\up_to_10000
```

`--frames 3 5 7 11 29 101` chooses representative profile curves. Defaults use those
frames within the requested upper bound; every successive prime frame still goes
into the CSV and heatmap. The symmetric tau grid must have an odd number of points
so zero is sampled exactly. Increase resolution to check features; a narrow tau
window can conceal oscillations. Runtime and profile storage scale with the number
of frames times the number of tau samples (about 59 MB of raw profiles for 1229 by 6001).

## Saved outputs

Open the [results gallery](results/README.md) without running Python.
The included initial example is only a smoke run (`--max-prime 101 --points 801`).

- `frames.csv`: discarded count, W(0), log W(0), B, log B, D(0), log D(0), chi,
  both coordinate widths, the PNT ratio, and the scaled width.
- `profiles.npz`: tau, frame primes, and natural-log normalized profiles for later plotting.
- `profiles.png`, `profiles_log.png`: representative profiles on linear and log10 scales.
- `central_suppression.png`, `chi.png`, `tau_quad.png`, `t_quad.png`: frame diagnostics.
- `heatmap.png`: log10 D, one equally spaced row per successive prime frame.
- `asymptotic_diagnostics.png`: chi/(2 sqrt(p) ln(p)) and tau_quad*p^(1/4)*sqrt(ln(p)).
- `summary.txt`, `parameters.json`: numerical summary and reproducible settings.

Send `frames.csv`, `summary.txt`, `parameters.json`, the heatmap, and the asymptotic
and width plots back for interpretation after a manual sweep. Raw powers can
underflow for much larger frames; the logarithmic columns remain the reference.
The brief's rough W(0) values are not used as expected values; the actual products
are reported for direct comparison. The CLI also compares analytic and finite-difference
curvature for representative frames.

The PNT estimate `chi ~ 2 sqrt(p) ln(p)` suggests a scaled-width reference `1/sqrt(2)`.
These are plotted as diagnostic references, without fitting or assuming convergence.
Inspect wider frame ranges and numerical resolution before interpreting trends;
a smoke test is not scientific evidence for a limiting scale.

## Interpretation boundaries

Primes enter through Euler factors and phases ln(q); they do not sit on a critical
line. The 1/p lines are deliberately created by the coordinate map. Finite history
factors are strictly positive here, so suppression alone is not a mass gap.
No mass, absolute vacuum-energy baseline, Dirac, KG, or Yang-Mills model is introduced.
The code is intended to expose collapsing, growing, or potentially stable scales
without forcing a physical analogy or proving a statement about zeta zeros.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_frame -p "test_*.py" -v
```

Tests check the sieve, recursion, complex/cosine/log products, central product,
curvature, coordinate mapping, and incremental sweep consistency.
