# Prime-factor phasors and zero-line combs

This small experiment extends `prime_frame`, `prime_frame_geometry`, and
`prime_frame_normalized`, reusing their finite history and coordinate helpers.
It separates exact factor cancellation from inherited zeta-zero geometry.

For `z=sigma+i*tau`, put `r=q^(-sigma)`. The two signed arrows are
`A1=1` and `A2=-r*exp(-i*tau*ln(q))`; their sum is `Fq=1-q^(-z)`.
The circle diagram instead draws the **unsigned** `q^(-z)` and its target `1`.

```text
|Fq|² = 1+r²-2r cos(tau ln(q))
      = (1-r)² + 4r sin²(tau ln(q)/2).
```

Both terms are nonnegative. A zero needs `r=1` and `tau ln(q)=2*pi*k`.
Since `q>1`, matching magnitudes forces `sigma=0`. Negative sigma does not
help: its circle has radius greater than one and misses the target as well.
At sigma=1/2 the global minimum is `(1-1/sqrt(q))² > 0`.

```text
Common:      z = 2*pi*i*k/ln(q)
Frame:       s_p = 2*z/p = 4*pi*i*k/(p*ln(q))
Normalized:  X=0, Y=4*pi*k/ln(q)
```

Larger primes have greater phase rate `ln(q)` and smaller comb spacing.
This is mathematical two-amplitude interference algebra, with no claim of
double-slit physics, quantum measurement, wave-particle duality, a physical
prime wave, an RH proof, or a mass gap.

For finite history, `|Rp|² = product_(q<p) |Fq|²`. This is **multiplication**,
not the many-amplitude sum `|sum Fq|²`; both are plotted to expose the difference.
Expanding the amplitude product is possible, but produces subset-indexed
terms with alternating signs, not a simple sum of one amplitude per prime.
On sigma=1/2 this product is exactly the earlier experiment's `W_p(tau)`;
its normalized profile is `D_p=W_p/B_p`. Thus those finite history oscillations
are precisely products of these factors. No new mechanism for zeta is inferred.

Every individual factor zero is simple. Distinct prime combs share only the
origin: a nonzero shared height would force equality of positive integer powers
of distinct primes. At the origin, the product has one zero per passed prime,
counted with multiplicity. On sigma=1/2 the finite product is strictly positive;
its global minimum is the product of the individual minima, attained at tau=0.

The inherited zeros plotted at `X=1` are critical-line samples from
`mpmath.zetazero`. A general zero `beta+i*gamma` would map to `X=2*beta`.
Factor zeros are a separate family, not inherited zeros moved onto zero.
No identical simple two-phasor mechanism is established for inherited zeros.
No infinite Euler product on the critical line is used.

## Manual run (PowerShell)

From `C:\Users\Dez\Desktop\Workspace\Primes`:

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_factor_phasors\run.py
```

Defaults: tau from -20 to 20, 2001 grid points, plus analytic zero heights;
frames 3, 5, 7, 11, 13, 17; three positive-height zeta samples for the geometry
figure only. Static phasor panels show a full cycle; no animation dependency.
The default output is `experiments/prime_factor_phasors/results/`, anchored
to the script directory. Matplotlib uses Agg and opens no windows.
Matching output files are overwritten; unrelated files are not deleted.
Use `--output` to keep a separate run:

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_factor_phasors\run.py --tau-max 30 --points 3001 --output .\experiments\prime_factor_phasors\results\wider
```

No large frame sweep is needed. A lightweight smoke run is:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_factor_phasors -p "test_*.py" -v
.\.venv\Scripts\python.exe .\experiments\prime_factor_phasors\run.py --points 401 --zeros 1 --output .\experiments\prime_factor_phasors\results\smoke
```

## Outputs and what to send back

| File | Purpose |
| --- | --- |
| `phasor_panels.png` | Signed arrows at five phases, sigma=0 versus 1/2 |
| `phasor_circles.png` | Circle radii for sigma=0, 1/2, -1/2; target at 1 |
| `single_prime_power.png` | q=2,3,5,7 curves, analytic minima and zero markers |
| `prime_comparison.png` | Phase rates and comb spacing comparison |
| `history_products.png` | Both sigma slices for six representative frames |
| `product_vs_sum.png` | Elementary powers, their product, and a different additive object |
| `zero_line_correspondence.png` | p=11 zeros beside rotated power curves with shared tau axis |
| `normalized_geometry_phasors.png` | X=0 factor combs, X=1 inherited samples, cancellation explanation |
| `frame_normalization.png` | q=2 tracked through six frames, raw heights and normalized collapse |
| `analytic_zeros.csv` | All in-window zeros for every passed prime in every history frame; all coordinates and spacings |
| `comb_spacings.csv` | Phase rate, common/normalized spacing, sigma=1/2 minima |
| `zeta_zeros.csv` | Source inherited samples |
| `parameters.json`, `summary.txt`, `README.md` | Settings, mathematical interpretation, saved image gallery |

Start by sending **phasor_panels**, **single_prime_power**, **history_products**,
**zero_line_correspondence**, and **normalized_geometry_phasors**, together with
`summary.txt` and `parameters.json`. Add `product_vs_sum` to discuss multiplication.

Analytic zero markers are not numerical zero detection. Floating-point evaluation
can leave tiny residuals at those locations. Inserting them into the sampling grid
helps show narrow zero dips; higher resolution may still reveal more detail between
them. History panels autoscale; sigma=1/2 uses a log scale to distinguish small
positive values from zeros. No power normalization or clipping is applied.
CSV rows preserve each factor's coincident origin. The normalized geometry figure
uses independent x/y scales and actual coordinates without jitter.
