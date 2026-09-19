# Prime-frame zero geometry

This experiment plots inherited nontrivial zeta zeros, passed-prime factor zero
combs, and their horizontal coordinate separation. It is independent of the
earlier `prime_frame` history-suppression experiment.

The common coordinate and frame coordinate satisfy `z=(p/2)*s_p`. Thus a zero
`rho=beta+i*gamma` maps to `s_p=2*rho/p`, and **p Re(s_p)=2 beta** is invariant.
For the sampled critical-line zeros (`beta=1/2`), the inherited-zero line is
`Re(s_p)=1/p`. A hypothetical off-line zero would retain its own beta under
the same map: the implementation does not project general inputs onto this line.

Every passed prime `q<p` labels a factor `1-q^(-z)`. Its zeros map to
`s_p=i*4*pi*k/(p*ln(q))`, all on `Re(s_p)=0`. The horizontal separation between
this axis and the mapped critical line is `Delta_p=1/p`. These are factor-zero
locations, not primes represented as points. At p=2 the comb axis is only a
reference: there are no passed primes. At subsequent frames all combs share
k=0; each factor's copy of this point is retained in the data.

The shifted line follows from the coordinate transformation. This visualization
does not establish RH, a physical mass gap, or a Yang–Mills result.

## Run from the repository root (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\experiments\prime_frame_geometry\run.py
```

Defaults: frames 2, 3, 5, 7, 11; first 20 positive-height zeta zeros (plus their
conjugates in static plots); first 5 zeros tracked; comb indices -8 through 8.
The runner uses matplotlib's Agg backend, saves PNGs, and opens no windows.

Extend the frames or preserve a separate run:

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_frame_geometry\run.py --frames 2 3 5 7 11 13 17 19 --zeros 20 --track-zeros 5 --k-max 12 --output .\experiments\prime_frame_geometry\results\extended
```

Frames must be prime and strictly increasing. `--track-zeros` is capped at
`--zeros`; `--k-max 0` shows only each comb's origin. `--dps` defaults to 30
decimal digits and must be at least 15. The runner obtains zeros once with
[mpmath.zetazero](https://mpmath.org/doc/current/functions/zeta.html#zetazero),
then converts to NumPy complex128 for plotting and CSVs. Raising `--dps` does
not increase the precision of the saved float64 coordinates. Large zero counts
take longer; no research-scale sweep is required.

## Outputs

Default destination: `experiments/prime_frame_geometry/results/`, anchored to the
script directory even when run from elsewhere. Reruns overwrite matching files;
use a new `--output` directory to preserve a run (unrelated older files are not
deleted). The generated `README.md` is a gallery of that run's figures.

| File | Contents |
| --- | --- |
| `frame_2.png`, etc. | A: individual frame views with all requested zeros and comb points |
| `frame_comparison.png` | B: frames side by side with shared x/y limits |
| `zero_trajectories.png` | C: the same positive-height zeros, connected with arrows toward increasing p |
| `frame_gap.png` | D: geometric horizontal separation 1/p versus frame prime |
| `zeta_zeros.csv` | Source zero index, beta, gamma |
| `inherited_zeros.csv` | Frame, zero index, conjugate sign, real/imag coordinates, p times real part |
| `passed_prime_combs.csv` | Frame, passed prime q, integer k, real/imag coordinates |
| `frame_gaps.csv` | Frame, gap, passed-prime count |
| `parameters.json`, `summary.txt` | Reproducible settings, versions, conceptual notes, first-zero sanity table |

Single-frame views autoscale; the comparison uses common limits that contain
every requested point. The x/y scales are independent to make the small real
separations legible; screen distances/angles are not conformal geometry. Colors
for q remain consistent within a run (cycle after 20 passed primes). Open
circles of differing sizes show coincident comb points without horizontal jitter.
The trajectory plot uses the positive-height zeros and draws every selected
inherited-zero line; conjugate trajectories are their reflection.

`model.frame_geometry(p, zero_list, k_max)` returns a frame state with an
ordered inherited-zero array and a dictionary of combs keyed by q. It can feed
later animation; the CSVs likewise preserve identity across frames. No animation
dependency is required.

## Lightweight checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_frame_geometry -p "test_*.py" -v
.\.venv\Scripts\python.exe .\experiments\prime_frame_geometry\run.py --zeros 10 --track-zeros 3 --k-max 5
```

The included saved example uses this 10-zero smoke command. Checks cover the
sample mapping, critical-line real parts, the general beta invariant, comb-axis
locations and factor equations, frame gaps, recursive passed-prime lists, and
inward motion. They check implementation consistency, not a scientific claim.
