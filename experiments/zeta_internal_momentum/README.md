# Zeta-quantized internal momentum in KG / Dirac

This experiment imposes `P_int,n = kappa * gamma_n` as a toy spectrum of internal
invariant momentum and inserts it into the ordinary Klein–Gordon and Dirac mass
slot, `P_int = m*c`. It does not change those equations. **At external rest,
`p_ext = 0`, internal invariant momentum can remain nonzero**, with
`E_n(0) = c*P_int,n` (equal to `P_int,n` in natural units).

The first nonzero ordinate is approximately `14.134725141734695`. With the
reference `P_0=0`, the **toy internal momentum gap** is `kappa*gamma_1`.
This is an imposed gap above a reference, not a derived physical vacuum gap.
The reference is included in the ladder CSV and as a dashed dispersion curve;
it is excluded from the zeta operator and mode calculations.

## Run manually in PowerShell

From the repository root, using its existing environment and requirements:

```powershell
.\.venv\Scripts\python.exe .\experiments\zeta_internal_momentum\run.py
```

This computes 8 modes, 1001 external momentum samples on `[-50,50]`, and four
directional states at mode 1, with `c=hbar=kappa=1`. It saves figures without
opening windows in `experiments/zeta_internal_momentum/results/normal/`.
Default paths are anchored to the script; relative explicit paths are relative
to the current working directory. A rerun replaces files with the same names.
No prime-product sweep is performed; existing CSVs are only read.

```powershell
# Configurable scale, units, mode, momentum range, and output:
.\.venv\Scripts\python.exe .\experiments\zeta_internal_momentum\run.py --zeros 10 --kappa 0.5 --c 2 --hbar 0.8 --p-max 40 --points 801 --directional-mode 3 --output .\experiments\zeta_internal_momentum\results\scaled

# Explicit bridge to an earlier finite-product run:
.\.venv\Scripts\python.exe .\experiments\zeta_internal_momentum\run.py --prime-features .\experiments\prime_zero_features\results\normal\nearest_features.csv

# Lightweight smoke run (the included gallery uses these settings):
.\.venv\Scripts\python.exe .\experiments\zeta_internal_momentum\run.py --zeros 5 --points 101 --p-max 30 --output .\experiments\zeta_internal_momentum\results\smoke

# Lightweight numerical tests:
.\.venv\Scripts\python.exe -m unittest discover -s experiments/zeta_internal_momentum -p "test_*.py" -v
```

`--zeros >= 1`, `--dps >= 15`, and positive finite `--kappa`, `--c`, `--hbar`,
`--p-max` are required. `--points` must be odd and at least 3 to include zero
exactly. Zeros are computed with `--dps` decimal precision (default 30), then
converted to float64 by the shared helper. Outputs/tests use float64 precision.
`--no-prime-features` disables the optional bridge.

## Equations and interpretation

With metric signature `(+---)`, `x^0=c*t`, and
`Box = c^(-2)*partial_t^2 - laplacian`, the equations are

```text
[Box + P_int,n^2 / hbar^2] phi_n = 0
[i hbar c gamma^mu partial_mu - c P_int,n] psi_n = 0

P_int,n = kappa gamma_n = m_n c
E_n^2 = p_ext^2 c^2 + P_int,n^2 c^2
      = p_ext^2 c^2 + m_n^2 c^4
```

Here `gamma^mu` denotes the usual Dirac matrices; `gamma_n` denotes the zeta-zero
ordinate. They are unrelated uses of gamma notation. Since `gamma_n` is
dimensionless, `kappa` supplies momentum units. `hbar` enters the wave equation
coefficient and the energy/momentum-to-frequency/wavenumber conversion; it does
not enter this dispersion expressed in energy and momentum. Choosing consistent
physical units requires supplying consistent `kappa`, `c`, and `hbar`.

The standard KG energy is evaluated independently as
`sqrt(p_ext^2*c^2 + (P_int/c)^2*c^4)` and compared with the internal-momentum
energy. The signed residual and its maximum absolute value are saved. Equality
up to floating-point roundoff is expected algebraically.

The numerical Dirac check constructs a Hermitian 4 by 4 Hamiltonian
`H = c*alpha_x*p_ext + beta*c*P_int,n`, with the conventional block matrices.
For up to three representative modes and five signed momenta including zero,
`eigvalsh(H)` is compared with `[-E,-E,+E,+E]`, retaining both degeneracies.
Dirac supplies its usual spinor structure; zeta quantization does not derive
spin. The negative branches carry only their standard Dirac mathematical meaning.

## Directional states and no double counting

For nonnegative directional content, define

```text
A = integral rho(n_hat) dOmega
B = integral rho(n_hat) n_hat dOmega
P_int^2 = A^2 - |B|^2
```

The helper stores **integrated weights**: `w_j=rho_j*dOmega_j` for smooth
quadrature samples, or the weights of delta measures for beams. Thus
`A=sum(w_j)` and `B=sum(w_j*n_hat_j)`. The smooth quadrature uses eight
Gauss–Legendre nodes in `cos(theta)` and sixteen equally spaced azimuths; it
integrates the moments of these constant/linear densities to roundoff.

The four shapes are:

- Isotropic constant density on the sphere.
- Equal antipodal beams along `+x` and `-x` (singular measures).
- Six axis beams with weights `(1,2,3,1,2,3)`, giving three balanced pairs.
- Smooth positive dipole `rho=1+0.6*n_x`, whose unnormalized moments are
  `A=4*pi`, `B=(0.8*pi,0,0)`, and `|B|/A=0.2`.

Each is divided by its initial positive invariant magnitude to produce a shape
`f` with `A_f^2-|B_f|^2=1`. Scaling by `lambda_n=kappa*gamma_n` gives
`rho_n=lambda_n*f` and invariant `lambda_n^2`. A single unbalanced null beam has
zero invariant and cannot be normalized this way; the helper rejects it.

The nonzero-B example is a comparison of directional configurations with the
same invariant, **not a claim that all four are rest-frame configurations of one
total four-momentum**. If `(A,B)` is identified with `(E/c,p)` of that same object,
then `B` is its spatial momentum, and its rest frame requires `B=0`.
Here the directional construction supplies the invariant parameter; the
dispersion sweep separately selects `p_ext`. We never add `|B|^2` to that
dispersion, nor equate every directional `A` with the external-rest energy.
The smooth dipole consequently has `A>P_int` while the balanced states have
`A=P_int`. Shared invariant magnitude does not specify all directional geometry.

Allowing `rho` alone does not quantize anything: rescaling it continuously
rescales the invariant continuously. The directional sphere represents
structure, while **the separate zeta rule imposes allowed magnitudes**.

## Abstract operator and product space

Define the finite toy operator directly as
`P_hat_int=diag(kappa*gamma_1,...,kappa*gamma_N)`. Its eigenvectors `chi_n` are
the coordinate basis and its square has eigenvalues `kappa^2*gamma_n^2`.
The saved matrices and coefficient table make this definition explicit.

```text
Phi(x,xi) = phi_n(x) chi_n(xi)
P_hat_int chi_n = kappa gamma_n chi_n
[Box_x + P_hat_int^2/hbar^2] Phi = 0
    => [Box_x + kappa^2 gamma_n^2/hbar^2] phi_n = 0
```

`xi` labels an abstract internal state, not an asserted extra spatial dimension.
This separation-of-variables construction is a finite diagonal definition, not
a first-principles derivation, a Hilbert–Pólya operator, or a dynamics for `rho`.
The directional samples illustrate realizations of the invariant; they are not
proved eigenfunctions of a differential operator on the sphere.

## Integration with existing experiments

The code reuses `prime_frame_geometry.model.first_zeta_zeros`, also used by the
normalized frames and finite-product feature experiment. It calls
`mpmath.zetazero`; finite-prime peaks never substitute for these zeros.
The [mpmath documentation](https://mpmath.org/doc/current/functions/zeta.html#zetazero)
describes the generator. Finite sampling does not establish RH.

`prime_frame*` supplies coordinate and normalization context;
`prime_factor_phasors` supplies the finite-product interference analogy.
`kg_worldline` studies a different time-dilation calculation and
`clifford_momentum` studies ordered quaternion evolution. Neither exposes this
sphere-invariant helper or a free Dirac matrix, so those implementations remain
unchanged and the new helpers live locally here.

If no CSV is specified, the runner tries
`prime_zero_features/results/normal/nearest_features.csv`, then the saved
`results/smoke/nearest_features.csv`. Missing files skip the bridge gracefully.
An explicit missing path does not silently fall back; a malformed existing file
reports an error before output is written. The exact source path and SHA-256
are saved in `parameters.json`, so smoke data cannot masquerade as a new sweep.

Only `dataset=real`, `replicate=0` entries for selected modes are mapped.
Source gamma values are checked against the independently generated modes;
`gamma_error=tau_peak-gamma_n` and `momentum_error=kappa*gamma_error`.
Missing/out-of-window peaks stay blank rather than becoming zero errors.
Bracket and track metadata are retained. The plot shows at most six modes;
the CSV retains all matching modes. Unbracketed peaks get a different marker.
Nearest features may switch identity, and offsets need not shrink monotonically.
Connecting guides do not imply continuity, convergence, or physical causation.
The ordinary Euler product is not assumed to converge on the critical line.

## Outputs and what to send back

Open the [included smoke gallery](results/smoke/README.md) or the generated
`results/normal/README.md`. Every run writes:

| File | Contents |
| --- | --- |
| `zeta_internal_modes.csv` | Baseline, true gamma, kappa, P_int, its square, previous gap, gap from zero, reference flag |
| `dispersion.csv` | Mode, gamma, p_ext, P_int, E, mass equivalent P_int/c, ordinary KG energy, signed energy residual |
| `directional_states.csv` | A, all B components, |B|, P_int, target, signed/relative magnitude residual, squared invariant and its residual |
| `dirac_eigenvalues.csv` | Expected ±E, all four computed eigenvalues as a JSON list, maximum absolute error |
| `internal_operator.csv`, `internal_operator.npz` | Eigenvalues, squared eigenvalues, KG coefficients, and both diagonal matrices |
| `parameters.json`, `summary.txt`, `README.md` | Settings, provenance, maximum numerical errors, and figure gallery |
| `prime_peak_mapping.csv` (optional) | True modes versus existing finite-product peak positions and reliability metadata |

The five required PNGs are `zeta_internal_momentum_spectrum.png`,
`dispersion_branches.png`, `directional_states_same_invariant.png`,
`kg_equivalence.png`, and `dirac_spectrum_check.png`. The optional bridge adds
`prime_peak_to_internal_mode.png`. Dot sizes in the sphere panels represent
integrated weights normalized **within each panel**, not a shared density scale
between smooth densities and delta beams.

For interpretation, send **`summary.txt`, `parameters.json`, the spectrum and
dispersion PNGs, `directional_states_same_invariant.png`, and
`directional_states.csv`**. Include `dirac_eigenvalues.csv` and the KG/Dirac check
PNGs to examine numerical equivalence. If present, also send the prime mapping
CSV/PNG and identify whether its source was a smoke or a normal run.

This model treats mass as invariant internal momentum by choice. It does not
claim that zeros literally are particle masses, that primes physically generate
particles, that RH is proved, or that a physical mass gap, Yang–Mills theory,
quantum gravity, spin, or the Hilbert–Pólya operator has been derived.
