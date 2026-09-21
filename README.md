# Prime and wave experiments

Independent numerical experiments, with a shared Python environment at the repository root.
Each experiment keeps its implementation, explanation, and saved results together:

```text
Primes/
  .venv/                         # shared local environment (not committed)
  requirements.txt
  experiments/
    prime_wave/
      prime_wave.py
      README.md
      results/
    kg_worldline/
      kg_sandbox.py
      README.md
      results/
    clifford_momentum/
      clifford_momentum_sandbox.py
      README.md
      results/
    prime_frame/
      model.py
      plotting.py
      run.py
      test_model.py
      README.md
      results/
    prime_frame_geometry/
      model.py
      plotting.py
      run.py
      test_model.py
      README.md
      results/
    prime_frame_normalized/
      model.py
      plotting.py
      run.py
      test_model.py
      README.md
      results/
    prime_factor_phasors/
      model.py
      plotting.py
      run.py
      test_model.py
      README.md
      results/
    prime_zero_features/
      model.py
      plotting.py
      run.py
      test_model.py
      README.md
      results/
    zeta_internal_momentum/
      model.py
      plotting.py
      run.py
      test_model.py
      README.md
      results/
```

## Setup (PowerShell)

From this directory (`C:\Users\Dez\Desktop\Workspace\Primes`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The local environment has already been recreated here using the installed numerical
packages. These commands are for a fresh checkout or rebuilding it. Activation is optional;
using the environment's Python directly avoids PowerShell activation-policy issues.

## Experiments

| Experiment | Question | Saved results |
| --- | --- | --- |
| [Prime wave](experiments/prime_wave/README.md) | What does a smooth logarithmic clock capture or miss about prime counts? | [Plot](experiments/prime_wave/results/figure.png) |
| [KG worldline](experiments/kg_worldline/README.md) | How does a worldline clock stretch under momentum and constant force? | [Plot](experiments/kg_worldline/results/figure.png) |
| [Clifford momentum](experiments/clifford_momentum/README.md) | What observed component arises from ordered rotations of a hidden generator? | [Plot](experiments/clifford_momentum/results/figure.png) |
| [Prime frame](experiments/prime_frame/README.md) | How does finite prime-history filtering suppress the center and change local widths? | [Results gallery](experiments/prime_frame/results/README.md) |
| [Prime-frame geometry](experiments/prime_frame_geometry/README.md) | Where do inherited zeta zeros and passed-prime factor zeros lie in each prime frame? | [Results gallery](experiments/prime_frame_geometry/results/README.md) |
| [Normalized prime frames](experiments/prime_frame_normalized/README.md) | Which geometry stays fixed after measuring both axes in units of the frame gap? | [Results gallery](experiments/prime_frame_normalized/results/README.md) |
| [Prime-factor phasors](experiments/prime_factor_phasors/README.md) | Why do individual factors cancel on the zero line, and how do their intensities multiply? | [Smoke gallery](experiments/prime_factor_phasors/results/smoke/README.md) |
| [Finite-product zero features](experiments/prime_zero_features/README.md) | Do finite-product extrema stabilize near zeta-zero heights, beyond scrambled controls? | [Smoke gallery](experiments/prime_zero_features/results/smoke/README.md) |
| [Zeta internal momentum](experiments/zeta_internal_momentum/README.md) | What does an imposed zeta invariant-momentum spectrum look like in ordinary KG/Dirac dispersion? | [Smoke gallery](experiments/zeta_internal_momentum/results/smoke/README.md) |
| [Blind prime spectrum](experiments/blind_prime_spectrum/README.md) | Can prime-only peak tracks predict a stable spectrum before loading any zeta zeros? | [Blind smoke gallery](experiments/blind_prime_spectrum/results/smoke/README.md) / [Evaluation](experiments/blind_prime_spectrum/results/smoke/evaluation_README.md) |

## Run and revisit

```powershell
.\.venv\Scripts\python.exe .\experiments\prime_wave\prime_wave.py --no-show
.\.venv\Scripts\python.exe .\experiments\kg_worldline\kg_sandbox.py --no-show
.\.venv\Scripts\python.exe .\experiments\clifford_momentum\clifford_momentum_sandbox.py --no-show
.\.venv\Scripts\python.exe .\experiments\prime_frame\run.py --max-prime 1000
.\.venv\Scripts\python.exe .\experiments\prime_frame_geometry\run.py
.\.venv\Scripts\python.exe .\experiments\prime_frame_normalized\run.py
.\.venv\Scripts\python.exe .\experiments\prime_factor_phasors\run.py
.\.venv\Scripts\python.exe .\experiments\prime_zero_features\run.py
.\.venv\Scripts\python.exe .\experiments\zeta_internal_momentum\run.py
```

The first three commands always save `figure.png`, `figure.txt` (summary), and
`figure.json` (settings) in their own `results/` folder. Omit `--no-show` to also
open a matplotlib window. `--save PATH` changes the image destination and writes
the summary/settings beside it. Prime frame saves a CSV, PNG plots, profile arrays,
settings, and a summary; it does not open windows.
Prime-frame geometry saves individual frame plots, a comparison panel, zero
trajectories, a horizontal-separation plot, coordinate CSVs, and run settings.
Its saved example uses 10 zeros; its default run uses 20.
Normalized prime frames reuse the geometry model and save raw/normalized pairs,
overlays, progressive history, an artificial off-line diagnostic, and invariant
CSVs. The default is a lightweight 10-zero run across eight representative frames.
Prime-factor phasors add signed-arrow panels, circle geometry, single-factor and
finite-product power curves, analytic zero CSVs, and the normalized zero-line
connection. The included results are a 401-point smoke run; the manual command
uses 2001 points. See its README for interpretation boundaries and output details.
Finite-product zero features compare independently detected peaks and minima with
mpmath zero ordinates, fixed-phase and shuffled-frequency controls, and resolution
checks. The included three-cutoff smoke run is only a plumbing check. Normal and
deep sweeps are manual; [its README](experiments/prime_zero_features/README.md)
lists exact commands, CSV definitions, phase diagnostics, and interpretation limits.
No naive critical-line Euler-product convergence is assumed.

Zeta internal momentum reuses the true zero generator, imposes `P_int,n=kappa*gamma_n`,
and checks ordinary KG/Dirac equivalence with `P_int=m*c`. Four directional states
share one invariant; existing finite-product peaks provide an optional comparison.
Its included results use five modes and 101 momentum samples. The manual default
uses eight modes and writes to `experiments/zeta_internal_momentum/results/normal/`.
This is an imposed toy quantization rule, not a derived physical mass gap.

Blind prime spectrum independently tracks hard-cutoff and tapered peaks, applies
fixed stability rules to real primes and seeded controls, and freezes all
predictions before loading zeros for evaluation. Only its smoke example was run;
normal/deep runs remain manual. [Its README](experiments/blind_prime_spectrum/README.md)
lists exact PowerShell commands and explains the integrity boundary. Unlike older
experiments, blind generation requires an empty output directory to preserve its
frozen predictions.

Defaults are anchored to the script's directory, independent of your working directory.
Rerunning replaces that experiment's default results. To preserve a comparison,
use `--save .\experiments\kg_worldline\results\variant.png` or prime frame's
`--output .\experiments\prime_frame\results\larger_sweep`.

Open the saved PNGs or Markdown gallery whenever you want to inspect a previous
run; Python is not needed to view them. Results are intentionally not gitignored.
The saved prime-frame example is a small smoke run through p=101, not a scientific
conclusion or the full default sweep. Its exact settings are in `parameters.json`.

Run the lightweight numerical tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_frame -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_frame_geometry -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_frame_normalized -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_factor_phasors -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/prime_zero_features -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/zeta_internal_momentum -p "test_*.py" -v
```

The original `prime-wave/` scripts now live under `experiments/`; their numerical
models have been retained. The original nested repository's Git history is preserved
at this root. These experiments explore mathematical constructions and hypotheses;
they do not establish a prime/mass correspondence.
