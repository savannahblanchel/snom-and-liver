# SNOM organoid / tonsil route inventory

This file records the current route split in this workspace, because several
branches share the same scripts and are separated mainly by command-line
arguments and output directories.

## Current scope

- Main samples: organoid and tonsil processed SNOM spectra.
- TA is grouped as organoid.
- `lei 1` and `lei 2` are still separated by `specimen_name + tune_wavenumber`:
  - `类器官1 ZX Cell 1__1200`
  - `类器官2 ZX Cell 2__1280`
- The processed organoid / tonsil data do not have matched raw background, so
  the main branch does not redo raw background normalization.

## Shared preprocessing layer

Script:

- `relative_spectrum_analysis.py`

Current output:

- `outputs/relative_analysis/group2_raw_reference`

Role:

- combines point spectra into sample-level mean spectra;
- writes within-sample dispersion summaries;
- provides the input used by all later physical / complex routes.

## Route A: phase0 amplitude-shape Lorentz fit

Script:

- `physical_batch_relative_fit.py`

Historical output:

- `outputs/physical_relative_fit_phase0`
- report: `outputs/physical_relative_report_phase0`

Canonical command:

```powershell
python .\physical_batch_relative_fit.py `
  --dataset group2_raw_reference=outputs\relative_analysis\group2_raw_reference `
  --output-dir outputs\physical_relative_fit_phase0 `
  --wmin 690 `
  --wmax 1750 `
  --stride 4 `
  --n-lorentz 3 `
  --phase-weight 0.0 `
  --phase-calibration-mode none
```

Meaning:

- amplitude controls the fit;
- phase is diagnostic only;
- the fitted Lorentz parameters are relative spectrum-shape parameters, not an
  absolute dielectric inversion.

Notes:

- The old output is preserved.
- The current script has extra PMMA-style and phase-calibration options, but the
  phase0 behavior is still reachable by explicitly setting
  `--phase-weight 0.0 --phase-calibration-mode none`.

## Route B: PMMA-style complex response branch

Script:

- `physical_batch_relative_fit.py`

Current outputs:

- `outputs/physical_relative_fit_complex_phasecal_offset`
- report: `outputs/physical_relative_report_complex_phasecal_offset`

Canonical command:

```powershell
python .\physical_batch_relative_fit.py `
  --dataset group2_raw_reference=outputs\relative_analysis\group2_raw_reference `
  --output-dir outputs\physical_relative_fit_complex_phasecal_offset `
  --wmin 690 `
  --wmax 1750 `
  --stride 4 `
  --n-lorentz 3 `
  --fit-space complex `
  --phase-calibration-mode offset
```

Meaning:

- builds `z = amp * exp(i * phi_corrected)`;
- fits Re/Im jointly;
- tests whether amplitude and phase can be described by one effective Lorentz
  response;
- useful as a PMMA-inspired validation / interpretation branch.

Current judgment:

- keep it as a comparison branch;
- do not use it as the main structural interpretation model yet, because its
  amplitude reproduction is worse than Route A / Route C.

## Route C: amplitude-shape fit with phase offset calibration

Script:

- `physical_batch_relative_fit.py`

Current preferred output:

- `outputs/physical_relative_fit_phasecal_offset_w005`
- report: `outputs/physical_relative_report_phasecal_offset_w005_corrected`

Canonical command:

```powershell
python .\physical_batch_relative_fit.py `
  --dataset group2_raw_reference=outputs\relative_analysis\group2_raw_reference `
  --output-dir outputs\physical_relative_fit_phasecal_offset_w005 `
  --wmin 690 `
  --wmax 1750 `
  --stride 4 `
  --n-lorentz 3 `
  --phase-weight 0.05 `
  --phase-calibration-mode offset
```

Meaning:

- keeps the amplitude-shape Lorentz route as the main fit;
- subtracts one batch-level phase offset estimated from low-structure windows;
- lets phase enter weakly, mainly to prevent completely inconsistent phase.

Current judgment:

- best current compromise;
- amplitude MSE is essentially unchanged from phase0;
- phase circular MSE is much better than phase0;
- Lorentz centers remain close to phase0, so the original amplitude-shape
  interpretation is not being rewritten by phase.

## Route D: phase slope calibration test

Script:

- `physical_batch_relative_fit.py`

Current outputs:

- `outputs/physical_relative_fit_phasecal_slope_smoke`
- `outputs/physical_relative_fit_complex_phasecal_slope`

Meaning:

- subtracts `phi0 + phi1 * (w - w0)`.

Current judgment:

- not recommended as a main branch;
- phase / complex metrics worsened in the current tests, suggesting the linear
  slope is removing real spectral structure or introducing instability.

## Route E: 2-Lorentz simplified control

Script:

- `physical_batch_relative_fit.py`

Current output:

- `outputs/physical_relative_fit_phase0_n2`
- report: `outputs/physical_relative_report_phase0_n2`

Meaning:

- reduced-degree-of-freedom control.

Current judgment:

- useful as a supplementary control only;
- worse amplitude fit than the 3-Lorentz route;
- apparent center stability is likely partly forced by lower flexibility.

## Route F: fixed background diagnostic

Script:

- `physical_batch_relative_fit.py`

Current output:

- `outputs/physical_relative_fit_phase0_fixed_bg_20260806`
- report: `outputs/physical_relative_report_phase0_fixed_bg_20260806`

Meaning:

- multiplies the model by one external reference spectrum from:
  `D:\undergra\na_nn\snom光谱1\2026-08-06 113047 PH PSP gan-ikzf-bg.txt`

Current judgment:

- diagnostic only;
- not the main branch because the reference is not a matched same-batch 1280
  background and worsened the 1280 amplitude fit.

## Route G: senior-style physics + ML platform

Script:

- `physical_ml_platform.py`

Current outputs:

- `outputs/physical_ml_platform_smoke`
- `outputs/physical_ml_platform_phase0warm_smoke`
- `outputs/physical_ml_platform_phase0warm_smoke2`

Related original code locations:

- liver/tumor adapted package:
  `D:\undergra\snom_project\github_ready\snom-and-liver`
- senior source fragments:
  `D:\undergra\na_nn\111`
  `D:\undergra\new`

Meaning:

- adapted self-contained platform route;
- synthetic spectra from a senior-style semi-infinite S-SNOM model;
- MLP pretraining for parameter initialization;
- gradient-based refinement on experimental sample-level spectra.

Current judgment:

- framework exists and is findable;
- current outputs are smoke / low-budget tests, not final scientific results;
- useful later if the project needs a more physics-complete forward platform,
  but it is not yet competitive with the compact relative fit on these data.

## Code-pollution status

- Route A, B, C, D, E, and F currently share `physical_batch_relative_fit.py`.
- The old phase0 result directories are preserved, but the old phase0 source is
  not a separate frozen script in this workspace.
- `physical_batch_relative_fit.py` now contains additional options:
  - `--fit-space amp_phase|complex`
  - `--phase-calibration-mode none|offset|slope`
  - `--phase-calibration-windows`
  - `--reference-file`
- The current default phase calibration mode is `none`, so old commands do not
  silently become offset-calibrated.
- To avoid ambiguity, every future command should explicitly write both:
  - `--fit-space ...`
  - `--phase-calibration-mode ...`

## Recommended naming convention from now on

- Main current branch:
  `physical_relative_fit_phasecal_offset_w005`
- PMMA-style comparison:
  `physical_relative_fit_complex_phasecal_offset`
- Historical phase0 baseline:
  `physical_relative_fit_phase0`
- Senior platform smoke / development:
  `physical_ml_platform_*`
