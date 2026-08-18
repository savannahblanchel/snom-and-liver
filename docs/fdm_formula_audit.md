# FDM Formula Audit Before Biological PIML Inversion

This note records how the clean repo relates to the legacy physical forward
model in `D:\undergra\na_nn\111\FDM`. The legacy folder remains read-only.

## Legacy Physics Path

Main Python implementation:

- `main.py`
  - `SsnomModel.compute_epsilon`: material dielectric function.
  - `SsnomModel.compute_reflectivity`: builds dielectric tensors and calls TMM.
  - `SsnomModel.integral_sinf`: semi-infinite/spheroid-inspired near-field
    demodulation integral.
  - `SsnomModel.compute_signal_sinf`: sample/Au reference normalized signal.
  - `SsnomModel.compute_signal`: full TMM reflectivity plus integral signal.
- `tmm.py`
  - `UltraOptimizedTMM`: Berreman-style anisotropic transfer matrix.
  - Returns p-polarized field reflection coefficient as `r[0, :]`.
- `mat_dielectric.py`
  - Material tensor generator, Si/Au and phonon dielectric functions.

Matlab reference implementation:

- `jifenhanshu1.m`
  - Semi-infinite integral with beta, tapping height, harmonic demodulation and
    far-field propagation factor.
- `reflectivity.m`
  - Transfer-matrix reflectivity for a layered material stack.
- `Copy_of_Spheroid.m`
  - Poles/residues spheroid model and harmonic demodulation.
- `passler_transfer_matrix_modular_k.m`
  - Original modular transfer matrix.

## Current Clean Repo Status

The current clean forward model in `src/snom_liver/physics/ssnom.py` is a
simplified baseline. It captures a near-field integral and reference
normalization, but it does not yet fully reproduce the legacy chain:

```text
dielectric tensor -> TMM r_p -> FDM/spheroid integral -> reference normalization
```

Therefore current PIML v1 is a workflow scaffold, not a chemically interpretable
physical inversion engine.

## Source Of Truth

The experimentally validated implementation is `FDM/main.py`. Treat Matlab
files as historical references for formulas and debugging, but do not use them
to override `main.py` behavior unless the original authors explicitly confirm a
bug in `main.py`.

Practical consequence: the clean repo should mimic the validated formula
structure from `SsnomModel`, but should not directly call or vendor the whole
legacy `main.py`. Experimental settings must come from this project's SNOM
files and command-line configuration, especially `Z`/tapping amplitude and tip
frequency.

## Required Alignment Before Structural Interpretation

1. Keep the clean repo's own forward model and parameter plumbing.
2. Pull in only the missing physics ideas from `main.py`: separate near-field
   beta from far-field p-polarized reflectivity, then use the FDM demodulation
   integral and reference normalization.
3. Read per-spectrum experimental parameters from SNOM headers when available,
   especially `Tapping Amplitude` as `Z` and `Tip Frequency` as `f`.
4. Add fuller TMM/layer support later only if the biological sample/substrate
   geometry requires it. Do not hard-code the legacy inorganic stack or legacy
   instrument constants.
5. Connect PIML to this clean, experiment-specific forward model for biological
   parameter stability and structural interpretation.
