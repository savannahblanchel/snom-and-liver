# Physics Refactor Notes

## Why This Refactor Exists

The legacy physics code is useful but not suitable as-is for a clean public
repository:

- `D:\undergra\na_nn\111\FDM\main.py` is a large all-in-one script containing
  forward modeling, fitting, plotting, parameter sampling, and experimental
  examples.
- `D:\undergra\new` contains newer PIML ideas, but the directory is not
  self-contained. Its long physical-model scripts import `tmm1`, which is not
  present in that folder.
- The original scripts are tuned to inorganic examples such as SiC/InAs and
  often normalize to Au reference, while this project currently describes the
  measured sample spectra as Si-referenced.

This repository therefore keeps the original code untouched and rebuilds a
small, inspectable physics path.

## Clean Physics Scope

Current module:

- `src/snom_liver/physics/dielectric.py`
- `src/snom_liver/physics/ssnom.py`
- `src/snom_liver/physical_fit.py`
- `src/snom_liver/physical_batch_fit.py`

Implemented now:

- constant Si reference dielectric function, `epsilon_Si = 13 + 0i`;
- simple Au/air placeholders for reference comparison;
- one-phonon Lorentz dielectric function;
- three-oscillator multi-Lorentz dielectric function;
- semi-infinite S-SNOM demodulated signal;
- sample/reference normalized amplitude and phase;
- measured-bg fitting mode for bg-normalized biological spectra;
- direct bg loading from original SNOM txt files;
- tip frequency and tapping amplitude loading from original SNOM txt headers;
- optional bounded global `amp_scale/amp_offset`;
- amplitude normalization modes: `median`, `zscore`, `snv`,
  `reference-band`, and `none`;
- relative-amplitude, shape, and circular-phase loss terms;
- gradient-based single-spectrum fitting.
- batch-level fitting with shared `g_factor/g_phase/amp_scale/amp_offset`
  and per-spectrum material parameters.

Project convention:

- `raw-reference` means the target is approximately `sample / Si`;
- `measured-bg` means the target is approximately `sample / measured_bg`, where
  `bg` was itself measured as `bg / Si`;
- `measured-bg` is the main biological-material route.

Still needs work:

- measured/reference-specific Si optical constants;
- better amplitude-baseline constraints, because `amp_offset` may otherwise
  absorb too much of the normalized amplitude;
- PIML inverse network from the newer scripts;
- full TMM multilayer geometry.

## Relationship To PIML

The clean physical model is the base layer. PIML should be added after the
forward model and normalization convention are trusted:

1. generate simulated spectra from physically allowed parameter ranges;
2. train a neural inverse model to predict parameter initial values;
3. refine those parameters by differentiable physical fitting;
4. compare reconstructed spectra and parameter distributions by tissue class.

This mirrors the useful idea in `D:\undergra\new`, but avoids copying a large,
non-self-contained script into the public project.
