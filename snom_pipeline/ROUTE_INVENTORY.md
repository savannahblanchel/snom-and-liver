# Route Inventory

This file records the current GitHub-ready route only.

## Active scope

- Samples: liver / tumor spectra from `D:\undergra\na_nn\snom光谱1`.
- Background: matched raw bg from the same batch.
- Working window: 690-1400 cm^-1.
- Model status: physics-constrained effective spectrum parameters, not absolute dielectric constants.

## Accepted physical route

1. Matched bg preprocessing with `preprocess_snom.py`.
2. Sample-level summary with `relative_spectrum_analysis.py`.
3. Physics + ML fitting with `physical_ml_platform.py`.
4. Stability screening with `physical_feature_selector.py`.
5. Unknown-spectrum fitting and exploratory classification with `physical_param_classifier.py`.

The accepted fit configuration uses:

- `n_oscillators = 2`;
- `reference_material = matched-bg`;
- `fixed_g_factor = 0.5`;
- `fixed_g_phase = 0.03`;
- differential-evolution initialization;
- local physics refinement;
- derivative spectrum loss;
- profiled nuisance amplitude / phase calibration;
- soft `liver-ftir` literature prior.

## Stable interpretation core

Use these as the primary stable feature pair:

- `ordered_wT_1`
- `ordered_wL_1`

These are effective oscillator center-like parameters. Features in the
1000-1250 cm^-1 region are allowed for this analysis, but they should be
reported as substrate-sensitive / coupled spectrum-shape features.

The curated classifier feature list is stored at
`snom_pipeline\config\stable_core_features_matched_bg_lit_deriv_de.csv`.
