# SNOM Pipeline

This repo keeps the current recommended SNOM route for liver / tumor spectra.
The output should be treated as physics-constrained effective spectrum
parameters, not as absolute dielectric constants.

## Current Data Route

Use the liver / tumor batch with matched raw background normalization.

Matched bg:

`D:\undergra\na_nn\snom光谱1\2026-08-06 113047 PH PSP gan-ikzf-bg.txt`

The working window is 690-1400 cm^-1.

```powershell
python snom_pipeline\preprocess_snom.py `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir snom_pipeline\outputs\group1_bg_normalized_matched_bg_clean1400 `
  --mode bg-normalized `
  --wavenumber-min 690 `
  --wavenumber-max 1400

python snom_pipeline\relative_spectrum_analysis.py `
  --dataset group1_bg_normalized_matched_bg_clean1400=snom_pipeline\outputs\group1_bg_normalized_matched_bg_clean1400 `
  --output-dir snom_pipeline\outputs\relative_analysis_liver_tumor_matched_bg_clean1400
```

## Physics + ML Fit

The accepted route uses:

- two ordered oscillators;
- fixed coupling `G = 0.5 * exp(i * 0.03)`;
- profiled nuisance calibration for amplitude scale and phase offset;
- point-level stability weighting;
- EMA task weighting for amplitude / phase losses;
- differential-evolution initialization before local refinement;
- soft literature guidance for liver / HCC FTIR bands;
- first-difference amplitude and phase shape losses.

```powershell
python snom_pipeline\physical_ml_platform.py `
  --dataset snom_pipeline\outputs\relative_analysis_liver_tumor_matched_bg_clean1400\group1_bg_normalized_matched_bg_clean1400 `
  --output-dir snom_pipeline\outputs\physical_ml_platform_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42 `
  --wmin 690 `
  --wmax 1400 `
  --stride 4 `
  --n-oscillators 2 `
  --synthetic-samples 256 `
  --pretrain-epochs 25 `
  --init-candidates 8 `
  --global-init de `
  --global-init-iters 2 `
  --global-init-popsize 3 `
  --refine-steps 60 `
  --phase-weight 0.2 `
  --derivative-weight 0.05 `
  --phase-derivative-weight 0.02 `
  --physics-recon-weight 0.5 `
  --physics-param-weight 0.5 `
  --reference-material matched-bg `
  --fixed-g-factor 0.5 `
  --fixed-g-phase 0.03 `
  --profile-nuisance `
  --point-stability-weight `
  --task-weight-mode ema `
  --literature-prior liver-ftir `
  --literature-center-weight 0.002 `
  --literature-gamma-weight 0.0002 `
  --literature-gamma-soft-max 120
```

The literature term is soft. It nudges oscillator centers toward reported liver
/ HCC FTIR bands and discourages very broad `gamma`, but it does not force a
biochemical assignment.

## Stable Features

The most stable current core features are:

- `ordered_wT_1`
- `ordered_wL_1`

Use these first for classification and interpretation. Other selected features
are secondary unless they repeat across seeds and cross-validation folds.

```powershell
python snom_pipeline\physical_feature_selector.py `
  --fit-dir snom_pipeline\outputs\physical_ml_platform_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42 `
  --output-dir snom_pipeline\outputs\physical_feature_selection_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42 `
  --tune-wavenumber 1000 `
  --n-oscillators 2 `
  --include-global `
  --allow-substrate-band-centers
```

The `--allow-substrate-band-centers` option keeps centers in the 1000-1250
cm^-1 substrate-sensitive region available for analysis. Report those features
as substrate-coupled effective parameters rather than direct molecular peaks.
For the final classifier, use the curated core list in
`snom_pipeline\config\stable_core_features_matched_bg_lit_deriv_de.csv`.

## Classification

The classifier refits an unknown spectrum with the same physical route, then
uses the selected stable parameters for a small-sample exploratory class
estimate.

```powershell
python snom_pipeline\physical_param_classifier.py `
  --fit-dir snom_pipeline\outputs\physical_ml_platform_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42 `
  --feature-selection snom_pipeline\config\stable_core_features_matched_bg_lit_deriv_de.csv `
  --spectrum path\to\unknown_processed_spectrum.csv `
  --output-dir snom_pipeline\outputs\unknown_prediction `
  --phase-weight 0.2 `
  --reference-material matched-bg `
  --synthetic-samples 512 `
  --global-init de `
  --derivative-weight 0.05 `
  --phase-derivative-weight 0.02
```

The output includes predicted class probabilities, fitted effective
parameters, nuisance calibration values, and an overlay plot. Because the
labeled set is small, classification accuracy should be reported as exploratory
cross-validation evidence, not as a deployed diagnostic model.

## Visual Checks

To plot the 12 liver / tumor waveforms before choosing the fitting window:

```powershell
python snom_pipeline\plot_liver_tumor_waveforms.py `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir "D:\undergra\na_nn\谱图1"
```

## Main Scripts

- `preprocess_snom.py`: parse SNOM text files and apply matched bg normalization.
- `relative_spectrum_analysis.py`: build sample-level spectra and dispersion summaries.
- `physical_ml_platform.py`: physics + ML fitting platform.
- `physical_feature_selector.py`: stability and separation based feature screening.
- `physical_param_classifier.py`: unknown spectrum refit and classification.
- `physical_relative_report.py`: predicted-vs-true plots and parameter stability reports.
- `plot_liver_tumor_waveforms.py`: waveform QC plotting.
