# snom-and-liver

Clean SNOM workflow for the current liver / tumor route.

This repository keeps only the current recommended path:

- matched-bg preprocessing
- sample-level relative spectrum aggregation
- physics + ML fitting
- stable feature selection
- exploratory classification

The current interpretation target is **physics-constrained effective spectrum parameters**, not absolute dielectric constants.

## What this repo is for

This project is a stage of SNOM processed-spectrum analysis for liver / tumor samples.
It is designed to answer three practical questions:

1. Can the processed spectra be reproduced with a physically constrained model?
2. Which fitted parameters are stable enough to explain across seeds and folds?
3. Do the stable parameters carry enough signal for exploratory classification?

## Repository layout

```text
snom_pipeline/
  preprocess_snom.py              matched-bg preprocessing
  relative_spectrum_analysis.py   sample-level aggregation and dispersion stats
  physical_ml_platform.py         physics + ML fitting platform
  physical_feature_selector.py    stable feature screening
  physical_param_classifier.py    unknown-spectrum refit and classification
  physical_relative_report.py     fit plots and stability reports
  plot_liver_tumor_waveforms.py   waveform QC
  config/
    stable_core_features_matched_bg_lit_deriv_de.csv
```

## Data requirements

The repository does **not** include raw SNOM text files.
To reproduce the route, prepare a local folder that contains:

- processed SNOM point-spectroscopy `.txt` files
- one matched background file in the same folder
- the background filename must contain `bg`
- sample filenames must preserve the specimen / class / tune information

Example:

```text
D:\undergra\na_nn\snom光谱1\
  2026-08-06 113047 PH PSP gan-ikzf-bg.txt
  2026-08-06 ... gan-....
  2026-08-06 ... zhong-....
```

The current route uses the liver / tumor batch, not the organoid / tonsil branch.

### Required columns

The parser expects Neaspec text files with a data header containing
`Row`, `Column`, `Wavenumber`, `O2A`, and `O2P`.
The three numerical columns used by this pipeline are:

| Column | Meaning | Used by |
| --- | --- | --- |
| `Wavenumber` | spectral coordinate in `cm^-1` | windowing and interpolation |
| `O2A` | amplitude channel | amplitude fit and classification |
| `O2P` | phase channel | phase fit and classification |

Extra columns are allowed and ignored.
Rows without a valid `Wavenumber` are dropped.

### Filename conventions

The current parser uses filename text to infer metadata:

| Filename text | Inferred meaning |
| --- | --- |
| `bg` | matched background spectrum |
| `gan` | liver |
| `zhong` | tumor |
| `-1000`, `-1200`, `-1280` | tune wavenumber |
| final `-number` | point identifier |

The exact naming rules are implemented in
`snom_pipeline/preprocess_snom.py`.
If a new dataset uses different names, either rename a copy of the files or update
the inference functions before running the pipeline.

## Environment

Recommended environment:

- Windows, macOS, or Linux
- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- PyTorch

Example installation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib scikit-learn torch
```

If PyTorch installation is different on the teammate's platform, install the
appropriate CPU or CUDA build first, then install the remaining packages.

## Data logic

The current preprocessing assumption is:

- sample spectra are already normalized relative to a reference
- matched background is also treated in the same normalized space
- background removal is done as a ratio-of-ratios

Practical rule:

- amplitude: divide sample by background
- phase: subtract background phase

Working window:

- `690-1400 cm^-1`

High-wavenumber tail is treated as power-related noise and is not part of the main analysis window.

## Processing order

The scripts are intended to be run in this order:

```text
SNOM txt + matched bg
        |
        v
preprocess_snom.py
        |
        v
relative_spectrum_analysis.py
        |
        v
physical_ml_platform.py
        |
        v
physical_feature_selector.py
        |
        v
physical_param_classifier.py
```

The first four stages build and validate the training route.
The classifier is used only after the physical fit and stable feature list have
been generated.

## Quick start

### 1. Preprocess raw txt files

```powershell
python snom_pipeline\preprocess_snom.py `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir snom_pipeline\outputs\group1_bg_normalized_matched_bg_clean1400 `
  --mode bg-normalized `
  --wavenumber-min 690 `
  --wavenumber-max 1400
```

### 2. Build sample-level relative spectra

```powershell
python snom_pipeline\relative_spectrum_analysis.py `
  --dataset group1_bg_normalized_matched_bg_clean1400=snom_pipeline\outputs\group1_bg_normalized_matched_bg_clean1400 `
  --output-dir snom_pipeline\outputs\relative_analysis_liver_tumor_matched_bg_clean1400
```

### 3. Run the physics + ML fit

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

### 4. Screen stable features

```powershell
python snom_pipeline\physical_feature_selector.py `
  --fit-dir snom_pipeline\outputs\physical_ml_platform_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42 `
  --output-dir snom_pipeline\outputs\physical_feature_selection_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42 `
  --tune-wavenumber 1000 `
  --n-oscillators 2 `
  --include-global `
  --allow-substrate-band-centers
```

### 5. Classify an unknown spectrum

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

## Main modeling choices

- two ordered oscillators
- fixed coupling `G = 0.5 * exp(i * 0.03)`
- profiled nuisance calibration for amplitude scale and phase offset
- point-level stability weighting
- EMA task weighting for amplitude / phase losses
- differential-evolution initialization before local refinement
- derivative-based spectrum-shape losses
- soft literature guidance for liver / HCC FTIR bands

These settings are chosen to improve **spectrum-shape reproducibility** and **parameter stability**.

## Stable parameters

The current most stable core parameters are:

- `ordered_wT_1`
- `ordered_wL_1`

Recommended interpretation:

- treat them as the main candidate explanatory parameters
- treat gamma / strength / nuisance terms as auxiliary
- treat centers in the 1000-1250 cm^-1 substrate-sensitive region conservatively

The final curated feature list is stored in:

```text
snom_pipeline\config\stable_core_features_matched_bg_lit_deriv_de.csv
```

## Outputs

Typical outputs are written to:

- `snom_pipeline\outputs\group1_bg_normalized_matched_bg_clean1400`
- `snom_pipeline\outputs\relative_analysis_liver_tumor_matched_bg_clean1400`
- `snom_pipeline\outputs\physical_ml_platform_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42`
- `snom_pipeline\outputs\physical_feature_selection_liver_tumor_matched_bg_clean1400_lit_deriv_de_seed42`

These folders include:

- normalized spectra
- sample-level summaries
- fit summaries
- parameter stability tables
- predicted-vs-true plots
- feature selection results

The most useful files are:

| File | Purpose |
| --- | --- |
| `metadata.csv` | input files, inferred labels, point IDs, and background mapping |
| `spectra_normalized.npz` | normalized amplitude, phase, and saved background arrays |
| `sample_level_summary.csv` | one row per sample-level spectrum |
| `platform_fit_summary.csv` | batch-level fit configuration and aggregate errors |
| `parameter_stability.csv` | fitted parameter means, standard deviations, and CV values |
| `sample_fit_results.csv` | fitted parameters for individual samples |
| `selected_physical_features.csv` | features passing the current stability/separation screen |
| `*_pred_vs_true.png` | predicted-versus-observed amplitude/phase overlay |

The output directories are ignored by Git because they can be large and depend
on local data. Keep important figures or tables separately if they need to be
shared.

## How to read the results

What the current model can support:

- broad amplitude-spectrum reproduction
- partial phase consistency
- stable center-like parameters
- small-sample exploratory classification

What it should **not** be claimed as:

- absolute dielectric inversion
- strict biochemical assignment
- clinical-grade classification

The soft literature prior nudges the fit toward reported FTIR bands, but it does not force a hard molecular label.

## Visual QC

To plot the liver / tumor waveforms before choosing the fitting window:

```powershell
python snom_pipeline\plot_liver_tumor_waveforms.py `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir "D:\undergra\na_nn\谱图1"
```

## For collaborators

If a teammate wants to reproduce this route, they need:

1. the repository
2. the raw SNOM txt folder
3. the matched background file in the same folder
4. the same filename conventions
5. the same processing window

GitHub alone is enough to reproduce the **method**.
GitHub plus the local data folder is needed to reproduce the **results**.

### Reproducibility checklist

Before comparing two runs, confirm that both users have the same:

- input txt files and matched background file
- filename conventions
- wavenumber window
- `n_oscillators`
- fixed `g_factor` and `g_phase`
- global initialization mode
- random seed
- PyTorch / SciPy versions where possible

Different seeds can produce slightly different fitted parameters.
The project therefore uses repeated appearance across seeds and cross-validation
as part of feature selection.

## Common problems

### No matched background is found

Check that the background file is in the same input directory and that its name
contains `bg`.

### A sample is assigned to `unknown`

Check the filename tokens used for class inference.
The current route expects `gan` for liver and `zhong` for tumor.

### `matched-bg` raises a missing-background error

The downstream physics model requires the preprocessed dataset to contain
background arrays.
Re-run preprocessing with `--mode bg-normalized`; do not use
`--mode raw-reference` for the matched-bg route.

### Results differ between machines

Check the data files, the random seed, the package versions, and the command-line
arguments before interpreting the difference as a scientific effect.

## Technical details

More parameter-level notes and command examples live in [`snom_pipeline/README.md`](snom_pipeline/README.md).
