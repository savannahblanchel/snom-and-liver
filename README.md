# snom-and-liver

Clean SNOM near-field spectroscopy workflow for tissue/cell classification and
physics-informed inversion.

This repository is intentionally separate from the legacy research scripts. The
legacy folders `D:\undergra\na_nn\111` and `D:\undergra\new` are treated as
references only; this repo contains the self-contained code path that can be
published, reviewed, and extended.

## Current Workflow

### 1. Preprocess SNOM spectra

For files with measured background spectra:

```powershell
python -m snom_liver.preprocess `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir "outputs\group1_bg_normalized" `
  --mode bg-normalized `
  --wavenumber-min 690 `
  --wavenumber-max 1800
```

For files exported only relative to the instrument reference:

```powershell
python -m snom_liver.preprocess `
  --input-dir "D:\undergra\na_nn\snom光谱2" `
  --input-dir "D:\undergra\na_nn\snom-lei1" `
  --output-dir "outputs\group2_raw_reference" `
  --mode raw-reference `
  --wavenumber-min 690 `
  --wavenumber-max 1800
```

Background normalization rule:

```text
O2A_norm(w) = O2A_sample(w) / O2A_background(w)
O2P_norm(w) = O2P_sample(w) - O2P_background(w)
```

### 2. PLS-DA baseline

```powershell
python -m snom_liver.plsda `
  --dataset group1_bg_normalized=outputs\group1_bg_normalized `
  --dataset group1_raw_reference=outputs\group1_raw_reference `
  --dataset group2_raw_reference=outputs\group2_raw_reference `
  --output-dir outputs\plsda_baseline `
  --feature-set amp `
  --feature-set phase `
  --feature-set amp_phase `
  --max-components 8
```

The baseline uses grouped cross-validation, normally grouped by `sample_id`.
For stricter checks, use `--group-col specimen_name`.

### 3. Clean physical inversion smoke test

The clean physics module currently implements:

- Si/Au/air reference material selection;
- single-phonon Lorentz dielectric function for comparison;
- three-oscillator multi-Lorentz dielectric function for biological spectra;
- semi-infinite S-SNOM forward model with separate near-field `beta` and
  far-field p-polarized reflectivity `r_p`, following the validated `FDM/main.py`
  formula structure without directly importing that long legacy script;
- direct measured-bg complex reference from the original SNOM txt file;
- per-spectrum `Tip Frequency` and `Tapping Amplitude` parsing from txt headers;
- trainable `g_factor/g_phase` coupling correction;
- optional global amplitude correction with bounded `amp_scale/amp_offset`;
- amplitude normalization modes: `median`, `zscore`, `snv`, `reference-band`, `none`;
- optional ALS amplitude baseline controls: `none`, `als-subtract`,
  `als-divide`;
- optional Lorentz center initialization from measured spectral peaks, with
  multi-start jitter controls in the batch fitter;
- combined loss: relative amplitude + amplitude shape + circular phase;
- gradient-based fitting for one processed spectrum.

Main biological-material mode:

```text
sample_saved = sample / Si
bg_saved = bg / Si
bg-normalized target = sample_saved / bg_saved = sample / measured_bg
```

Therefore `measured-bg` is the main fitting mode for bg-normalized datasets. The
model first predicts `sample / Si`, then normalizes the prediction by the
measured bg spectrum read directly from the original `background_file`.

```powershell
python -m snom_liver.physical_fit `
  --dataset "outputs\group1_bg_normalized" `
  --output-json "outputs\physics\group1_si_smoke.json" `
  --fit-mode measured-bg `
  --model multi-lorentz `
  --reference-material si `
  --epochs 10 `
  --step 20 `
  --amp-normalization reference-band `
  --reference-band-min 1500 `
  --reference-band-max 1600 `
  --amp-baseline none `
  --center-init fixed `
  --no-use-amp-scale `
  --no-use-amp-offset
```

For physics fitting, prefer passing the dataset folder rather than only the
`.npz`, because the folder contains metadata with `source_file` and
`background_file` paths.

Comparison modes:

- `--fit-mode raw-reference`: fit spectra as `sample / Si`.
- `--fit-mode processed-direct`: fit processed spectra directly without measured
  bg correction. This is mostly a control mode.
- `--fit-mode auto`: use `measured-bg` for `bg-normalized` datasets and
  `raw-reference` otherwise.

This is still a fitting baseline, not final chemistry. The next step is a
batch-level fitter where one experimental batch shares instrument parameters
such as `g_factor/g_phase`, while each spectrum owns only material parameters.

### 4. Batch-level physical fitter

For one experimental batch, use shared instrument/nuisance parameters and
per-spectrum material parameters:

```text
shared across batch: g_factor, g_phase
optional shared controls: amp_scale, amp_offset
per spectrum: multi-Lorentz material parameters
fixed/default: R, L
read from txt header: Tip Frequency, Tapping Amplitude
```

CPU smoke test:

```powershell
python -m snom_liver.physical_batch_fit `
  --dataset "outputs\group1_bg_normalized" `
  --output-json "outputs\physics\group1_batch_shared_smoke.json" `
  --fit-mode measured-bg `
  --model multi-lorentz `
  --reference-material si `
  --max-spectra 3 `
  --epochs 3 `
  --step 50 `
  --amp-normalization reference-band `
  --reference-band-min 1500 `
  --reference-band-max 1600 `
  --amp-baseline none `
  --center-init fixed `
  --no-use-amp-scale `
  --no-use-amp-offset
```

`amp_scale` and `amp_offset` are intentionally disabled by default when using
amplitude normalization. Enable them only as a control experiment, because they
can otherwise absorb non-material baseline differences.

Useful controls after the first residual/stability check:

- `--amp-baseline als-divide` or `--amp-baseline als-subtract`: optional ALS
  amplitude baseline correction before amplitude normalization.
- `--center-init peaks`: initialize multi-Lorentz centers from prominent
  measured amplitude deviations instead of fixed centers.
- `--n-starts 3 --center-jitter 35`: run several center initializations and keep
  the best batch fit.

Current caution: on the first 20-spectrum group1 check, ALS with fixed centers
slightly improves amplitude MSE and correlation, but peak-based center
initialization performs worse. Do not interpret fitted Lorentz centers as
chemical peak assignments until residuals improve and centers move away from
initialization in a stable, class-specific way.

### 5. PIML-assisted physical inversion

The PIML path follows the structure of the newer scripts in `D:\undergra\new`:
train an inverse network on spectra generated by the clean physical forward
model, use the network to predict material-parameter initial values for
experimental spectra, then run a physics-model fine-tune. PIML is not used as a
classifier at this stage.

Train a synthetic inverse model:

```powershell
python -m snom_liver.piml.train `
  --output-dir "outputs\piml\group1_multilorentz_v1" `
  --n-samples 4000 `
  --epochs 80 `
  --batch-size 64 `
  --wmin 700 `
  --wmax 1600 `
  --step 30 `
  --amp-normalization reference-band `
  --reference-band-min 1500 `
  --reference-band-max 1600 `
  --device cpu
```

Use the trained model to initialize and fine-tune experimental spectra:

```powershell
python -m snom_liver.piml.invert `
  --dataset "outputs\group1_bg_normalized" `
  --checkpoint "outputs\piml\group1_multilorentz_v1\piml_checkpoint.json" `
  --model-state "outputs\piml\group1_multilorentz_v1\piml_model.pt" `
  --output-dir "outputs\piml\group1_inversion_v1" `
  --indices 0 1 2 33 34 35 `
  --fit-mode measured-bg `
  --epochs 80 `
  --step 30 `
  --amp-normalization reference-band `
  --reference-band-min 1500 `
  --reference-band-max 1600 `
  --no-fit-g
```

The inversion CSV writes flattened parameter columns such as `fit_center_1`,
`fit_strength_1`, `fit_gamma_1`, and `fit_eps_inf`. Use these only after checking
within-class stability, residual structure, and agreement with physically
reasonable spectral features.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data Policy

Raw data and generated outputs are ignored by git. Keep large local datasets
outside the repository, then pass their paths to the command-line scripts.
