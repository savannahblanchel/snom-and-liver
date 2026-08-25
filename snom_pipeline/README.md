# SNOM Pipeline

This folder starts the three-layer SNOM workflow.

## Layer 1: experimental normalization or processed-spectrum summary

Script:

Three comparison datasets:

```powershell
# Group 1: snom光谱1, with bg normalization
python snom_pipeline/preprocess_snom.py `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir "snom_pipeline\outputs\group1_bg_normalized" `
  --mode bg-normalized `
  --wavenumber-min 690 `
  --wavenumber-max 1800

# Group 1 control: snom光谱1, raw exported O2A/O2P
python snom_pipeline/preprocess_snom.py `
  --input-dir "D:\undergra\na_nn\snom光谱1" `
  --output-dir "snom_pipeline\outputs\group1_raw_reference" `
  --mode raw-reference `
  --wavenumber-min 690 `
  --wavenumber-max 1800

# Group 2: snom光谱2, raw exported O2A/O2P because no bg file is available
python snom_pipeline/preprocess_snom.py `
  --input-dir "D:\undergra\na_nn\snom光谱2" `
  --input-dir "D:\undergra\na_nn\snom-lei1" `
  --output-dir "snom_pipeline\outputs\group2_raw_reference" `
  --mode raw-reference `
  --wavenumber-min 690 `
  --wavenumber-max 1800
```

Background-normalization rule:

```text
O2A_norm(w) = O2A_sample(w) / O2A_background(w)
O2P_norm(w) = O2P_sample(w) - O2P_background(w)
```

Output files in each dataset folder:

- `metadata.csv`: all discovered sample points and processing status.
- `metadata_normalized.csv`: points that have usable processed spectra.
- `spectra_normalized.npz`: arrays for bg-normalized mode.
- Raw-reference mode writes `spectra_raw_reference.npz`.
- `summary.csv`: dataset status and QC summary.

For organoid and tonsil, the current working assumption is that the exported spectra are already processed spectra. In that case, the next step is the relative summary:

```powershell
python snom_pipeline\relative_spectrum_analysis.py `
  --dataset group1_bg_normalized=snom_pipeline\outputs\group1_bg_normalized `
  --dataset group2_raw_reference=snom_pipeline\outputs\group2_raw_reference `
  --output-dir snom_pipeline\outputs\relative_analysis
```

This writes sample-level means, within-sample dispersion, and class-level summaries for later physical fitting.

## Layer 2: batch-level relative physical fit

For the organoid / tonsil branch, use the processed spectra summary and fit a shared batch gain plus per-sample Lorentz parameters:

```powershell
python snom_pipeline\physical_batch_relative_fit.py `
  --dataset group2_raw_reference=snom_pipeline\outputs\relative_analysis\group2_raw_reference `
  --output-dir snom_pipeline\outputs\physical_relative_fit `
  --wmin 690 `
  --wmax 1750 `
  --stride 4 `
  --n-lorentz 3 `
  --phase-weight 0.0
```

This keeps `G` batch-shared and leaves the material parameters sample-specific.
Phase is diagnostic only in the current phase0 line, so the default fit should stay on `--phase-weight 0.0`.

Optional fixed-reference test, using one exported bg spectrum as a multiplicative background layer:

```powershell
python snom_pipeline\physical_batch_relative_fit.py `
  --dataset group2_raw_reference=snom_pipeline\outputs\relative_analysis\group2_raw_reference `
  --output-dir snom_pipeline\outputs\physical_relative_fit_phase0_fixed_bg_20260806 `
  --wmin 690 `
  --wmax 1750 `
  --stride 4 `
  --n-lorentz 3 `
  --phase-weight 0.0 `
  --reference-file "D:\undergra\na_nn\snom光谱1\2026-08-06 113047 PH PSP gan-ikzf-bg.txt"
```

This is a diagnostic branch, not the main model. With the non-matched bg above,
the 1280 amplitude fit became worse, so the current main line should remain the
no-fixed-bg phase0 fit unless a matched same-batch substrate/reference spectrum
is available.

## Layer 2b: senior-style physics + ML platform

If you want the full physics-plus-ML route for the organoid / tonsil branch, use the adapted platform entry:

```powershell
python snom_pipeline\physical_ml_platform.py `
  --dataset snom_pipeline\outputs\relative_analysis\group2_raw_reference `
  --output-dir snom_pipeline\outputs\physical_ml_platform `
  --wmin 690 `
  --wmax 1750 `
  --stride 4 `
  --n-oscillators 3 `
  --phase-weight 0.0
```

This keeps the senior-style flow:
- synthetic spectra for parameter pretraining
- MLP parameter prediction from amplitude features
- physics refinement on experimental samples

TA is grouped with organoid in this route.

## Layer 3: visual report

Render predicted-vs-true curves and parameter stability plots:

```powershell
python snom_pipeline\physical_relative_report.py `
  --fit-dir snom_pipeline\outputs\physical_relative_fit `
  --spectra-root snom_pipeline\outputs\relative_analysis `
  --output-dir snom_pipeline\outputs\physical_relative_report
```

The report writes one folder per batch with:
- predicted-vs-true overlays
- residual plots
- class-wise parameter stability boxplots
- stability tables for later inspection
- `center_parameter_shortlist.csv` for center terms that have low CV and no boundary hits

Current status:

- `group1_bg_normalized`: `snom光谱1`, 60 bg-normalized point spectra.
- `group1_raw_reference`: `snom光谱1`, 60 raw-reference point spectra.
- `group2_raw_reference`: `snom光谱2` + `snom-lei1`, 83 raw-reference point spectra.
- Group 2 labels: `bian` -> `人扁桃体_14_PD1_1.500`, `TA` -> `TA0038190758_PD1_1.500`, `lei 1` -> `类器官1 ZX Cell 1`, `lei 2` -> `类器官2 ZX Cell 2`.
- Wavenumber range requested: 690-1800 cm^-1.
- `lei 1` and `lei 2` are kept separate in the physical layer by `specimen_name + tune_wavenumber`, e.g. `类器官1 ZX Cell 1__1200` and `类器官2 ZX Cell 2__1280`.
- In the phase0 line, phase is used only as a diagnostic channel; the main fit objective is amplitude shape.
- `group1_bg_normalized` drops background-zero wavenumbers: 1676, 1677, 1678 cm^-1.

## Layer 2: PLS-DA baseline classification

Script:

```powershell
python code\snom_pipeline\classification_plsda.py `
  --dataset group1_bg_normalized=outputs\group1_bg_normalized `
  --dataset group1_raw_reference=outputs\group1_raw_reference `
  --dataset group2_raw_reference=outputs\group2_raw_reference `
  --output-dir outputs\plsda_baseline `
  --feature-set amp `
  --feature-set phase `
  --feature-set amp_phase `
  --max-components 8
```

The baseline uses leave-one-sample-group-out evaluation based on `sample_id`.
It reports both point-level accuracy and sample-level aggregated accuracy.
