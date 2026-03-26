# napari-hdf5-activity

**napari-hdf5-activity** is a napari plugin for automated quantitative analysis of locomotor activity and behavioral rhythms from long-term timelapse recordings. It is designed for chronobiology and behavioral neuroscience research, with particular support for small aquatic invertebrates and other model organisms in multi-well or free-field recording setups.

## What It Does

The plugin reads HDF5 timelapse files, Zarr directory stores, and AVI video files, detects organism regions of interest (ROIs) automatically using the Hough Circle Transform, and quantifies frame-to-frame pixel changes as a proxy for movement. It then classifies movement states using a hysteresis threshold algorithm and computes activity fraction, quiescence, and behavioral sleep bouts across the full recording duration.

A dedicated Extended Analysis tab provides six complementary methods for detecting, quantifying, and comparing circadian and ultradian rhythms across ROIs.

## Target Organisms

The plugin has been developed and validated for:

- **Nematostella vectensis** (starlet sea anemone) — circadian and tidal rhythm analysis
- **Caenorhabditis elegans** — activity and sleep bout quantification
- **Danio rerio** (zebrafish) — larval locomotor activity
- **Drosophila melanogaster** — locomotor rhythmicity
- Other small organisms imaged in well-plate or open-field timelapse setups

## Supported File Formats

- **HDF5** (.h5, .hdf5) — stacked-frame or individual-frame structures, with embedded metadata and LED lighting data
- **Zarr** (.zarr directory stores) — treated identically to HDF5 stacked-frame datasets
- **AVI** — single files or batches with temporal concatenation

## Key Analysis Methods

### Movement Analysis
- Pixel-difference movement quantification, normalized per ROI area
- Three threshold methods: Baseline (from first N minutes), Calibration (reference recording), and Adaptive (sliding window)
- Hysteresis state detection to prevent spurious state switching
- Jump correction for sudden signal artifacts
- Optional linear detrending

### Circadian Rhythm Detection (Extended Analysis Tab)

| Method | What It Provides |
|--------|-----------------|
| **Chi² Periodogram** (Sokolove & Bushell 1978) | Bonferroni-corrected significance testing across m=100 periods; threshold ≈ 15.2 for α=0.05 |
| **FFT Power Spectrum** | Full spectral decomposition with permutation significance (1000 shuffles, max-power test) |
| **Cosinor Analysis** | MESOR, Amplitude, and Acrophase with F(2, n−3) individual test and Nelson et al. (1979) population F-test |
| **ROI Similarity Matrix** | Pairwise cross-correlation with Bonferroni-corrected t-test and hierarchical clustering |
| **Coherence Analysis** | Welch magnitude-squared coherence with Bonferroni correction across all ROI pairs |
| **Phase Clustering** | Hilbert-transform instantaneous phase, circular mean, Phase Locking Value (descriptive) |

## Key Outputs

- **Excel export**: movement traces, activity fraction, sleep bouts, statistics, parameters, and metadata (7 sheets)
- **Publication-quality plots**: movement traces, activity fraction, lighting conditions (from HDF5 LED data), sleep patterns, actograms, periodograms, phase polar plots
- **Circadian results export**: CSV and Excel with per-ROI periodogram data and rhythm parameters

## Installation

```bash
pip install napari-hdf5-activity
```

**Dependencies**: napari >= 0.4.17, numpy, h5py, zarr, opencv-python, matplotlib, scikit-image, scipy

**Optional** (for Excel export): pandas, openpyxl

## Typical Workflow

1. Load HDF5/Zarr/AVI file — first frame displayed automatically
2. Detect ROIs (Hough Circle Transform with adjustable parameters)
3. Fine-tune ROI positions with the ROI Editor if needed
4. Configure and run movement analysis (baseline/calibration/adaptive method)
5. Generate plots and export to Excel
6. Run Extended Analysis for circadian rhythm detection and population statistics

## Performance

- Multiprocessing support: ROI-level parallelization with automatic core detection (3–4× speedup with 4 cores on large datasets)
- Memory-efficient: only the first frame is loaded for ROI detection; full dataset loaded on demand during analysis
- Dynamic RAM management adapts queue size to available system memory
