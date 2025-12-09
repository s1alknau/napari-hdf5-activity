# napari-hdf5-activity Plugin: Technical Summary for Publication

## Overview

napari-hdf5-activity is an open-source Python plugin for the napari image viewer that provides automated quantification of organism movement and behavioral state classification from time-lapse video recordings. The plugin supports both HDF5-format timelapse recordings and standard AVI video files, enabling high-throughput analysis of behavioral patterns in small model organisms.

**Repository**: https://github.com/s1alknau/napari-hdf5-activity
**License**: MIT
**Platform**: Windows 11, Python 3.9+
**Integration**: napari ecosystem

---

## Key Features

### 1. Automated Region of Interest (ROI) Detection
- **Algorithm**: Circular Hough Transform-based detection
- **Purpose**: Automatically identifies and segments individual organisms in video frames
- **Parameters**: Configurable size range (min/max radius), detection sensitivity, minimum separation distance
- **Output**: Circular binary masks defining pixel boundaries for each organism

### 2. Movement Quantification Pipeline

#### Input Processing
- **Supported formats**:
  - HDF5 files with dual structure support (stacked frames: `[N, H, W]` or individual frames: `/frames/frame_XXXX`)
  - AVI video files (single or batch processing with temporal concatenation)
- **Memory optimization**: Lazy loading with chunk-based processing
- **Frame sampling**: Configurable interval (default: 5 seconds between analyzed frames)

#### Movement Detection Algorithm
The core movement quantification algorithm operates on a per-ROI basis:

1. **ROI Mask Creation** (one-time setup):
   - Binary circular masks created during ROI detection phase
   - Mask defines which pixels belong to each organism
   - Example: ROI with radius 100px contains ~31,400 pixels

2. **Pixel-Level Movement Calculation** (per frame):
   ```
   For each frame t and ROI:
     - Extract pixels: pixels_t = frame[t][mask], pixels_t-1 = frame[t-1][mask]
     - Calculate differences: diff = |pixels_t - pixels_t-1|
     - Sum changes: total_change = Σ(diff)
     - Normalize: movement_value = total_change / pixel_count
   ```

3. **Efficiency**: Only pixels within ROI masks are processed (~1,000-50,000 pixels per ROI), avoiding computation on millions of background pixels per frame.

**Movement Value Output**:
- Units: Average pixel intensity change per pixel within ROI
- Range: 0-255 (8-bit) or 0-65535 (16-bit images)
- Interpretation: Dataset-specific, relative to baseline thresholds

#### Data Preprocessing Options
- **MATLAB-compatible normalization**: Optional pixel value normalization for compatibility with legacy MATLAB analysis pipelines
- **Detrending**: Polynomial trend removal to compensate for long-term signal drift (photobleaching, LED intensity changes)
- **Jump correction**: Automated detection and correction of sudden signal discontinuities caused by external disturbances

### 3. Threshold-Based State Classification

#### Baseline Method
- **Approach**: Calculate activity thresholds from initial recording period
- **Calculation**: For each ROI, compute mean ± (multiplier × std) from baseline window (default: first 200 minutes)
- **Thresholds**:
  - Upper threshold: baseline_mean + (multiplier × baseline_std)
  - Lower threshold: baseline_mean - (multiplier × baseline_std)
- **Critical implementation detail**: Baseline thresholds calculated from normalized data **BEFORE** detrending to preserve signal characteristics
- **Use case**: Standard analysis of individual recordings

#### Calibration Method
- **Approach**: Transfer thresholds from separate calibration recording to experimental datasets
- **Workflow**:
  1. Process calibration recording to establish reference thresholds
  2. Apply calibration thresholds to all experimental recordings
- **Use case**: Standardized analysis across multiple recordings, batch experiments

#### Adaptive Method
- **Approach**: Dynamic threshold adjustment using sliding window
- **Use case**: Variable environmental conditions, long-duration recordings

### 4. Hysteresis-Based Movement Detection

To prevent rapid state flickering due to signal noise, movement classification uses a hysteresis algorithm:

```
State Machine:
  IF current_state == QUIESCENT:
    IF value > upper_threshold:
      current_state = MOVEMENT

  IF current_state == MOVEMENT:
    IF value < lower_threshold:
      current_state = QUIESCENT
```

**Advantage**: State transitions require crossing both upper and lower thresholds, providing robust classification in noisy signals.

**Output**: Binary movement classification (0 = quiescent, 1 = movement) at frame-level temporal resolution

### 5. Behavioral State Classification

#### Activity Fraction
- **Method**: Time-binned calculation of movement percentage
- **Calculation**: `activity_fraction = (movement_frames / total_frames) per bin`
- **Default bin size**: 60 seconds (configurable)
- **Output**: Continuous activity percentage (0-100%) over time

#### Quiescence Detection
- **Threshold-based classification**: Bins with activity < threshold (default: 50%) classified as quiescent
- **Output**: Binary quiescence state per time bin

#### Sleep Bout Identification
- **Method**: Temporal consolidation of sustained quiescence periods
- **Minimum duration**: Configurable threshold (default: 8 minutes)
- **Output**: Sleep bout intervals with start time, end time, and duration
- **Purpose**: Distinguishes sustained behavioral sleep from brief pauses in activity

### 6. Circadian Rhythm Analysis (Fischer Z-Transformation)

Extended analysis module for detecting periodic behavioral patterns:

- **Method**: Fischer's periodogram analysis
- **Configurable period range**: User-defined search window (e.g., 12-36 hours for circadian rhythms)
- **Statistical testing**: Chi-square significance testing (default α = 0.05)
- **Output**:
  - Dominant periods with statistical significance
  - Fischer Z-scores across period range
  - Sleep/wake phase classification based on periodogram
  - Visual periodogram plots per ROI

### 7. LED-Based Lighting Detection

Automatic extraction and visualization of lighting schedules from experimental metadata:

- **Light phase detection**: White LED power > 0.5%
- **Dark phase detection**: White LED power ≤ 0.5% (IR illumination only)
- **Data source**: HDF5 timeseries metadata
- **Integration**: Overlaid on activity plots for circadian analysis

### 8. Multiprocessing Implementation

To accelerate analysis of large datasets, the plugin implements true parallel processing using Python's `multiprocessing` module.

#### Technical Architecture

**Challenge**: Python's Global Interpreter Lock (GIL) prevents true parallelism with threading for CPU-bound tasks.

**Solution**: Process-based parallelism
- Each worker is a separate Python process with independent GIL
- Enables genuine parallel execution on multi-core CPUs
- Implementation: `multiprocessing.Pool` with automatic task distribution

#### Processing Pipeline

```
Phase 1: Preprocessing (Sequential - Required)
  - Load frames and create ROI masks
  - Calculate pixel differences within ROI masks
  - Compute baseline thresholds (requires data from all ROIs)
  - Optional: Apply detrending and jump correction

Phase 2: ROI-Level Processing (Parallelizable)
  - Distribute ROIs across worker processes
  - Each worker independently processes assigned ROIs:
    * Hysteresis-based movement detection
    * Activity fraction calculation
  - Workers execute simultaneously on separate CPU cores

Phase 3: Post-processing (Sequential - Required)
  - Aggregate results from all workers
  - Cross-ROI quiescence detection
  - Sleep bout identification
  - Statistical calculations
```

#### Performance Characteristics

Tested on Windows 11, Python 3.9, 4-core CPU:

**Large Dataset (10 ROIs, 10,000 frames ~ 14 hours recording)**:
- Sequential (1 core): 29.9 seconds
- Parallel (4 cores): 12.4 seconds
- **Speedup: 2.42x**

**Small Dataset (6 ROIs, 5,000 frames ~ 7 hours recording)**:
- Sequential (1 core): 4.5 seconds
- Parallel (4 cores): 4.2 seconds
- **Speedup: 1.05x** (overhead dominates)

**Analysis**: Multiprocessing overhead (~2 seconds for process creation) dominates for small datasets. Speedup becomes beneficial for datasets >8,000 frames (>10 hours recording).

**Amdahl's Law Impact**: Maximum speedup limited by sequential portions (preprocessing and post-processing comprise ~30% of total time).

#### Implementation Details
- **Process creation method**: `spawn` (Windows standard)
- **Overhead**: ~1-2 seconds for 4 worker processes
- **Memory model**: Each process receives copy of input data (no shared memory)
- **Automatic parallelization decision**: `use_parallel = (num_processes > 1) AND (num_rois ≥ 2)`

### 9. Interactive Frame Viewer

Real-time visualization module for dataset exploration:

- **Navigation**: Frame-by-frame stepping, slider-based seeking
- **Playback**: Configurable FPS (1-60 FPS)
- **Time overlay**: Frame timestamps with white text (50% larger font for visibility)
- **Video export**: Export selected frame ranges as MP4 or animated GIF with configurable FPS
- **Support**: Both HDF5 and AVI datasets

---

## Data Export and Visualization

### Excel Export Format
Comprehensive analysis results exported to multi-sheet Excel workbook (`.xlsx`):

1. **Movement_Data**: Raw movement values per ROI over time
2. **Activity_Fraction**: Time-binned activity percentages
3. **Sleep_Data**: Sleep bout intervals (start, end, duration)
4. **Quiescence_Binned**: Binary quiescence classification per time bin
5. **Statistics**: Summary statistics per ROI (mean, std, total sleep time, etc.)
6. **Parameters**: Complete record of analysis parameters for reproducibility
7. **Metadata**: File information, recording duration, frame intervals

### Visualization Outputs
- **Movement traces**: Continuous movement values with overlaid thresholds
- **Activity fraction plots**: Time-binned activity percentages with lighting conditions
- **Sleep pattern rasters**: Temporal distribution of sleep bouts across ROIs
- **Periodogram plots**: Fischer Z-scores across period range (circadian analysis)
- **Lighting conditions**: Automatic extraction and overlay of light/dark phases

All plots exportable as PNG and PDF formats.

---

## Technical Implementation

### Software Architecture
- **Language**: Python 3.9+
- **Core framework**: napari (image visualization platform)
- **Key dependencies**:
  - `numpy`: Numerical operations
  - `h5py`: HDF5 file reading
  - `opencv-python`: AVI video decoding
  - `scikit-image`: Image processing utilities
  - `matplotlib`: Plotting and visualization
  - `pandas`, `openpyxl`: Excel export

### Code Organization
- **Modular design**: Separate modules for reading, calculation, and GUI
- **Single unified calculation module**: All analysis logic consolidated in `_calc.py`
- **Worker functions**: Top-level functions for multiprocessing compatibility
- **Error handling**: Comprehensive exception handling with user-facing error messages

### Memory Optimization Strategies
1. **Lazy loading**: Only first frame loaded for ROI detection
2. **Chunk-based processing**: Large datasets processed in configurable chunks (default: 20 frames)
3. **Streaming analysis**: AVI batch processing loads, analyzes, and discards frames sequentially
4. **ROI-masked computation**: Only pixels within ROI boundaries processed (not entire frames)

---

## Validation and Compatibility

### MATLAB Compatibility
- **Normalization option**: Optional MATLAB-compatible preprocessing for direct comparison with legacy analysis pipelines
- **Baseline calculation**: Matches MATLAB approach (computed before detrending)
- **Movement calculation**: Equivalent to MATLAB's `framePixelChange` methodology

### Data Integrity
- **Reproducibility**: All analysis parameters saved with results for exact replication
- **Validation**: Multiprocessing implementation produces numerically identical results to sequential processing (verified: max difference < 10⁻⁶)

### Testing
- **Unit tests**: Movement detection, baseline calculation, hysteresis algorithm
- **Performance tests**: `test_multiprocessing_timing.py` validates speedup measurements
- **Integration tests**: End-to-end analysis pipeline validation

---

## Use Cases

1. **Circadian rhythm studies**: Long-duration recordings (24-72 hours) with periodic pattern detection
2. **Sleep research**: Quantification of sleep bout frequency, duration, and temporal distribution
3. **Pharmacological screening**: High-throughput behavioral phenotyping with standardized thresholds (calibration method)
4. **Environmental response studies**: Activity quantification under varying light/temperature conditions
5. **Developmental studies**: Temporal dynamics of activity patterns across life stages

---

## Advantages and Limitations

### Advantages
- **Open-source**: Freely available, modifiable, and extensible
- **Automated**: Minimal manual intervention required after ROI detection
- **High-throughput**: Batch processing capabilities with multiprocessing acceleration
- **Flexible**: Multiple threshold methods for different experimental designs
- **Integrated**: Seamless integration with napari ecosystem
- **Reproducible**: Complete parameter logging and deterministic algorithms
- **Memory-efficient**: Handles large datasets through chunking and streaming

### Limitations
- **ROI detection assumptions**: Assumes organisms are approximately circular
- **Single-plane analysis**: No tracking of depth movement (2D projection only)
- **Static ROIs**: Does not track organism locomotion across frame (position assumed constant)
- **Threshold sensitivity**: Baseline method performance depends on quality of baseline period
- **Platform-specific optimization**: Performance testing conducted on Windows 11 only

---

## Future Development Directions

Potential extensions discussed during development:

1. **Cross-platform optimization**: Performance testing and optimization for Linux and macOS
2. **GPU acceleration**: CUDA-based processing for ultra-high-throughput applications
3. **Advanced ROI tracking**: Integration of object tracking for mobile organisms
4. **Machine learning integration**: Deep learning-based behavioral state classification
5. **Real-time analysis**: Streaming analysis during data acquisition

---

## Conclusion

napari-hdf5-activity provides a comprehensive, validated, and efficient solution for automated behavioral analysis of time-lapse recordings. The combination of robust movement quantification, flexible threshold methods, and parallel processing acceleration enables high-throughput phenotyping studies. The plugin's integration with the napari ecosystem and adherence to open-source principles facilitates reproducibility and community-driven development.

---

## Citation

If you use this plugin in your research, please cite:

```
[Your publication details here]

Software available at: https://github.com/s1alknau/napari-hdf5-activity
```

---

## Acknowledgments

Developed by: s1alknau (https://github.com/s1alknau)
Development environment: Windows 11, Python 3.9
Testing: Comprehensive validation with synthetic and experimental datasets

---

## Technical Specifications Summary

| Feature | Specification |
|---------|--------------|
| **Platform** | Windows 11, Python 3.9+ |
| **File formats** | HDF5 (dual structure), AVI |
| **ROI detection** | Circular Hough Transform |
| **Movement algorithm** | Frame-difference with ROI masking |
| **Temporal resolution** | Configurable (default: 5s intervals) |
| **Threshold methods** | Baseline, Calibration, Adaptive |
| **State classification** | Hysteresis-based (2-threshold) |
| **Behavioral outputs** | Movement, Activity, Quiescence, Sleep |
| **Circadian analysis** | Fischer periodogram (12-36h typical) |
| **Parallelization** | Process-based multiprocessing |
| **Speedup (large data)** | 2.4x (4 cores, 10 ROIs, 10k frames) |
| **Memory optimization** | Chunk-based, streaming, ROI-masked |
| **Export formats** | Excel (.xlsx), PNG, PDF, MP4, GIF |
| **License** | MIT |

---

**Document version**: 1.0
**Last updated**: December 2025
**Generated for**: Publication submission
