# napari-hdf5-activity

[![License MIT](https://img.shields.io/pypi/l/napari-hdf5-activity.svg?color=green)](https://github.com/s1alknau/napari-hdf5-activity/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-hdf5-activity.svg?color=green)](https://pypi.org/project/napari-hdf5-activity)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-hdf5-activity.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-hdf5-activity)](https://napari-hub.org/plugins/napari-hdf5-activity)

A napari plugin for analyzing activity and movement behavior from HDF5, Zarr, and AVI timelapse recordings.

----------------------------------

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Recent Updates (2025)](#recent-updates-2025)
- [Changelog](#changelog)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Parameter Guide](#parameter-guide)
- [AVI File Support](#avi-file-support)
- [Output Files](#output-files)
- [Technical Details](#technical-details)
  - [HDF5 Structure Support](#hdf5-structure-support)
  - [Complete Processing Pipeline](#complete-processing-pipeline)
  - [Multiprocessing Implementation](#multiprocessing-implementation-and-performance)
  - [LED-Based Lighting Detection](#led-based-lighting-detection)
- [📚 Additional Documentation](#-additional-documentation)
- [Troubleshooting](#troubleshooting)
- [Scientific Background](#scientific-background)
  - [Chi² Periodogram](#chi-periodogram-for-circadian-rhythm-detection)
  - [Frame Viewer](#frame-viewer)
- [📐 Mathematical Documentation](#-mathematical-documentation)
  - [Movement Analysis Pipeline](#movement-analysis-pipeline-mathematics)
    - [ROI Detection (Hough Transform)](#roi-detection-hough-circle-transform)
    - [Movement Quantification](#movement-quantification-pixel-differences)
    - [Baseline Thresholds](#baseline-threshold-calculation)
    - [Preprocessing (Detrending, Jump Correction)](#preprocessing-methods)
    - [Hysteresis State Detection](#movement-state-detection-hysteresis)
    - [Activity Fraction & Sleep Bouts](#activity-fraction-and-sleep-detection)
  - [Rhythmic Pattern Analysis](#rhythmic-pattern-analysis-mathematics)
    - [Chi² Periodogram](#chi-periodogram-mathematical-details)
    - [FFT Power Spectrum](#fft-power-spectrum-mathematical-details)
    - [Cosinor Analysis](#cosinor-analysis-mathematical-details)
    - [Method Comparison](#comparison-of-rhythmic-analysis-methods)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Issues](#issues)
- [Acknowledgments](#acknowledgments)

----------------------------------

## Overview

**napari-hdf5-activity** enables automated quantitative analysis of organism behavior in long-term timelapse recordings. Whether you're studying circadian rhythms, sleep-wake cycles, or locomotor activity patterns, this plugin transforms hours of video data into actionable insights.

### Why Use This Plugin?

**For Chronobiology & Behavioral Neuroscience Researchers:**
- Analyze circadian rhythms and ultradian cycles in model organisms (C. elegans, Drosophila, zebrafish, Nematostella, etc.)
- Automatically detect sleep/wake states based on movement patterns
- Quantify activity levels across light/dark cycles with LED-based lighting detection
- Statistical periodogram analysis (Chi² periodogram, FFT) to identify dominant rhythmic patterns

**Key Advantages:**
- **Automated ROI detection**: No manual tracking required - automatically identifies and monitors multiple organisms
- **Optimized for large datasets**: Process hours of high-resolution recordings with multiprocessing (3-4× speedup)
- **Publication-ready outputs**: Excel/CSV files with statistical metrics, publication-quality plots
- **Flexible file support**: Works with both HDF5 timelapse files and standard AVI videos
- **Memory efficient**: Dynamic RAM management adapts to your system's capabilities

**Typical Use Cases:**
- Screening mutant phenotypes for altered activity/sleep patterns
- Monitoring drug effects on circadian behavior
- Analyzing developmental changes in locomotor activity
- Comparing activity patterns across environmental conditions

----------------------------------

## 📚 Additional Documentation

Detailed documentation for advanced features and workflows:

### Analysis & Methods
- **[Extended Analysis Guide](EXTENDED_ANALYSIS.md)** - Comprehensive guide for rhythmic pattern analysis
  - Chi² Periodogram (Bonferroni-corrected, threshold ≈ 15.2)
  - FFT Power Spectrum (permutation significance, 1000 shuffles)
  - Cosinor Analysis (Nelson F-test for population)
  - ROI Similarity Matrix (Bonferroni-corrected pairwise t-test)
  - Coherence Analysis (Welch, Bonferroni-corrected)
  - Phase Clustering (descriptive, PLV heuristics)
  - Scientific rationale and best practices

### File Format Support
- **[AVI Integration Guide](AVI_INTEGRATION_README.md)** - Working with AVI video files
  - Single and batch AVI processing
  - Temporal concatenation
  - Frame interval configuration
  - Best practices for video analysis

### Performance & Optimization
- **[Performance Optimizations](PERFORMANCE_OPTIMIZATIONS.md)** - Performance tuning and benchmarks
  - RGB→Grayscale conversion optimization (10-100× speedup)
  - Dynamic RAM management
  - Worker thread improvements
  - Real-world benchmark results (3.35× speedup with 4 processes)
  - Hardware-specific recommendations

### User Guides
- **[User Guide](docs/USER_GUIDE.md)** - Detailed usage instructions
- **[Circadian Analysis Guide](docs/CIRCADIAN_ANALYSIS.md)** - Circadian rhythm analysis workflows

----------------------------------

## Features

### File Format Support
- **HDF5 files**: Dual structure support (stacked frames and individual frames)
- **Zarr files**: Directory store (`.zarr` directories) support
- **AVI video files**: Single or batch processing with temporal concatenation
- **Memory-efficient loading**: Only first frame loaded for ROI detection, full dataset loaded during analysis

### Analysis Capabilities
- **Automated ROI Detection**: Detect regions of interest (organisms) automatically
- **Movement Analysis**: Pixel-difference based movement quantification
- **Multiple Threshold Methods**:
  - **Baseline**: Uses first N frames to establish baseline activity
  - **Calibration**: Reference-based thresholding from calibration recordings
  - **Adaptive**: Dynamic threshold adjustment during analysis
- **Hysteresis Algorithm**: Robust state detection with upper and lower thresholds
- **Sleep/Wake Detection**: Automated classification of activity states
- **Extended Analysis**: Six complementary circadian rhythm methods
  - **Chi² Periodogram** (Sokolove & Bushell 1978): statistical period detection with Bonferroni-corrected significance (threshold ≈ 15.2 for α=0.05, m=100)
  - **FFT Power Spectrum**: spectral decomposition with permutation significance (1000 shuffles, max-power test)
  - **Cosinor Analysis**: amplitude, MESOR, and acrophase with Nelson et al. (1979) population F-test
  - **ROI Similarity Matrix**: pairwise cross-correlation with Bonferroni-corrected t-test and hierarchical clustering
  - **Coherence Analysis**: Welch magnitude-squared coherence with Bonferroni correction
  - **Phase Clustering**: Hilbert-transform phase extraction with PLV (descriptive)
  - Configurable period range (0-100 hours)
  - Visual plots for all ROIs
- **Frame Viewer**: Interactive dataset playback with export capabilities
  - Frame-by-frame navigation with slider
  - Playback controls with adjustable FPS (1-60 FPS)
  - Time overlay in white text (50% larger for better visibility)
  - **Video/GIF Export**: Export selected frame ranges as MP4 or animated GIF
    - Configurable frame range (start/end frames)
    - Adjustable export FPS
    - Time stamps included in exported videos
  - Support for both HDF5 and AVI datasets
- **Jump Correction**: Detect and correct sudden signal jumps in time-series data
  - Rolling standard deviation-based detection
  - Automatic correction by subtracting jump magnitude
  - Optional feature (can be enabled/disabled)
- **Multiprocessing Support**: Parallel ROI processing for faster analysis
  - Automatic CPU core detection and utilization
  - 2.3x speedup with 4 cores on typical datasets
  - Seamless integration (no configuration needed)

## Recent Updates (2025)

### Latest Branch: refactor/widget-split-zarr-support

- **Zarr support**: Directory store (`.zarr` directories) are now supported alongside HDF5 and AVI files; includes realistic test datasets
- **ROI Editor**: "Edit ROI Circles" button opens a napari Points layer with ring symbols; circles can be moved (but not resized); "Apply Edits" rebuilds the masks from the updated positions
- **Recording duration fix**: Bin size is added to the duration estimate so that 72-hour recordings are recognized correctly rather than appearing as 71.x hours
- **Cycle-count warnings**: Warnings about too few cycles for analysis are logged only — they no longer appear as overlaid text on plots
- **Population cosinor**: Rayleigh test replaced by the Nelson et al. (1979) F-test: F(dfn=2, dfd=2(n−1))

### Major Features Added
- **Jump Correction for Time-Series Preprocessing**: Automatically detect and correct sudden signal jumps caused by equipment vibrations or external disturbances
  - Uses rolling standard deviation-based detection
  - Corrects jumps by subtracting magnitude from subsequent values
  - Optional feature accessible via "Enable Jump Correction" checkbox in Analysis tab
  - Works with both detrending enabled and disabled

- **Frame Viewer Video/GIF Export**: Export selected frame ranges from your recordings
  - Export as MP4 video or animated GIF
  - Configurable frame range (start/end frames)
  - Adjustable export FPS (1-60 FPS)
  - Time stamps automatically included in exported videos
  - Accessible via "Export Video/GIF" section in Frame Viewer tab

- **Improved Frame Viewer Display**: Better visibility for time overlay
  - Time text now displayed in white (previously blue)
  - Font size increased by 50% for easier reading
  - Time format: "Time: X.XX s" in lower-left corner

- **Consolidated Multiprocessing Architecture**: Single unified calculation module
  - All multiprocessing logic integrated into `_calc.py`
  - Pre-calculated baselines passed to worker functions
  - More maintainable codebase without separate parallel module

### Critical Bug Fixes

**IMPORTANT - Baseline Calculation Fix:**
- **Fixed**: Baseline thresholds are now calculated from normalized data **BEFORE** detrending
- **Why this matters**: Detrending removes trends across the entire video, which was distorting baseline calculations. The baseline should reflect the actual signal characteristics at the start of the recording, not detrended values.
- **Impact**: This fix ensures more accurate movement detection thresholds
- **Applies to**: Both Baseline Method and Calibration Method
- **Backward compatible**: This is a bugfix that improves accuracy without changing output format


**Other Bug Fixes:**
- **Fixed**: Performance metrics calculation error (TypeError: start_time was None)
  - Root cause: Qt signal race condition where cleanup happened before metrics capture
  - Solution: Capture start_time at beginning of analysis_finished() method

- **Fixed**: AVI batch processing plot time range auto-update
  - Plots now automatically adjust to full duration of all AVI files
  - Previously only showed first 1000 minutes even for longer recordings

- **Fixed**: Save Results Excel export crash
  - Corrected function call to `_save_results_excel_to_path()`
  - Fixed AttributeError in parameters sheet from incorrect `getattr()` pattern

- **Fixed**: Documentation for hysteresis thresholds
  - Clarified that thresholds are symmetric: `mean ± (multiplier × std)`
  - Updated both parameter table and Movement Calculation section

### Compatibility Notes
- 100% backward compatible with previous versions
- Excel export format unchanged (7 sheets, same structure)
- Metadata export format unchanged
- All bug fixes apply to both old and new file formats
- New features (jump correction, video export) are optional and don't affect existing workflows

## Changelog

### Version 0.3.2 (2025) - Feature/Multiprocessing Merge
**Major Features:**
- **Jump Correction**: Detect and correct sudden signal jumps in time-series data
  - Rolling standard deviation-based detection
  - Optional feature via checkbox
  - Compatible with detrending
- **Frame Viewer Export**: Video/GIF export with frame range selection
  - MP4 and animated GIF support
  - Configurable FPS (1-60)
  - Time stamps included
- **Improved Frame Viewer Display**: White text (was blue), 50% larger font
- **Consolidated Architecture**: Single `_calc.py` module for multiprocessing

**Critical Bug Fixes:**
- **Baseline Calculation**: Now calculated from normalized data BEFORE detrending (both baseline and calibration methods)
  - This was causing incorrect thresholds when detrending was enabled
  - Verified with test suite (difference < 0.000001)
- **Performance Metrics**: Fixed TypeError when start_time was None (Qt signal race condition)
- **AVI Batch Plot Range**: Plot time range now auto-updates to full recording duration
- **Save Results**: Fixed Excel export crash and AttributeError in parameters sheet
- **Documentation**: Clarified symmetric hysteresis thresholds in multiple sections

**Compatibility:**
- 100% backward compatible
- Excel/CSV formats unchanged
- All changes are improvements or bugfixes

### Version 0.3.1 (2025)
- **Multiprocessing support**: True parallel processing for baseline analysis
  - ROI-level parallelization using Python's `multiprocessing.Pool`
  - Automatic core count detection (cpu_count() - 1)
  - 2-5x speedup for multi-ROI datasets
  - Python 3.9+ compatible
- Enhanced "Number of Processes" parameter now functional

### Version 0.3.0 (2025)
- Added Extended Analysis tab with Chi² periodogram (Sokolove & Bushell 1978)
- Periodogram visualization for circadian rhythm detection
- Statistical significance testing for periodic patterns
- Sleep/wake phase identification
- Frame Viewer for interactive dataset playback
- Time overlay in frames (calculated from metadata)
- Playback controls with adjustable FPS
- Flexible period range configuration (0-100 hours)

### Version 0.2.0 (2025)
- Added AVI video file support
- Memory-efficient loading (first frame only for ROI detection)
- Batch processing for multiple AVI files
- LED-based lighting condition detection
- Modular calculation system (_calc.py modules)
- Enhanced metadata handling
- Improved plot generation

### Version 0.1.0 (2024)
- Initial release
- HDF5 dual structure support
- ROI detection
- Movement analysis with multiple threshold methods
- Basic plotting and export

### Visualization
- **Real-time Plots**: Movement traces, activity fractions, sleep patterns
- **Lighting Conditions**: Automatic detection and visualization of light/dark phases from LED data
- **Multi-ROI Display**: Color-coded plots for multiple organisms
- **Export Options**: Save plots as PNG/PDF, export data to Excel/CSV

### Video Processing
- **Frame Interval Control**: Configurable sampling rate (default: 5 seconds)
- **Batch Processing**: Process multiple AVI files as continuous timeseries
- **Temporal Concatenation**: Automatic time offset calculation for sequential videos
- **LED Data Integration**: Extract lighting schedules from metadata

## Installation

### From PyPI

```bash
pip install napari-hdf5-activity
```

### From Source

```bash
git clone https://github.com/s1alknau/napari-hdf5-activity.git
cd napari-hdf5-activity
pip install -e .
```

### Dependencies

Required:
- `napari >= 0.4.17`
- `numpy`
- `h5py`
- `opencv-python` (for AVI support)
- `matplotlib`
- `qtpy`
- `scikit-image`

Optional:
- `pandas` (for Excel export)
- `openpyxl` (for Excel export)

## Quick Start

### 1. Launch napari with plugin

```bash
napari
```

Then: `Plugins` → `napari-hdf5-activity`

### 2. Load Data

**HDF5 File:**
- Click "Load File" → Select `.h5` or `.hdf5` file
- Plugin automatically detects structure type (stacked/individual frames)

**AVI File(s):**
- Click "Load File" → Select one or multiple `.avi` files
- For batch: Hold Ctrl/Cmd and select multiple files
- Files are concatenated temporally (Video1: 0-600s, Video2: 600-1200s, etc.)

**Directory:**
- Click "Load Directory" → Select folder containing HDF5 or AVI files
- All files of same type are loaded automatically

### 3. Detect ROIs

- Navigate to "ROI Detection" tab
- Adjust parameters:
  - Min/Max Radius: Size range of organisms
  - DP Parameter: Detection sensitivity
  - Min Distance: Minimum separation between ROIs
- Click "Detect ROIs"
- ROIs appear as colored circles

### 4. Analyze Movement

- Navigate to "Movement Analysis" tab
- Select threshold method:
  - **Baseline**: For standard recordings
  - **Calibration**: For reference-based analysis
  - **Adaptive**: For variable conditions
- Adjust parameters:
  - Frame Interval: Time between frames (default: 5s)
  - Baseline Duration: Duration for baseline calculation
  - Threshold Multiplier: Sensitivity adjustment
- Click "Process Data"

### 5. Generate Plots

- Navigate to "Results" tab
- Click "Generate Plots"
- View:
  - Movement traces with thresholds
  - Activity fraction over time
  - Lighting conditions (automatic from LED data)
  - Sleep/wake patterns

### 6. Export Results

- Click "Export to Excel" for comprehensive data export
- Click "Save All Plots" for figure export (PNG/PDF)

### 7. Extended Analysis (Circadian Rhythms)

- Navigate to "Extended Analysis" tab
- **Prerequisites**: Run main analysis first (step 4-5)
- Configure parameters:
  - **Minimum Period**: Start of period range (hours, e.g., 12h)
  - **Maximum Period**: End of period range (hours, e.g., 36h)
  - **Significance Level**: Statistical threshold (default: 0.05)
  - **Phase Threshold**: Sleep/wake classification threshold (0-1)
- Click "Run Circadian Analysis"
- View results:
  - **Text Results**: Statistical summary for each ROI
  - **Periodogram Plot**: Visual representation of periodic patterns
  - Green ROI titles indicate significant circadian rhythms
  - Red markers show dominant periods
- Export results to CSV/Excel

### 8. Frame Viewer

- Navigate to "Frame Viewer" tab
- Click "Load Data" to load current dataset into viewer
- Use controls:
  - **Slider**: Navigate to specific frames
  - **|◀ / ◀**: Jump to first/previous frame
  - **▶ Play**: Start/pause playback
  - **▶ / ▶|**: Next frame / jump to last frame
  - **FPS Control**: Adjust playback speed (1-60 FPS)
- Time overlay displayed in frame (red text, lower left)
- Frame info panel shows statistics (shape, min/max, mean)

## Usage Examples

### Example 1: Single HDF5 Recording

**Step-by-step workflow:**

1. **Launch napari and open plugin**
   ```bash
   napari
   ```
   - Menu: `Plugins` → `napari-hdf5-activity: HDF5 Activity Analysis`

2. **Load HDF5 file** (Input tab)
   - Click "Load File"
   - Select your `timelapse.h5` file
   - First frame is displayed in napari viewer

3. **Detect ROIs** (Input tab)
   - Adjust Min/Max Radius based on organism size (e.g., 380-420 pixels)
   - Click "Detect ROIs"
   - Verify that all organisms are detected correctly

4. **Configure analysis** (Analysis tab)
   - Frame Interval: 5.0 seconds (check metadata)
   - Select "Baseline Method" tab
   - Baseline Duration: 200 minutes
   - Threshold Multiplier: 0.10
   - Enable Detrending: ✓

5. **Run analysis**
   - Click "Start Analysis"
   - Wait for processing to complete (~2-5 minutes)

6. **View results** (Results tab)
   - Click "Generate Plots"
   - View movement traces, activity fractions, sleep patterns
   - Click "Export to Excel" to save data

7. **Circadian analysis** (Extended Analysis tab - optional)
   - Set Period Range: Min 12h, Max 36h
   - Click "Run Circadian Analysis"
   - View periodogram plot and statistical results

### Example 2: Batch AVI Processing

**Through napari UI:**
1. Click "Load File"
2. Hold Ctrl/Cmd and select multiple AVI files
3. Plugin loads all videos as temporal batch
4. Detect ROIs on first frame
5. Process Data → analyzes all frames
6. Generate Plots and Export

**Or load from directory:**
1. Click "Load Directory" → Select folder with AVIs
2. All AVI files loaded as batch (sorted alphabetically)
3. Continue with ROI detection and analysis

### Example 3: Calibration-Based Analysis

**Step 1: Load calibration recording**
1. Analysis → Calibration Method tab
2. "Select Calibration File" → Choose calibration.h5
3. "Load Calibration Dataset"

**Step 2: Detect calibration ROIs**
1. Input tab → "Detect ROIs" on calibration data
2. Verify ROI detection

**Step 3: Process calibration baseline**
1. Calibration tab → "Process Calibration Baseline"
2. Baseline statistics are calculated

**Step 4: Load and analyze main dataset**
1. "Select Main Dataset File" → Choose experimental.h5
2. "Load Main Dataset"
3. Detect ROIs on main dataset
4. Process Data (uses calibration thresholds)

## Parameter Guide

### ROI Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Radius | 100 | Minimum organism size (pixels) |
| Max Radius | 120 | Maximum organism size (pixels) |
| DP Parameter | 0.5 | Hough transform sensitivity (lower = more sensitive) |
| Min Distance | 150 | Minimum separation between ROIs (pixels) |
| Param1 (Edge) | 40 | Canny edge detection threshold |
| Param2 (Center) | 40 | Circle center detection threshold |

### Movement Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| Frame Interval | 5.0 s | Time between analyzed frames |
| Baseline Duration | 5.0 min | Duration for baseline calculation |
| Threshold Multiplier | 0.1 | Sensitivity factor for movement detection |
| Upper Threshold Factor | 1.0 | Hysteresis upper threshold |
| Lower Threshold Factor | 1.0 | Hysteresis lower threshold |
| Chunk Size | 20 | Frames per processing chunk |
| Num Processes | 4 | Number of CPU cores for parallel processing |

### Threshold Methods

**Baseline Method:**
- Uses first N minutes of recording
- Calculates mean + (multiplier × std) for each ROI
- Best for: Stable conditions, single recordings

**Calibration Method:**
- Uses separate calibration recording
- Transfers thresholds to experimental data
- Best for: Multiple recordings, standardized protocols

**Adaptive Method:**
- Dynamically adjusts thresholds
- Sliding window baseline calculation
- Best for: Variable conditions, long recordings

## AVI File Support

### Frame Sampling

AVI videos are sampled at configurable intervals (default: 5 seconds):

| Video FPS | Interval | Frames Sampled | Effective FPS |
|-----------|----------|----------------|---------------|
| 30 FPS    | 5s       | Every 150th    | 0.2 FPS       |
| 5 FPS     | 5s       | Every 25th     | 0.2 FPS       |

Frame interval is automatically calculated based on video FPS and target interval (default: 5s).

### Memory Efficiency

- **Loading**: Only first frame loaded (~2 MB instead of 500 MB)
- **ROI Detection**: Performed on first frame
- **Analysis**: All frames loaded on-demand during processing
- **Benefit**: Fast UI, minimal memory footprint for preview

## Output Files

### Excel Export

**Filename**: `[original]_analysis_[timestamp].xlsx`

**Sheets**:
- `Movement_Data`: Raw movement values per ROI
- `Activity_Fraction`: Percentage of time active
- `Sleep_Data`: Sleep bout detection
- `Quiescence_Binned`: Quiescence in time bins
- `Statistics`: Summary statistics per ROI
- `Parameters`: Analysis parameters used
- `Metadata`: File and recording information

### Plot Export

**Filename**: `[original]_plots_[timestamp]/`

**Files**:
- `movement_traces.png`: Movement over time
- `activity_fraction.png`: Activity percentage
- `lighting_conditions.png`: Light/dark phases
- `sleep_pattern.png`: Sleep bout visualization
- Each plot also saved as `.pdf`

## Technical Details

### HDF5 Structure Support

**Stacked Frames:**
```
/frames [dataset: (N, H, W) or (N, H, W, C)]
```

**Individual Frames:**
```
/frames/
  ├── frame_0000
  ├── frame_0001
  └── ...
```

### Complete Processing Pipeline

This section explains the full analysis pipeline from raw video frames to behavioral classifications.

#### Step 1: ROI-Level Movement Quantification

**Input**:
- Raw video frames (grayscale, 8-bit or 16-bit)
- Pre-detected ROIs with circular masks (created during ROI detection phase)

**Process Description**:
This step quantifies how much each organism moves by comparing consecutive video frames. The algorithm calculates the absolute difference in pixel brightness between each frame and the previous frame, focusing only on pixels within each detected ROI (Region of Interest - the circular area around each organism).

**Prerequisites**:
- ROI Detection must be performed FIRST to create circular masks
- Each mask defines which pixels belong to each organism
- Mask is binary: 1 for pixels inside ROI circle, 0 for pixels outside

**Detailed Algorithm**:
```
Setup (done once during ROI detection):
  Create circular mask for each ROI
  → Binary mask: 1 for pixels inside circle, 0 for pixels outside
  → Example: ROI with radius 100px contains ~31,400 pixels

For each frame t (repeated for all frames):
  For each ROI:
    1. Extract ROI pixels from both frames using mask:
       → pixels_current = frame[t][mask_bool]
       → pixels_previous = frame[t-1][mask_bool]
       → Only pixels inside the circular ROI are extracted
       → Example: ROI with 1000 pixels → extract 1000 values from each frame

    2. Calculate pixel-wise differences (ONLY for ROI pixels):
       → diff_pixels = abs(pixels_current - pixels_previous)
       → Compare frame-to-frame change for each pixel in ROI
       → Example: If pixel was 100, now 115 → difference = 15
       → IMPORTANT: Background pixels are NEVER computed!

    3. Sum absolute differences: total_change = sum(diff_pixels)
       → Add up all pixel changes within the ROI
       → Gives total amount of change in the organism's area
       → Example: If 1000 pixels each changed by ~15 → total ≈ 15000

    4. Normalize by ROI area: movement_value = total_change / roi_pixel_count
       → Divide by number of pixels in ROI to get average change per pixel
       → This makes values comparable between different sized organisms
       → Example: 15000 / 1000 pixels = 15.0 average change per pixel
```

**What the Numbers Mean**:
- **Movement Value**: Average pixel intensity change per pixel within ROI
- **Theoretical Range**: 0 to 255 (for 8-bit images) or 0 to 65535 (for 16-bit images)
- **Typical Observed Range**: Depends on organism size, contrast, and ROI area
  - Small organisms (100-500 pixels): Often 50-500
  - Medium organisms (500-2000 pixels): Often 100-2000
  - Large organisms (>2000 pixels): Often 500-5000+
- **Interpretation** (RELATIVE to your baseline thresholds, not absolute):
  - Values are dataset-specific and depend on:
    - **Organism contrast**: High contrast organisms produce higher values
    - **ROI size**: Larger ROIs accumulate more pixel changes
    - **Movement type**: Fast vs. slow, whole-body vs. partial
  - **Use baseline thresholds to classify movement**, not absolute numbers
  - Example from your data: Baseline mean ~130, std ~20, threshold ~132
    - Below 130: Quiescent
    - Above 132: Active
    - Values typically range 100-200 in this example
- **Example**: Movement value of 150 with baseline mean=130 indicates organism is active (above threshold)

**Physical Meaning**:
When an organism moves, its body position changes relative to the background. Dark pixels become light (or vice versa) as the organism moves across the frame. The movement value captures the magnitude of these brightness changes as a proxy for physical movement.

**Y-Axis in Movement Plots**: "Movement (pixel intensity change)" - raw pixel difference values

#### Step 2: Data Normalization

**Input**: Raw movement values from Step 1

**Processing**:
```
For each ROI:
  normalized_data = raw_movement_values (no minimum subtraction)
```

**Note**: The data represents direct frame-to-frame pixel changes.

**Units After Normalization**: Same as raw movement values (0-255 or 0-65535)

#### Step 3: Baseline Calculation (CRITICAL - Before Detrending!)

**Input**: Normalized movement data

**Baseline Window**: First N minutes of recording (configurable, default: 200 minutes)

**Calculation**:
```
For each ROI:
  baseline_window = normalized_data[0:baseline_duration]
  baseline_mean = mean(baseline_window)
  baseline_std = std(baseline_window)

  upper_threshold = baseline_mean + (multiplier × baseline_std)
  lower_threshold = baseline_mean - (multiplier × baseline_std)
```

**Why Before Detrending?**: Detrending removes long-term trends across the ENTIRE video, which would distort the baseline thresholds. The baseline should reflect actual signal characteristics during the baseline period, not detrended values.

**Units**: Same as movement values (pixel intensity change)

**Threshold Interpretation**:
- **Upper Threshold**: Movement above this = organism is ACTIVE
- **Lower Threshold**: Movement below this = organism is QUIESCENT
- **Between Thresholds**: Hysteresis zone - state remains unchanged (prevents flickering)

**Example**:
```
Baseline Mean = 12.5 (average pixel change during baseline)
Baseline Std = 3.2
Multiplier = 0.1

Upper Threshold = 12.5 + (0.1 × 3.2) = 12.82
Lower Threshold = 12.5 - (0.1 × 3.2) = 12.18
```

#### Step 4: Optional Preprocessing

**4a. Jump Correction (Optional)**:
```
If enabled:
  For each ROI:
    1. Calculate rolling std deviation (window = min(20 frames, len/5))
    2. Find frame-to-frame differences: diff = values[t] - values[t-1]
    3. Detect jumps: abs(diff) > jump_threshold_factor × median(rolling_std)
    4. Correct jumps: values[t:] -= jump_magnitude
```

**When to Use**: When equipment vibrations or external disturbances cause sudden signal shifts

**4b. Detrending (Optional)**:
```
If enabled:
  For each ROI:
    1. Fit polynomial trend line (degree 1-3) to entire dataset
    2. Subtract trend: detrended_data = normalized_data - trend_line
```

**When to Use**: When long-term signal drift occurs (photobleaching, LED intensity changes)

**Important**: Baseline thresholds calculated in Step 3 are NOT recalculated after detrending - they are preserved from the original normalized data.

#### Step 5: Movement State Detection (Hysteresis Algorithm)

**Input**: Processed data + baseline thresholds (from Step 3)

**Hysteresis State Machine**:
```
For each time point t:
  current_value = processed_data[t]

  If current_state == QUIESCENT:
    if current_value > upper_threshold:
      current_state = MOVEMENT
      movement_binary[t] = 1
    else:
      movement_binary[t] = 0

  Elif current_state == MOVEMENT:
    if current_value < lower_threshold:
      current_state = QUIESCENT
      movement_binary[t] = 0
    else:
      movement_binary[t] = 1
```

**Output**: Binary movement array (0 = quiescent, 1 = movement)

**Why Hysteresis?**: Prevents rapid state flickering due to noise. State only changes when signal crosses both upper AND lower thresholds.

**Y-Axis in Binary Movement Plots**: 0 (quiescent) or 1 (movement)

#### Step 6: Activity Fraction Calculation

**Input**: Binary movement array

**Time Binning**:
```
bin_size = 60 seconds (default, configurable)

For each time bin:
  movement_frames = count(movement_binary[bin] == 1)
  total_frames = count(all_frames[bin])

  activity_fraction[bin] = movement_frames / total_frames
```

**Units**: Fraction (0.0 to 1.0) or Percentage (0% to 100%)

**Y-Axis in Activity Fraction Plots**: "Activity Fraction" (0.0-1.0) or "% Active" (0-100%)

**Example**:
```
Bin size = 60 seconds
Frame interval = 5 seconds
Frames per bin = 60/5 = 12 frames

If 8 out of 12 frames show movement:
Activity fraction = 8/12 = 0.667 (66.7%)
```

#### Step 7: Quiescence Detection

**Input**: Activity fraction data

**Threshold-Based Classification**:
```
quiescence_threshold = 0.5 (default, configurable)

For each time bin:
  if activity_fraction[bin] < quiescence_threshold:
    quiescence_binary[bin] = 1  # Organism is quiescent
  else:
    quiescence_binary[bin] = 0  # Organism is active
```

**Y-Axis in Quiescence Plots**: Binary (0 = active, 1 = quiescent)

#### Step 8: Sleep Bout Detection

**Input**: Quiescence binary data

**Temporal Consolidation**:
```
sleep_threshold_minutes = 8 (default, configurable)

For each quiescence period:
  if duration >= sleep_threshold_minutes:
    classify_as_SLEEP
  else:
    classify_as_SHORT_QUIESCENCE (not sleep)
```

**Output**: Sleep bouts with start time, end time, and duration

**Units**:
- **Start/End Time**: Minutes or hours from recording start
- **Duration**: Minutes

**Sleep Bout Characteristics**:
- **Minimum duration**: Defined by sleep_threshold_minutes parameter
- **Represents**: Sustained periods of inactivity (behavioral sleep)
- **Excludes**: Brief pauses in activity

#### Summary: Data Flow and Units

```
Raw Frames (8-bit grayscale)
  ↓
Movement Values (0-255 pixel intensity change)
  ↓
Normalized Data (0-255, no minimum subtraction)
  ↓
Baseline Thresholds (mean ± multiplier×std, in pixel units)
  ↓
[Optional: Jump Correction + Detrending]
  ↓
Binary Movement (0 or 1)
  ↓
Activity Fraction (0.0-1.0, per 60-second bin)
  ↓
Quiescence Binary (0 or 1)
  ↓
Sleep Bouts (start, end, duration in minutes)
```

#### Understanding Plot Y-Axes

1. **Movement Trace Plot**:
   - Y-axis: "Movement (pixel intensity change)"
   - Units: Average pixel value change per pixel within ROI
   - Range: 0-255 (8-bit) or 0-65535 (16-bit)
   - Horizontal lines: Upper/lower thresholds from baseline

2. **Activity Fraction Plot**:
   - Y-axis: "Activity Fraction" or "% Active"
   - Units: Fraction (0.0-1.0) or Percentage (0-100%)
   - Time bins: Default 60 seconds

3. **Sleep Pattern Plot**:
   - Y-axis: ROI labels
   - Horizontal bars: Sleep bouts
   - Bar length: Sleep duration (minutes)

4. **Periodogram (Chi² / Fisher Analysis)**:
   - X-axis: "Period (hours)"
   - Y-axis: Labeled "Z-score" in the plot, but the quantity is the chi-squared statistic Z(T) = n × (r_cos² + r_sin²), which follows χ²(df=2) under H₀
   - Range: 0 to ~30+ (higher = stronger rhythm)
   - Horizontal dashed line: Bonferroni-corrected significance threshold ≈ 15.2 (for α=0.05, m=100 tested periods)

### Multiprocessing Implementation and Performance

The plugin implements **true multiprocessing** (not multithreading) using Python's `multiprocessing` module to bypass the Global Interpreter Lock (GIL) and achieve genuine parallel execution on multi-core CPUs.

#### Why Multiprocessing Instead of Multithreading?

**Python's Global Interpreter Lock (GIL) Problem:**
- Python's GIL prevents multiple threads from executing Python bytecode simultaneously
- Multithreading in Python is only useful for I/O-bound tasks (waiting for disk, network, etc.)
- For CPU-bound tasks (like our pixel-difference calculations), multithreading provides NO speedup
- The GIL essentially serializes CPU-bound operations across threads

**Our Solution: Multiprocessing**
- `multiprocessing` module creates separate Python processes (not threads)
- Each process has its own Python interpreter and memory space
- Each process has its own GIL, so they can run truly in parallel
- Ideal for CPU-bound tasks like ROI movement detection
- Fully tested and optimized for Windows 11 with Python 3.9+

#### Technical Implementation

**Complete Processing Pipeline:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN PROCESS (GUI)                           │
│                   Single Python Interpreter                     │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: PREPROCESSING (Sequential - Cannot be parallelized)  │
├─────────────────────────────────────────────────────────────────┤
│  1. Create ROI masks (circular masks for each detected ROI)    │
│     → Defines which pixels belong to each organism             │
│                                                                 │
│  2. Load HDF5/AVI frames                                        │
│                                                                 │
│  3. Calculate pixel differences WITHIN each ROI mask:           │
│     For each ROI:                                               │
│       diff = abs(frame[t] - frame[t-1]) * circular_mask        │
│       movement = sum(diff) / count(mask_pixels)                │
│     → Only pixels inside ROI are used for calculation          │
│                                                                 │
│  4. Normalize data (optional)                                   │
│                                                                 │
│  5. Calculate baseline thresholds (mean ± std)                  │
│     → Must use data from ALL ROIs                               │
│                                                                 │
│  6. Optional: Detrending & Jump Correction                      │
│                                                                 │
│  Time: ~2-3 seconds for 10 ROIs, 10000 frames                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │    PARALLELIZATION DECISION            │
         │  IF (num_processes > 1) AND (rois ≥ 2) │
         └────────────────────────────────────────┘
                      ▼           ▼
              ┌───────┘           └───────┐
              │ YES                   NO  │
              ▼                           ▼
┌──────────────────────────┐   ┌──────────────────────┐
│  PARALLEL PATH           │   │  SEQUENTIAL PATH     │
│  (4 Worker Processes)    │   │  (Main Process)      │
└──────────────────────────┘   └──────────────────────┘
              ▼                           ▼
┌──────────────────────────┐   ┌──────────────────────┐
│  CREATE PROCESS POOL     │   │  Process Each ROI    │
│  (~2 seconds overhead)   │   │  in Main Process     │
├──────────────────────────┤   │                      │
│  Worker 1  Worker 2      │   │  For each ROI:       │
│  Worker 3  Worker 4      │   │  • Movement Detection│
│                          │   │  • Activity Fraction │
│  Each = New Python       │   │                      │
│         Interpreter!     │   │  Time: ~26s          │
└──────────────────────────┘   │  (10 ROIs)           │
              ▼                 └──────────────────────┘
┌──────────────────────────┐              │
│  DISTRIBUTE ROI TASKS    │              │
├──────────────────────────┤              │
│  Task Queue:             │              │
│  [ROI 0][ROI 1][ROI 2]   │              │
│  [ROI 3][ROI 4][ROI 5]   │              │
│  [ROI 6][ROI 7][ROI 8]   │              │
│  [ROI 9]                 │              │
└──────────────────────────┘              │
              ▼                            │
┌──────────────────────────┐              │
│  PARALLEL EXECUTION      │              │
├──────────────────────────┤              │
│                          │              │
│  Worker 1: ROI 0 → ROI 4 │              │
│  Worker 2: ROI 1 → ROI 5 │              │
│  Worker 3: ROI 2 → ROI 6 │              │
│  Worker 4: ROI 3 → ROI 7 │              │
│                          │              │
│  Each Worker:            │              │
│  1. Movement Detection   │              │
│     (Hysteresis)         │              │
│  2. Activity Fraction    │              │
│     (Binning)            │              │
│                          │              │
│  Time: ~7s (parallel)    │              │
│  vs. ~26s (sequential)   │              │
└──────────────────────────┘              │
              ▼                            │
┌──────────────────────────┐              │
│  COLLECT RESULTS         │              │
├──────────────────────────┤              │
│  From all 4 workers:     │              │
│  • ROI 0 results         │              │
│  • ROI 1 results         │              │
│  • ...                   │              │
│  • ROI 9 results         │              │
└──────────────────────────┘              │
              ▼                            │
              └────────────┬───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: POST-PROCESSING (Sequential - Needs all ROI data)    │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Quiescence detection (compare activity across ROIs)         │
│  ✓ Sleep bout identification (temporal consolidation)          │
│  ✓ ROI color assignment for plots                              │
│  ✓ Statistical calculations                                    │
│                                                                 │
│  Time: ~1 second                                                │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESULTS READY                                │
│  • Movement data per ROI                                        │
│  • Activity fractions                                           │
│  • Sleep bouts                                                  │
│  • Statistics                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **Phase 1 (Preprocessing)**: MUST be sequential - needs all data at once
- **Phase 2 (ROI Processing)**: CAN be parallelized - each ROI is independent
- **Phase 3 (Post-processing)**: MUST be sequential - needs all results together

**Time Breakdown Example (10 ROIs, 10000 frames):**

| Phase | Sequential (1 core) | Parallel (4 cores) | Saved Time |
|-------|--------------------|--------------------|------------|
| **Preprocessing** | 2s | 2s | 0s (cannot parallelize) |
| **Overhead** | 0s | 2s | -2s (process creation) |
| **ROI Processing** | 26s | 7s | +19s (4x faster) |
| **Post-processing** | 1s | 1s | 0s (cannot parallelize) |
| **TOTAL** | **29s** | **12s** | **+17s saved** |

**Speedup: 2.42x** (not 4x due to overhead and sequential portions)

**Code Implementation** (from `_calc.py:1343-1380`):

**Step 1: Automatic Parallelization Decision**
```python
# Decide: Parallel or Sequential?
use_parallel = num_processes > 1 and len(processed_data) >= 2

if use_parallel:
    print(f"Using parallel processing with {num_processes} processes")
else:
    print("Using sequential processing (single core)")
```

**Step 2: Worker Function** (`_process_single_roi_movement`):
```python
def _process_single_roi_movement(args):
    """
    Worker function executed in separate process.

    Each worker receives:
    - roi_id: ROI identifier (0, 1, 2, ...)
    - data: Preprocessed movement values for this ROI
    - baseline_mean: Pre-calculated baseline (from Step 3)
    - upper_threshold: Pre-calculated threshold (from Step 3)
    - lower_threshold: Pre-calculated threshold (from Step 3)
    - bin_size_seconds: Time binning parameter
    - frame_interval: Frame sampling rate

    Returns:
    - (roi_id, results_dict) with movement_data and fraction_data
    """
    (roi_id, data, baseline_mean, upper_threshold,
     lower_threshold, bin_size_seconds, frame_interval) = args

    # Step 1: Hysteresis movement detection
    movement_data = define_movement_with_hysteresis(...)

    # Step 2: Bin fraction movement
    fraction_data = bin_fraction_movement(...)

    return roi_id, {"movement_data": movement_data,
                    "fraction_data": fraction_data}
```

**Step 3: Parallel Execution with Process Pool**
```python
if use_parallel:
    # Prepare arguments for each ROI
    roi_args = [
        (roi_id, processed_data[roi_id], baseline_means[roi_id],
         upper_thresholds[roi_id], lower_thresholds[roi_id],
         bin_size_seconds, frame_interval)
        for roi_id in processed_data.keys()
    ]

    # Create process pool and distribute work
    with Pool(processes=num_processes) as pool:
        roi_results = pool.map(_process_single_roi_movement, roi_args)

    # Aggregate results from workers
    movement_data = {}
    fraction_data = {}
    for roi_id, results in roi_results:
        movement_data[roi_id] = results["movement_data"]
        fraction_data[roi_id] = results["fraction_data"]
```

#### Why This Design Works

**1. Preprocessing is Sequential (Necessary)**
- Frame loading from disk: I/O-bound, no benefit from parallelization
- Baseline calculation: Needs data from ALL ROIs, can't parallelize
- Detrending/jump correction: Per-ROI but fast, overhead not worth it

**2. Movement Detection is Parallel (CPU-Intensive)**
- Hysteresis algorithm: Independent per ROI, perfect for parallelization
- Activity fraction calculation: Per-ROI, no dependencies
- Each ROI takes ~1-5 seconds on typical datasets
- Linear scaling: 4 cores → 4x speedup (minus small overhead)

**3. Post-Processing is Sequential (Cross-ROI)**
- Quiescence/sleep detection: Needs comparison across ROIs
- Plot generation: GUI operations, must be in main thread
- Result aggregation: Fast, not worth parallelizing

#### Performance Characteristics

**Speedup Measurements** (actual test results on Windows 10, Python 3.9):

**Test 1: Large Dataset (10 ROIs, 10000 frames ~ 14 hours of recording)**
```
Configuration          Time      Speedup    CPU Usage
────────────────────────────────────────────────────
1 process (sequential)  29.9s    1.0x       ~25% (1 core)
2 processes             20.7s    1.45x      ~50% (2 cores)
4 processes             12.4s    2.42x      ~90% (4 cores)
8 processes             13.2s    2.27x      ~95% (all cores)
```

**Test 2: Small Dataset (6 ROIs, 5000 frames ~ 7 hours of recording)**
```
Configuration          Time      Speedup    CPU Usage
────────────────────────────────────────────────────
1 process (sequential)   4.5s    1.0x       ~25% (1 core)
2 processes              4.9s    0.92x      ~50% (2 cores)
4 processes              4.2s    1.05x      ~90% (4 cores)
8 processes              3.9s    1.15x      ~95% (all cores)
```

**Key Insight: Multiprocessing Benefit Depends on Dataset Size**
- **Large datasets (>8000 frames)**: Significant speedup (2.4x with 4 cores)
- **Small datasets (<6000 frames)**: Minimal benefit or slower due to overhead
- **Threshold**: Multiprocessing becomes beneficial at ~7000-8000 frames
- **Recommendation**: For recordings <5 hours, use 1 process (sequential)

**Why Not Linear Speedup?**
1. **Process Creation Overhead**: ~0.5-1 second to spawn processes (fixed cost)
2. **Data Serialization**: Arguments must be pickled and sent to workers
3. **Result Collection**: Results must be collected and deserialized
4. **Amdahl's Law**: Sequential portions (preprocessing, post-processing) limit speedup
5. **Dataset Size Matters**: Overhead dominates for small datasets

**Optimal Number of Processes:**
- **Rule of thumb**: `num_processes = cpu_count() - 1`
- **Why -1?**: Leaves one core for OS and GUI responsiveness
- **Diminishing returns**: Beyond 4-8 processes, overhead dominates
- **Memory consideration**: Each process needs ~200-500 MB RAM

#### When Parallel Processing is Used

**Automatic Criteria:**
```python
use_parallel = (num_processes > 1) AND (num_rois >= 2)
```

**Examples:**
- ✅ 6 ROIs, 4 processes → Parallel (4 workers, 2 ROIs per batch)
- ✅ 2 ROIs, 2 processes → Parallel (each ROI on separate core)
- ❌ 1 ROI, 4 processes → Sequential (only 1 ROI, can't parallelize)
- ❌ 6 ROIs, 1 process → Sequential (user disabled parallel)

**Currently Supported Methods:**
- ✅ **Baseline Method**: Full parallel support
- ❌ **Calibration Method**: Sequential (uses cross-ROI baseline transfer)
- ❌ **Adaptive Method**: Sequential (sliding window across time, not ROIs)

#### Memory Considerations

**Process Memory Model:**
- Each process gets **copy** of input data (not shared memory)
- Memory usage: `base_memory + (num_processes × per_roi_memory)`
- Typical per-ROI memory: ~100-300 MB for 10,000 frames
- Example: 6 ROIs, 4 processes, 10k frames → ~2-3 GB total RAM

**Why Not Shared Memory?**
- Python's `multiprocessing.shared_memory` (Python 3.8+) has limited NumPy support
- Pickling overhead is acceptable for typical dataset sizes (< 1 GB per ROI)
- Simplicity and cross-platform compatibility prioritized over max performance

#### Limitations and Future Work

**Current Limitations:**
1. **ROI-level parallelization only**: Can't parallelize single very large ROI
2. **No GPU support**: All processing on CPU (adding GPU would require CUDA/OpenCL)
3. **Calibration/Adaptive not parallelized**: These methods have cross-ROI dependencies

**Potential Future Improvements:**
1. **Frame-level parallelization**: For single-ROI datasets, parallelize across time
2. **Hybrid parallelization**: Combine ROI-level and frame-level for very large datasets
3. **GPU acceleration**: Use PyTorch/CuPy for pixel difference calculations
4. **Shared memory**: Use `multiprocessing.shared_memory` for very large datasets (>5 GB)

**Why Not Implemented Yet:**
- Current implementation handles 95% of use cases efficiently
- Added complexity not justified by typical dataset sizes
- Cross-platform compatibility is priority (GPU requires CUDA, which is NVIDIA-only)

#### Debugging and Monitoring

**Enable Parallel Processing Logging:**
The plugin automatically logs when parallel processing is used:
```
[INFO] Using parallel processing with 4 processes for 6 ROIs
[INFO] ROI 0 processing time: 1.8s
[INFO] ROI 1 processing time: 1.9s
...
[INFO] Total parallel processing time: 2.1s (includes overhead)
```

**Common Issues:**

1. **"Parallel processing enabled but no speedup"**
   - Cause: Dataset too small, overhead dominates
   - Solution: Use sequential processing for <1000 frames

2. **"Memory error with 8 processes"**
   - Cause: Each process copies full dataset
   - Solution: Reduce `num_processes` or process fewer ROIs at once

3. **"GUI freezes during analysis"**
   - Cause: Not an issue! Multiprocessing runs in background
   - GUI should remain responsive even during heavy computation

#### Implementation Notes (Windows 11)

**Process Creation Method:**
- Uses `spawn` method (starts fresh Python interpreter for each worker)
- Process creation overhead: ~1-2 seconds for 4 workers
- `if __name__ == '__main__':` guard handled internally by the plugin
- Each worker process gets its own copy of input data (no shared memory)

#### Performance Guidelines

**When to Use Parallel Processing:**
- ✅ Multiple ROIs (≥2) to process
- ✅ Large datasets with long recordings (>8000 frames / >10 hours)
- ✅ Multi-core CPU available (≥2 cores)
- ✅ Baseline analysis method

**When NOT to Use Parallel Processing:**
- ❌ Small datasets (<6000 frames / <8 hours) - overhead exceeds benefit
- ❌ Single ROI - no parallelization possible
- ❌ Calibration/Adaptive methods - not yet parallelized

**Recommended Settings (for large datasets >8000 frames):**
```
Number of ROIs    Recommended Processes    Expected Speedup
──────────────────────────────────────────────────────────
1 ROI             1 (sequential)           1.0x (no benefit)
2-3 ROIs          2-3                      1.3-1.6x
4-6 ROIs          4                        1.8-2.4x
7-12 ROIs         4-6                      2.0-2.8x
>12 ROIs          6-8                      2.2-3.0x
```

**Important Notes:**
- Beyond 8 processes, overhead typically outweighs benefits
- Speedup values are for large datasets (>8000 frames)
- Small datasets (<6000 frames) show minimal or negative speedup

#### AVI Batch Processing and Multiprocessing

The AVI batch processing pipeline **also uses multiprocessing** for the baseline analysis step:

**AVI Processing Pipeline:**
1. **Frame Loading** (sequential): Videos are loaded sequentially to maintain temporal order
   - Uses streaming analysis to minimize memory usage
   - Loads, analyzes, and discards frames in chunks
   - Multiple videos can be loaded in parallel using `ThreadPoolExecutor` (I/O-bound)

2. **Movement Detection** (sequential): Frame-to-frame differences calculated during streaming
   - Processes one video at a time to preserve temporal continuity
   - Chunk-based processing (default: 100 frames per chunk)

3. **Baseline Analysis** (parallel multiprocessing): After ROI movement data is collected, the same multiprocessing system is used
   - Calls `run_baseline_analysis()` from `_calc.py` with `num_processes` parameter
   - Distributes ROIs across CPU cores (same as HDF5 processing)
   - Each process independently calculates: hysteresis thresholds, movement classification, binned fraction data

**Key Differences Between HDF5 and AVI Processing:**
- **HDF5**: Frame loading can be parallelized across ROIs, entire pipeline uses multiprocessing
- **AVI**: Frame loading is sequential (maintains temporal order), but baseline analysis uses multiprocessing

**Why Sequential Frame Loading for AVI?**
- AVI videos must be processed in temporal order to construct continuous timeseries
- Video files are accessed sequentially by nature (seeking is expensive)
- Streaming approach prevents memory overflow with large video batches
- Multiprocessing is applied where it matters most: computationally intensive baseline analysis

**Performance Example (10 ROIs, 12 AVI files, ~60 minutes total, 10000 frames):**
```
Stage                     Time      Parallel Method
────────────────────────────────────────────────────────
Frame Loading             ~3-5 min  ThreadPoolExecutor (I/O)
Movement Detection        ~2-3 min  Sequential (streaming)
Baseline Analysis         ~12 sec   multiprocessing.Pool (CPU, 4 cores)
  (sequential)            ~30 sec   Single core
────────────────────────────────────────────────────────
Total (with 4 cores)      ~5-8 min  Hybrid parallel approach
Total (sequential)        ~6-9 min
Speedup from multiproc    ~18 sec   Saved in baseline analysis stage
```

The multiprocessing speedup for baseline analysis is the same as HDF5 processing (2.4x with 4 cores for large datasets), but represents a smaller fraction of total AVI processing time because video decoding dominates.

**Note:** For small AVI batches (<5 hours total), baseline analysis takes only 3-5 seconds and multiprocessing overhead may exceed benefits. Use sequential processing (1 core) for such cases.

### LED-Based Lighting Detection

- **Light Phase**: White LED power > 0.5%
- **Dark Phase**: White LED power ≤ 0.5% (IR only)
- **IR LED**: Continuous 100% for video recording
- **Source**: HDF5 timeseries only (not available for AVI files)

## Troubleshooting

### Issue: No ROIs detected

**Solutions:**
- Adjust Min/Max Radius to match organism size
- Decrease DP Parameter (e.g., 0.3) for more sensitivity
- Check first frame contrast (use "Debug HDF5 Structure")

### Issue: AVI files not loading

**Solutions:**
- Install opencv: `pip install opencv-python`
- Verify AVI codec is supported (MJPEG, H264, etc.)
- Check if file is corrupted

### Issue: "Structure detection failed" error

**Solutions:**
- File is likely AVI, not HDF5 - use Load File for AVIs
- Check HDF5 file integrity with `h5py`
- Try "Load Directory" for batch processing

### Issue: Analysis very slow

**Solutions:**
- Increase Chunk Size (e.g., 100 frames)
- Reduce Num Processes (memory vs. speed tradeoff)
- Use smaller time window (Start/End Time)
- For AVI: Consider reducing frame interval

### Issue: Memory error during AVI analysis

**Solutions:**
- Process fewer videos at once
- Increase frame interval (e.g., 10s instead of 5s)
- Reduce Chunk Size
- Close other applications

## Scientific Background

### Chi² Periodogram for Circadian Rhythm Detection

The plugin implements the **Chi² periodogram** (Sokolove & Bushell 1978) for detecting periodic patterns in activity data, particularly useful for identifying circadian rhythms in biological timeseries. This method tests for correlations between the time series and sine/cosine waves at different candidate periods.

#### What is a Periodogram?

A periodogram is a statistical tool that identifies periodic (repeating) patterns in timeseries data. It answers two key questions:
1. **Does the organism show rhythmic activity?** (statistical significance test)
2. **If so, at what period?** (e.g., 24 hours for circadian rhythms)

#### Mathematical Foundation

**Input Requirements**:
- Time series data: Activity fraction over time (from Step 6 of processing pipeline)
- Minimum 10 data points required
- Longer recordings provide better resolution (recommended: ≥2-3 days for circadian analysis)

**Algorithm Steps**:

1. **Period Testing Range**:
   ```
   Default: 12-36 hours (captures circadian and ultradian rhythms)
   Number of test periods: 100 (evenly spaced)

   Example periods tested:
   12.0h, 12.24h, 12.48h, ..., 35.76h, 36.0h
   ```

2. **For Each Test Period T**:
   ```
   a. Calculate angular frequency: ω = 2π / T
   b. Generate cosine wave: cos_wave = cos(ω × t)
   c. Generate sine wave: sin_wave = sin(ω × t)
   d. Calculate Pearson correlations:
      - r_cos = correlation(activity_data, cos_wave)
      - r_sin = correlation(activity_data, sin_wave)
   e. Compute chi-squared statistic: Z(T) = n × (r_cos² + r_sin²)
      (where n = number of data points)
      NOTE: the plot y-axis is labeled "Z-score" but the quantity is Z(T)
   ```

3. **Statistical Significance with Bonferroni Correction**:
   ```
   Z(T) follows chi-square distribution (df=2) under H₀

   With m = 100 tested periods, Bonferroni-corrected threshold at α = 0.05:
   Critical Z ≈ 15.2   (= χ²(1 − 0.05/100, df=2))

   Interpretation:
   - Z(T) > 15.2: Statistically significant rhythm (Bonferroni-corrected)
   - Z(T) < 15.2: No significant rhythm detected

   The uncorrected threshold (5.99) is NOT used — testing 100 periods without
   correction would yield ~5 false positives per analysis on average.
   ```

4. **Dominant Period Identification**:
   ```
   Dominant period = period with maximum Z-score
   p-value = 1 - χ²_cdf(max_Z, df=2)
   ```

#### Periodogram Plot Interpretation

**X-Axis: Period (hours)**
- Range: Minimum period (e.g., 12h) to Maximum period (e.g., 36h)
- Resolution: 100 test points
- Covers circadian (24h) and ultradian (<24h) rhythms

**Y-Axis: Chi-squared statistic Z(T) (labeled "Z-score" in plot)**
- Range: Typically 0 to 30+
- **Z(T) > 15.2**: Statistically significant after Bonferroni correction (α=0.05, m=100)
- **Z(T) > 20**: Highly significant
- Higher Z(T) values indicate stronger, more consistent rhythms

**Visual Elements**:
1. **Horizontal dashed line**: Bonferroni-corrected threshold ≈ 15.2 (α=0.05, m=100)
2. **Red marker**: Dominant period (peak Z(T))
3. **Green title**: ROI has significant rhythm (above Bonferroni threshold)
4. **Black title**: No significant rhythm detected

#### Example Interpretations

**Case 1: Strong Circadian Rhythm**
```
Periodogram shows:
- Sharp peak at 24.0 hours
- Z-score = 18.5 (well above 5.99)
- p-value = 0.0001

Interpretation:
Organism exhibits robust 24-hour circadian rhythm, likely entrained to
light/dark cycles. High Z-score indicates consistent phase relationship
across entire recording.

Biological meaning:
- Strong clock-driven behavior
- Reliable entrainment to environmental cycles
- Good candidate for circadian rhythm studies
```

**Case 2: Ultradian Rhythm**
```
Periodogram shows:
- Peak at 12.0 hours
- Z(T) = 22.4  (above Bonferroni threshold ≈ 15.2 → significant)
- Secondary peak at 24.0 hours (Z = 16.8, also significant)

Interpretation:
Organism shows twice-daily (ultradian) activity pattern. Could indicate:
- Bimodal activity (e.g., dawn/dusk activity)
- Harmonic of 24h rhythm
- Response to twice-daily feeding schedule

Next steps:
- Check lighting conditions (are there two light phases?)
- Examine activity fraction plot for two daily peaks
- Compare with control group in constant conditions
```

**Case 3: Free-Running Period**
```
Periodogram shows:
- Peak at 25.2 hours (not 24.0h)
- Z(T) = 19.1  (above Bonferroni threshold ≈ 15.2 → significant)
- No light/dark data available (constant darkness)

Interpretation:
Organism's endogenous circadian period is ~25.2 hours (longer than 24h).
This is a "free-running" rhythm in the absence of external time cues.

Biological meaning:
- Demonstrates endogenous clock (not driven by environment)
- Period slightly longer than Earth's rotation (common in many organisms)
- Useful for studying internal clock mechanisms
```

**Case 4: No Significant Rhythm**
```
Periodogram shows:
- Flat profile, no clear peaks
- Maximum Z(T) = 8.5  (below Bonferroni threshold ≈ 15.2 → not significant)
- per-test p-value = e^(-8.5/2) ≈ 0.014  (but does not survive correction)

Interpretation:
No statistically significant periodic pattern detected. Possible reasons:
1. Organism is arrhythmic (lacks circadian clock)
2. Recording too short (need more cycles)
3. Highly variable activity obscures rhythm
4. Developmental stage lacks rhythmicity

Next steps:
- Extend recording duration (try 5-7 days)
- Check for masking effects (light directly suppressing activity)
- Try stronger entrainment conditions (stronger LD cycles)
- Examine individual days for day-to-day variability
```

**Case 5: Multiple Significant Periods**
```
Periodogram shows:
- Peak 1 at 24.0h (Z = 32.5  → significant, above 15.2)
- Peak 2 at 12.0h (Z = 18.7  → significant, above 15.2)
- Peak 3 at 8.0h  (Z = 9.1   → NOT significant, below 15.2)

Interpretation:
Multiple rhythmic components detected:
- 24h: Fundamental circadian rhythm (dominant)
- 12h: Second harmonic (ultradian, also significant)
- 8h: Sub-threshold; not counted as significant after Bonferroni correction

This is common in complex behaviors with multiple regulatory mechanisms.

Analysis approach:
- Focus on dominant period (24h) for circadian studies
- Secondary peaks may reflect meal timing, tidal cycles, or other factors
- Use FFT to see the full spectral picture alongside the Chi² periodogram
```

#### Parameter Selection Guidelines

**Minimum Period (Default: 12 hours)**:
- Set based on expected rhythm range
- For circadian only: 20-22 hours
- For ultradian + circadian: 8-12 hours
- For infradian: Increase to 24-48 hours

**Maximum Period (Default: 36 hours)**:
- Should be < recording_duration / 2
- For 3-day recording: Max ~36h allows 2 full cycles
- For 7-day recording: Can test up to 84h (3.5 days)
- Longer periods need longer recordings for reliable detection

**Significance Level (Default: 0.05)**:
- 0.05: Standard (95% confidence)
- 0.01: Conservative (99% confidence, fewer false positives)
- 0.10: Exploratory (90% confidence, more sensitive)

**Recording Duration Recommendations**:
```
Target Period    Minimum Recording    Recommended
12h ultradian    1 day               2-3 days
24h circadian    2 days              3-5 days
48h infradian    4 days              7-10 days
>72h rhythms     1 week              2-3 weeks
```

#### Common Pitfalls and Solutions

**Problem 1: "No significant rhythm, but I see daily patterns in the plot"**
- **Cause**: High day-to-day variability
- **Solution**: Check if rhythm phase shifts across days. Try longer recordings or more stringent entrainment.

**Problem 2: "Multiple similar peaks, can't determine dominant period"**
- **Cause**: Broad spectral power, noisy rhythm
- **Solution**: Increase bin size for activity fraction, smooth data, or use bandpass filtering.

**Problem 3: "Peak at wrong period (e.g., 23.1h instead of 24.0h)"**
- **Cause**: Limited frequency resolution (only 100 test periods)
- **Solution**: This is normal - report dominant period as detected. Resolution = (max-min)/100.

**Problem 4: "Z-scores very low despite clear activity patterns"**
- **Cause**: Activity patterns are not sinusoidal (e.g., square wave LD response)
- **Solution**: The Chi² periodogram tests for sinusoidal rhythms. Try the FFT power spectrum for non-sinusoidal patterns, or reduce bin size to preserve waveform detail.

#### Technical Notes

**Data Preprocessing for Fisher Analysis**:
```
Input: Activity fraction data (Step 6 output)
- Already binned (default 60-second bins)
- Values range 0.0-1.0 (fraction of time active)
- No additional normalization applied

Note: Analysis uses activity fraction, NOT raw movement values
```

**Sampling Interval Consideration**:
```
Fisher analysis inherits sampling from activity fraction:
- Bin size = 60 seconds → 60 samples per hour
- For 24h period: ~1440 samples per cycle
- Nyquist frequency: Can detect periods down to 120 seconds

In practice:
- Circadian analysis: 60s bins are excellent
- Ultradian (<12h): 60s bins are adequate
- Very fast rhythms (<1h): Consider finer binning
```

**Statistical Power**:
```
Longer recordings = higher n = higher Z-scores (for same rhythm strength)

Example:
Same rhythm amplitude, different recording lengths:
- 1 day (n=1440): Z = 8.5 (marginally significant)
- 3 days (n=4320): Z = 25.5 (highly significant)
- 7 days (n=10080): Z = 59.5 (extremely significant)

Recommendation: Aim for 3-5 days minimum for circadian studies
```

#### Use Cases in Research

1. **Circadian Clock Studies**:
   - Quantify rhythm robustness (Z-score strength)
   - Measure endogenous period (free-running conditions)
   - Assess entrainment quality (peak at 24h vs. other periods)

2. **Drug/Treatment Effects**:
   - Compare Z-scores between control and treated groups
   - Detect period changes (e.g., 24h → 23h after treatment)
   - Identify arrhythmicity (loss of significant peak)

3. **Developmental Studies**:
   - Track rhythm emergence during development
   - Quantify rhythm strength at different life stages
   - Identify critical periods for rhythm establishment

4. **Environmental Entrainment**:
   - Verify light/dark cycle entrainment (peak at LD period)
   - Test non-24h cycles (e.g., 20h or 28h)
   - Study zeitgeber strength (how strongly environment drives rhythm)

5. **Comparative Chronobiology**:
   - Compare circadian periods across species
   - Identify strain/genotype differences in rhythm parameters
   - Quantify inter-individual variability within populations

### Frame Viewer

The Frame Viewer provides interactive exploration of raw video data with temporal context:

- **Time overlay**: Each frame shows elapsed time based on recording interval
- **Metadata integration**: Time calculated from HDF5 metadata or AVI frame rate
- **Memory efficient**: Loads frames on-demand during playback
- **Analysis verification**: Visual confirmation of ROI detection and movement events

---

## 📐 Mathematical Documentation

This section provides comprehensive mathematical foundations for all analysis methods implemented in napari-hdf5-activity, including complete formulas, derivations, and statistical foundations for both movement analysis and rhythmic pattern detection.

### Movement Analysis Pipeline (Mathematics)

#### ROI Detection (Hough Circle Transform)

**Circle Equation:**
```
(x - a)² + (y - b)² = r²
```
Where (a, b) = circle center, r = radius

**Hough Transform:**

For each edge point (x, y), map to parameter space:
```
a = x + r · cos(θ)
b = y + r · sin(θ)  for θ ∈ [0, 2π]
```

**Accumulator Array:**
```
A[a, b, r] = Σ_{(x,y) ∈ Edges} δ[(x-a)² + (y-b)² - r²]
```

Detected circles = local maxima in A[a,b,r] above threshold.

**OpenCV Parameters:**
- **dp**: Accumulator resolution = image_resolution / dp
- **minDist**: Minimum center-to-center distance
- **param1**: Canny edge detector threshold
- **param2**: Accumulator threshold for circle detection
- **minRadius, maxRadius**: Radius constraints

**ROI Mask Generation:**
```
M_i[x, y] = {1  if (x - a_i)² + (y - b_i)² ≤ r_i²
            {0  otherwise

N_i = Σ_{x,y} M_i[x, y] ≈ π · r_i²  (pixel count)
```

#### Movement Quantification (Pixel Differences)

**Frame-to-Frame Difference:**
```
D_i,t[x, y] = |I_t[x, y] - I_{t-1}[x, y]|  for pixels where M_i[x,y] = 1
```

**Total Change in ROI:**
```
C_i,t = Σ_{x,y ∈ ROI_i} |I_t[x,y] - I_{t-1}[x,y]|
```

**Normalized Movement Value:**
```
m_i,t = C_i,t / N_i
```
- **m_i,t** = average pixel intensity change per pixel
- Units: 0-255 (8-bit) or 0-65535 (16-bit)
- Physical meaning: Magnitude of brightness changes due to organism movement

**Normalization:**
```
m̃_i,t = m_i,t  (no minimum subtraction)
```

#### Baseline Threshold Calculation

**Statistical Estimators (from baseline period t ∈ [0, T_baseline]):**
```
μ̂_i = (1/T_baseline) · Σ_{t=0}^{T_baseline-1} m̃_i,t  (sample mean)

σ̂_i = √[(1/(T_baseline-1)) · Σ_{t=0}^{T_baseline-1} (m̃_i,t - μ̂_i)²]  (sample std)
```

**Threshold Calculation:**
```
θ_upper,i = μ̂_i + λ · σ̂_i
θ_lower,i = μ̂_i - λ · σ̂_i
```
Where **λ** = threshold multiplier (default: 0.1)

**Interpretation:**
- **θ_upper**: Movement above this → ACTIVE state
- **θ_lower**: Movement below this → QUIESCENT state
- **Hysteresis gap**: H = 2λσ̂_i (prevents flickering)

**Three Threshold Methods:**

1. **Baseline Method:**
   ```
   Use first N minutes of recording
   → Calculate μ̂_i, σ̂_i from baseline window
   → Apply same thresholds to entire recording
   ```

2. **Calibration Method:**
   ```
   Process separate calibration recording
   → Transfer thresholds to main dataset
   → Standardizes across experiments
   ```

3. **Adaptive Method:**
   ```
   Sliding window: θ_i(t) adapts over time
   μ̂_i(t) = mean(m̃_i,τ) for τ ∈ [t - W/2, t + W/2]
   → Handles non-stationary baselines
   ```

**Important:** Thresholds calculated from **normalized data BEFORE detrending** (preserves true baseline statistics).

#### Preprocessing Methods

**Detrending (Polynomial Regression):**
```
Trend(t) = β_0 + β_1·t + β_2·t² + ... + β_p·t^p

Least squares:  β̂ = (X'X)⁻¹X'y

Detrended:  m'_i,t = m̃_i,t - Trend(t)
```
- Removes slow drift (photobleaching, LED changes)
- Typical degree: p = 1 (linear) or p = 2 (quadratic)

**Jump Correction (Outlier Detection):**
```
Rolling Std:  σ_rolling(t) = std(m̃_i,τ) for τ ∈ [t - W/2, t + W/2]

Frame difference:  Δ_t = m̃_i,t - m̃_i,t-1

Jump detected if:  |Δ_t| > κ · median(σ_rolling)  (κ = 3.0 default)

Correction:  m̃_i,t ← m̃_i,t - jump_magnitude  for all t ≥ t_jump
```
- Removes sudden signal shifts (vibrations, bumps)

#### Movement State Detection (Hysteresis)

**State Machine:**
```
States:  𝒮 = {QUIESCENT, MOVEMENT}

Transitions:
  QUIESCENT → MOVEMENT:  if m'_i,t > θ_upper,i
  MOVEMENT → QUIESCENT:  if m'_i,t < θ_lower,i
```

**Binary Output:**
```
b_i,t = {1  if state = MOVEMENT
        {0  if state = QUIESCENT
```

**Schmitt Trigger Analogy:**
```
       ┌───────┐
  θ_u ─┤       ├─ HIGH (MOVEMENT)
       │  Gap  │
  θ_l ─┤       ├─ LOW (QUIESCENT)
       └───────┘
```

**Algorithm:**
```python
s ← QUIESCENT  # Initial state

for t = 1 to T:
    if s == QUIESCENT:
        if m'[t] > θ_upper:
            s ← MOVEMENT
            b[t] ← 1
        else:
            b[t] ← 0
    elif s == MOVEMENT:
        if m'[t] < θ_lower:
            s ← QUIESCENT
            b[t] ← 0
        else:
            b[t] ← 1
```

**Noise Immunity:**
- Hysteresis gap H = θ_upper - θ_lower = 2λσ̂_i
- Signal fluctuations < H do not cause state changes

#### Activity Fraction and Sleep Detection

**Time Binning:**
```
Bin k = [t_k, t_{k+1})  where  t_k = k · B

B = bin_size_seconds / frame_interval

Activity fraction:  α_i,k = (1/|Bin_k|) · Σ_{t ∈ Bin_k} b_i,t
```
- **α_i,k** ∈ [0, 1] = fraction of time active in bin k
- Typical bin size: 60 seconds

**Statistical Properties:**
```
Mean activity:  ᾱ_i = (1/K) · Σ_{k=0}^{K-1} α_i,k

Standard error:  SE(α_i,k) = √[α_i,k(1 - α_i,k) / |Bin_k|]

95% CI:  α̂ ± 1.96 · SE(α̂)
```

**Quiescence Detection:**
```
q_i,k = {1  (QUIESCENT)  if α_i,k < ψ
        {0  (ACTIVE)      if α_i,k ≥ ψ

where ψ = quiescence threshold (default: 0.5)
```

**Sleep Bout Identification:**
```
Sleep Bout = consecutive quiescent bins with duration ≥ D_min

Duration = (number of consecutive q_i,k = 1) · bin_size

Default:  D_min = 8 minutes
```

**Sleep Statistics:**
```
Total Sleep Time (TST):  Σ_{bout ∈ 𝒮_i} duration(bout)

Number of Bouts:  N_bouts,i = |𝒮_i|

Average Bout Duration:  <duration>_i = TST_i / N_bouts,i

Sleep Efficiency:  SE_i = TST_i / Total_Recording_Duration
```

---

### Rhythmic Pattern Analysis (Mathematics)

#### Chi² Periodogram (Mathematical Details)

**Core Principle:**

Tests for sinusoidal rhythms by correlating data with cosine/sine waves at m=100 candidate periods. Named after the χ²(df=2) null distribution of the test statistic.

**For test period T:**
```
Angular frequency:  ω = 2π / T

Reference waves:
  C(t) = cos(ω · t)
  S(t) = sin(ω · t)

Correlations:
  r_cos = corr(y, C)
  r_sin = corr(y, S)

Squared coherence:  R²(T) = r_cos² + r_sin²

Chi-squared statistic:  Z(T) = n · R²(T)
  (labeled "Z-score" in plot output)
```

**Statistical Significance (Bonferroni-corrected):**
```
Under H₀ (no rhythm):  Z(T) ~ χ²(df=2)

With m = 100 tested periods, Bonferroni-corrected thresholds:
  α = 0.10:  critical_Z ≈ 13.4   [χ²(1 - 0.10/100, df=2)]
  α = 0.05:  critical_Z ≈ 15.2   [χ²(1 - 0.05/100, df=2)]
  α = 0.01:  critical_Z ≈ 19.5   [χ²(1 - 0.01/100, df=2)]

Per-test p-value:  p = e^(-Z/2)   (closed form for χ²(df=2))
  NOTE: significance decision uses Bonferroni threshold on Z(T),
        not the raw per-test p-value

Significant if:  Z(T) > critical_Z  (Bonferroni-corrected)
```

**Dominant Period:**
```
T_dom = argmax_T Z(T)

Test range:  T ∈ [T_min, T_max]  (100 evenly-spaced periods)
```

**Nyquist Constraint:**
```
Maximum testable period:  T_max ≤ recording_duration / 2

Example: 24-hour recording → can test up to 12-hour periods
```

**Interpretation (Bonferroni-corrected, α=0.05):**
- **Z(T) > 20**: Very strong, highly significant rhythm
- **Z(T) 15-20**: Significant (above Bonferroni threshold ≈ 15.2)
- **Z(T) 6-15**: Below threshold; not significant after correction
- **Z(T) < 6**: No significant rhythm

#### FFT Power Spectrum (Mathematical Details)

**Discrete Fourier Transform:**
```
Y[k] = Σ_{n=0}^{N-1} y[n] · e^(-2πikn/N)

Power spectrum:  P[k] = |Y[k]|²

Frequency bins:  f[k] = k / (N · Δt)

Period conversion:  τ[k] = 1 / f[k]
```

**Window Functions (reduce spectral leakage):**

Hann (default):
```
w[n] = 0.5 - 0.5 · cos(2πn / (N-1))
```

Hamming:
```
w[n] = 0.54 - 0.46 · cos(2πn / (N-1))
```

Blackman:
```
w[n] = 0.42 - 0.5 · cos(2πn / (N-1)) + 0.08 · cos(4πn / (N-1))
```

**Windowed Data:**
```
y_windowed[n] = y[n] · w[n]
```

**Permutation Test for Significance:**

Unlike the Chi² periodogram (which has an analytical χ²(df=2) distribution), FFT requires empirical significance testing. The permutation test uses the **maximum** power over the full period range to correctly handle the multiple-frequencies problem.

**Algorithm:**
```
1. Compute observed MAXIMUM power over the period range:  P_obs = max(P[T_min..T_max])

2. Generate null distribution (n_perm = 1000 permutations):
     For i = 1 to n_perm:
       a. Randomly shuffle data:  y_perm = random_permutation(y)
       b. Apply same preprocessing (detrending, windowing)
       c. Compute FFT:  P_perm[i] = max power over same period range

3. Calculate p-value:
     p = (count of P_perm[i] ≥ P_obs) / n_perm

4. Significant if:  p < α
```

**Rationale:**

Permutation test:
- **Non-parametric** (no distribution assumptions)
- **Exact** (given enough permutations)
- **Controls for data properties** (preserves amplitude distribution, variance)

**Interpretation:**
- p < 0.05: Dominant period is significantly stronger than expected by chance
- p ≥ 0.05: No significant periodic pattern

#### Cosinor Analysis (Mathematical Details)

**Cosine Model:**
```
y(t) = M + A · cos(ω·t + φ) + ε(t)
```
Where:
- **M** = MESOR (mean level)
- **A** = Amplitude (half peak-to-trough distance)
- **φ** = Acrophase (phase at peak)
- **ω** = 2π/τ (angular frequency, τ = assumed period)

**Linear Regression Form:**

Using trigonometric identity:
```
y(t) = M + β·cos(ω·t) + γ·sin(ω·t) + ε(t)

where:
  β = A · cos(φ)
  γ = -A · sin(φ)
```

**Least Squares Estimation:**
```
Design matrix X:
  [1   cos(ω·t₁)  sin(ω·t₁)]
  [1   cos(ω·t₂)  sin(ω·t₂)]
  [⋮   ⋮          ⋮        ]
  [1   cos(ω·t_n)  sin(ω·t_n)]

Parameter estimates:  θ̂ = (X'X)⁻¹X'y = [M̂, β̂, γ̂]'
```

**Rhythm Parameters:**
```
MESOR:  M̂ = M̂  (directly from regression)

Amplitude:  Â = √(β̂² + γ̂²)

Acrophase:  φ̂ = arctan2(-γ̂, β̂)

Peak time:  t_peak = -φ̂ / ω  (mod τ)
```

**F-Test for Significance:**
```
Null hypothesis:  H₀: β = 0 and γ = 0  (no rhythm)

F-statistic:  F = (RegSS / 2) / (RSS / (n - 3))

Under H₀:  F ~ F(df₁=2, df₂=n-3)

p-value:  p = P(F_{2, n-3} ≥ F_obs)

Significant if:  p < α
```

**Goodness of Fit:**
```
R² = RegSS / TSS = 1 - (RSS / TSS)

Interpretation:
  R² = proportion of variance explained by cosine model
  R² > 0.30: Strong rhythm
  0.10 < R² < 0.30: Moderate rhythm
  R² < 0.10: Weak or no rhythm
```

**Confidence Intervals:**
```
Standard errors from covariance matrix:
  Cov(θ̂) = σ̂² · (X'X)⁻¹

95% CI:
  M̂ ± t_{n-3, 0.025} · SE(M̂)
  Â ± t_{n-3, 0.025} · SE(Â)  (delta method)
  φ̂ ± t_{n-3, 0.025} · SE(φ̂)  (delta method)
```

#### Comparison of Rhythmic Analysis Methods

| Method | Best For | Significance Test | Period Detection |
|--------|----------|-------------------|------------------|
| **Chi² Periodogram** | Statistical period detection | Bonferroni-corrected χ²(df=2); threshold ≈ 15.2 | Scans 100 test periods |
| **FFT** | Exploratory analysis, harmonics | Permutation (1000x, max-power test) | Full frequency spectrum |
| **Cosinor** | Quantifying rhythm parameters | F(2, n−3) individual; Nelson F-test for population | Assumes known period |

**When to Use Each:**

1. **Chi² Periodogram:**
   - ✓ Hypothesis testing with analytical χ²(df=2) distribution
   - ✓ Bonferroni correction for 100 periods built in (threshold ≈ 15.2)
   - ✓ Standard circadian analysis (Sokolove & Bushell 1978)
   - ✓ Fast computation
   - ✗ Limited to 100 test periods (use FFT for finer resolution)

2. **FFT Power Spectrum:**
   - ✓ Discovering unexpected periods
   - ✓ Harmonic analysis
   - ✓ Continuous frequency spectrum
   - ✓ Permutation test correctly handles multiple-frequency problem
   - ✗ Power in arbitrary units (not comparable across recordings)

3. **Cosinor Analysis:**
   - ✓ MESOR, Amplitude, Acrophase estimation
   - ✓ Confidence intervals
   - ✓ Nelson F-test for population-level rhythm
   - ✗ Requires known period
   - ✗ Cannot detect period

**Integrated Workflow:**
```
Step 1: Run Chi² Periodogram and FFT
  → Detect if rhythm exists
  → Estimate dominant period T_dom

Step 2: Verify concordance
  → If Chi² and FFT agree → proceed
  → If disagree → investigate data quality or extend recording

Step 3: Run Cosinor with period = T_dom
  → Quantify MESOR, Amplitude, Acrophase
  → Report with confidence intervals and Nelson population F-test

Step 4: Biological interpretation
  → Relate parameters to experimental conditions
```

**Expected Concordance:**
```
Strong 24h rhythm:
  Chi² Periodogram: Peak at 24h, Z(T) = 32.5 > 15.2 → Bonferroni-significant ✓
  FFT: Peak at 24h, permutation p = 0.001 ✓
  Cosinor (24h): A = 0.28, F(2, n−3) = 45.3, p < 0.001 ✓

  → All methods agree: strong circadian rhythm
```

**Key Differences:**
- **Chi² Periodogram**: Bonferroni-corrected threshold (≈ 15.2); analytical χ²(df=2) distribution
- **FFT**: Tests all frequencies, max-power permutation p-value; power in arbitrary units
- **Cosinor**: Assumes period, parametric F-test; Nelson F-test for population

---

## Citation

If you use this plugin in your research, please cite:

```
@software{napari_hdf5_activity,
  author = {s1alknau},
  title = {napari-hdf5-activity: Activity analysis plugin for napari},
  year = {2025},
  url = {https://github.com/s1alknau/napari-hdf5-activity}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/s1alknau/napari-hdf5-activity.git
cd napari-hdf5-activity
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## License

Distributed under the terms of the [MIT](http://opensource.org/licenses/MIT) license, "napari-hdf5-activity" is free and open source software.

## Issues

If you encounter any problems, please [file an issue](https://github.com/s1alknau/napari-hdf5-activity/issues) with:
- Operating system and version
- Python version
- napari version
- Error message and full traceback
- Minimal example to reproduce the issue

## Acknowledgments

This plugin was developed for analyzing activity and sleep behavior in marine organisms (Nematostella vectensis and other cnidarians) but can be adapted for any timelapse movement analysis.

---

**Author**: s1alknau
**Repository**: https://github.com/s1alknau/napari-hdf5-activity
**AVI Documentation**: See [AVI_INTEGRATION_README.md](AVI_INTEGRATION_README.md) for AVI file support details
