# napari-hdf5-activity

[![License MIT](https://img.shields.io/pypi/l/napari-hdf5-activity.svg?color=green)](https://github.com/s1alknau/napari-hdf5-activity/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-hdf5-activity.svg?color=green)](https://pypi.org/project/napari-hdf5-activity)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-hdf5-activity.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-hdf5-activity)](https://napari-hub.org/plugins/napari-hdf5-activity)

A napari plugin for analyzing activity and movement behavior from HDF5 timelapse recordings and AVI video files.

----------------------------------

## Features

### File Format Support
- **HDF5 files**: Dual structure support (stacked frames and individual frames)
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
- **Extended Analysis**: Fischer Z-transformation for circadian rhythm detection
  - Periodogram analysis for detecting periodic patterns
  - Statistical significance testing (Chi-square)
  - Sleep/wake phase identification based on dominant periods
  - Configurable period range (0-100 hours)
  - Visual periodogram plots for all ROIs
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
- **Verification**: Test suite confirms baseline difference < 0.000001 between detrended and non-detrended modes

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

**Input**: Raw video frames (grayscale, 8-bit or 16-bit)

**Process Description**:
This step quantifies how much each organism moves by comparing consecutive video frames. The algorithm calculates the absolute difference in pixel brightness between each frame and the previous frame, focusing only on pixels within each detected ROI (Region of Interest - the circular area around each organism).

**Detailed Algorithm**:
```
For each ROI at time t:
  1. Calculate pixel-wise difference: diff_pixels = abs(frame[t] - frame[t-1])
     → Compare current frame with previous frame, pixel by pixel
     → Take absolute value to get magnitude of change (positive number)
     → Example: If pixel was 100, now 115 → difference = 15

  2. Apply circular ROI mask: masked_diff = diff_pixels * circular_mask
     → Isolate only pixels inside the circular ROI boundary
     → Ignore pixels outside the organism's area
     → Circular mask = 1 inside ROI, 0 outside ROI

  3. Sum absolute differences: total_change = sum(masked_diff)
     → Add up all pixel changes within the ROI
     → Gives total amount of change in the organism's area
     → Example: If ROI has 1000 pixels and each changed by ~15 → total ≈ 15000

  4. Normalize by ROI area: movement_value = total_change / sum(circular_mask)
     → Divide by number of pixels in ROI to get average change per pixel
     → This makes values comparable between different sized organisms
     → Example: 15000 / 1000 pixels = 15.0 average change per pixel
```

**What the Numbers Mean**:
- **Movement Value**: Average pixel intensity change per pixel within ROI
- **Range**: 0 to 255 (for 8-bit images) or 0 to 65535 (for 16-bit images)
- **Interpretation**:
  - **0**: No movement detected (organism completely still)
  - **1-5**: Very subtle movement (small positional shifts)
  - **5-20**: Moderate movement (typical slow crawling or body contractions)
  - **20-50**: Strong movement (rapid locomotion or large body changes)
  - **>50**: Very strong movement (fast swimming or major morphology changes)
- **Example**: Movement value of 15.3 means pixels changed by an average of 15.3 intensity units between frames

**Physical Meaning**:
When an organism moves, its body position changes relative to the background. Dark pixels become light (or vice versa) as the organism moves across the frame. The movement value captures the magnitude of these brightness changes as a proxy for physical movement.

**Y-Axis in Movement Plots**: "Movement (pixel intensity change)" - raw pixel difference values

#### Step 2: MATLAB-Compatible Normalization

**Input**: Raw movement values from Step 1

**Processing**:
```
For each ROI:
  normalized_data = raw_movement_values (no minimum subtraction)
```

**Note**: True MATLAB compatibility means NO minimum subtraction. The data represents direct frame-to-frame pixel changes, matching MATLAB's `framePixelChange` calculation.

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

4. **Periodogram (Fisher Analysis)**:
   - X-axis: "Period (hours)"
   - Y-axis: "Fischer Z-Score" (dimensionless)
   - Range: 0 to ~30+ (higher = stronger rhythm)
   - Horizontal line: Significance threshold (chi-square, df=2)

### Multiprocessing Performance

The plugin supports true multiprocessing using Python's `multiprocessing` module for CPU-bound analysis tasks:

**Parallel Processing:**
- **ROI-level parallelization**: Each ROI is processed in a separate CPU core
- **Automatic selection**: Parallel processing enabled when `num_processes > 1` and `num_rois >= 2`
- **Optimal core usage**: Automatically uses `cpu_count() - 1` cores (leaves one for system)
- **Python 3.9+ compatible**: Uses `multiprocessing.Pool` for cross-platform compatibility

**When to use parallel processing:**
- Multiple ROIs (≥2) to process
- Large datasets with long recordings
- Multi-core CPU available
- Baseline analysis method (currently supported)

**Performance guidelines:**
- 2-4 ROIs: Use `num_processes=2-4` for ~2x speedup
- 5-10 ROIs: Use `num_processes=4-8` for ~3-5x speedup
- Single ROI: Parallel processing automatically disabled (no benefit)

**Note:** Calibration and Adaptive methods currently use sequential processing.

### LED-Based Lighting Detection

- **Light Phase**: White LED power > 0.5%
- **Dark Phase**: White LED power ≤ 0.5% (IR only)
- **IR LED**: Continuous 100% for video recording
- **Source**: HDF5 timeseries only (not available for AVI files)

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

## Scientific Background

### Fischer Z-Transformation for Circadian Rhythm Detection

The plugin implements **Fischer's Z-transformation periodogram** for detecting periodic patterns in activity data, particularly useful for identifying circadian rhythms in biological timeseries. This method tests for correlations between the time series and sine/cosine waves at different periods.

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

2. **For Each Test Period P**:
   ```
   a. Calculate angular frequency: ω = 2π / P
   b. Generate cosine wave: cos_wave = cos(ω × t)
   c. Generate sine wave: sin_wave = sin(ω × t)
   d. Calculate correlations:
      - r_cos = correlation(activity_data, cos_wave)
      - r_sin = correlation(activity_data, sin_wave)
   e. Calculate coherence squared: C² = r_cos² + r_sin²
   f. Calculate Fischer Z-score: Z = n × C²
      (where n = number of data points)
   ```

3. **Statistical Significance**:
   ```
   Z-scores follow chi-square distribution (df=2)

   Significance threshold (α = 0.05):
   Critical Z = 5.99 (chi-square critical value, 2 df, p<0.05)

   Interpretation:
   - Z > 5.99: Statistically significant rhythm (p < 0.05)
   - Z < 5.99: No significant rhythm detected
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

**Y-Axis: Fischer Z-Score (dimensionless)**
- Range: Typically 0 to 30+
- **Z > 5.99**: Statistically significant (p < 0.05)
- **Z > 9.21**: Highly significant (p < 0.01)
- **Z > 13.82**: Very highly significant (p < 0.001)
- Higher Z-scores indicate stronger, more consistent rhythms

**Visual Elements**:
1. **Horizontal line**: Critical threshold (Z = 5.99 for α=0.05)
2. **Red marker**: Dominant period (peak Z-score)
3. **Green title**: ROI has significant rhythm (p < 0.05)
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
- Z-score = 10.2
- Secondary peak at 24.0 hours (Z = 7.1)

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
- Z-score = 12.4
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
- Maximum Z-score = 4.2 (below 5.99 threshold)
- p-value = 0.12

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
- Peak 1 at 24.0h (Z = 15.3)
- Peak 2 at 12.0h (Z = 8.7)
- Peak 3 at 8.0h (Z = 6.2)

Interpretation:
Multiple rhythmic components detected:
- 24h: Fundamental circadian rhythm
- 12h: Second harmonic (ultradian)
- 8h: Third harmonic or independent rhythm

This is common in complex behaviors with multiple regulatory mechanisms.

Analysis approach:
- Focus on dominant period (24h) for circadian studies
- Secondary peaks may reflect meal timing, tidal cycles, or other factors
- Use filtering to isolate specific frequency components if needed
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
- **Solution**: Fischer Z tests for sinusoidal rhythms. Try autocorrelation analysis for non-sinusoidal patterns.

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
- Added Extended Analysis tab with Fischer Z-transformation
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

## Acknowledgments

This plugin was developed for analyzing activity and sleep behavior in marine organisms (Nematostella vectensis and other cnidarians) but can be adapted for any timelapse movement analysis.

---

**Author**: s1alknau
**Repository**: https://github.com/s1alknau/napari-hdf5-activity
**AVI Documentation**: See [AVI_INTEGRATION_README.md](AVI_INTEGRATION_README.md) for AVI file support details
