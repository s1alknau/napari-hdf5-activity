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
- **Frame Viewer**: Interactive dataset playback
  - Frame-by-frame navigation with slider
  - Playback controls with adjustable FPS
  - Time overlay (calculated from frame interval)
  - Support for both HDF5 and AVI datasets

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
| Chunk Size | 50 | Frames per processing chunk |
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

### Movement Calculation

1. **Pixel Difference**: `diff = abs(frame[t] - frame[t-1])`
2. **ROI Masking**: Apply circular mask to each ROI
3. **Normalization**: `movement = sum(diff * mask) / sum(mask)`
4. **Baseline Calculation**: `mean` and `std` from first N minutes
5. **Symmetric Hysteresis Thresholds**:
   - Upper: `mean + (multiplier × std)`
   - Lower: `mean - (multiplier × std)`
   - State changes require crossing both thresholds for noise resistance

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

### Fischer Z-Transformation

The plugin implements **Fischer Z-transformation** for detecting periodic patterns in activity data, particularly useful for identifying circadian rhythms in biological timeseries.

**What is a Periodogram?**

A periodogram is a statistical tool that identifies periodic (repeating) patterns in timeseries data. It answers: "Does the organism show rhythmic activity, and if so, at what period (e.g., 24 hours)?"

**How it works:**

1. **Input**: Movement activity data over time (e.g., 3 days of recording)
2. **Analysis**: Tests multiple period lengths (e.g., 12h, 18h, 24h, 30h, 36h)
3. **Output**: Z-scores indicating strength of each period
4. **Significance**: Chi-square test determines if patterns are statistically significant

**Interpretation:**

- **High Z-score peak at 24h**: Strong circadian (24-hour) rhythm
- **Peak at 12h**: Ultradian (twice daily) activity pattern
- **No significant peaks**: Irregular/random activity
- **Green ROI title**: Statistically significant rhythm detected
- **Red marker**: Dominant period (strongest rhythm)

**Use cases:**
- Detect light/dark cycle entrainment
- Identify free-running circadian periods
- Compare rhythmicity between experimental groups
- Detect ultradian or infradian rhythms

### Frame Viewer

The Frame Viewer provides interactive exploration of raw video data with temporal context:

- **Time overlay**: Each frame shows elapsed time based on recording interval
- **Metadata integration**: Time calculated from HDF5 metadata or AVI frame rate
- **Memory efficient**: Loads frames on-demand during playback
- **Analysis verification**: Visual confirmation of ROI detection and movement events

## Changelog

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
