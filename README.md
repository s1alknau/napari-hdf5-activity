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

## Usage Examples

### Example 1: Single HDF5 Recording

```python
import napari

# Launch napari
viewer = napari.Viewer()

# Open HDF5 file through plugin
# Plugins → napari-hdf5-activity
# Load File → Select timelapse.h5
# Detect ROIs → Process Data → Generate Plots
```

### Example 2: Batch AVI Processing

**Prepare metadata file** (`metadata.json`):
```json
{
  "video_settings": {
    "fps": 5.0,
    "frame_interval": 5.0
  },
  "lighting_conditions": {
    "white_led": {
      "status": "scheduled",
      "light_periods": [
        {"start_hour": 7, "end_hour": 19, "description": "Day 1"}
      ]
    }
  }
}
```

**Process videos:**
```bash
python process_avi_batch.py --dir /path/to/videos --interval 5.0
```

Or through napari UI:
1. Load Directory → Select folder with AVIs
2. Plugin loads all videos as temporal batch
3. Detect ROIs on first frame
4. Process Data → analyzes all frames
5. Lighting conditions automatically shown from metadata

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

### Metadata Format

Place `metadata.json` next to your AVI files:

```json
{
  "video_settings": {
    "fps": 5.0,
    "frame_interval": 5.0,
    "camera": "IR camera",
    "illumination": "continuous IR, white LED for light phases"
  },
  "lighting_conditions": {
    "ir_led": {
      "status": "always_on",
      "power_percent": 100
    },
    "white_led": {
      "status": "scheduled",
      "schedule_type": "custom",
      "light_periods": [
        {"start_hour": 7, "end_hour": 19, "description": "Day 1: 07:00-19:00"},
        {"start_hour": 31, "end_hour": 43, "description": "Day 2: 07:00-19:00"}
      ]
    }
  }
}
```

**Generate metadata:**
```bash
python create_avi_metadata.py --output metadata.json --fps 5.0 --interval 5.0 --days 3
```

### Memory Efficiency

- **Loading**: Only first frame loaded (~2 MB instead of 500 MB)
- **ROI Detection**: Performed on first frame
- **Analysis**: All frames loaded on-demand during processing
- **Benefit**: Fast UI, minimal memory footprint for preview

## Parameter Guide

### ROI Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Radius | 380 | Minimum organism size (pixels) |
| Max Radius | 420 | Maximum organism size (pixels) |
| DP Parameter | 0.5 | Hough transform sensitivity (lower = more sensitive) |
| Min Distance | 150 | Minimum separation between ROIs (pixels) |
| Param1 (Edge) | 40 | Canny edge detection threshold |
| Param2 (Center) | 40 | Circle center detection threshold |

### Movement Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| Frame Interval | 5.0 s | Time between analyzed frames |
| Baseline Duration | 5.0 min | Duration for baseline calculation |
| Threshold Multiplier | 3.0 | Sensitivity factor for movement detection |
| Upper Threshold Factor | 1.0 | Hysteresis upper threshold |
| Lower Threshold Factor | 0.5 | Hysteresis lower threshold |
| Chunk Size | 50 | Frames per processing chunk |
| Num Processes | 4 | Parallel processing workers |

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
- Check metadata.json is in same folder as AVI files
- Verify AVI codec is supported (MJPEG, H264, etc.)

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
4. **Thresholding**: Compare to baseline + (multiplier × std)
5. **Hysteresis**: Separate upper/lower thresholds for state changes

### LED-Based Lighting Detection

- **Light Phase**: White LED power > 0.5%
- **Dark Phase**: White LED power ≤ 0.5% (IR only)
- **IR LED**: Continuous 100% for video recording
- **Source**: HDF5 timeseries or AVI metadata

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

## Changelog

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
**Documentation**: See [AVI_USAGE_GUIDE.md](AVI_USAGE_GUIDE.md) for detailed AVI workflow
