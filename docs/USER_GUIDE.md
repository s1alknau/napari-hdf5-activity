# napari-hdf5-activity User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Input Tab](#input-tab)
3. [Analysis Tab](#analysis-tab)
4. [Results Tab](#results-tab)
5. [Extended Analysis Tab](#extended-analysis-tab)
6. [Frame Viewer Tab](#frame-viewer-tab)
7. [Workflow Examples](#workflow-examples)
8. [Tips and Tricks](#tips-and-tricks)
9. [Processing Zarr Recordings Without the Plugin](#processing-zarr-recordings-without-the-plugin)

## Getting Started

### Prerequisites

- **napari** installed (version >= 0.4.17)
- **Python** 3.8 or higher
- **napari-hdf5-activity** plugin installed

### Launching the Plugin

1. Open napari:
   ```bash
   napari
   ```

2. Activate the plugin:
   - Menu: `Plugins` → `napari-hdf5-activity: HDF5 Activity Analysis`
   - The plugin widget appears on the right side of the napari window

### Plugin Layout

The plugin has 5 tabs:
- **Input**: File loading and ROI detection
- **Analysis**: Movement analysis configuration
- **Results**: Plot generation and data export
- **Extended Analysis**: Circadian rhythm detection
- **Frame Viewer**: Interactive dataset playback

---

## Input Tab

### Loading Data

#### HDF5 Files

**Single File:**
1. Click **"Load File"**
2. Select your `.h5` or `.hdf5` file
3. Plugin automatically detects structure type (stacked frames vs. individual frames)
4. First frame is displayed in the napari viewer

**Directory of Files:**
1. Click **"Load Directory"**
2. Select folder containing HDF5 files
3. All HDF5 files are loaded
4. Select a file from dropdown to view

**Supported HDF5 Structures:**
```
# Stacked frames
/frames [dataset: (N, H, W)]

# Individual frames
/frames/
  ├── frame_0000
  ├── frame_0001
  └── ...

# With metadata
/metadata (optional)
```

#### Zarr Files

**Directory Stores (.zarr directories):**
1. Click **"Load File"**
2. Select the `.zarr` directory
3. Plugin reads the Zarr array and treats it identically to an HDF5 stacked-frames dataset

Large stores are processed in parallel, one storage chunk at a time (the
recording plugin writes 50-frame chunks), so each chunk is decompressed only
once. A 46 GB recording (51 840 frames, 1024 × 1224 px) takes about 1.5 min with
four worker processes. The number of workers adapts to the memory Windows can
still commit, so the analysis slows down rather than stalls when RAM is tight —
see [Performance](../PERFORMANCE_OPTIMIZATIONS.md).

To read a Zarr recording in your own scripts, without the plugin, see
[Processing Zarr Recordings Without the Plugin](#processing-zarr-recordings-without-the-plugin).

#### AVI Files

**Single Video:**
1. Click **"Load File"**
2. Select `.avi` file
3. First frame is extracted and displayed

**Multiple Videos (Batch):**
1. Click **"Load File"**
2. Hold **Ctrl** (Windows/Linux) or **Cmd** (Mac)
3. Select multiple AVI files
4. Videos are concatenated temporally:
   - Video 1: 0 - 600 seconds
   - Video 2: 600 - 1200 seconds
   - Etc.

**Memory Efficiency:**
- Only first frame is loaded initially (~2 MB instead of 500 MB)
- Full dataset loaded during analysis

#### Format comparison

| Feature | HDF5 | Zarr | AVI |
|---------|------|------|-----|
| Frame Loading | Direct dataset access | Direct array access | opencv video reader |
| LED Data | ✓ Available in timeseries | ✓ Available in timeseries | ✗ Not available |
| Lighting Plot | ✓ Automatic from LED data | ✓ Automatic from LED data | ✗ Not shown |
| Metadata | ✓ Embedded in file | ✗ Basic (shape, dtype) | ✗ Basic (FPS, duration) |
| Batch Processing | Directory scan | Single directory store | Directory scan or Ctrl multi-select |
| Movement Analysis | ✓ Identical | ✓ Identical | ✓ Identical |
| Threshold Methods | ✓ All methods | ✓ All methods | ✓ All methods |
| Export | ✓ Excel/CSV/Plots | ✓ Excel/CSV/Plots | ✓ Excel/CSV/Plots |

**Note:** Lighting conditions plots require LED power data in the `timeseries`
group, which the recording plugin writes to **both HDF5 and Zarr** stores
(`white_led_power` / `ir_led_power`). AVI files carry no LED data, so lighting
plots are unavailable for them.

### ROI Detection

**What are ROIs?**
Regions of Interest (ROIs) are circular areas that represent individual organisms in your recording. The plugin detects these automatically using Hough Circle Transform.

**Detection Parameters:**

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| **Min Radius** | 300-400 px | Minimum organism size |
| **Max Radius** | 400-500 px | Maximum organism size |
| **DP Parameter** | 0.3-0.8 | Detection sensitivity (lower = more sensitive) |
| **Min Distance** | 100-200 px | Minimum separation between organisms |
| **Param1** | 30-50 | Edge detection threshold |
| **Param2** | 30-50 | Circle detection threshold |

**Steps:**
1. Adjust parameters based on your organism size
2. Click **"Detect ROIs"**
3. ROIs appear as colored circles on the image
4. Each ROI is automatically numbered and color-coded

**Manual Adjustment:**
- If detection fails, try:
  - Decrease DP parameter (more sensitive)
  - Adjust radius range
  - Check image contrast
- You can re-run detection with new parameters (previous ROIs are cleared)

**ROI Editor (fine-tuning positions):**
1. Click **"Edit ROI Circles"** — a napari Points layer appears showing ring symbols at each ROI center
2. Drag points to correct positions (move only — resizing is not supported in the editor)
3. Click **"Apply Edits"** — masks are rebuilt from the updated positions
4. The editor layer is removed automatically after applying

**ROI Visualization:**
- Each ROI has a unique color (repeated in plots)
- ROI labels show the ID number
- ROI layers can be toggled on/off in napari

---

## Analysis Tab

### Basic Parameters

**Frame Interval (seconds):**
- Time between analyzed frames
- Default: 5.0 seconds
- HDF5: Usually specified in metadata
- AVI: Calculated from video FPS and sampling interval

**End Time (seconds):**
- Total duration to analyze
- Default: Full recording
- Useful for testing on subsets

**Chunk Size:**
- Number of frames processed per batch
- Default: 20 frames
- Increase for faster processing (uses more memory)
- Decrease if memory errors occur

**Number of Processes:**
- Parallel processing workers
- Default: 4
- Set to number of CPU cores for maximum speed
- Reduce if system becomes unresponsive

### Threshold Calculation Methods

The plugin offers three methods for determining movement thresholds:

#### 1. Baseline Method

**When to use:**
- Standard recordings with stable conditions
- First-time analysis
- Single recordings

**Parameters:**
- **Baseline Duration**: Length of the baseline window used to compute the mean
  and standard deviation of the activity signal. On loading a dataset the plugin
  sets this to the **full recording** by default (matching the methods used in
  the paper); shorten it only to take the baseline from a specific window.
- **Threshold Multiplier**: Sensitivity factor (default: 0.10)
  - Higher = less sensitive (fewer movements detected)
  - Lower = more sensitive (more movements detected)

**How it works:**
```
Threshold = baseline_mean + (multiplier × baseline_std)
```

**Configuration:**
1. Navigate to **"Baseline Method"** tab
2. Set baseline duration (minutes)
3. Set threshold multiplier
4. Enable/disable detrending
5. Enable/disable jump correction

#### 2. Calibration Method

**When to use:**
- Multiple recordings with same experimental setup
- Comparing across days or conditions
- Standardized threshold across samples

**Workflow:**
1. **Load Calibration File:**
   - Click "Select Calibration File"
   - Choose calibration recording (HDF5 or AVI)
   - Click "Load Calibration Dataset"

2. **Detect Calibration ROIs:**
   - Switch to Input tab
   - Detect ROIs on calibration data
   - Verify detection quality

3. **Process Calibration Baseline:**
   - Return to Analysis → Calibration tab
   - Click "Process Calibration Baseline"
   - Baseline statistics are calculated and stored

4. **Load Main Dataset:**
   - Click "Select Main Dataset File"
   - Choose experimental recording
   - Click "Load Main Dataset"

5. **Detect Main ROIs:**
   - Switch to Input tab
   - Detect ROIs on main dataset

6. **Analyze:**
   - Return to Analysis tab
   - Click "Start Analysis"
   - Calibration thresholds are applied

**Parameters:**
- **Calibration Multiplier**: Threshold factor for calibration baseline
- Same as threshold multiplier in baseline method

**Advantages:**
- Consistent thresholds across recordings
- Reduces batch effects
- Standardized analysis pipeline

#### 3. Adaptive Method

**When to use:**
- Noisy recordings or variable signal quality
- When the threshold should adapt to the data's noise level automatically

**Parameters:**
- **Analysis Duration (min)**: the sample period whose signal quality is analysed
  (default: 15 min, range 5–120)
- **Base Multiplier**: base threshold multiplier before quality scaling
  (default: 2.5, range 1.0–5.0)

**How it works:**
- Analyses the sample period and computes signal-quality metrics: signal-to-noise
  ratio (SNR = mean / std), coefficient of variation (CV = std / mean), and the
  percentage of outliers (IQR fencing).
- For high-noise signals (CV > 0.5 or > 10 % outliers) the **median** replaces the
  mean as the baseline estimator (robust to outliers).
- The base multiplier is scaled by SNR and CV — relaxed for noisy signals (fewer
  false positives), tightened for clean signals (higher sensitivity).
- Movement is then detected with the **same hysteresis logic** as the baseline method.

**Configuration:**
1. Navigate to **"Adaptive Method"** tab
2. Set the analysis duration
3. Set the base multiplier

### Additional Options

**Enable Detrending:**
- Removes linear trends in the data
- Useful for recordings with gradual brightness changes
- Recommended: ON for most analyses

**Enable Jump Correction:**
- Detects and corrects sudden shifts in baseline
- Useful for recordings with technical artifacts
- Recommended: ON if you observe sudden jumps

**Steps at light/dark transitions.** Even with no animal movement, |ΔI| is not
zero: camera and LED noise always contribute, and that noise floor grows with
image brightness. When the illumination switches between phases (white LED on/off,
IR power changing), the floor jumps — visible as steps in the activity trace at
ZT 12, 24, 36, … The step height differs per ROI because the illumination change
is not uniform across the plate. Two options remove it; both read the light/dark
phases from the LED data in the file (`white_led_power`, or `phase_str`) and
only one can be active at a time:

**Adaptive Illumination Baseline:**
- Shifts each light/dark period so its resting floor (15th percentile) matches
  the median floor of all periods — an **additive** correction
- The shift ramps smoothly over ±2 min around each transition
- Recommended: ON when the activity trace steps at light/dark transitions

**Per-Phase Normalization:**
- Divides each light/dark period by its own resting floor — a **multiplicative**
  correction that equalises the floor *and* scales the noise width with it
- Intended for setups where the noise width changes noticeably with the
  illumination, not only the level
- On the recordings tested so far both options levelled the steps equally well
  (residual step 1–5 %) and Per-Phase Normalization brought no extra benefit for
  the noise width — try both and keep the one that gives the flatter baseline
  for your rig

### Running the Analysis

1. Ensure data and ROIs are loaded
2. Select threshold method (Baseline/Calibration/Adaptive)
3. Configure parameters
4. Click **"Start Analysis"**
5. Progress bar shows processing status
6. Analysis log displays in real-time
7. When complete, navigate to Results tab

**Processing Time:**
- Depends on: File size, number of ROIs, chunk size, number of processes
- Typical: 1-5 minutes for 3-day recording with 4 ROIs

---

## Results Tab

### Generate Plots

After analysis completes:

1. Click **"Generate Plots"**
2. Plots are generated and displayed
3. Each plot appears in a separate matplotlib figure

### Plot Types

**1. Movement Traces**
- Raw movement values over time
- One line per ROI (color-coded)
- Shows baseline threshold (dashed line)
- Lighting conditions (yellow/gray background)

**2. Activity Fraction**
- Percentage of time active per time bin
- Binned data (default: 60-second bins)
- Useful for comparing overall activity levels

**3. Lighting Conditions**
- Light (yellow) and dark (gray) phases
- Extracted from LED metadata (HDF5 only)
- Manual annotation possible for AVI files

**4. Sleep Pattern**
- Sleep bouts detected based on quiescence threshold
- Horizontal bars indicate sleep episodes
- Useful for visualizing circadian sleep patterns

### Customizing Plots

**Plot Options (checkboxes):**
- **Show Baseline Mean**: Display baseline threshold line
- **Show Deviation Band**: Show ±1 SD band around baseline
- **Show Detection Threshold**: Highlight threshold line
- **Show Threshold Stats**: Add text annotations with threshold values

**Adjusting Parameters:**
- **Bin Size**: Adjust time bins for activity fraction plots
- **Quiescence Threshold**: Change sleep detection sensitivity
- **Sleep Threshold**: Minimum duration to classify as sleep bout

### Exporting Results

There are three ways to save results. Only one of them produces a file the
plugin can load again:

| Button | Where | Writes | Reload in plugin |
|--------|-------|--------|------------------|
| **Save Results** | Results tab | Excel workbook | – |
| **Save Metadata (Excel)** | Results tab | `<name>_metadata.xlsx` | – |
| **Save Results to HDF5** | Extended Analysis tab | `<name>_comprehensive_results.h5` | ✓ **Load Results from HDF5**, batch pooling |

**Save Results (Excel)** — one workbook with the sheets `Summary`,
`Raw_Intensity`, `Movement`, `Fraction_Movement`, `Sleep`, `Quiescence`,
`Lighting_Conditions` (when LED data exists) and `Real_Amplitude_MATLAB` /
`Real_Amplitude_Sensor`.

**Save Metadata (Excel)** — the recording's metadata and telemetry
(temperature, humidity, LED, timing) plus the analysis results, in one Excel
file. Despite the HDF5 metadata it contains, it does **not** create an HDF5
file. (Up to plugin version 0.2.0 this button was labelled *Save with HDF5
Metadata*.)

**Save Results to HDF5** — everything needed to continue later: activity
traces, movement/sleep data, thresholds, ROI masks, LED and temperature
records, analysis parameters and the extended-analysis results. This is the
file type the **Batch Datasets** group pools and the Temperature Control reads
when no raw recording is loaded.

Every save ends with a dialog: on success it names the file and folder — after
checking that the file actually exists on disk — and on failure it states the
reason (for example a missing `openpyxl` or no write permission).

**Plot Export:**
1. Click **"Save All Plots"**
2. Choose save directory
3. All plots saved as PNG and PDF
4. High resolution (300 DPI) suitable for publication

---

## Extended Analysis Tab

### Overview

The Extended Analysis tab provides six complementary methods for detecting periodic patterns and synchronization in activity data. The primary period-detection method is the **Chi² periodogram** (Sokolove & Bushell 1978), which uses a Bonferroni-corrected significance threshold (≈ 15.2 for α = 0.05, m = 100 tested periods). See the [Extended Analysis Guide](../EXTENDED_ANALYSIS.md) and [Circadian Analysis Guide](CIRCADIAN_ANALYSIS.md) for full method details.

### Prerequisites

- Main analysis must be completed first
- Sufficient recording duration (minimum 3-4 cycles)
- For 24h rhythm: At least 3 days of data

### Parameters

**Period Range:**
- **Minimum Period**: Lower bound of search range (0-100 hours)
  - Example: 12h for circadian/ultradian
- **Maximum Period**: Upper bound of search range (0-100 hours)
  - Example: 36h for circadian/infradian
- **Range selection tips:**
  - Circadian only: 18-30h
  - Ultradian: 6-18h
  - Broad search: 0-48h

**Statistical Parameters:**
- **Significance Level (α)**: False positive rate (default: 0.05)
  - 0.05 = 5% chance of false positive
  - Decrease for more stringent testing (0.01)
  - Increase for exploratory analysis (0.10)
- **Phase Threshold**: Sleep/wake classification cutoff (default: 0.5)
  - 0.5 = median split
  - Higher = stricter wake criteria
  - Lower = stricter sleep criteria

### Running Analysis

1. Navigate to **"Extended Analysis"** tab
2. Set period range based on expected rhythms
3. Set significance level (usually 0.05)
4. Set phase threshold (usually 0.5)
5. Click **"Run Circadian Analysis"**
6. Wait for processing (typically 10-30 seconds)
7. View results

### Interpreting Results

**Text Output:**
```
ROI 1: SIGNIFICANT circadian rhythm detected
  Dominant Period: 24.2 ± 0.8 hours
  Z-score: 8.45 (p < 0.001)
  Predicted Sleep Phases:
    - Phase 1: 0.0 - 12.1 hours
    - Phase 2: 24.2 - 36.3 hours
  Predicted Wake Phases:
    - Phase 1: 12.1 - 24.2 hours
```

**Periodogram Plot:**
- **X-axis**: Period (hours)
- **Y-axis**: Labeled "Z-score" but represents the chi-squared statistic Z(T) = n × (r_cos² + r_sin²)
- **Blue line**: Z(T) values across all tested periods
- **Gray dashed line**: Bonferroni-corrected significance threshold (≈ 15.2 for α=0.05, m=100)
- **Red/colored marker**: Dominant period (peak Z(T))
- **Green title**: Significant rhythm detected (Z(T) > Bonferroni threshold)
- **Black title**: No significant rhythm

**Biological Interpretation:**

| Peak Location | Interpretation |
|---------------|----------------|
| **24h** | Circadian rhythm (day/night cycle) |
| **12h** | Ultradian rhythm (twice-daily activity) |
| **48h+** | Infradian rhythm (multi-day cycles) |
| **No peak** | Arrhythmic or irregular activity |

### Exporting Extended Analysis

- **Export Current Analysis (Excel)** — the method currently shown
- **Export All Analyses (Excel)** — every method run in this session, one
  workbook
- **Save Results to HDF5** — the complete state including these results, so it
  can be reloaded (see [Exporting Results](#exporting-results))

---

## Frame Viewer Tab

### Overview

The Frame Viewer provides interactive exploration of your raw dataset with temporal context.

### Loading Data

1. Navigate to **"Frame Viewer"** tab
2. Click **"Load Data"**
3. Plugin loads the current dataset (HDF5 or AVI)
4. First frame is displayed in a new layer: "Frame Viewer"

**Time Overlay:**
- Red text in lower-left corner shows elapsed time
- Format: `t = 125.0s (2.08min)`
- Calculated from frame interval metadata

### Navigation Controls

**Slider:**
- Drag to navigate to any frame
- Shows frame number and time

**Button Controls:**
- **|◀**: Jump to first frame
- **◀**: Previous frame
- **▶ Play**: Start/pause playback
- **▶**: Next frame
- **▶|**: Jump to last frame

**Playback Speed:**
- **FPS Control**: Adjust playback speed (1-60 FPS)
- Default: 10 FPS
- Higher FPS = faster playback
- Lower FPS = frame-by-frame inspection

### Frame Information Panel

Displays statistics for current frame:
- Frame number (e.g., 26 / 1000)
- Time (seconds, minutes, hours)
- Image shape (height x width)
- Data type (uint8, uint16, etc.)
- Min/Max pixel values
- Mean pixel value

### Use Cases

**1. Verify ROI Detection:**
- Load data in Frame Viewer
- Navigate through frames
- Confirm ROIs track organisms correctly
- Check for drift or occlusion

**2. Identify Movement Events:**
- Play through recording
- Observe when organisms move
- Correlate with movement plots in Results tab

**3. Visual Quality Control:**
- Check for technical artifacts
- Identify frames with poor contrast
- Detect position shifts or rotation

**4. Presentation and Documentation:**
- Export frames at specific timepoints
- Create screenshots for publications
- Generate supplementary videos

---

## Workflow Examples

### Example 1: Basic HDF5 Analysis

**Scenario:** Single 3-day recording of Nematostella, 6 organisms

**Steps:**
1. Load HDF5 file (Input tab)
2. Detect ROIs (adjust min/max radius as needed)
3. Configure analysis (Analysis tab):
   - Frame interval: 5 seconds
   - Baseline method: full-recording baseline (default), 0.10 multiplier
   - Enable detrending: ON
4. Start analysis
5. Generate plots (Results tab)
6. Export to Excel
7. Run Extended Analysis (24h circadian rhythm expected)
8. Export circadian results

**Expected Output:**
- Movement traces showing day/night activity
- Lighting conditions (if LED data available)
- Sleep patterns with ~24h periodicity
- Circadian analysis with significant 24h peak

### Example 2: AVI Batch Processing

**Scenario:** 10 AVI files (1 hour each), total 10 hours

**Steps:**
1. Load all AVI files as batch (Input tab)
   - Hold Ctrl/Cmd and select all 10 files
   - Plugin concatenates temporally
2. Detect ROIs on first frame
3. Configure analysis:
   - Frame interval: 5 seconds (auto-calculated from FPS)
   - Baseline method: Use first 30 minutes
4. Start analysis (processes all 10 videos)
5. Generate plots
6. Export results

**Expected Output:**
- Continuous 10-hour movement traces
- Seamless transitions between videos
- Consistent ROI tracking

### Example 3: Calibration-Based Comparison

**Scenario:** Compare 5 experimental recordings using shared calibration

**Steps:**

**Day 0 - Calibration:**
1. Load calibration recording
2. Detect ROIs
3. Process calibration baseline (Analysis → Calibration tab)
4. Save results

**Day 1-5 - Experimental Recordings:**
For each recording:
1. Load calibration file (Analysis → Calibration tab)
2. Load calibration dataset
3. Process calibration baseline (uses saved parameters)
4. Load experimental recording
5. Detect ROIs on experimental data
6. Start analysis (uses calibration thresholds)
7. Generate plots and export

**Advantages:**
- Identical thresholds for all recordings
- Comparable across days
- Reduces variability

### Example 4: Long-Term Adaptive Analysis

**Scenario:** 7-day developmental study with changing activity levels

**Steps:**
1. Load 7-day HDF5 recording
2. Detect ROIs
3. Configure analysis:
   - Adaptive method
   - Adaptive duration: 6 hours
   - Recalculation interval: 60 minutes
   - Adaptive multiplier: 0.10
4. Start analysis
5. Generate plots
6. Export results
7. Extended analysis with broad period range (12-48h)

**Expected Output:**
- Adaptive thresholds follow developmental changes
- Long-term circadian stability analysis
- Potential detection of developmental rhythm changes

---

## Tips and Tricks

### ROI Detection

**Problem: Too many false positives**
- Increase DP parameter (e.g., 0.6 → 0.8)
- Increase min distance between ROIs
- Increase Param2 (circle detection threshold)

**Problem: Missing ROIs**
- Decrease DP parameter (e.g., 0.5 → 0.3)
- Expand radius range
- Adjust edge detection (Param1)

**Problem: Overlapping ROIs**
- Increase min distance
- Manual exclusion (note ROI IDs to exclude)

### Movement Analysis

**Problem: Too sensitive (detecting noise)**
- Increase threshold multiplier
- Increase baseline duration
- Enable detrending

**Problem: Too insensitive (missing movements)**
- Decrease threshold multiplier
- Check ROI detection quality
- Verify frame interval is correct

**Problem: Analysis very slow**
- Increase chunk size (e.g., 50 → 100)
- Reduce number of ROIs
- Use subset of data (set End Time)

**Problem: Memory errors**
- Decrease chunk size
- Reduce number of processes
- Close other applications
- Analyze shorter segments

### Extended Analysis

**Problem: No significant rhythms**
- Recording too short (extend to 5-7 days)
- Period range wrong (adjust min/max)
- Significance level too strict (increase α to 0.10)
- Organism arrhythmic (biological variation)

**Problem: Multiple peaks**
- Identify dominant peak (highest Z-score)
- Check for harmonics (24h peak + 12h peak)
- Verify biological plausibility

### Frame Viewer

**Problem: Slow playback**
- Decrease FPS
- Close other napari layers
- Use smaller subset (crop spatially)

**Problem: Time overlay incorrect**
- Verify frame interval in Analysis tab
- Check HDF5 metadata
- For AVI: Ensure correct FPS detection

### General

**Problem: Plugin freezes**
- Check napari console for errors
- Restart napari
- Update to latest plugin version

**Problem: Plots don't appear**
- Check if matplotlib backend is working
- Try closing and reopening Results tab
- Regenerate plots

**Problem: Export fails**
- Check write permissions for save directory
- Ensure sufficient disk space
- For Excel: Install pandas and openpyxl

---

## Processing Zarr Recordings Without the Plugin

A Zarr recording is an ordinary directory of compressed array chunks. Any
Python environment with `zarr` and `numpy` can read it — useful for custom
analyses, other tracking tools, or checking a recording on a machine without
napari. Nothing has to be converted first.

```bash
pip install "zarr>=3" numpy pandas   # zarr-python 3 reads v2 and v3 stores
```

### Store layout

| Path | Content |
|------|---------|
| *(root attributes)* | Recording metadata: `frame_interval` (s), `created_human`, `light_duration_min`, `dark_duration_min`, `start_with_light`, LED powers per phase, frame counts, … |
| `images/frames` | All frames, shape `(T, H, W)`, `uint8`, chunks of 50 frames, Blosc/LZ4 compressed. Older stores keep them at `frames` |
| `timeseries/<name>` | One value per frame, e.g. `recording_elapsed_sec`, `temperature_celsius`, `humidity_percent`, `white_led_power` (%), `ir_led_power` (%), `phase_str` (`light`/`dark`), `frame_mean_intensity` |
| `rois` | Reserved for ROI definitions (empty in current recordings) |

Data is read lazily: indexing an array decompresses only the chunks that
contain the requested frames. A three-day recording can be 40–50 GB on disk,
so never call `frames[:]` on the whole array.

### Look inside

```python
import zarr

root = zarr.open("D:/recordings/nematostella_timelapse_20260522_122817.zarr", mode="r")
print(list(root.keys()))                   # top-level groups
print(sorted(root["timeseries"].keys()))   # one value per frame each
print(dict(root.attrs))                    # recording metadata

frames = root["images/frames"]             # older stores: root["frames"]
print(frames.shape, frames.dtype, frames.chunks)
first = frames[0]                          # decompresses only the 50-frame chunk holding it
```

### Telemetry as a table

```python
import pandas as pd

ts = root["timeseries"]
columns = ["recording_elapsed_sec", "temperature_celsius", "humidity_percent",
           "white_led_power", "ir_led_power", "phase_str", "frame_mean_intensity"]
telemetry = pd.DataFrame({k: ts[k][:] for k in columns if k in ts})

# Failed sensor reads are stored as -2.1 or 0.0 °C - drop and interpolate
temp = telemetry["temperature_celsius"].where(telemetry["temperature_celsius"].between(5, 45))
telemetry["temperature_celsius"] = temp.interpolate(limit_direction="both")
```

The plugin applies the same cleaning (plus a spike test) when it reads
temperature — see [Temperature Control](CIRCADIAN_ANALYSIS.md#temperature-control).

### Activity per ROI

The script below computes the plugin's raw activity signal — the mean absolute
frame-to-frame difference inside each circular ROI, divided by the dtype maximum
— one storage chunk at a time, so memory use stays at a few hundred MB for any
recording length. On a real recording it matches the plugin's values to within
10⁻⁹. It stops before thresholding, normalisation and sleep detection; for those,
load the result into the plugin or apply your own method.

```python
import numpy as np
import zarr

path = "D:/recordings/nematostella_timelapse_20260522_122817.zarr"
root = zarr.open(path, mode="r")
frames = root["images/frames"] if "images" in root else root["frames"]
interval_s = float(root.attrs.get("frame_interval", 5.0))

# ROI circles as (centre_x, centre_y, radius) in pixels - e.g. read off the
# plugin's ROI layer, or from your own detection
rois = [(200, 250, 120), (600, 250, 120), (1000, 250, 120),
        (200, 750, 120), (600, 750, 120), (1000, 750, 120)]
height, width = frames.shape[1:3]
yy, xx = np.mgrid[0:height, 0:width]
masks = [(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2 for cx, cy, r in rois]

step = frames.chunks[0]                          # read whole storage chunks only
norm = float(np.iinfo(frames.dtype).max)         # 255 for uint8
signed = np.int16 if frames.dtype.itemsize == 1 else np.int32  # no wrap-around
activity = [[] for _ in masks]
previous = None                                  # last frame of the previous chunk

for start in range(0, frames.shape[0], step):
    block = frames[start:start + step].astype(signed)
    if previous is not None:                     # keep the diff across the chunk border
        block = np.concatenate([previous[None], block])
    previous = block[-1]
    for values, mask in zip(activity, masks):
        pixels = block[:, mask]                  # (frames, pixels inside the ROI)
        values.extend((np.abs(np.diff(pixels, axis=0)).mean(axis=1) / norm).tolist())

activity = np.array(activity)                    # (n_rois, n_frames - 1)
time_h = np.arange(activity.shape[1]) * interval_s / 3600.0   # time of the earlier frame
np.savetxt("activity.csv", np.column_stack([time_h, activity.T]), delimiter=",",
           header="time_h," + ",".join(f"roi_{i + 1}" for i in range(len(rois))))
```

On one core this runs at roughly 130 frames/s for 1024 × 1224 frames, so a
three-day recording (51 840 frames) takes about 6–7 minutes; the plugin's
parallel pipeline needs about 1.5 minutes for the same file.

### Viewing the frames elsewhere

- **napari** without the plugin: `napari.Viewer().add_image(frames)` — napari
  reads the Zarr array lazily, one chunk per displayed frame.
- **Fiji**: *File › Import › HDF5/N5/Zarr/OME-NGFF* opens Zarr **v2** stores
  (stores with `.zgroup` files); whether v3 stores (`zarr.json`) open depends on
  the Fiji/N5 version.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+O** | Open file dialog (napari) |
| **Ctrl+S** | Save current layer (napari) |
| **Ctrl+W** | Close napari window |
| **Space** | Toggle playback (Frame Viewer, when focused) |
| **←/→** | Previous/next frame (Frame Viewer) |

---

## Support and Resources

- **GitHub Issues**: https://github.com/s1alknau/napari-hdf5-activity/issues
- **Documentation**: https://s1alknau.github.io/Nematostella-time-series/
- **Installation**: not yet on PyPI or the napari hub — install from source,
  `pip install "napari-hdf5-activity[zarr] @ git+https://github.com/s1alknau/napari-hdf5-activity.git"`
- **napari Documentation**: https://napari.org/

---

## Glossary

- **ROI**: Region of Interest, circular area around an organism
- **HDF5**: Hierarchical Data Format version 5, scientific data container
- **AVI**: Audio Video Interleave, video file format
- **Baseline**: Reference period for threshold calculation
- **Threshold**: Movement detection cutoff value
- **Hysteresis**: Two-threshold system for state detection
- **Circadian**: ~24-hour biological rhythm
- **Ultradian**: Biological rhythm < 24 hours
- **Infradian**: Biological rhythm > 24 hours
- **Periodogram**: Frequency-domain representation of timeseries
- **Z-score**: Standardized measure of statistical significance
- **FPS**: Frames per second
