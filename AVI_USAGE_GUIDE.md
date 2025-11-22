# AVI File Support - Usage Guide

## Overview

The napari-hdf5-activity plugin now supports **both HDF5 and AVI video files** through the same UI buttons and analysis pipeline. All features work identically for both file types.

## Loading AVI Files

### Method 1: Load File Button

**Single AVI File:**
1. Click "Load File" button
2. Select a single `.avi` file
3. Plugin will load it with the default 5-second frame interval

**Multiple AVI Files (Batch):**
1. Click "Load File" button
2. Hold Ctrl (Windows/Linux) or Cmd (Mac) and select multiple `.avi` files
3. Plugin will automatically:
   - Concatenate videos temporally (Video 1: 0-600s, Video 2: 600-1200s, etc.)
   - Sample frames at 5-second intervals (0.2 FPS effective)
   - Create continuous timestamps across all videos

### Method 2: Load Directory Button

1. Click "Load Directory" button
2. Select a folder containing AVI and/or HDF5 files
3. Plugin will scan and display:
   - Number of HDF5 files found
   - Number of AVI files found
   - List of files in the log

## Frame Interval and Sampling

**Default Frame Interval: 5 seconds (0.2 FPS effective)**

This matches the HDF5 workflow. The plugin automatically samples frames from your AVI videos:

| Video FPS | Frames Sampled | Effective Rate |
|-----------|----------------|----------------|
| 30 FPS    | Every 150th    | 0.2 FPS        |
| 5 FPS     | Every 25th     | 0.2 FPS        |
| Any FPS   | Automatic      | 0.2 FPS        |

**Example Timeline:**
```
t = 0s    → Frame 1 (from Video 1)
t = 5s    → Frame 2 (from Video 1)
t = 10s   → Frame 3 (from Video 1)
...
t = 600s  → Frame 121 (end of Video 1)
t = 605s  → Frame 122 (start of Video 2)
...
```

## Metadata for AVI Files

### Required: metadata.json

Place a `metadata.json` file next to your AVI file(s):

```
Videos/
  ├── video_001.avi
  ├── video_002.avi
  └── metadata.json
```

### Quick Setup: Generate Metadata

Use the helper script to create a metadata file:

```bash
cd napari-hdf5-activity
python create_avi_metadata.py --output "C:/path/to/your/videos/metadata.json"
```

**With custom settings:**
```bash
python create_avi_metadata.py \
  --output metadata.json \
  --fps 5.0 \
  --interval 5.0 \
  --light-start 7 \
  --light-end 19 \
  --days 3
```

### Metadata Format

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
      "power_percent": 100,
      "note": "IR LED always on for video recording"
    },
    "white_led": {
      "status": "scheduled",
      "schedule_type": "custom",
      "light_periods": [
        {"start_hour": 7, "end_hour": 19, "description": "Day 1: 07:00-19:00"},
        {"start_hour": 31, "end_hour": 43, "description": "Day 2: 07:00-19:00"},
        {"start_hour": 55, "end_hour": 67, "description": "Day 3: 07:00-19:00"}
      ]
    }
  }
}
```

## Analysis Pipeline

Once loaded, **all analysis features work identically** for AVI and HDF5 files:

### 1. ROI Detection
- Click "ROI Detection" tab
- Click "Detect ROIs" button
- Works on first frame of AVI just like HDF5

### 2. Movement Analysis
- Click "Movement Analysis" tab
- Select threshold method (Baseline, Calibration, or Adaptive)
- Click "Process Data" button
- Same hysteresis algorithm as HDF5

### 3. Plot Generation
- Click "Generate Plots" button
- **Lighting Conditions plot** automatically shows light/dark phases from metadata
- All plots identical to HDF5 output

## Lighting Conditions Plot

The plugin automatically detects light/dark phases for AVI files:

- **Light Phase** = White LED ON (shown in yellow)
- **Dark Phase** = Only IR LED ON (shown in gray)

**Important:** AVI videos are recorded with **continuous IR illumination**. The white LED is only on during light phases. Animals only respond to white LED changes.

## Batch Processing from Command Line

For scripted workflows, use the standalone batch processor:

```bash
python process_avi_batch.py --dir "C:/path/to/videos"
```

**Options:**
```bash
# Specific videos
python process_avi_batch.py --videos video1.avi video2.avi video3.avi

# Custom frame interval
python process_avi_batch.py --dir "C:/path/to/videos" --interval 5.0

# Without GUI (data loading only)
python process_avi_batch.py --dir "C:/path/to/videos" --no-gui
```

## Troubleshooting

### "AVI support not available"
Install OpenCV:
```bash
pip install opencv-python
```

### "No metadata found"
The plugin will use default settings (5s interval, 12h light cycle). For accurate lighting analysis, create a `metadata.json` file.

### Single vs. Batch Loading
- **Single file:** Processes one AVI independently
- **Multiple files:** Concatenates temporally (for continuous recordings split across files)

### Frame Interval Mismatch
If your metadata specifies a different interval, the plugin will use that value. Default is always 5.0 seconds.

## Example Workflow

1. **Prepare videos:**
   ```
   C:/Experiments/20240627_Nema/
     ├── video_001.avi
     ├── video_002.avi
     ├── video_003.avi
     └── metadata.json  (created with create_avi_metadata.py)
   ```

2. **Open napari:**
   ```bash
   napari
   ```

3. **Load plugin:**
   - Plugins → napari-hdf5-activity

4. **Load videos:**
   - Click "Load File"
   - Select all 3 AVI files (Ctrl+Click)
   - Plugin loads as continuous 3-video timeseries

5. **Detect ROIs:**
   - ROI Detection tab → "Detect ROIs"

6. **Process:**
   - Movement Analysis tab
   - Select "Adaptive" threshold
   - "Process Data"

7. **Generate plots:**
   - "Generate Plots"
   - Lighting conditions automatically shown

8. **Export:**
   - Results saved to Excel
   - Plots saved as PNG/PDF

## Comparison: HDF5 vs AVI

| Feature | HDF5 | AVI |
|---------|------|-----|
| Frame loading | Direct dataset access | opencv video reader |
| Frame interval | Stored in HDF5 | From metadata.json |
| LED data | Stored in timeseries | From metadata.json |
| Batch processing | Directory scan | Multi-select or directory scan |
| ROI detection | ✓ | ✓ |
| Movement analysis | ✓ | ✓ |
| Hysteresis algorithm | ✓ | ✓ |
| Lighting plot | ✓ | ✓ |
| Excel export | ✓ | ✓ |

## Notes

- First frame of each AVI is sampled at t=0
- Temporal concatenation preserves exact timing across videos
- IR LED assumed always on (100%) during recording
- White LED schedule determines light/dark phases
- All threshold methods (Baseline/Calibration/Adaptive) work identically
- Plot generation and export features unchanged
