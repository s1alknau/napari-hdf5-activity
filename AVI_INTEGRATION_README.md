# AVI File Integration for napari-hdf5-activity

## Overview

The napari-hdf5-activity plugin now supports AVI video files in addition to HDF5 files. This enables analysis of videos using the same movement analysis pipeline.

## Features

### Supported Operations
- **ROI Detection**: Automatic detection on first frame
- **Movement Analysis**: Pixel-difference based movement quantification
- **Threshold Methods**: Baseline, Calibration, and Adaptive
- **Batch Processing**: Load multiple AVI files as temporal sequence
- **Memory Efficient**: Only first frame loaded for ROI detection, full data loaded during analysis

### Frame Sampling
- Default frame interval: **5 seconds** (matching HDF5 workflow)
- Automatic calculation based on video FPS
- Example: 30 FPS video → samples every 150th frame (30 × 5 = 150)

## Usage

### 1. Load Single AVI File

**Via UI:**
1. Click "Load File"
2. Select `.avi` file
3. First frame displayed for ROI detection

**Via napari:**
```python
import napari
viewer = napari.Viewer()
viewer.open('video.avi', plugin='napari-hdf5-activity')
```

### 2. Load Multiple AVI Files (Batch)

**Via UI:**
1. Click "Load File"
2. Hold Ctrl/Cmd and select multiple `.avi` files
3. Files loaded as continuous timeseries with temporal concatenation

Example:
- video_001.avi: 10 minutes → t = 0 to 600s
- video_002.avi: 10 minutes → t = 600 to 1200s
- video_003.avi: 10 minutes → t = 1200 to 1800s

**Via napari:**
```python
import napari
viewer = napari.Viewer()
viewer.open(['video1.avi', 'video2.avi', 'video3.avi'],
            plugin='napari-hdf5-activity')
```

### 3. Load Directory with AVIs

**Via UI:**
1. Click "Load Directory"
2. Select folder containing `.avi` files
3. All AVIs loaded as batch (sorted alphabetically)

### 4. Analyze

Same workflow as HDF5:
1. **ROI Detection** → Detect ROIs
2. **Movement Analysis** → Select method and Process Data
3. **Results** → Generate Plots and Export

## Technical Details

### Frame Interval Calculation

```python
video_fps = 30.0  # From video metadata
target_interval = 5.0  # seconds
frames_per_sample = int(video_fps * target_interval)  # = 150
```

This ensures consistent temporal resolution regardless of source video FPS.

### Temporal Concatenation

When loading multiple videos:
1. Each video's duration calculated from FPS and frame count
2. Timestamps calculated with cumulative offset:
   ```python
   for video_idx, video in enumerate(videos):
       for frame in sampled_frames:
           timestamp = frame_idx / video_fps + time_offset
       time_offset += video.duration
   ```

### Memory Management

**Loading Phase:**
- Only first frame loaded (~2-4 MB)
- Metadata extracted (FPS, duration, resolution)
- ROI detection performed

**Analysis Phase:**
- All frames loaded on-demand
- Processed in chunks for memory efficiency
- Results stored progressively

## Differences from HDF5

| Feature | HDF5 | AVI |
|---------|------|-----|
| Frame Loading | Direct dataset access | opencv video reader |
| LED Data | ✓ Available in timeseries | ✗ Not available |
| Lighting Plot | ✓ Automatic from LED data | ✗ Not shown |
| Metadata | ✓ Embedded in file | ✗ Basic (FPS, duration) |
| Batch Processing | Directory scan | Multi-select or directory |
| Movement Analysis | ✓ Identical | ✓ Identical |
| Threshold Methods | ✓ All methods | ✓ All methods |
| Export | ✓ Excel/CSV/Plots | ✓ Excel/CSV/Plots |

**Note:** Lighting conditions plots are only available for HDF5 files where LED power data is stored in the timeseries.

## Requirements

```bash
pip install opencv-python
```

The plugin will show an error if opencv is not installed when trying to load AVI files.

## Troubleshooting

### AVI file won't load
- **Solution:** Install opencv-python: `pip install opencv-python`
- Check codec is supported (MJPEG, H264, etc.)

### First frame not showing
- Check if file is actually AVI (not corrupted)
- Try loading through "Load Directory" instead
- Check log for error messages

### Memory error during batch loading
- Load fewer files at once
- Increase frame interval (reduces total frames)
- Close other applications

### Analysis slower than HDF5
- AVI decoding is slower than HDF5 direct access
- Consider converting to HDF5 for repeated analysis
- Reduce frame interval if acceptable for analysis

## Example Workflow

```python
# Complete analysis workflow
import napari
from napari_hdf5_activity import HDF5AnalysisWidget

# 1. Launch napari
viewer = napari.Viewer()

# 2. Load AVI files
viewer.open(['day1.avi', 'day2.avi', 'day3.avi'],
            plugin='napari-hdf5-activity')

# 3. Open plugin widget
widget = HDF5AnalysisWidget(viewer)
viewer.window.add_dock_widget(widget)

# 4. Detect ROIs (via UI)
# 5. Process Data (via UI)
# 6. Generate Plots (via UI)
# 7. Export Results (via UI)
```

## MATLAB Compatibility

The AVI support is compatible with the MATLAB workflow:
- Same frame sampling approach (frameRateOffline)
- Same movement metric (pixel difference)
- Compatible with ROI.mat ellipse/polygon definitions
- Results comparable to MATLAB output

## Performance

Typical processing times (3 days, 30 FPS videos, 5s interval):

| Operation | Time |
|-----------|------|
| Load first frame | < 1 second |
| ROI detection | 1-2 seconds |
| Load all frames (3 videos) | 5-10 seconds |
| Movement analysis | 10-30 seconds |
| Plot generation | 2-5 seconds |
| Export to Excel | 3-8 seconds |

**Note:** Times vary based on video codec, resolution, and hardware.
