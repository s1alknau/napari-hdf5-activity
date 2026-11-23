# Performance Optimizations

This document describes the performance optimizations implemented in napari-hdf5-activity for processing large datasets with high resolution and RGB channels.

---

## Table of Contents
1. [Overview](#overview)
2. [RGB to Grayscale Conversion](#rgb-to-grayscale-conversion)
3. [Dynamic RAM Management](#dynamic-ram-management)
4. [Parallel Processing on Windows](#parallel-processing-on-windows)
5. [Chunking, Border Frames and Failed Workers](#chunking-border-frames-and-failed-workers)
6. [Worker Thread Management](#worker-thread-management)
7. [Recommended Settings](#recommended-settings)
8. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

Processing large HDF5 files with high resolution (e.g., 1920×1080) and RGB channels can be computationally intensive. We have implemented several optimizations to significantly improve processing speed while maintaining memory safety.

### Key Optimizations
- **Vectorized RGB→Grayscale conversion**: 10-100× faster than frame-by-frame processing
- **Memory-aware worker count**: Sized from the memory Windows can still commit, not just free RAM
- **Lean worker processes**: Workers no longer import napari or CuPy (~390 MB instead of ~1 GB each)
- **Chunk-aligned, gap-free parallel reading**: Each Zarr storage chunk is decompressed once; no frame difference is lost at chunk borders; chunks of a crashed worker are recomputed
- **Improved worker thread management**: Prevents duplicate analyses and ensures clean shutdown

---

## RGB to Grayscale Conversion

### Problem
Old HDF5 files with RGB channels require conversion to grayscale for movement analysis. The original implementation used frame-by-frame conversion with OpenCV:

```python
# OLD (SLOW): Frame-by-frame conversion
grayscale_stack = np.array(
    [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in image_stack]
)
```

**Issues:**
- One function call per frame (10,000 frames = 10,000 calls)
- Python loop overhead
- Memory inefficient (creates temporary lists)

### Solution
Vectorized NumPy operation using `tensordot`:

```python
# NEW (FAST): Vectorized conversion
weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
grayscale_stack = np.tensordot(image_stack, weights, axes=([3], [0]))
grayscale_stack = grayscale_stack.astype(image_stack.dtype)
```

**Benefits:**
- Single operation for entire stack
- Uses optimized BLAS/LAPACK routines
- Memory efficient (no intermediate arrays)
- Standard ITU-R 601-2 luma transform (same as cv2.cvtColor)

### Performance Impact

| Resolution | Frames | Old Method | New Method | Speedup |
|------------|--------|------------|------------|---------|
| 640×480 RGB | 1,000 | ~5s | ~0.5s | **10×** |
| 1920×1080 RGB | 1,000 | ~15s | ~1s | **15×** |
| 1920×1080 RGB | 10,000 | ~150s | ~10s | **15×** |

**Real-world impact:** A 2-hour RGB recording (10,000 frames) that previously took 2.5 minutes to convert now takes only 10 seconds.

---

## Dynamic RAM Management

### Problem
Processing large files in parallel can exhaust system memory, leading to
swapping, out-of-memory errors, and — on Windows — worker processes that cannot
even be started.

On Windows the limit that matters is the **commit limit** (physical RAM + page
file), not free RAM. Every process reserves commit for memory it *may* touch;
once the limit is reached, allocations fail even with gigabytes of RAM still
free. In one observed case 14 GB of RAM were free while the commit charge was at
35.5 / 35.5 GB, and new worker processes hung while starting.

### Solution
**Number of workers** — `calculate_safe_process_count()` budgets ~400 MB per
worker (measured peak ~390 MB, see below) plus the ROI masks, against

```python
available = min(psutil.virtual_memory().available,   # free RAM
                free_commit_charge)                     # Windows only, GlobalMemoryStatusEx
```

keeping 2 GB in reserve for napari and the OS. The CPU-based default is
`min(4, 75 % of cores)` on Windows. When memory is tight, fewer workers start and
the analysis gets slower instead of stalling. The Analysis tab applies the same
check to the *Number of Processes* setting before each run.

**Chunk size** — `calculate_optimal_chunk_size()` sizes chunks from the frame
size and the memory budget, divided by the number of workers, and then rounds to
whole Zarr storage chunks (see below).

**Tasks in flight** — at most `max(4, 3 × workers)` chunks are submitted at a
time; a new one is submitted whenever one finishes. Finished results are merged
immediately, so memory use stays flat regardless of file length.

```
Processing pipeline:
┌──────────────────────────────────────────┐
│ Recording (e.g. 1037 chunks of 50 frames)│
└──────────────┬───────────────────────────┘
               ▼
    ┌──────────────────────┐
    │ In flight: ≤ 3×N     │  ← refilled as chunks finish
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │ Process pool (N)     │  ← N from the memory check
    └──────────┬───────────┘
               ▼
   Results merged + border diffs added
```

**If the page file is small**, the commit limit is low even on machines with
plenty of RAM. Letting Windows manage the page file size (*System Properties →
Advanced → Performance → Virtual memory → Automatically manage*) raises it on
demand.

---

## Parallel Processing on Windows

Windows has no `fork`: every worker is a fresh Python interpreter that, by
default, first re-runs the parent's main script. Inside napari that script is
`napari-script.py`, so every worker imported all of napari (Qt, vispy, numba, …)
it never uses. Workers starting at the same time also wrote napari's theme icons
to the same cache files and collided:

```
PermissionError: [Errno 13] Permission denied: '...\napari\Cache\...\_themes\dark\3D.svg'
```

— the pool broke and the analysis fell back to a single process.

All three process pools (frame reading, multi-file batches, per-ROI movement
detection) are now created inside `worker_spawn_environment()`
(`_mp_utils.py`). While a pool lives it

1. hides the parent's `__main__.__file__`, so workers do not re-run the main
   script — the worker functions live in this package, not in `__main__`;
2. sets a flag that makes workers skip the CuPy import — they compute on NumPy
   masks, and initialising CUDA reserves ~1 GB of commit per process;
3. runs BLAS/OpenMP single-threaded per worker (`OPENBLAS_NUM_THREADS=1` etc.) —
   the parallelism comes from the processes, and a many-threaded OpenBLAS
   reserves hundreds of MB per process.

Everything is restored afterwards, also when the pool raises.

| Per worker | Main script executed | Commit charge |
|------------|----------------------|---------------|
| Before | parent + every worker | ~1070 MB |
| After | parent only | ~390 MB |

---

## Chunking, Border Frames and Failed Workers

**Storage-aligned chunks.** The recording plugin writes Zarr frames in storage
chunks of 50 frames. A task that straddles two storage chunks decompresses both,
so task boundaries are rounded to whole storage chunks
(`storage_chunk_frames`, read from the store).

**Border frames.** Activity is the difference between consecutive frames, so a
chunk of *n* frames yields *n − 1* values; the difference between the last frame
of one chunk and the first of the next used to be lost — about 2 % of all values
with 50-frame chunks. Workers now return their first and last frame, and the
main process computes the border differences from them. No frame is read twice,
and parallel and sequential processing give identical results.

**Failed workers.** If a worker dies (e.g. out of memory), its chunk — and any
chunk not yet submitted once the pool is broken — is recomputed in the main
process instead of being dropped. Previously all remaining chunks were
discarded without a message, leaving gaps in the activity trace. Only a chunk
that fails a second time is reported as missing.

**ROI bounding boxes.** Each ROI is evaluated on its mask's bounding box instead
of the full frame, and the cast to a signed type (int16 for uint8, int32 for
uint16 — int16 overflowed above 32 767) happens after cropping.

**Sequential fallback.** When no pool can be used, the Zarr store is opened once
for the whole run instead of once per chunk; for zip stores that meant parsing
the entire central directory every time.

---

## Worker Thread Management

### Problem
Analysis runs in background threads. Previous implementation had issues:
- Multiple analyses could run simultaneously (progress bar confusion)
- Stop button didn't cleanly terminate workers
- Worker references not properly cleared
- Callbacks could fire after stop requested

### Solution

#### 1. Prevent Duplicate Analyses
```python
def run_analysis(self):
    # Check if analysis is already running
    if hasattr(self, 'current_worker') and self.current_worker is not None:
        self._log_message("⚠️ Analysis already running!")
        return
    # ... continue with analysis
```

#### 2. Clean Worker Shutdown
```python
def stop_analysis(self):
    # Set cancellation flag
    self._cancel_requested = True

    # Disconnect signals to prevent callbacks
    if self.current_worker is not None:
        self.current_worker.returned.disconnect()
        self.current_worker.errored.disconnect()
        self.current_worker.finished.disconnect()

        # Clear worker reference
        self.current_worker = None
```

#### 3. Proper Cleanup
```python
def _analysis_done(self):
    # Always clear worker reference when done
    self.current_worker = None
    self._cancel_requested = False
    # ... reset UI state
```

**Benefits:**
- Single analysis at a time (clear progress indication)
- Clean stop/restart workflow
- No zombie threads
- No orphaned callbacks

---

## Recommended Settings

### For External SSD with Limited RAM (8-16 GB)

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| **Chunk Size** | 50-100 frames | Balance between I/O and RAM |
| **Processes** | 2-3 | Avoid RAM exhaustion |
| **Expected RAM** | ~300-600 MB | Manageable for most systems |

### For High-End System (32+ GB RAM, NVMe SSD)

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| **Chunk Size** | 100-200 frames | Maximize sequential I/O |
| **Processes** | 4-6 | Full CPU utilization |
| **Expected RAM** | ~2-4 GB | Plenty of headroom |

### For HDD Storage

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| **Chunk Size** | 150-200 frames | Minimize seek operations |
| **Processes** | 2-3 | I/O bound, not CPU bound |

**Note:** The dynamic RAM management system automatically adjusts queue size, so these are starting recommendations. The system will adapt to your specific hardware.

---

## Performance Benchmarks

### Real-World Test Results

**Test Configuration:**
- **File**: Nematostella timelapse (IR-only, grayscale)
- **Resolution**: 1024×1224 pixels
- **Frames**: 6,103 (~8.5 hours at 5s intervals)
- **ROIs**: 6
- **Storage**: External SSD
- **System**: Mid-range (limited RAM)

### Results by Configuration

| Chunk Size | Processes | Time | FPS | Speedup | Efficiency |
|------------|-----------|------|-----|---------|------------|
| 20 | 1 | 108.7s | 56.2 | 1.00× | 100% |
| 20 | 2 | 57.3s | 106.5 | 1.90× | 95% |
| 20 | 4 | 34.2s | 178.3 | 3.18× | 80% |
| **50** | **1** | **101.2s** | **60.3** | **1.07×** | **107%** |
| **50** | **2** | **55.2s** | **110.5** | **1.97×** | **99%** |
| **50** | **4** | **32.5s** | **188.1** | **3.35×** | **84%** ← **Best** |
| 100 | 1 | 98.9s | 61.7 | 1.10× | 110% |
| 100 | 2 | 54.1s | 112.9 | 2.01× | 101% |
| 100 | 4 | 33.9s | 180.3 | 3.21× | 80% |

**Key Findings:**

✅ **Optimal Configuration (External SSD):**
- Chunk Size: **50 frames**
- Processes: **4**
- Processing time: **32.5 seconds** (vs 108.7s sequential)
- Speedup: **3.35×** (84% of theoretical 4× maximum)
- Processing rate: **188 frames/second**

✅ **RAM-Safe Configuration:**
- Chunk Size: **100 frames**
- Processes: **2**
- Processing time: **54.1 seconds**
- Speedup: **2.01×** (still very good!)
- Lower memory footprint

✅ **Chunk Size Impact:**
- Larger chunks (50-100) slightly faster than small chunks (20)
- Improvement: ~7-10% due to reduced I/O overhead
- Optimal: 50 frames for external SSD

### Three-Day Zarr Recording (2026-11)

| | |
|---|---|
| File | Zarr directory store, 46 GB on disk |
| Frames | 51 840 (72 h at 5 s), 1024 × 1224, uint8, 50-frame storage chunks |
| ROIs | 6 |
| Workers | 4 (Windows, 32 GB RAM) |
| **Time** | **84 s — 615 frames/s** |
| Values per ROI | 51 839 of 51 839 (no gaps) |
| For comparison | ~130 frames/s single-threaded with plain `zarr` + NumPy (see the User Guide, *Processing Zarr Recordings Without the Plugin*) |

When a second analysis runs at the same time, both slow down markedly — they
compete for CPU, memory and disk.

### Scalability Analysis

**Speedup by Process Count:**
- 1→2 processes: **1.90× speedup** (95% efficient)
- 1→4 processes: **3.18× speedup** (80% efficient)

**Excellent scaling!** The ~20% overhead at 4 processes is typical and acceptable, caused by:
- Process communication overhead
- I/O contention
- Dynamic queue management

### CPU Utilization

| Configuration | CPU Usage | Notes |
|---------------|-----------|-------|
| 1 process | ~25% | Single core utilized |
| 2 processes | ~50% | Good parallel efficiency |
| 4 processes | ~80% | Excellent utilization |

**Much improved** from old implementation (20-40% usage)!

### Recommended Workflow

1. **First Run**: Use default settings (chunk=50, processes=auto)
2. **Monitor**: Check RAM usage and CPU load
3. **Adjust if needed**:
   - High RAM usage (>80%): Reduce chunk size or processes
   - Low CPU usage (<50%): Increase chunk size for better I/O
   - External SSD: chunk=50-100
   - HDD: chunk=150-200

---

## Technical Details

### RGB to Grayscale Formula
Both methods use the ITU-R 601-2 luma transform:
```
Y = 0.299*R + 0.587*G + 0.114*B
```

**Why these weights?**
- Based on human perception (green is brightest, blue is darkest)
- Standard used by cv2.cvtColor
- Ensures consistent results with other tools

### Memory Estimation
```python
# Single frame memory
frame_memory = width × height × channels × bytes_per_pixel

# Chunk memory
chunk_memory = frame_memory × frames_per_chunk

# Total process memory (worst case)
total_memory = chunk_memory × num_processes × 2  # ×2 for processing overhead
```

**Example (1920×1080 RGB, uint8, chunk=100, 4 processes):**
```
Frame: 1920 × 1080 × 3 × 1 = 6.2 MB
Chunk: 6.2 MB × 100 = 620 MB
Total: 620 MB × 4 × 2 ≈ 5 GB (peak usage)
```

### Tasks in Flight
```python
max_in_flight = max(4, num_processes * 3)   # submitted at once, refilled as they finish
```

---

## Troubleshooting

### High RAM Usage
**Symptoms:** System slowing down, RAM >90%

**Solutions:**
1. Reduce chunk size: 100 → 50 frames
2. Reduce processes: 4 → 2
3. Close other applications
4. Consider processing in batches

### Low CPU Usage (<40%)
**Symptoms:** Long processing time, disk activity high

**Solutions:**
1. Increase chunk size: 50 → 100 frames
2. Check storage speed (HDD vs SSD)
3. Reduce number of processes (I/O bound)

### Analysis Won't Stop
**Symptoms:** Stop button clicked but analysis continues

**Expected behavior:** Analysis stops after current chunk completes (1-30 seconds depending on chunk size)

**If it doesn't stop:**
1. Wait for current chunk to finish
2. Check log for "STOP requested"
3. Restart application if frozen

### "Multi-processing failed" / analysis runs single-threaded
**Symptoms:** Log shows `Multi-processing failed (...)` or `MemoryError reading
chunk`, processing is much slower than expected

**Causes and solutions:**
1. Commit limit reached (Windows): close memory-heavy programs or enlarge the
   page file — see [Dynamic RAM Management](#dynamic-ram-management)
2. Two analyses running in parallel (e.g. two napari windows) share CPU, RAM and
   disk; run them one after the other
3. The log now prints the exception type and traceback — include it when
   reporting a problem

### "Analysis already running" Message
**Cause:** Trying to start new analysis while one is running

**Solution:** Wait for current analysis to finish or click Stop first

---

## Future Optimizations

Potential areas for further improvement:

1. **GPU acceleration** for RGB→Gray conversion (OpenGL/CUDA)
2. **Compression-aware chunking** for HDF5 files
3. **Incremental processing** for very large files (>100 GB)
4. **Smart chunk sizing** based on file structure analysis
5. **Parallel I/O** with memory-mapped files

---

## References

- NumPy tensordot documentation: https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html
- ITU-R BT.601-7: https://www.itu.int/rec/R-REC-BT.601
- psutil documentation: https://psutil.readthedocs.io/
- Python ProcessPoolExecutor: https://docs.python.org/3/library/concurrent.futures.html

---

**Last Updated:** 2026-11-23
**Version:** 1.1
