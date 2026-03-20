"""create_test_zarr.py — Generate a synthetic Zarr test file with 6 ROIs.

Produces ``tests/test_data_6rois.zarr`` that mirrors the structure expected by
napari-hdf5-activity:

  /frames                   uint16 (200, 256, 256)   — stacked image frames
  /timeseries/
    frame_mean              float64 (200,)
    frame_min               float64 (200,)
    frame_max               float64 (200,)
    frame_std               float64 (200,)
    temperature             float64 (200,)
    humidity                float64 (200,)
    actual_intervals        float64 (200,)
    expected_intervals      float64 (200,)
    frame_drift             float64 (200,)
    cumulative_drift        float64 (200,)
    led_white_power_percent float64 (200,)
    led_ir_power_percent    float64 (200,)
  /metadata
    frame_interval          scalar  300.0  (seconds, i.e. 5-min bins)
    total_frames            scalar  200
    experiment_name         scalar  "test_6roi_zarr"
    creation_date           scalar  ISO-8601 string

Six circular ROI blobs are embedded in the frames.  Each ROI pulses with a
distinct circadian-like period (20–28 h) so that period-finding algorithms have
something to detect.

Run from the repository root::

    python tests/create_test_zarr.py
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import zarr

_ZARR_V3 = tuple(int(x) for x in zarr.__version__.split(".")[:2]) >= (3, 0)

# zarr v3 uses zarr.codecs.BloscCodec; v2 uses numcodecs.Blosc.
# Pick the right one so compressor= works regardless of zarr version.
_Blosc = None
if _ZARR_V3:
    try:
        from zarr.codecs import BloscCodec as _BloscCodec
        _Blosc = _BloscCodec(cname="lz4", clevel=3)
    except ImportError:
        pass
else:
    try:
        from numcodecs import Blosc as _BloscClass
        _Blosc = _BloscClass(cname="lz4", clevel=3)
    except ImportError:
        _Blosc = None


def _create_array(group, name, data, chunks, dtype, compressor=None):
    """Create a zarr array inside *group* — compatible with zarr v2 and v3.

    - zarr v2: ``group.create_dataset(name, data=data, ...)``
    - zarr v3: ``group.create_dataset`` was removed; use ``group.create_array``
               or direct assignment ``group[name] = zarr.array(...)``.
    """
    # zarr v3 create_dataset requires shape= explicitly even when data= is given
    kwargs = dict(data=data, shape=data.shape, chunks=chunks, dtype=dtype)
    if compressor is not None:
        # zarr v3 uses compressors= (list); v2 uses compressor=
        if _ZARR_V3:
            kwargs["compressors"] = [compressor]
        else:
            kwargs["compressor"] = compressor
    # Try v2 API first (create_dataset), fall back to v3 (create_array)
    create_fn = getattr(group, "create_dataset", None) or getattr(group, "create_array", None)
    if create_fn is None:
        raise RuntimeError("Cannot find create_dataset or create_array on zarr Group")
    return create_fn(name, **kwargs)


def _require_group(store, name):
    """Open or create a sub-group — compatible with zarr v2 and v3."""
    require_fn = getattr(store, "require_group", None)
    if require_fn is not None:
        return require_fn(name)
    # zarr v3 fallback: open_group with create mode
    return zarr.open_group(store=store, path=name, mode="a")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_PATH_ZIP = os.path.join(os.path.dirname(__file__), "test_data_6rois.zarr")
OUT_PATH_DIR = os.path.join(os.path.dirname(__file__), "test_data_6rois_dir.zarr")

FRAME_INTERVAL = 5.0    # seconds per frame
DURATION_H = 72         # total recording duration in hours
N_FRAMES = int(DURATION_H * 3600 / FRAME_INTERVAL)  # 51 840 frames
FRAME_H = 256
FRAME_W = 256
WRITE_CHUNK = 500       # frames written per batch to limit RAM usage
DTYPE = np.uint16
MAX_INTENSITY = 3000    # baseline background level (out of 65535)
ROI_PEAK = 1500         # additional signal on top of baseline for active ROI

# Six ROI centres and radii (centre_y, centre_x, radius)
ROIS = [
    (50,  50,  18),   # ROI 1 — upper-left
    (50,  128, 18),   # ROI 2 — upper-centre
    (50,  206, 18),   # ROI 3 — upper-right
    (206, 50,  18),   # ROI 4 — lower-left
    (206, 128, 18),   # ROI 5 — lower-centre
    (206, 206, 18),   # ROI 6 — lower-right
]

# Each ROI gets its own period (hours) so the analysis can distinguish them
ROI_PERIODS_H = [24.0, 22.0, 26.0, 20.0, 28.0, 23.5]

RNG_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roi_mask(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    """Boolean mask for a circular ROI."""
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2




# ---------------------------------------------------------------------------
# Build frames array
# ---------------------------------------------------------------------------


def build_frames_chunk(start: int, end: int, rng: np.random.Generator) -> np.ndarray:
    """Build a chunk of frames [start, end) without holding all frames in RAM."""
    n = end - start
    frames = np.full((n, FRAME_H, FRAME_W), MAX_INTENSITY, dtype=np.float32)
    frames += rng.normal(0, 50, size=frames.shape)

    # Pre-compute ROI masks
    for idx, ((cy, cx, r), period_h) in enumerate(zip(ROIS, ROI_PERIODS_H)):
        mask = _roi_mask(FRAME_H, FRAME_W, cy, cx, r)
        phase = idx * 1.5
        # Compute signal only for this chunk's time range
        t_h = np.arange(start, end) * FRAME_INTERVAL / 3600.0
        signal = ROI_PEAK * (1 + np.cos(2 * np.pi * (t_h - phase) / period_h)) / 2
        signal += rng.normal(0, ROI_PEAK * 0.05, size=n)
        signal = np.clip(signal, 0, None)
        frames[:, mask] += signal[:, np.newaxis]

    return np.clip(frames, 0, 65535).astype(DTYPE)


# ---------------------------------------------------------------------------
# Build telemetry timeseries
# ---------------------------------------------------------------------------


def build_timeseries(rng: np.random.Generator) -> dict:
    """Build telemetry timeseries analytically (no full frame array needed)."""
    ts = {}
    t_h = np.arange(N_FRAMES) * FRAME_INTERVAL / 3600.0

    # Approximate frame statistics from the signal model
    # Mean = background + average ROI contribution
    avg_roi_signal = ROI_PEAK * 0.5  # average of cosine over full cycles
    roi_pixel_fraction = sum(
        np.pi * r ** 2 for _, _, r in ROIS
    ) / (FRAME_H * FRAME_W)
    ts["frame_mean"] = np.full(N_FRAMES, MAX_INTENSITY + avg_roi_signal * roi_pixel_fraction) \
                       + rng.normal(0, 5, N_FRAMES)
    ts["frame_min"]  = np.full(N_FRAMES, float(MAX_INTENSITY - 150)) + rng.normal(0, 2, N_FRAMES)
    ts["frame_max"]  = np.full(N_FRAMES, float(MAX_INTENSITY + ROI_PEAK)) + rng.normal(0, 10, N_FRAMES)
    ts["frame_std"]  = np.full(N_FRAMES, 80.0) + rng.normal(0, 3, N_FRAMES)

    # Environment: temperature cycles around 21 °C with slow drift
    ts["temperature"] = (21.0 + 0.5 * np.sin(2 * np.pi * t_h / 24.0)
                         + rng.normal(0, 0.1, N_FRAMES))
    ts["humidity"]    = (55.0 + 3.0 * np.sin(2 * np.pi * t_h / 24.0 + 1.0)
                         + rng.normal(0, 0.5, N_FRAMES))

    # Timing: small jitter around nominal interval
    expected = np.full(N_FRAMES, FRAME_INTERVAL, dtype=np.float64)
    jitter   = rng.normal(0, 0.5, N_FRAMES)
    actual   = expected + jitter
    ts["expected_intervals"] = expected
    ts["actual_intervals"]   = actual
    ts["frame_drift"]        = jitter
    ts["cumulative_drift"]   = np.cumsum(jitter)

    # LED power — constant during recording
    ts["led_white_power_percent"] = np.full(N_FRAMES, 0.0,   dtype=np.float64)
    ts["led_ir_power_percent"]    = np.full(N_FRAMES, 100.0, dtype=np.float64)

    return {k: v.astype(np.float64) for k, v in ts.items()}


# ---------------------------------------------------------------------------
# Write Zarr store
# ---------------------------------------------------------------------------


def _open_store_for_write(out_path: str, store_type: str):
    """Open a zarr store for writing.  *store_type* is 'zip' or 'dir'."""
    import shutil
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    elif os.path.isfile(out_path):
        os.remove(out_path)

    if store_type == "zip":
        ZipStore = (
            getattr(zarr, "ZipStore", None)
            or getattr(getattr(zarr, "storage", None), "ZipStore", None)
        )
        if ZipStore is None:
            raise RuntimeError("Cannot find ZipStore in this zarr installation")
        try:
            backend = ZipStore(out_path, mode="w")
        except TypeError:
            backend = ZipStore(out_path)
        return zarr.open(backend, mode="w")
    else:
        # Directory store — zarr.open on a path creates it automatically
        return zarr.open(out_path, mode="w")


def create_zarr(out_path: str, store_type: str = "zip") -> None:
    rng = np.random.default_rng(RNG_SEED)

    label = "zip store" if store_type == "zip" else "directory store"
    print(f"Writing Zarr {label} -> {out_path}")
    print(f"  {N_FRAMES} frames  ({FRAME_H}x{FRAME_W})  "
          f"interval={FRAME_INTERVAL}s  duration={DURATION_H}h")

    store = _open_store_for_write(out_path, store_type)

    # /frames  — pre-allocate then fill in chunks to limit RAM usage
    # zarr v3 uses compressors= (list of codec instances); v2 uses compressor=
    codec_kwargs = {}
    if _Blosc is not None:
        codec_kwargs = {"compressors": [_Blosc]} if _ZARR_V3 else {"compressor": _Blosc}
    create_fn = getattr(store, "create_dataset", None) or getattr(store, "create_array", None)
    # Directory stores: use large time-axis chunks to avoid 50k+ chunk files.
    # Zip stores: chunk size less critical (single file), keep 1 for compat.
    time_chunk = WRITE_CHUNK if store_type == "dir" else 1
    frames_arr = create_fn(
        "frames",
        shape=(N_FRAMES, FRAME_H, FRAME_W),
        chunks=(time_chunk, FRAME_H, FRAME_W),
        dtype=DTYPE,
        **codec_kwargs,
    )

    n_chunks = (N_FRAMES + WRITE_CHUNK - 1) // WRITE_CHUNK
    for chunk_idx in range(n_chunks):
        start = chunk_idx * WRITE_CHUNK
        end   = min(start + WRITE_CHUNK, N_FRAMES)
        chunk_rng = np.random.default_rng(RNG_SEED + chunk_idx)
        frames_arr[start:end] = build_frames_chunk(start, end, chunk_rng)
        print(f"  frames {end}/{N_FRAMES} ({100*end/N_FRAMES:.0f}%)", end="\r", flush=True)
    print()

    # /timeseries/<name>
    print("Computing telemetry timeseries ...")
    ts_data = build_timeseries(rng)
    ts_group = _require_group(store, "timeseries")
    for name, arr in ts_data.items():
        _create_array(ts_group, name, arr, chunks=(N_FRAMES,), dtype=np.float64)

    # /metadata  — scalar attributes stored as a group with array datasets
    meta = _require_group(store, "metadata")
    meta.attrs["frame_interval"]   = FRAME_INTERVAL
    meta.attrs["total_frames"]     = N_FRAMES
    meta.attrs["experiment_name"]  = "test_6roi_zarr"
    meta.attrs["creation_date"]    = datetime.now().isoformat()
    meta.attrs["frame_height"]     = FRAME_H
    meta.attrs["frame_width"]      = FRAME_W
    meta.attrs["n_rois"]           = len(ROIS)
    meta.attrs["roi_centres"]      = [[cy, cx] for cy, cx, _ in ROIS]
    meta.attrs["roi_radii"]        = [r for _, _, r in ROIS]
    meta.attrs["roi_periods_h"]    = ROI_PERIODS_H
    meta.attrs["notes"] = (
        f"Synthetic test dataset: 6 circular ROIs, each oscillating with a "
        f"distinct circadian period (20-28 h). "
        f"Frame interval = {FRAME_INTERVAL}s, duration = {DURATION_H}h. "
        f"Created by tests/create_test_zarr.py."
    )

    # Top-level store attributes
    store.attrs["file_format"]    = "zarr"
    store.attrs["plugin_version"] = "napari-hdf5-activity"
    store.attrs["zarr_version"]   = zarr.__version__

    print("Done.")
    print(f"  /frames              shape={store['frames'].shape}  dtype={store['frames'].dtype}")
    print(f"  /timeseries keys:    {sorted(ts_group.keys())}")
    print(f"  /metadata attrs:     {sorted(meta.attrs.keys())}")
    _print_roi_summary()


def _print_roi_summary() -> None:
    print("\nROI summary:")
    header = f"  {'ROI':<6} {'Centre (y,x)':<16} {'Radius':>6}  {'Period (h)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, ((cy, cx, r), period) in enumerate(zip(ROIS, ROI_PERIODS_H), start=1):
        print(f"  ROI {i:<2} ({cy:>3},{cx:>3})         {r:>6}  {period:>10.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Zarr test datasets")
    parser.add_argument(
        "--store", choices=["zip", "dir", "both"], default="both",
        help="Store type to generate: zip store, directory store, or both (default: both)",
    )
    args = parser.parse_args()

    if args.store in ("zip", "both"):
        create_zarr(OUT_PATH_ZIP, store_type="zip")
    if args.store in ("dir", "both"):
        create_zarr(OUT_PATH_DIR, store_type="dir")
