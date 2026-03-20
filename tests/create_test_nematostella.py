"""create_test_nematostella.py — Realistic Nematostella activity simulation.

Generates a Zarr directory store where each of 6 ROIs shows realistic
activity bouts (contraction/movement) modulated by a circadian rhythm.
The inter-frame pixel differences within each ROI's mask will show clear
activity bouts that the analysis pipeline can detect and quantify.

Activity model
--------------
Each ROI is an independent Markov chain with two states:

  QUIESCENT → body pixels are stable (small frame-to-frame drift)
              → low inter-frame difference signal
  ACTIVE    → body pixels reshuffle each frame (animal moves/contracts)
              → high inter-frame difference signal

Transition rates are modulated by a circadian sinusoid so that each animal
has a preferred active phase. Bout and rest durations are exponentially
distributed.

Physical parameters are calibrated so that:
  - Quiescent RMS pixel change ≈ 70 ADU/pixel
  - Active    RMS pixel change ≈ 2100 ADU/pixel
  (typical for a 12-bit camera with 16-bit storage)

Run from the repository root::

    python tests/create_test_nematostella.py [--store zip|dir|both]
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import zarr

_ZARR_V3 = tuple(int(x) for x in zarr.__version__.split(".")[:2]) >= (3, 0)

# Codec — pick the right API for zarr v2 vs v3
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
        pass

# ---------------------------------------------------------------------------
# Recording configuration
# ---------------------------------------------------------------------------

FRAME_INTERVAL = 5.0          # seconds per frame
DURATION_H     = 72           # total duration (hours)
N_FRAMES       = int(DURATION_H * 3600 / FRAME_INTERVAL)   # 51 840
FRAME_H, FRAME_W = 256, 256
WRITE_CHUNK    = 500          # frames per write batch (limits RAM)
DTYPE          = np.uint16
RNG_SEED       = 7

# ---------------------------------------------------------------------------
# Image intensity model
# ---------------------------------------------------------------------------

BACKGROUND       = 3000   # ADU  — camera background (out of 65535)
BODY_INTENSITY   = 7000   # ADU  — additional brightness of animal body
CAMERA_NOISE_SD  =   50   # ADU  — Gaussian readout noise everywhere
BODY_NOISE_QUIET =   50   # ADU  — pixel-to-pixel noise during quiescence
BODY_NOISE_ACTIVE= 1500   # ADU  — pixel noise during activity (motion)

# ---------------------------------------------------------------------------
# ROI layout  (centre_y, centre_x, radius_px)
# ---------------------------------------------------------------------------

ROIS = [
    ( 50,  50, 18),   # ROI 1 — upper-left
    ( 50, 128, 18),   # ROI 2 — upper-centre
    ( 50, 206, 18),   # ROI 3 — upper-right
    (206,  50, 18),   # ROI 4 — lower-left
    (206, 128, 18),   # ROI 5 — lower-centre
    (206, 206, 18),   # ROI 6 — lower-right
]

# ---------------------------------------------------------------------------
# Per-ROI circadian + behavioural parameters
# ---------------------------------------------------------------------------

# Circadian period (hours) — small natural variation around 24 h
ROI_PERIODS_H = [24.0, 23.5, 24.8, 24.2, 23.8, 25.0]

# Phase of PEAK ACTIVITY relative to recording start (hours)
# Stagger the animals so they are not all active at the same time
ROI_ACTIVE_PHASE_H = [0.0, 6.0, 12.0, 18.0, 3.0, 9.0]

# Mean bout duration (minutes)   — exponentially distributed
ROI_BOUT_MIN  = [8,  6, 10,  7,  9,  5]

# Mean rest duration (minutes) — exponentially distributed
# Determines overall fraction of time spent active
ROI_REST_MIN  = [40, 30, 50, 35, 45, 25]

OUT_PATH_ZIP = os.path.join(os.path.dirname(__file__), "test_nematostella_6rois.zarr")
OUT_PATH_DIR = os.path.join(os.path.dirname(__file__), "test_nematostella_6rois_dir.zarr")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roi_mask(cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:FRAME_H, :FRAME_W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def simulate_activity_states(rng: np.random.Generator) -> np.ndarray:
    """Simulate Markov-chain activity states for all ROIs.

    Returns bool array of shape (N_FRAMES, N_ROIS).
    Transition rates are modulated by a circadian sinusoid.
    """
    n_rois = len(ROIS)
    states = np.zeros((N_FRAMES, n_rois), dtype=bool)
    t_h = np.arange(N_FRAMES) * FRAME_INTERVAL / 3600.0   # time in hours

    for i in range(n_rois):
        period  = ROI_PERIODS_H[i]
        phase   = ROI_ACTIVE_PHASE_H[i]
        bout_f  = ROI_BOUT_MIN[i]  * 60 / FRAME_INTERVAL   # mean bout in frames
        rest_f  = ROI_REST_MIN[i]  * 60 / FRAME_INTERVAL   # mean rest in frames

        # p_off is constant (geometric bout duration)
        p_off = 1.0 / bout_f

        # p_on modulated by circadian phase
        # circ oscillates 0–1; p_on_base set so overall mean ≈ f_active
        p_on_base = 1.0 / rest_f
        circ = 0.5 * (1.0 + np.sin(2 * np.pi * (t_h - phase) / period))
        # Range: 0.25× … 1.75× of base rate
        p_on = p_on_base * (0.25 + 1.5 * circ)

        state = False
        for t in range(N_FRAMES):
            if state:
                if rng.random() < p_off:
                    state = False
            else:
                if rng.random() < p_on[t]:
                    state = True
            states[t, i] = state

    return states


def build_frames_chunk(
    start: int,
    end: int,
    rng: np.random.Generator,
    activity_states: np.ndarray,
    roi_masks: list,
    roi_pixels_prev: list,     # current stable pixel values per ROI (modified in-place)
) -> np.ndarray:
    """Generate frames [start, end) as uint16.

    During QUIESCENCE each ROI's pixels drift slightly from the previous frame
    → low inter-frame difference.
    During ACTIVITY each ROI's pixels are redrawn independently each frame
    → large inter-frame difference (simulates movement / contraction).
    """
    n = end - start

    # Background + camera noise
    frames = np.full((n, FRAME_H, FRAME_W), BACKGROUND, dtype=np.float32)
    frames += rng.normal(0, CAMERA_NOISE_SD, size=frames.shape).astype(np.float32)

    for t in range(n):
        abs_t = start + t
        for roi_idx, (mask, (cy, cx, r)) in enumerate(zip(roi_masks, ROIS)):
            n_px = int(mask.sum())
            is_active = activity_states[abs_t, roi_idx]

            if is_active:
                # Animal moving: new random texture each frame
                new_vals = (
                    BACKGROUND + BODY_INTENSITY
                    + rng.normal(0, BODY_NOISE_ACTIVE, size=n_px)
                ).astype(np.float32)
            else:
                # Animal still: tiny drift from previous stable texture
                new_vals = np.clip(
                    roi_pixels_prev[roi_idx]
                    + rng.normal(0, BODY_NOISE_QUIET, size=n_px),
                    BACKGROUND,
                    65535,
                ).astype(np.float32)

            frames[t][mask] = new_vals
            roi_pixels_prev[roi_idx] = new_vals   # carry state to next frame

    return np.clip(frames, 0, 65535).astype(DTYPE)


def build_timeseries(
    rng: np.random.Generator,
    activity_states: np.ndarray,
) -> dict:
    """Build telemetry timeseries.

    frame_mean reflects the fraction of ROI pixels that are active
    (active pixels are brighter → slightly elevated mean).
    """
    ts: dict[str, np.ndarray] = {}
    t_h = np.arange(N_FRAMES) * FRAME_INTERVAL / 3600.0

    # Compute total ROI area and per-frame active pixel fraction
    roi_masks = [_roi_mask(cy, cx, r) for (cy, cx, r) in ROIS]
    roi_areas  = np.array([m.sum() for m in roi_masks], dtype=float)
    total_roi_pixels = roi_areas.sum()
    total_pixels = float(FRAME_H * FRAME_W)

    # Fraction of all pixels that are active ROI pixels at each frame
    active_fraction = (activity_states.astype(float) * roi_areas).sum(axis=1) / total_pixels

    base_mean = BACKGROUND + (total_roi_pixels / total_pixels) * BODY_INTENSITY
    ts["frame_mean"] = (
        base_mean + active_fraction * BODY_NOISE_ACTIVE * 0.5
        + rng.normal(0, 5, N_FRAMES)
    )
    ts["frame_min"] = np.full(N_FRAMES, float(BACKGROUND - 100)) + rng.normal(0, 2, N_FRAMES)
    ts["frame_max"] = (
        np.where(
            activity_states.any(axis=1),
            float(BACKGROUND + BODY_INTENSITY + 3 * BODY_NOISE_ACTIVE),
            float(BACKGROUND + BODY_INTENSITY + 3 * BODY_NOISE_QUIET),
        )
        + rng.normal(0, 10, N_FRAMES)
    )
    ts["frame_std"] = (
        80.0 + 50.0 * active_fraction + rng.normal(0, 3, N_FRAMES)
    )

    # Environment
    ts["temperature"] = (
        21.0 + 0.5 * np.sin(2 * np.pi * t_h / 24.0)
        + rng.normal(0, 0.1, N_FRAMES)
    )
    ts["humidity"] = (
        55.0 + 3.0 * np.sin(2 * np.pi * t_h / 24.0 + 1.0)
        + rng.normal(0, 0.5, N_FRAMES)
    )

    # Timing
    expected = np.full(N_FRAMES, FRAME_INTERVAL, dtype=np.float64)
    jitter   = rng.normal(0, 0.3, N_FRAMES)
    ts["expected_intervals"] = expected
    ts["actual_intervals"]   = expected + jitter
    ts["frame_drift"]        = jitter
    ts["cumulative_drift"]   = np.cumsum(jitter)

    # LED — IR illumination only (typical for bioluminescence / brightfield)
    ts["led_white_power_percent"] = np.full(N_FRAMES, 0.0,   dtype=np.float64)
    ts["led_ir_power_percent"]    = np.full(N_FRAMES, 100.0, dtype=np.float64)

    return {k: v.astype(np.float64) for k, v in ts.items()}


# ---------------------------------------------------------------------------
# Store helpers (zarr v2 / v3 compat)
# ---------------------------------------------------------------------------


def _open_store_for_write(out_path: str, store_type: str):
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
        return zarr.open(out_path, mode="w")


def _codec_kwargs() -> dict:
    if _Blosc is None:
        return {}
    return {"compressors": [_Blosc]} if _ZARR_V3 else {"compressor": _Blosc}


def _create_fn(group):
    return getattr(group, "create_dataset", None) or getattr(group, "create_array", None)


def _require_group(store, name: str):
    fn = getattr(store, "require_group", None)
    if fn is not None:
        return fn(name)
    return zarr.open_group(store=store, path=name, mode="a")


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------


def create_zarr(out_path: str, store_type: str = "dir") -> None:
    rng = np.random.default_rng(RNG_SEED)

    label = "zip store" if store_type == "zip" else "directory store"
    print(f"Writing Nematostella test dataset ({label}) -> {out_path}")
    print(f"  {N_FRAMES} frames  ({FRAME_H}×{FRAME_W})  "
          f"interval={FRAME_INTERVAL}s  duration={DURATION_H}h")

    # --- Simulate activity state sequence (trivial memory: 6 × 51840 bools) ---
    print("Simulating activity state sequences ...")
    activity_states = simulate_activity_states(rng)
    for i in range(len(ROIS)):
        frac = activity_states[:, i].mean()
        print(f"  ROI {i+1}: {frac*100:.1f}% active  "
              f"(period={ROI_PERIODS_H[i]}h, active-phase={ROI_ACTIVE_PHASE_H[i]}h)")

    # --- Open store ---
    store = _open_store_for_write(out_path, store_type)

    # --- /frames — pre-allocate, fill chunk by chunk ---
    ck = WRITE_CHUNK if store_type == "dir" else 1   # large chunks for dir store
    frames_arr = _create_fn(store)(
        "frames",
        shape=(N_FRAMES, FRAME_H, FRAME_W),
        chunks=(ck, FRAME_H, FRAME_W),
        dtype=DTYPE,
        **_codec_kwargs(),
    )

    roi_masks = [_roi_mask(cy, cx, r) for (cy, cx, r) in ROIS]

    # Initialise each ROI's pixel state to quiescent body intensity
    roi_pixels_prev = []
    for mask in roi_masks:
        n_px = int(mask.sum())
        init = (BACKGROUND + BODY_INTENSITY
                + rng.normal(0, BODY_NOISE_QUIET * 5, size=n_px)).astype(np.float32)
        roi_pixels_prev.append(np.clip(init, BACKGROUND, 65535))

    n_chunks = (N_FRAMES + WRITE_CHUNK - 1) // WRITE_CHUNK
    for ci in range(n_chunks):
        start = ci * WRITE_CHUNK
        end   = min(start + WRITE_CHUNK, N_FRAMES)
        chunk_rng = np.random.default_rng(RNG_SEED + ci + 1000)
        chunk = build_frames_chunk(
            start, end, chunk_rng, activity_states, roi_masks, roi_pixels_prev
        )
        frames_arr[start:end] = chunk
        print(f"  frames {end}/{N_FRAMES} ({100*end/N_FRAMES:.0f}%)", end="\r", flush=True)
    print()

    # --- /timeseries ---
    print("Computing telemetry timeseries ...")
    ts_data = build_timeseries(rng, activity_states)
    ts_group = _require_group(store, "timeseries")
    for name, arr in ts_data.items():
        _create_fn(ts_group)(
            name,
            shape=arr.shape,
            data=arr,
            chunks=(N_FRAMES,),
            dtype=np.float64,
        )

    # --- /metadata ---
    meta = _require_group(store, "metadata")
    meta.attrs["frame_interval"]   = FRAME_INTERVAL
    meta.attrs["total_frames"]     = N_FRAMES
    meta.attrs["experiment_name"]  = "test_nematostella_6rois"
    meta.attrs["creation_date"]    = datetime.now().isoformat()
    meta.attrs["frame_height"]     = FRAME_H
    meta.attrs["frame_width"]      = FRAME_W
    meta.attrs["n_rois"]           = len(ROIS)
    meta.attrs["roi_centres"]      = [[cy, cx] for cy, cx, _ in ROIS]
    meta.attrs["roi_radii"]        = [r for _, _, r in ROIS]
    meta.attrs["roi_periods_h"]    = ROI_PERIODS_H
    meta.attrs["roi_active_phase_h"] = ROI_ACTIVE_PHASE_H
    meta.attrs["roi_bout_min"]     = ROI_BOUT_MIN
    meta.attrs["roi_rest_min"]     = ROI_REST_MIN
    meta.attrs["notes"] = (
        "Synthetic Nematostella activity dataset: 6 ROIs with realistic "
        "activity bouts (Markov chain, exponential bout/rest durations) "
        "modulated by individual circadian rhythms. "
        f"Frame interval={FRAME_INTERVAL}s, duration={DURATION_H}h. "
        "Created by tests/create_test_nematostella.py."
    )

    store.attrs["file_format"]    = "zarr"
    store.attrs["plugin_version"] = "napari-hdf5-activity"
    store.attrs["zarr_version"]   = zarr.__version__

    # --- Summary ---
    print("Done.")
    print(f"  /frames        shape={store['frames'].shape}  dtype={store['frames'].dtype}")
    print(f"  /timeseries    keys={sorted(ts_group.keys())}")
    _print_summary(activity_states)


def _print_summary(activity_states: np.ndarray) -> None:
    print("\nROI summary:")
    header = (f"  {'ROI':<5} {'Centre':>10} {'Rad':>4}  "
              f"{'Period(h)':>9}  {'Phase(h)':>8}  "
              f"{'Bout(min)':>9}  {'Rest(min)':>9}  {'%Active':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, ((cy, cx, r), period, phase, bout, rest) in enumerate(
        zip(ROIS, ROI_PERIODS_H, ROI_ACTIVE_PHASE_H,
            ROI_BOUT_MIN, ROI_REST_MIN), start=1
    ):
        frac = activity_states[:, i - 1].mean() * 100
        print(f"  ROI {i:<2}  ({cy:>3},{cx:>3})  {r:>4}  "
              f"{period:>9.1f}  {phase:>8.1f}  "
              f"{bout:>9}  {rest:>9}  {frac:>6.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic Nematostella activity test datasets"
    )
    parser.add_argument(
        "--store", choices=["zip", "dir", "both"], default="dir",
        help="Store type: zip store, directory store, or both (default: dir)",
    )
    args = parser.parse_args()

    if args.store in ("dir", "both"):
        create_zarr(OUT_PATH_DIR, store_type="dir")
    if args.store in ("zip", "both"):
        create_zarr(OUT_PATH_ZIP, store_type="zip")
