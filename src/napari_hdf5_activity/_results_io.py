"""
_results_io.py - Comprehensive HDF5 Results Save/Load System

This module provides functions to save and load ALL analysis results:
- Core analysis results (movement data, thresholds, quiescence bouts)
- Extended analysis results (Fisher, FFT, Cosinor, Similarity, Coherence, Phase)
- Analysis parameters
- Metadata

Allows for post-hoc analysis of different cycles and periods without re-processing.
"""

import h5py
import numpy as np
from typing import Dict, Any, Optional
import json


def save_comprehensive_results(
    file_path: str,
    core_results: Dict[str, Any],
    extended_results: Optional[Dict[str, Any]] = None,
    analysis_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Save ALL analysis results to HDF5 file.

    Args:
        file_path: Output HDF5 file path
        core_results: Dictionary with core analysis results
            {
                'merged_results': {roi_id: [(time, value), ...]},
                'fraction_data': {roi_id: [(time, fraction), ...]},
                'roi_summary': {roi_id: {'mean': ..., 'std': ..., ...}},
                'thresholds': {roi_id: {'upper': ..., 'lower': ..., ...}},
                'quiescence_bouts': {roi_id: [{'start': ..., 'end': ..., ...}, ...]},
            }
        extended_results: Dictionary with extended analysis results
            {
                'fisher': {roi_id: {'periods': [...], 'z_scores': [...], 'p_values': [...], ...}},
                'fft': {roi_id: {'frequencies': [...], 'power': [...], 'dominant_period': ...}},
                'cosinor': {roi_id: {period: {'mesor': ..., 'amplitude': ..., 'acrophase': ..., ...}}},
                'similarity': {'matrix': [[...]], 'roi_ids': [...], 'lag_matrix': [[...]]},
                'coherence': {(roi_i, roi_j): {'frequencies': [...], 'coherence': [...]}},
                'phase': {roi_id: {'phase': ..., 'amplitude': ..., 'period': ...}},
            }
        analysis_params: Analysis parameters used
        metadata: Additional metadata

    Returns:
        bool: True if successful
    """
    try:
        with h5py.File(file_path, "w") as f:
            # ================================================================
            # 1. CORE ANALYSIS RESULTS
            # ================================================================
            core_group = f.create_group("core_analysis")

            # Save movement data per ROI
            if "merged_results" in core_results:
                movement_group = core_group.create_group("movement_data")
                for roi_id, data in core_results["merged_results"].items():
                    roi_group = movement_group.create_group(f"roi_{roi_id}")

                    # Convert to structured array
                    dtype = np.dtype([("time", "f8"), ("value", "f8")])
                    array = np.array(data, dtype=dtype)
                    roi_group.create_dataset("timeseries", data=array)

                    # Metadata
                    roi_group.attrs["roi_id"] = roi_id
                    roi_group.attrs["n_samples"] = len(data)
                    if data:
                        times = [t for t, _ in data]
                        roi_group.attrs["duration_seconds"] = max(times) - min(times)

            # Save fraction data per ROI
            if "fraction_data" in core_results:
                fraction_group = core_group.create_group("fraction_data")
                for roi_id, data in core_results["fraction_data"].items():
                    roi_group = fraction_group.create_group(f"roi_{roi_id}")

                    dtype = np.dtype([("time", "f8"), ("fraction", "f8")])
                    array = np.array(data, dtype=dtype)
                    roi_group.create_dataset("timeseries", data=array)

            # Save binary movement data per ROI (1=moving, 0=not)
            if "movement_data_binary" in core_results:
                mov_bin_group = core_group.create_group("movement_data_binary")
                for roi_id, data in core_results["movement_data_binary"].items():
                    roi_group = mov_bin_group.create_group(f"roi_{roi_id}")
                    dtype = np.dtype([("time", "f8"), ("state", "i4")])
                    array = np.array([(t, int(v)) for t, v in data], dtype=dtype)
                    roi_group.create_dataset("timeseries", data=array)

            # Save quiescence data per ROI (1=quiescent, 0=active)
            if "quiescence_data" in core_results:
                quiescence_group = core_group.create_group("quiescence_data")
                for roi_id, data in core_results["quiescence_data"].items():
                    roi_group = quiescence_group.create_group(f"roi_{roi_id}")
                    dtype = np.dtype([("time", "f8"), ("state", "i4")])
                    array = np.array([(t, int(v)) for t, v in data], dtype=dtype)
                    roi_group.create_dataset("timeseries", data=array)

            # Save sleep data per ROI (1=sleeping, 0=awake)
            if "sleep_data" in core_results:
                sleep_grp = core_group.create_group("sleep_data")
                for roi_id, data in core_results["sleep_data"].items():
                    roi_group = sleep_grp.create_group(f"roi_{roi_id}")
                    dtype = np.dtype([("time", "f8"), ("state", "i4")])
                    array = np.array([(t, int(v)) for t, v in data], dtype=dtype)
                    roi_group.create_dataset("timeseries", data=array)

            # Save ROI colors ({roi_id: color_string_or_tuple})
            if "roi_colors" in core_results:
                colors_group = core_group.create_group("roi_colors")
                for roi_id, color in core_results["roi_colors"].items():
                    if isinstance(color, (list, tuple)):
                        colors_group.attrs[f"roi_{roi_id}"] = json.dumps(list(color))
                    else:
                        colors_group.attrs[f"roi_{roi_id}"] = str(color)

            # Save ROI statistics (threshold analysis results)
            if "roi_statistics" in core_results:
                roi_stats_group = core_group.create_group("roi_statistics")
                for roi_id, stats in core_results["roi_statistics"].items():
                    roi_group = roi_stats_group.create_group(f"roi_{roi_id}")
                    for key, value in stats.items():
                        if isinstance(value, (int, float, str, bool,
                                              np.integer, np.floating)):
                            roi_group.attrs[key] = value
                        elif isinstance(value, (tuple, list)):
                            # e.g. data_range = (min, max)
                            roi_group.create_dataset(
                                key, data=np.array(value, dtype="f8")
                            )

            # Save ROI summary statistics
            if "roi_summary" in core_results:
                summary_group = core_group.create_group("roi_summary")
                for roi_id, stats in core_results["roi_summary"].items():
                    roi_group = summary_group.create_group(f"roi_{roi_id}")
                    for key, value in stats.items():
                        roi_group.attrs[key] = value

            # Save thresholds
            if "thresholds" in core_results:
                threshold_group = core_group.create_group("thresholds")
                for roi_id, thresh in core_results["thresholds"].items():
                    roi_group = threshold_group.create_group(f"roi_{roi_id}")
                    for key, value in thresh.items():
                        roi_group.attrs[key] = value

            # Save quiescence bouts
            if "quiescence_bouts" in core_results:
                bouts_group = core_group.create_group("quiescence_bouts")
                for roi_id, bouts in core_results["quiescence_bouts"].items():
                    roi_group = bouts_group.create_group(f"roi_{roi_id}")

                    # Convert to structured array
                    if bouts:
                        dtype = np.dtype(
                            [
                                ("start_time", "f8"),
                                ("end_time", "f8"),
                                ("duration", "f8"),
                                ("mean_movement", "f8"),
                            ]
                        )
                        bout_array = np.array(
                            [
                                (
                                    b["start_time"],
                                    b["end_time"],
                                    b["duration"],
                                    b["mean_movement"],
                                )
                                for b in bouts
                            ],
                            dtype=dtype,
                        )
                        roi_group.create_dataset("bouts", data=bout_array)
                        roi_group.attrs["n_bouts"] = len(bouts)

            # Save LED / lighting data (times, white_powers, ir_powers)
            if "led_data" in core_results and core_results["led_data"]:
                led = core_results["led_data"]
                led_group = core_group.create_group("led_data")
                for key in ("times", "white_powers", "ir_powers"):
                    if key in led and led[key] is not None:
                        led_group.create_dataset(key, data=np.array(led[key], dtype="f8"))

            # Save ROI masks — binary arrays (uint8, one per ROI)
            try:
                if "masks" in core_results and core_results["masks"]:
                    masks_group = core_group.create_group("roi_masks")
                    saved = 0
                    for i, mask in enumerate(core_results["masks"]):
                        arr = np.asarray(mask, dtype=np.uint8)
                        if arr.ndim == 2 and arr.size > 0:
                            masks_group.create_dataset(
                                f"mask_{i}", data=arr,
                                compression="gzip", compression_opts=4,
                            )
                            saved += 1
                    masks_group.attrs["n_masks"] = saved
            except Exception as _me:
                print(f"Warning: could not save ROI masks: {_me}")

            # Save detected circle parameters [x, y, radius] (uint16, shape N×3)
            try:
                circles = core_results.get("original_circles")
                if circles is not None:
                    arr = np.asarray(circles)
                    if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                        core_group.create_dataset(
                            "roi_circles",
                            data=arr.astype(np.uint16),
                        )
            except Exception as _ce:
                print(f"Warning: could not save ROI circles: {_ce}")

            # ================================================================
            # 2. EXTENDED ANALYSIS RESULTS
            # ================================================================
            if extended_results:
                extended_group = f.create_group("extended_analysis")

                # Fisher Z-Test results
                if "fisher" in extended_results:
                    fisher_group = extended_group.create_group("fisher_z_test")
                    for roi_id, results in extended_results["fisher"].items():
                        roi_group = fisher_group.create_group(f"roi_{roi_id}")

                        # Save arrays
                        if "periods" in results:
                            roi_group.create_dataset(
                                "periods", data=np.array(results["periods"])
                            )
                        if "z_scores" in results:
                            roi_group.create_dataset(
                                "z_scores", data=np.array(results["z_scores"])
                            )
                        if "p_values" in results:
                            roi_group.create_dataset(
                                "p_values", data=np.array(results["p_values"])
                            )

                        # Save scalar results
                        for key in [
                            "dominant_period",
                            "dominant_z_score",
                            "dominant_p_value",
                        ]:
                            if key in results:
                                roi_group.attrs[key] = results[key]

                # FFT Power Spectrum results
                if "fft" in extended_results:
                    fft_group = extended_group.create_group("fft_spectrum")
                    for roi_id, results in extended_results["fft"].items():
                        roi_group = fft_group.create_group(f"roi_{roi_id}")

                        if "frequencies" in results:
                            roi_group.create_dataset(
                                "frequencies", data=np.array(results["frequencies"])
                            )
                        if "power" in results:
                            roi_group.create_dataset(
                                "power", data=np.array(results["power"])
                            )
                        if "periods" in results:
                            roi_group.create_dataset(
                                "periods", data=np.array(results["periods"])
                            )

                        if "dominant_period" in results:
                            roi_group.attrs["dominant_period"] = results[
                                "dominant_period"
                            ]
                        if "dominant_power" in results:
                            roi_group.attrs["dominant_power"] = results[
                                "dominant_power"
                            ]

                # Cosinor Analysis results
                if "cosinor" in extended_results:
                    cosinor_group = extended_group.create_group("cosinor")
                    for roi_id, period_results in extended_results["cosinor"].items():
                        roi_group = cosinor_group.create_group(f"roi_{roi_id}")

                        # Each period has its own subgroup
                        for period, results in period_results.items():
                            period_group = roi_group.create_group(f"period_{period}h")

                            for key in [
                                "mesor",
                                "amplitude",
                                "acrophase",
                                "r_squared",
                                "p_value",
                            ]:
                                if key in results:
                                    period_group.attrs[key] = results[key]

                            # Save confidence intervals if available
                            for key in ["mesor_ci", "amplitude_ci", "acrophase_ci"]:
                                if key in results:
                                    period_group.create_dataset(
                                        key, data=np.array(results[key])
                                    )

                # ROI Similarity Matrix
                if "similarity" in extended_results:
                    similarity_group = extended_group.create_group("roi_similarity")

                    if "matrix" in extended_results["similarity"]:
                        similarity_group.create_dataset(
                            "correlation_matrix",
                            data=np.array(extended_results["similarity"]["matrix"]),
                        )
                    if "lag_matrix" in extended_results["similarity"]:
                        similarity_group.create_dataset(
                            "lag_matrix",
                            data=np.array(extended_results["similarity"]["lag_matrix"]),
                        )
                    if "roi_ids" in extended_results["similarity"]:
                        similarity_group.create_dataset(
                            "roi_ids",
                            data=np.array(extended_results["similarity"]["roi_ids"]),
                        )

                # Coherence Analysis
                if "coherence" in extended_results:
                    coherence_group = extended_group.create_group("coherence")

                    for (roi_i, roi_j), results in extended_results[
                        "coherence"
                    ].items():
                        pair_group = coherence_group.create_group(
                            f"roi_{roi_i}_vs_roi_{roi_j}"
                        )

                        if "frequencies" in results:
                            pair_group.create_dataset(
                                "frequencies", data=np.array(results["frequencies"])
                            )
                        if "coherence" in results:
                            pair_group.create_dataset(
                                "coherence", data=np.array(results["coherence"])
                            )

                # Phase Clustering
                if "phase" in extended_results:
                    phase_group = extended_group.create_group("phase_clustering")

                    for roi_id, results in extended_results["phase"].items():
                        roi_group = phase_group.create_group(f"roi_{roi_id}")

                        for key in ["phase", "amplitude", "period"]:
                            if key in results:
                                roi_group.attrs[key] = results[key]

                # Sleep Phase Results (from sleep data analysis)
                if "sleep_phase_results" in extended_results:
                    sleep_group = extended_group.create_group("sleep_phase_analysis")
                    sleep_results = extended_results["sleep_phase_results"]

                    # Handle different structures (Fisher vs Cosinor)
                    if isinstance(sleep_results, dict):
                        # Check if it has roi_results (Cosinor structure)
                        if "roi_results" in sleep_results:
                            for roi_id, roi_data in sleep_results[
                                "roi_results"
                            ].items():
                                if isinstance(roi_id, int):
                                    roi_group = sleep_group.create_group(
                                        f"roi_{roi_id}"
                                    )
                                    best = roi_data.get("best_result", {})
                                    roi_group.attrs["best_period"] = roi_data.get(
                                        "best_period", 0
                                    )
                                    roi_group.attrs["sleep_phase"] = best.get(
                                        "acrophase", 0
                                    )
                                    roi_group.attrs["mesor"] = best.get("mesor", 0)
                                    roi_group.attrs["amplitude"] = best.get(
                                        "amplitude", 0
                                    )
                                    roi_group.attrs["significant"] = best.get(
                                        "significant", False
                                    )
                        else:
                            # Fisher/FFT structure - ROI IDs as direct keys
                            for roi_id, roi_data in sleep_results.items():
                                if isinstance(roi_id, int):
                                    roi_group = sleep_group.create_group(
                                        f"roi_{roi_id}"
                                    )
                                    periodogram = roi_data.get("periodogram", {})
                                    roi_group.attrs["dominant_period"] = (
                                        periodogram.get("dominant_period", 0)
                                    )
                                    roi_group.attrs["z_score"] = periodogram.get(
                                        "dominant_z_score", 0
                                    )
                                    roi_group.attrs["is_significant"] = periodogram.get(
                                        "is_significant", False
                                    )

                # Save method info
                if "method_index" in extended_results:
                    extended_group.attrs["method_index"] = extended_results[
                        "method_index"
                    ]
                if "method_name" in extended_results:
                    extended_group.attrs["method_name"] = extended_results[
                        "method_name"
                    ]

                # Save full results as JSON blob for lossless round-trip
                if "full_results_json" in extended_results:
                    json_bytes = np.frombuffer(
                        extended_results["full_results_json"].encode("utf-8"),
                        dtype=np.uint8,
                    )
                    extended_group.create_dataset(
                        "full_results_json", data=json_bytes, compression="gzip"
                    )

            # ================================================================
            # 3. ANALYSIS PARAMETERS
            # ================================================================
            if analysis_params:
                params_group = f.create_group("analysis_parameters")

                # Core analysis parameters
                if "core" in analysis_params:
                    core_params = params_group.create_group("core")
                    for key, value in analysis_params["core"].items():
                        core_params.attrs[key] = value

                # Extended analysis parameters
                if "extended" in analysis_params:
                    extended_params = params_group.create_group("extended")
                    for key, value in analysis_params["extended"].items():
                        extended_params.attrs[key] = value

            # ================================================================
            # 4. METADATA
            # ================================================================
            if metadata:
                meta_group = f.create_group("metadata")
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, dict):
                        # Save as JSON string
                        meta_group.attrs[key] = json.dumps(value)

            # General attributes
            f.attrs["saved_from"] = "napari-hdf5-activity comprehensive results"
            f.attrs["version"] = "2.0"
            f.attrs["save_timestamp"] = str(np.datetime64("now"))

        return True

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error saving comprehensive results: {e}\n{tb}")
        raise RuntimeError(f"{e}\n\nFull traceback:\n{tb}") from e


def load_comprehensive_results(file_path: str) -> Dict[str, Any]:
    """
    Load ALL analysis results from HDF5 file.

    Supports both old format (v1.x - with 'results' group) and
    new format (v2.x - with 'core_analysis' group).

    Args:
        file_path: Input HDF5 file path

    Returns:
        Dictionary with all results:
        {
            'core_analysis': {...},
            'extended_analysis': {...},
            'analysis_parameters': {...},
            'metadata': {...},
        }
    """
    results = {}

    try:
        with h5py.File(file_path, "r") as f:
            # ================================================================
            # BACKWARD COMPATIBILITY: Detect old format (v1.x)
            # ================================================================
            if "results" in f and "core_analysis" not in f:
                # OLD FORMAT: Load from 'results' group
                results["core_analysis"] = {}

                results_group = f["results"]
                movement_data = {}
                fraction_data = {}

                for roi_key in results_group.keys():
                    if roi_key.startswith("roi_"):
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = results_group[roi_key]

                        # Load processed_data as movement_data
                        if "processed_data" in roi_group:
                            array = roi_group["processed_data"][:]
                            movement_data[roi_id] = [
                                (float(t), float(v)) for t, v in array
                            ]

                        # Load fraction_data
                        if "fraction_data" in roi_group:
                            array = roi_group["fraction_data"][:]
                            fraction_data[roi_id] = [
                                (float(t), float(v)) for t, v in array
                            ]

                results["core_analysis"]["movement_data"] = movement_data
                results["core_analysis"]["fraction_data"] = fraction_data

                # Load old metadata and parameters
                if "metadata" in f:
                    results["metadata"] = dict(f["metadata"].attrs)

                if "analysis_parameters" in f:
                    params = dict(f["analysis_parameters"].attrs)
                    results["analysis_parameters"] = {
                        "core": {},
                        "extended": {
                            "min_period_hours": params.get("min_period_hours"),
                            "max_period_hours": params.get("max_period_hours"),
                            "significance_level": params.get("significance_level"),
                            "analysis_method": params.get("analysis_method"),
                        },
                    }

                # No extended analysis results in old format
                results["extended_analysis"] = None

                print(
                    "Loaded HDF5 file in OLD format (v1.x) - backward compatibility mode"
                )
                return results

            # ================================================================
            # 1. LOAD CORE ANALYSIS RESULTS (NEW FORMAT v2.x)
            # ================================================================
            if "core_analysis" in f:
                core_group = f["core_analysis"]
                results["core_analysis"] = {}

                # Load movement data
                if "movement_data" in core_group:
                    movement_data = {}
                    for roi_key in core_group["movement_data"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["movement_data"][roi_key]

                        if "timeseries" in roi_group:
                            array = roi_group["timeseries"][:]
                            movement_data[roi_id] = [
                                (float(t), float(v)) for t, v in array
                            ]

                    results["core_analysis"]["merged_results"] = movement_data

                # Load fraction data
                if "fraction_data" in core_group:
                    fraction_data = {}
                    for roi_key in core_group["fraction_data"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["fraction_data"][roi_key]

                        if "timeseries" in roi_group:
                            array = roi_group["timeseries"][:]
                            fraction_data[roi_id] = [
                                (float(t), float(v)) for t, v in array
                            ]

                    results["core_analysis"]["fraction_data"] = fraction_data

                # Load binary movement data
                if "movement_data_binary" in core_group:
                    mov_bin = {}
                    for roi_key in core_group["movement_data_binary"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["movement_data_binary"][roi_key]
                        if "timeseries" in roi_group:
                            array = roi_group["timeseries"][:]
                            mov_bin[roi_id] = [(float(t), int(v)) for t, v in array]
                    results["core_analysis"]["movement_data"] = mov_bin

                # Load quiescence data
                if "quiescence_data" in core_group:
                    quiescence_data = {}
                    for roi_key in core_group["quiescence_data"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["quiescence_data"][roi_key]
                        if "timeseries" in roi_group:
                            array = roi_group["timeseries"][:]
                            quiescence_data[roi_id] = [
                                (float(t), int(v)) for t, v in array
                            ]
                    results["core_analysis"]["quiescence_data"] = quiescence_data

                # Load sleep data
                if "sleep_data" in core_group:
                    sleep_data = {}
                    for roi_key in core_group["sleep_data"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["sleep_data"][roi_key]
                        if "timeseries" in roi_group:
                            array = roi_group["timeseries"][:]
                            sleep_data[roi_id] = [
                                (float(t), int(v)) for t, v in array
                            ]
                    results["core_analysis"]["sleep_data"] = sleep_data

                # Load ROI colors
                if "roi_colors" in core_group:
                    roi_colors = {}
                    for key, value in core_group["roi_colors"].attrs.items():
                        roi_id = int(key.split("_")[1])
                        try:
                            roi_colors[roi_id] = json.loads(value)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            roi_colors[roi_id] = value
                    results["core_analysis"]["roi_colors"] = roi_colors

                # Load ROI statistics
                if "roi_statistics" in core_group:
                    roi_statistics = {}
                    for roi_key in core_group["roi_statistics"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["roi_statistics"][roi_key]
                        stats = dict(roi_group.attrs)
                        # Load tuple/array datasets (e.g. data_range)
                        for ds_key in roi_group.keys():
                            stats[ds_key] = tuple(
                                roi_group[ds_key][:].tolist()
                            )
                        roi_statistics[roi_id] = stats
                    results["core_analysis"]["roi_statistics"] = roi_statistics

                # Load ROI summary
                if "roi_summary" in core_group:
                    roi_summary = {}
                    for roi_key in core_group["roi_summary"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["roi_summary"][roi_key]

                        roi_summary[roi_id] = dict(roi_group.attrs)

                    results["core_analysis"]["roi_summary"] = roi_summary

                # Load thresholds
                if "thresholds" in core_group:
                    thresholds = {}
                    for roi_key in core_group["thresholds"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["thresholds"][roi_key]

                        thresholds[roi_id] = dict(roi_group.attrs)

                    results["core_analysis"]["thresholds"] = thresholds

                # Load quiescence bouts
                if "quiescence_bouts" in core_group:
                    bouts_data = {}
                    for roi_key in core_group["quiescence_bouts"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = core_group["quiescence_bouts"][roi_key]

                        if "bouts" in roi_group:
                            bout_array = roi_group["bouts"][:]
                            bouts_data[roi_id] = [
                                {
                                    "start_time": float(b["start_time"]),
                                    "end_time": float(b["end_time"]),
                                    "duration": float(b["duration"]),
                                    "mean_movement": float(b["mean_movement"]),
                                }
                                for b in bout_array
                            ]

                    results["core_analysis"]["quiescence_bouts"] = bouts_data

                # Load LED / lighting data
                if "led_data" in core_group:
                    led_group = core_group["led_data"]
                    led_data = {}
                    for key in ("times", "white_powers", "ir_powers"):
                        if key in led_group:
                            led_data[key] = led_group[key][:].tolist()
                    results["core_analysis"]["led_data"] = led_data

                # Load ROI masks
                if "roi_masks" in core_group:
                    masks_group = core_group["roi_masks"]
                    n = int(masks_group.attrs.get("n_masks", 0))
                    masks = [masks_group[f"mask_{i}"][:].astype(bool) for i in range(n)]
                    results["core_analysis"]["masks"] = masks

                # Load detected circle parameters
                if "roi_circles" in core_group:
                    results["core_analysis"]["original_circles"] = core_group["roi_circles"][:].copy()

            # ================================================================
            # 2. LOAD EXTENDED ANALYSIS RESULTS
            # ================================================================
            if "extended_analysis" in f:
                extended_group = f["extended_analysis"]
                results["extended_analysis"] = {}

                # Load Fisher Z-Test results
                if "fisher_z_test" in extended_group:
                    fisher_data = {}
                    for roi_key in extended_group["fisher_z_test"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = extended_group["fisher_z_test"][roi_key]

                        fisher_data[roi_id] = {}

                        if "periods" in roi_group:
                            fisher_data[roi_id]["periods"] = roi_group["periods"][
                                :
                            ].tolist()
                        if "z_scores" in roi_group:
                            fisher_data[roi_id]["z_scores"] = roi_group["z_scores"][
                                :
                            ].tolist()
                        if "p_values" in roi_group:
                            fisher_data[roi_id]["p_values"] = roi_group["p_values"][
                                :
                            ].tolist()

                        # Load attributes
                        fisher_data[roi_id].update(dict(roi_group.attrs))

                    results["extended_analysis"]["fisher"] = fisher_data

                # Load FFT results
                if "fft_spectrum" in extended_group:
                    fft_data = {}
                    for roi_key in extended_group["fft_spectrum"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = extended_group["fft_spectrum"][roi_key]

                        fft_data[roi_id] = {}

                        if "frequencies" in roi_group:
                            fft_data[roi_id]["frequencies"] = roi_group["frequencies"][
                                :
                            ].tolist()
                        if "power" in roi_group:
                            fft_data[roi_id]["power"] = roi_group["power"][:].tolist()
                        if "periods" in roi_group:
                            fft_data[roi_id]["periods"] = roi_group["periods"][
                                :
                            ].tolist()

                        fft_data[roi_id].update(dict(roi_group.attrs))

                    results["extended_analysis"]["fft"] = fft_data

                # Load Cosinor results
                if "cosinor" in extended_group:
                    cosinor_data = {}
                    for roi_key in extended_group["cosinor"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = extended_group["cosinor"][roi_key]

                        cosinor_data[roi_id] = {}

                        # Each period is a subgroup
                        for period_key in roi_group.keys():
                            period = float(
                                period_key.replace("period_", "").replace("h", "")
                            )
                            period_group = roi_group[period_key]

                            cosinor_data[roi_id][period] = dict(period_group.attrs)

                            # Load confidence intervals
                            for key in ["mesor_ci", "amplitude_ci", "acrophase_ci"]:
                                if key in period_group:
                                    cosinor_data[roi_id][period][key] = period_group[
                                        key
                                    ][:].tolist()

                    results["extended_analysis"]["cosinor"] = cosinor_data

                # Load Similarity Matrix
                if "roi_similarity" in extended_group:
                    similarity_data = {}

                    if "correlation_matrix" in extended_group["roi_similarity"]:
                        similarity_data["matrix"] = extended_group["roi_similarity"][
                            "correlation_matrix"
                        ][:].tolist()
                    if "lag_matrix" in extended_group["roi_similarity"]:
                        similarity_data["lag_matrix"] = extended_group[
                            "roi_similarity"
                        ]["lag_matrix"][:].tolist()
                    if "roi_ids" in extended_group["roi_similarity"]:
                        similarity_data["roi_ids"] = extended_group["roi_similarity"][
                            "roi_ids"
                        ][:].tolist()

                    results["extended_analysis"]["similarity"] = similarity_data

                # Load Coherence results
                if "coherence" in extended_group:
                    coherence_data = {}
                    for pair_key in extended_group["coherence"].keys():
                        # Parse "roi_1_vs_roi_2"
                        parts = pair_key.split("_vs_")
                        roi_i = int(parts[0].replace("roi_", ""))
                        roi_j = int(parts[1].replace("roi_", ""))

                        pair_group = extended_group["coherence"][pair_key]
                        coherence_data[(roi_i, roi_j)] = {}

                        if "frequencies" in pair_group:
                            coherence_data[(roi_i, roi_j)]["frequencies"] = pair_group[
                                "frequencies"
                            ][:].tolist()
                        if "coherence" in pair_group:
                            coherence_data[(roi_i, roi_j)]["coherence"] = pair_group[
                                "coherence"
                            ][:].tolist()

                    results["extended_analysis"]["coherence"] = coherence_data

                # Load Phase Clustering
                if "phase_clustering" in extended_group:
                    phase_data = {}
                    for roi_key in extended_group["phase_clustering"].keys():
                        roi_id = int(roi_key.split("_")[1])
                        roi_group = extended_group["phase_clustering"][roi_key]

                        phase_data[roi_id] = dict(roi_group.attrs)

                    results["extended_analysis"]["phase"] = phase_data

                # Load method index/name from group attributes
                results["extended_analysis"]["method_index"] = int(
                    extended_group.attrs.get("method_index", 0)
                )
                results["extended_analysis"]["method_name"] = str(
                    extended_group.attrs.get("method_name", "Unknown")
                )

                # Load full results JSON blob (lossless round-trip, new saves only)
                if "full_results_json" in extended_group:
                    try:
                        import json as _json

                        json_bytes = extended_group["full_results_json"][:]
                        json_str = bytes(json_bytes.tolist()).decode("utf-8")
                        raw = _json.loads(json_str)

                        # JSON converts integer dict keys to strings — restore them
                        def _restore_int_keys(obj):
                            if not isinstance(obj, dict):
                                return obj
                            out = {}
                            for k, v in obj.items():
                                try:
                                    k = int(k)
                                except (ValueError, TypeError):
                                    pass
                                out[k] = _restore_int_keys(v)
                            return out

                        results["extended_analysis"]["results"] = _restore_int_keys(raw)
                    except Exception as _e:
                        print(f"Could not load full_results_json: {_e}")

            # ================================================================
            # 3. LOAD ANALYSIS PARAMETERS
            # ================================================================
            if "analysis_parameters" in f:
                params_group = f["analysis_parameters"]
                results["analysis_parameters"] = {}

                if "core" in params_group:
                    results["analysis_parameters"]["core"] = dict(
                        params_group["core"].attrs
                    )

                if "extended" in params_group:
                    results["analysis_parameters"]["extended"] = dict(
                        params_group["extended"].attrs
                    )

            # ================================================================
            # 4. LOAD METADATA
            # ================================================================
            if "metadata" in f:
                meta_group = f["metadata"]
                results["metadata"] = {}

                for key, value in meta_group.attrs.items():
                    # Try to parse JSON strings
                    if isinstance(value, str) and value.startswith("{"):
                        try:
                            results["metadata"][key] = json.loads(value)
                        except:
                            results["metadata"][key] = value
                    else:
                        results["metadata"][key] = value

            # Load file-level attributes
            results["file_info"] = dict(f.attrs)

        return results

    except Exception as e:
        print(f"Error loading comprehensive results: {e}")
        import traceback

        traceback.print_exc()
        return {}


def extract_subset_by_time_range(
    results: Dict[str, Any], start_time: float, end_time: float
) -> Dict[str, Any]:
    """
    Extract a subset of results for a specific time range (cycle analysis).

    Args:
        results: Full results dictionary from load_comprehensive_results()
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Subset of results containing only data within the time range
    """
    subset = {"core_analysis": {}, "time_range": {"start": start_time, "end": end_time}}

    if "core_analysis" not in results:
        return subset

    core = results["core_analysis"]

    # Filter movement data
    if "merged_results" in core:
        filtered_movement = {}
        for roi_id, data in core["merged_results"].items():
            filtered = [(t, v) for t, v in data if start_time <= t <= end_time]
            if filtered:
                filtered_movement[roi_id] = filtered
        subset["core_analysis"]["merged_results"] = filtered_movement

    # Filter fraction data
    if "fraction_data" in core:
        filtered_fraction = {}
        for roi_id, data in core["fraction_data"].items():
            filtered = [(t, v) for t, v in data if start_time <= t <= end_time]
            if filtered:
                filtered_fraction[roi_id] = filtered
        subset["core_analysis"]["fraction_data"] = filtered_fraction

    # Filter quiescence bouts
    if "quiescence_bouts" in core:
        filtered_bouts = {}
        for roi_id, bouts in core["quiescence_bouts"].items():
            filtered = [
                b
                for b in bouts
                if start_time <= b["start_time"] <= end_time
                or start_time <= b["end_time"] <= end_time
            ]
            if filtered:
                filtered_bouts[roi_id] = filtered
        subset["core_analysis"]["quiescence_bouts"] = filtered_bouts

    return subset
