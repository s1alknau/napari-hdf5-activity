"""
_calc_parallel.py - Multiprocessing support for analysis calculations

This module provides parallel processing capabilities for ROI-level analysis
using Python's multiprocessing module (Python 3.9+ compatible).
"""

from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count


# =============================================================================
# WORKER FUNCTIONS FOR MULTIPROCESSING
# =============================================================================


def _process_single_roi_baseline(
    args: Tuple[int, List[Tuple[float, float]], float, float, float, float],
) -> Tuple[int, Dict[str, Any]]:
    """
    Process a single ROI for baseline analysis (worker function for multiprocessing).

    Args:
        args: Tuple of (roi_id, data, baseline_duration_minutes, multiplier,
                       frame_interval, bin_size_seconds)

    Returns:
        Tuple of (roi_id, results_dict)
    """
    from ._calc import (
        compute_threshold_baseline_hysteresis,
        define_movement_with_hysteresis,
        bin_fraction_movement,
    )

    (
        roi_id,
        data,
        baseline_duration_minutes,
        multiplier,
        frame_interval,
        bin_size_seconds,
    ) = args

    results = {}

    if not data:
        results["status"] = "no_data"
        results["baseline_mean"] = 0.0
        results["upper_threshold"] = 0.0
        results["lower_threshold"] = 0.0
        results["movement_data"] = []
        results["fraction_data"] = []
        return roi_id, results

    try:
        # Step 1: Compute baseline threshold
        baseline_mean, upper_thresh, lower_thresh, stats = (
            compute_threshold_baseline_hysteresis(
                data, baseline_duration_minutes, multiplier, frame_interval
            )
        )

        results["baseline_mean"] = baseline_mean
        results["upper_threshold"] = upper_thresh
        results["lower_threshold"] = lower_thresh
        results["statistics"] = stats

        # Step 2: Hysteresis movement detection
        baseline_means_single = {roi_id: baseline_mean}
        upper_thresholds_single = {roi_id: upper_thresh}
        lower_thresholds_single = {roi_id: lower_thresh}
        data_single = {roi_id: data}

        movement_data_dict = define_movement_with_hysteresis(
            data_single,
            baseline_means_single,
            upper_thresholds_single,
            lower_thresholds_single,
        )
        results["movement_data"] = movement_data_dict.get(roi_id, [])

        # Step 3: Bin fraction movement
        fraction_data_dict = bin_fraction_movement(
            movement_data_dict, bin_size_seconds, frame_interval
        )
        results["fraction_data"] = fraction_data_dict.get(roi_id, [])

        results["status"] = "success"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        results["baseline_mean"] = 0.0
        results["upper_threshold"] = 0.0
        results["lower_threshold"] = 0.0
        results["movement_data"] = []
        results["fraction_data"] = []

    return roi_id, results


# =============================================================================
# PARALLEL ANALYSIS FUNCTIONS
# =============================================================================


def run_baseline_analysis_parallel(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
    use_improved_detrending: bool = True,
    baseline_duration_minutes: float = 200.0,
    multiplier: float = 1.0,
    frame_interval: float = 5.0,
    num_processes: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run baseline analysis with multiprocessing support.

    Processes each ROI in parallel using separate CPU cores.

    Args:
        merged_results: Dictionary mapping ROI IDs to time-series data
        enable_matlab_norm: Apply MATLAB-style normalization
        enable_detrending: Apply detrending
        use_improved_detrending: Use improved detrending algorithm
        baseline_duration_minutes: Duration for baseline calculation
        multiplier: Threshold multiplier
        frame_interval: Time between frames (seconds)
        num_processes: Number of parallel processes (None = auto-detect)
        **kwargs: Additional parameters

    Returns:
        Complete analysis results dictionary
    """
    from ._calc import (
        apply_matlab_normalization_to_merged_results,
        improved_full_dataset_detrending,
        bin_quiescence,
        define_sleep_periods,
    )

    # Determine number of processes
    if num_processes is None or num_processes < 1:
        num_processes = max(1, cpu_count() - 1)
    num_processes = min(num_processes, len(merged_results))  # Don't use more than ROIs

    analysis_results = {
        "method": "baseline",
        "parameters": {
            "enable_matlab_norm": enable_matlab_norm,
            "enable_detrending": enable_detrending,
            "baseline_duration_minutes": baseline_duration_minutes,
            "multiplier": multiplier,
            "frame_interval": frame_interval,
            "matlab_compatible": True,
            "num_processes": num_processes,
            "parallel": True,
        },
    }

    # Step 1: Preprocessing (sequential - shared across all ROIs)
    if enable_matlab_norm:
        normalized_data = apply_matlab_normalization_to_merged_results(merged_results)
    else:
        normalized_data = merged_results

    if enable_detrending and use_improved_detrending:
        processed_data = improved_full_dataset_detrending(normalized_data)
    else:
        processed_data = normalized_data

    analysis_results["processed_data"] = processed_data

    # Step 2: Parallel ROI processing
    bin_size_seconds = kwargs.get("bin_size_seconds", 60)

    # Prepare arguments for parallel processing
    roi_args = [
        (
            roi_id,
            data,
            baseline_duration_minutes,
            multiplier,
            frame_interval,
            bin_size_seconds,
        )
        for roi_id, data in processed_data.items()
    ]

    # Process ROIs in parallel
    if num_processes > 1 and len(roi_args) > 1:
        with Pool(processes=num_processes) as pool:
            roi_results = pool.map(_process_single_roi_baseline, roi_args)
    else:
        # Fallback to sequential processing
        roi_results = [_process_single_roi_baseline(args) for args in roi_args]

    # Step 3: Aggregate results from parallel workers
    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}
    movement_data = {}
    fraction_data = {}

    for roi_id, results in roi_results:
        baseline_means[roi_id] = results["baseline_mean"]
        upper_thresholds[roi_id] = results["upper_threshold"]
        lower_thresholds[roi_id] = results["lower_threshold"]
        roi_statistics[roi_id] = results.get(
            "statistics", {"status": results["status"]}
        )
        movement_data[roi_id] = results["movement_data"]
        fraction_data[roi_id] = results["fraction_data"]

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
            "movement_data": movement_data,
            "fraction_data": fraction_data,
        }
    )

    # Step 4: Post-processing (sequential - needs all ROI data)
    quiescence_threshold = kwargs.get("quiescence_threshold", 0.5)
    quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)
    analysis_results["quiescence_data"] = quiescence_data

    sleep_threshold_minutes = kwargs.get("sleep_threshold_minutes", 8)
    sleep_data = define_sleep_periods(
        quiescence_data, sleep_threshold_minutes, bin_size_seconds
    )
    analysis_results["sleep_data"] = sleep_data

    # Add ROI colors
    try:
        from ._reader import get_roi_colors

        roi_colors = get_roi_colors(sorted(processed_data.keys()))
    except Exception:
        roi_colors = {
            roi: f"C{i}" for i, roi in enumerate(sorted(processed_data.keys()))
        }

    analysis_results["roi_colors"] = roi_colors

    return analysis_results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_optimal_process_count(num_rois: int, max_processes: int = None) -> int:
    """
    Determine optimal number of processes for parallel processing.

    Args:
        num_rois: Number of ROIs to process
        max_processes: Maximum number of processes to use (None = auto)

    Returns:
        Optimal number of processes
    """
    available_cores = cpu_count()

    # Leave one core for system
    recommended = max(1, available_cores - 1)

    # Don't spawn more processes than ROIs
    optimal = min(recommended, num_rois)

    # Apply user limit if specified
    if max_processes is not None and max_processes > 0:
        optimal = min(optimal, max_processes)

    return optimal


def should_use_parallel(num_rois: int, num_processes: int) -> bool:
    """
    Determine if parallel processing should be used.

    Args:
        num_rois: Number of ROIs to process
        num_processes: Requested number of processes

    Returns:
        True if parallel processing should be used
    """
    # Only use parallel if:
    # 1. More than 1 process requested
    # 2. More than 1 ROI to process
    # 3. Enough ROIs to benefit from parallelization (>= 2)
    return num_processes > 1 and num_rois >= 2
