"""
_calc_calibration.py - Calibration-based threshold calculation

This module handles calibration method using pre-computed baseline statistics
from sedated animal recordings.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def run_calibration_analysis_with_precomputed_baseline(
    merged_results: Dict[int, List[Tuple[float, float]]],
    calibration_baseline_statistics: Dict[int, Dict[str, Any]],
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
    use_improved_detrending: bool = True,
    frame_interval: float = 5.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run calibration analysis using pre-computed baseline statistics.

    Args:
        merged_results: Main dataset from reader
        calibration_baseline_statistics: Pre-computed thresholds from calibration
        enable_matlab_norm: Apply MATLAB normalization
        enable_detrending: Apply detrending
        frame_interval: Time between frames
        **kwargs: Additional parameters (bin_size_seconds, quiescence_threshold, etc.)

    Returns:
        Complete analysis results dictionary
    """

    if not calibration_baseline_statistics:
        raise ValueError("No calibration baseline statistics provided")

    analysis_results = {
        "method": "calibration_precomputed",
        "parameters": {
            "enable_matlab_norm": enable_matlab_norm,
            "enable_detrending": enable_detrending,
            "frame_interval": frame_interval,
            "uses_precomputed_calibration_baseline": True,
            "calibration_rois_processed": len(calibration_baseline_statistics),
        },
    }

    # Step 1: Preprocess main dataset
    if enable_matlab_norm:
        from ._calc import apply_matlab_normalization_to_merged_results

        normalized_data = apply_matlab_normalization_to_merged_results(
            merged_results, enable_matlab_norm=True
        )
    else:
        normalized_data = merged_results

    if enable_detrending:
        if use_improved_detrending:
            from ._calc import improved_full_dataset_detrending

            processed_data = improved_full_dataset_detrending(normalized_data)
        else:
            processed_data = normalized_data
    else:
        processed_data = normalized_data

    analysis_results["processed_data"] = processed_data

    # Step 2: Apply calibration thresholds
    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}

    successful_matches = 0
    missing_calibration = 0

    for roi in processed_data.keys():
        if roi in calibration_baseline_statistics:
            cal_stats = calibration_baseline_statistics[roi]

            baseline_means[roi] = cal_stats["baseline_mean"]
            upper_thresholds[roi] = cal_stats["upper_threshold"]
            lower_thresholds[roi] = cal_stats["lower_threshold"]

            roi_statistics[roi] = {
                "method": "calibration_precomputed",
                "baseline_mean": cal_stats["baseline_mean"],
                "baseline_std": cal_stats.get("baseline_std", 0.0),
                "upper_threshold": cal_stats["upper_threshold"],
                "lower_threshold": cal_stats["lower_threshold"],
                "threshold_band": cal_stats.get("threshold_band", 0.0),
                "multiplier": cal_stats.get("multiplier", 1.0),
                "uses_precomputed_baseline": True,
                "status": "success",
            }

            successful_matches += 1

        else:
            # Fallback for ROIs without calibration data
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 1.0
            lower_thresholds[roi] = 0.0

            roi_statistics[roi] = {
                "method": "calibration_precomputed",
                "status": "no_calibration_data_for_roi",
                "baseline_mean": 0.0,
                "upper_threshold": 1.0,
                "lower_threshold": 0.0,
            }

            missing_calibration += 1

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
        }
    )

    # Step 3: Apply hysteresis movement detection
    from ._calc import define_movement_with_hysteresis

    movement_data = define_movement_with_hysteresis(
        processed_data, baseline_means, upper_thresholds, lower_thresholds
    )
    analysis_results["movement_data"] = movement_data

    # Step 4: Behavioral analysis
    successful_movement_data = {
        roi: data
        for roi, data in movement_data.items()
        if roi in roi_statistics and roi_statistics[roi]["status"] == "success"
    }

    if successful_movement_data:
        from ._calc import bin_fraction_movement, bin_quiescence, define_sleep_periods

        bin_size_seconds = kwargs.get("bin_size_seconds", 60)
        fraction_data = bin_fraction_movement(
            successful_movement_data, bin_size_seconds, frame_interval
        )
        analysis_results["fraction_data"] = fraction_data

        quiescence_threshold = kwargs.get("quiescence_threshold", 0.5)
        quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)
        analysis_results["quiescence_data"] = quiescence_data

        sleep_threshold_minutes = kwargs.get("sleep_threshold_minutes", 8)
        sleep_data = define_sleep_periods(
            quiescence_data, sleep_threshold_minutes, bin_size_seconds
        )
        analysis_results["sleep_data"] = sleep_data
    else:
        analysis_results["fraction_data"] = {}
        analysis_results["quiescence_data"] = {}
        analysis_results["sleep_data"] = {}

    # Add ROI colors
    try:
        from ._reader import get_roi_colors

        roi_colors = get_roi_colors(sorted(processed_data.keys()))
    except:
        roi_colors = {
            roi: f"C{i}" for i, roi in enumerate(sorted(processed_data.keys()))
        }

    analysis_results["roi_colors"] = roi_colors

    # Calculate summary statistics
    movement_summary = {}
    for roi, movements in movement_data.items():
        if (
            movements
            and roi in roi_statistics
            and roi_statistics[roi]["status"] == "success"
        ):
            movement_count = sum(1 for _, m in movements if m == 1)
            total_count = len(movements)
            movement_summary[roi] = (
                (movement_count / total_count * 100) if total_count > 0 else 0
            )

    avg_movement = np.mean(list(movement_summary.values())) if movement_summary else 0

    analysis_results["summary"] = {
        "total_rois": len(processed_data),
        "successful_calibration_matches": successful_matches,
        "missing_calibration_data": missing_calibration,
        "success_rate_percent": (
            (successful_matches / len(processed_data) * 100) if processed_data else 0
        ),
        "average_movement_percentage": avg_movement,
        "method_description": "Pre-computed calibration baseline applied to main experimental dataset",
    }

    return analysis_results


def process_calibration_baseline(
    calibration_file_path: str,
    masks: List[np.ndarray],
    frame_interval: float = 5.0,
    calibration_multiplier: float = 1.0,
    chunk_size: int = 50,
    num_processes: int = 1,
    progress_callback: Optional[callable] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Process calibration file to create baseline statistics.

    Args:
        calibration_file_path: Path to calibration HDF5 file
        masks: ROI masks for processing
        frame_interval: Time between frames
        calibration_multiplier: Multiplier for threshold calculation
        chunk_size: Processing chunk size
        num_processes: Number of processes to use
        progress_callback: Progress reporting function

    Returns:
        Dictionary of baseline statistics per ROI
    """

    try:
        # Process calibration file
        from ._reader import process_single_file_in_parallel_dual_structure

        _, calibration_roi_changes, calibration_duration = (
            process_single_file_in_parallel_dual_structure(
                calibration_file_path,
                masks,
                chunk_size=chunk_size,
                progress_callback=progress_callback,
                frame_interval=frame_interval,
                num_processes=num_processes,
            )
        )

        if not calibration_roi_changes:
            raise Exception("No calibration data obtained from processing")

        # Apply preprocessing to calibration data
        from ._calc import (
            apply_matlab_normalization_to_merged_results,
            improved_full_dataset_detrending,
        )

        normalized_calibration = apply_matlab_normalization_to_merged_results(
            calibration_roi_changes, enable_matlab_norm=True
        )

        processed_calibration = improved_full_dataset_detrending(normalized_calibration)

        # Calculate baseline statistics for each ROI
        calibration_baseline_statistics = {}

        for roi, data in processed_calibration.items():
            if not data:
                continue

            # Extract all values from complete calibration dataset
            values = np.array([val for _, val in data])

            # Calculate comprehensive statistics
            cal_mean = np.mean(values)
            cal_std = np.std(values)

            # Calculate hysteresis thresholds
            threshold_band = calibration_multiplier * cal_std
            upper_threshold = cal_mean + threshold_band
            lower_threshold = cal_mean - threshold_band  # ← FIXED: Remove max(0, ...)

            calibration_baseline_statistics[roi] = {
                "baseline_mean": cal_mean,
                "baseline_std": cal_std,
                "upper_threshold": upper_threshold,
                "lower_threshold": lower_threshold,
                "threshold_band": threshold_band,
                "multiplier": calibration_multiplier,
                "data_points": len(values),
                "duration_minutes": calibration_duration / 60,
                "data_range": (float(np.min(values)), float(np.max(values))),
            }

        return calibration_baseline_statistics

    except Exception as e:
        raise Exception(f"Calibration baseline processing failed: {e}")


def integrate_calibration_analysis_with_widget(widget) -> bool:
    """Integration function for calibration analysis with napari widget."""
    try:
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message("No merged_results available for calibration analysis")
            return False

        if not (
            hasattr(widget, "calibration_baseline_processed")
            and widget.calibration_baseline_processed
            and hasattr(widget, "calibration_baseline_statistics")
            and widget.calibration_baseline_statistics
        ):
            widget._log_message("No calibration baseline statistics available")
            return False

        # Extract parameters
        frame_interval = widget.frame_interval.value()
        enable_detrending = widget.enable_detrending.isChecked()

        # Run calibration analysis
        calibration_results = run_calibration_analysis_with_precomputed_baseline(
            merged_results=widget.merged_results,
            calibration_baseline_statistics=widget.calibration_baseline_statistics,
            enable_matlab_norm=True,
            enable_detrending=enable_detrending,
            frame_interval=frame_interval,
            bin_size_seconds=widget.bin_size_seconds.value(),
            quiescence_threshold=widget.quiescence_threshold.value(),
            sleep_threshold_minutes=widget.sleep_threshold_minutes.value(),
        )

        # Update widget with results
        widget.merged_results = calibration_results.get(
            "processed_data", widget.merged_results
        )
        widget.roi_baseline_means = calibration_results.get("baseline_means", {})
        widget.roi_upper_thresholds = calibration_results.get("upper_thresholds", {})
        widget.roi_lower_thresholds = calibration_results.get("lower_thresholds", {})
        widget.roi_statistics = calibration_results.get("roi_statistics", {})
        widget.movement_data = calibration_results.get("movement_data", {})
        widget.fraction_data = calibration_results.get("fraction_data", {})
        widget.quiescence_data = calibration_results.get("quiescence_data", {})
        widget.sleep_data = calibration_results.get("sleep_data", {})

        # Calculate band widths
        widget.roi_band_widths = {}
        for roi in widget.roi_baseline_means:
            if (
                roi in widget.roi_upper_thresholds
                and roi in widget.roi_lower_thresholds
            ):
                upper = widget.roi_upper_thresholds[roi]
                lower = widget.roi_lower_thresholds[roi]
                widget.roi_band_widths[roi] = (upper - lower) / 2

        widget._log_message("Calibration analysis integration completed successfully")
        return True

    except Exception as e:
        widget._log_message(f"Calibration analysis integration failed: {str(e)}")
        return False


# Legacy function for backward compatibility
def run_calibration_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]], **kwargs
) -> Dict[str, Any]:
    """Legacy calibration analysis function."""

    # Check if we have pre-computed statistics
    if "calibration_baseline_statistics" in kwargs:
        return run_calibration_analysis_with_precomputed_baseline(
            merged_results, **kwargs
        )

    # Legacy workflow - process calibration file on-the-fly
    calibration_file_path = kwargs.get("calibration_file_path")
    masks = kwargs.get("masks", [])

    if not calibration_file_path or not masks:
        raise ValueError(
            "Legacy calibration workflow requires calibration_file_path and masks"
        )

    # Process calibration baseline
    calibration_baseline_statistics = process_calibration_baseline(
        calibration_file_path=calibration_file_path,
        masks=masks,
        frame_interval=kwargs.get("frame_interval", 5.0),
        calibration_multiplier=kwargs.get("calibration_multiplier", 1.0),
    )

    # Run analysis with computed baseline
    return run_calibration_analysis_with_precomputed_baseline(
        merged_results, calibration_baseline_statistics, **kwargs
    )
