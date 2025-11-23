"""
_calc_adaptive.py - Adaptive threshold calculation for HDF5 analysis

This module handles adaptive threshold calculation that automatically adjusts
based on signal-to-noise ratio and data variability.

The adaptive method:
1. Analyzes a sample period of data
2. Calculates signal-to-noise ratio (SNR)
3. Determines coefficient of variation (CV)
4. Adaptively sets threshold based on these metrics
5. Provides hysteresis thresholds for stable movement detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ._calc import (
    validate_frame_difference_data,
    apply_matlab_normalization_to_merged_results,
    define_movement_with_hysteresis,
    bin_fraction_movement,
    bin_quiescence,
    define_sleep_periods,
)


def compute_threshold_adaptive_hysteresis(
    data: List[Tuple[float, float]],
    analysis_duration_frames: int,
    base_multiplier: float = 2.5,
    frame_interval: float = 5.0,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute adaptive hysteresis thresholds based on signal characteristics.

    The adaptive method analyzes the signal-to-noise ratio and variability
    to automatically determine appropriate thresholds.

    Args:
        data: List of (time, value) tuples - Frame difference sums
        analysis_duration_frames: Number of frames to analyze for adaptation
        base_multiplier: Base multiplier for threshold calculation
        frame_interval: Time between frames

    Returns:
        Tuple of (baseline_mean, upper_threshold, lower_threshold, statistics_dict)
    """
    if not data or len(data) < analysis_duration_frames:
        return (
            0.0,
            0.0,
            0.0,
            {"method": "adaptive_hysteresis", "status": "insufficient_data"},
        )

    # Sort data and take analysis period
    sorted_data = sorted(data, key=lambda x: x[0])
    analysis_duration_seconds = analysis_duration_frames * frame_interval
    start_time = sorted_data[0][0]
    end_time = start_time + analysis_duration_seconds

    # Select analysis data by time range
    analysis_data = [(t, v) for t, v in sorted_data if start_time <= t < end_time]

    if len(analysis_data) < 10:
        return (
            0.0,
            0.0,
            0.0,
            {"method": "adaptive_hysteresis", "status": "insufficient_analysis_data"},
        )

    # Extract values for analysis
    values = np.array([val for _, val in analysis_data])

    print(f"Adaptive threshold calculation:")
    print(f"  Analysis period: {len(analysis_data)} frames")
    print(f"  Value range: {np.min(values):.1f} to {np.max(values):.1f}")
    print(f"  Mean: {np.mean(values):.1f}")
    print(f"  Std: {np.std(values):.1f}")

    # Calculate signal characteristics
    mean_val = np.mean(values)
    std_val = np.std(values)
    median_val = np.median(values)

    # Calculate signal-to-noise ratio (SNR)
    # SNR = mean / std (higher SNR = cleaner signal)
    snr = mean_val / std_val if std_val > 0 else 0

    # Calculate coefficient of variation (CV)
    # CV = std / mean (lower CV = more stable signal)
    cv = std_val / mean_val if mean_val > 0 else 1.0

    # Calculate robustness metrics
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25

    # Detect outliers using IQR method
    outlier_threshold_lower = q25 - 1.5 * iqr
    outlier_threshold_upper = q75 + 1.5 * iqr
    outliers = values[
        (values < outlier_threshold_lower) | (values > outlier_threshold_upper)
    ]
    outlier_percentage = len(outliers) / len(values) * 100

    print(f"  Signal characteristics:")
    print(f"    SNR: {snr:.2f}")
    print(f"    CV: {cv:.3f}")
    print(f"    IQR: {iqr:.1f}")
    print(f"    Outliers: {outlier_percentage:.1f}%")

    # Adaptive threshold calculation based on signal characteristics

    # Step 1: Determine baseline reference
    # Use median for noisy signals, mean for clean signals
    if cv > 0.5 or outlier_percentage > 10:
        # Noisy signal - use robust median
        baseline_mean = median_val
        baseline_std = iqr / 1.349  # Convert IQR to std equivalent
        print(f"    Using robust median baseline: {baseline_mean:.1f}")
    else:
        # Clean signal - use mean
        baseline_mean = mean_val
        baseline_std = std_val
        print(f"    Using mean baseline: {baseline_mean:.1f}")

    # Step 2: Adaptive multiplier based on signal quality
    if snr > 10:
        # High SNR - can use smaller multiplier
        adaptive_multiplier = base_multiplier * 0.7
        print(
            f"    High SNR detected - reducing multiplier to {adaptive_multiplier:.2f}"
        )
    elif snr > 5:
        # Medium SNR - use base multiplier
        adaptive_multiplier = base_multiplier
        print(
            f"    Medium SNR detected - using base multiplier {adaptive_multiplier:.2f}"
        )
    else:
        # Low SNR - increase multiplier for stability
        adaptive_multiplier = base_multiplier * 1.3
        print(
            f"    Low SNR detected - increasing multiplier to {adaptive_multiplier:.2f}"
        )

    # Step 3: Adjust for coefficient of variation
    if cv > 1.0:
        # Very variable signal - increase multiplier further
        cv_adjustment = 1.0 + (cv - 1.0) * 0.5
        adaptive_multiplier *= cv_adjustment
        print(
            f"    High variability (CV={cv:.3f}) - adjusting multiplier to {adaptive_multiplier:.2f}"
        )
    elif cv < 0.2:
        # Very stable signal - can reduce multiplier
        cv_adjustment = 0.8 + cv
        adaptive_multiplier *= cv_adjustment
        print(
            f"    Low variability (CV={cv:.3f}) - adjusting multiplier to {adaptive_multiplier:.2f}"
        )

    # Step 4: Calculate hysteresis thresholds
    threshold_band = adaptive_multiplier * baseline_std
    upper_threshold = baseline_mean + threshold_band
    lower_threshold = baseline_mean - threshold_band

    # Step 5: Ensure thresholds are reasonable
    if lower_threshold < 0:
        lower_threshold = max(0, baseline_mean - baseline_std)
        print(f"    Adjusted negative lower threshold to {lower_threshold:.1f}")

    if upper_threshold < baseline_mean * 1.1:
        upper_threshold = baseline_mean * 1.2
        print(f"    Adjusted too-low upper threshold to {upper_threshold:.1f}")

    print(f"  Final adaptive thresholds:")
    print(f"    Baseline: {baseline_mean:.1f}")
    print(f"    Upper: {upper_threshold:.1f}")
    print(f"    Lower: {lower_threshold:.1f}")
    print(f"    Band width: ±{threshold_band:.1f}")

    # Compile statistics
    statistics = {
        "method": "adaptive_hysteresis",
        "baseline_mean": baseline_mean,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "threshold_band": threshold_band,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "snr": snr,
        "cv": cv,
        "iqr": iqr,
        "outlier_percentage": outlier_percentage,
        "base_multiplier": base_multiplier,
        "adaptive_multiplier": adaptive_multiplier,
        "analysis_frames": len(analysis_data),
        "data_range": (np.min(values), np.max(values)),
        "status": "success",
        "signal_quality": "high" if snr > 10 else "medium" if snr > 5 else "low",
        "signal_stability": "high" if cv < 0.2 else "medium" if cv < 0.5 else "low",
        "uses_robust_baseline": cv > 0.5 or outlier_percentage > 10,
    }

    return baseline_mean, upper_threshold, lower_threshold, statistics


def run_adaptive_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
    use_improved_detrending: bool = True,
    analysis_duration_frames: int = 180,
    base_multiplier: float = 2.5,
    frame_interval: float = 5.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Complete adaptive analysis pipeline.

    Args:
        merged_results: Raw frame difference data from Reader
        enable_matlab_norm: Whether to apply MATLAB min-subtraction
        enable_detrending: Whether to apply detrending
        use_improved_detrending: Whether to use improved detrending
        analysis_duration_frames: Number of frames to analyze for adaptation
        base_multiplier: Base multiplier for threshold calculation
        frame_interval: Time between frames
        **kwargs: Additional parameters

    Returns:
        Complete adaptive analysis results dictionary
    """
    print("🚀 RUNNING ADAPTIVE ANALYSIS PIPELINE")
    print("=" * 60)

    analysis_results = {
        "method": "adaptive",
        "parameters": {
            "enable_matlab_norm": enable_matlab_norm,
            "enable_detrending": enable_detrending,
            "use_improved_detrending": use_improved_detrending,
            "analysis_duration_frames": analysis_duration_frames,
            "base_multiplier": base_multiplier,
            "frame_interval": frame_interval,
        },
    }

    # Step 1: Apply preprocessing (FIX THIS SECTION)
    if enable_detrending:
        print("📊 Step 1: Applying detrending and MATLAB normalization...")

        # Apply MATLAB normalization first
        if enable_matlab_norm:
            from ._calc import apply_matlab_normalization_to_merged_results

            normalized_data = apply_matlab_normalization_to_merged_results(
                merged_results
            )
        else:
            normalized_data = merged_results

        # Then apply detrending
        if use_improved_detrending:
            from ._calc import improved_full_dataset_detrending

            processed_data = improved_full_dataset_detrending(normalized_data)
        else:
            from ._calc import normalize_and_detrend_merged_results

            processed_data = normalize_and_detrend_merged_results(normalized_data)
    else:
        print("📊 Step 1: Applying MATLAB normalization only...")
        if enable_matlab_norm:
            from ._calc import apply_matlab_normalization_to_merged_results

            processed_data = apply_matlab_normalization_to_merged_results(
                merged_results
            )
        else:
            processed_data = merged_results

    analysis_results["processed_data"] = processed_data

    # Step 2: Calculate adaptive hysteresis thresholds
    print("📊 Step 2: Computing adaptive hysteresis thresholds...")
    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}

    for roi, data in processed_data.items():
        if not data:
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {"method": "adaptive_hysteresis", "status": "no_data"}
            continue

        try:
            baseline_mean, upper_thresh, lower_thresh, stats = (
                compute_threshold_adaptive_hysteresis(
                    data, analysis_duration_frames, base_multiplier, frame_interval
                )
            )

            baseline_means[roi] = baseline_mean
            upper_thresholds[roi] = upper_thresh
            lower_thresholds[roi] = lower_thresh
            roi_statistics[roi] = stats

            # Log results for first few ROIs
            if roi <= 3:
                print(f"ROI {roi} adaptive results:")
                print(f"  Signal quality: {stats['signal_quality']}")
                print(f"  Signal stability: {stats['signal_stability']}")
                print(f"  SNR: {stats['snr']:.2f}")
                print(f"  CV: {stats['cv']:.3f}")
                print(f"  Final multiplier: {stats['adaptive_multiplier']:.2f}")

        except Exception as e:
            print(f"Error computing adaptive thresholds for ROI {roi}: {e}")
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {
                "method": "adaptive_hysteresis",
                "status": f"error: {e}",
            }

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
        }
    )

    # Step 3: Movement detection with hysteresis
    print("📊 Step 3: Detecting movement with hysteresis...")
    movement_data = define_movement_with_hysteresis(
        processed_data, baseline_means, upper_thresholds, lower_thresholds
    )
    analysis_results["movement_data"] = movement_data

    # Step 4: Calculate fraction movement and behavior analysis
    print("📊 Step 4: Calculating fraction movement...")
    fraction_data = bin_fraction_movement(
        movement_data, bin_size_seconds=60, frame_interval=frame_interval
    )
    analysis_results["fraction_data"] = fraction_data

    print("📊 Step 5: Detecting quiescence and sleep periods...")
    quiescence_data = bin_quiescence(fraction_data, quiescence_threshold=0.5)
    analysis_results["quiescence_data"] = quiescence_data

    sleep_data = define_sleep_periods(
        quiescence_data, sleep_threshold_minutes=8, bin_size_seconds=60
    )
    analysis_results["sleep_data"] = sleep_data

    # Step 6: Generate adaptive analysis summary
    print("📊 Step 6: Generating adaptive analysis summary...")
    summary_stats = _generate_adaptive_summary(roi_statistics)
    analysis_results["summary_stats"] = summary_stats

    print("✅ ADAPTIVE ANALYSIS PIPELINE COMPLETE")
    print("=" * 60)

    return analysis_results


def _generate_adaptive_summary(
    roi_statistics: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate summary statistics for adaptive analysis.

    Args:
        roi_statistics: ROI statistics from adaptive analysis

    Returns:
        Dictionary with summary statistics
    """
    successful_rois = [
        roi for roi, stats in roi_statistics.items() if stats.get("status") == "success"
    ]

    if not successful_rois:
        return {"status": "no_successful_analysis"}

    # Collect metrics
    snr_values = [roi_statistics[roi]["snr"] for roi in successful_rois]
    cv_values = [roi_statistics[roi]["cv"] for roi in successful_rois]
    multiplier_values = [
        roi_statistics[roi]["adaptive_multiplier"] for roi in successful_rois
    ]

    # Signal quality distribution
    signal_qualities = [
        roi_statistics[roi]["signal_quality"] for roi in successful_rois
    ]
    quality_counts = {
        "high": signal_qualities.count("high"),
        "medium": signal_qualities.count("medium"),
        "low": signal_qualities.count("low"),
    }

    # Signal stability distribution
    signal_stabilities = [
        roi_statistics[roi]["signal_stability"] for roi in successful_rois
    ]
    stability_counts = {
        "high": signal_stabilities.count("high"),
        "medium": signal_stabilities.count("medium"),
        "low": signal_stabilities.count("low"),
    }

    # Robust baseline usage
    robust_baseline_count = sum(
        1
        for roi in successful_rois
        if roi_statistics[roi].get("uses_robust_baseline", False)
    )

    summary = {
        "total_rois": len(roi_statistics),
        "successful_rois": len(successful_rois),
        "snr_stats": {
            "mean": np.mean(snr_values),
            "std": np.std(snr_values),
            "min": np.min(snr_values),
            "max": np.max(snr_values),
        },
        "cv_stats": {
            "mean": np.mean(cv_values),
            "std": np.std(cv_values),
            "min": np.min(cv_values),
            "max": np.max(cv_values),
        },
        "multiplier_stats": {
            "mean": np.mean(multiplier_values),
            "std": np.std(multiplier_values),
            "min": np.min(multiplier_values),
            "max": np.max(multiplier_values),
        },
        "signal_quality_distribution": quality_counts,
        "signal_stability_distribution": stability_counts,
        "robust_baseline_usage": {
            "count": robust_baseline_count,
            "percentage": (robust_baseline_count / len(successful_rois)) * 100,
        },
        "recommendations": _generate_adaptive_recommendations(roi_statistics),
    }

    return summary


def _generate_adaptive_recommendations(
    roi_statistics: Dict[int, Dict[str, Any]],
) -> List[str]:
    """
    Generate recommendations based on adaptive analysis results.

    Args:
        roi_statistics: ROI statistics from adaptive analysis

    Returns:
        List of recommendation strings
    """
    recommendations = []

    successful_rois = [
        roi for roi, stats in roi_statistics.items() if stats.get("status") == "success"
    ]

    if not successful_rois:
        recommendations.append("No successful ROI analysis - check data quality")
        return recommendations

    # Analyze signal quality
    signal_qualities = [
        roi_statistics[roi]["signal_quality"] for roi in successful_rois
    ]
    low_quality_count = signal_qualities.count("low")

    if low_quality_count > len(successful_rois) * 0.5:
        recommendations.append(
            "Many ROIs have low signal quality - consider increasing frame interval or improving imaging conditions"
        )

    # Analyze coefficient of variation
    cv_values = [roi_statistics[roi]["cv"] for roi in successful_rois]
    high_cv_count = sum(1 for cv in cv_values if cv > 0.5)

    if high_cv_count > len(successful_rois) * 0.3:
        recommendations.append(
            "High signal variability detected - adaptive method automatically adjusted thresholds"
        )

    # Analyze multiplier distribution
    multiplier_values = [
        roi_statistics[roi]["adaptive_multiplier"] for roi in successful_rois
    ]
    avg_multiplier = np.mean(multiplier_values)

    if avg_multiplier > 3.0:
        recommendations.append(
            "High average multiplier indicates challenging signal conditions - results may be conservative"
        )
    elif avg_multiplier < 1.5:
        recommendations.append(
            "Low average multiplier indicates good signal quality - results should be sensitive"
        )

    # Robust baseline usage
    robust_count = sum(
        1
        for roi in successful_rois
        if roi_statistics[roi].get("uses_robust_baseline", False)
    )

    if robust_count > len(successful_rois) * 0.5:
        recommendations.append(
            "Many ROIs required robust baseline calculation - data may have outliers or noise"
        )

    # Overall assessment
    if len(successful_rois) == len(roi_statistics):
        recommendations.append("All ROIs successfully analyzed with adaptive method")
    elif len(successful_rois) > len(roi_statistics) * 0.8:
        recommendations.append(
            "Most ROIs successfully analyzed - good adaptive analysis results"
        )
    else:
        recommendations.append(
            "Some ROIs failed analysis - check data quality for failed ROIs"
        )

    return recommendations


def integrate_adaptive_analysis_with_widget(widget) -> bool:
    """
    Integration function for adaptive analysis with napari widget.

    Args:
        widget: The napari widget instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message("No merged_results available for adaptive analysis")
            return False

        # Get parameters from widget
        frame_interval = widget.frame_interval.value()

        # Calculate analysis duration in frames
        analysis_duration_minutes = widget.adaptive_duration_minutes.value()
        analysis_duration_frames = int(
            (analysis_duration_minutes * 60) / frame_interval
        )

        widget._log_message("Running adaptive analysis pipeline...")

        # Run adaptive analysis
        adaptive_results = run_adaptive_analysis(
            merged_results=widget.merged_results,
            enable_matlab_norm=True,
            enable_detrending=widget.enable_detrending.isChecked(),
            use_improved_detrending=True,
            analysis_duration_frames=analysis_duration_frames,
            base_multiplier=widget.adaptive_base_multiplier.value(),
            frame_interval=frame_interval,
        )

        # Update widget with adaptive results
        widget.merged_results = adaptive_results.get(
            "processed_data", widget.merged_results
        )
        widget.roi_baseline_means = adaptive_results.get("baseline_means", {})
        widget.roi_upper_thresholds = adaptive_results.get("upper_thresholds", {})
        widget.roi_lower_thresholds = adaptive_results.get("lower_thresholds", {})
        widget.roi_statistics = adaptive_results.get("roi_statistics", {})
        widget.movement_data = adaptive_results.get("movement_data", {})
        widget.fraction_data = adaptive_results.get("fraction_data", {})
        widget.quiescence_data = adaptive_results.get("quiescence_data", {})
        widget.sleep_data = adaptive_results.get("sleep_data", {})

        # Calculate band widths for compatibility
        widget.roi_band_widths = {}
        for roi in widget.roi_baseline_means:
            if (
                roi in widget.roi_upper_thresholds
                and roi in widget.roi_lower_thresholds
            ):
                upper = widget.roi_upper_thresholds[roi]
                lower = widget.roi_lower_thresholds[roi]
                widget.roi_band_widths[roi] = (upper - lower) / 2

        # Log adaptive analysis summary
        summary = adaptive_results.get("summary_stats", {})
        if summary:
            widget._log_message(f"Adaptive Analysis Summary:")
            widget._log_message(
                f"  Successful ROIs: {summary.get('successful_rois', 0)}"
            )
            widget._log_message(
                f"  Average SNR: {summary.get('snr_stats', {}).get('mean', 0):.2f}"
            )
            widget._log_message(
                f"  Average CV: {summary.get('cv_stats', {}).get('mean', 0):.3f}"
            )

            # Log recommendations
            recommendations = summary.get("recommendations", [])
            for rec in recommendations:
                widget._log_message(f"  • {rec}")

        widget._log_message("Adaptive analysis completed successfully")
        return True

    except Exception as e:
        widget._log_message(f"Adaptive analysis failed: {str(e)}")
        import traceback

        widget._log_message(f"Traceback: {traceback.format_exc()}")
        return False


# =============================================================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# =============================================================================


def compute_threshold_adaptive(
    data: List[Tuple[float, float]],
    analysis_duration_frames: int,
    base_multiplier: float = 2.5,
) -> Tuple[float, Dict[str, Any]]:
    """
    LEGACY COMPATIBILITY: Single threshold adaptive calculation.

    Args:
        data: List of (time, value) tuples
        analysis_duration_frames: Number of frames to analyze
        base_multiplier: Base multiplier for threshold calculation

    Returns:
        Tuple of (threshold, statistics)
    """
    baseline_mean, upper_threshold, lower_threshold, stats = (
        compute_threshold_adaptive_hysteresis(
            data, analysis_duration_frames, base_multiplier
        )
    )

    # Return upper threshold for backwards compatibility
    legacy_stats = stats.copy()
    legacy_stats["threshold"] = upper_threshold

    return upper_threshold, legacy_stats
