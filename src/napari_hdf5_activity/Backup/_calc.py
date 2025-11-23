"""
_calc.py - Core calculation module for HDF5 analysis (Baseline Method Only)

This module contains core computational functions and the baseline threshold method.
Other methods are now in separate modules:
- _calc_adaptive.py: Adaptive threshold calculation
- _calc_calibration.py: Calibration-based threshold calculation

SIMPLIFIED: Now focuses only on baseline method and core utilities.
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable


def validate_analysis_parameters(
    frame_interval: float, chunk_size: int, baseline_duration_minutes: float
) -> Tuple[bool, str]:
    """
    Validate analysis parameters before starting computation.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if frame_interval <= 0:
        return False, "Frame interval must be positive"

    if chunk_size <= 0:
        return False, "Chunk size must be positive"

    if baseline_duration_minutes <= 0:
        return False, "Baseline duration must be positive"

    return True, ""


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================


def validate_frame_difference_data(
    merged_results: Dict[int, List[Tuple[float, float]]], stage: str = "Input"
):
    """
    Validate frame difference sum data (NOT expecting 0-1 range).

    Args:
        merged_results: Data from processing pipeline
        stage: Stage description for logging
    """
    print(f"=== VALIDATING FRAME DIFFERENCE DATA: {stage} ===")

    for roi, data in list(merged_results.items())[:2]:  # Check first 2 ROIs
        if not data:
            continue

        values = [val for _, val in data[:100]]  # First 100 values

        print(f"ROI {roi} ({stage}):")
        print(f"  Range: {np.min(values):.1f} to {np.max(values):.1f}")
        print(f"  Mean: {np.mean(values):.1f}")
        print(f"  Std: {np.std(values):.1f}")

        # Correct validation for frame difference sums:
        if np.max(values) > 10000:
            print(f"  ⚠️  Very high values - check ROI size or processing")
        elif np.max(values) < 1:
            print(f"  ⚠️  Very low values - check sensitivity")
        elif 10 <= np.max(values) <= 1000:
            print(f"  ✅ Values in expected range for frame difference analysis")
        else:
            print(
                f"  ℹ️  Values: {np.min(values):.1f}-{np.max(values):.1f} (may be normal)"
            )
        break  # Only check first ROI for brevity


def validate_detrending_effectiveness(
    original_data: Dict[int, List[Tuple[float, float]]],
    detrended_data: Dict[int, List[Tuple[float, float]]],
):
    """
    Validate that detrending actually reduced drift.
    """
    print(f"=== DETRENDING EFFECTIVENESS ANALYSIS ===")

    for roi in list(original_data.keys())[:3]:  # Check first 3 ROIs
        if (
            roi not in detrended_data
            or not original_data[roi]
            or not detrended_data[roi]
        ):
            continue

        # Calculate original drift (slope over time)
        orig_data = original_data[roi]
        detr_data = detrended_data[roi]

        if len(orig_data) < 10:
            continue

        orig_times = [t for t, _ in orig_data]
        orig_values = [v for _, v in orig_data]
        detr_values = [v for _, v in detr_data]

        # Calculate linear trends
        orig_slope = np.polyfit(orig_times, orig_values, 1)[0]
        detr_slope = np.polyfit(orig_times, detr_values, 1)[0]

        # Calculate drift reduction
        drift_reduction = abs(orig_slope) - abs(detr_slope)
        reduction_percent = (
            (drift_reduction / abs(orig_slope)) * 100 if orig_slope != 0 else 0
        )

        print(f"ROI {roi}:")
        print(f"  Original slope: {orig_slope:.6f} units/second")
        print(f"  Detrended slope: {detr_slope:.6f} units/second")
        print(f"  Drift reduction: {reduction_percent:.1f}%")

        if reduction_percent > 50:
            print(f"  ✅ Good detrending effectiveness")
        elif reduction_percent > 20:
            print(f"  ⚠️  Moderate detrending effectiveness")
        else:
            print(f"  ❌ Poor detrending - may need parameter adjustment")

    print("=" * 50)


def validate_matlab_normalization(
    original_data: Dict[int, List[Tuple[float, float]]],
    normalized_data: Dict[int, List[Tuple[float, float]]],
):
    """
    Validate that MATLAB-style normalization worked correctly.
    """
    print(f"=== MATLAB NORMALIZATION VALIDATION ===")

    for roi in list(original_data.keys())[:3]:  # Check first 3 ROIs
        if roi not in normalized_data:
            continue

        orig_values = [val for _, val in original_data[roi]]
        norm_values = [val for _, val in normalized_data[roi]]

        if not orig_values or not norm_values:
            continue

        orig_min = np.min(orig_values)
        orig_max = np.max(orig_values)
        norm_min = np.min(norm_values)
        norm_max = np.max(norm_values)

        print(f"ROI {roi} MATLAB Normalization:")
        print(
            f"  Original: {orig_min:.1f} to {orig_max:.1f} (range: {orig_max-orig_min:.1f})"
        )
        print(
            f"  Normalized: {norm_min:.1f} to {norm_max:.1f} (range: {norm_max-norm_min:.1f})"
        )

        # Check if normalization is correct
        if abs(norm_min) < 0.001:  # Should be ~0
            print(f"  ✅ Minimum correctly set to 0")
        else:
            print(f"  ❌ ERROR: Minimum not 0 ({norm_min:.3f})")

        # Check if range is preserved
        range_preserved = abs((orig_max - orig_min) - (norm_max - norm_min)) < 0.001
        if range_preserved:
            print(f"  ✅ Range preserved")
        else:
            print(f"  ❌ ERROR: Range not preserved")

    print("=" * 50)


# =============================================================================
# MATLAB-STYLE NORMALIZATION FUNCTIONS
# =============================================================================


def apply_matlab_normalization_to_merged_results(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_matlab_norm: bool = True,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Apply MATLAB-style normalization to merged_results from Reader.

    Replicates: normPixelChange = pixelChaStich - min(pixelChaStich);

    Args:
        merged_results: Raw frame difference data from Reader
        enable_matlab_norm: Whether to apply MATLAB-style min-subtraction

    Returns:
        MATLAB-normalized frame difference data
    """
    if not enable_matlab_norm:
        print("📊 MATLAB normalization disabled - using raw Reader output")
        return merged_results

    print(f"=== MATLAB-STYLE NORMALIZATION ===")
    print(
        f"Applying: normPixelChange = pixelChaStich - min(pixelChaStich) for each ROI"
    )
    print(f"Input: {len(merged_results)} ROIs from Reader")

    # Validate input data from Reader
    validate_frame_difference_data(merged_results, "Raw_Reader_Output")

    normalized_results = {}

    for roi, data in merged_results.items():
        if not data:
            normalized_results[roi] = []
            continue

        # Extract times and intensity values
        times = [t for t, _ in data]
        intensities = np.array([val for _, val in data])

        # MATLAB logic: subtract minimum (per ROI)
        min_intensity = np.min(intensities)
        normalized_intensities = intensities - min_intensity

        # Reconstruct data with normalized values
        normalized_data = list(zip(times, normalized_intensities))
        normalized_results[roi] = normalized_data

        # Log normalization statistics for first few ROIs
        if roi <= 3:
            print(f"ROI {roi} MATLAB normalization:")
            print(
                f"  Original range: {np.min(intensities):.1f} to {np.max(intensities):.1f}"
            )
            print(f"  Minimum subtracted: {min_intensity:.1f}")
            print(
                f"  Normalized range: {np.min(normalized_intensities):.1f} to {np.max(normalized_intensities):.1f}"
            )
            print(f"  Data points: {len(intensities)}")

    # Validate normalized data
    validate_frame_difference_data(normalized_results, "MATLAB_Normalized_Output")
    validate_matlab_normalization(merged_results, normalized_results)

    print("=" * 50)
    return normalized_results


# =============================================================================
# DETRENDING FUNCTIONS
# =============================================================================


def detect_and_remove_jumps_aggressive(
    times: np.ndarray, values: np.ndarray, jump_threshold_factor: float = 1.5
) -> Tuple[np.ndarray, List[int]]:
    """
    More aggressive jump detection with lower threshold for better results.

    Args:
        times: Time array
        values: Value array
        jump_threshold_factor: Factor for jump detection (lowered from 3.0 to 1.5)

    Returns:
        Tuple of (corrected_values, jump_indices)
    """
    if len(values) < 10:
        return values, []

    # Use smaller window for more sensitive detection
    window_size = min(20, len(values) // 5)  # Smaller window than before
    if window_size < 5:
        return values, []

    # Calculate differences
    diffs = np.diff(values)

    # Rolling standard deviation of differences
    rolling_std = []
    for i in range(len(diffs)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(diffs), i + window_size // 2 + 1)
        window_diffs = diffs[start_idx:end_idx]
        rolling_std.append(np.std(window_diffs))

    rolling_std = np.array(rolling_std)

    # More aggressive threshold
    jump_threshold = jump_threshold_factor * np.median(rolling_std)
    jump_indices = np.where(np.abs(diffs) > jump_threshold)[0]

    if len(jump_indices) == 0:
        return values, []

    print(
        f"    Aggressive jump detection: Found {len(jump_indices)} jumps with threshold {jump_threshold:.1f}"
    )

    # Correct jumps by adjusting subsequent values
    corrected_values = values.copy()

    for jump_idx in jump_indices:
        jump_size = diffs[jump_idx]
        print(f"    Corrected jump at index {jump_idx}: {jump_size:.1f}")

        # Subtract the jump from all subsequent values
        corrected_values[jump_idx + 1 :] -= jump_size

    return corrected_values, list(jump_indices)


def remove_polynomial_trend(
    times: np.ndarray, values: np.ndarray, degree: int = 2
) -> np.ndarray:
    """
    Remove polynomial trend (better for curved drift than linear-only).

    Args:
        times: Time array
        values: Value array
        degree: Polynomial degree (2 = quadratic, handles curved drift)

    Returns:
        Detrended values
    """
    if len(values) < degree + 1:
        return values

    try:
        # Fit polynomial
        poly_coeffs = np.polyfit(times, values, degree)
        poly_trend = np.polyval(poly_coeffs, times)

        # Remove trend while preserving mean level
        mean_value = np.mean(values)
        detrended = values - poly_trend + np.mean(poly_trend)

        # Ensure we preserve the original scale
        if np.mean(detrended) < mean_value * 0.5:  # If we've shifted too much
            detrended = values - (poly_trend - poly_trend[0])  # Alternative approach

        trend_range = np.ptp(poly_trend)
        print(
            f"    Polynomial detrending: degree {degree}, trend range {trend_range:.1f}"
        )

        return detrended

    except Exception as e:
        print(f"    Polynomial detrending failed: {e}")
        return values


def remove_linear_drift(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Remove any remaining linear drift after polynomial correction.

    Args:
        times: Time array
        values: Value array

    Returns:
        Linear-detrended values
    """
    if len(values) < 10:
        return values

    try:
        # Fit linear trend
        slope, intercept = np.polyfit(times, values, 1)
        linear_trend = slope * times + intercept

        # Check if drift is significant
        total_drift = abs(slope * (times[-1] - times[0]))
        drift_percentage = (
            (total_drift / np.mean(values)) * 100 if np.mean(values) > 0 else 0
        )

        if drift_percentage > 1.0:  # Only remove if > 1% drift
            # Remove linear component but preserve starting level
            detrended = values - (linear_trend - intercept)
            print(
                f"    Linear detrending: {drift_percentage:.1f}% drift removed (slope: {slope:.6f})"
            )
            return detrended
        else:
            print(
                f"    Linear detrending: {drift_percentage:.1f}% drift - no correction needed"
            )
            return values

    except Exception as e:
        print(f"    Linear detrending failed: {e}")
        return values


def improved_full_dataset_detrending(
    merged_results: Dict[int, List[Tuple[float, float]]],
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Apply improved detrending to the COMPLETE dataset (not just baseline).

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples

    Returns:
        Dictionary with fully detrended values
    """
    detrended_results = {}

    for roi, data in merged_results.items():
        if not data or len(data) < 20:
            detrended_results[roi] = data
            continue

        try:
            # Sort data by time
            sorted_data = sorted(data, key=lambda x: x[0])
            times = np.array([t for t, _ in sorted_data])
            values = np.array([val for _, val in sorted_data])

            print(
                f"ROI {roi}: Processing {len(values)} points, range {np.min(values):.0f}-{np.max(values):.0f}"
            )

            # Step 1: More aggressive jump detection and correction
            values_jump_corrected = detect_and_remove_jumps_aggressive(times, values)[0]

            # Step 2: Remove polynomial trend (handles curved drift better than linear)
            values_poly_detrended = remove_polynomial_trend(
                times, values_jump_corrected, degree=2
            )

            # Step 3: Remove any remaining linear drift
            values_final = remove_linear_drift(times, values_poly_detrended)

            # Reconstruct data
            detrended_data = list(zip(times, values_final))
            detrended_results[roi] = detrended_data

            # Report effectiveness
            original_drift = np.ptp(values)  # peak-to-peak
            final_drift = np.ptp(values_final)
            reduction = (
                (1 - final_drift / original_drift) * 100 if original_drift > 0 else 0
            )

            # Calculate slope reduction
            original_slope = np.polyfit(times, values, 1)[0]
            final_slope = np.polyfit(times, values_final, 1)[0]
            slope_reduction = (
                (1 - abs(final_slope) / abs(original_slope)) * 100
                if original_slope != 0
                else 0
            )

            print(
                f"  ✅ ROI {roi}: Drift reduced by {reduction:.1f}% ({original_drift:.0f} → {final_drift:.0f})"
            )
            print(
                f"      Slope reduced by {slope_reduction:.1f}% ({original_slope:.6f} → {final_slope:.6f})"
            )

        except Exception as e:
            print(f"ROI {roi}: Full dataset detrending failed: {e}")
            detrended_results[roi] = data

    return detrended_results


def robust_detrend_baseline(
    times: np.ndarray, values: np.ndarray, enable_jump_correction: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Robust detrending that handles jumps and non-linear trends.

    Args:
        times: Time array
        values: Value array
        enable_jump_correction: Whether to correct sudden jumps

    Returns:
        Tuple of (detrended_values, detrend_info)
    """
    original_values = values.copy()
    detrend_info = {
        "jump_correction_applied": False,
        "jumps_detected": 0,
        "linear_detrending_applied": False,
        "drift_slope": 0.0,
        "drift_percentage": 0.0,
    }

    # Step 1: Jump correction
    if enable_jump_correction:
        corrected_values, jump_indices = detect_and_remove_jumps_aggressive(
            times, values
        )

        if len(jump_indices) > 0:
            detrend_info["jump_correction_applied"] = True
            detrend_info["jumps_detected"] = len(jump_indices)
            detrend_info["jump_indices"] = jump_indices
            values = corrected_values
            print(f"    Applied jump correction: {len(jump_indices)} jumps removed")

    # Step 2: Linear detrending on jump-corrected data
    if len(values) >= 10:
        try:
            slope, intercept = np.polyfit(times, values, 1)
            trend_line = slope * times + intercept

            # Check if significant linear trend remains
            total_drift = abs(slope * (times[-1] - times[0]))
            drift_percentage = (
                (total_drift / np.mean(original_values)) * 100
                if np.mean(original_values) > 0
                else 0
            )

            detrend_info["drift_slope"] = slope
            detrend_info["drift_percentage"] = drift_percentage

            if drift_percentage > 2:  # Significant remaining trend
                detrended_values = values - (trend_line - intercept)
                detrend_info["linear_detrending_applied"] = True
                print(
                    f"    Applied linear detrending: {drift_percentage:.1f}% drift removed"
                )
            else:
                detrended_values = values
                print(
                    f"    No significant linear trend remaining: {drift_percentage:.1f}%"
                )

        except Exception as e:
            detrended_values = values
            print(f"    Linear detrending failed: {e}")
    else:
        detrended_values = values

    return detrended_values, detrend_info


def normalize_and_detrend_merged_results(
    merged_results: Dict[int, List[Tuple[float, float]]],
    use_improved_detrending: bool = True,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Apply detrending to frame difference sum data.

    Args:
        merged_results: Frame difference sum data from Reader
        baseline_duration_seconds: Duration for baseline analysis (legacy)
        use_improved_detrending: Whether to use improved full-dataset detrending

    Returns:
        Detrended frame difference data
    """
    if merged_results is None or not merged_results:
        print("ERROR: Invalid merged_results input!")
        return {}

    print(f"Processing {len(merged_results)} ROIs for FULL DATASET DETRENDING")

    # Validate input data
    validate_frame_difference_data(merged_results, "Before_Detrending")

    try:
        if use_improved_detrending:
            print("🚀 Using IMPROVED full-dataset detrending...")

            # Add debug info about dataset size
            total_points = sum(len(data) for data in merged_results.values())
            first_roi_data = next(iter(merged_results.values()))
            if first_roi_data:
                duration_hours = (first_roi_data[-1][0] - first_roi_data[0][0]) / 3600
                print(
                    f"  Dataset: {total_points} total points, {duration_hours:.1f} hours duration"
                )

            # Apply improved full-dataset detrending
            detrended_results = improved_full_dataset_detrending(merged_results)

        else:
            print("Using legacy detrending...")
            detrended_results = merged_results

        if not detrended_results:
            print("ERROR: Detrending function returned empty dict!")
            return merged_results

        print(f"✅ Successfully applied detrending to {len(detrended_results)} ROIs")

        # Validate detrending effectiveness
        validate_detrending_effectiveness(merged_results, detrended_results)

        return detrended_results

    except Exception as e:
        print(f"ERROR in detrending: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return merged_results


# =============================================================================
# BASELINE THRESHOLD CALCULATION (HYSTERESIS)
# =============================================================================

# def compute_threshold_baseline_hysteresis(data: List[Tuple[float, float]],
#                                         threshold_block_count: int,
#                                         multiplier: float = 1.0,
#                                         enable_detrending: bool = True,
#                                         enable_jump_correction: bool = True,
#                                         frame_interval: float = 5.0) -> Tuple[float, float, float, Dict[str, Any]]:
#     """
#     Compute hysteresis thresholds using baseline method with robust detrending.

#     Args:
#         data: List of (time, value) tuples - Frame difference sums from Reader
#         threshold_block_count: Number of frames for baseline
#         multiplier: Threshold multiplier for band calculation
#         enable_detrending: Whether to apply detrending to baseline data
#         enable_jump_correction: Whether to correct sudden jumps/plateaus
#         frame_interval: Time between frames

#     Returns:
#         Tuple of (baseline_mean, upper_threshold, lower_threshold, statistics_dict)
#     """
#     if not data or len(data) < threshold_block_count:
#         return 0.0, 0.0, 0.0, {'method': 'baseline_hysteresis', 'status': 'insufficient_data'}

#     # Sort data and take baseline period BY TIME RANGE
#     sorted_data = sorted(data, key=lambda x: x[0])

#     # KORREKTUR: baseline_duration_seconds statt baseline_duration_minutes
#     baseline_duration_seconds = threshold_block_count * frame_interval  # In Sekunden!
#     start_time = sorted_data[0][0]
#     end_time = start_time + baseline_duration_seconds

#     # Select data by TIME RANGE instead of frame count
#     baseline_data = [(t, v) for t, v in sorted_data if start_time <= t < end_time]

#     # Log die korrekte Information
#     baseline_duration_minutes = baseline_duration_seconds / 60.0  # Für Anzeige
#     print(f"Baseline calculation: {len(baseline_data)} frames over {baseline_duration_minutes:.1f} minutes")

#     # Extract time and values for baseline
#     times = np.array([t for t, _ in baseline_data])
#     values = np.array([val for _, val in baseline_data])

#     # Store original statistics
#     original_mean = np.mean(values)
#     original_std = np.std(values)

#     print(f"  Raw baseline mean: {original_mean:.1f}, std: {original_std:.1f}")

#     # Apply robust detrending if enabled
#     if enable_detrending and len(values) >= 10:
#         try:
#             print(f"  Applying robust detrending (jumps={enable_jump_correction})...")
#             detrended_values, detrend_info = robust_detrend_baseline(times, values, enable_jump_correction)

#             # Use detrended values for threshold calculation
#             mean_val = np.mean(detrended_values)
#             std_val = np.std(detrended_values)

#             processed_values_for_stats = detrended_values.copy()

#             print(f"  Robust detrending results:")
#             print(f"    Jump correction: {detrend_info['jump_correction_applied']} ({detrend_info['jumps_detected']} jumps)")
#             print(f"    Linear detrending: {detrend_info['linear_detrending_applied']} ({detrend_info['drift_percentage']:.1f}% drift)")
#             print(f"    Detrended mean: {mean_val:.1f}")
#             print(f"    Detrended std: {std_val:.1f}")

#             was_detrended = detrend_info['jump_correction_applied'] or detrend_info['linear_detrending_applied']
#             status = 'robust_detrending_applied'

#             drift_slope = detrend_info['drift_slope']
#             detrending_details = detrend_info

#         except Exception as e:
#             # If detrending fails, fall back to original values
#             mean_val = original_mean
#             std_val = original_std
#             drift_slope = 0.0
#             status = f'robust_detrending_failed_{str(e)[:20]}'
#             was_detrended = False
#             detrending_details = {}
#             processed_values_for_stats = values.copy()
#             print(f"  Robust detrending failed: {e}")
#     else:
#         # Use original values without detrending
#         mean_val = original_mean
#         std_val = original_std
#         drift_slope = 0.0
#         was_detrended = False
#         detrending_details = {}
#         processed_values_for_stats = values.copy()

#         if not enable_detrending:
#             status = 'detrending_disabled'
#             print(f"  Detrending: disabled by user")
#         else:
#             status = 'insufficient_data_for_detrending'
#             print(f"  Detrending: insufficient data ({len(values)} points)")

#     # Calculate hysteresis thresholds
#     baseline_mean = mean_val
#     threshold_band = multiplier * std_val
#     upper_threshold = baseline_mean + threshold_band
#     lower_threshold = baseline_mean - threshold_band

#     print(f"  Hysteresis threshold calculation:")
#     print(f"    Baseline Mean: {baseline_mean:.1f}")
#     print(f"    Band Width: ±{threshold_band:.1f} (multiplier: {multiplier:.2f})")
#     print(f"    Upper Threshold: {upper_threshold:.1f}")
#     print(f"    Lower Threshold: {lower_threshold:.1f}")

#     # Basic sanity checks
#     if upper_threshold < 0 or lower_threshold < 0:
#         old_upper = upper_threshold
#         old_lower = lower_threshold
#         if baseline_mean > threshold_band:
#             upper_threshold = baseline_mean + threshold_band
#             lower_threshold = max(0, baseline_mean - threshold_band)
#         else:
#             upper_threshold = baseline_mean * 2
#             lower_threshold = baseline_mean * 0.5
#         status += '_negative_threshold_corrected'
#         print(f"    → Corrected negative thresholds: {old_lower:.1f}/{old_upper:.1f} → {lower_threshold:.1f}/{upper_threshold:.1f}")

#     # Check for invalid values
#     if np.isnan(upper_threshold) or np.isinf(upper_threshold) or np.isnan(lower_threshold) or np.isinf(lower_threshold):
#         upper_threshold = np.percentile(values, 75)
#         lower_threshold = np.percentile(values, 25)
#         baseline_mean = np.median(values)
#         status += '_nan_threshold_corrected'
#         print(f"    → Corrected invalid thresholds using percentiles: {lower_threshold:.1f}/{upper_threshold:.1f}")

#     print(f"  Final hysteresis thresholds:")
#     print(f"    Baseline Mean: {baseline_mean:.1f}")
#     print(f"    Upper Threshold: {upper_threshold:.1f}")
#     print(f"    Lower Threshold: {lower_threshold:.1f}")
#     print(f"  Status: {status}")

#     statistics = {
#         'method': 'baseline_hysteresis',
#         'baseline_mean': baseline_mean,
#         'upper_threshold': upper_threshold,
#         'lower_threshold': lower_threshold,
#         'threshold_band': threshold_band,
#         'mean': mean_val,
#         'std': std_val,
#         'multiplier': multiplier,
#         'baseline_frames': len(baseline_data),
#         'baseline_duration_minutes': baseline_duration_minutes,  # Neu: korrekte Minuten-Angabe
#         'baseline_duration_seconds': baseline_duration_seconds,  # Neu: Sekunden-Angabe
#         'enable_detrending': enable_detrending,
#         'enable_jump_correction': enable_jump_correction,
#         'was_detrended': was_detrended,
#         'drift_slope': drift_slope,
#         'original_mean': original_mean,
#         'original_std': original_std,
#         'data_range': (np.min(values), np.max(values)),
#         'detrending_details': detrending_details,
#         'status': status,
#         'processed_baseline_values': processed_values_for_stats,
#         'original_baseline_values': values,
#         'baseline_times': times,
#         'should_plot_processed_data': enable_detrending and was_detrended,
#         'baseline_data_for_plotting': list(zip(times, processed_values_for_stats))
#     }


#     return baseline_mean, upper_threshold, lower_threshold, statistics
def compute_threshold_baseline_hysteresis(
    data: List[Tuple[float, float]],
    baseline_duration_minutes: float,
    multiplier: float = 1.0,
    enable_detrending: bool = True,
    enable_jump_correction: bool = True,
    frame_interval: float = 5.0,
    data_already_detrended: bool = False,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute hysteresis thresholds using baseline method.

    WICHTIG: Diese Funktion arbeitet auf bereits vorverarbeiteten Daten!

    Args:
        data: List of (time, value) tuples - bereits normalisiert und ggf. detrended
        baseline_duration_minutes: Duration of baseline period in MINUTES
        multiplier: Threshold multiplier for band calculation
        enable_detrending: IGNORIERT - nur für Backwards Compatibility
        enable_jump_correction: IGNORIERT - nur für Backwards Compatibility
        frame_interval: Time between frames in seconds
        data_already_detrended: IGNORIERT - Daten sind immer vorverarbeitet
    """

    if not data:
        return 0.0, 0.0, 0.0, {"method": "baseline_hysteresis", "status": "no_data"}

    # Sort data by time
    sorted_data = sorted(data, key=lambda x: x[0])

    # Berechne Zeitbereich für Baseline
    baseline_duration_seconds = baseline_duration_minutes * 60
    start_time = sorted_data[0][0]
    end_time = start_time + baseline_duration_seconds

    # Wähle Daten im Baseline-Zeitbereich
    baseline_data = [(t, v) for t, v in sorted_data if start_time <= t < end_time]

    # Prüfe ob genug Daten vorhanden sind
    min_required_frames = max(10, int(baseline_duration_seconds / frame_interval * 0.8))
    if len(baseline_data) < min_required_frames:
        return (
            0.0,
            0.0,
            0.0,
            {
                "method": "baseline_hysteresis",
                "status": "insufficient_data",
                "found_frames": len(baseline_data),
                "required_frames": min_required_frames,
            },
        )

    print(f"  Baseline calculation for ROI:")
    print(f"    Duration: {baseline_duration_minutes:.1f} minutes")
    print(f"    Frames in baseline period: {len(baseline_data)}")

    # Extract values for baseline period (times not needed for simple stats)
    times = np.array([t for t, _ in baseline_data])
    values = np.array([val for _, val in baseline_data])

    # Berechne Statistiken auf den BEREITS VORVERARBEITETEN Daten
    mean_val = np.mean(values)
    std_val = np.std(values)

    print(f"    Baseline statistics: mean={mean_val:.1f}, std={std_val:.1f}")

    # Berechne Hysterese-Schwellenwerte
    baseline_mean = mean_val
    threshold_band = multiplier * std_val
    upper_threshold = baseline_mean + threshold_band
    lower_threshold = baseline_mean - threshold_band

    print(f"    Hysteresis thresholds:")
    print(f"      Baseline Mean: {baseline_mean:.1f}")
    print(f"      Band Width: ±{threshold_band:.1f} (multiplier: {multiplier})")
    print(f"      Upper Threshold: {upper_threshold:.1f}")
    print(f"      Lower Threshold: {lower_threshold:.1f}")

    # Sanity checks
    if upper_threshold < 0 or lower_threshold < 0:
        print(f"    WARNING: Negative thresholds detected!")
        # Bei negativen Schwellenwerten setze lower auf 0
        if lower_threshold < 0:
            lower_threshold = 0
            print(f"      Adjusted lower threshold to 0")

    if (
        np.isnan(upper_threshold)
        or np.isinf(upper_threshold)
        or np.isnan(lower_threshold)
        or np.isinf(lower_threshold)
    ):
        print(f"    ERROR: Invalid thresholds! Using percentiles as fallback")
        # Fallback to percentiles
        upper_threshold = np.percentile(values, 75)
        lower_threshold = np.percentile(values, 25)
        baseline_mean = np.median(values)

    statistics = {
        "method": "baseline_hysteresis",
        "baseline_mean": baseline_mean,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "threshold_band": threshold_band,
        "mean": mean_val,
        "std": std_val,
        "multiplier": multiplier,
        "baseline_frames": len(baseline_data),
        "baseline_duration_minutes": baseline_duration_minutes,
        "baseline_duration_seconds": baseline_duration_seconds,
        "frame_interval": frame_interval,
        "data_range": (np.min(values), np.max(values)),
        "status": "calculated_from_preprocessed_data",
        "baseline_times": times.tolist(),
        "baseline_values": values.tolist(),
    }

    return baseline_mean, upper_threshold, lower_threshold, statistics


# =============================================================================
# HYSTERESIS-BASED MOVEMENT DETECTION
# =============================================================================


def define_movement_with_hysteresis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    roi_baseline_means: Dict[int, float],
    roi_upper_thresholds: Dict[int, float],
    roi_lower_thresholds: Dict[int, float],
) -> Dict[int, List[Tuple[float, int]]]:
    """
    Define movement with hysteresis logic to prevent flicker near thresholds.

    Hysteresis Logic:
    - Signal > Upper Threshold → Movement = TRUE
    - Signal < Lower Threshold → Movement = FALSE
    - Signal between thresholds → State remains unchanged (no flicker)

    Args:
        merged_results: ROI intensity data (frame difference sums)
        roi_baseline_means: Baseline means per ROI
        roi_upper_thresholds: Upper thresholds per ROI
        roi_lower_thresholds: Lower thresholds per ROI

    Returns:
        Movement data with hysteresis logic applied
    """
    movement_data = {}

    print(f"=== HYSTERESIS-BASED MOVEMENT DETECTION ===")

    for roi, data in merged_results.items():
        if roi not in roi_upper_thresholds or roi not in roi_lower_thresholds:
            movement_data[roi] = []
            continue

        upper_thresh = roi_upper_thresholds[roi]
        lower_thresh = roi_lower_thresholds[roi]
        baseline = roi_baseline_means[roi]

        # Sort data by time
        sorted_data = sorted(data, key=lambda x: x[0])

        if not sorted_data:
            movement_data[roi] = []
            continue

        # Determine initial state based on first value relative to baseline
        first_value = sorted_data[0][1]
        if first_value > upper_thresh:
            current_movement_state = 1  # Movement
        elif first_value < lower_thresh:
            current_movement_state = 0  # No movement
        else:
            # First value in hysteresis band: use baseline as reference
            current_movement_state = 1 if first_value > baseline else 0

        roi_movement = []
        state_changes = 0
        values_above_upper = 0
        values_below_lower = 0
        values_in_band = 0

        for time_point, value in sorted_data:
            # Hysteresis logic
            if current_movement_state == 0:  # Currently: No Movement
                if value > upper_thresh:
                    current_movement_state = 1  # Switch to Movement
                    state_changes += 1
                    values_above_upper += 1
                # Else: state remains unchanged (prevents flicker)
            else:  # Currently: Movement
                if value < lower_thresh:
                    current_movement_state = 0  # Switch to No Movement
                    state_changes += 1
                    values_below_lower += 1
                # Else: state remains unchanged (prevents flicker)

            # Count values in different regions for statistics
            if value > upper_thresh:
                values_above_upper += 1
            elif value < lower_thresh:
                values_below_lower += 1
            else:
                values_in_band += 1

            roi_movement.append((time_point, current_movement_state))

        movement_data[roi] = roi_movement

        # Log statistics for first few ROIs
        if roi <= 3:
            total_points = len(sorted_data)
            movement_percentage = (
                (sum(m for _, m in roi_movement) / total_points * 100)
                if total_points > 0
                else 0
            )
            band_percentage = (
                (values_in_band / total_points * 100) if total_points > 0 else 0
            )

            print(f"ROI {roi} Hysteresis Analysis:")
            print(
                f"  Thresholds: Lower={lower_thresh:.1f}, Baseline={baseline:.1f}, Upper={upper_thresh:.1f}"
            )
            print(f"  Data points: {total_points}")
            print(f"  State changes: {state_changes} (reduced flicker)")
            print(
                f"  Values above upper: {values_above_upper} ({values_above_upper/total_points*100:.1f}%)"
            )
            print(
                f"  Values below lower: {values_below_lower} ({values_below_lower/total_points*100:.1f}%)"
            )
            print(f"  Values in band: {values_in_band} ({band_percentage:.1f}%)")
            print(f"  Final movement: {movement_percentage:.1f}% of time")

    print("=" * 50)
    return movement_data


# =============================================================================
# ACTIVITY ANALYSIS FUNCTIONS
# =============================================================================


def bin_fraction_movement(
    movement_data: Dict[int, List[Tuple[float, int]]],
    bin_size_seconds: int,
    frame_interval: float,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    CORRECTED: Calculate fraction movement for HYSTERESIS-based movement detection.

    With hysteresis, movement_data contains state changes, not individual frame classifications.
    Each data point represents: (time, current_movement_state)

    The state remains constant until the signal crosses a threshold boundary.

    Fraction = (Time spent in movement state) / (Total time in bin)

    Args:
        movement_data: Dictionary mapping ROI ID to list of (time, movement_state) tuples
                      where movement_state is 0 (no movement) or 1 (movement)
        bin_size_seconds: Size of time bins in seconds
        frame_interval: Time interval between frames (for reference only)

    Returns:
        Dictionary mapping ROI ID to list of (bin_center_time, fraction_movement) tuples
    """
    fraction_data = {}

    print(f"=== HYSTERESIS FRACTION MOVEMENT CALCULATION ===")
    print(f"Bin size: {bin_size_seconds}s")
    print(f"Method: Calculate time spent in movement state within each bin")

    for roi, data in movement_data.items():
        if not data:
            fraction_data[roi] = []
            continue

        # Sort data by time
        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            fraction_data[roi] = []
            continue

        # Get time range
        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]
        total_duration = (end_time - start_time) / 60

        print(f"\nROI {roi} hysteresis fraction calculation:")
        print(f"  State change points: {len(sorted_data)}")
        print(f"  Time range: {start_time:.1f}s to {end_time:.1f}s")
        print(f"  Duration: {total_duration:.1f} minutes")

        # Create time bins
        first_bin_start = (start_time // bin_size_seconds) * bin_size_seconds
        bin_edges = []
        current_bin_start = first_bin_start
        while current_bin_start < end_time:
            bin_edges.append(current_bin_start)
            current_bin_start += bin_size_seconds
        bin_edges.append(current_bin_start)

        print(f"  Number of bins: {len(bin_edges) - 1}")

        roi_fractions = []

        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2
            bin_duration = bin_end - bin_start

            # Calculate time spent in movement state within this bin
            movement_time = 0.0

            # Find all state periods that overlap with this bin
            for j in range(len(sorted_data)):
                current_time = sorted_data[j][0]
                current_state = sorted_data[j][1]

                # Determine when this state ends
                if j + 1 < len(sorted_data):
                    next_time = sorted_data[j + 1][0]
                else:
                    next_time = end_time  # Last state continues to end

                # Check if this state period overlaps with current bin
                state_start = max(current_time, bin_start)
                state_end = min(next_time, bin_end)

                if state_start < state_end:  # There is overlap
                    overlap_duration = state_end - state_start

                    if current_state == 1:  # Movement state
                        movement_time += overlap_duration

            # Calculate fraction of time spent moving
            if bin_duration > 0:
                fraction_movement = movement_time / bin_duration
            else:
                fraction_movement = 0.0

            # Ensure fraction is in valid range
            fraction_movement = max(0.0, min(1.0, fraction_movement))

            roi_fractions.append((bin_center, fraction_movement))

            # Debug first few bins for first ROI
            if roi == sorted(movement_data.keys())[0] and i < 3:
                print(f"    Bin {i+1}: {bin_start:.1f}s-{bin_end:.1f}s")
                print(f"      Bin duration: {bin_duration:.1f}s")
                print(f"      Movement time: {movement_time:.1f}s")
                print(f"      Fraction: {fraction_movement:.3f}")

        fraction_data[roi] = roi_fractions

        # Summary statistics for first ROI
        if roi == sorted(movement_data.keys())[0]:
            fractions = [f for _, f in roi_fractions]
            print(f"  Fraction movement summary:")
            print(f"    Range: {min(fractions):.3f} to {max(fractions):.3f}")
            print(f"    Mean: {np.mean(fractions):.3f}")
            print(f"    Std: {np.std(fractions):.3f}")

            # Distribution analysis
            zero_bins = sum(1 for f in fractions if f == 0.0)
            full_bins = sum(1 for f in fractions if f == 1.0)
            partial_bins = len(fractions) - zero_bins - full_bins

            print(f"    Distribution:")
            print(
                f"      No movement (0.0): {zero_bins} bins ({zero_bins/len(fractions)*100:.1f}%)"
            )
            print(
                f"      Partial movement: {partial_bins} bins ({partial_bins/len(fractions)*100:.1f}%)"
            )
            print(
                f"      Full movement (1.0): {full_bins} bins ({full_bins/len(fractions)*100:.1f}%)"
            )

    print("=" * 50)
    return fraction_data


def bin_quiescence(
    fraction_data: Dict[int, List[Tuple[float, float]]],
    quiescence_threshold: float = 0.5,
) -> Dict[int, List[Tuple[float, int]]]:
    """
    CORRECTED: Calculate quiescence based on fraction movement data.

    CORRECT LOGIC:
    - fraction_movement < quiescence_threshold → Quiescent (1)
    - fraction_movement >= quiescence_threshold → Active (0)

    This makes sense for quiescence analysis:
    - 1 = "Yes, animal is quiescent" (low movement)
    - 0 = "No, animal is not quiescent" (high movement)

    Args:
        fraction_data: Dictionary mapping ROI ID to list of (time, fraction_movement) tuples
        quiescence_threshold: Threshold below which animal is considered quiescent

    Returns:
        Dictionary mapping ROI ID to list of (time, quiescence_binary) tuples
        where 1 = quiescent, 0 = active
    """
    quiescence_data = {}

    print(f"=== CORRECTED QUIESCENCE CALCULATION ===")
    print(
        f"Quiescence threshold: {quiescence_threshold} ({quiescence_threshold*100}% movement)"
    )
    print(f"CORRECT logic:")
    print(f"  fraction_movement < {quiescence_threshold} → Quiescent (1) ✅")
    print(f"  fraction_movement >= {quiescence_threshold} → Active (0)")

    for roi, data in fraction_data.items():
        quiescent_roi_data = []

        quiescent_bins = 0
        active_bins = 0

        for time_point, fraction_movement in data:
            # CORRECTED LOGIC: Quiescent when movement is LOW
            if fraction_movement < quiescence_threshold:
                quiescence_state = 1  # Quiescent (low movement)
                quiescent_bins += 1
            else:  # fraction_movement >= quiescence_threshold
                quiescence_state = 0  # Active (high movement)
                active_bins += 1

            quiescent_roi_data.append((time_point, quiescence_state))

        quiescence_data[roi] = quiescent_roi_data

        # Log summary for first ROI
        if roi == sorted(fraction_data.keys())[0]:
            total_bins = len(quiescent_roi_data)
            if total_bins > 0:
                percent_quiescent = (quiescent_bins / total_bins) * 100
                percent_active = (active_bins / total_bins) * 100

                print(f"\n  ROI {roi} summary:")
                print(f"    Total bins: {total_bins}")
                print(
                    f"    Quiescent bins (1): {quiescent_bins} ({percent_quiescent:.1f}%)"
                )
                print(f"    Active bins (0): {active_bins} ({percent_active:.1f}%)")

                # Show examples
                sample_size = min(5, len(data))
                print(f"    First {sample_size} examples:")
                for i in range(sample_size):
                    time_point, fraction = data[i]
                    _, quiescence = quiescent_roi_data[i]
                    status = "Quiescent (1)" if quiescence == 1 else "Active (0)"
                    print(
                        f"      t={time_point/60:.1f}min: fraction={fraction:.3f} → {status}"
                    )

    print("=" * 50)
    return quiescence_data


def define_sleep_periods(
    quiescence_data: Dict[int, List[Tuple[float, int]]],
    sleep_threshold_minutes: int = 8,
    bin_size_seconds: int = 60,
) -> Dict[int, List[Tuple[float, int]]]:
    """
    CORRECTED: Define sleep periods based on sustained quiescence.

    CORRECTED LOGIC:
    - quiescence_data contains: 1 = quiescent, 0 = active
    - Sleep = sustained periods of quiescence (consecutive 1s) for ≥ sleep_threshold_minutes

    Args:
        quiescence_data: Dictionary with (time, quiescence_state) where 1=quiescent, 0=active
        sleep_threshold_minutes: Minimum duration of quiescence to be considered sleep
        bin_size_seconds: Size of time bins used in fraction calculation

    Returns:
        Dictionary mapping ROI ID to list of (time, sleep_binary) tuples
        where 1 = sleep, 0 = not sleep
    """
    sleep_data = {}
    min_bins_for_sleep = (sleep_threshold_minutes * 60) // bin_size_seconds

    print(f"=== CORRECTED SLEEP PERIOD DETECTION ===")
    print(f"Sleep threshold: {sleep_threshold_minutes} minutes")
    print(f"Bin size: {bin_size_seconds} seconds")
    print(f"Minimum bins for sleep: {min_bins_for_sleep}")
    print(f"CORRECTED logic: Sustained quiescence (consecutive 1s) → Sleep (1)")

    for roi, data in quiescence_data.items():
        if not data:
            sleep_data[roi] = []
            continue

        sleep_roi_data = []

        # Convert to arrays for easier processing
        times = np.array([t for t, _ in data])
        quiescence_states = np.array([q for _, q in data])  # 1=quiescent, 0=active

        # Find continuous quiescence periods (consecutive 1s)
        sleep_state = np.zeros_like(quiescence_states)

        i = 0
        while i < len(quiescence_states):
            if quiescence_states[i] == 1:  # Start of potential sleep (quiescent)
                # Count consecutive quiescent bins
                consecutive_quiescent_count = 0
                j = i
                while j < len(quiescence_states) and quiescence_states[j] == 1:
                    consecutive_quiescent_count += 1
                    j += 1

                # If quiescent period is long enough, mark as sleep
                if consecutive_quiescent_count >= min_bins_for_sleep:
                    sleep_state[i:j] = 1  # Mark as sleep
                    print(
                        f"    ROI {roi}: Sleep period detected from index {i} to {j-1} ({consecutive_quiescent_count} bins = {consecutive_quiescent_count * bin_size_seconds / 60:.1f}min)"
                    )

                i = j
            else:
                i += 1

        # Convert back to list of tuples
        sleep_roi_data = list(zip(times, sleep_state))
        sleep_data[roi] = sleep_roi_data

        # Summary for first ROI
        if roi == sorted(quiescence_data.keys())[0]:
            sleep_bins = sum(sleep_state)
            total_bins = len(sleep_state)
            sleep_percent = (sleep_bins / total_bins * 100) if total_bins > 0 else 0
            sleep_time_minutes = (sleep_bins * bin_size_seconds) / 60

            print(f"\n  ROI {roi} sleep analysis:")
            print(f"    Total bins: {total_bins}")
            print(f"    Sleep bins: {sleep_bins} ({sleep_percent:.1f}%)")
            print(f"    Sleep time: {sleep_time_minutes:.1f} minutes")

            # Count sleep episodes
            sleep_episodes = 0
            in_sleep = False
            for state in sleep_state:
                if state == 1 and not in_sleep:
                    sleep_episodes += 1
                    in_sleep = True
                elif state == 0:
                    in_sleep = False

            print(f"    Sleep episodes: {sleep_episodes}")
            if sleep_episodes > 0:
                avg_episode_length = sleep_time_minutes / sleep_episodes
                print(f"    Average episode: {avg_episode_length:.1f} minutes")

    print("=" * 50)
    return sleep_data


def validate_quiescence_and_sleep_logic(
    fraction_data: Dict[int, List[Tuple[float, float]]],
    quiescence_threshold: float = 0.5,
    sleep_threshold_minutes: int = 8,
    bin_size_seconds: int = 60,
) -> None:
    """
    VALIDATION FUNCTION: Test the corrected quiescence and sleep logic.

    This function helps verify that the logic is working correctly.
    """
    print(f"\n🧪 === VALIDATING CORRECTED QUIESCENCE & SLEEP LOGIC ===")

    if not fraction_data:
        print("❌ No fraction data to validate")
        return

    # Test quiescence calculation
    quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)

    # Test sleep calculation
    sleep_data = define_sleep_periods(
        quiescence_data, sleep_threshold_minutes, bin_size_seconds
    )

    # Validation checks
    for roi in list(fraction_data.keys())[:2]:  # Check first 2 ROIs
        if roi not in quiescence_data or roi not in sleep_data:
            continue

        fraction_values = [f for _, f in fraction_data[roi]]
        quiescence_values = [q for _, q in quiescence_data[roi]]
        sleep_values = [s for _, s in sleep_data[roi]]

        print(f"\n🔍 ROI {roi} validation:")

        # Check quiescence logic
        for i in range(min(10, len(fraction_values))):
            fraction = fraction_values[i]
            quiescence = quiescence_values[i]
            expected_quiescence = 1 if fraction < quiescence_threshold else 0

            status = "✅" if quiescence == expected_quiescence else "❌"
            print(
                f"  {status} Fraction: {fraction:.3f} → Quiescent: {quiescence} (expected: {expected_quiescence})"
            )

            if quiescence != expected_quiescence:
                print(
                    f"      ❌ LOGIC ERROR: fraction {fraction:.3f} should give quiescence {expected_quiescence}"
                )
                return

        # Check sleep-quiescence relationship
        sleep_without_quiescence = 0
        for i in range(len(sleep_values)):
            if sleep_values[i] == 1 and quiescence_values[i] == 0:
                sleep_without_quiescence += 1

        if sleep_without_quiescence > 0:
            print(
                f"  ❌ ERROR: {sleep_without_quiescence} sleep periods without quiescence!"
            )
        else:
            print(f"  ✅ All sleep periods occur during quiescence")

        # Summary statistics
        total_bins = len(fraction_values)
        quiescent_bins = sum(quiescence_values)
        sleep_bins = sum(sleep_values)

        print(f"  📊 Statistics:")
        print(f"    Total bins: {total_bins}")
        print(f"    Quiescent: {quiescent_bins} ({quiescent_bins/total_bins*100:.1f}%)")
        print(f"    Sleep: {sleep_bins} ({sleep_bins/total_bins*100:.1f}%)")
        print(
            f"    Sleep/Quiescence ratio: {sleep_bins/max(1,quiescent_bins)*100:.1f}%"
        )

    print(f"\n✅ VALIDATION COMPLETE - Logic appears correct!")
    print(f"=" * 60)


# def bin_activity_data_for_lighting(fraction_data: Dict[int, List[Tuple[float, float]]],
#                                  bin_minutes: int) -> Dict[int, List[Tuple[float, float]]]:
#     """
#     Bin activity data for lighting analysis visualization.

#     Args:
#         fraction_data: Dictionary mapping ROI ID to list of (time, fraction) tuples
#         bin_minutes: Bin size in minutes

#     Returns:
#         Dictionary with binned activity data
#     """
#     bin_size_seconds = bin_minutes * 60
#     binned_data = {}

#     for roi, data in fraction_data.items():
#         if not data:
#             binned_data[roi] = []
#             continue

#         # Sort data by time
#         sorted_data = sorted(data, key=lambda x: x[0])

#         # Create bins
#         start_time = sorted_data[0][0]
#         end_time = sorted_data[-1][0]

#         binned_roi_data = []
#         current_time = start_time

#         while current_time < end_time:
#             bin_end = current_time + bin_size_seconds

#             # Get data points in this bin
#             bin_data = [val for t, val in sorted_data
#                        if current_time <= t < bin_end]

#             if bin_data:
#                 # Average activity in this bin
#                 avg_activity = np.mean(bin_data)
#                 bin_center = current_time + (bin_size_seconds / 2)
#                 binned_roi_data.append((bin_center, avg_activity))

#             current_time = bin_end

#         binned_data[roi] = binned_roi_data


#     return binned_data
def bin_activity_data_for_lighting(
    fraction_data: Dict[int, List[Tuple[float, float]]], bin_minutes: int = 30
) -> Dict[int, List[Tuple[float, float]]]:
    """
    UPDATED: Bin activity data for circadian/lighting analysis visualization.

    For circadian analysis, we want to see ACTIVITY patterns over longer time periods.
    This function bins the fraction movement data into larger time windows to reveal
    daily patterns and lighting effects.

    LOGIC: Higher values = more active periods (good for circadian analysis)

    Args:
        fraction_data: Dictionary mapping ROI ID to list of (time, fraction_movement) tuples
        bin_minutes: Bin size in minutes (30 min default for circadian analysis)

    Returns:
        Dictionary with binned activity data for lighting visualization
        Higher values = more active periods
    """
    bin_size_seconds = bin_minutes * 60
    binned_data = {}

    print(f"=== ACTIVITY BINNING FOR LIGHTING/CIRCADIAN ANALYSIS ===")
    print(f"Bin size: {bin_minutes} minutes ({bin_size_seconds} seconds)")
    print(f"Purpose: Reveal daily activity patterns and lighting effects")
    print(f"Output: Higher values = more active periods")

    for roi, data in fraction_data.items():
        if not data:
            binned_data[roi] = []
            continue

        # Sort data by time
        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            binned_data[roi] = []
            continue

        # Get time range
        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]
        total_duration_hours = (end_time - start_time) / 3600

        print(f"\nROI {roi} lighting analysis:")
        print(f"  Data points: {len(sorted_data)}")
        print(f"  Time range: {start_time:.1f}s to {end_time:.1f}s")
        print(f"  Duration: {total_duration_hours:.1f} hours")

        # Create time bins aligned to nice boundaries
        # Start from the beginning of the first hour for cleaner circadian analysis
        first_hour_start = (start_time // 3600) * 3600  # Round down to nearest hour

        binned_roi_data = []
        current_time = first_hour_start

        bin_count = 0
        while current_time < end_time:
            bin_end = current_time + bin_size_seconds

            # Get data points in this bin
            bin_data = []
            for t, fraction_movement in sorted_data:
                if current_time <= t < bin_end:
                    bin_data.append(fraction_movement)

            if bin_data:
                # Calculate average activity in this bin
                # Higher fraction_movement = more active = good for circadian analysis
                avg_activity = np.mean(bin_data)
                bin_center = current_time + (bin_size_seconds / 2)
                binned_roi_data.append((bin_center, avg_activity))
                bin_count += 1

                # Log first few bins for debugging
                if bin_count <= 3 and roi == sorted(fraction_data.keys())[0]:
                    hour_of_day = (bin_center % (24 * 3600)) / 3600
                    print(
                        f"    Bin {bin_count}: hour {hour_of_day:.1f}, activity {avg_activity:.3f}"
                    )

            current_time = bin_end

        binned_data[roi] = binned_roi_data

        # Log summary for first ROI
        if roi == sorted(fraction_data.keys())[0]:
            if binned_roi_data:
                activities = [activity for _, activity in binned_roi_data]
                print(f"  Binned summary:")
                print(f"    Total bins: {len(binned_roi_data)}")
                print(
                    f"    Activity range: {np.min(activities):.3f} to {np.max(activities):.3f}"
                )
                print(f"    Mean activity: {np.mean(activities):.3f}")
                print(f"    Activity std: {np.std(activities):.3f}")

                # Find peak activity periods
                peak_threshold = np.mean(activities) + np.std(activities)
                peak_bins = sum(
                    1 for activity in activities if activity > peak_threshold
                )
                print(
                    f"    Peak activity bins: {peak_bins}/{len(activities)} ({peak_bins/len(activities)*100:.1f}%)"
                )

    print("=" * 50)
    return binned_data


def bin_quiescence_data_for_lighting(
    quiescence_data: Dict[int, List[Tuple[float, int]]], bin_minutes: int = 30
) -> Dict[int, List[Tuple[float, float]]]:
    """
    NEW FUNCTION: Bin quiescence data for lighting analysis.

    This is an alternative approach for circadian analysis that focuses on
    quiescent periods rather than activity. Some researchers prefer this
    because it shows "rest patterns" more clearly.

    LOGIC: Higher values = more quiescent periods (good for rest/sleep analysis)

    Args:
        quiescence_data: Dictionary mapping ROI ID to list of (time, quiescence_binary) tuples
        bin_minutes: Bin size in minutes

    Returns:
        Dictionary with binned quiescence fractions for lighting analysis
        Higher values = more quiescent periods
    """
    bin_size_seconds = bin_minutes * 60
    binned_data = {}

    print(f"=== QUIESCENCE BINNING FOR LIGHTING/CIRCADIAN ANALYSIS ===")
    print(f"Bin size: {bin_minutes} minutes")
    print(f"Purpose: Reveal daily rest patterns and lighting effects")
    print(f"Output: Higher values = more quiescent periods")

    for roi, data in quiescence_data.items():
        if not data:
            binned_data[roi] = []
            continue

        # Sort data by time
        sorted_data = sorted(data, key=lambda x: x[0])

        # Get time range
        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]

        # Create time bins
        binned_roi_data = []
        current_time = start_time

        while current_time < end_time:
            bin_end = current_time + bin_size_seconds

            # Get data points in this bin
            bin_data = []
            for t, quiescence_state in sorted_data:
                if current_time <= t < bin_end:
                    bin_data.append(quiescence_state)

            if bin_data:
                # Calculate fraction of time spent quiescent in this bin
                fraction_quiescent = np.mean(
                    bin_data
                )  # Since quiescence_state is 0 or 1
                bin_center = current_time + (bin_size_seconds / 2)
                binned_roi_data.append((bin_center, fraction_quiescent))

            current_time = bin_end

        binned_data[roi] = binned_roi_data

    print("=" * 50)
    return binned_data


def choose_lighting_analysis_data(
    fraction_data: Dict[int, List[Tuple[float, float]]] = None,
    quiescence_data: Dict[int, List[Tuple[float, int]]] = None,
    bin_minutes: int = 30,
    analysis_focus: str = "activity",
) -> Dict[int, List[Tuple[float, float]]]:
    """
    HELPER FUNCTION: Choose appropriate data for lighting/circadian analysis.

    This function helps decide whether to use activity-based or quiescence-based
    binning for circadian analysis based on the research focus.

    Args:
        fraction_data: Fraction movement data
        quiescence_data: Quiescence binary data
        bin_minutes: Bin size for analysis
        analysis_focus: 'activity' or 'rest' - determines which approach to use

    Returns:
        Binned data appropriate for lighting analysis
    """
    print(f"=== CHOOSING LIGHTING ANALYSIS APPROACH ===")
    print(f"Analysis focus: {analysis_focus}")

    if analysis_focus.lower() == "activity":
        print("Using ACTIVITY-based approach (fraction movement)")
        print("Higher values = more active periods")
        print("Good for: Activity rhythm analysis, feeding patterns, exploration")

        if fraction_data:
            return bin_activity_data_for_lighting(fraction_data, bin_minutes)
        else:
            print("❌ No fraction data available for activity analysis")
            return {}

    elif analysis_focus.lower() == "rest":
        print("Using REST-based approach (quiescence periods)")
        print("Higher values = more quiescent periods")
        print("Good for: Sleep rhythm analysis, rest patterns, circadian rest")

        if quiescence_data:
            return bin_quiescence_data_for_lighting(quiescence_data, bin_minutes)
        else:
            print("❌ No quiescence data available for rest analysis")
            return {}

    else:
        print(f"❌ Unknown analysis focus: {analysis_focus}")
        print("   Valid options: 'activity' or 'rest'")
        return {}


def validate_lighting_analysis_data(
    original_fraction_data: Dict[int, List[Tuple[float, float]]],
    binned_activity_data: Dict[int, List[Tuple[float, float]]],
    binned_quiescence_data: Dict[int, List[Tuple[float, float]]],
    bin_minutes: int = 30,
) -> None:
    """
    VALIDATION FUNCTION: Verify that lighting analysis data makes sense.

    This function helps ensure the binning process preserved the essential
    characteristics of the original data.
    """
    print(f"\n🧪 === VALIDATING LIGHTING ANALYSIS DATA ===")

    if not original_fraction_data:
        print("❌ No original fraction data to validate against")
        return

    for roi in list(original_fraction_data.keys())[:2]:  # Check first 2 ROIs
        print(f"\n🔍 ROI {roi} validation:")

        # Original data statistics
        orig_fractions = [f for _, f in original_fraction_data[roi]]
        orig_mean = np.mean(orig_fractions)
        orig_std = np.std(orig_fractions)

        print(f"  Original data:")
        print(f"    Points: {len(orig_fractions)}")
        print(f"    Mean fraction: {orig_mean:.3f}")
        print(f"    Std: {orig_std:.3f}")

        # Activity binned data
        if roi in binned_activity_data:
            activity_values = [a for _, a in binned_activity_data[roi]]
            activity_mean = np.mean(activity_values)

            print(f"  Activity binned ({bin_minutes}min bins):")
            print(f"    Points: {len(activity_values)}")
            print(f"    Mean activity: {activity_mean:.3f}")
            print(
                f"    Compression ratio: {len(orig_fractions)/len(activity_values):.1f}x"
            )

            # Check if means are similar (should be!)
            mean_diff = abs(activity_mean - orig_mean)
            if mean_diff < 0.05:  # Within 5%
                print(f"    ✅ Mean preserved: diff = {mean_diff:.3f}")
            else:
                print(f"    ⚠️ Mean changed significantly: diff = {mean_diff:.3f}")

        # Quiescence binned data
        if roi in binned_quiescence_data:
            quiescence_values = [q for _, q in binned_quiescence_data[roi]]
            quiescence_mean = np.mean(quiescence_values)

            print(f"  Quiescence binned ({bin_minutes}min bins):")
            print(f"    Points: {len(quiescence_values)}")
            print(f"    Mean quiescence: {quiescence_mean:.3f}")

            # Activity and quiescence should be complementary
            expected_activity = 1.0 - quiescence_mean  # Rough approximation
            activity_mean = (
                np.mean([a for _, a in binned_activity_data[roi]])
                if roi in binned_activity_data
                else 0
            )

            complementary_diff = abs(activity_mean - expected_activity)
            if complementary_diff < 0.2:  # Within 20%
                print(
                    f"    ✅ Activity/quiescence relationship reasonable: diff = {complementary_diff:.3f}"
                )
            else:
                print(
                    f"    ⚠️ Activity/quiescence relationship unclear: diff = {complementary_diff:.3f}"
                )

    print(f"\n✅ LIGHTING ANALYSIS VALIDATION COMPLETE")
    print(f"=" * 60)


def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
    """
    Calculate performance metrics for analysis.

    Args:
        start_time: Analysis start time
        total_frames: Total number of frames processed

    Returns:
        Dictionary with performance metrics
    """
    try:
        import psutil

        elapsed_time = time.time() - start_time
        fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        return {
            "elapsed_time": elapsed_time,
            "fps": fps,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "total_frames": total_frames,
        }
    except ImportError:
        elapsed_time = time.time() - start_time
        fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        return {
            "elapsed_time": elapsed_time,
            "fps": fps,
            "cpu_percent": 0,
            "memory_percent": 0,
            "total_frames": total_frames,
        }


# =============================================================================
# MAIN BASELINE ANALYSIS FUNCTION
# =============================================================================


def run_baseline_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
    use_improved_detrending: bool = True,
    baseline_duration_minutes: float = 200.0,
    multiplier: float = 1.0,
    frame_interval: float = 5.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    FIXED VERSION: Properly respects user-selected baseline duration.

    The baseline calculation now correctly samples the signal for the exact
    duration specified by the user in the widget.
    """
    print("🚀 RUNNING BASELINE ANALYSIS PIPELINE - USER DURATION RESPECTED")
    print("=" * 60)
    print(f"📊 User-selected baseline duration: {baseline_duration_minutes} minutes")

    analysis_results = {
        "method": "baseline",
        "parameters": {
            "enable_matlab_norm": enable_matlab_norm,
            "enable_detrending": enable_detrending,
            "use_improved_detrending": use_improved_detrending,
            "baseline_duration_minutes": baseline_duration_minutes,
            "multiplier": multiplier,
            "frame_interval": frame_interval,
        },
    }

    # Step 1: Apply preprocessing
    print("\n📊 Step 1: Preprocessing data...")

    if enable_matlab_norm:
        print("    Applying MATLAB normalization (min-subtraction)...")
        normalized_data = apply_matlab_normalization_to_merged_results(merged_results)
    else:
        print("    Skipping MATLAB normalization")
        normalized_data = merged_results

    if enable_detrending:
        print("    Applying detrending...")
        if use_improved_detrending:
            processed_data = improved_full_dataset_detrending(normalized_data)
        else:
            processed_data = normalize_and_detrend_merged_results(normalized_data)
    else:
        print("    Skipping detrending")
        processed_data = normalized_data

    analysis_results["processed_data"] = processed_data

    # Step 2: Calculate baseline using USER-SPECIFIED duration
    print(f"\n📊 Step 2: Computing baseline from processed signal...")
    print(f"    Baseline duration: {baseline_duration_minutes} minutes (user-selected)")
    print(f"    Frame interval: {frame_interval} seconds")
    print(f"    Threshold multiplier: {multiplier}")

    baseline_duration_seconds = baseline_duration_minutes * 60
    expected_baseline_points = int(baseline_duration_seconds / frame_interval)

    print(f"    Expected signal points in baseline: {expected_baseline_points}")

    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}

    for roi, data in processed_data.items():
        if not data:
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {"method": "baseline", "status": "no_data"}
            continue

        # Sort signal by time
        sorted_data = sorted(data, key=lambda x: x[0])
        total_points = len(sorted_data)

        print(f"\n    ROI {roi} - Baseline calculation:")
        print(f"      Total signal points: {total_points}")
        print(
            f"      Signal time span: {sorted_data[0][0]:.1f}s to {sorted_data[-1][0]:.1f}s"
        )
        print(
            f"      Signal duration: {(sorted_data[-1][0] - sorted_data[0][0])/60:.1f} minutes"
        )

        # Define baseline time range
        signal_start_time = sorted_data[0][0]
        baseline_end_time = signal_start_time + baseline_duration_seconds

        print(
            f"      Baseline time range: {signal_start_time:.1f}s to {baseline_end_time:.1f}s"
        )

        # Sample ALL signal values within the baseline duration
        baseline_signal_values = []
        baseline_times = []

        for time_point, intensity_change in sorted_data:
            if signal_start_time <= time_point <= baseline_end_time:
                baseline_signal_values.append(intensity_change)
                baseline_times.append(time_point)

        baseline_points_found = len(baseline_signal_values)
        print(f"      Signal points in baseline: {baseline_points_found}")

        # Check coverage
        if baseline_points_found == 0:
            print(f"      ❌ No baseline data found!")
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {"method": "baseline", "status": "no_baseline_data"}
            continue
        elif baseline_points_found < 10:
            print(f"      ⚠️ Very few baseline points ({baseline_points_found})")

        # Calculate actual baseline duration and coverage
        if len(baseline_times) > 1:
            actual_baseline_duration = (baseline_times[-1] - baseline_times[0]) / 60
        else:
            actual_baseline_duration = 0

        coverage_percent = (
            (baseline_points_found / expected_baseline_points) * 100
            if expected_baseline_points > 0
            else 0
        )

        print(f"      Actual baseline duration: {actual_baseline_duration:.1f} minutes")
        print(f"      Coverage: {coverage_percent:.1f}% of requested duration")

        # Calculate baseline statistics from the sampled signal
        baseline_array = np.array(baseline_signal_values)

        baseline_mean = np.mean(baseline_array)
        baseline_std = np.std(baseline_array)
        baseline_median = np.median(baseline_array)
        baseline_min = np.min(baseline_array)
        baseline_max = np.max(baseline_array)

        print(f"      Baseline signal statistics:")
        print(f"        Range: {baseline_min:.1f} to {baseline_max:.1f}")
        print(f"        Mean: {baseline_mean:.1f}")
        print(f"        Median: {baseline_median:.1f}")
        print(f"        Std: {baseline_std:.1f}")
        print(f"        Sample values: {baseline_array[:5]}")

        # Calculate hysteresis thresholds
        threshold_band = multiplier * baseline_std
        upper_threshold = baseline_mean + threshold_band
        lower_threshold = baseline_mean - threshold_band

        # Ensure lower threshold is not negative
        if lower_threshold < 0:
            lower_threshold = 0
            print(f"        Adjusted negative lower threshold to 0")

        print(f"      Hysteresis thresholds:")
        print(f"        Baseline mean: {baseline_mean:.1f}")
        print(
            f"        Upper threshold: {upper_threshold:.1f} (mean + {multiplier}×std)"
        )
        print(
            f"        Lower threshold: {lower_threshold:.1f} (mean - {multiplier}×std)"
        )
        print(f"        Band width: ±{threshold_band:.1f}")

        # Validate that baseline makes sense
        if baseline_mean <= 0:
            print(
                f"      ⚠️ WARNING: Baseline mean is {baseline_mean:.1f} - check data quality"
            )
        if threshold_band <= 0:
            print(
                f"      ⚠️ WARNING: Threshold band is {threshold_band:.1f} - no variation in baseline"
            )

        # Store results
        baseline_means[roi] = baseline_mean
        upper_thresholds[roi] = upper_threshold
        lower_thresholds[roi] = lower_threshold

        roi_statistics[roi] = {
            "method": "baseline",
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "baseline_median": baseline_median,
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "threshold_band": threshold_band,
            "multiplier": multiplier,
            "baseline_signal_points": baseline_points_found,
            "expected_baseline_points": expected_baseline_points,
            "baseline_duration_minutes": actual_baseline_duration,
            "requested_baseline_minutes": baseline_duration_minutes,
            "coverage_percent": coverage_percent,
            "data_preprocessing": {
                "matlab_norm_applied": enable_matlab_norm,
                "detrending_applied": enable_detrending,
            },
            "signal_range": (float(baseline_min), float(baseline_max)),
            "status": "success",
        }

    # Log summary
    successful_rois = sum(
        1 for stats in roi_statistics.values() if stats.get("status") == "success"
    )
    print(
        f"\n    ✅ Baseline calculation complete: {successful_rois}/{len(processed_data)} ROIs successful"
    )

    # Show summary of baseline calculations
    if successful_rois > 0:
        print(f"\n    📊 Baseline Summary:")
        for roi in sorted(baseline_means.keys())[:3]:  # Show first 3 ROIs
            if roi in roi_statistics and roi_statistics[roi]["status"] == "success":
                stats = roi_statistics[roi]
                print(f"      ROI {roi}:")
                print(f"        Baseline mean: {baseline_means[roi]:.1f}")
                print(f"        Coverage: {stats['coverage_percent']:.1f}%")
                print(
                    f"        Duration: {stats['baseline_duration_minutes']:.1f}/{baseline_duration_minutes} min"
                )

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
        }
    )

    # Step 3: Movement detection with hysteresis
    print("\n📊 Step 3: Detecting movement with hysteresis...")
    movement_data = define_movement_with_hysteresis(
        processed_data, baseline_means, upper_thresholds, lower_thresholds
    )
    analysis_results["movement_data"] = movement_data

    # Log movement summary
    movement_summary = {}
    for roi, movements in movement_data.items():
        if movements:
            movement_count = sum(1 for _, m in movements if m == 1)
            total_count = len(movements)
            movement_summary[roi] = (
                (movement_count / total_count * 100) if total_count > 0 else 0
            )

    if movement_summary:
        avg_movement = np.mean(list(movement_summary.values()))
        print(f"    Average movement across ROIs: {avg_movement:.1f}%")
    else:
        avg_movement = 0

    # Step 4: Calculate fraction movement and behavior analysis
    print("\n📊 Step 4: Calculating behavioral metrics...")
    bin_size_seconds = kwargs.get("bin_size_seconds", 60)
    fraction_data = bin_fraction_movement(
        movement_data, bin_size_seconds, frame_interval
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

    # Add ROI colors
    try:
        from ._reader import get_roi_colors

        roi_colors = get_roi_colors(sorted(processed_data.keys()))
    except:
        roi_colors = {
            roi: f"C{i}" for i, roi in enumerate(sorted(processed_data.keys()))
        }

    analysis_results["roi_colors"] = roi_colors

    # Add comprehensive summary
    analysis_results["summary"] = {
        "total_rois": len(processed_data),
        "successful_baseline_calculations": successful_rois,
        "baseline_duration_minutes": baseline_duration_minutes,
        "user_selected_duration": baseline_duration_minutes,
        "data_preprocessing": {
            "matlab_norm_applied": enable_matlab_norm,
            "detrending_applied": enable_detrending,
            "processing_type": (
                "processed" if (enable_matlab_norm or enable_detrending) else "raw"
            ),
        },
        "average_movement_percentage": avg_movement if movement_summary else 0,
    }

    print("\n✅ BASELINE ANALYSIS PIPELINE COMPLETE")
    print(f"   Processed {len(processed_data)} ROIs")
    print(f"   User baseline duration: {baseline_duration_minutes} minutes (respected)")
    print(f"   Successful calculations: {successful_rois}/{len(processed_data)}")
    print("=" * 60)

    return analysis_results


# =============================================================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# =============================================================================


def run_complete_hdf5_compatible_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]], **kwargs
) -> Dict[str, Any]:
    """
    SIMPLIFIED: Direct baseline analysis without routing complexity.

    Args:
        merged_results: Raw frame difference data from Reader
        **kwargs: All analysis parameters including threshold_method

    Returns:
        Complete analysis results dictionary
    """
    print("🚀 SIMPLIFIED LEGACY COMPATIBILITY: Using direct baseline analysis")

    # Extract threshold method for logging
    threshold_method = kwargs.get("threshold_method", "baseline")
    print(f"📊 Requested method: {threshold_method}")

    if threshold_method != "baseline":
        print(f"⚠️  Only baseline method available in simplified mode")

    # Remove threshold_method from kwargs to avoid conflicts
    clean_kwargs = {k: v for k, v in kwargs.items() if k != "threshold_method"}

    print("🚀 Running baseline analysis directly...")
    try:
        results = run_baseline_analysis(merged_results, **clean_kwargs)
        print(
            f"✅ Baseline analysis completed with {len(results.get('baseline_means', {}))} ROIs"
        )
        return results
    except Exception as e:
        print(f"❌ Baseline analysis failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        raise


def run_complete_matlab_compatible_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]], **kwargs
) -> Dict[str, Any]:
    """
    BACKWARDS COMPATIBILITY: Legacy MATLAB-compatible function.

    Args:
        merged_results: Raw frame difference data from Reader
        **kwargs: Analysis parameters

    Returns:
        Complete analysis results dictionary
    """
    print(
        "🔄 Legacy MATLAB compatibility: Using baseline analysis with MATLAB normalization"
    )

    # Force MATLAB normalization for this legacy function
    kwargs["enable_matlab_norm"] = True
    kwargs["threshold_method"] = "baseline"  # MATLAB compatibility uses baseline

    return run_complete_hdf5_compatible_analysis(merged_results, **kwargs)


def process_with_matlab_compatibility(
    merged_results: Dict[int, List[Tuple[float, float]]], **kwargs
) -> Dict[str, Any]:
    """
    BACKWARDS COMPATIBILITY: Legacy MATLAB processing function.

    Args:
        merged_results: Raw frame difference data from Reader
        **kwargs: Analysis parameters

    Returns:
        Complete analysis results dictionary
    """
    print(
        "🔄 Legacy MATLAB processing: Using baseline analysis with MATLAB normalization"
    )

    # Ensure MATLAB compatibility
    kwargs["enable_matlab_norm"] = True
    kwargs["enable_detrending"] = kwargs.get("enable_detrending", True)
    kwargs["threshold_method"] = "baseline"

    return run_complete_hdf5_compatible_analysis(merged_results, **kwargs)


def compute_roi_thresholds_unified(
    merged_results: Dict[int, List[Tuple[float, float]]],
    threshold_method: str,
    **kwargs,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]]]:
    """
    BACKWARDS COMPATIBILITY: Unified function that now routes to modular system.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        threshold_method: Method to use ('baseline', 'adaptive', 'calibration')
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (roi_thresholds, roi_statistics) - for backwards compatibility
    """
    try:
        # Try to use the new modular system
        from ._calc_integration import run_analysis_with_method

        print(
            f"🔄 Legacy threshold calculation: Routing '{threshold_method}' to new system"
        )

        results = run_analysis_with_method(merged_results, threshold_method, **kwargs)

        # Return in old format for backwards compatibility
        roi_thresholds = results["upper_thresholds"].copy()
        roi_statistics = results["roi_statistics"].copy()

        # Add compatibility data
        for roi in roi_statistics:
            roi_statistics[roi].update(
                {
                    "threshold": roi_thresholds.get(roi, 0.0),
                    "baseline_mean": results["baseline_means"].get(roi, 0.0),
                    "upper_threshold": results["upper_thresholds"].get(roi, 0.0),
                    "lower_threshold": results["lower_thresholds"].get(roi, 0.0),
                    "uses_hysteresis": True,
                }
            )

        return roi_thresholds, roi_statistics

    except ImportError:
        # Fallback to baseline only if new system not available
        if threshold_method != "baseline":
            print(
                f"⚠️ Method '{threshold_method}' not available, falling back to baseline"
            )

        results = run_baseline_analysis(merged_results, **kwargs)

        # Return in old format
        roi_thresholds = results["upper_thresholds"].copy()
        roi_statistics = results["roi_statistics"].copy()

        for roi in roi_statistics:
            roi_statistics[roi].update(
                {
                    "threshold": roi_thresholds.get(roi, 0.0),
                    "baseline_mean": results["baseline_means"].get(roi, 0.0),
                    "upper_threshold": results["upper_thresholds"].get(roi, 0.0),
                    "lower_threshold": results["lower_thresholds"].get(roi, 0.0),
                    "uses_hysteresis": True,
                }
            )

        return roi_thresholds, roi_statistics


def compute_roi_thresholds_hysteresis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    threshold_method: str,
    **kwargs,
) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, Dict[str, Any]]
]:
    """
    BACKWARDS COMPATIBILITY: Hysteresis function that now only supports baseline method.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        threshold_method: Method to use (only 'baseline' supported in this module)
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (roi_baseline_means, roi_upper_thresholds, roi_lower_thresholds, roi_statistics)
    """
    if threshold_method != "baseline":
        raise ValueError(
            f"Only 'baseline' method is supported in this module. "
            f"For '{threshold_method}' method, use _calc_{threshold_method}.py"
        )

    # Run baseline analysis
    results = run_baseline_analysis(merged_results, **kwargs)

    return (
        results["baseline_means"],
        results["upper_thresholds"],
        results["lower_thresholds"],
        results["roi_statistics"],
    )


def define_movement_from_thresholds(
    merged_results: Dict[int, List[Tuple[float, float]]],
    roi_thresholds: Dict[int, float],
) -> Dict[int, List[Tuple[float, int]]]:
    """
    BACKWARDS COMPATIBILITY: Movement detection using only upper thresholds.

    This function is kept for backwards compatibility but recommends using hysteresis.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        roi_thresholds: Dictionary mapping ROI ID to threshold values

    Returns:
        Dictionary mapping ROI ID to list of (time, movement_binary) tuples
    """
    print("⚠️  Using legacy movement detection without hysteresis!")
    print("    Recommend using define_movement_with_hysteresis() for better results")

    movement_data = {}

    for roi, data in merged_results.items():
        if roi not in roi_thresholds:
            movement_data[roi] = []
            continue

        threshold = roi_thresholds[roi]
        movement_data[roi] = [(t, 1 if val > threshold else 0) for t, val in data]

    return movement_data


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================


def integrate_baseline_analysis_with_widget(widget) -> bool:
    """
    Integration function for baseline analysis with napari widget.

    Args:
        widget: The napari widget instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if we have merged_results
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message("❌ No merged_results available for baseline analysis")
            return False

        widget._log_message(
            f"✅ Found merged_results: {len(widget.merged_results)} ROIs"
        )

        # Get parameters from widget
        try:
            frame_interval = widget.frame_interval.value()
            baseline_duration_minutes = widget.baseline_duration_minutes.value()
            threshold_multiplier = widget.threshold_multiplier.value()
            enable_detrending = widget.enable_detrending.isChecked()
            enable_jump_correction = widget.enable_jump_correction.isChecked()

            widget._log_message(
                f"📊 Parameters: frame_interval={frame_interval}, baseline={baseline_duration_minutes}min"
            )

        except Exception as e:
            widget._log_message(f"❌ Error extracting parameters: {e}")
            return False

        widget._log_message("🚀 Running baseline analysis pipeline...")

        # Calculate threshold block count
        threshold_block_count = int((baseline_duration_minutes * 60) / frame_interval)
        widget._log_message(f"📊 Threshold block count: {threshold_block_count} frames")

        # Run baseline analysis
        try:
            baseline_results = run_baseline_analysis(
                merged_results=widget.merged_results,
                enable_matlab_norm=True,
                enable_detrending=enable_detrending,
                use_improved_detrending=True,
                threshold_block_count=threshold_block_count,
                multiplier=threshold_multiplier,
                frame_interval=frame_interval,
                enable_jump_correction=enable_jump_correction,
            )

            widget._log_message(f"✅ Baseline analysis completed successfully")

        except Exception as e:
            widget._log_message(f"❌ Baseline analysis failed: {e}")
            import traceback

            widget._log_message(f"Traceback: {traceback.format_exc()}")
            return False

        # Check if we got results
        if not baseline_results or "baseline_means" not in baseline_results:
            widget._log_message("❌ Baseline analysis returned no results")
            return False

        widget._log_message(
            f"📊 Analysis returned {len(baseline_results.get('baseline_means', {}))} ROI results"
        )

        # Update widget with baseline results
        try:
            widget.merged_results = baseline_results.get(
                "processed_data", widget.merged_results
            )
            widget.roi_baseline_means = baseline_results.get("baseline_means", {})
            widget.roi_upper_thresholds = baseline_results.get("upper_thresholds", {})
            widget.roi_lower_thresholds = baseline_results.get("lower_thresholds", {})
            widget.roi_statistics = baseline_results.get("roi_statistics", {})
            widget.movement_data = baseline_results.get("movement_data", {})
            widget.fraction_data = baseline_results.get("fraction_data", {})
            widget.quiescence_data = baseline_results.get("quiescence_data", {})
            widget.sleep_data = baseline_results.get("sleep_data", {})

            widget._log_message("✅ Updated widget with baseline results")

        except Exception as e:
            widget._log_message(f"❌ Error updating widget: {e}")
            return False

        # Calculate band widths for compatibility
        try:
            widget.roi_band_widths = {}
            for roi in widget.roi_baseline_means:
                if (
                    roi in widget.roi_upper_thresholds
                    and roi in widget.roi_lower_thresholds
                ):
                    upper = widget.roi_upper_thresholds[roi]
                    lower = widget.roi_lower_thresholds[roi]
                    widget.roi_band_widths[roi] = (upper - lower) / 2

            widget._log_message(
                f"✅ Calculated band widths for {len(widget.roi_band_widths)} ROIs"
            )

        except Exception as e:
            widget._log_message(f"⚠️ Could not calculate band widths: {e}")
            widget.roi_band_widths = {}

        widget._log_message("🎉 Baseline analysis integration completed successfully")
        return True

    except Exception as e:
        widget._log_message(f"❌ Baseline analysis integration failed: {str(e)}")
        import traceback

        widget._log_message(f"Traceback: {traceback.format_exc()}")
        return False


def test_baseline_analysis_direct(
    merged_results: Dict[int, List[Tuple[float, float]]],
) -> bool:
    """
    Test function to verify baseline analysis works with minimal parameters.

    Args:
        merged_results: Raw data to test with

    Returns:
        bool: True if successful, False otherwise
    """
    print("🧪 === TESTING BASELINE ANALYSIS DIRECTLY ===")

    if not merged_results:
        print("❌ No input data provided")
        return False

    print(f"✅ Input data: {len(merged_results)} ROIs")

    # Test with minimal parameters
    test_params = {
        "enable_matlab_norm": True,
        "enable_detrending": True,
        "threshold_block_count": 10,  # Very small for testing
        "multiplier": 1.0,
        "frame_interval": 5.0,
    }

    try:
        print("🚀 Running run_baseline_analysis with minimal parameters...")
        results = run_baseline_analysis(merged_results, **test_params)

        if results:
            print(f"✅ SUCCESS: Got results with keys: {list(results.keys())}")

            # Check required components
            required_keys = [
                "baseline_means",
                "upper_thresholds",
                "lower_thresholds",
                "movement_data",
            ]
            all_good = True

            for key in required_keys:
                if key in results and results[key]:
                    print(f"  ✅ {key}: {len(results[key])} ROIs")
                else:
                    print(f"  ❌ {key}: Missing or empty")
                    all_good = False

            return all_good
        else:
            print("❌ FAILED: No results returned")
            return False

    except Exception as e:
        print(f"❌ FAILED: Exception during baseline analysis: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False
