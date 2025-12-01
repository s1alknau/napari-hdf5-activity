# """
# _calc.py - Core baseline analysis calculations

# This module contains ONLY baseline-specific functions and core utilities.
# Other methods are in separate modules:
# - _calc_adaptive.py: Adaptive threshold calculation
# - _calc_calibration.py: Calibration-based threshold calculation
# - _calc_integration.py: Method routing and integration
# """

# import os
# import time
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Any


# # =============================================================================
# # CORE PREPROCESSING FUNCTIONS
# # =============================================================================

# def apply_matlab_normalization_to_merged_results(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     enable_matlab_norm: bool = True
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """Apply MATLAB-style normalization: subtract minimum per ROI."""
#     if not enable_matlab_norm:
#         return merged_results

#     normalized_results = {}

#     for roi, data in merged_results.items():
#         if not data:
#             normalized_results[roi] = []
#             continue

#         times = [t for t, _ in data]
#         intensities = np.array([val for _, val in data])

#         # MATLAB logic: subtract minimum per ROI
#         min_intensity = np.min(intensities)
#         normalized_intensities = intensities - min_intensity

#         normalized_results[roi] = list(zip(times, normalized_intensities))

#     return normalized_results


# def improved_full_dataset_detrending(
#     merged_results: Dict[int, List[Tuple[float, float]]]
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """Apply improved detrending to complete dataset."""
#     detrended_results = {}

#     for roi, data in merged_results.items():
#         if not data or len(data) < 20:
#             detrended_results[roi] = data
#             continue

#         try:
#             sorted_data = sorted(data, key=lambda x: x[0])
#             times = np.array([t for t, _ in sorted_data])
#             values = np.array([val for _, val in sorted_data])

#             # Remove polynomial trend (handles curved drift)
#             if len(values) >= 10:
#                 poly_coeffs = np.polyfit(times, values, 2)
#                 poly_trend = np.polyval(poly_coeffs, times)
#                 values_detrended = values - poly_trend + np.mean(poly_trend)
#             else:
#                 values_detrended = values

#             # Remove any remaining linear drift
#             if len(values_detrended) >= 10:
#                 slope, intercept = np.polyfit(times, values_detrended, 1)
#                 total_drift = abs(slope * (times[-1] - times[0]))
#                 drift_percentage = (total_drift / np.mean(values)) * 100 if np.mean(values) > 0 else 0

#                 if drift_percentage > 1.0:  # Only remove if > 1% drift
#                     linear_trend = slope * times + intercept
#                     values_final = values_detrended - (linear_trend - intercept)
#                 else:
#                     values_final = values_detrended
#             else:
#                 values_final = values_detrended

#             detrended_results[roi] = list(zip(times, values_final))

#         except Exception as e:
#             print(f"Detrending failed for ROI {roi}: {e}")
#             detrended_results[roi] = data

#     return detrended_results


# # =============================================================================
# # BASELINE THRESHOLD CALCULATION
# # =============================================================================

# def compute_threshold_baseline_hysteresis(
#     data: List[Tuple[float, float]],
#     baseline_duration_minutes: float,
#     multiplier: float = 1.0,
#     frame_interval: float = 5.0,
#     **kwargs  # For backward compatibility
# ) -> Tuple[float, float, float, Dict[str, Any]]:
#     """Compute hysteresis thresholds using baseline method."""

#     if not data:
#         return 0.0, 0.0, 0.0, {'method': 'baseline_hysteresis', 'status': 'no_data'}

#     # Sort data by time
#     sorted_data = sorted(data, key=lambda x: x[0])

#     # Calculate baseline time range
#     baseline_duration_seconds = baseline_duration_minutes * 60
#     start_time = sorted_data[0][0]
#     end_time = start_time + baseline_duration_seconds

#     # Select baseline data
#     baseline_data = [(t, v) for t, v in sorted_data if start_time <= t < end_time]

#     # Check minimum data requirement
#     min_required_frames = max(10, int(baseline_duration_seconds / frame_interval * 0.8))
#     if len(baseline_data) < min_required_frames:
#         return 0.0, 0.0, 0.0, {
#             'method': 'baseline_hysteresis',
#             'status': 'insufficient_data',
#             'found_frames': len(baseline_data),
#             'required_frames': min_required_frames
#         }

#     # Calculate statistics
#     times = np.array([t for t, _ in baseline_data])
#     values = np.array([val for _, val in baseline_data])

#     mean_val = np.mean(values)
#     std_val = np.std(values)

#     # Calculate hysteresis thresholds
#     baseline_mean = mean_val
#     threshold_band = multiplier * std_val
#     upper_threshold = baseline_mean + threshold_band
#     lower_threshold = max(0, baseline_mean - threshold_band)  # Ensure non-negative

#     # Validate thresholds
#     if np.isnan(upper_threshold) or np.isinf(upper_threshold):
#         upper_threshold = np.percentile(values, 75)
#         lower_threshold = np.percentile(values, 25)
#         baseline_mean = np.median(values)

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
#         'baseline_duration_minutes': baseline_duration_minutes,
#         'frame_interval': frame_interval,
#         'data_range': (np.min(values), np.max(values)),
#         'status': 'calculated_from_preprocessed_data'
#     }

#     return baseline_mean, upper_threshold, lower_threshold, statistics


# # =============================================================================
# # HYSTERESIS MOVEMENT DETECTION
# # =============================================================================

# def define_movement_with_hysteresis(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     roi_baseline_means: Dict[int, float],
#     roi_upper_thresholds: Dict[int, float],
#     roi_lower_thresholds: Dict[int, float]
# ) -> Dict[int, List[Tuple[float, int]]]:
#     """Define movement using hysteresis logic to prevent threshold flicker."""

#     movement_data = {}

#     for roi, data in merged_results.items():
#         if roi not in roi_upper_thresholds or roi not in roi_lower_thresholds:
#             movement_data[roi] = []
#             continue

#         upper_thresh = roi_upper_thresholds[roi]
#         lower_thresh = roi_lower_thresholds[roi]
#         baseline = roi_baseline_means[roi]

#         sorted_data = sorted(data, key=lambda x: x[0])

#         if not sorted_data:
#             movement_data[roi] = []
#             continue

#         # Determine initial state
#         first_value = sorted_data[0][1]
#         if first_value > upper_thresh:
#             current_movement_state = 1
#         elif first_value < lower_thresh:
#             current_movement_state = 0
#         else:
#             current_movement_state = 1 if first_value > baseline else 0

#         roi_movement = []

#         for time_point, value in sorted_data:
#             # Hysteresis logic
#             if current_movement_state == 0:  # Currently: No Movement
#                 if value > upper_thresh:
#                     current_movement_state = 1  # Switch to Movement
#             else:  # Currently: Movement
#                 if value < lower_thresh:
#                     current_movement_state = 0  # Switch to No Movement

#             roi_movement.append((time_point, current_movement_state))

#         movement_data[roi] = roi_movement

#     return movement_data


# # =============================================================================
# # BEHAVIORAL ANALYSIS FUNCTIONS
# # =============================================================================

# def bin_fraction_movement(
#     movement_data: Dict[int, List[Tuple[float, int]]],
#     bin_size_seconds: int,
#     frame_interval: float
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """Calculate fraction movement using hysteresis state data."""

#     fraction_data = {}

#     for roi, data in movement_data.items():
#         if not data:
#             fraction_data[roi] = []
#             continue

#         sorted_data = sorted(data, key=lambda x: x[0])

#         if len(sorted_data) < 2:
#             fraction_data[roi] = []
#             continue

#         start_time = sorted_data[0][0]
#         end_time = sorted_data[-1][0]

#         # Create time bins
#         first_bin_start = (start_time // bin_size_seconds) * bin_size_seconds
#         bin_edges = []
#         current_bin_start = first_bin_start
#         while current_bin_start < end_time:
#             bin_edges.append(current_bin_start)
#             current_bin_start += bin_size_seconds
#         bin_edges.append(current_bin_start)

#         roi_fractions = []

#         for i in range(len(bin_edges) - 1):
#             bin_start = bin_edges[i]
#             bin_end = bin_edges[i + 1]
#             bin_center = (bin_start + bin_end) / 2
#             bin_duration = bin_end - bin_start

#             # Calculate time spent in movement state
#             movement_time = 0.0

#             for j in range(len(sorted_data)):
#                 current_time = sorted_data[j][0]
#                 current_state = sorted_data[j][1]

#                 # Determine when this state ends
#                 next_time = sorted_data[j + 1][0] if j + 1 < len(sorted_data) else end_time

#                 # Check overlap with current bin
#                 state_start = max(current_time, bin_start)
#                 state_end = min(next_time, bin_end)

#                 if state_start < state_end and current_state == 1:
#                     movement_time += (state_end - state_start)

#             fraction_movement = movement_time / bin_duration if bin_duration > 0 else 0.0
#             fraction_movement = max(0.0, min(1.0, fraction_movement))

#             roi_fractions.append((bin_center, fraction_movement))

#         fraction_data[roi] = roi_fractions

#     return fraction_data


# def bin_quiescence(
#     fraction_data: Dict[int, List[Tuple[float, float]]],
#     quiescence_threshold: float = 0.5
# ) -> Dict[int, List[Tuple[float, int]]]:
#     """Calculate quiescence: 1 = quiescent (low movement), 0 = active (high movement)."""

#     quiescence_data = {}

#     for roi, data in fraction_data.items():
#         quiescent_roi_data = []

#         for time_point, fraction_movement in data:
#             # Quiescent when movement is LOW
#             quiescence_state = 1 if fraction_movement < quiescence_threshold else 0
#             quiescent_roi_data.append((time_point, quiescence_state))

#         quiescence_data[roi] = quiescent_roi_data

#     return quiescence_data


# def define_sleep_periods(
#     quiescence_data: Dict[int, List[Tuple[float, int]]],
#     sleep_threshold_minutes: int = 8,
#     bin_size_seconds: int = 60
# ) -> Dict[int, List[Tuple[float, int]]]:
#     """Define sleep as sustained quiescence periods."""

#     sleep_data = {}
#     min_bins_for_sleep = (sleep_threshold_minutes * 60) // bin_size_seconds

#     for roi, data in quiescence_data.items():
#         if not data:
#             sleep_data[roi] = []
#             continue

#         times = np.array([t for t, _ in data])
#         quiescence_states = np.array([q for _, q in data])

#         sleep_state = np.zeros_like(quiescence_states)

#         i = 0
#         while i < len(quiescence_states):
#             if quiescence_states[i] == 1:  # Start of quiescent period
#                 consecutive_count = 0
#                 j = i
#                 while j < len(quiescence_states) and quiescence_states[j] == 1:
#                     consecutive_count += 1
#                     j += 1

#                 # Mark as sleep if long enough
#                 if consecutive_count >= min_bins_for_sleep:
#                     sleep_state[i:j] = 1

#                 i = j
#             else:
#                 i += 1

#         sleep_data[roi] = list(zip(times, sleep_state))

#     return sleep_data


# def bin_activity_data_for_lighting(
#     fraction_data: Dict[int, List[Tuple[float, float]]],
#     bin_minutes: int = 30
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """Bin activity data for circadian/lighting analysis."""

#     bin_size_seconds = bin_minutes * 60
#     binned_data = {}

#     for roi, data in fraction_data.items():
#         if not data:
#             binned_data[roi] = []
#             continue

#         sorted_data = sorted(data, key=lambda x: x[0])

#         if len(sorted_data) < 2:
#             binned_data[roi] = []
#             continue

#         start_time = sorted_data[0][0]
#         end_time = sorted_data[-1][0]

#         first_hour_start = (start_time // 3600) * 3600

#         binned_roi_data = []
#         current_time = first_hour_start

#         while current_time < end_time:
#             bin_end = current_time + bin_size_seconds

#             bin_data = [val for t, val in sorted_data if current_time <= t < bin_end]

#             if bin_data:
#                 avg_activity = np.mean(bin_data)
#                 bin_center = current_time + (bin_size_seconds / 2)
#                 binned_roi_data.append((bin_center, avg_activity))

#             current_time = bin_end

#         binned_data[roi] = binned_roi_data

#     return binned_data


# # =============================================================================
# # MAIN BASELINE ANALYSIS FUNCTION
# # =============================================================================

# def run_baseline_analysis(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     enable_matlab_norm: bool = True,
#     enable_detrending: bool = True,
#     use_improved_detrending: bool = True,
#     baseline_duration_minutes: float = 200.0,
#     multiplier: float = 1.0,
#     frame_interval: float = 5.0,
#     **kwargs
# ) -> Dict[str, Any]:
#     """Run complete baseline analysis pipeline."""

#     analysis_results = {
#         'method': 'baseline',
#         'parameters': {
#             'enable_matlab_norm': enable_matlab_norm,
#             'enable_detrending': enable_detrending,
#             'baseline_duration_minutes': baseline_duration_minutes,
#             'multiplier': multiplier,
#             'frame_interval': frame_interval
#         }
#     }

#     # Step 1: Preprocessing
#     if enable_matlab_norm:
#         normalized_data = apply_matlab_normalization_to_merged_results(merged_results)
#     else:
#         normalized_data = merged_results

#     if enable_detrending and use_improved_detrending:
#         processed_data = improved_full_dataset_detrending(normalized_data)
#     else:
#         processed_data = normalized_data

#     analysis_results['processed_data'] = processed_data

#     # Step 2: Baseline threshold calculation
#     baseline_means = {}
#     upper_thresholds = {}
#     lower_thresholds = {}
#     roi_statistics = {}

#     for roi, data in processed_data.items():
#         if not data:
#             baseline_means[roi] = 0.0
#             upper_thresholds[roi] = 0.0
#             lower_thresholds[roi] = 0.0
#             roi_statistics[roi] = {'method': 'baseline', 'status': 'no_data'}
#             continue

#         baseline_mean, upper_thresh, lower_thresh, stats = compute_threshold_baseline_hysteresis(
#             data, baseline_duration_minutes, multiplier, frame_interval
#         )

#         baseline_means[roi] = baseline_mean
#         upper_thresholds[roi] = upper_thresh
#         lower_thresholds[roi] = lower_thresh
#         roi_statistics[roi] = stats

#     analysis_results.update({
#         'baseline_means': baseline_means,
#         'upper_thresholds': upper_thresholds,
#         'lower_thresholds': lower_thresholds,
#         'roi_statistics': roi_statistics
#     })

#     # Step 3: Movement detection
#     movement_data = define_movement_with_hysteresis(
#         processed_data, baseline_means, upper_thresholds, lower_thresholds
#     )
#     analysis_results['movement_data'] = movement_data

#     # Step 4: Behavioral analysis
#     bin_size_seconds = kwargs.get('bin_size_seconds', 60)
#     fraction_data = bin_fraction_movement(movement_data, bin_size_seconds, frame_interval)
#     analysis_results['fraction_data'] = fraction_data

#     quiescence_threshold = kwargs.get('quiescence_threshold', 0.5)
#     quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)
#     analysis_results['quiescence_data'] = quiescence_data

#     sleep_threshold_minutes = kwargs.get('sleep_threshold_minutes', 8)
#     sleep_data = define_sleep_periods(quiescence_data, sleep_threshold_minutes, bin_size_seconds)
#     analysis_results['sleep_data'] = sleep_data

#     # Add ROI colors
#     try:
#         from ._reader import get_roi_colors
#         roi_colors = get_roi_colors(sorted(processed_data.keys()))
#     except:
#         roi_colors = {roi: f'C{i}' for i, roi in enumerate(sorted(processed_data.keys()))}

#     analysis_results['roi_colors'] = roi_colors

#     return analysis_results


# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
#     """Calculate performance metrics."""
#     try:
#         import psutil
#         elapsed_time = time.time() - start_time
#         fps = total_frames / elapsed_time if elapsed_time > 0 else 0
#         cpu_percent = psutil.cpu_percent(interval=None)
#         memory_percent = psutil.virtual_memory().percent

#         return {
#             'elapsed_time': elapsed_time,
#             'fps': fps,
#             'cpu_percent': cpu_percent,
#             'memory_percent': memory_percent,
#             'total_frames': total_frames
#         }
#     except ImportError:
#         elapsed_time = time.time() - start_time
#         return {
#             'elapsed_time': elapsed_time,
#             'fps': total_frames / elapsed_time if elapsed_time > 0 else 0,
#             'cpu_percent': 0,
#             'memory_percent': 0,
#             'total_frames': total_frames
#         }


# # =============================================================================
# # BACKWARD COMPATIBILITY FUNCTIONS
# # =============================================================================
# # =============================================================================
# # PURE ANALYSIS FUNCTIONS (for centralized preprocessing)
# # =============================================================================

# def run_baseline_analysis_pure(
#     preprocessed_data: Dict[int, List[Tuple[float, float]]],
#     **kwargs
# ) -> Dict[str, Any]:
#     """
#     Pure baseline analysis function that works on already-preprocessed data.
#     No internal preprocessing - used by centralized integration pipeline.

#     Args:
#         preprocessed_data: Already preprocessed data (normalized, detrended, etc.)
#         **kwargs: Analysis parameters

#     Returns:
#         Analysis results dictionary
#     """

#     analysis_results = {
#         'method': 'baseline',
#         'parameters': {
#             'baseline_duration_minutes': kwargs.get('baseline_duration_minutes', 200.0),
#             'multiplier': kwargs.get('multiplier', 1.0),
#             'frame_interval': kwargs.get('frame_interval', 5.0),
#             'preprocessing_skipped': True  # Indicates this is the pure version
#         }
#     }

#     # Use the preprocessed data directly (no internal preprocessing)
#     processed_data = preprocessed_data
#     analysis_results['processed_data'] = processed_data

#     # Step 2: Baseline threshold calculation
#     baseline_means = {}
#     upper_thresholds = {}
#     lower_thresholds = {}
#     roi_statistics = {}

#     for roi, data in processed_data.items():
#         if not data:
#             baseline_means[roi] = 0.0
#             upper_thresholds[roi] = 0.0
#             lower_thresholds[roi] = 0.0
#             roi_statistics[roi] = {'method': 'baseline', 'status': 'no_data'}
#             continue

#         baseline_mean, upper_thresh, lower_thresh, stats = compute_threshold_baseline_hysteresis(
#             data,
#             kwargs.get('baseline_duration_minutes', 200.0),
#             kwargs.get('multiplier', 1.0),
#             kwargs.get('frame_interval', 5.0)
#         )

#         baseline_means[roi] = baseline_mean
#         upper_thresholds[roi] = upper_thresh
#         lower_thresholds[roi] = lower_thresh
#         roi_statistics[roi] = stats

#     analysis_results.update({
#         'baseline_means': baseline_means,
#         'upper_thresholds': upper_thresholds,
#         'lower_thresholds': lower_thresholds,
#         'roi_statistics': roi_statistics
#     })

#     # Step 3: Movement detection
#     movement_data = define_movement_with_hysteresis(
#         processed_data, baseline_means, upper_thresholds, lower_thresholds
#     )
#     analysis_results['movement_data'] = movement_data

#     # Step 4: Behavioral analysis
#     bin_size_seconds = kwargs.get('bin_size_seconds', 60)
#     frame_interval = kwargs.get('frame_interval', 5.0)
#     fraction_data = bin_fraction_movement(movement_data, bin_size_seconds, frame_interval)
#     analysis_results['fraction_data'] = fraction_data

#     quiescence_threshold = kwargs.get('quiescence_threshold', 0.5)
#     quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)
#     analysis_results['quiescence_data'] = quiescence_data

#     sleep_threshold_minutes = kwargs.get('sleep_threshold_minutes', 8)
#     sleep_data = define_sleep_periods(quiescence_data, sleep_threshold_minutes, bin_size_seconds)
#     analysis_results['sleep_data'] = sleep_data

#     # Add ROI colors
#     try:
#         from ._reader import get_roi_colors
#         roi_colors = get_roi_colors(sorted(processed_data.keys()))
#     except:
#         roi_colors = {roi: f'C{i}' for i, roi in enumerate(sorted(processed_data.keys()))}

#     analysis_results['roi_colors'] = roi_colors

#     return analysis_results

# def integrate_baseline_analysis_with_widget(widget) -> bool:
#     """Integration function for baseline analysis with napari widget."""
#     try:
#         if not hasattr(widget, 'merged_results') or not widget.merged_results:
#             widget._log_message("No merged_results available for baseline analysis")
#             return False

#         # Extract parameters
#         frame_interval = widget.frame_interval.value()
#         baseline_duration_minutes = widget.baseline_duration_minutes.value()
#         threshold_multiplier = widget.threshold_multiplier.value()
#         enable_detrending = widget.enable_detrending.isChecked()

#         # Run analysis
#         baseline_results = run_baseline_analysis(
#             merged_results=widget.merged_results,
#             enable_matlab_norm=True,
#             enable_detrending=enable_detrending,
#             baseline_duration_minutes=baseline_duration_minutes,
#             multiplier=threshold_multiplier,
#             frame_interval=frame_interval
#         )

#         # Update widget
#         widget.merged_results = baseline_results.get('processed_data', widget.merged_results)
#         widget.roi_baseline_means = baseline_results.get('baseline_means', {})
#         widget.roi_upper_thresholds = baseline_results.get('upper_thresholds', {})
#         widget.roi_lower_thresholds = baseline_results.get('lower_thresholds', {})
#         widget.roi_statistics = baseline_results.get('roi_statistics', {})
#         widget.movement_data = baseline_results.get('movement_data', {})
#         widget.fraction_data = baseline_results.get('fraction_data', {})
#         widget.quiescence_data = baseline_results.get('quiescence_data', {})
#         widget.sleep_data = baseline_results.get('sleep_data', {})

#         # Calculate band widths
#         widget.roi_band_widths = {}
#         for roi in widget.roi_baseline_means:
#             if roi in widget.roi_upper_thresholds and roi in widget.roi_lower_thresholds:
#                 upper = widget.roi_upper_thresholds[roi]
#                 lower = widget.roi_lower_thresholds[roi]
#                 widget.roi_band_widths[roi] = (upper - lower) / 2

#         return True

#     except Exception as e:
#         widget._log_message(f"Baseline analysis integration failed: {str(e)}")
#         return False


# # Legacy aliases for backward compatibility
# run_complete_hdf5_compatible_analysis = run_baseline_analysis
# test_baseline_analysis_direct = lambda merged_results: bool(run_baseline_analysis(merged_results))
"""
_calc.py - Core baseline analysis calculations

This module contains ONLY baseline-specific functions and core utilities.
Other methods are in separate modules:
- _calc_adaptive.py: Adaptive threshold calculation
- _calc_calibration.py: Calibration-based threshold calculation
- _calc_integration.py: Method routing and integration
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any


# =============================================================================
# CORE PREPROCESSING FUNCTIONS
# =============================================================================


def apply_matlab_normalization_to_merged_results(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_matlab_norm: bool = True,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Apply true MATLAB-style processing: NO minimum subtraction like real MATLAB.

    MATLAB code does: im2double(rgb2gray(frame)) and direct frame differences.
    MATLAB does NOT subtract minimum values - it only handles pixel range conversion.
    Since our data is already frame differences, we return as-is to match MATLAB behavior.

    Real MATLAB: framePixelChange = sum(sum(abs(frameLast{1,n} - frame)));
    No minimum subtraction in MATLAB processing!
    """
    if not enable_matlab_norm:
        return merged_results

    # True MATLAB behavior: no minimum subtraction
    # Our data is already frame differences (like MATLAB's framePixelChange)
    return merged_results


def detect_and_remove_jumps(
    times: np.ndarray, values: np.ndarray, jump_threshold_factor: float = 1.5
) -> Tuple[np.ndarray, List[int]]:
    """
    Detect and correct sudden jumps in time-series data.

    Identifies abrupt changes (jumps) in the signal by comparing frame-to-frame
    differences against a rolling standard deviation threshold. Corrects jumps
    by subtracting the jump magnitude from all subsequent values.

    Args:
        times: Time array (not currently used, kept for API compatibility)
        values: Value array
        jump_threshold_factor: Factor for jump detection threshold (default: 1.5)
                              Lower values = more sensitive detection

    Returns:
        Tuple of (corrected_values, jump_indices)
    """
    if len(values) < 10:
        return values, []

    # Use smaller window for more sensitive detection
    window_size = min(20, len(values) // 5)
    if window_size < 5:
        return values, []

    # Calculate frame-to-frame differences
    diffs = np.diff(values)

    # Calculate rolling standard deviation of differences
    rolling_std = []
    for i in range(len(diffs)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(diffs), i + window_size // 2 + 1)
        window_diffs = diffs[start_idx:end_idx]
        rolling_std.append(np.std(window_diffs))

    rolling_std = np.array(rolling_std)

    # Detect jumps using threshold based on median rolling std
    jump_threshold = jump_threshold_factor * np.median(rolling_std)
    jump_indices = np.where(np.abs(diffs) > jump_threshold)[0]

    if len(jump_indices) == 0:
        return values, []

    # Correct jumps by adjusting subsequent values
    corrected_values = values.copy()

    for jump_idx in jump_indices:
        jump_size = diffs[jump_idx]
        # Subtract the jump from all subsequent values
        corrected_values[jump_idx + 1 :] -= jump_size

    return corrected_values, list(jump_indices)


def improved_full_dataset_detrending(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_jump_correction: bool = False,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Apply improved detrending to complete dataset.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        enable_jump_correction: Whether to apply jump correction before detrending

    Returns:
        Dictionary with detrended values
    """
    detrended_results = {}

    for roi, data in merged_results.items():
        if not data or len(data) < 20:
            detrended_results[roi] = data
            continue

        try:
            sorted_data = sorted(data, key=lambda x: x[0])
            times = np.array([t for t, _ in sorted_data])
            values = np.array([val for _, val in sorted_data])

            # Step 1: Jump correction (if enabled)
            if enable_jump_correction:
                values, jump_indices = detect_and_remove_jumps(times, values)
                if len(jump_indices) > 0:
                    print(f"ROI {roi}: Corrected {len(jump_indices)} jumps")

            # Step 2: Remove polynomial trend (handles curved drift)
            if len(values) >= 10:
                poly_coeffs = np.polyfit(times, values, 2)
                poly_trend = np.polyval(poly_coeffs, times)
                values_detrended = values - poly_trend + np.mean(poly_trend)
            else:
                values_detrended = values

            # Step 3: Remove any remaining linear drift
            if len(values_detrended) >= 10:
                slope, intercept = np.polyfit(times, values_detrended, 1)
                total_drift = abs(slope * (times[-1] - times[0]))
                drift_percentage = (
                    (total_drift / np.mean(values)) * 100 if np.mean(values) > 0 else 0
                )

                if drift_percentage > 1.0:  # Only remove if > 1% drift
                    linear_trend = slope * times + intercept
                    values_final = values_detrended - (linear_trend - intercept)
                else:
                    values_final = values_detrended
            else:
                values_final = values_detrended

            detrended_results[roi] = list(zip(times, values_final))

        except Exception as e:
            print(f"Detrending failed for ROI {roi}: {e}")
            detrended_results[roi] = data

    return detrended_results


# =============================================================================
# BASELINE THRESHOLD CALCULATION
# =============================================================================


def compute_threshold_baseline_hysteresis(
    data: List[Tuple[float, float]],
    baseline_duration_minutes: float,
    multiplier: float = 1.0,
    frame_interval: float = 5.0,
    **kwargs,  # For backward compatibility
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Compute hysteresis thresholds using baseline method."""

    if not data:
        return 0.0, 0.0, 0.0, {"method": "baseline_hysteresis", "status": "no_data"}

    # Sort data by time
    sorted_data = sorted(data, key=lambda x: x[0])

    # Calculate baseline time range
    baseline_duration_seconds = baseline_duration_minutes * 60
    start_time = sorted_data[0][0]
    end_time = start_time + baseline_duration_seconds

    # Select baseline data
    baseline_data = [(t, v) for t, v in sorted_data if start_time <= t < end_time]

    # Check minimum data requirement
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

    # Calculate statistics
    values = np.array([val for _, val in baseline_data])

    mean_val = np.mean(values)
    std_val = np.std(values)

    # Calculate hysteresis thresholds
    baseline_mean = mean_val
    threshold_band = multiplier * std_val
    upper_threshold = baseline_mean + threshold_band
    lower_threshold = baseline_mean - threshold_band  # ← ENTFERNE max(0, ...)

    # Validate thresholds
    if np.isnan(upper_threshold) or np.isinf(upper_threshold):
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
        "frame_interval": frame_interval,
        "data_range": (np.min(values), np.max(values)),
        "status": "calculated_from_preprocessed_data",
    }

    return baseline_mean, upper_threshold, lower_threshold, statistics


# =============================================================================
# HYSTERESIS MOVEMENT DETECTION
# =============================================================================


def define_movement_with_hysteresis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    roi_baseline_means: Dict[int, float],
    roi_upper_thresholds: Dict[int, float],
    roi_lower_thresholds: Dict[int, float],
) -> Dict[int, List[Tuple[float, int]]]:
    """Define movement using hysteresis logic to prevent threshold flicker."""

    movement_data = {}

    for roi, data in merged_results.items():
        if roi not in roi_upper_thresholds or roi not in roi_lower_thresholds:
            movement_data[roi] = []
            continue

        upper_thresh = roi_upper_thresholds[roi]
        lower_thresh = roi_lower_thresholds[roi]
        baseline = roi_baseline_means[roi]

        sorted_data = sorted(data, key=lambda x: x[0])

        if not sorted_data:
            movement_data[roi] = []
            continue

        # Determine initial state
        first_value = sorted_data[0][1]
        if first_value > upper_thresh:
            current_movement_state = 1
        elif first_value < lower_thresh:
            current_movement_state = 0
        else:
            current_movement_state = 1 if first_value > baseline else 0

        roi_movement = []

        for time_point, value in sorted_data:
            # Hysteresis logic
            if current_movement_state == 0:  # Currently: No Movement
                if value > upper_thresh:
                    current_movement_state = 1  # Switch to Movement
            else:  # Currently: Movement
                if value < lower_thresh:
                    current_movement_state = 0  # Switch to No Movement

            roi_movement.append((time_point, current_movement_state))

        movement_data[roi] = roi_movement

    return movement_data


# =============================================================================
# BEHAVIORAL ANALYSIS FUNCTIONS
# =============================================================================


def bin_fraction_movement(
    movement_data: Dict[int, List[Tuple[float, int]]],
    bin_size_seconds: int,
    frame_interval: float,
) -> Dict[int, List[Tuple[float, float]]]:
    """Calculate fraction movement using hysteresis state data."""

    fraction_data = {}

    for roi, data in movement_data.items():
        if not data:
            fraction_data[roi] = []
            continue

        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            fraction_data[roi] = []
            continue

        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]

        # Create time bins
        first_bin_start = (start_time // bin_size_seconds) * bin_size_seconds
        bin_edges = []
        current_bin_start = first_bin_start
        while current_bin_start < end_time:
            bin_edges.append(current_bin_start)
            current_bin_start += bin_size_seconds
        bin_edges.append(current_bin_start)

        roi_fractions = []

        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2
            bin_duration = bin_end - bin_start

            # Calculate time spent in movement state
            movement_time = 0.0

            for j in range(len(sorted_data)):
                current_time = sorted_data[j][0]
                current_state = sorted_data[j][1]

                # Determine when this state ends
                next_time = (
                    sorted_data[j + 1][0] if j + 1 < len(sorted_data) else end_time
                )

                # Check overlap with current bin
                state_start = max(current_time, bin_start)
                state_end = min(next_time, bin_end)

                if state_start < state_end and current_state == 1:
                    movement_time += state_end - state_start

            fraction_movement = (
                movement_time / bin_duration if bin_duration > 0 else 0.0
            )
            fraction_movement = max(0.0, min(1.0, fraction_movement))

            roi_fractions.append((bin_center, fraction_movement))

        fraction_data[roi] = roi_fractions

    return fraction_data


def bin_quiescence(
    fraction_data: Dict[int, List[Tuple[float, float]]],
    quiescence_threshold: float = 0.5,
) -> Dict[int, List[Tuple[float, int]]]:
    """Calculate quiescence: 1 = quiescent (low movement), 0 = active (high movement)."""

    quiescence_data = {}

    for roi, data in fraction_data.items():
        quiescent_roi_data = []

        for time_point, fraction_movement in data:
            # Quiescent when movement is LOW
            quiescence_state = 1 if fraction_movement < quiescence_threshold else 0
            quiescent_roi_data.append((time_point, quiescence_state))

        quiescence_data[roi] = quiescent_roi_data

    return quiescence_data


def define_sleep_periods(
    quiescence_data: Dict[int, List[Tuple[float, int]]],
    sleep_threshold_minutes: int = 8,
    bin_size_seconds: int = 60,
) -> Dict[int, List[Tuple[float, int]]]:
    """Define sleep as sustained quiescence periods."""

    sleep_data = {}
    min_bins_for_sleep = (sleep_threshold_minutes * 60) // bin_size_seconds

    for roi, data in quiescence_data.items():
        if not data:
            sleep_data[roi] = []
            continue

        times = np.array([t for t, _ in data])
        quiescence_states = np.array([q for _, q in data])

        sleep_state = np.zeros_like(quiescence_states)

        i = 0
        while i < len(quiescence_states):
            if quiescence_states[i] == 1:  # Start of quiescent period
                consecutive_count = 0
                j = i
                while j < len(quiescence_states) and quiescence_states[j] == 1:
                    consecutive_count += 1
                    j += 1

                # Mark as sleep if long enough
                if consecutive_count >= min_bins_for_sleep:
                    sleep_state[i:j] = 1

                i = j
            else:
                i += 1

        sleep_data[roi] = list(zip(times, sleep_state))

    return sleep_data


def bin_activity_data_for_lighting(
    fraction_data: Dict[int, List[Tuple[float, float]]], bin_minutes: int = 30
) -> Dict[int, List[Tuple[float, float]]]:
    """Bin activity data for circadian/lighting analysis."""

    bin_size_seconds = bin_minutes * 60
    binned_data = {}

    for roi, data in fraction_data.items():
        if not data:
            binned_data[roi] = []
            continue

        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            binned_data[roi] = []
            continue

        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]

        first_hour_start = (start_time // 3600) * 3600

        binned_roi_data = []
        current_time = first_hour_start

        while current_time < end_time:
            bin_end = current_time + bin_size_seconds

            bin_data = [val for t, val in sorted_data if current_time <= t < bin_end]

            if bin_data:
                avg_activity = np.mean(bin_data)
                bin_center = current_time + (bin_size_seconds / 2)
                binned_roi_data.append((bin_center, avg_activity))

            current_time = bin_end

        binned_data[roi] = binned_roi_data

    return binned_data


# =============================================================================
# MULTIPROCESSING WORKER FUNCTION
# =============================================================================


def _process_single_roi_movement(
    args: Tuple[int, List[Tuple[float, float]], float, float, float, float, float],
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function for parallel ROI movement detection with pre-calculated baseline.

    Args:
        args: Tuple of (roi_id, data, baseline_mean, upper_threshold,
                       lower_threshold, bin_size_seconds, frame_interval)

    Returns:
        Tuple of (roi_id, results_dict)
    """
    (
        roi_id,
        data,
        baseline_mean,
        upper_threshold,
        lower_threshold,
        bin_size_seconds,
        frame_interval,
    ) = args

    results = {}

    if not data:
        results["movement_data"] = []
        results["fraction_data"] = []
        return roi_id, results

    try:
        # Step 1: Hysteresis movement detection using pre-calculated baselines
        baseline_means_single = {roi_id: baseline_mean}
        upper_thresholds_single = {roi_id: upper_threshold}
        lower_thresholds_single = {roi_id: lower_threshold}
        data_single = {roi_id: data}

        movement_data_dict = define_movement_with_hysteresis(
            data_single,
            baseline_means_single,
            upper_thresholds_single,
            lower_thresholds_single,
        )
        results["movement_data"] = movement_data_dict.get(roi_id, [])

        # Step 2: Bin fraction movement
        fraction_data_dict = bin_fraction_movement(
            movement_data_dict, bin_size_seconds, frame_interval
        )
        results["fraction_data"] = fraction_data_dict.get(roi_id, [])

    except Exception as e:
        results["error"] = str(e)
        results["movement_data"] = []
        results["fraction_data"] = []

    return roi_id, results


# =============================================================================
# MAIN BASELINE ANALYSIS FUNCTION (with integrated multiprocessing)
# =============================================================================


def run_baseline_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
    use_improved_detrending: bool = True,
    enable_jump_correction: bool = False,
    baseline_duration_minutes: float = 200.0,
    multiplier: float = 1.0,
    frame_interval: float = 5.0,
    num_processes: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run complete baseline analysis pipeline with MATLAB-compatible processing.

    Automatically chooses between sequential and parallel processing based on
    num_processes parameter and number of ROIs.

    Args:
        merged_results: Dictionary mapping ROI IDs to time-series data
        enable_matlab_norm: Apply MATLAB-style normalization
        enable_detrending: Apply detrending to remove drift
        use_improved_detrending: Use improved detrending algorithm
        enable_jump_correction: Detect and correct sudden jumps before detrending
        baseline_duration_minutes: Duration for baseline calculation
        multiplier: Threshold multiplier
        frame_interval: Time between frames (seconds)
        num_processes: Number of parallel processes (1 = sequential)
        **kwargs: Additional parameters

    Returns:
        Complete analysis results dictionary
    """
    from multiprocessing import Pool, cpu_count

    # Determine if we should use parallel processing
    num_rois = len(merged_results)
    if num_processes is None or num_processes < 1:
        num_processes = max(1, cpu_count() - 1)
    num_processes = min(num_processes, num_rois)  # Don't use more than ROIs
    use_parallel = num_processes > 1 and num_rois >= 2

    analysis_results = {
        "method": "baseline",
        "parameters": {
            "enable_matlab_norm": enable_matlab_norm,
            "enable_detrending": enable_detrending,
            "enable_jump_correction": enable_jump_correction,
            "baseline_duration_minutes": baseline_duration_minutes,
            "multiplier": multiplier,
            "frame_interval": frame_interval,
            "matlab_compatible": True,
            "num_processes": num_processes,
            "parallel": use_parallel,
        },
    }

    # Step 1: Preprocessing (sequential - shared across all ROIs)
    if enable_matlab_norm:
        normalized_data = apply_matlab_normalization_to_merged_results(merged_results)
    else:
        normalized_data = merged_results

    # Step 1a: Calculate baseline thresholds from normalized data (BEFORE detrending)
    # This ensures baseline reflects the original signal, not the detrended signal
    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}

    for roi, data in normalized_data.items():
        if not data:
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {"method": "baseline", "status": "no_data"}
            continue

        baseline_mean, upper_thresh, lower_thresh, stats = (
            compute_threshold_baseline_hysteresis(
                data, baseline_duration_minutes, multiplier, frame_interval
            )
        )

        baseline_means[roi] = baseline_mean
        upper_thresholds[roi] = upper_thresh
        lower_thresholds[roi] = lower_thresh
        roi_statistics[roi] = stats

    # Step 1b: Apply detrending and jump correction (if enabled)
    if enable_detrending and use_improved_detrending:
        processed_data = improved_full_dataset_detrending(
            normalized_data, enable_jump_correction=enable_jump_correction
        )
    else:
        processed_data = normalized_data

    analysis_results["processed_data"] = processed_data

    # Step 2: ROI-level processing (parallel or sequential)
    # Movement detection uses processed_data but pre-calculated baselines
    bin_size_seconds = kwargs.get("bin_size_seconds", 60)

    if use_parallel:
        # Parallel processing using multiprocessing.Pool
        roi_args = [
            (
                roi_id,
                processed_data[roi_id],
                baseline_means[roi_id],
                upper_thresholds[roi_id],
                lower_thresholds[roi_id],
                bin_size_seconds,
                frame_interval,
            )
            for roi_id in processed_data.keys()
        ]

        with Pool(processes=num_processes) as pool:
            roi_results = pool.map(_process_single_roi_movement, roi_args)

        # Aggregate results from parallel workers
        movement_data = {}
        fraction_data = {}

        for roi_id, results in roi_results:
            movement_data[roi_id] = results["movement_data"]
            fraction_data[roi_id] = results["fraction_data"]

    else:
        # Sequential processing
        # Movement detection using processed_data with pre-calculated baselines
        movement_data = define_movement_with_hysteresis(
            processed_data, baseline_means, upper_thresholds, lower_thresholds
        )

        # Fraction movement
        fraction_data = bin_fraction_movement(
            movement_data, bin_size_seconds, frame_interval
        )

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

    # Step 3: Post-processing (sequential - needs all ROI data)
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
# PURE ANALYSIS FUNCTIONS (for centralized preprocessing)
# =============================================================================


def run_baseline_analysis_pure(
    preprocessed_data: Dict[int, List[Tuple[float, float]]], **kwargs
) -> Dict[str, Any]:
    """
    Pure baseline analysis function that works on already-preprocessed data.
    No internal preprocessing - used by centralized integration pipeline.
    Now MATLAB-compatible.
    """

    analysis_results = {
        "method": "baseline",
        "parameters": {
            "baseline_duration_minutes": kwargs.get("baseline_duration_minutes", 200.0),
            "multiplier": kwargs.get("multiplier", 1.0),
            "frame_interval": kwargs.get("frame_interval", 5.0),
            "preprocessing_skipped": True,
            "matlab_compatible": True,
        },
    }

    # Use the preprocessed data directly (no internal preprocessing)
    processed_data = preprocessed_data
    analysis_results["processed_data"] = processed_data

    # Step 2: Baseline threshold calculation
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

        baseline_mean, upper_thresh, lower_thresh, stats = (
            compute_threshold_baseline_hysteresis(
                data,
                kwargs.get("baseline_duration_minutes", 200.0),
                kwargs.get("multiplier", 1.0),
                kwargs.get("frame_interval", 5.0),
            )
        )

        baseline_means[roi] = baseline_mean
        upper_thresholds[roi] = upper_thresh
        lower_thresholds[roi] = lower_thresh
        roi_statistics[roi] = stats

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
        }
    )

    # Step 3: Movement detection
    movement_data = define_movement_with_hysteresis(
        processed_data, baseline_means, upper_thresholds, lower_thresholds
    )
    analysis_results["movement_data"] = movement_data

    # Step 4: Behavioral analysis
    bin_size_seconds = kwargs.get("bin_size_seconds", 60)
    frame_interval = kwargs.get("frame_interval", 5.0)
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
    except Exception:
        roi_colors = {
            roi: f"C{i}" for i, roi in enumerate(sorted(processed_data.keys()))
        }

    analysis_results["roi_colors"] = roi_colors

    return analysis_results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
    """Calculate performance metrics."""
    # Handle case where start_time might be None
    if start_time is None:
        return {
            "elapsed_time": 0.0,
            "fps": 0.0,
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "total_frames": total_frames,
        }

    try:
        import psutil

        elapsed_time = time.time() - start_time
        fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent

        return {
            "elapsed_time": elapsed_time,
            "fps": fps,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "total_frames": total_frames,
        }
    except ImportError:
        elapsed_time = time.time() - start_time
        return {
            "elapsed_time": elapsed_time,
            "fps": total_frames / elapsed_time if elapsed_time > 0 else 0,
            "cpu_percent": 0,
            "memory_percent": 0,
            "total_frames": total_frames,
        }


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================


def integrate_baseline_analysis_with_widget(widget) -> bool:
    """Integration function for baseline analysis with napari widget."""
    try:
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message("No merged_results available for baseline analysis")
            return False

        # Extract parameters
        frame_interval = widget.frame_interval.value()
        baseline_duration_minutes = widget.baseline_duration_minutes.value()
        threshold_multiplier = widget.threshold_multiplier.value()
        enable_detrending = widget.enable_detrending.isChecked()

        # Run analysis with MATLAB-compatible processing
        baseline_results = run_baseline_analysis(
            merged_results=widget.merged_results,
            enable_matlab_norm=True,  # Now uses true MATLAB processing
            enable_detrending=enable_detrending,
            baseline_duration_minutes=baseline_duration_minutes,
            multiplier=threshold_multiplier,
            frame_interval=frame_interval,
        )

        # Update widget
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

        return True

    except Exception as e:
        widget._log_message(f"Baseline analysis integration failed: {str(e)}")
        return False


# =============================================================================
# PARALLEL PROCESSING WRAPPER
# =============================================================================


# Legacy aliases for backward compatibility
run_complete_hdf5_compatible_analysis = run_baseline_analysis


def test_baseline_analysis_direct(merged_results):
    """Test function for baseline analysis."""
    return bool(run_baseline_analysis(merged_results))
