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
#             logger.warning(f"Detrending failed for ROI {roi}: {e}")
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

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


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


def detect_jumps_from_frame_mean(
    frame_times_s: np.ndarray,
    frame_mean: np.ndarray,
    jump_threshold_factor: float = 3.0,
) -> List[Tuple[float, float]]:
    """
    Detect illumination jumps from HDF5 frame_mean telemetry.

    frame_mean (average pixel intensity per frame) is a clean, low-noise signal
    where sudden illumination changes appear as clear step jumps. Using it is much
    more reliable than detecting jumps in the noisy activity signal itself.

    Args:
        frame_times_s: Timestamps in seconds (same length as frame_mean)
        frame_mean: Average pixel intensity per frame from HDF5 timeseries
        jump_threshold_factor: How many sigma above median diff qualifies as jump

    Returns:
        List of (time_s, offset) tuples — offset is the jump magnitude
        (positive = sudden brightness increase, negative = decrease).
        Caller subtracts offset from all subsequent activity values.
    """
    if len(frame_mean) < 10:
        return []

    diffs = np.diff(frame_mean.astype(float))
    median_abs = np.median(np.abs(diffs))
    if median_abs == 0:
        return []

    threshold = jump_threshold_factor * median_abs
    jump_mask = np.abs(diffs) > threshold
    jump_indices = np.where(jump_mask)[0]

    # Merge consecutive jump indices (same physical event)
    merged = []
    for idx in jump_indices:
        if merged and idx == merged[-1][0] + 1:
            # Accumulate consecutive frames into one jump
            merged[-1] = (merged[-1][0], merged[-1][1] + diffs[idx])
        else:
            merged.append((idx, diffs[idx]))

    result = []
    for idx, offset in merged:
        t = float(frame_times_s[idx + 1]) if idx + 1 < len(frame_times_s) else float(frame_times_s[idx])
        result.append((t, float(offset)))

    return result


def correct_activity_with_frame_mean_jumps(
    times: np.ndarray,
    values: np.ndarray,
    jump_list: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Apply frame_mean-derived jump offsets to activity signal.

    For each detected jump (time_s, offset), all activity values at or after
    time_s are shifted by -offset so the signal level is continuous.

    Args:
        times: Time array of the activity signal (seconds)
        values: Activity values
        jump_list: Output of detect_jumps_from_frame_mean()

    Returns:
        Corrected values array (same length as input)
    """
    if not jump_list:
        return values

    corrected = values.copy()
    cumulative_offset = 0.0
    # Sort by time just in case
    for jump_time, offset in sorted(jump_list, key=lambda x: x[0]):
        mask = times >= jump_time
        corrected[mask] -= offset
        cumulative_offset += offset

    logger.debug(f"Jump correction: applied {len(jump_list)} frame_mean jumps, total offset {cumulative_offset:.3f}")
    return corrected


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
    enable_detrending: bool = True,
    frame_mean_data: Optional[Dict] = None,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Apply improved detrending to complete dataset.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        enable_jump_correction: Whether to apply jump correction before detrending
        enable_detrending: Whether to apply polynomial detrending
        frame_mean_data: Optional dict with 'times' (s) and 'values' (pixel intensity)
                         from HDF5 timeseries/frame_mean. When provided and
                         enable_jump_correction is True, jumps are derived from
                         frame_mean (more reliable) instead of the noisy activity signal.

    Returns:
        Dictionary with detrended values
    """
    # Pre-compute frame_mean jumps once (shared across all ROIs) if available
    frame_mean_jumps = []
    if enable_jump_correction and frame_mean_data:
        try:
            fm_times = np.array(frame_mean_data["times"], dtype=float)
            fm_values = np.array(frame_mean_data["values"], dtype=float)
            frame_mean_jumps = detect_jumps_from_frame_mean(fm_times, fm_values)
            logger.info(f"Frame mean jump detection: found {len(frame_mean_jumps)} jumps")
        except Exception as e:
            logger.warning(f"Could not use frame_mean for jump detection: {e}; falling back to signal-based")

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
                if frame_mean_jumps:
                    # Use frame_mean-derived jumps (preferred: clean signal)
                    values = correct_activity_with_frame_mean_jumps(times, values, frame_mean_jumps)
                else:
                    # Fallback: detect jumps in activity signal directly
                    values, jump_indices = detect_and_remove_jumps(times, values)
                    if len(jump_indices) > 0:
                        logger.debug(f"ROI {roi}: Corrected {len(jump_indices)} jumps (signal-based)")

            # Step 2: Remove polynomial trend (handles curved drift)
            if enable_detrending and len(values) >= 10:
                poly_coeffs = np.polyfit(times, values, 2)
                poly_trend = np.polyval(poly_coeffs, times)
                values_detrended = values - poly_trend + np.mean(poly_trend)
            else:
                values_detrended = values

            # Step 3: Remove any remaining linear drift (only if detrending enabled)
            if enable_detrending and len(values_detrended) >= 10:
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
            logger.warning(f"Detrending failed for ROI {roi}: {e}")
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

    # Calculate baseline time range — cap to actual recording duration
    recording_duration_seconds = sorted_data[-1][0] - sorted_data[0][0]
    baseline_duration_seconds = min(
        baseline_duration_minutes * 60, recording_duration_seconds
    )
    start_time = sorted_data[0][0]
    end_time = start_time + baseline_duration_seconds

    # Select baseline data
    baseline_data = [(t, v) for t, v in sorted_data if start_time <= t < end_time]

    # Check minimum data requirement.
    # Use the ACTUAL time step between consecutive data points, not frame_interval.
    # frame_interval is the raw camera interval (e.g. 5 s), but data may already
    # be binned (e.g. 5-min bins = 300 s).  Using frame_interval would demand
    # 25 920 frames for a 72 h baseline when only 864 are available.
    if len(sorted_data) >= 2:
        actual_dt = (sorted_data[-1][0] - sorted_data[0][0]) / (len(sorted_data) - 1)
    else:
        actual_dt = frame_interval
    actual_dt = max(actual_dt, frame_interval)  # never smaller than raw frame interval

    min_required_frames = max(2, int(baseline_duration_seconds / actual_dt * 0.5))
    if len(baseline_data) < min_required_frames:
        # Fallback: use full-signal percentiles so thresholds are never (0, 0, 0)
        full_vals = np.array([v for _, v in sorted_data])
        fb_mean   = float(np.median(full_vals))
        fb_upper  = float(np.percentile(full_vals, 75))
        fb_lower  = float(np.percentile(full_vals, 25))
        return (
            fb_mean,
            fb_upper,
            fb_lower,
            {
                "method": "baseline_hysteresis",
                "status": "insufficient_data_fallback_percentiles",
                "found_frames": len(baseline_data),
                "required_frames": min_required_frames,
            },
        )

    # Calculate statistics
    values = np.array([val for _, val in baseline_data])
    full_values = np.array([v for _, v in sorted_data])

    mean_val = np.mean(values)
    std_val = np.std(values)
    full_std = float(np.std(full_values))

    # Fallback: if the baseline std is too small (< 5 % of full-signal std, or
    # exactly 0 — which happens when detrending flattens the baseline period),
    # use the full-signal std so the threshold band is never zero.
    std_source = "baseline"
    if full_std > 0 and std_val < 0.05 * full_std:
        std_val = full_std
        std_source = "full_signal_fallback"
        logger.debug(
            f"Baseline std too small ({std_val:.2e} < 5% of full std {full_std:.2e}); "
            "using full-signal std as fallback."
        )

    # Calculate hysteresis thresholds
    baseline_mean = mean_val
    threshold_band = multiplier * std_val
    upper_threshold = baseline_mean + threshold_band
    # Clamp lower threshold to the minimum actually observed during the baseline
    # period — the signal never reaches 0 (absolute frame differences always have
    # a camera-noise floor), so 0 is not a meaningful lower bound.
    min_baseline = float(np.min(values))
    lower_threshold = max(min_baseline, baseline_mean - threshold_band)

    # Validate thresholds
    if np.isnan(upper_threshold) or np.isinf(upper_threshold):
        upper_threshold = np.percentile(full_values, 75)
        lower_threshold = np.percentile(full_values, 25)
        baseline_mean = np.median(values)

    statistics = {
        "method": "baseline_hysteresis",
        "baseline_mean": baseline_mean,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "threshold_band": threshold_band,
        "mean": mean_val,
        "std": std_val,
        "std_source": std_source,
        "multiplier": multiplier,
        "baseline_frames": len(baseline_data),
        "baseline_duration_minutes": baseline_duration_minutes,
        "frame_interval": frame_interval,
        "data_range": (float(np.min(full_values)), float(np.max(full_values))),
        "min_baseline": min_baseline,
        "status": "calculated_from_preprocessed_data",
    }

    return baseline_mean, upper_threshold, lower_threshold, statistics


# =============================================================================
# ROI ACTIVITY CHECK (EMPTY WELL DETECTION)
# =============================================================================


def check_roi_activity(
    merged_results: Dict[int, List[Tuple[float, float]]],
    relative_threshold: float = 0.3,
) -> Dict[int, bool]:
    """Check which ROIs have sufficient signal amplitude to be considered active.

    Empty wells produce sensor noise that looks similar in range to active wells
    when using per-pixel normalization.  This function uses the **standard deviation**
    of each ROI's time-series as the activity metric and compares it against the
    median std across all ROIs.  Active wells have higher variability (movement
    periods vs. rest) while empty wells have uniform low-level noise.

    If there are fewer than 2 ROIs, all are treated as active (no comparison possible).

    Args:
        merged_results: Raw (un-normalized) frame difference data per ROI.
        relative_threshold: Fraction of the median std below which a ROI is
            considered inactive.  Default 0.3 (30 % of median).

    Returns:
        Dictionary mapping ROI ID to boolean (True = active, False = inactive/empty).
    """
    import numpy as np

    # First pass: compute std per ROI
    roi_stds: Dict[int, float] = {}
    for roi, data in merged_results.items():
        if not data or len(data) < 10:
            roi_stds[roi] = 0.0
            continue
        values = np.array([v for _, v in data])
        roi_stds[roi] = float(np.std(values))

    # Need at least 2 ROIs to do a meaningful comparison
    non_zero_stds = [s for s in roi_stds.values() if s > 0]
    if len(non_zero_stds) < 2:
        return {roi: True for roi in merged_results}

    median_std = float(sorted(non_zero_stds)[len(non_zero_stds) // 2])
    activity_threshold = relative_threshold * median_std

    roi_active = {}
    for roi, std_val in roi_stds.items():
        roi_active[roi] = std_val >= activity_threshold

    logger.debug(f"Activity check (std-based): median std = {median_std:.6e}, "
                 f"threshold = {activity_threshold:.6e}")
    for roi in sorted(roi_stds.keys()):
        status = "ACTIVE" if roi_active[roi] else "INACTIVE"
        logger.debug(f"  ROI {roi}: std = {roi_stds[roi]:.6e} -> {status}")

    return roi_active


def apply_minmax_normalization(
    data_dict: Dict[int, List[Tuple[float, float]]],
    baseline_means: Dict[int, float],
    upper_thresholds: Dict[int, float],
    lower_thresholds: Dict[int, float],
    roi_active: Optional[Dict[int, bool]] = None,
) -> Tuple[
    Dict[int, List[Tuple[float, float]]],
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
]:
    """Apply MinMax normalization to processed data AND consistently transform
    baseline means and thresholds using the same per-ROI scaling factors.

    For inactive ROIs, data is set to all zeros.

    Args:
        data_dict: Time-series data per ROI.
        baseline_means: Per-ROI baseline means.
        upper_thresholds: Per-ROI upper hysteresis thresholds.
        lower_thresholds: Per-ROI lower hysteresis thresholds.
        roi_active: Optional activity flags. If None, all ROIs treated as active.

    Returns:
        Tuple of (normalized_data, normalized_baselines, normalized_upper, normalized_lower).
    """
    norm_data = {}
    norm_baselines = {}
    norm_upper = {}
    norm_lower = {}

    for roi, data in data_dict.items():
        is_active = roi_active.get(roi, True) if roi_active else True

        if not data or not is_active:
            norm_data[roi] = [(t, 0.0) for t, _ in data] if data else []
            norm_baselines[roi] = 0.0
            norm_upper[roi] = 0.0
            norm_lower[roi] = 0.0
            continue

        values = [v for _, v in data]
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val

        logger.debug(f"MinMax ROI {roi}: min={min_val:.6e}, max={max_val:.6e}, range={val_range:.6e}, active={is_active}")

        if val_range > 0:
            norm_data[roi] = [(t, (v - min_val) / val_range) for t, v in data]
            norm_baselines[roi] = (baseline_means.get(roi, 0.0) - min_val) / val_range
            norm_upper[roi] = (upper_thresholds.get(roi, 0.0) - min_val) / val_range
            norm_lower[roi] = (lower_thresholds.get(roi, 0.0) - min_val) / val_range
        else:
            norm_data[roi] = [(t, 0.0) for t, _ in data]
            norm_baselines[roi] = 0.0
            norm_upper[roi] = 0.0
            norm_lower[roi] = 0.0

    return norm_data, norm_baselines, norm_upper, norm_lower


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
    """Calculate fraction movement using hysteresis state data.

    Uses a two-pointer sweep (O(n+m)) instead of a nested loop (O(n*m))
    for efficient binning over large datasets.
    """
    fraction_data = {}

    for roi, data in movement_data.items():
        if not data:
            fraction_data[roi] = []
            continue

        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            fraction_data[roi] = []
            continue

        # Convert to numpy arrays for fast indexing
        times  = np.array([t for t, _ in sorted_data], dtype=np.float64)
        states = np.array([s for _, s in sorted_data], dtype=np.int8)
        end_time = times[-1]

        # Build bin edges
        first_bin_start = (times[0] // bin_size_seconds) * bin_size_seconds
        bin_starts = np.arange(
            first_bin_start,
            end_time + bin_size_seconds,
            bin_size_seconds,
            dtype=np.float64,
        )

        roi_fractions = []
        ptr = 0  # two-pointer: advance through data once across all bins

        for bin_start in bin_starts[:-1]:
            bin_end = bin_start + bin_size_seconds
            bin_center = (bin_start + bin_end) / 2.0

            # Advance pointer past entries that end before this bin
            while ptr + 1 < len(times) and times[ptr + 1] <= bin_start:
                ptr += 1

            # Accumulate movement time within [bin_start, bin_end)
            movement_time = 0.0
            j = ptr
            while j < len(times):
                t_curr = times[j]
                if t_curr >= bin_end:
                    break
                t_next = times[j + 1] if j + 1 < len(times) else end_time
                if states[j] == 1:
                    overlap_start = max(t_curr, bin_start)
                    overlap_end   = min(t_next, bin_end)
                    if overlap_end > overlap_start:
                        movement_time += overlap_end - overlap_start
                j += 1

            fraction = movement_time / bin_size_seconds
            roi_fractions.append((bin_center, max(0.0, min(1.0, fraction))))

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


def bin_and_normalize_movement(
    merged_results: Dict[int, List[Tuple[float, float]]],
    bin_size_seconds: int,
) -> Dict[int, List[Tuple[float, float]]]:
    """Bin raw pixel changes and min/max normalize per ROI.

    Creates a continuous 0-1 normalized movement signal comparable to
    Aguillon et al. (2023) "Normalized Movement (a.u.)" where total distance
    is summed per hourly bin and min/max normalized per animal.

    Args:
        merged_results: Raw frame-differencing data {roi: [(time, pixel_change), ...]}
        bin_size_seconds: Bin size in seconds (e.g. 3600 for hourly bins)

    Returns:
        Dict mapping ROI ID to list of (bin_center_time, normalized_value) tuples
        where normalized_value is 0-1 (min/max per ROI).
    """
    normalized_data = {}

    for roi, data in merged_results.items():
        if not data:
            normalized_data[roi] = []
            continue

        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            normalized_data[roi] = []
            continue

        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]

        # Create time bins
        first_bin_start = (start_time // bin_size_seconds) * bin_size_seconds
        bin_edges = np.arange(
            first_bin_start,
            end_time + bin_size_seconds,
            bin_size_seconds,
        )

        # Vectorised binning using searchsorted (O(N log N) instead of O(N*B))
        times_arr = np.array([t for t, _ in sorted_data])
        values_arr = np.array([v for _, v in sorted_data])
        bin_indices = np.searchsorted(times_arr, bin_edges)

        roi_binned = []
        for i in range(len(bin_edges) - 1):
            bin_center = float((bin_edges[i] + bin_edges[i + 1]) / 2)
            s, e = int(bin_indices[i]), int(bin_indices[i + 1])
            bin_sum = float(values_arr[s:e].sum())
            roi_binned.append((bin_center, bin_sum))

        # Min/Max normalize per ROI
        if roi_binned:
            vals = np.array([v for _, v in roi_binned])
            min_val = float(vals.min())
            max_val = float(vals.max())
            val_range = max_val - min_val

            if val_range > 0:
                normalized_data[roi] = [
                    (t, (v - min_val) / val_range) for t, v in roi_binned
                ]
            else:
                normalized_data[roi] = [(t, 0.0) for t, _ in roi_binned]
        else:
            normalized_data[roi] = []

    return normalized_data


def rebin_fraction_movement(
    fraction_data: Dict[int, List[Tuple[float, float]]],
    plot_bin_minutes: int,
    original_bin_seconds: int = 60,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Re-bin fraction movement data for visualization purposes.

    Takes already-binned fraction movement data (e.g., 60s bins) and
    aggregates it into larger bins (e.g., 60 minutes) for clearer visualization.

    Args:
        fraction_data: Dict mapping ROI ID to list of (time, fraction_movement) tuples
        plot_bin_minutes: New bin size in minutes for visualization
        original_bin_seconds: Original bin size in seconds (default: 60s)

    Returns:
        Dict mapping ROI ID to list of (bin_center_time, mean_fraction_movement) tuples

    Example:
        # Original data: 60s bins with fraction values 0-1
        # Re-binned: 60 minute bins with averaged fraction values
        rebinned = rebin_fraction_movement(fraction_data, plot_bin_minutes=60)
    """
    rebinned_data = {}
    plot_bin_seconds = plot_bin_minutes * 60

    for roi, data in fraction_data.items():
        if not data:
            rebinned_data[roi] = []
            continue

        sorted_data = sorted(data, key=lambda x: x[0])

        if len(sorted_data) < 2:
            rebinned_data[roi] = sorted_data
            continue

        start_time = sorted_data[0][0]
        end_time = sorted_data[-1][0]

        # Create new time bins
        first_bin_start = (start_time // plot_bin_seconds) * plot_bin_seconds
        bin_edges = []
        current_bin_start = first_bin_start
        while current_bin_start < end_time:
            bin_edges.append(current_bin_start)
            current_bin_start += plot_bin_seconds
        bin_edges.append(current_bin_start)

        roi_rebinned = []

        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2

            # Collect all fraction values in this bin
            bin_fractions = []
            for time_point, fraction_value in sorted_data:
                if bin_start <= time_point < bin_end:
                    bin_fractions.append(fraction_value)

            # Calculate mean fraction for this bin
            if bin_fractions:
                mean_fraction = np.mean(bin_fractions)
                roi_rebinned.append((bin_center, mean_fraction))

        rebinned_data[roi] = roi_rebinned

    return rebinned_data


def define_sleep_periods(
    quiescence_data: Dict[int, List[Tuple[float, int]]],
    sleep_threshold_minutes: int = 8,
    bin_size_seconds: int = 60,
) -> Dict[int, List[Tuple[float, int]]]:
    """Define sleep as sustained quiescence periods."""

    sleep_data = {}
    min_bins_for_sleep = max(1, (sleep_threshold_minutes * 60) // bin_size_seconds)

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


def calculate_sleep_quality_hourly(
    sleep_data: Dict[int, List[Tuple[float, int]]],
    bin_size_minutes: int = 60,
    data_bin_seconds: int = 60,
) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
    """
    Calculate hourly sleep quality metrics (MATLAB-compatible).

    Computes per ROI and per hourly bin:
    - sleep_minutes: Total minutes of sleep per hour
    - transitions: Number of sleep<->wake transitions per hour
    - bout_length: Mean sleep bout length (minutes) per hour

    Args:
        sleep_data: Binary sleep data {roi: [(time_s, 0/1), ...]}
        bin_size_minutes: Hourly bin size (default 60 min)
        data_bin_seconds: Resolution of input data bins (default 60s)

    Returns:
        Dict with keys 'sleep_minutes', 'transitions', 'bout_length',
        each mapping to {roi: [(bin_center_s, value), ...]}
    """
    bin_size_seconds = bin_size_minutes * 60
    minutes_per_data_bin = data_bin_seconds / 60.0

    result = {
        "sleep_minutes": {},
        "transitions": {},
        "bout_length": {},
        "sleep_hours_per_day": {},
    }

    for roi, data in sleep_data.items():
        if not data:
            for key in result:
                result[key][roi] = []
            continue

        sorted_data = sorted(data, key=lambda x: x[0])
        times = np.array([t for t, _ in sorted_data])
        states = np.array([s for _, s in sorted_data])

        start_time = times[0]
        end_time = times[-1]
        # Align to hour boundaries
        hour_start = (start_time // bin_size_seconds) * bin_size_seconds

        sleep_min_roi = []
        transitions_roi = []
        bout_length_roi = []

        current_bin_start = hour_start
        while current_bin_start < end_time:
            bin_end = current_bin_start + bin_size_seconds
            bin_center = current_bin_start + bin_size_seconds / 2

            # Get data points in this bin
            mask = (times >= current_bin_start) & (times < bin_end)
            bin_states = states[mask]

            if len(bin_states) == 0:
                current_bin_start = bin_end
                continue

            # 1. Sleep minutes: count sleep bins × minutes per bin
            sleep_bins_count = int(np.sum(bin_states))
            sleep_min = sleep_bins_count * minutes_per_data_bin
            sleep_min_roi.append((bin_center, float(sleep_min)))

            # 2. Transitions: count state changes (|diff|)
            if len(bin_states) > 1:
                trans_count = int(np.sum(np.abs(np.diff(bin_states))))
            else:
                trans_count = 0
            transitions_roi.append((bin_center, float(trans_count)))

            # 3. Mean bout length: find consecutive sleep runs
            bout_lengths = []
            i = 0
            while i < len(bin_states):
                if bin_states[i] == 1:
                    j = i
                    while j < len(bin_states) and bin_states[j] == 1:
                        j += 1
                    bout_len_min = (j - i) * minutes_per_data_bin
                    bout_lengths.append(bout_len_min)
                    i = j
                else:
                    i += 1

            mean_bout = float(np.mean(bout_lengths)) if bout_lengths else 0.0
            bout_length_roi.append((bin_center, mean_bout))

            current_bin_start = bin_end

        result["sleep_minutes"][roi] = sleep_min_roi
        result["transitions"][roi] = transitions_roi
        result["bout_length"][roi] = bout_length_roi

        # 4. Sleep duration per 24 h: total sleep hours per calendar day
        day_size_s = 24 * 3600.0
        day_start = (start_time // day_size_s) * day_size_s
        sleep_per_day = []
        while day_start < end_time:
            day_end = day_start + day_size_s
            day_center = day_start + day_size_s / 2
            mask = (times >= day_start) & (times < day_end)
            day_states = states[mask]
            if len(day_states) > 0:
                sleep_h = float(np.sum(day_states)) * minutes_per_data_bin / 60.0
                sleep_per_day.append((day_center, sleep_h))
            day_start = day_end
        result["sleep_hours_per_day"][roi] = sleep_per_day

    return result


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


def extract_illumination_periods(
    led_data: Dict[str, list],
    recording_start_s: float = 0.0,
    recording_end_s: Optional[float] = None,
    min_period_s: float = 60.0,
) -> List[Tuple[float, float, str]]:
    """
    Extract light/dark periods from LED timeseries data.

    Returns:
        List of (start_s, end_s, phase) where phase is 'light' or 'dark'.
        Empty list if LED data is unusable.
    """
    try:
        times = np.array(led_data.get("times", []), dtype=float)
        white = np.array(led_data.get("white_powers", []), dtype=float)
        if len(times) == 0 or len(white) == 0:
            return []

        phases = np.where(white > 0, "light", "dark")
        periods = []
        start = times[0]
        current = phases[0]

        for i in range(1, len(times)):
            if phases[i] != current:
                end = times[i]
                if end - start >= min_period_s:
                    periods.append((max(start, recording_start_s), end, current))
                start = times[i]
                current = phases[i]

        # Last period
        end = times[-1] if recording_end_s is None else recording_end_s
        if end - start >= min_period_s:
            periods.append((max(start, recording_start_s), end, current))

        return periods
    except Exception as e:
        logger.warning(f"extract_illumination_periods failed: {e}")
        return []


def equalize_signal_per_illumination_period(
    data: List[Tuple[float, float]],
    periods: List[Tuple[float, float, str]],
    floor_percentile: float = 15.0,
    use_mode: bool = False,
    transition_ramp_minutes: float = 2.0,
) -> List[Tuple[float, float]]:
    """
    Level-correct activity signal so all illumination periods share the same baseline floor.

    For each L/D period the resting floor is estimated as a low percentile of
    the period's frame-to-frame differences (default 15%). This is robust
    across periods with very different activity proportions because the lower
    tail of the distribution is dominated by quiet frames regardless of the
    overall activity level. A histogram-mode estimator is available via
    use_mode=True for special cases, but it can pick the wrong baseline when
    the active-frame count exceeds the resting-frame count (the mode then
    sits on the activity peak, not the resting peak), so it is OFF by default.

    All floors are aligned to a global reference (median of the per-period
    floors) by adding a per-frame shift. To avoid an audible "click" at L/D
    boundaries the per-frame shift is convolved with a Gaussian whose width
    is set by transition_ramp_minutes, so the correction ramps smoothly
    across the transition instead of stepping discontinuously.

    Args:
        data: List of (time_s, value) tuples (frame-to-frame differences)
        periods: List of (start_s, end_s, phase) from extract_illumination_periods()
        floor_percentile: Percentile used as fallback floor when
                          use_mode=False (default 15).
        use_mode: When True (default), use the histogram mode of each period's
                  values as the resting floor — robust to very active animals.
                  When False, fall back to the legacy floor_percentile behaviour.
        transition_ramp_minutes: Smoothing half-width applied to the per-frame
                                  shift around L/D boundaries (default 2.0).
                                  Set to 0 to keep hard step transitions.

    Returns:
        Level-corrected data as List of (time_s, value) tuples
    """
    if not data or not periods:
        return data

    times = np.array([t for t, _ in data])
    values = np.array([v for _, v in data])

    def _estimate_floor(period_values: np.ndarray) -> Optional[float]:
        if len(period_values) < 5:
            return None
        if use_mode:
            # Histogram peak = mode of the value distribution = where the
            # signal sits most of the time. Activity bursts pile up in the
            # upper tail and don't displace the peak — far more robust than
            # a low percentile when the animal is frequently active.
            n_bins = max(50, int(np.sqrt(len(period_values))))
            hist, edges = np.histogram(period_values, bins=n_bins)
            peak = int(np.argmax(hist))
            return float(0.5 * (edges[peak] + edges[peak + 1]))
        return float(np.percentile(period_values, floor_percentile))

    period_floors: List[Optional[float]] = []
    period_masks: List[np.ndarray] = []
    for start_s, end_s, _ in periods:
        mask = (times >= start_s) & (times < end_s)
        period_floors.append(_estimate_floor(values[mask]))
        period_masks.append(mask)

    valid_floors = [f for f in period_floors if f is not None]
    if not valid_floors:
        return data

    global_floor = float(np.median(valid_floors))

    # Step 1: per-frame shift as a piecewise-constant step function.
    per_frame_shift = np.zeros_like(values, dtype=float)
    for mask, floor in zip(period_masks, period_floors):
        if floor is None:
            continue
        per_frame_shift[mask] = global_floor - floor

    # Step 2: blur the step function with a Gaussian so frames near a L/D
    # boundary receive a smoothly-interpolated shift instead of a jump.
    if transition_ramp_minutes > 0 and len(times) > 1:
        try:
            from scipy.ndimage import gaussian_filter1d
            dt = float(np.median(np.diff(times)))
            if dt > 0:
                sigma_frames = (transition_ramp_minutes * 60.0 / dt) / 2.0
                if sigma_frames >= 0.5:
                    per_frame_shift = gaussian_filter1d(
                        per_frame_shift, sigma=sigma_frames, mode="nearest"
                    )
        except ImportError:
            pass  # scipy missing — keep hard step (fail-safe)

    corrected = np.clip(values + per_frame_shift, 0.0, None)

    logger.debug(
        f"Signal equalization: {len(valid_floors)} periods, "
        f"floors {[f'{f:.4f}' for f in valid_floors]}, global ref={global_floor:.4f}, "
        f"floor_method={'mode' if use_mode else f'p{floor_percentile:.0f}'}, "
        f"transition_ramp_min={transition_ramp_minutes:.1f}"
    )
    return list(zip(times, corrected))


def _process_single_roi_movement(
    args: Tuple,
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function for parallel ROI movement detection with pre-calculated baseline.

    Args:
        args: Tuple of (roi_id, data, baseline_mean, upper_threshold,
                       lower_threshold, bin_size_seconds, frame_interval, is_active)

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
        is_active,
    ) = args

    results = {}

    if not data:
        results["movement_data"] = []
        results["fraction_data"] = []
        return roi_id, results

    # Inactive ROI (empty well): force all movement to 0
    if not is_active:
        sorted_data = sorted(data, key=lambda x: x[0])
        zero_movement = [(t, 0) for t, _ in sorted_data]
        results["movement_data"] = zero_movement
        movement_data_dict = {roi_id: zero_movement}
        fraction_data_dict = bin_fraction_movement(
            movement_data_dict, bin_size_seconds, frame_interval
        )
        results["fraction_data"] = fraction_data_dict.get(roi_id, [])
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
    frame_mean_data: Optional[Dict] = None,
    adaptive_illumination_baseline: bool = False,
    led_data: Optional[Dict] = None,
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

    # Check available RAM: each worker needs ~300 MB for Python + modules on Windows.
    # If free RAM < num_processes * 350 MB, fall back to sequential to avoid OOM hang.
    try:
        import psutil
        free_mb = psutil.virtual_memory().available / (1024 ** 2)
        required_mb = num_processes * 350
        if free_mb < required_mb:
            logger.warning(
                f"Low RAM ({free_mb:.0f} MiB free, {required_mb:.0f} MiB needed for "
                f"{num_processes} workers) — forcing sequential processing"
            )
            num_processes = 1
    except ImportError:
        pass

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

    # Step 1.1: Amplitude check for inactive ROIs (empty wells)
    roi_active = check_roi_activity(normalized_data)
    analysis_results["roi_active"] = roi_active
    inactive_count = sum(1 for v in roi_active.values() if not v)
    if inactive_count > 0:
        logger.info(f"Detected {inactive_count} inactive ROIs (empty wells)")

    # Step 1a: Apply detrending and jump correction (if enabled) FIRST,
    # so baseline thresholds are computed on the same signal used for movement detection.
    logger.debug(f"Step 1a: Detrending enabled={enable_detrending}, jump_correction={enable_jump_correction}")
    if (enable_detrending and use_improved_detrending) or enable_jump_correction:
        processed_data = improved_full_dataset_detrending(
            normalized_data,
            enable_jump_correction=enable_jump_correction,
            enable_detrending=enable_detrending and use_improved_detrending,
            frame_mean_data=frame_mean_data,
        )
    else:
        processed_data = normalized_data
    logger.debug(f"Step 1a complete: {len(processed_data)} ROIs")

    # Step 1b: Equalize signal level across illumination periods (adaptive baseline).
    # Must run BEFORE threshold computation so thresholds and movement detection
    # both operate on the same equalized signal space.
    if adaptive_illumination_baseline and led_data:
        all_times = [t for data in processed_data.values() for t, _ in data]
        rec_start = min(all_times) if all_times else 0.0
        rec_end = max(all_times) if all_times else None
        periods = extract_illumination_periods(led_data, rec_start, rec_end)
        if periods:
            logger.info(f"Adaptive illumination baseline: equalizing signal across {len(periods)} periods")
            processed_data = {
                roi: equalize_signal_per_illumination_period(data, periods)
                for roi, data in processed_data.items()
            }
        else:
            logger.warning("Adaptive illumination baseline: no usable periods found, using global baseline")

    # Step 1c: Calculate baseline thresholds from equalized/processed data.
    # Thresholds are now in the same signal space as movement detection.
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
                data, baseline_duration_minutes, multiplier, frame_interval
            )
        )

        baseline_means[roi] = baseline_mean
        upper_thresholds[roi] = upper_thresh
        lower_thresholds[roi] = lower_thresh
        roi_statistics[roi] = stats

    # Step 1d (optional): override thresholds with pre-computed fixed values.
    # _fixed_upper_thresholds are in post-MinMax [0-1] space (as entered by the user).
    # We de-normalise them per ROI so that after apply_minmax_normalization they map
    # back to exactly the user-specified value — giving identical threshold lines for
    # all ROIs in the displayed [0-1] plot.
    if "_fixed_upper_thresholds" in kwargs:
        _f_upper = kwargs.pop("_fixed_upper_thresholds", {})
        _f_lower = kwargs.pop("_fixed_lower_thresholds", {})
        for roi in list(processed_data.keys()):
            if roi not in _f_upper:
                continue
            vals = np.array([v for _, v in processed_data[roi]]) if processed_data[roi] else np.array([0.0])
            min_roi = float(np.min(vals))
            range_roi = float(np.max(vals)) - min_roi
            fixed_norm_upper = _f_upper[roi]
            fixed_norm_lower = _f_lower.get(roi, fixed_norm_upper * 0.8)
            if range_roi > 0:
                # De-normalise: threshold_preminmax = fixed_norm * range + min
                upper_thresholds[roi] = fixed_norm_upper * range_roi + min_roi
                lower_thresholds[roi] = fixed_norm_lower * range_roi + min_roi
            else:
                upper_thresholds[roi] = fixed_norm_upper
                lower_thresholds[roi] = fixed_norm_lower
            baseline_means[roi] = (upper_thresholds[roi] + lower_thresholds[roi]) / 2

    # Step 2: ROI-level processing (parallel or sequential)
    # Movement detection uses processed_data but pre-calculated baselines
    bin_size_seconds = kwargs.get("bin_size_seconds", 60)
    logger.debug(f"Step 2: Movement detection (parallel={use_parallel}, processes={num_processes})")

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
                roi_active.get(roi_id, True),
            )
            for roi_id in processed_data.keys()
        ]

        logger.debug(f"Starting Pool with {num_processes} processes...")
        try:
            with Pool(processes=num_processes) as pool:
                roi_results = pool.map(_process_single_roi_movement, roi_args)
            logger.debug(f"Pool complete: {len(roi_results)} results")
        except (MemoryError, OSError) as exc:
            logger.warning(
                f"Parallel Pool failed ({exc}); falling back to sequential processing"
            )
            roi_results = [_process_single_roi_movement(a) for a in roi_args]

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

        # Force inactive ROIs to zero movement
        for roi in movement_data:
            if not roi_active.get(roi, True):
                movement_data[roi] = [(t, 0) for t, _ in processed_data.get(roi, [])]

        # Fraction movement
        fraction_data = bin_fraction_movement(
            movement_data, bin_size_seconds, frame_interval
        )

    # Step 2.5: Apply MinMax normalization AFTER movement detection (for display)
    logger.debug(f"Applying MinMax normalization to {len(processed_data)} ROIs...")
    norm_processed, norm_baselines, norm_upper, norm_lower = apply_minmax_normalization(
        processed_data, baseline_means, upper_thresholds, lower_thresholds, roi_active
    )

    analysis_results.update(
        {
            "processed_data": norm_processed,
            "baseline_means": norm_baselines,
            "upper_thresholds": norm_upper,
            "lower_thresholds": norm_lower,
            # Keep raw (pre-MinMax) data for amplitude view toggle
            "processed_data_raw": processed_data,
            "baseline_means_raw": baseline_means,
            "upper_thresholds_raw": upper_thresholds,
            "lower_thresholds_raw": lower_thresholds,
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
            roi: f"C{(roi - 1) % 10}" for roi in sorted(processed_data.keys())
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
