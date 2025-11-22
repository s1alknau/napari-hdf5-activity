# """
# _calc_integration.py - Clean integration module for HDF5 analysis methods

# This module provides routing between baseline, adaptive, and calibration methods
# and handles method validation and performance metrics.
# """

# from typing import Dict, List, Tuple, Optional, Any
# import os
# import time


# def run_analysis_with_method(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     method: str,
#     **kwargs
# ) -> Dict[str, Any]:
#     """
#     Main analysis routing function.

#     Args:
#         merged_results: Raw frame difference data from Reader
#         method: Analysis method ('baseline', 'adaptive', 'calibration')
#         **kwargs: Method-specific parameters

#     Returns:
#         Complete analysis results dictionary
#     """
#     if method.lower() == 'baseline':
#         from ._calc import run_baseline_analysis
#         return run_baseline_analysis(merged_results, **kwargs)

#     elif method.lower() == 'adaptive':
#         from ._calc_adaptive import run_adaptive_analysis
#         return run_adaptive_analysis(merged_results, **kwargs)

#     elif method.lower() == 'calibration':
#         # Check if we have pre-computed calibration baseline
#         if 'calibration_baseline_statistics' in kwargs:
#             from ._calc_calibration import run_calibration_analysis_with_precomputed_baseline
#             return run_calibration_analysis_with_precomputed_baseline(merged_results, **kwargs)
#         else:
#             from ._calc_calibration import run_calibration_analysis
#             return run_calibration_analysis(merged_results, **kwargs)

#     else:
#         raise ValueError(f"Unknown analysis method: {method}. "
#                         f"Supported methods: 'baseline', 'adaptive', 'calibration'")


# def get_analysis_summary(analysis_results: Dict[str, Any]) -> str:
#     """Generate summary string for any analysis method."""
#     try:
#         method = analysis_results.get('method', 'unknown')
#         params = analysis_results.get('parameters', {})
#         roi_count = len(analysis_results.get('baseline_means', {}))

#         # Calculate movement statistics
#         movement_data = analysis_results.get('movement_data', {})
#         if movement_data:
#             import numpy as np
#             movement_percentages = []
#             for roi_data in movement_data.values():
#                 if roi_data:
#                     movement_pct = np.mean([m for _, m in roi_data]) * 100
#                     movement_percentages.append(movement_pct)
#             avg_movement = np.mean(movement_percentages) if movement_percentages else 0
#         else:
#             avg_movement = 0

#         # Calculate sleep statistics
#         sleep_data = analysis_results.get('sleep_data', {})
#         if sleep_data:
#             sleep_percentages = []
#             for roi_data in sleep_data.values():
#                 if roi_data:
#                     sleep_pct = np.mean([s for _, s in roi_data]) * 100
#                     sleep_percentages.append(sleep_pct)
#             avg_sleep = np.mean(sleep_percentages) if sleep_percentages else 0
#         else:
#             avg_sleep = 0

#         # Method-specific information
#         method_info = ""
#         if method == 'baseline':
#             multiplier = params.get('multiplier', 'unknown')
#             detrending = 'Yes' if params.get('enable_detrending', False) else 'No'
#             method_info = f"Multiplier: {multiplier}, Detrending: {detrending}"

#         elif method == 'adaptive':
#             base_multiplier = params.get('base_multiplier', 'unknown')
#             duration = params.get('analysis_duration_frames', 'unknown')
#             method_info = f"Base multiplier: {base_multiplier}, Analysis frames: {duration}"

#         elif method == 'calibration':
#             cal_multiplier = params.get('calibration_multiplier', 'unknown')
#             method_info = f"Calibration multiplier: {cal_multiplier}"

#         summary = f"""Analysis Summary:
# Method: {method.upper()}
# ROIs analyzed: {roi_count}
# {method_info}
# MATLAB normalization: {'Yes' if params.get('enable_matlab_norm', False) else 'No'}

# Results:
# - Average movement: {avg_movement:.1f}%
# - Average sleep: {avg_sleep:.1f}%
# - Status: {'Successful' if roi_count > 0 else 'Failed'}"""

#         return summary

#     except Exception as e:
#         return f"Error generating summary: {e}"


# def validate_hdf5_timing_in_data(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     frame_interval: float = 5.0
# ) -> Dict[str, Any]:
#     """Validate HDF5 timing consistency."""
#     if not merged_results:
#         return {"timing_type": "no_data", "needs_correction": False}

#     try:
#         # Get sample data from first ROI
#         first_roi_data = next(iter(merged_results.values()))
#         if len(first_roi_data) < 3:
#             return {"timing_type": "insufficient_data", "needs_correction": False}

#         # Calculate actual intervals
#         times = [t for t, _ in first_roi_data[:10]]  # First 10 points
#         intervals = [times[i+1] - times[i] for i in range(len(times)-1)]

#         import numpy as np
#         avg_interval = np.mean(intervals)
#         interval_std = np.std(intervals)

#         # Check consistency
#         tolerance = max(1.0, frame_interval * 0.1)  # 10% tolerance
#         needs_correction = abs(avg_interval - frame_interval) > tolerance
#         interval_consistent = interval_std < (frame_interval * 0.05)  # 5% variation

#         return {
#             "timing_type": "hdf5_analysis",
#             "first_time": times[0],
#             "avg_interval": avg_interval,
#             "expected_interval": frame_interval,
#             "interval_consistent": interval_consistent,
#             "needs_hdf5_correction": needs_correction,
#             "recommended_action": "Apply timing correction" if needs_correction else "No correction needed"
#         }

#     except Exception as e:
#         return {"timing_type": "error", "needs_correction": False, "error": str(e)}


# def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
#     """Calculate performance metrics for analysis."""
#     try:
#         import psutil

#         elapsed_time = time.time() - start_time
#         fps = total_frames / elapsed_time if elapsed_time > 0 else 0

#         cpu_percent = psutil.cpu_percent(interval=None)
#         memory_info = psutil.virtual_memory()
#         memory_percent = memory_info.percent

#         return {
#             'elapsed_time': elapsed_time,
#             'fps': fps,
#             'cpu_percent': cpu_percent,
#             'memory_percent': memory_percent,
#             'total_frames': total_frames
#         }
#     except ImportError:
#         elapsed_time = time.time() - start_time
#         fps = total_frames / elapsed_time if elapsed_time > 0 else 0

#         return {
#             'elapsed_time': elapsed_time,
#             'fps': fps,
#             'cpu_percent': 0,
#             'memory_percent': 0,
#             'total_frames': total_frames
#         }


# def export_results_for_matlab(analysis_results: Dict[str, Any], output_dir: str) -> List[str]:
#     """Export analysis results in MATLAB-compatible format."""
#     import csv
#     import os
#     from datetime import datetime

#     created_files = []
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     try:
#         # Export basic results
#         basic_file = os.path.join(output_dir, f"analysis_results_{timestamp}.csv")

#         with open(basic_file, 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)

#             # Header
#             writer.writerow(["# MATLAB-Compatible Analysis Results"])
#             writer.writerow([f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
#             writer.writerow([f"# Method: {analysis_results.get('method', 'unknown')}"])
#             writer.writerow([])

#             # Parameters
#             writer.writerow(["# Analysis Parameters"])
#             params = analysis_results.get('parameters', {})
#             for key, value in params.items():
#                 writer.writerow([key, str(value)])
#             writer.writerow([])

#             # ROI summary
#             writer.writerow(["# ROI Summary"])
#             writer.writerow(["ROI_ID", "Baseline_Mean", "Upper_Threshold", "Lower_Threshold"])

#             baseline_means = analysis_results.get('baseline_means', {})
#             upper_thresholds = analysis_results.get('upper_thresholds', {})
#             lower_thresholds = analysis_results.get('lower_thresholds', {})

#             for roi in sorted(baseline_means.keys()):
#                 writer.writerow([
#                     roi,
#                     baseline_means.get(roi, 0),
#                     upper_thresholds.get(roi, 0),
#                     lower_thresholds.get(roi, 0)
#                 ])

#         created_files.append(basic_file)

#         # Export time series data if available
#         movement_data = analysis_results.get('movement_data', {})
#         if movement_data:
#             movement_file = os.path.join(output_dir, f"movement_data_{timestamp}.csv")

#             with open(movement_file, 'w', newline='', encoding='utf-8') as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(["Time_Seconds", "ROI_ID", "Movement_Binary"])

#                 for roi, data in movement_data.items():
#                     for time_sec, movement in data:
#                         writer.writerow([time_sec, roi, int(movement)])

#             created_files.append(movement_file)

#         return created_files

#     except Exception as e:
#         print(f"Error exporting MATLAB results: {e}")
#         return []


# def quick_method_test(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     methods: List[str] = None
# ) -> str:
#     """Quick test of analysis methods."""
#     if methods is None:
#         methods = ['baseline']  # Only baseline for quick test

#     if not merged_results:
#         return "No input data provided for testing"

#     results = []

#     for method in methods:
#         try:
#             # Test with minimal parameters
#             if method == 'baseline':
#                 kwargs = {
#                     'baseline_duration_minutes': 10.0,  # Small for testing
#                     'multiplier': 1.0,
#                     'frame_interval': 5.0,
#                 }
#             else:
#                 continue  # Skip other methods for now

#             # Run analysis
#             analysis_results = run_analysis_with_method(merged_results, method, **kwargs)

#             if analysis_results and 'baseline_means' in analysis_results:
#                 roi_count = len(analysis_results['baseline_means'])
#                 results.append(f"{method}: SUCCESS ({roi_count} ROIs)")
#             else:
#                 results.append(f"{method}: FAILED (no results)")

#         except Exception as e:
#             results.append(f"{method}: ERROR ({str(e)[:50]})")

#     return "\n".join(results)


# # Backward compatibility functions
# def integrate_analysis_with_widget(widget) -> bool:
#     """Main integration function that routes to appropriate analysis method."""
#     try:
#         if not hasattr(widget, 'merged_results') or not widget.merged_results:
#             widget._log_message("No merged_results available for analysis")
#             return False

#         # Determine method from widget
#         method_text = widget.threshold_method.currentText()

#         if "Baseline" in method_text:
#             from ._calc import integrate_baseline_analysis_with_widget
#             return integrate_baseline_analysis_with_widget(widget)

#         elif "Adaptive" in method_text:
#             from ._calc_adaptive import integrate_adaptive_analysis_with_widget
#             return integrate_adaptive_analysis_with_widget(widget)

#         elif "Calibration" in method_text:
#             from ._calc_calibration import integrate_calibration_analysis_with_widget
#             return integrate_calibration_analysis_with_widget(widget)

#         else:
#             widget._log_message(f"Unknown threshold method: {method_text}")
#             return False

#     except Exception as e:
#         widget._log_message(f"Analysis integration failed: {str(e)}")
#         return False


# # Legacy aliases for backward compatibility
# integrate_hdf5_analysis_with_widget = integrate_analysis_with_widget
# quick_analysis_test = quick_method_test
# """
# _calc_integration.py - Complete integration module with centralized preprocessing

# This module provides centralized preprocessing and routing between analysis methods.
# All preprocessing (MATLAB norm, detrending, jump correction) happens here to ensure
# consistency across all analysis methods.

# Architecture:
# 1. Centralized preprocessing pipeline
# 2. Method routing to simplified analysis functions
# 3. Comprehensive validation and error handling
# 4. Performance metrics and metadata tracking
# """

# from typing import Dict, List, Tuple, Optional, Any
# import os
# import time
# import numpy as np


# # =============================================================================
# # CENTRALIZED PREPROCESSING FUNCTIONS
# # =============================================================================

# def apply_matlab_normalization_to_merged_results(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     enable_matlab_norm: bool = True
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """
#     Apply MATLAB-style normalization: subtract minimum per ROI.

#     Args:
#         merged_results: Raw intensity data per ROI
#         enable_matlab_norm: Whether to apply normalization

#     Returns:
#         Normalized data with same structure
#     """
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


# def apply_improved_detrending(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     use_improved_detrending: bool = True
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """
#     Apply improved detrending to remove linear and polynomial drift.

#     Args:
#         merged_results: Input data per ROI
#         use_improved_detrending: Use polynomial + linear detrending

#     Returns:
#         Detrended data with same structure
#     """
#     detrended_results = {}

#     for roi, data in merged_results.items():
#         if not data or len(data) < 20:
#             detrended_results[roi] = data
#             continue

#         try:
#             sorted_data = sorted(data, key=lambda x: x[0])
#             times = np.array([t for t, _ in sorted_data])
#             values = np.array([val for _, val in sorted_data])

#             if use_improved_detrending:
#                 # Remove polynomial trend (handles curved drift)
#                 if len(values) >= 10:
#                     poly_coeffs = np.polyfit(times, values, 2)
#                     poly_trend = np.polyval(poly_coeffs, times)
#                     values_detrended = values - poly_trend + np.mean(poly_trend)
#                 else:
#                     values_detrended = values

#                 # Remove any remaining linear drift
#                 if len(values_detrended) >= 10:
#                     slope, intercept = np.polyfit(times, values_detrended, 1)
#                     total_drift = abs(slope * (times[-1] - times[0]))
#                     drift_percentage = (total_drift / np.mean(values)) * 100 if np.mean(values) > 0 else 0

#                     if drift_percentage > 1.0:  # Only remove if > 1% drift
#                         linear_trend = slope * times + intercept
#                         values_final = values_detrended - (linear_trend - intercept)
#                     else:
#                         values_final = values_detrended
#                 else:
#                     values_final = values_detrended
#             else:
#                 # Simple linear detrending only
#                 if len(values) >= 10:
#                     slope, intercept = np.polyfit(times, values, 1)
#                     linear_trend = slope * times + intercept
#                     values_final = values - (linear_trend - intercept)
#                 else:
#                     values_final = values

#             detrended_results[roi] = list(zip(times, values_final))

#         except Exception as e:
#             print(f"Detrending failed for ROI {roi}: {e}")
#             detrended_results[roi] = data

#     return detrended_results


# def apply_jump_correction(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     sensitivity: float = 3.0
# ) -> Dict[int, List[Tuple[float, float]]]:
#     """
#     Apply jump correction to detect and correct sudden intensity jumps/plateaus.

#     Args:
#         merged_results: Input data per ROI
#         sensitivity: Sensitivity for jump detection (sigma multiplier)

#     Returns:
#         Jump-corrected data with same structure
#     """
#     corrected_results = {}

#     for roi, data in merged_results.items():
#         if not data or len(data) < 10:
#             corrected_results[roi] = data
#             continue

#         try:
#             sorted_data = sorted(data, key=lambda x: x[0])
#             times = np.array([t for t, _ in sorted_data])
#             values = np.array([val for _, val in sorted_data])

#             # Calculate first differences
#             diffs = np.diff(values)

#             if len(diffs) < 5:
#                 corrected_results[roi] = data
#                 continue

#             # Detect jumps using robust statistics
#             median_diff = np.median(np.abs(diffs))
#             mad = np.median(np.abs(diffs - np.median(diffs)))  # Median Absolute Deviation
#             threshold = sensitivity * mad * 1.4826  # Scale factor for normal distribution

#             # Fallback if MAD is too small
#             if threshold < median_diff * 0.1:
#                 threshold = sensitivity * np.std(diffs)

#             # Find jump points
#             jump_indices = np.where(np.abs(diffs) > threshold)[0]

#             if len(jump_indices) > 0:
#                 corrected_values = values.copy()

#                 for jump_idx in jump_indices:
#                     # Correct jump by interpolation
#                     if jump_idx > 0 and jump_idx < len(values) - 2:
#                         # Use median of surrounding values for robust correction
#                         surrounding_indices = []
#                         for offset in [-2, -1, 2, 3]:
#                             idx = jump_idx + offset
#                             if 0 <= idx < len(values):
#                                 surrounding_indices.append(idx)

#                         if len(surrounding_indices) >= 2:
#                             surrounding_values = values[surrounding_indices]
#                             corrected_values[jump_idx + 1] = np.median(surrounding_values)

#                 corrected_results[roi] = list(zip(times, corrected_values))
#             else:
#                 corrected_results[roi] = data

#         except Exception as e:
#             print(f"Jump correction failed for ROI {roi}: {e}")
#             corrected_results[roi] = data

#     return corrected_results


# def apply_preprocessing_pipeline(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     **kwargs
# ) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[str, Any]]:
#     """
#     Apply complete preprocessing pipeline in the correct order.

#     Processing Order:
#     1. MATLAB Normalization (subtract minimum per ROI)
#     2. Detrending (remove drift)
#     3. Jump Correction (fix sudden jumps/plateaus)

#     Args:
#         merged_results: Raw intensity data
#         **kwargs: Preprocessing parameters

#     Returns:
#         Tuple of (processed_data, preprocessing_metadata)
#     """
#     # Extract preprocessing parameters
#     enable_matlab_norm = kwargs.get('enable_matlab_norm', True)
#     enable_detrending = kwargs.get('enable_detrending', False)
#     enable_jump_correction = kwargs.get('enable_jump_correction', False)
#     use_improved_detrending = kwargs.get('use_improved_detrending', True)

#     # Track what processing was applied
#     processing_steps = []
#     processing_metadata = {
#         'steps_applied': [],
#         'parameters_used': {},
#         'processing_order': [],
#         'roi_count': len(merged_results),
#         'original_data_points': sum(len(data) for data in merged_results.values())
#     }

#     # Start with original data
#     current_data = merged_results

#     # Step 1: MATLAB Normalization
#     if enable_matlab_norm:
#         print("Applying MATLAB normalization...")
#         current_data = apply_matlab_normalization_to_merged_results(current_data, True)
#         processing_steps.append('matlab_normalization')
#         processing_metadata['steps_applied'].append('matlab_normalization')
#         processing_metadata['parameters_used']['matlab_normalization'] = {'enabled': True}
#         processing_metadata['processing_order'].append('1_matlab_normalization')

#     # Step 2: Detrending
#     if enable_detrending:
#         detrend_type = 'improved' if use_improved_detrending else 'linear'
#         print(f"Applying {detrend_type} detrending...")
#         current_data = apply_improved_detrending(current_data, use_improved_detrending)
#         processing_steps.append(f'{detrend_type}_detrending')
#         processing_metadata['steps_applied'].append('detrending')
#         processing_metadata['parameters_used']['detrending'] = {
#             'enabled': True,
#             'type': detrend_type,
#             'use_improved': use_improved_detrending
#         }
#         processing_metadata['processing_order'].append(f'2_{detrend_type}_detrending')

#     # Step 3: Jump Correction
#     if enable_jump_correction:
#         jump_sensitivity = kwargs.get('jump_sensitivity', 3.0)
#         print(f"Applying jump correction (sensitivity: {jump_sensitivity})...")
#         current_data = apply_jump_correction(current_data, jump_sensitivity)
#         processing_steps.append('jump_correction')
#         processing_metadata['steps_applied'].append('jump_correction')
#         processing_metadata['parameters_used']['jump_correction'] = {
#             'enabled': True,
#             'sensitivity': jump_sensitivity
#         }
#         processing_metadata['processing_order'].append('3_jump_correction')

#     # Final metadata
#     processing_metadata.update({
#         'total_steps': len(processing_steps),
#         'processing_summary': ' → '.join(processing_steps) if processing_steps else 'no_preprocessing',
#         'final_data_points': sum(len(data) for data in current_data.values())
#     })

#     print(f"Preprocessing complete: {processing_metadata['processing_summary']}")

#     return current_data, processing_metadata


# # =============================================================================
# # PARAMETER VALIDATION
# # =============================================================================

# def validate_preprocessing_params(method: str, **kwargs) -> Dict[str, Any]:
#     """
#     Validate and standardize preprocessing parameters across all methods.

#     Args:
#         method: Analysis method name
#         **kwargs: Parameters to validate

#     Returns:
#         Validated and standardized parameters
#     """
#     # Required parameters for all methods
#     required_params = ['frame_interval']

#     # Check for missing required parameters
#     missing_required = [p for p in required_params if p not in kwargs]
#     if missing_required:
#         raise ValueError(f"Missing required parameters: {missing_required}")

#     # Standardize parameter names and defaults
#     standardized = kwargs.copy()

#     # Ensure boolean flags have correct defaults
#     standardized.setdefault('enable_matlab_norm', True)
#     standardized.setdefault('enable_detrending', False)
#     standardized.setdefault('enable_jump_correction', False)
#     standardized.setdefault('use_improved_detrending', True)
#     standardized.setdefault('jump_sensitivity', 3.0)

#     # Validate parameter types and ranges
#     try:
#         standardized['frame_interval'] = float(standardized['frame_interval'])
#         if standardized['frame_interval'] <= 0:
#             raise ValueError("Frame interval must be positive")
#     except (ValueError, TypeError):
#         raise ValueError("Frame interval must be a positive number")

#     try:
#         standardized['jump_sensitivity'] = float(standardized['jump_sensitivity'])
#         if standardized['jump_sensitivity'] <= 0:
#             standardized['jump_sensitivity'] = 3.0
#     except (ValueError, TypeError):
#         standardized['jump_sensitivity'] = 3.0

#     # Log preprocessing configuration
#     preprocessing_steps = []
#     if standardized.get('enable_matlab_norm', False):
#         preprocessing_steps.append("MATLAB normalization")
#     if standardized.get('enable_detrending', False):
#         detrend_type = "improved" if standardized.get('use_improved_detrending', False) else "linear"
#         preprocessing_steps.append(f"{detrend_type} detrending")
#     if standardized.get('enable_jump_correction', False):
#         sensitivity = standardized.get('jump_sensitivity', 3.0)
#         preprocessing_steps.append(f"jump correction (σ={sensitivity})")

#     preprocessing_info = ', '.join(preprocessing_steps) if preprocessing_steps else 'none'
#     print(f"Preprocessing pipeline: {preprocessing_info}")

#     return standardized


# # =============================================================================
# # MAIN ANALYSIS ROUTING WITH CENTRALIZED PREPROCESSING
# # =============================================================================

# def run_analysis_with_method(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     method: str,
#     **kwargs
# ) -> Dict[str, Any]:
#     """
#     Main analysis routing function with centralized preprocessing.

#     Args:
#         merged_results: Raw frame difference data from Reader
#         method: Analysis method ('baseline', 'adaptive', 'calibration')
#         **kwargs: Method-specific parameters

#     Returns:
#         Complete analysis results dictionary with preprocessing metadata
#     """
#     if not merged_results:
#         raise ValueError("No input data provided for analysis")

#     # Validate and standardize preprocessing parameters
#     try:
#         validated_kwargs = validate_preprocessing_params(method, **kwargs)
#     except Exception as e:
#         raise ValueError(f"Parameter validation failed: {e}")

#     print(f"Running {method.upper()} analysis with {len(merged_results)} ROIs")

#     # Apply centralized preprocessing pipeline
#     try:
#         preprocessed_data, preprocessing_metadata = apply_preprocessing_pipeline(
#             merged_results, **validated_kwargs
#         )
#     except Exception as e:
#         raise RuntimeError(f"Preprocessing failed: {e}")

#     # Route to appropriate analysis method (now preprocessing-free)
#     try:
#         if method.lower() == 'baseline':
#             from ._calc import run_baseline_analysis_pure
#             results = run_baseline_analysis_pure(preprocessed_data, **validated_kwargs)

#         elif method.lower() == 'adaptive':
#             from ._calc_adaptive import run_adaptive_analysis_pure
#             results = run_adaptive_analysis_pure(preprocessed_data, **validated_kwargs)

#         elif method.lower() == 'calibration':
#             # Check if we have pre-computed calibration baseline
#             if 'calibration_baseline_statistics' in validated_kwargs:
#                 from ._calc_calibration import run_calibration_analysis_with_precomputed_baseline_pure
#                 results = run_calibration_analysis_with_precomputed_baseline_pure(
#                     preprocessed_data, **validated_kwargs
#                 )
#             else:
#                 from ._calc_calibration import run_calibration_analysis_pure
#                 results = run_calibration_analysis_pure(preprocessed_data, **validated_kwargs)

#         else:
#             raise ValueError(f"Unknown analysis method: {method}. "
#                            f"Supported methods: 'baseline', 'adaptive', 'calibration'")

#         # Add comprehensive metadata to results
#         if isinstance(results, dict):
#             results['preprocessing_metadata'] = preprocessing_metadata
#             results['integration_version'] = '3.0_centralized_preprocessing'
#             results['method'] = method.lower()
#             results['processed_data'] = preprocessed_data  # Include preprocessed data

#             # Add preprocessing summary to parameters
#             if 'parameters' not in results:
#                 results['parameters'] = {}
#             results['parameters'].update({
#                 'preprocessing_applied': preprocessing_metadata['steps_applied'],
#                 'preprocessing_order': preprocessing_metadata['processing_order'],
#                 'preprocessing_summary': preprocessing_metadata['processing_summary']
#             })

#         # Validate results
#         _validate_analysis_results(results, method)

#         return results

#     except ImportError as e:
#         raise ImportError(f"Analysis method '{method}' module not available: {e}")
#     except Exception as e:
#         raise RuntimeError(f"Analysis failed for method '{method}': {e}")


# def _validate_analysis_results(results: Dict[str, Any], method: str) -> None:
#     """Validate that analysis results contain expected components."""
#     required_keys = ['baseline_means', 'upper_thresholds', 'lower_thresholds', 'method']

#     missing_keys = [key for key in required_keys if key not in results]
#     if missing_keys:
#         raise ValueError(f"Analysis results missing required keys: {missing_keys}")

#     # Check that we have data for at least one ROI
#     if not results.get('baseline_means'):
#         raise ValueError("Analysis produced no ROI results")

#     roi_count = len(results['baseline_means'])
#     print(f"Analysis completed successfully: {roi_count} ROIs processed")


# # =============================================================================
# # ANALYSIS SUMMARY AND DIAGNOSTICS
# # =============================================================================

# def get_analysis_summary(analysis_results: Dict[str, Any]) -> str:
#     """Generate comprehensive summary string for any analysis method."""
#     try:
#         method = analysis_results.get('method', 'unknown')
#         params = analysis_results.get('parameters', {})
#         roi_count = len(analysis_results.get('baseline_means', {}))

#         # Get preprocessing information
#         preprocessing_meta = analysis_results.get('preprocessing_metadata', {})
#         preprocessing_summary = preprocessing_meta.get('processing_summary', 'none')
#         preprocessing_order = preprocessing_meta.get('processing_order', [])

#         # Calculate movement statistics
#         movement_data = analysis_results.get('movement_data', {})
#         if movement_data:
#             movement_percentages = []
#             for roi_data in movement_data.values():
#                 if roi_data:
#                     movement_pct = np.mean([m for _, m in roi_data]) * 100
#                     movement_percentages.append(movement_pct)
#             avg_movement = np.mean(movement_percentages) if movement_percentages else 0
#         else:
#             avg_movement = 0

#         # Calculate sleep statistics
#         sleep_data = analysis_results.get('sleep_data', {})
#         if sleep_data:
#             sleep_percentages = []
#             for roi_data in sleep_data.values():
#                 if roi_data:
#                     sleep_pct = np.mean([s for _, s in roi_data]) * 100
#                     sleep_percentages.append(sleep_pct)
#             avg_sleep = np.mean(sleep_percentages) if sleep_percentages else 0
#         else:
#             avg_sleep = 0

#         # Calculate data quality metrics
#         baseline_means = analysis_results.get('baseline_means', {})
#         if baseline_means:
#             mean_values = list(baseline_means.values())
#             baseline_consistency = np.std(mean_values) / np.mean(mean_values) if mean_values else 0
#         else:
#             baseline_consistency = 0

#         # Method-specific information
#         method_info = ""
#         if method == 'baseline':
#             multiplier = params.get('multiplier', 'unknown')
#             baseline_duration = params.get('baseline_duration_minutes', 'unknown')
#             method_info = f"Multiplier: {multiplier}, Baseline duration: {baseline_duration} min"

#         elif method == 'adaptive':
#             base_multiplier = params.get('base_multiplier', 'unknown')
#             duration = params.get('analysis_duration_frames', 'unknown')
#             method_info = f"Base multiplier: {base_multiplier}, Analysis frames: {duration}"

#         elif method == 'calibration':
#             cal_multiplier = params.get('calibration_multiplier', 'unknown')
#             precomputed = 'Yes' if 'calibration_baseline_statistics' in params else 'No'
#             method_info = f"Calibration multiplier: {cal_multiplier}, Pre-computed baseline: {precomputed}"

#         # Quality assessment
#         quality_indicators = []
#         if baseline_consistency < 0.1:
#             quality_indicators.append("Good baseline consistency")
#         elif baseline_consistency < 0.3:
#             quality_indicators.append("Moderate baseline consistency")
#         else:
#             quality_indicators.append("High baseline variability")

#         if 0 < avg_movement < 100:
#             quality_indicators.append("Reasonable movement levels")
#         elif avg_movement == 0:
#             quality_indicators.append("No movement detected")
#         else:
#             quality_indicators.append("High movement activity")

#         quality_text = ', '.join(quality_indicators)

#         # Data processing statistics
#         original_points = preprocessing_meta.get('original_data_points', 0)
#         final_points = preprocessing_meta.get('final_data_points', 0)
#         processing_efficiency = (final_points / original_points * 100) if original_points > 0 else 100

#         summary = f"""Analysis Summary:
# Method: {method.upper()}
# ROIs analyzed: {roi_count}
# {method_info}

# Preprocessing Pipeline: {preprocessing_summary}
# Processing steps: {len(preprocessing_order)} steps applied
# Data retention: {processing_efficiency:.1f}% ({final_points}/{original_points} points)

# Results:
# - Average movement: {avg_movement:.1f}%
# - Average sleep: {avg_sleep:.1f}%
# - Baseline consistency (CV): {baseline_consistency:.3f}
# - Quality indicators: {quality_text}

# Status: {'Successful' if roi_count > 0 else 'Failed'}
# Integration version: {analysis_results.get('integration_version', '1.0')}"""

#         return summary

#     except Exception as e:
#         return f"Error generating summary: {e}"


# def validate_hdf5_timing_in_data(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     frame_interval: float = 5.0
# ) -> Dict[str, Any]:
#     """Enhanced HDF5 timing validation with comprehensive diagnostics."""
#     if not merged_results:
#         return {"timing_type": "no_data", "needs_correction": False}

#     try:
#         # Get sample data from first ROI
#         first_roi_data = next(iter(merged_results.values()))
#         if len(first_roi_data) < 3:
#             return {"timing_type": "insufficient_data", "needs_correction": False}

#         # Calculate actual intervals from more data points for better accuracy
#         sample_size = min(50, len(first_roi_data))
#         times = [t for t, _ in first_roi_data[:sample_size]]
#         intervals = [times[i+1] - times[i] for i in range(len(times)-1)]

#         avg_interval = np.mean(intervals)
#         interval_std = np.std(intervals)
#         median_interval = np.median(intervals)

#         # Enhanced tolerance calculations
#         tolerance = max(1.0, frame_interval * 0.1)  # 10% tolerance
#         tight_tolerance = frame_interval * 0.05     # 5% for consistency check

#         needs_correction = abs(avg_interval - frame_interval) > tolerance
#         interval_consistent = interval_std < tight_tolerance

#         # Detect drift patterns
#         drift_trend = 0
#         if len(intervals) > 10:
#             # Linear regression to detect systematic drift
#             x = np.arange(len(intervals))
#             drift_trend = np.polyfit(x, intervals, 1)[0]  # Slope

#         # Categorize timing quality
#         if interval_std < frame_interval * 0.01:
#             timing_quality = "excellent"
#         elif interval_std < frame_interval * 0.03:
#             timing_quality = "good"
#         elif interval_std < frame_interval * 0.05:
#             timing_quality = "acceptable"
#         else:
#             timing_quality = "poor"

#         return {
#             "timing_type": "hdf5_analysis",
#             "first_time": times[0],
#             "last_time": times[-1],
#             "total_duration": times[-1] - times[0],
#             "sample_size": sample_size,
#             "avg_interval": avg_interval,
#             "median_interval": median_interval,
#             "expected_interval": frame_interval,
#             "interval_std": interval_std,
#             "interval_consistent": interval_consistent,
#             "timing_quality": timing_quality,
#             "drift_trend": drift_trend,
#             "needs_hdf5_correction": needs_correction,
#             "recommended_action": "Apply timing correction" if needs_correction else "No correction needed",
#             "diagnostics": {
#                 "deviation_from_expected": avg_interval - frame_interval,
#                 "coefficient_of_variation": interval_std / avg_interval if avg_interval > 0 else 0,
#                 "max_interval": np.max(intervals),
#                 "min_interval": np.min(intervals)
#             }
#         }

#     except Exception as e:
#         return {"timing_type": "error", "needs_correction": False, "error": str(e)}


# def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
#     """Calculate comprehensive performance metrics for analysis."""
#     try:
#         import psutil

#         elapsed_time = time.time() - start_time
#         fps = total_frames / elapsed_time if elapsed_time > 0 else 0

#         cpu_percent = psutil.cpu_percent(interval=None)
#         memory_info = psutil.virtual_memory()
#         memory_percent = memory_info.percent
#         memory_available_gb = memory_info.available / (1024**3)

#         # Performance categorization
#         if fps > 1000:
#             performance_rating = "excellent"
#         elif fps > 500:
#             performance_rating = "good"
#         elif fps > 100:
#             performance_rating = "acceptable"
#         else:
#             performance_rating = "slow"

#         return {
#             'elapsed_time': elapsed_time,
#             'fps': fps,
#             'cpu_percent': cpu_percent,
#             'memory_percent': memory_percent,
#             'memory_available_gb': memory_available_gb,
#             'total_frames': total_frames,
#             'performance_rating': performance_rating,
#             'frames_per_second_per_core': fps / psutil.cpu_count() if psutil.cpu_count() > 0 else fps
#         }
#     except ImportError:
#         elapsed_time = time.time() - start_time
#         fps = total_frames / elapsed_time if elapsed_time > 0 else 0

#         performance_rating = "good" if fps > 100 else "acceptable" if fps > 50 else "slow"

#         return {
#             'elapsed_time': elapsed_time,
#             'fps': fps,
#             'cpu_percent': 0,
#             'memory_percent': 0,
#             'memory_available_gb': 0,
#             'total_frames': total_frames,
#             'performance_rating': performance_rating,
#             'frames_per_second_per_core': fps
#         }


# # =============================================================================
# # EXPORT FUNCTIONS
# # =============================================================================

# def export_results_for_matlab(analysis_results: Dict[str, Any], output_dir: str) -> List[str]:
#     """Export analysis results in comprehensive MATLAB-compatible format."""
#     import csv
#     import json
#     from datetime import datetime

#     created_files = []
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     method = analysis_results.get('method', 'unknown')

#     try:
#         # 1. Export basic results with comprehensive preprocessing information
#         basic_file = os.path.join(output_dir, f"analysis_results_{method}_{timestamp}.csv")

#         with open(basic_file, 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)

#             # Enhanced header with preprocessing info
#             writer.writerow(["# MATLAB-Compatible Analysis Results"])
#             writer.writerow([f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
#             writer.writerow([f"# Method: {method}"])
#             writer.writerow([f"# Integration Version: {analysis_results.get('integration_version', '1.0')}"])
#             writer.writerow([])

#             # Comprehensive preprocessing information
#             preprocessing_meta = analysis_results.get('preprocessing_metadata', {})
#             if preprocessing_meta:
#                 writer.writerow(["# Centralized Preprocessing Pipeline"])
#                 writer.writerow([f"# Processing Summary: {preprocessing_meta.get('processing_summary', 'none')}"])
#                 writer.writerow([f"# Steps Applied: {len(preprocessing_meta.get('steps_applied', []))}"])
#                 writer.writerow([f"# Processing Order: {' → '.join(preprocessing_meta.get('processing_order', []))}"])
#                 writer.writerow([f"# Original Data Points: {preprocessing_meta.get('original_data_points', 0)}"])
#                 writer.writerow([f"# Final Data Points: {preprocessing_meta.get('final_data_points', 0)}"])

#                 # Parameter details for each step
#                 params_used = preprocessing_meta.get('parameters_used', {})
#                 for step, step_params in params_used.items():
#                     writer.writerow([f"# {step}_parameters", str(step_params)])
#                 writer.writerow([])

#             # Analysis parameters
#             writer.writerow(["# Analysis Parameters"])
#             params = analysis_results.get('parameters', {})
#             for key, value in params.items():
#                 if key != 'calibration_baseline_statistics':  # Skip large nested data
#                     writer.writerow([key, str(value)])
#             writer.writerow([])

#             # ROI summary with enhanced metrics
#             writer.writerow(["# ROI Summary"])
#             writer.writerow(["ROI_ID", "Baseline_Mean", "Upper_Threshold", "Lower_Threshold", "Band_Width", "Quality_Score"])

#             baseline_means = analysis_results.get('baseline_means', {})
#             upper_thresholds = analysis_results.get('upper_thresholds', {})
#             lower_thresholds = analysis_results.get('lower_thresholds', {})

#             for roi in sorted(baseline_means.keys()):
#                 baseline = baseline_means.get(roi, 0)
#                 upper = upper_thresholds.get(roi, 0)
#                 lower = lower_thresholds.get(roi, 0)
#                 band_width = upper - lower

#                 # Simple quality score based on band width relative to baseline
#                 quality_score = band_width / baseline if baseline > 0 else 0

#                 writer.writerow([
#                     roi, baseline, upper, lower, band_width, f"{quality_score:.3f}"
#                 ])

#         created_files.append(basic_file)

#         # 2. Export comprehensive time series data
#         time_series_file = os.path.join(output_dir, f"timeseries_data_{method}_{timestamp}.csv")

#         # Combine all time series data types
#         all_data_types = [
#             ('movement_data', 'Movement_Binary'),
#             ('fraction_data', 'Fraction_Movement'),
#             ('sleep_data', 'Sleep_Binary'),
#             ('quiescence_data', 'Quiescence_Binary')
#         ]

#         with open(time_series_file, 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["Time_Seconds", "ROI_ID", "Data_Type", "Value"])

#             for data_key, data_type in all_data_types:
#                 data_dict = analysis_results.get(data_key, {})
#                 for roi, data_list in data_dict.items():
#                     for time_sec, value in data_list:
#                         writer.writerow([time_sec, roi, data_type, value])

#         created_files.append(time_series_file)

#         # 3. Export comprehensive metadata as JSON
#         metadata_file = os.path.join(output_dir, f"analysis_metadata_{method}_{timestamp}.json")

#         metadata = {
#             'analysis_info': {
#                 'method': method,
#                 'integration_version': analysis_results.get('integration_version', '1.0'),
#                 'generation_time': datetime.now().isoformat(),
#                 'roi_count': len(baseline_means)
#             },
#             'preprocessing_pipeline': analysis_results.get('preprocessing_metadata', {}),
#             'analysis_parameters': {k: v for k, v in analysis_results.get('parameters', {}).items()
#                                   if k != 'calibration_baseline_statistics'},  # Exclude large data
#             'performance_summary': {
#                 'data_retention_percent': (
#                     preprocessing_meta.get('final_data_points', 0) /
#                     preprocessing_meta.get('original_data_points', 1) * 100
#                 ) if preprocessing_meta.get('original_data_points', 0) > 0 else 100
#             }
#         }

#         with open(metadata_file, 'w') as f:
#             json.dump(metadata, f, indent=2, default=str)

#         created_files.append(metadata_file)

#         return created_files

#     except Exception as e:
#         print(f"Error exporting MATLAB results: {e}")
#         return created_files  # Return what we managed to create


# # =============================================================================
# # TESTING AND VALIDATION
# # =============================================================================

# def quick_method_test(
#     merged_results: Dict[int, List[Tuple[float, float]]],
#     methods: List[str] = None
# ) -> str:
#     """Enhanced quick test of analysis methods with preprocessing validation."""
#     if methods is None:
#         methods = ['baseline']  # Start with baseline for quick test

#     if not merged_results:
#         return "No input data provided for testing"

#     results = []
#     roi_count = len(merged_results)
#     sample_data_length = len(next(iter(merged_results.values()))) if merged_results else 0

#     results.append(f"Test data: {roi_count} ROIs, {sample_data_length} time points each")
#     results.append("")

#     for method in methods:
#         try:
#             # Test with comprehensive preprocessing enabled
#             if method == 'baseline':
#                 kwargs = {
#                     'baseline_duration_minutes': 10.0,
#                     'multiplier': 1.0,
#                     'frame_interval': 5.0,
#                     'enable_matlab_norm': True,
#                     'enable_detrending': True,
#                     'enable_jump_correction': True,
#                     'use_improved_detrending': True,
#                     'jump_sensitivity': 3.0
#                 }
#             elif method == 'adaptive':
#                 kwargs = {
#                     'adaptive_duration_minutes': 15.0,
#                     'adaptive_multiplier': 2.5,
#                     'frame_interval': 5.0,
#                     'enable_matlab_norm': True,
#                     'enable_detrending': True,
#                     'enable_jump_correction': False
#                 }
#             elif method == 'calibration':
#                 # Skip calibration test unless we have calibration data
#                 results.append(f"{method}: SKIPPED (requires calibration data)")
#                 continue
#             else:
#                 results.append(f"{method}: UNKNOWN METHOD")
#                 continue

#             # Run analysis with timing
#             start_time = time.time()
#             analysis_results = run_analysis_with_method(merged_results, method, **kwargs)
#             test_duration = time.time() - start_time

#             if analysis_results and 'baseline_means' in analysis_results:
#                 processed_roi_count = len(analysis_results['baseline_means'])
#                 fps = sample_data_length * roi_count / test_duration if test_duration > 0 else 0

#                 # Check preprocessing was applied
#                 preprocessing_meta = analysis_results.get('preprocessing_metadata', {})
#                 preprocessing_summary = preprocessing_meta.get('processing_summary', 'none')
#                 steps_applied = preprocessing_meta.get('steps_applied', [])

#                 results.append(f"{method.upper()}: SUCCESS")
#                 results.append(f"  - ROIs processed: {processed_roi_count}/{roi_count}")
#                 results.append(f"  - Processing speed: {fps:.1f} fps")
#                 results.append(f"  - Duration: {test_duration:.2f}s")
#                 results.append(f"  - Preprocessing: {preprocessing_summary}")
#                 results.append(f"  - Steps applied: {', '.join(steps_applied)}")
#             else:
#                 results.append(f"{method.upper()}: FAILED (no results returned)")

#         except Exception as e:
#             results.append(f"{method.upper()}: ERROR - {str(e)[:100]}")

#         results.append("")

#     return "\n".join(results)


# # =============================================================================
# # WIDGET INTEGRATION WITH CENTRALIZED PREPROCESSING
# # =============================================================================

# def integrate_analysis_with_widget(widget) -> bool:
#     """
#     Enhanced integration function with centralized preprocessing.
#     """
#     try:
#         # Validate widget state
#         if not hasattr(widget, 'merged_results') or not widget.merged_results:
#             widget._log_message("No merged_results available for analysis")
#             return False

#         if not hasattr(widget, 'threshold_method'):
#             widget._log_message("No threshold method selector found in widget")
#             return False

#         # Determine method from widget
#         method_text = widget.threshold_method.currentText()
#         widget._log_message(f"Starting centralized preprocessing + {method_text} analysis")

#         # Extract all parameters from widget
#         kwargs = {
#             'frame_interval': widget.frame_interval.value(),
#             'enable_matlab_norm': True,  # Always enabled
#             'enable_detrending': getattr(widget, 'enable_detrending', None),
#             'enable_jump_correction': getattr(widget, 'enable_jump_correction', None),
#             'use_improved_detrending': True,
#             'jump_sensitivity': 3.0,
#             'bin_size_seconds': widget.bin_size_seconds.value(),
#             'quiescence_threshold': widget.quiescence_threshold.value(),
#             'sleep_threshold_minutes': widget.sleep_threshold_minutes.value()
#         }

#         # Extract checkbox values safely
#         if hasattr(widget, 'enable_detrending') and hasattr(widget.enable_detrending, 'isChecked'):
#             kwargs['enable_detrending'] = widget.enable_detrending.isChecked()
#         else:
#             kwargs['enable_detrending'] = False

#         if hasattr(widget, 'enable_jump_correction') and hasattr(widget.enable_jump_correction, 'isChecked'):
#             kwargs['enable_jump_correction'] = widget.enable_jump_correction.isChecked()
#         else:
#             kwargs['enable_jump_correction'] = False

#         # Add method-specific parameters
#         if "Baseline" in method_text:
#             kwargs.update({
#                 'baseline_duration_minutes': widget.baseline_duration_minutes.value(),
#                 'multiplier': widget.threshold_multiplier.value()
#             })
#             method = 'baseline'

#         elif "Adaptive" in method_text:
#             kwargs.update({
#                 'adaptive_duration_minutes': widget.adaptive_duration_minutes.value(),
#                 'adaptive_multiplier': widget.adaptive_base_multiplier.value()
#             })
#             method = 'adaptive'

#         elif "Calibration" in method_text:
#             kwargs.update({
#                 'calibration_multiplier': widget.calibration_multiplier.value()
#             })
#             # Add calibration baseline if available
#             if hasattr(widget, 'calibration_baseline_statistics') and widget.calibration_baseline_statistics:
#                 kwargs['calibration_baseline_statistics'] = widget.calibration_baseline_statistics
#             method = 'calibration'

#         else:
#             widget._log_message(f"Unknown threshold method: {method_text}")
#             return False

#         # Run centralized analysis
#         widget._log_message("Running analysis with centralized preprocessing...")
#         analysis_results = run_analysis_with_method(widget.merged_results, method, **kwargs)

#         # Update widget with results
#         widget.merged_results = analysis_results.get('processed_data', widget.merged_results)
#         widget.roi_baseline_means = analysis_results.get('baseline_means', {})
#         widget.roi_upper_thresholds = analysis_results.get('upper_thresholds', {})
#         widget.roi_lower_thresholds = analysis_results.get('lower_thresholds', {})
#         widget.roi_statistics = analysis_results.get('roi_statistics', {})
#         widget.movement_data = analysis_results.get('movement_data', {})
#         widget.fraction_data = analysis_results.get('fraction_data', {})
#         widget.quiescence_data = analysis_results.get('quiescence_data', {})
#         widget.sleep_data = analysis_results.get('sleep_data', {})
#         widget.roi_colors = analysis_results.get('roi_colors', {})

#         # Calculate band widths for plotting
#         widget.roi_band_widths = {}
#         for roi in widget.roi_baseline_means:
#             if roi in widget.roi_upper_thresholds and roi in widget.roi_lower_thresholds:
#                 upper = widget.roi_upper_thresholds[roi]
#                 lower = widget.roi_lower_thresholds[roi]
#                 widget.roi_band_widths[roi] = (upper - lower) / 2

#         # Log preprocessing results
#         preprocessing_meta = analysis_results.get('preprocessing_metadata', {})
#         widget._log_message(f"Preprocessing applied: {preprocessing_meta.get('processing_summary', 'none')}")
#         widget._log_message(f"Analysis integration successful: {method} with centralized preprocessing")

#         return True

#     except Exception as e:
#         widget._log_message(f"Analysis integration failed: {str(e)}")
#         import traceback
#         widget._log_message(f"Traceback: {traceback.format_exc()}")
#         return False


# # =============================================================================
# # LEGACY COMPATIBILITY
# # =============================================================================

# # Legacy aliases for backward compatibility
# integrate_hdf5_analysis_with_widget = integrate_analysis_with_widget
# quick_analysis_test = quick_method_test
"""
_calc_integration.py - Simplified integration module for HDF5 analysis methods

This module provides routing between baseline, adaptive, and calibration methods
and handles method validation and performance metrics.
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import time
import numpy as np


# =============================================================================
# MAIN ANALYSIS ROUTING
# =============================================================================


def run_analysis_with_method(
    merged_results: Dict[int, List[Tuple[float, float]]], method: str, **kwargs
) -> Dict[str, Any]:
    """
    Simplified analysis routing function.

    Args:
        merged_results: Raw frame difference data from Reader
        method: Analysis method ('baseline', 'adaptive', 'calibration')
        **kwargs: Method-specific parameters

    Returns:
        Complete analysis results dictionary
    """
    if not merged_results:
        raise ValueError("No input data provided for analysis")

    print(f"Running {method.upper()} analysis with {len(merged_results)} ROIs")

    # Route to appropriate analysis method (each handles own preprocessing)
    try:
        if method.lower() == "baseline":
            from ._calc import run_baseline_analysis

            results = run_baseline_analysis(merged_results, **kwargs)

        elif method.lower() == "adaptive":
            from ._calc_adaptive import run_adaptive_analysis

            results = run_adaptive_analysis(merged_results, **kwargs)

        elif method.lower() == "calibration":
            # Check if we have pre-computed calibration baseline
            if "calibration_baseline_statistics" in kwargs:
                from ._calc_calibration import (
                    run_calibration_analysis_with_precomputed_baseline,
                )

                results = run_calibration_analysis_with_precomputed_baseline(
                    merged_results, **kwargs
                )
            else:
                from ._calc_calibration import run_calibration_analysis

                results = run_calibration_analysis(merged_results, **kwargs)

        else:
            raise ValueError(
                f"Unknown analysis method: {method}. "
                f"Supported methods: 'baseline', 'adaptive', 'calibration'"
            )

        # Add integration metadata
        if isinstance(results, dict):
            results["integration_version"] = "2.0_simplified_routing"
            results["method"] = method.lower()

        # Validate results
        _validate_analysis_results(results, method)

        return results

    except ImportError as e:
        raise ImportError(f"Analysis method '{method}' module not available: {e}")
    except Exception as e:
        raise RuntimeError(f"Analysis failed for method '{method}': {e}")


def _validate_analysis_results(results: Dict[str, Any], method: str) -> None:
    """Validate that analysis results contain expected components."""
    required_keys = ["baseline_means", "upper_thresholds", "lower_thresholds", "method"]

    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(f"Analysis results missing required keys: {missing_keys}")

    # Check that we have data for at least one ROI
    if not results.get("baseline_means"):
        raise ValueError("Analysis produced no ROI results")

    roi_count = len(results["baseline_means"])
    print(f"Analysis completed successfully: {roi_count} ROIs processed")


# =============================================================================
# ANALYSIS SUMMARY AND DIAGNOSTICS
# =============================================================================


def get_analysis_summary(analysis_results: Dict[str, Any]) -> str:
    """Generate summary string for any analysis method."""
    try:
        method = analysis_results.get("method", "unknown")
        params = analysis_results.get("parameters", {})
        roi_count = len(analysis_results.get("baseline_means", {}))

        # Calculate movement statistics
        movement_data = analysis_results.get("movement_data", {})
        if movement_data:
            movement_percentages = []
            for roi_data in movement_data.values():
                if roi_data:
                    movement_pct = np.mean([m for _, m in roi_data]) * 100
                    movement_percentages.append(movement_pct)
            avg_movement = np.mean(movement_percentages) if movement_percentages else 0
        else:
            avg_movement = 0

        # Calculate sleep statistics
        sleep_data = analysis_results.get("sleep_data", {})
        if sleep_data:
            sleep_percentages = []
            for roi_data in sleep_data.values():
                if roi_data:
                    sleep_pct = np.mean([s for _, s in roi_data]) * 100
                    sleep_percentages.append(sleep_pct)
            avg_sleep = np.mean(sleep_percentages) if sleep_percentages else 0
        else:
            avg_sleep = 0

        # Calculate data quality metrics
        baseline_means = analysis_results.get("baseline_means", {})
        if baseline_means:
            mean_values = list(baseline_means.values())
            baseline_consistency = (
                np.std(mean_values) / np.mean(mean_values) if mean_values else 0
            )
        else:
            baseline_consistency = 0

        # Method-specific information
        method_info = ""
        if method == "baseline":
            multiplier = params.get("multiplier", "unknown")
            baseline_duration = params.get("baseline_duration_minutes", "unknown")
            detrending = "Yes" if params.get("enable_detrending", False) else "No"
            method_info = f"Multiplier: {multiplier}, Baseline duration: {baseline_duration} min, Detrending: {detrending}"

        elif method == "adaptive":
            base_multiplier = params.get("base_multiplier", "unknown")
            duration = params.get("analysis_duration_frames", "unknown")
            method_info = (
                f"Base multiplier: {base_multiplier}, Analysis frames: {duration}"
            )

        elif method == "calibration":
            cal_multiplier = params.get("calibration_multiplier", "unknown")
            precomputed = "Yes" if "calibration_baseline_statistics" in params else "No"
            method_info = f"Calibration multiplier: {cal_multiplier}, Pre-computed baseline: {precomputed}"

        # Quality assessment
        quality_indicators = []
        if baseline_consistency < 0.1:
            quality_indicators.append("Good baseline consistency")
        elif baseline_consistency < 0.3:
            quality_indicators.append("Moderate baseline consistency")
        else:
            quality_indicators.append("High baseline variability")

        if 0 < avg_movement < 100:
            quality_indicators.append("Reasonable movement levels")
        elif avg_movement == 0:
            quality_indicators.append("No movement detected")
        else:
            quality_indicators.append("High movement activity")

        quality_text = ", ".join(quality_indicators)

        summary = f"""Analysis Summary:
Method: {method.upper()}
ROIs analyzed: {roi_count}
{method_info}
MATLAB normalization: {'Yes' if params.get('enable_matlab_norm', False) else 'No'}

Results:
- Average movement: {avg_movement:.1f}%
- Average sleep: {avg_sleep:.1f}%
- Baseline consistency (CV): {baseline_consistency:.3f}
- Quality indicators: {quality_text}

Status: {'Successful' if roi_count > 0 else 'Failed'}
Integration version: {analysis_results.get('integration_version', '1.0')}"""

        return summary

    except Exception as e:
        return f"Error generating summary: {e}"


def validate_hdf5_timing_in_data(
    merged_results: Dict[int, List[Tuple[float, float]]], frame_interval: float = 5.0
) -> Dict[str, Any]:
    """Enhanced HDF5 timing validation with comprehensive diagnostics."""
    if not merged_results:
        return {"timing_type": "no_data", "needs_correction": False}

    try:
        # Get sample data from first ROI
        first_roi_data = next(iter(merged_results.values()))
        if len(first_roi_data) < 3:
            return {"timing_type": "insufficient_data", "needs_correction": False}

        # Calculate actual intervals from more data points for better accuracy
        sample_size = min(50, len(first_roi_data))
        times = [t for t, _ in first_roi_data[:sample_size]]
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)
        median_interval = np.median(intervals)

        # Enhanced tolerance calculations
        tolerance = max(1.0, frame_interval * 0.1)  # 10% tolerance
        tight_tolerance = frame_interval * 0.05  # 5% for consistency check

        needs_correction = abs(avg_interval - frame_interval) > tolerance
        interval_consistent = interval_std < tight_tolerance

        # Detect drift patterns
        drift_trend = 0
        if len(intervals) > 10:
            # Linear regression to detect systematic drift
            x = np.arange(len(intervals))
            drift_trend = np.polyfit(x, intervals, 1)[0]  # Slope

        # Categorize timing quality
        if interval_std < frame_interval * 0.01:
            timing_quality = "excellent"
        elif interval_std < frame_interval * 0.03:
            timing_quality = "good"
        elif interval_std < frame_interval * 0.05:
            timing_quality = "acceptable"
        else:
            timing_quality = "poor"

        return {
            "timing_type": "hdf5_analysis",
            "first_time": times[0],
            "last_time": times[-1],
            "total_duration": times[-1] - times[0],
            "sample_size": sample_size,
            "avg_interval": avg_interval,
            "median_interval": median_interval,
            "expected_interval": frame_interval,
            "interval_std": interval_std,
            "interval_consistent": interval_consistent,
            "timing_quality": timing_quality,
            "drift_trend": drift_trend,
            "needs_hdf5_correction": needs_correction,
            "recommended_action": (
                "Apply timing correction"
                if needs_correction
                else "No correction needed"
            ),
            "diagnostics": {
                "deviation_from_expected": avg_interval - frame_interval,
                "coefficient_of_variation": (
                    interval_std / avg_interval if avg_interval > 0 else 0
                ),
                "max_interval": np.max(intervals),
                "min_interval": np.min(intervals),
            },
        }

    except Exception as e:
        return {"timing_type": "error", "needs_correction": False, "error": str(e)}


def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics for analysis."""
    try:
        import psutil

        elapsed_time = time.time() - start_time
        fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        memory_available_gb = memory_info.available / (1024**3)

        # Performance categorization
        if fps > 1000:
            performance_rating = "excellent"
        elif fps > 500:
            performance_rating = "good"
        elif fps > 100:
            performance_rating = "acceptable"
        else:
            performance_rating = "slow"

        return {
            "elapsed_time": elapsed_time,
            "fps": fps,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_gb": memory_available_gb,
            "total_frames": total_frames,
            "performance_rating": performance_rating,
            "frames_per_second_per_core": (
                fps / psutil.cpu_count() if psutil.cpu_count() > 0 else fps
            ),
        }
    except ImportError:
        elapsed_time = time.time() - start_time
        fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        performance_rating = (
            "good" if fps > 100 else "acceptable" if fps > 50 else "slow"
        )

        return {
            "elapsed_time": elapsed_time,
            "fps": fps,
            "cpu_percent": 0,
            "memory_percent": 0,
            "memory_available_gb": 0,
            "total_frames": total_frames,
            "performance_rating": performance_rating,
            "frames_per_second_per_core": fps,
        }


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_results_for_matlab(
    analysis_results: Dict[str, Any], output_dir: str
) -> List[str]:
    """Export analysis results in MATLAB-compatible format."""
    import csv
    import json
    from datetime import datetime

    created_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = analysis_results.get("method", "unknown")

    try:
        # Export basic results
        basic_file = os.path.join(
            output_dir, f"analysis_results_{method}_{timestamp}.csv"
        )

        with open(basic_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(["# MATLAB-Compatible Analysis Results"])
            writer.writerow(
                [f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
            )
            writer.writerow([f"# Method: {method}"])
            writer.writerow(
                [
                    f"# Integration Version: {analysis_results.get('integration_version', '1.0')}"
                ]
            )
            writer.writerow([])

            # Analysis parameters
            writer.writerow(["# Analysis Parameters"])
            params = analysis_results.get("parameters", {})
            for key, value in params.items():
                if key != "calibration_baseline_statistics":  # Skip large nested data
                    writer.writerow([key, str(value)])
            writer.writerow([])

            # ROI summary
            writer.writerow(["# ROI Summary"])
            writer.writerow(
                [
                    "ROI_ID",
                    "Baseline_Mean",
                    "Upper_Threshold",
                    "Lower_Threshold",
                    "Band_Width",
                ]
            )

            baseline_means = analysis_results.get("baseline_means", {})
            upper_thresholds = analysis_results.get("upper_thresholds", {})
            lower_thresholds = analysis_results.get("lower_thresholds", {})

            for roi in sorted(baseline_means.keys()):
                baseline = baseline_means.get(roi, 0)
                upper = upper_thresholds.get(roi, 0)
                lower = lower_thresholds.get(roi, 0)
                band_width = upper - lower

                writer.writerow([roi, baseline, upper, lower, band_width])

        created_files.append(basic_file)

        # Export time series data
        time_series_file = os.path.join(
            output_dir, f"timeseries_data_{method}_{timestamp}.csv"
        )

        # Combine all time series data types
        all_data_types = [
            ("movement_data", "Movement_Binary"),
            ("fraction_data", "Fraction_Movement"),
            ("sleep_data", "Sleep_Binary"),
            ("quiescence_data", "Quiescence_Binary"),
        ]

        with open(time_series_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time_Seconds", "ROI_ID", "Data_Type", "Value"])

            for data_key, data_type in all_data_types:
                data_dict = analysis_results.get(data_key, {})
                for roi, data_list in data_dict.items():
                    for time_sec, value in data_list:
                        writer.writerow([time_sec, roi, data_type, value])

        created_files.append(time_series_file)

        # Export metadata as JSON
        metadata_file = os.path.join(
            output_dir, f"analysis_metadata_{method}_{timestamp}.json"
        )

        metadata = {
            "analysis_info": {
                "method": method,
                "integration_version": analysis_results.get(
                    "integration_version", "1.0"
                ),
                "generation_time": datetime.now().isoformat(),
                "roi_count": len(baseline_means),
            },
            "analysis_parameters": {
                k: v
                for k, v in analysis_results.get("parameters", {}).items()
                if k != "calibration_baseline_statistics"
            },  # Exclude large data
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        created_files.append(metadata_file)

        return created_files

    except Exception as e:
        print(f"Error exporting MATLAB results: {e}")
        return created_files  # Return what we managed to create


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================


def quick_method_test(
    merged_results: Dict[int, List[Tuple[float, float]]], methods: List[str] = None
) -> str:
    """Quick test of analysis methods."""
    if methods is None:
        methods = ["baseline"]  # Start with baseline for quick test

    if not merged_results:
        return "No input data provided for testing"

    results = []
    roi_count = len(merged_results)
    sample_data_length = (
        len(next(iter(merged_results.values()))) if merged_results else 0
    )

    results.append(
        f"Test data: {roi_count} ROIs, {sample_data_length} time points each"
    )
    results.append("")

    for method in methods:
        try:
            # Test with minimal parameters
            if method == "baseline":
                kwargs = {
                    "baseline_duration_minutes": 10.0,
                    "multiplier": 1.0,
                    "frame_interval": 5.0,
                    "enable_matlab_norm": True,
                    "enable_detrending": False,  # Keep simple for testing
                }
            elif method == "adaptive":
                kwargs = {
                    "adaptive_duration_minutes": 15.0,
                    "adaptive_multiplier": 2.5,
                    "frame_interval": 5.0,
                    "enable_matlab_norm": True,
                    "enable_detrending": False,
                }
            elif method == "calibration":
                # Skip calibration test unless we have calibration data
                results.append(f"{method}: SKIPPED (requires calibration data)")
                continue
            else:
                results.append(f"{method}: UNKNOWN METHOD")
                continue

            # Run analysis with timing
            start_time = time.time()
            analysis_results = run_analysis_with_method(
                merged_results, method, **kwargs
            )
            test_duration = time.time() - start_time

            if analysis_results and "baseline_means" in analysis_results:
                processed_roi_count = len(analysis_results["baseline_means"])
                fps = (
                    sample_data_length * roi_count / test_duration
                    if test_duration > 0
                    else 0
                )

                results.append(f"{method.upper()}: SUCCESS")
                results.append(f"  - ROIs processed: {processed_roi_count}/{roi_count}")
                results.append(f"  - Processing speed: {fps:.1f} fps")
                results.append(f"  - Duration: {test_duration:.2f}s")
            else:
                results.append(f"{method.upper()}: FAILED (no results returned)")

        except Exception as e:
            results.append(f"{method.upper()}: ERROR - {str(e)[:100]}")

        results.append("")

    return "\n".join(results)


# =============================================================================
# WIDGET INTEGRATION
# =============================================================================


def integrate_analysis_with_widget(widget) -> bool:
    """
    Main integration function that routes to appropriate analysis method.
    """
    try:
        # Validate widget state
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message("No merged_results available for analysis")
            return False

        if not hasattr(widget, "threshold_method"):
            widget._log_message("No threshold method selector found in widget")
            return False

        # Determine method from widget
        method_text = widget.threshold_method.currentText()
        widget._log_message(f"Starting {method_text} analysis")

        # Extract parameters from widget
        kwargs = {
            "frame_interval": widget.frame_interval.value(),
            "enable_matlab_norm": True,  # Always enabled
            "bin_size_seconds": widget.bin_size_seconds.value(),
            "quiescence_threshold": widget.quiescence_threshold.value(),
            "sleep_threshold_minutes": widget.sleep_threshold_minutes.value(),
        }

        # Extract checkbox values safely
        if hasattr(widget, "enable_detrending") and hasattr(
            widget.enable_detrending, "isChecked"
        ):
            kwargs["enable_detrending"] = widget.enable_detrending.isChecked()
        else:
            kwargs["enable_detrending"] = False

        if hasattr(widget, "enable_jump_correction") and hasattr(
            widget.enable_jump_correction, "isChecked"
        ):
            kwargs["enable_jump_correction"] = widget.enable_jump_correction.isChecked()
        else:
            kwargs["enable_jump_correction"] = False

        # Add method-specific parameters
        if "Baseline" in method_text:
            kwargs.update(
                {
                    "baseline_duration_minutes": widget.baseline_duration_minutes.value(),
                    "multiplier": widget.threshold_multiplier.value(),
                }
            )
            method = "baseline"

        elif "Adaptive" in method_text:
            kwargs.update(
                {
                    "adaptive_duration_minutes": widget.adaptive_duration_minutes.value(),
                    "adaptive_multiplier": widget.adaptive_base_multiplier.value(),
                }
            )
            method = "adaptive"

        elif "Calibration" in method_text:
            kwargs.update(
                {"calibration_multiplier": widget.calibration_multiplier.value()}
            )
            # Add calibration baseline if available
            if (
                hasattr(widget, "calibration_baseline_statistics")
                and widget.calibration_baseline_statistics
            ):
                kwargs["calibration_baseline_statistics"] = (
                    widget.calibration_baseline_statistics
                )
            method = "calibration"

        else:
            widget._log_message(f"Unknown threshold method: {method_text}")
            return False

        # Run analysis
        widget._log_message("Running analysis...")
        analysis_results = run_analysis_with_method(
            widget.merged_results, method, **kwargs
        )

        # Update widget with results
        widget.merged_results = analysis_results.get(
            "processed_data", widget.merged_results
        )
        widget.roi_baseline_means = analysis_results.get("baseline_means", {})
        widget.roi_upper_thresholds = analysis_results.get("upper_thresholds", {})
        widget.roi_lower_thresholds = analysis_results.get("lower_thresholds", {})
        widget.roi_statistics = analysis_results.get("roi_statistics", {})
        widget.movement_data = analysis_results.get("movement_data", {})
        widget.fraction_data = analysis_results.get("fraction_data", {})
        widget.quiescence_data = analysis_results.get("quiescence_data", {})
        widget.sleep_data = analysis_results.get("sleep_data", {})
        widget.roi_colors = analysis_results.get("roi_colors", {})

        # Calculate band widths for plotting
        widget.roi_band_widths = {}
        for roi in widget.roi_baseline_means:
            if (
                roi in widget.roi_upper_thresholds
                and roi in widget.roi_lower_thresholds
            ):
                upper = widget.roi_upper_thresholds[roi]
                lower = widget.roi_lower_thresholds[roi]
                widget.roi_band_widths[roi] = (upper - lower) / 2

        widget._log_message(f"Analysis integration successful: {method}")

        return True

    except Exception as e:
        widget._log_message(f"Analysis integration failed: {str(e)}")
        import traceback

        widget._log_message(f"Traceback: {traceback.format_exc()}")
        return False


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Legacy aliases for backward compatibility
integrate_hdf5_analysis_with_widget = integrate_analysis_with_widget
quick_analysis_test = quick_method_test
