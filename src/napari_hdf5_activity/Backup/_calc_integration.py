"""
_calc_integration.py - Integration module for all HDF5 analysis calculation methods

This module provides a unified interface for all calculation methods and handles
the routing between baseline, adaptive, and calibration methods.

This module should be imported by the main widget to access all calculation
functionality through a single interface.
"""

from typing import Dict, List, Tuple, Optional, Any
import os

# Add these to the END of your _calc_integration.py file

# Import missing functions from _calc.py
try:
    from ._calc import (
        define_movement_with_hysteresis,
        bin_fraction_movement,
        define_sleep_periods,
        bin_activity_data_for_lighting,
        get_performance_metrics,
        bin_quiescence,
        run_complete_hdf5_compatible_analysis,
        process_with_matlab_compatibility,
        compute_roi_thresholds_unified,
        compute_roi_thresholds_hysteresis,
        integrate_baseline_analysis_with_widget as integrate_hdf5_analysis_with_widget,  # Alias
        test_baseline_analysis_direct as quick_analysis_test,  # Alias
    )

    print("✅ Successfully imported calculation functions from _calc.py")
except ImportError as e:
    print(f"⚠️ Could not import some functions from _calc.py: {e}")

# Import timing validation function
try:
    from ._calc import validate_hdf5_timing_in_data
except ImportError:

    def validate_hdf5_timing_in_data(merged_results, frame_interval=5.0):
        """Dummy function for compatibility."""
        return {"timing_type": "unknown", "needs_correction": False}


# Ensure backwards compatibility aliases
try:
    integrate_analysis_with_widget = integrate_hdf5_analysis_with_widget
    quick_method_test = quick_analysis_test
except NameError:
    print("⚠️ Some compatibility functions not available")


# Add the missing simple utility functions
def get_available_methods() -> List[str]:
    """Get list of available analysis methods."""
    return ["baseline", "adaptive", "calibration"]


def validate_method_parameters(method: str, **kwargs) -> Tuple[bool, str]:
    """Validate parameters for a specific analysis method."""
    if method.lower() not in ["baseline", "adaptive", "calibration"]:
        return False, f"Unknown method: {method}"

    # Basic validation
    if method.lower() == "baseline":
        if "baseline_duration_minutes" not in kwargs:
            return False, "baseline_duration_minutes required for baseline method"
        if kwargs.get("baseline_duration_minutes", 0) <= 0:
            return False, "baseline_duration_minutes must be positive"

    return True, ""


def run_analysis_with_method(
    merged_results: Dict[int, List[Tuple[float, float]]], method: str, **kwargs
) -> Dict[str, Any]:
    """
    Run analysis with the specified method.

    Args:
        merged_results: Raw frame difference data from Reader
        method: Analysis method ('baseline', 'adaptive', 'calibration')
        **kwargs: Method-specific parameters

    Returns:
        Complete analysis results dictionary
    """
    if method.lower() == "baseline":
        from ._calc import run_baseline_analysis

        return run_baseline_analysis(merged_results, **kwargs)

    elif method.lower() == "adaptive":
        from ._calc_adaptive import run_adaptive_analysis

        return run_adaptive_analysis(merged_results, **kwargs)

    elif method.lower() == "calibration":
        from ._calc_calibration import run_calibration_analysis

        return run_calibration_analysis(merged_results, **kwargs)

    else:
        raise ValueError(
            f"Unknown analysis method: {method}. "
            f"Supported methods: 'baseline', 'adaptive', 'calibration'"
        )


def integrate_analysis_with_widget(widget) -> bool:
    """
    Main integration function that routes to the appropriate analysis method.

    This function should be called from the main widget's analysis completion handler.

    Args:
        widget: The napari widget instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message("No merged_results available for analysis")
            return False

        # Determine method from widget
        method_text = widget.threshold_method.currentText()

        if "Baseline" in method_text:
            method = "baseline"
            from ._calc import integrate_baseline_analysis_with_widget

            return integrate_baseline_analysis_with_widget(widget)

        elif "Adaptive" in method_text:
            method = "adaptive"
            from ._calc_adaptive import integrate_adaptive_analysis_with_widget

            return integrate_adaptive_analysis_with_widget(widget)

        elif "Calibration" in method_text:
            method = "calibration"
            from ._calc_calibration import integrate_calibration_analysis_with_widget

            return integrate_calibration_analysis_with_widget(widget)

        else:
            widget._log_message(f"Unknown threshold method: {method_text}")
            return False

    except Exception as e:
        widget._log_message(f"Analysis integration failed: {str(e)}")
        import traceback

        widget._log_message(f"Traceback: {traceback.format_exc()}")
        return False


def get_method_requirements(method: str) -> Dict[str, Any]:
    """
    Get the requirements for a specific analysis method.

    Args:
        method: Analysis method name

    Returns:
        Dictionary with method requirements
    """
    requirements = {
        "baseline": {
            "required_parameters": [
                "threshold_block_count",
                "multiplier",
                "frame_interval",
            ],
            "optional_parameters": [
                "enable_matlab_norm",
                "enable_detrending",
                "enable_jump_correction",
            ],
            "required_data": ["merged_results"],
            "description": "Uses first N frames to establish baseline and calculate thresholds",
        },
        "adaptive": {
            "required_parameters": [
                "analysis_duration_frames",
                "base_multiplier",
                "frame_interval",
            ],
            "optional_parameters": ["enable_matlab_norm", "enable_detrending"],
            "required_data": ["merged_results"],
            "description": "Automatically adapts thresholds based on signal-to-noise ratio and variability",
        },
        "calibration": {
            "required_parameters": [
                "calibration_file_path",
                "masks",
                "percentile_threshold",
                "calibration_multiplier",
            ],
            "optional_parameters": [
                "enable_matlab_norm",
                "enable_detrending",
                "frame_interval",
            ],
            "required_data": ["merged_results", "calibration_file"],
            "description": "Uses sedated animal recordings to determine noise baseline",
        },
    }

    return requirements.get(method.lower(), {})


def validate_method_parameters(method: str, **kwargs) -> Tuple[bool, str]:
    """
    Validate parameters for a specific analysis method.

    Args:
        method: Analysis method name
        **kwargs: Parameters to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    requirements = get_method_requirements(method)

    if not requirements:
        return False, f"Unknown method: {method}"

    # Check required parameters
    required_params = requirements.get("required_parameters", [])
    missing_params = [param for param in required_params if param not in kwargs]

    if missing_params:
        return False, f"Missing required parameters for {method}: {missing_params}"

    # Method-specific validation
    if method.lower() == "baseline":
        if kwargs.get("threshold_block_count", 0) <= 0:
            return False, "threshold_block_count must be positive"
        if kwargs.get("multiplier", 0) <= 0:
            return False, "multiplier must be positive"

    elif method.lower() == "adaptive":
        if kwargs.get("analysis_duration_frames", 0) <= 0:
            return False, "analysis_duration_frames must be positive"
        if kwargs.get("base_multiplier", 0) <= 0:
            return False, "base_multiplier must be positive"

    elif method.lower() == "calibration":
        calibration_file = kwargs.get("calibration_file_path")
        if not calibration_file or not os.path.exists(calibration_file):
            return False, "Valid calibration_file_path is required"
        if not kwargs.get("masks"):
            return False, "ROI masks are required for calibration method"
        if not (0 <= kwargs.get("percentile_threshold", -1) <= 100):
            return False, "percentile_threshold must be between 0 and 100"

    return True, ""


def get_analysis_summary(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a unified summary string for any analysis method.

    Args:
        analysis_results: Results from any analysis method

    Returns:
        Formatted summary string
    """
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
                    import numpy as np

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
                    import numpy as np

                    sleep_pct = np.mean([s for _, s in roi_data]) * 100
                    sleep_percentages.append(sleep_pct)
            avg_sleep = np.mean(sleep_percentages) if sleep_percentages else 0
        else:
            avg_sleep = 0

        # Method-specific information
        method_info = ""
        if method == "baseline":
            multiplier = params.get("multiplier", "unknown")
            detrending = "✅" if params.get("enable_detrending", False) else "❌"
            method_info = f"Multiplier: {multiplier}, Detrending: {detrending}"

        elif method == "adaptive":
            base_multiplier = params.get("base_multiplier", "unknown")
            duration = params.get("analysis_duration_frames", "unknown")
            method_info = (
                f"Base multiplier: {base_multiplier}, Analysis frames: {duration}"
            )

        elif method == "calibration":
            percentile = params.get("percentile_threshold", "unknown")
            cal_multiplier = params.get("calibration_multiplier", "unknown")
            method_info = f"Percentile: {percentile}%, Multiplier: {cal_multiplier}"

        summary = f"""=== ANALYSIS SUMMARY ===
Method: {method.upper()}
ROIs analyzed: {roi_count}
{method_info}
MATLAB normalization: {'✅' if params.get('enable_matlab_norm', False) else '❌'}

Results:
• Average movement: {avg_movement:.1f}%
• Average sleep: {avg_sleep:.1f}%
• Analysis: {'Successful' if roi_count > 0 else 'Failed'}
========================"""

        return summary

    except Exception as e:
        return f"Error generating summary: {e}"


def export_method_comparison(
    results_dict: Dict[str, Dict[str, Any]], output_file: str
) -> bool:
    """
    Export comparison of results from different methods.

    Args:
        results_dict: Dictionary mapping method names to analysis results
        output_file: Path to output CSV file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import csv

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(["# Method Comparison Results"])
            writer.writerow(["# Generated:", "Timestamp"])
            writer.writerow([])

            # Method summary
            writer.writerow(
                ["Method", "ROIs_Analyzed", "Avg_Movement_%", "Avg_Sleep_%", "Status"]
            )

            for method_name, results in results_dict.items():
                roi_count = len(results.get("baseline_means", {}))

                # Calculate averages
                movement_data = results.get("movement_data", {})
                if movement_data:
                    import numpy as np

                    movement_percentages = []
                    for roi_data in movement_data.values():
                        if roi_data:
                            movement_pct = np.mean([m for _, m in roi_data]) * 100
                            movement_percentages.append(movement_pct)
                    avg_movement = (
                        np.mean(movement_percentages) if movement_percentages else 0
                    )
                else:
                    avg_movement = 0

                sleep_data = results.get("sleep_data", {})
                if sleep_data:
                    sleep_percentages = []
                    for roi_data in sleep_data.values():
                        if roi_data:
                            sleep_pct = np.mean([s for _, s in roi_data]) * 100
                            sleep_percentages.append(sleep_pct)
                    avg_sleep = np.mean(sleep_percentages) if sleep_percentages else 0
                else:
                    avg_sleep = 0

                status = "Success" if roi_count > 0 else "Failed"

                writer.writerow(
                    [
                        method_name,
                        roi_count,
                        f"{avg_movement:.1f}",
                        f"{avg_sleep:.1f}",
                        status,
                    ]
                )

            writer.writerow([])

            # Detailed ROI comparison
            writer.writerow(["# Detailed ROI Comparison"])

            # Get all ROIs
            all_rois = set()
            for results in results_dict.values():
                all_rois.update(results.get("baseline_means", {}).keys())

            if all_rois:
                # Header for detailed comparison
                header = ["ROI"]
                for method_name in results_dict.keys():
                    header.extend(
                        [
                            f"{method_name}_Baseline",
                            f"{method_name}_Upper",
                            f"{method_name}_Lower",
                            f"{method_name}_Movement_%",
                        ]
                    )
                writer.writerow(header)

                # Data for each ROI
                for roi in sorted(all_rois):
                    row = [roi]

                    for method_name, results in results_dict.items():
                        baseline = results.get("baseline_means", {}).get(roi, 0)
                        upper = results.get("upper_thresholds", {}).get(roi, 0)
                        lower = results.get("lower_thresholds", {}).get(roi, 0)

                        # Calculate movement percentage for this ROI
                        movement_data = results.get("movement_data", {}).get(roi, [])
                        if movement_data:
                            import numpy as np

                            movement_pct = np.mean([m for _, m in movement_data]) * 100
                        else:
                            movement_pct = 0

                        row.extend(
                            [
                                f"{baseline:.1f}",
                                f"{upper:.1f}",
                                f"{lower:.1f}",
                                f"{movement_pct:.1f}",
                            ]
                        )

                    writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error exporting method comparison: {e}")
        return False


def quick_method_test(
    merged_results: Dict[int, List[Tuple[float, float]]],
    methods: List[str] = None,
    **common_kwargs,
) -> Dict[str, str]:
    """
    Quick test of multiple analysis methods for comparison.

    Args:
        merged_results: Data from Reader
        methods: List of methods to test (default: all available)
        **common_kwargs: Common parameters for all methods

    Returns:
        Dictionary mapping method names to result summaries
    """
    if methods is None:
        methods = ["baseline", "adaptive"]  # Calibration requires file

    results = {}

    for method in methods:
        try:
            print(f"\n=== Testing {method.upper()} method ===")

            # Method-specific parameters
            if method == "baseline":
                kwargs = {
                    "threshold_block_count": 120,
                    "multiplier": 1.0,
                    "frame_interval": 5.0,
                    **common_kwargs,
                }
            elif method == "adaptive":
                kwargs = {
                    "analysis_duration_frames": 180,
                    "base_multiplier": 2.5,
                    "frame_interval": 5.0,
                    **common_kwargs,
                }
            elif method == "calibration":
                # Skip calibration if no file provided
                if "calibration_file_path" not in common_kwargs:
                    results[method] = "Skipped - no calibration file provided"
                    continue
                kwargs = {
                    "percentile_threshold": 95.0,
                    "calibration_multiplier": 1.0,
                    "frame_interval": 5.0,
                    **common_kwargs,
                }

            # Run analysis
            analysis_results = run_analysis_with_method(
                merged_results, method, **kwargs
            )

            # Generate summary
            summary = get_analysis_summary(analysis_results)
            results[method] = summary

        except Exception as e:
            results[method] = f"Failed: {str(e)}"

    return results


# =============================================================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# =============================================================================


def compute_roi_thresholds_unified(
    merged_results: Dict[int, List[Tuple[float, float]]],
    threshold_method: str,
    **kwargs,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]]]:
    """
    BACKWARDS COMPATIBILITY: Unified threshold calculation with method routing.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        threshold_method: Method to use ('baseline', 'adaptive', 'calibration')
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (roi_thresholds, roi_statistics) - for backwards compatibility
    """
    try:
        # Route to appropriate method
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

    except Exception as e:
        print(f"Error in unified threshold calculation: {e}")
        # Return empty results
        empty_thresholds = {roi: 0.0 for roi in merged_results.keys()}
        empty_statistics = {
            roi: {"method": threshold_method, "status": f"error: {e}"}
            for roi in merged_results.keys()
        }
        return empty_thresholds, empty_statistics


def compute_roi_thresholds_hysteresis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    threshold_method: str,
    **kwargs,
) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, Dict[str, Any]]
]:
    """
    BACKWARDS COMPATIBILITY: Hysteresis threshold calculation with method routing.

    Args:
        merged_results: Dictionary mapping ROI ID to list of (time, value) tuples
        threshold_method: Method to use ('baseline', 'adaptive', 'calibration')
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (roi_baseline_means, roi_upper_thresholds, roi_lower_thresholds, roi_statistics)
    """
    try:
        # Route to appropriate method
        results = run_analysis_with_method(merged_results, threshold_method, **kwargs)

        return (
            results["baseline_means"],
            results["upper_thresholds"],
            results["lower_thresholds"],
            results["roi_statistics"],
        )

    except Exception as e:
        print(f"Error in hysteresis threshold calculation: {e}")
        # Return empty results
        empty_dict = {roi: 0.0 for roi in merged_results.keys()}
        empty_statistics = {
            roi: {"method": threshold_method, "status": f"error: {e}"}
            for roi in merged_results.keys()
        }
        return empty_dict, empty_dict, empty_dict, empty_statistics


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_available_methods() -> List[str]:
    """
    Get list of available analysis methods.

    Returns:
        List of available method names
    """
    return ["baseline", "adaptive", "calibration"]


def get_method_description(method: str) -> str:
    """
    Get description of a specific analysis method.

    Args:
        method: Method name

    Returns:
        Method description string
    """
    descriptions = {
        "baseline": (
            "Baseline Method: Uses the first N frames to establish a baseline "
            "and calculates thresholds based on mean ± multiplier × std. "
            "Good for stable recording conditions."
        ),
        "adaptive": (
            "Adaptive Method: Automatically adjusts thresholds based on "
            "signal-to-noise ratio and coefficient of variation. "
            "Best for varying recording conditions."
        ),
        "calibration": (
            "Calibration Method: Uses recordings of sedated animals to "
            "determine noise baseline. Provides excellent specificity "
            "when calibration data is available."
        ),
    }

    return descriptions.get(method.lower(), "Unknown method")


def suggest_method_for_data(
    merged_results: Dict[int, List[Tuple[float, float]]],
) -> str:
    """
    Suggest the best analysis method based on data characteristics.

    Args:
        merged_results: Data to analyze

    Returns:
        Suggested method name with reasoning
    """
    if not merged_results:
        return "No data available for analysis"

    # Analyze data characteristics
    try:
        import numpy as np

        # Sample from first ROI
        first_roi_data = next(iter(merged_results.values()))
        if len(first_roi_data) < 100:
            return "baseline - Insufficient data for adaptive analysis"

        values = np.array(
            [val for _, val in first_roi_data[:1000]]
        )  # Sample first 1000 points

        # Calculate metrics
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
        value_range = np.ptp(values)
        mean_value = np.mean(values)

        # Decision logic
        if cv > 0.8:
            return "adaptive - High variability detected, adaptive method will handle this best"
        elif cv < 0.2 and value_range / mean_value < 0.5:
            return "baseline - Stable signal, baseline method is sufficient"
        elif mean_value > 100 and cv > 0.3:
            return "adaptive - Medium-high activity with some variability"
        else:
            return "baseline - Standard conditions, baseline method recommended"

    except Exception as e:
        return f"baseline - Error analyzing data: {e}"


def create_method_comparison_report(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Create a comprehensive comparison report of different methods.

    Args:
        results_dict: Dictionary mapping method names to analysis results

    Returns:
        Formatted comparison report
    """
    if not results_dict:
        return "No results to compare"

    report = "=" * 60 + "\n"
    report += "METHOD COMPARISON REPORT\n"
    report += "=" * 60 + "\n\n"

    # Summary table
    report += "SUMMARY TABLE:\n"
    report += "-" * 60 + "\n"
    report += f"{'Method':<12} {'ROIs':<6} {'Avg_Movement':<12} {'Avg_Sleep':<10} {'Status':<10}\n"
    report += "-" * 60 + "\n"

    for method_name, results in results_dict.items():
        roi_count = len(results.get("baseline_means", {}))

        # Calculate averages
        movement_data = results.get("movement_data", {})
        if movement_data and roi_count > 0:
            import numpy as np

            movement_percentages = []
            for roi_data in movement_data.values():
                if roi_data:
                    movement_pct = np.mean([m for _, m in roi_data]) * 100
                    movement_percentages.append(movement_pct)
            avg_movement = np.mean(movement_percentages) if movement_percentages else 0
        else:
            avg_movement = 0

        sleep_data = results.get("sleep_data", {})
        if sleep_data and roi_count > 0:
            sleep_percentages = []
            for roi_data in sleep_data.values():
                if roi_data:
                    sleep_pct = np.mean([s for _, s in roi_data]) * 100
                    sleep_percentages.append(sleep_pct)
            avg_sleep = np.mean(sleep_percentages) if sleep_percentages else 0
        else:
            avg_sleep = 0

        status = "Success" if roi_count > 0 else "Failed"

        report += f"{method_name:<12} {roi_count:<6} {avg_movement:<12.1f} {avg_sleep:<10.1f} {status:<10}\n"

    report += "\n"

    # Detailed analysis
    for method_name, results in results_dict.items():
        report += f"DETAILED ANALYSIS - {method_name.upper()}:\n"
        report += "-" * 40 + "\n"

        if "summary_stats" in results:
            summary = results["summary_stats"]
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if key != "recommendations":
                        report += f"  {key}: {value}\n"

                # Add recommendations
                if "recommendations" in summary:
                    report += "  Recommendations:\n"
                    for rec in summary["recommendations"]:
                        report += f"    • {rec}\n"
            else:
                report += f"  Summary: {summary}\n"
        else:
            report += f"  {get_analysis_summary(results)}\n"

        report += "\n"

    return report


# Add these to the end of _calc_integration.py


def validate_hdf5_timing_in_data(
    merged_results: Dict[int, List[Tuple[float, float]]], frame_interval: float = 5.0
) -> Dict[str, Any]:
    """
    Validate HDF5 timing consistency.

    Args:
        merged_results: Data to analyze
        frame_interval: Expected frame interval

    Returns:
        Dictionary with timing diagnostics
    """
    if not merged_results:
        return {"timing_type": "no_data", "needs_correction": False}

    try:
        # Get sample data from first ROI
        first_roi_data = next(iter(merged_results.values()))
        if len(first_roi_data) < 3:
            return {"timing_type": "insufficient_data", "needs_correction": False}

        # Calculate actual intervals
        times = [t for t, _ in first_roi_data[:10]]  # First 10 points
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        import numpy as np

        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)

        # Check consistency
        tolerance = max(1.0, frame_interval * 0.1)  # 10% tolerance
        needs_correction = abs(avg_interval - frame_interval) > tolerance
        interval_consistent = interval_std < (frame_interval * 0.05)  # 5% variation

        return {
            "timing_type": "hdf5_analysis",
            "first_time": times[0],
            "avg_interval": avg_interval,
            "expected_interval": frame_interval,
            "interval_consistent": interval_consistent,
            "needs_hdf5_correction": needs_correction,
            "recommended_action": (
                "Apply timing correction"
                if needs_correction
                else "No correction needed"
            ),
        }

    except Exception as e:
        return {"timing_type": "error", "needs_correction": False, "error": str(e)}


def get_performance_metrics(start_time: float, total_frames: int) -> Dict[str, Any]:
    """
    Calculate performance metrics for analysis.

    Args:
        start_time: Analysis start time
        total_frames: Total number of frames processed

    Returns:
        Dictionary with performance metrics
    """
    import time

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


def export_results_for_matlab(
    analysis_results: Dict[str, Any], output_dir: str
) -> List[str]:
    """
    Export analysis results in MATLAB-compatible format.

    Args:
        analysis_results: Complete analysis results dictionary
        output_dir: Output directory for files

    Returns:
        List of created file paths
    """
    import csv
    import os
    from datetime import datetime

    created_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Export basic results
        basic_file = os.path.join(output_dir, f"analysis_results_{timestamp}.csv")

        with open(basic_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(["# MATLAB-Compatible Analysis Results"])
            writer.writerow(
                [f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
            )
            writer.writerow([f"# Method: {analysis_results.get('method', 'unknown')}"])
            writer.writerow([])

            # Parameters
            writer.writerow(["# Analysis Parameters"])
            params = analysis_results.get("parameters", {})
            for key, value in params.items():
                writer.writerow([key, str(value)])
            writer.writerow([])

            # ROI summary
            writer.writerow(["# ROI Summary"])
            writer.writerow(
                ["ROI_ID", "Baseline_Mean", "Upper_Threshold", "Lower_Threshold"]
            )

            baseline_means = analysis_results.get("baseline_means", {})
            upper_thresholds = analysis_results.get("upper_thresholds", {})
            lower_thresholds = analysis_results.get("lower_thresholds", {})

            for roi in sorted(baseline_means.keys()):
                writer.writerow(
                    [
                        roi,
                        baseline_means.get(roi, 0),
                        upper_thresholds.get(roi, 0),
                        lower_thresholds.get(roi, 0),
                    ]
                )

        created_files.append(basic_file)

        # Export time series data if available
        movement_data = analysis_results.get("movement_data", {})
        if movement_data:
            movement_file = os.path.join(output_dir, f"movement_data_{timestamp}.csv")

            with open(movement_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time_Seconds", "ROI_ID", "Movement_Binary"])

                for roi, data in movement_data.items():
                    for time_sec, movement in data:
                        writer.writerow([time_sec, roi, int(movement)])

            created_files.append(movement_file)

        return created_files

    except Exception as e:
        print(f"Error exporting MATLAB results: {e}")
        return []
