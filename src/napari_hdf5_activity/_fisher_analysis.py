"""
_fisher_analysis.py - Fischer Z-transformation for periodic pattern detection

This module implements Fischer's Z-transformation to detect periodic patterns
in activity data. The method identifies the dominant period(s) and their
statistical significance, but does not directly classify sleep/wake timing.
For actual sleep/wake timing, use the Main Analysis Quiescence and Sleep plots.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats


def fisher_z_periodogram(
    time_series: np.ndarray,
    sampling_interval: float = 5.0,
    min_period_hours: float = 12.0,
    max_period_hours: float = 36.0,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Apply Fischer's Z-transformation to detect periodic patterns in time series.

    Args:
        time_series: 1D array of activity values (e.g., movement fraction)
        sampling_interval: Time interval between samples in seconds
        min_period_hours: Minimum period to test (hours)
        max_period_hours: Maximum period to test (hours)
        significance_level: Statistical significance threshold

    Returns:
        Dictionary containing:
        - periods: Array of tested periods (hours)
        - z_scores: Fischer Z-scores for each period
        - significant_periods: Periods with significant rhythms
        - dominant_period: Most prominent period (hours)
        - p_value: Statistical significance of dominant period
    """
    if len(time_series) < 10:
        return {
            "periods": np.array([]),
            "z_scores": np.array([]),
            "significant_periods": [],
            "dominant_period": None,
            "p_value": 1.0,
            "error": "Time series too short for analysis",
        }

    # Convert sampling interval to hours
    sampling_hours = sampling_interval / 3600.0

    # Calculate total duration
    total_duration_hours = len(time_series) * sampling_hours

    # Generate test periods (in hours)
    # Ensure we have enough data points per period
    min_period = max(min_period_hours, 3 * sampling_hours)
    max_period = min(max_period_hours, total_duration_hours / 2)

    # Create period range (test periods from min to max)
    n_periods = 100
    periods = np.linspace(min_period, max_period, n_periods)

    z_scores = np.zeros(n_periods)

    # Calculate Z-score for each period
    for idx, period_hours in enumerate(periods):
        # Convert period to number of samples
        period_samples = period_hours / sampling_hours

        # Calculate frequency (cycles per sample)
        freq = 1.0 / period_samples

        # Calculate angular frequency
        omega = 2 * np.pi * freq

        # Create time indices
        t = np.arange(len(time_series))

        # Calculate cosine and sine components
        cos_component = np.cos(omega * t)
        sin_component = np.sin(omega * t)

        # Calculate correlation with time series
        r_cos = np.corrcoef(time_series, cos_component)[0, 1]
        r_sin = np.corrcoef(time_series, sin_component)[0, 1]

        # Handle NaN correlations
        if np.isnan(r_cos):
            r_cos = 0
        if np.isnan(r_sin):
            r_sin = 0

        # Calculate squared coherence (power)
        coherence_sq = r_cos**2 + r_sin**2

        # Chi² periodogram test statistic: n × (r_cos² + r_sin²)
        # Follows chi-square distribution with df=2 under null hypothesis of no periodicity
        n = len(time_series)
        z_scores[idx] = n * coherence_sq

    # Find dominant period
    max_z_idx = np.argmax(z_scores)
    dominant_period = periods[max_z_idx]
    max_z_score = z_scores[max_z_idx]

    # Calculate p-value using chi-square distribution (df=2)
    # Raw single-frequency p-value (for display)
    p_value = 1 - stats.chi2.cdf(max_z_score, df=2)

    # Bonferroni-corrected threshold: testing m periods simultaneously
    # chi²(1 - alpha/m, df=2) — guards against false positives across the periodogram
    m = len(periods)
    corrected_alpha = significance_level / m
    critical_z = stats.chi2.ppf(1 - corrected_alpha, df=2)
    significant_mask = z_scores > critical_z
    significant_periods = periods[significant_mask].tolist()

    return {
        "periods": periods,
        "test_periods": periods.tolist(),   # alias expected by generate_circadian_summary
        "z_scores": z_scores,
        "significant_periods": significant_periods,
        "dominant_period": dominant_period,
        "dominant_z_score": max_z_score,
        "p_value": p_value,
        "is_significant": max_z_score > critical_z,
        "critical_z": critical_z,
        "n_periods_tested": m,
        "sampling_hours": sampling_hours,
        "total_duration_hours": total_duration_hours,
        "actual_min_period": float(min_period),
        "actual_max_period": float(max_period),
        "requested_max_period": float(max_period_hours),
        "period_capped": float(max_period) < float(max_period_hours),
    }


def analyze_roi_circadian_patterns(
    movement_data: Dict[int, List[Tuple[float, float]]],
    sampling_interval: float = 5.0,
    min_period_hours: float = 12.0,
    max_period_hours: float = 36.0,
    significance_level: float = 0.05,
    phase_threshold: float = 0.5,
    bin_size_seconds: int = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze circadian patterns for all ROIs using Fischer Z-transformation.

    Args:
        movement_data: Dictionary mapping ROI ID to list of (time, value) tuples
                      Can be fraction_data (binned, 0-1) or processed_data (raw signal)
        sampling_interval: Time interval between samples (seconds)
        min_period_hours: Minimum period to test (hours)
        max_period_hours: Maximum period to test (hours)
        significance_level: Statistical significance threshold
        phase_threshold: Threshold for sleep/wake classification (unused, kept for compatibility)
        bin_size_seconds: Optional bin size for averaging data (e.g., 60 for 1-minute bins)
                         If None, data is used as-is. Useful for high-resolution raw data.

    Returns:
        Dictionary mapping ROI ID to analysis results
    """
    results = {}

    for roi_id, data in movement_data.items():
        if not data or len(data) < 10:
            results[roi_id] = {
                "error": "Insufficient data for analysis",
                "n_samples": len(data) if data else 0,
            }
            continue

        # Extract values and times
        times = np.array([t for t, _ in data])
        values = np.array([v for _, v in data])

        # Apply binning if requested (for high-resolution raw data)
        if bin_size_seconds is not None and bin_size_seconds > 0:
            values, effective_interval = _bin_data(times, values, bin_size_seconds)
        else:
            effective_interval = sampling_interval

        # Run Chi² periodogram
        periodogram = fisher_z_periodogram(
            values,
            sampling_interval=effective_interval,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
            significance_level=significance_level,
        )

        # Combine results
        results[roi_id] = {
            "periodogram": periodogram,
            "n_samples": len(values),
            "mean_activity": np.mean(values),
            "std_activity": np.std(values),
            "data_type": "binned" if bin_size_seconds else "raw",
            "effective_sampling_interval": effective_interval,
        }

    return results


def _bin_data(
    times: np.ndarray,
    values: np.ndarray,
    bin_size_seconds: int,
) -> Tuple[np.ndarray, float]:
    """
    Bin data into time windows by averaging.

    Args:
        times: Array of time points (seconds)
        values: Array of data values
        bin_size_seconds: Size of time bins (seconds)

    Returns:
        Tuple of (binned_values, effective_sampling_interval)
    """
    if len(times) == 0:
        return np.array([]), bin_size_seconds

    # Create bins
    start_time = times[0]
    end_time = times[-1]
    bin_edges = np.arange(start_time, end_time + bin_size_seconds, bin_size_seconds)

    # Assign data points to bins and average
    binned_values = []
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Find all values in this bin
        mask = (times >= bin_start) & (times < bin_end)
        bin_data = values[mask]

        if len(bin_data) > 0:
            binned_values.append(np.mean(bin_data))
        else:
            # No data in this bin - use interpolation or skip
            if len(binned_values) > 0:
                binned_values.append(binned_values[-1])  # Forward fill
            else:
                binned_values.append(0.0)

    return np.array(binned_values), float(bin_size_seconds)


def generate_circadian_summary(results: Dict[int, Dict[str, Any]]) -> str:
    """
    Generate a human-readable summary of circadian analysis results.

    Args:
        results: Dictionary of analysis results from analyze_roi_circadian_patterns

    Returns:
        Formatted summary string
    """
    summary_lines = ["=" * 60, "RHYTHMIC PATTERN ANALYSIS SUMMARY", "=" * 60, ""]

    n_rois = len(results)
    n_significant = sum(
        1
        for r in results.values()
        if r.get("periodogram", {}).get("is_significant", False)
    )

    summary_lines.append(f"Total ROIs analyzed: {n_rois}")
    summary_lines.append(
        f"ROIs with significant rhythms: {n_significant} ({n_significant/n_rois*100:.1f}%)"
    )
    summary_lines.append("")

    # Diagnostic checks for period range issues
    warnings = []
    boundary_count = 0
    detected_periods = []

    for roi_id, result in sorted(results.items()):
        if "error" in result:
            continue

        periodogram = result.get("periodogram", {})
        if not periodogram.get("is_significant", False):
            continue

        dominant_period = periodogram.get("dominant_period", 0)
        if dominant_period > 0:
            detected_periods.append(dominant_period)

        # Check if period is at boundary
        test_periods = periodogram.get("test_periods", [])
        if len(test_periods) > 0:
            min_period = min(test_periods)
            max_period = max(test_periods)
            period_range = max_period - min_period

            # Check if dominant period is at boundary (within 5%)
            boundary_threshold = period_range * 0.05
            if (
                abs(dominant_period - max_period) < boundary_threshold
                or abs(dominant_period - min_period) < boundary_threshold
            ):
                boundary_count += 1

    # Add warnings if issues detected
    if boundary_count > n_significant * 0.3:  # More than 30% at boundaries
        warnings.append(
            f"⚠️  WARNING: {boundary_count}/{n_significant} ROIs have dominant periods at range boundaries.\n"
            f"   This suggests the period range may be too narrow.\n"
            f"   Consider expanding the period range to capture true rhythms."
        )

    # Check if detected periods cluster at extremes
    if len(detected_periods) >= 3:
        detected_periods_array = np.array(detected_periods)
        min_detected = detected_periods_array.min()
        max_detected = detected_periods_array.max()

        # Get the analysis period range from first valid result
        for result in results.values():
            if "error" not in result:
                periodogram = result.get("periodogram", {})
                test_periods = periodogram.get("test_periods", [])
                if len(test_periods) > 0:
                    analysis_min = min(test_periods)
                    analysis_max = max(test_periods)

                    # Check if many periods are beyond typical ranges
                    if max_detected > 12.0 and analysis_max < 24.0:
                        warnings.append(
                            f"ℹ️  INFO: Some detected periods exceed 12h (max: {max_detected:.1f}h).\n"
                            f"   Consider extending max period to 24-36h for circadian analysis."
                        )
                    elif min_detected < 2.0 and analysis_min > 1.0:
                        warnings.append(
                            f"ℹ️  INFO: Some detected periods below 2h (min: {min_detected:.1f}h).\n"
                            f"   Consider reducing min period to 0.5-1h for ultradian analysis."
                        )
                    break

    if warnings:
        summary_lines.extend(warnings)
        summary_lines.append("")
        summary_lines.append("=" * 60)
        summary_lines.append("")

    for roi_id, result in sorted(results.items()):
        summary_lines.append(f"ROI {roi_id}:")

        if "error" in result:
            summary_lines.append(f"  ⚠️  {result['error']}")
            summary_lines.append("")
            continue

        periodogram = result.get("periodogram", {})

        if periodogram.get("is_significant", False):
            dominant_period = periodogram.get("dominant_period", 0)

            # Check for boundary warning for this specific ROI
            boundary_marker = ""
            test_periods = periodogram.get("test_periods", [])
            if len(test_periods) > 0:
                min_p = min(test_periods)
                max_p = max(test_periods)
                boundary_threshold = (max_p - min_p) * 0.05

                if abs(dominant_period - max_p) < boundary_threshold:
                    boundary_marker = f" ⚠️ (at upper boundary {max_p:.2f}h)"
                elif abs(dominant_period - min_p) < boundary_threshold:
                    boundary_marker = f" ⚠️ (at lower boundary {min_p:.2f}h)"

            summary_lines.append(
                f"  ✓ Significant rhythm detected (p={periodogram['p_value']:.4f})"
            )
            summary_lines.append(
                f"    Dominant period: {dominant_period:.2f} hours{boundary_marker}"
            )
            summary_lines.append(f"    Z-score: {periodogram['dominant_z_score']:.2f}")
        else:
            summary_lines.append(
                f"  ✗ No significant rhythm (p={periodogram.get('p_value', 1.0):.4f})"
            )

        summary_lines.append("")

    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)
