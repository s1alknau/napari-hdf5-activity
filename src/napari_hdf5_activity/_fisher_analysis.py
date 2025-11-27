"""
_fisher_analysis.py - Fischer Z-transformation for circadian pattern detection

This module implements Fischer's Z-transformation to detect recurring sleep/wake
patterns in activity data. The method is particularly useful for identifying
circadian rhythms and periodic behavioral patterns.
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

        # Fischer's Z-transformation
        n = len(time_series)
        z_scores[idx] = n * coherence_sq

    # Find dominant period
    max_z_idx = np.argmax(z_scores)
    dominant_period = periods[max_z_idx]
    max_z_score = z_scores[max_z_idx]

    # Calculate p-value using chi-square distribution (df=2)
    p_value = 1 - stats.chi2.cdf(max_z_score, df=2)

    # Find all significant periods
    critical_z = stats.chi2.ppf(1 - significance_level, df=2)
    significant_mask = z_scores > critical_z
    significant_periods = periods[significant_mask].tolist()

    return {
        "periods": periods,
        "z_scores": z_scores,
        "significant_periods": significant_periods,
        "dominant_period": dominant_period,
        "dominant_z_score": max_z_score,
        "p_value": p_value,
        "is_significant": p_value < significance_level,
        "critical_z": critical_z,
        "sampling_hours": sampling_hours,
        "total_duration_hours": total_duration_hours,
    }


def detect_sleep_wake_phases(
    time_series: np.ndarray,
    dominant_period: float,
    sampling_interval: float = 5.0,
    phase_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect sleep and wake phases based on the dominant circadian period.

    Args:
        time_series: 1D array of activity values
        dominant_period: Dominant period detected (hours)
        sampling_interval: Time interval between samples (seconds)
        phase_threshold: Threshold for classifying sleep vs wake

    Returns:
        Dictionary containing detected phases and their timing
    """
    sampling_hours = sampling_interval / 3600.0
    period_samples = dominant_period / sampling_hours

    # Calculate frequency
    freq = 1.0 / period_samples
    omega = 2 * np.pi * freq

    # Create time indices
    t = np.arange(len(time_series))

    # Calculate cosine and sine components
    cos_component = np.cos(omega * t)
    sin_component = np.sin(omega * t)

    # Fit the rhythm using linear regression
    # Activity = A*cos(ωt) + B*sin(ωt) + C
    X = np.column_stack([cos_component, sin_component, np.ones(len(time_series))])
    coeffs, _, _, _ = np.linalg.lstsq(X, time_series, rcond=None)

    A, B, C = coeffs

    # Calculate fitted rhythm
    fitted_rhythm = A * cos_component + B * sin_component + C

    # Calculate amplitude and phase
    amplitude = np.sqrt(A**2 + B**2)
    phase_offset = np.arctan2(B, A)

    # Normalize fitted rhythm to [0, 1] for phase detection
    rhythm_normalized = (fitted_rhythm - fitted_rhythm.min()) / (
        fitted_rhythm.max() - fitted_rhythm.min() + 1e-10
    )

    # Detect phases (wake = high activity, sleep = low activity)
    wake_mask = rhythm_normalized > phase_threshold
    sleep_mask = ~wake_mask

    # Find phase transitions
    transitions = np.diff(wake_mask.astype(int))
    wake_onsets = np.where(transitions == 1)[0] + 1
    sleep_onsets = np.where(transitions == -1)[0] + 1

    # Convert to time in hours
    time_hours = t * sampling_hours

    # Create phase list with timing
    wake_phases = []
    for onset_idx in wake_onsets:
        # Find corresponding offset
        next_sleep = sleep_onsets[sleep_onsets > onset_idx]
        if len(next_sleep) > 0:
            offset_idx = next_sleep[0]
            wake_phases.append(
                {
                    "start_hours": time_hours[onset_idx],
                    "end_hours": time_hours[offset_idx],
                    "duration_hours": time_hours[offset_idx] - time_hours[onset_idx],
                    "start_idx": onset_idx,
                    "end_idx": offset_idx,
                }
            )

    sleep_phases = []
    for onset_idx in sleep_onsets:
        # Find corresponding offset
        next_wake = wake_onsets[wake_onsets > onset_idx]
        if len(next_wake) > 0:
            offset_idx = next_wake[0]
            sleep_phases.append(
                {
                    "start_hours": time_hours[onset_idx],
                    "end_hours": time_hours[offset_idx],
                    "duration_hours": time_hours[offset_idx] - time_hours[onset_idx],
                    "start_idx": onset_idx,
                    "end_idx": offset_idx,
                }
            )

    return {
        "fitted_rhythm": fitted_rhythm,
        "rhythm_normalized": rhythm_normalized,
        "amplitude": amplitude,
        "phase_offset": phase_offset,
        "wake_phases": wake_phases,
        "sleep_phases": sleep_phases,
        "n_wake_phases": len(wake_phases),
        "n_sleep_phases": len(sleep_phases),
        "wake_fraction": np.sum(wake_mask) / len(wake_mask),
        "sleep_fraction": np.sum(sleep_mask) / len(sleep_mask),
    }


def analyze_roi_circadian_patterns(
    movement_data: Dict[int, List[Tuple[float, float]]],
    sampling_interval: float = 5.0,
    min_period_hours: float = 12.0,
    max_period_hours: float = 36.0,
    significance_level: float = 0.05,
    phase_threshold: float = 0.5,
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze circadian patterns for all ROIs using Fischer Z-transformation.

    Args:
        movement_data: Dictionary mapping ROI ID to list of (time, movement) tuples
        sampling_interval: Time interval between samples (seconds)
        min_period_hours: Minimum period to test (hours)
        max_period_hours: Maximum period to test (hours)
        significance_level: Statistical significance threshold
        phase_threshold: Threshold for sleep/wake classification

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

        # Extract movement values (ignore time for now, assume regular sampling)
        movement_values = np.array([m for _, m in data])

        # Run Fisher periodogram
        periodogram = fisher_z_periodogram(
            movement_values,
            sampling_interval=sampling_interval,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
            significance_level=significance_level,
        )

        # If significant rhythm detected, analyze sleep/wake phases
        if periodogram.get("is_significant", False):
            phase_analysis = detect_sleep_wake_phases(
                movement_values,
                periodogram["dominant_period"],
                sampling_interval=sampling_interval,
                phase_threshold=phase_threshold,
            )
        else:
            phase_analysis = {
                "error": "No significant circadian rhythm detected",
                "wake_phases": [],
                "sleep_phases": [],
            }

        # Combine results
        results[roi_id] = {
            "periodogram": periodogram,
            "phase_analysis": phase_analysis,
            "n_samples": len(movement_values),
            "mean_activity": np.mean(movement_values),
            "std_activity": np.std(movement_values),
        }

    return results


def generate_circadian_summary(results: Dict[int, Dict[str, Any]]) -> str:
    """
    Generate a human-readable summary of circadian analysis results.

    Args:
        results: Dictionary of analysis results from analyze_roi_circadian_patterns

    Returns:
        Formatted summary string
    """
    summary_lines = ["=" * 60, "CIRCADIAN PATTERN ANALYSIS SUMMARY", "=" * 60, ""]

    n_rois = len(results)
    n_significant = sum(
        1
        for r in results.values()
        if r.get("periodogram", {}).get("is_significant", False)
    )

    summary_lines.append(f"Total ROIs analyzed: {n_rois}")
    summary_lines.append(
        f"ROIs with significant circadian rhythms: {n_significant} ({n_significant/n_rois*100:.1f}%)"
    )
    summary_lines.append("")

    for roi_id, result in sorted(results.items()):
        summary_lines.append(f"ROI {roi_id}:")

        if "error" in result:
            summary_lines.append(f"  ⚠️  {result['error']}")
            summary_lines.append("")
            continue

        periodogram = result.get("periodogram", {})
        phase_analysis = result.get("phase_analysis", {})

        if periodogram.get("is_significant", False):
            summary_lines.append(
                f"  ✓ Significant circadian rhythm detected (p={periodogram['p_value']:.4f})"
            )
            summary_lines.append(
                f"    Dominant period: {periodogram['dominant_period']:.2f} hours"
            )
            summary_lines.append(f"    Z-score: {periodogram['dominant_z_score']:.2f}")

            if "wake_phases" in phase_analysis:
                n_wake = len(phase_analysis["wake_phases"])
                n_sleep = len(phase_analysis["sleep_phases"])
                wake_frac = phase_analysis.get("wake_fraction", 0) * 100
                sleep_frac = phase_analysis.get("sleep_fraction", 0) * 100

                summary_lines.append(f"    Wake phases detected: {n_wake}")
                summary_lines.append(f"    Sleep phases detected: {n_sleep}")
                summary_lines.append(
                    f"    Wake fraction: {wake_frac:.1f}% | Sleep fraction: {sleep_frac:.1f}%"
                )
        else:
            summary_lines.append(
                f"  ✗ No significant rhythm (p={periodogram.get('p_value', 1.0):.4f})"
            )

        summary_lines.append("")

    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)
