"""
_circadian_fft.py - FFT-based circadian rhythm analysis

This module implements Fast Fourier Transform (FFT) based methods for detecting
and analyzing circadian rhythms in activity data. FFT is complementary to the
Fisher Z-transformation and provides standard spectral analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import signal


def _permutation_test_fft(
    time_series: np.ndarray,
    observed_power: float,
    sampling_interval: float,
    target_period: float,
    window: str = "hann",
    n_permutations: int = 1000,
) -> float:
    """
    Perform permutation test to assess significance of observed FFT power.

    This test randomizes the time series data while preserving its amplitude
    distribution, then computes the FFT power at the target period. The p-value
    is the proportion of permutations that produce power >= observed power.

    Args:
        time_series: Original time series data
        observed_power: Observed power at the dominant period
        sampling_interval: Time between samples (seconds)
        target_period: Period to test (hours)
        window: Window function to apply
        n_permutations: Number of random permutations (default: 1000)

    Returns:
        p_value: Proportion of permutations with power >= observed power
    """
    # Convert target period to frequency
    target_freq = 1.0 / (target_period * 3600.0)  # Hz

    # Store powers from permutations
    permuted_powers = []

    for _ in range(n_permutations):
        # Randomly shuffle the time series (breaks temporal structure)
        permuted = np.random.permutation(time_series)

        # Detrend
        detrended = signal.detrend(permuted)

        # Apply window
        if window:
            if window == "hann":
                win = np.hanning(len(detrended))
            elif window == "hamming":
                win = np.hamming(len(detrended))
            elif window == "blackman":
                win = np.blackman(len(detrended))
            else:
                win = np.ones(len(detrended))
            detrended = detrended * win

        # Compute FFT
        n_fft = max(2 ** int(np.ceil(np.log2(len(detrended) * 4))), len(detrended) * 4)
        fft_values = np.fft.rfft(detrended, n=n_fft)
        power_spectrum = np.abs(fft_values) ** 2

        # Get frequencies
        frequencies = np.fft.rfftfreq(n_fft, d=sampling_interval)

        # Find power at target frequency (closest bin)
        freq_idx = np.argmin(np.abs(frequencies - target_freq))
        permuted_powers.append(power_spectrum[freq_idx])

    # Calculate p-value: proportion of permuted powers >= observed power
    p_value = np.sum(np.array(permuted_powers) >= observed_power) / n_permutations

    return p_value


def fft_periodogram(
    time_series: np.ndarray,
    sampling_interval: float = 5.0,
    min_period_hours: float = 12.0,
    max_period_hours: float = 36.0,
    window: str = "hann",
    significance_level: float = 0.05,
    n_permutations: int = 1000,
) -> Dict[str, Any]:
    """
    Compute FFT-based power spectrum to detect periodic patterns with significance testing.

    Args:
        time_series: 1D array of activity values
        sampling_interval: Time interval between samples (seconds)
        min_period_hours: Minimum period to analyze (hours)
        max_period_hours: Maximum period to analyze (hours)
        window: Window function for FFT ('hann', 'hamming', 'blackman', None)
        significance_level: Statistical significance threshold (default: 0.05)
        n_permutations: Number of permutations for significance test (default: 1000)

    Returns:
        Dictionary containing:
        - frequencies: Array of frequencies (Hz)
        - periods: Array of periods (hours)
        - power_spectrum: Power spectral density
        - dominant_period: Most prominent period (hours)
        - dominant_frequency: Corresponding frequency (Hz)
        - dominant_power: Power at dominant frequency
        - p_value: Statistical significance of dominant period (permutation test)
        - is_significant: Boolean indicating if rhythm is significant
        - frequency_peaks: List of significant peaks (period, power)
    """
    if len(time_series) < 10:
        return {
            "error": "Time series too short for FFT analysis",
            "frequencies": np.array([]),
            "periods": np.array([]),
            "power_spectrum": np.array([]),
        }

    # Detrend data (remove linear trend)
    detrended = signal.detrend(time_series)

    # Apply window function to reduce spectral leakage
    if window:
        if window == "hann":
            win = np.hanning(len(detrended))
        elif window == "hamming":
            win = np.hamming(len(detrended))
        elif window == "blackman":
            win = np.blackman(len(detrended))
        else:
            win = np.ones(len(detrended))
        detrended = detrended * win

    # Compute FFT with zero-padding for better frequency resolution
    # Pad to next power of 2 for efficiency, minimum 4x original length
    n_fft = max(2 ** int(np.ceil(np.log2(len(detrended) * 4))), len(detrended) * 4)
    fft_values = np.fft.rfft(detrended, n=n_fft)
    power_spectrum = np.abs(fft_values) ** 2

    # Frequency bins (use n_fft for zero-padded FFT)
    frequencies = np.fft.rfftfreq(n_fft, d=sampling_interval)

    # Convert to periods (in hours)
    # Avoid division by zero
    periods = np.zeros_like(frequencies)
    nonzero_mask = frequencies > 0
    periods[nonzero_mask] = (1.0 / frequencies[nonzero_mask]) / 3600.0  # to hours

    # Filter for relevant period range
    period_mask = (periods >= min_period_hours) & (periods <= max_period_hours)
    relevant_periods = periods[period_mask]
    relevant_power = power_spectrum[period_mask]
    relevant_freqs = frequencies[period_mask]

    if len(relevant_power) == 0:
        return {
            "error": "No data in specified period range",
            "frequencies": frequencies,
            "periods": periods,
            "power_spectrum": power_spectrum,
        }

    # Find dominant period: Simply use the maximum power in the relevant range
    # This is the most straightforward approach and should match Fisher Z-transformation
    # which also finds the period with maximum correlation (analogous to maximum power)
    max_power_idx = np.argmax(relevant_power)
    dominant_period = relevant_periods[max_power_idx]
    dominant_frequency = relevant_freqs[max_power_idx]
    dominant_power = relevant_power[max_power_idx]

    # Perform permutation test for statistical significance
    # This tests whether the observed power is significantly higher than what would
    # be expected by chance (i.e., from a non-periodic time series)
    p_value = _permutation_test_fft(
        time_series=time_series,
        observed_power=dominant_power,
        sampling_interval=sampling_interval,
        target_period=dominant_period,
        window=window,
        n_permutations=n_permutations,
    )

    is_significant = p_value < significance_level

    # Find peaks in power spectrum for reporting purposes
    # Use scipy's find_peaks with low prominence threshold to catch all significant peaks
    peak_indices, peak_properties = signal.find_peaks(
        relevant_power, prominence=np.max(relevant_power) * 0.1
    )

    frequency_peaks = [
        {
            "period_hours": relevant_periods[i],
            "frequency_hz": relevant_freqs[i],
            "power": relevant_power[i],
            "prominence": peak_properties["prominences"][idx],
        }
        for idx, i in enumerate(peak_indices)
    ]

    # Sort by power (descending)
    frequency_peaks = sorted(frequency_peaks, key=lambda x: x["power"], reverse=True)

    return {
        "frequencies": frequencies,
        "periods": periods,
        "power_spectrum": power_spectrum,
        "relevant_periods": relevant_periods,
        "relevant_power": relevant_power,
        "dominant_period": dominant_period,
        "dominant_frequency": dominant_frequency,
        "dominant_power": dominant_power,
        "p_value": p_value,
        "is_significant": is_significant,
        "significance_level": significance_level,
        "frequency_peaks": frequency_peaks,
        "n_samples": len(time_series),
        "sampling_interval": sampling_interval,
        "window": window,
    }


def analyze_roi_fft_patterns(
    movement_data: Dict[int, List[Tuple[float, float]]],
    sampling_interval: float = 5.0,
    min_period_hours: float = 12.0,
    max_period_hours: float = 36.0,
    bin_size_seconds: int = None,
    window: str = "hann",
    significance_level: float = 0.05,
    n_permutations: int = 1000,
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze circadian patterns for all ROIs using FFT with significance testing.

    Args:
        movement_data: Dictionary mapping ROI ID to list of (time, value) tuples
        sampling_interval: Time interval between samples (seconds)
        min_period_hours: Minimum period to analyze (hours)
        max_period_hours: Maximum period to analyze (hours)
        bin_size_seconds: Optional bin size for averaging data
        window: Window function for FFT
        significance_level: Statistical significance threshold (default: 0.05)
        n_permutations: Number of permutations for significance test (default: 1000)

    Returns:
        Dictionary mapping ROI ID to FFT analysis results (includes p-values)
    """
    from ._fisher_analysis import _bin_data

    results = {}

    for roi_id, data in movement_data.items():
        if not data or len(data) < 10:
            results[roi_id] = {
                "error": "Insufficient data for FFT analysis",
                "n_samples": len(data) if data else 0,
            }
            continue

        # Extract values and times
        times = np.array([t for t, _ in data])
        values = np.array([v for _, v in data])

        # Apply binning if requested
        if bin_size_seconds is not None and bin_size_seconds > 0:
            values, effective_interval = _bin_data(times, values, bin_size_seconds)
        else:
            effective_interval = sampling_interval

        # Run FFT periodogram with significance testing
        fft_result = fft_periodogram(
            values,
            sampling_interval=effective_interval,
            min_period_hours=min_period_hours,
            max_period_hours=max_period_hours,
            window=window,
            significance_level=significance_level,
            n_permutations=n_permutations,
        )

        # Add metadata
        fft_result["roi_id"] = roi_id
        fft_result["mean_activity"] = np.mean(values)
        fft_result["std_activity"] = np.std(values)
        fft_result["data_type"] = "binned" if bin_size_seconds else "raw"
        fft_result["effective_sampling_interval"] = effective_interval

        results[roi_id] = fft_result

    return results


def compare_fisher_fft(
    fisher_results: Dict[int, Dict[str, Any]], fft_results: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare Fisher Z-transformation and FFT results.

    Args:
        fisher_results: Results from Fisher analysis
        fft_results: Results from FFT analysis

    Returns:
        Comparison summary with agreement metrics
    """
    comparison = {
        "roi_comparisons": {},
        "overall_agreement": 0.0,
        "mean_period_difference": 0.0,
    }

    agreements = []
    period_diffs = []

    for roi_id in fisher_results.keys():
        if roi_id not in fft_results:
            continue

        fisher_res = fisher_results[roi_id]
        fft_res = fft_results[roi_id]

        # Skip if errors
        if "error" in fisher_res or "error" in fft_res:
            continue

        fisher_period = fisher_res.get("periodogram", {}).get("dominant_period", 0)
        fft_period = fft_res.get("dominant_period", 0)

        if fisher_period > 0 and fft_period > 0:
            period_diff = abs(fisher_period - fft_period)
            period_diffs.append(period_diff)

            # Consider agreement if periods within 2 hours
            agrees = period_diff < 2.0
            agreements.append(agrees)

            comparison["roi_comparisons"][roi_id] = {
                "fisher_period": fisher_period,
                "fft_period": fft_period,
                "difference_hours": period_diff,
                "agrees": agrees,
            }

    if agreements:
        comparison["overall_agreement"] = np.mean(agreements) * 100  # Percentage
        comparison["mean_period_difference"] = np.mean(period_diffs)
        comparison["std_period_difference"] = np.std(period_diffs)

    return comparison


def generate_fft_summary(results: Dict[int, Dict[str, Any]]) -> str:
    """
    Generate human-readable summary of FFT analysis results.

    Args:
        results: Dictionary of FFT analysis results

    Returns:
        Formatted summary string
    """
    summary_lines = [
        "=" * 60,
        "FFT POWER SPECTRUM ANALYSIS",
        "=" * 60,
        "",
    ]

    n_rois = len(results)
    n_valid = sum(1 for r in results.values() if "error" not in r)
    n_significant = sum(1 for r in results.values() if r.get("is_significant", False))

    summary_lines.append(f"Total ROIs analyzed: {n_rois}")
    summary_lines.append(f"ROIs with valid FFT results: {n_valid}")
    summary_lines.append(
        f"ROIs with significant rhythms: {n_significant} ({n_significant/n_rois*100:.1f}%)"
    )
    summary_lines.append("")

    # Diagnostic checks for period range issues
    warnings = []
    boundary_count = 0
    mismatch_count = 0

    for roi_id, result in sorted(results.items()):
        if "error" in result:
            continue

        dominant_period = result.get("dominant_period", 0)
        peaks = result.get("frequency_peaks", [])

        if not peaks or dominant_period == 0:
            continue

        # Get period range from first valid result
        if "relevant_periods" in result and len(result["relevant_periods"]) > 0:
            min_period = result["relevant_periods"].min()
            max_period = result["relevant_periods"].max()

            # Check if dominant period is at boundary (within 5%)
            boundary_threshold = (max_period - min_period) * 0.05
            if (
                abs(dominant_period - max_period) < boundary_threshold
                or abs(dominant_period - min_period) < boundary_threshold
            ):
                boundary_count += 1

            # Check if top peak has much higher power than dominant
            if len(peaks) > 0:
                top_peak_power = peaks[0]["power"]
                dominant_power = result.get("dominant_power", 0)

                # If top peak is not the dominant period and has similar/higher power
                if (
                    abs(peaks[0]["period_hours"] - dominant_period) > 0.5
                    and top_peak_power >= dominant_power * 0.9
                ):
                    mismatch_count += 1

    # Add warnings if issues detected
    if boundary_count > n_valid * 0.3:  # More than 30% at boundaries
        warnings.append(
            f"⚠️  WARNING: {boundary_count}/{n_valid} ROIs have dominant periods at range boundaries.\n"
            f"   This suggests the period range may be too narrow.\n"
            f"   Consider expanding the period range to capture true rhythms."
        )

    if mismatch_count > 0:
        warnings.append(
            f"⚠️  WARNING: {mismatch_count}/{n_valid} ROIs show mismatches between\n"
            f"   dominant period and top frequency components.\n"
            f"   This indicates the period range may not match your data."
        )

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

        dominant_period = result.get("dominant_period", 0)
        dominant_power = result.get("dominant_power", 0)
        p_value = result.get("p_value", 1.0)
        is_significant = result.get("is_significant", False)
        n_peaks = len(result.get("frequency_peaks", []))

        summary_lines.append(f"  Dominant period: {dominant_period:.2f} hours")
        summary_lines.append(f"  Spectral power: {dominant_power:.2e}")

        # Show significance
        if is_significant:
            summary_lines.append(f"  ✓ Significant rhythm detected (p={p_value:.4f})")
        else:
            summary_lines.append(f"  ✗ No significant rhythm (p={p_value:.4f})")

        summary_lines.append(f"  Number of peaks detected: {n_peaks}")

        # Check for boundary issue in this specific ROI
        if "relevant_periods" in result and len(result["relevant_periods"]) > 0:
            min_p = result["relevant_periods"].min()
            max_p = result["relevant_periods"].max()
            boundary_threshold = (max_p - min_p) * 0.05

            if abs(dominant_period - max_p) < boundary_threshold:
                summary_lines.append(f"  ⚠️  Period at upper boundary ({max_p:.2f}h)")
            elif abs(dominant_period - min_p) < boundary_threshold:
                summary_lines.append(f"  ⚠️  Period at lower boundary ({min_p:.2f}h)")

        # Show top 3 peaks
        peaks = result.get("frequency_peaks", [])[:3]
        if peaks:
            summary_lines.append("  Top frequency components:")
            for i, peak in enumerate(peaks, 1):
                marker = ""
                # Mark if this peak has significantly different period than dominant
                if i == 1 and abs(peak["period_hours"] - dominant_period) > 0.5:
                    marker = " ⚠️"
                summary_lines.append(
                    f"    {i}. {peak['period_hours']:.2f}h (power: {peak['power']:.2e}){marker}"
                )

        summary_lines.append("")

    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)
