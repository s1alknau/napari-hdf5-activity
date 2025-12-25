"""
_circadian_coherence.py - Coherence and phase synchronization analysis

This module implements frequency-domain methods to analyze how well different ROIs
synchronize at specific frequencies (especially circadian frequencies around 24h).
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import signal


def calculate_coherence(
    signal1: np.ndarray,
    signal2: np.ndarray,
    sampling_interval: float = 60.0,
    nperseg: int = None,
    target_period_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Calculate magnitude-squared coherence between two signals.

    Coherence measures the correlation between two signals as a function of frequency.
    Values range from 0 (no correlation) to 1 (perfect correlation) at each frequency.

    Args:
        signal1: First activity time series
        signal2: Second activity time series
        sampling_interval: Time between samples (seconds)
        nperseg: Length of each segment for Welch's method (default: 256 or len//8)
        target_period_hours: Target period to analyze (default: 24h)

    Returns:
        Dictionary containing:
        - frequencies: Array of frequencies (Hz)
        - periods: Array of periods (hours)
        - coherence: Coherence values at each frequency
        - circadian_coherence: Coherence at target period
        - is_synchronized: Whether ROIs are synchronized at target frequency
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

    if len(signal1) < 10:
        return {"error": "Signals too short for coherence analysis"}

    # Auto-select segment length if not provided
    if nperseg is None:
        nperseg = min(256, len(signal1) // 8)
        nperseg = max(nperseg, 16)  # Minimum segment size

    # Calculate coherence using Welch's method
    sampling_freq = 1.0 / sampling_interval
    freqs, coherence = signal.coherence(
        signal1, signal2, fs=sampling_freq, nperseg=nperseg, noverlap=nperseg // 2
    )

    # Convert frequencies to periods (hours)
    periods = np.zeros_like(freqs)
    nonzero_mask = freqs > 0
    periods[nonzero_mask] = (1.0 / freqs[nonzero_mask]) / 3600.0

    # Find coherence at target period
    # Look for period within ±20% of target (adaptive range)
    period_tolerance = 0.2  # 20% tolerance
    min_period = target_period_hours * (1 - period_tolerance)
    max_period = target_period_hours * (1 + period_tolerance)
    target_mask = (periods >= min_period) & (periods <= max_period)

    if np.any(target_mask):
        circadian_coherence = np.max(coherence[target_mask])
        circadian_period_idx = np.where(target_mask)[0][
            np.argmax(coherence[target_mask])
        ]
        circadian_period = periods[circadian_period_idx]
        circadian_frequency = freqs[circadian_period_idx]
    else:
        # If no data in target range, use maximum coherence
        circadian_coherence = np.max(coherence)
        circadian_period_idx = np.argmax(coherence)
        circadian_period = (
            periods[circadian_period_idx] if len(periods) > 0 else target_period_hours
        )
        circadian_frequency = (
            freqs[circadian_period_idx]
            if len(freqs) > 0
            else 1.0 / (target_period_hours * 3600)
        )

    # Synchronization threshold: coherence > 0.6 at target frequency
    is_synchronized = circadian_coherence > 0.6

    return {
        "frequencies": freqs,
        "periods": periods,
        "coherence": coherence,
        "circadian_coherence": circadian_coherence,
        "circadian_period": circadian_period,
        "circadian_frequency": circadian_frequency,
        "is_synchronized": is_synchronized,
        "max_coherence": np.max(coherence),
        "max_coherence_period": (
            periods[np.argmax(coherence)] if len(periods) > 0 else 0
        ),
        "target_period_hours": target_period_hours,
    }


def calculate_coherence_matrix(
    movement_data: Dict[int, List[Tuple[float, float]]],
    sampling_interval: float = 5.0,
    bin_size_seconds: int = 60,
    target_period_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Calculate coherence matrix for all ROI pairs at a specific period.

    Args:
        movement_data: Dictionary mapping ROI ID to (time, value) tuples
        sampling_interval: Time interval between samples (seconds)
        bin_size_seconds: Bin size for data averaging
        target_period_hours: Period to analyze (default: 24h)

    Returns:
        Dictionary with coherence matrix and ROI information
    """
    from ._fisher_analysis import _bin_data

    # Prepare data
    roi_ids = sorted(movement_data.keys())
    roi_signals = {}

    for roi_id in roi_ids:
        data = movement_data[roi_id]
        if not data or len(data) < 10:
            continue

        times = np.array([t for t, _ in data])
        values = np.array([v for _, v in data])

        if bin_size_seconds:
            values, _ = _bin_data(times, values, bin_size_seconds)

        roi_signals[roi_id] = values

    # Calculate coherence for all pairs
    n_rois = len(roi_signals)
    roi_list = sorted(roi_signals.keys())
    coherence_matrix = np.zeros((n_rois, n_rois))
    pairwise_coherence = {}

    for i, roi1 in enumerate(roi_list):
        for j, roi2 in enumerate(roi_list):
            if i == j:
                coherence_matrix[i, j] = 1.0
                continue

            # Skip if already computed (symmetric)
            if j < i:
                coherence_matrix[i, j] = coherence_matrix[j, i]
                continue

            result = calculate_coherence(
                roi_signals[roi1],
                roi_signals[roi2],
                sampling_interval=(
                    bin_size_seconds if bin_size_seconds else sampling_interval
                ),
                target_period_hours=target_period_hours,
            )

            if "error" not in result:
                coherence_matrix[i, j] = result["circadian_coherence"]
                pairwise_coherence[(roi1, roi2)] = result

    return {
        "roi_ids": roi_list,
        "coherence_matrix": coherence_matrix,
        "pairwise_coherence": pairwise_coherence,
        "n_rois": n_rois,
        "target_period_hours": target_period_hours,
    }


def phase_synchronization_index(
    signal1: np.ndarray,
    signal2: np.ndarray,
    dominant_period_hours: float,
    sampling_interval: float = 60.0,
) -> Dict[str, Any]:
    """
    Calculate phase synchronization index between two signals.

    The phase synchronization index (PSI) measures how consistently two signals
    maintain a constant phase relationship over time.

    Args:
        signal1: First activity signal
        signal2: Second activity signal
        dominant_period_hours: Dominant circadian period (e.g., 24 hours)
        sampling_interval: Time between samples (seconds)

    Returns:
        Dictionary with synchronization metrics
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

    if len(signal1) < 10:
        return {"error": "Signals too short for phase analysis"}

    # Extract instantaneous phase using Hilbert transform
    analytic1 = signal.hilbert(signal1 - np.mean(signal1))
    analytic2 = signal.hilbert(signal2 - np.mean(signal2))

    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    # Calculate phase difference
    phase_diff = phase2 - phase1

    # Wrap to [-π, π]
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

    # Calculate phase locking value (PLV)
    # PLV measures consistency of phase relationship
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    # Calculate mean phase difference
    mean_phase_diff = np.arctan2(
        np.mean(np.sin(phase_diff)), np.mean(np.cos(phase_diff))
    )

    # Convert to hours
    mean_phase_diff_hours = (mean_phase_diff / (2 * np.pi)) * dominant_period_hours

    # Standard deviation of phase difference (circular)
    phase_std = np.sqrt(-2 * np.log(plv)) if plv > 0 else np.pi

    # Determine synchronization quality
    if plv > 0.8:
        sync_quality = "strong"
    elif plv > 0.5:
        sync_quality = "moderate"
    elif plv > 0.3:
        sync_quality = "weak"
    else:
        sync_quality = "none"

    return {
        "phase_locking_value": plv,
        "mean_phase_difference_radians": mean_phase_diff,
        "mean_phase_difference_hours": mean_phase_diff_hours,
        "phase_std_radians": phase_std,
        "synchronization_quality": sync_quality,
        "is_synchronized": plv > 0.5,
    }


def detect_phase_clusters(
    movement_data: Dict[int, List[Tuple[float, float]]],
    dominant_period_hours: float = 24.0,
    sampling_interval: float = 5.0,
    bin_size_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Detect groups of ROIs with similar circadian phases.

    Args:
        movement_data: Dictionary mapping ROI ID to (time, value) tuples
        dominant_period_hours: Period to analyze (default: 24h)
        sampling_interval: Time interval between samples (seconds)
        bin_size_seconds: Bin size for data averaging

    Returns:
        Dictionary with phase cluster information
    """
    from ._fisher_analysis import _bin_data

    # Extract phase for each ROI
    roi_phases = {}

    for roi_id, data in movement_data.items():
        if not data or len(data) < 10:
            continue

        times = np.array([t for t, _ in data])
        values = np.array([v for _, v in data])

        if bin_size_seconds:
            values, _ = _bin_data(times, values, bin_size_seconds)

        # Calculate phase using Hilbert transform
        analytic_signal = signal.hilbert(values - np.mean(values))
        instantaneous_phase = np.angle(analytic_signal)

        # Mean phase (circular mean)
        mean_phase = np.arctan2(
            np.mean(np.sin(instantaneous_phase)), np.mean(np.cos(instantaneous_phase))
        )

        # Convert to hours (peak activity time)
        phase_hours = (
            (mean_phase / (2 * np.pi)) * dominant_period_hours
        ) % dominant_period_hours

        roi_phases[roi_id] = {
            "phase_radians": mean_phase,
            "phase_hours": phase_hours,
            "amplitude": np.mean(np.abs(analytic_signal)),
        }

    # Cluster ROIs by phase (simple binning into 4 quadrants)
    clusters = {
        "early_active": [],
        "mid_active": [],
        "late_active": [],
        "night_active": [],
    }

    for roi_id, phase_info in roi_phases.items():
        phase_h = phase_info["phase_hours"]

        if 0 <= phase_h < 6:
            clusters["early_active"].append(roi_id)
        elif 6 <= phase_h < 12:
            clusters["mid_active"].append(roi_id)
        elif 12 <= phase_h < 18:
            clusters["late_active"].append(roi_id)
        else:
            clusters["night_active"].append(roi_id)

    return {
        "roi_phases": roi_phases,
        "phase_clusters": clusters,
        "n_rois": len(roi_phases),
        "dominant_period_hours": dominant_period_hours,
    }


def generate_coherence_summary(coherence_results: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of coherence analysis.

    Args:
        coherence_results: Results from coherence analysis

    Returns:
        Formatted summary string
    """
    summary_lines = [
        "=" * 60,
        "COHERENCE & PHASE SYNCHRONIZATION ANALYSIS",
        "=" * 60,
        "",
    ]

    n_rois = coherence_results["n_rois"]
    target_period = coherence_results.get("target_period_hours", 24.0)

    summary_lines.append(f"Total ROIs analyzed: {n_rois}")
    summary_lines.append(f"Target period: {target_period:.1f} hours")
    summary_lines.append("")

    # Find highly coherent pairs
    coherent_pairs = []
    for (roi1, roi2), result in coherence_results["pairwise_coherence"].items():
        if result["is_synchronized"]:
            coherent_pairs.append(
                {
                    "roi1": roi1,
                    "roi2": roi2,
                    "coherence": result["circadian_coherence"],
                    "period": result["circadian_period"],
                }
            )

    coherent_pairs = sorted(coherent_pairs, key=lambda x: x["coherence"], reverse=True)

    summary_lines.append(f"Synchronized pairs (coherence > 0.6): {len(coherent_pairs)}")

    if coherent_pairs:
        summary_lines.append("")
        summary_lines.append("Top 5 most coherent ROI pairs:")
        for i, pair in enumerate(coherent_pairs[:5], 1):
            summary_lines.append(
                f"  {i}. ROI {pair['roi1']} ↔ ROI {pair['roi2']}: "
                f"coherence={pair['coherence']:.3f} at {pair['period']:.1f}h"
            )

    summary_lines.append("")
    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)


def export_coherence_to_excel(file_path: str, coherence_results: Dict) -> None:
    """
    Export Coherence Analysis results to Excel format.

    Args:
        file_path: Path to save Excel file
        coherence_results: Results from calculate_coherence_matrix

    Creates Excel file with sheets:
    - Coherence_Matrix: Full coherence matrix
    - Pairwise_Coherence: Detailed pairwise results
    - Parameters: Analysis parameters
    """
    import pandas as pd

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        # Sheet 1: Coherence Matrix
        coherence_matrix = coherence_results.get("coherence_matrix", np.array([]))
        roi_ids = coherence_results.get("roi_ids", [])
        target_period = coherence_results.get("target_period_hours", 24.0)

        if len(coherence_matrix) > 0:
            coherence_df = pd.DataFrame(
                coherence_matrix,
                index=[f"ROI {r}" for r in roi_ids],
                columns=[f"ROI {r}" for r in roi_ids],
            )
            coherence_df.to_excel(writer, sheet_name="Coherence_Matrix")

        # Sheet 2: Pairwise Coherence
        pairwise_coherence = coherence_results.get("pairwise_coherence", {})
        pairwise_data = []

        for (roi1, roi2), result in pairwise_coherence.items():
            pairwise_data.append(
                {
                    "ROI_1": roi1,
                    "ROI_2": roi2,
                    "Coherence": result.get("circadian_coherence", 0),
                    "Coherence_Period": result.get("circadian_period", target_period),
                    "Max_Coherence": result.get("max_coherence", 0),
                    "Max_Coherence_Period": result.get("max_coherence_period", 0),
                    "Synchronized": (
                        "Yes" if result.get("is_synchronized", False) else "No"
                    ),
                }
            )

        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df = pairwise_df.sort_values("Coherence", ascending=False)
        pairwise_df.to_excel(writer, sheet_name="Pairwise_Coherence", index=False)

        # Sheet 3: Parameters
        params_df = pd.DataFrame(
            {
                "Parameter": ["Target Period", "Number of ROIs"],
                "Value": [f"{target_period:.1f} hours", str(len(roi_ids))],
            }
        )
        params_df.to_excel(writer, sheet_name="Parameters", index=False)
