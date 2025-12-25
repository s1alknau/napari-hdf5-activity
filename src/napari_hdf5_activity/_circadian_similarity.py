"""
_circadian_similarity.py - ROI similarity and clustering analysis

This module implements methods to compare circadian rhythms between different ROIs,
identify similar patterns, and group ROIs based on their activity profiles.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import signal
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def calculate_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag_hours: float = 12.0,
    sampling_interval: float = 60.0,
) -> Dict[str, Any]:
    """
    Calculate cross-correlation between two activity signals.

    This measures how similar two signals are and detects time shifts (phase differences).

    Args:
        signal1: First activity time series
        signal2: Second activity time series
        max_lag_hours: Maximum time lag to consider (hours)
        sampling_interval: Time between samples (seconds)

    Returns:
        Dictionary containing:
        - correlation: Cross-correlation values
        - lags: Time lags (hours)
        - max_correlation: Maximum correlation coefficient
        - optimal_lag_hours: Time shift at maximum correlation
        - is_similar: Whether signals are significantly similar
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

    if len(signal1) < 10:
        return {"error": "Signals too short for correlation analysis"}

    # Normalize signals
    s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
    s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

    # Calculate cross-correlation
    correlation = signal.correlate(s1, s2, mode="same", method="auto")
    correlation = correlation / len(s1)  # Normalize

    # Calculate lags
    max_lag_samples = int((max_lag_hours * 3600) / sampling_interval)
    max_lag_samples = min(max_lag_samples, len(correlation) // 2)

    center = len(correlation) // 2
    lag_range = slice(center - max_lag_samples, center + max_lag_samples + 1)

    correlation_window = correlation[lag_range]
    lag_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lag_hours = lag_samples * sampling_interval / 3600.0

    # Find maximum correlation
    max_corr_idx = np.argmax(correlation_window)
    max_correlation = correlation_window[max_corr_idx]
    optimal_lag = lag_hours[max_corr_idx]

    # Determine significance (correlation > 0.5 is considered similar)
    is_similar = max_correlation > 0.5

    return {
        "correlation": correlation_window,
        "lags": lag_hours,
        "max_correlation": max_correlation,
        "optimal_lag_hours": optimal_lag,
        "is_similar": is_similar,
        "zero_lag_correlation": correlation_window[max_lag_samples],  # At lag=0
    }


def calculate_roi_correlation_matrix(
    movement_data: Dict[int, List[Tuple[float, float]]],
    sampling_interval: float = 5.0,
    bin_size_seconds: int = 60,
    max_lag_hours: float = 12.0,
) -> Dict[str, Any]:
    """
    Calculate pairwise correlations between all ROIs.

    Args:
        movement_data: Dictionary mapping ROI ID to (time, value) tuples
        sampling_interval: Time interval between samples (seconds)
        bin_size_seconds: Bin size for data averaging
        max_lag_hours: Maximum lag for cross-correlation

    Returns:
        Dictionary containing correlation matrix and ROI pairs
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

    # Calculate all pairwise correlations
    n_rois = len(roi_signals)
    roi_list = sorted(roi_signals.keys())
    correlation_matrix = np.zeros((n_rois, n_rois))
    lag_matrix = np.zeros((n_rois, n_rois))
    pairwise_results = {}

    for i, roi1 in enumerate(roi_list):
        for j, roi2 in enumerate(roi_list):
            if i == j:
                correlation_matrix[i, j] = 1.0
                lag_matrix[i, j] = 0.0
                continue

            # Skip if already computed (symmetric)
            if j < i:
                correlation_matrix[i, j] = correlation_matrix[j, i]
                lag_matrix[i, j] = -lag_matrix[j, i]  # Reverse lag
                continue

            result = calculate_cross_correlation(
                roi_signals[roi1],
                roi_signals[roi2],
                max_lag_hours=max_lag_hours,
                sampling_interval=(
                    bin_size_seconds if bin_size_seconds else sampling_interval
                ),
            )

            if "error" not in result:
                correlation_matrix[i, j] = result["max_correlation"]
                lag_matrix[i, j] = result["optimal_lag_hours"]
                pairwise_results[(roi1, roi2)] = result

    return {
        "roi_ids": roi_list,
        "correlation_matrix": correlation_matrix,
        "lag_matrix": lag_matrix,
        "pairwise_results": pairwise_results,
        "n_rois": n_rois,
    }


def find_similar_roi_pairs(
    correlation_results: Dict[str, Any], threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find ROI pairs with high similarity.

    Args:
        correlation_results: Results from calculate_roi_correlation_matrix
        threshold: Minimum correlation to consider similar

    Returns:
        List of similar ROI pairs with their correlation and lag
    """
    similar_pairs = []

    for (roi1, roi2), result in correlation_results["pairwise_results"].items():
        if result["max_correlation"] >= threshold:
            similar_pairs.append(
                {
                    "roi1": roi1,
                    "roi2": roi2,
                    "correlation": result["max_correlation"],
                    "lag_hours": result["optimal_lag_hours"],
                    "synchronized": abs(result["optimal_lag_hours"]) < 1.0,
                }
            )

    # Sort by correlation (descending)
    similar_pairs = sorted(similar_pairs, key=lambda x: x["correlation"], reverse=True)

    return similar_pairs


def hierarchical_clustering(
    correlation_matrix: np.ndarray, roi_ids: List[int], method: str = "average"
) -> Dict[str, Any]:
    """
    Perform hierarchical clustering on ROIs based on correlation.

    Args:
        correlation_matrix: NxN correlation matrix
        roi_ids: List of ROI IDs
        method: Linkage method ('average', 'single', 'complete', 'ward')

    Returns:
        Dictionary with clustering results
    """
    # Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0

    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(condensed_dist, method=method)

    # Cut tree to get clusters (e.g., at 0.5 distance = 0.5 correlation threshold)
    cluster_labels = hierarchy.fcluster(linkage_matrix, t=0.5, criterion="distance")

    # Group ROIs by cluster
    clusters = {}
    for roi_id, cluster_id in zip(roi_ids, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(roi_id)

    return {
        "linkage_matrix": linkage_matrix,
        "cluster_labels": cluster_labels,
        "clusters": clusters,
        "n_clusters": len(clusters),
        "roi_ids": roi_ids,
    }


def calculate_phase_difference(
    signal1: np.ndarray,
    signal2: np.ndarray,
    dominant_period_hours: float,
    sampling_interval: float = 60.0,
) -> Dict[str, Any]:
    """
    Calculate phase difference between two signals at a specific frequency.

    Args:
        signal1: First activity signal
        signal2: Second activity signal
        dominant_period_hours: Period to analyze (e.g., 24 hours)
        sampling_interval: Time between samples (seconds)

    Returns:
        Dictionary with phase difference information
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

    # Calculate angular frequency
    period_samples = (dominant_period_hours * 3600) / sampling_interval
    omega = 2 * np.pi / period_samples

    # Create time index
    t = np.arange(len(signal1))

    # Fit sinusoid to extract phase
    def fit_sinusoid(signal):
        cos_component = np.cos(omega * t)
        sin_component = np.sin(omega * t)

        # Linear regression
        X = np.column_stack([cos_component, sin_component, np.ones(len(signal))])
        coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)

        A, B, C = coeffs
        phase = np.arctan2(B, A)
        amplitude = np.sqrt(A**2 + B**2)

        return phase, amplitude

    phase1, amp1 = fit_sinusoid(signal1)
    phase2, amp2 = fit_sinusoid(signal2)

    # Calculate phase difference (-π to π)
    phase_diff = phase2 - phase1
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))  # Wrap to [-π, π]

    # Convert to hours
    phase_diff_hours = (phase_diff / (2 * np.pi)) * dominant_period_hours

    # Determine if in-phase or anti-phase
    in_phase = abs(phase_diff_hours) < (dominant_period_hours * 0.25)
    anti_phase = abs(phase_diff_hours) > (dominant_period_hours * 0.75)

    return {
        "phase_difference_radians": phase_diff,
        "phase_difference_hours": phase_diff_hours,
        "amplitude1": amp1,
        "amplitude2": amp2,
        "in_phase": in_phase,
        "anti_phase": anti_phase,
    }


def generate_similarity_summary(
    correlation_results: Dict[str, Any],
    clustering_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate human-readable summary of similarity analysis.

    Args:
        correlation_results: Results from correlation analysis
        clustering_results: Optional clustering results

    Returns:
        Formatted summary string
    """
    summary_lines = [
        "=" * 60,
        "ROI SIMILARITY ANALYSIS",
        "=" * 60,
        "",
    ]

    n_rois = correlation_results["n_rois"]

    summary_lines.append(f"Total ROIs analyzed: {n_rois}")
    summary_lines.append(f"Total pairwise comparisons: {n_rois * (n_rois - 1) // 2}")
    summary_lines.append("")

    # Find similar pairs
    similar_pairs = find_similar_roi_pairs(correlation_results, threshold=0.7)

    summary_lines.append(f"Highly similar pairs (r > 0.7): {len(similar_pairs)}")
    if similar_pairs:
        summary_lines.append("")
        summary_lines.append("Top 5 most similar ROI pairs:")
        for i, pair in enumerate(similar_pairs[:5], 1):
            sync_status = (
                "synchronized"
                if pair["synchronized"]
                else f"lag: {pair['lag_hours']:.1f}h"
            )
            summary_lines.append(
                f"  {i}. ROI {pair['roi1']} ↔ ROI {pair['roi2']}: "
                f"r={pair['correlation']:.3f} ({sync_status})"
            )

    summary_lines.append("")

    # Clustering summary
    if clustering_results:
        n_clusters = clustering_results["n_clusters"]
        summary_lines.append(
            f"Hierarchical clustering: {n_clusters} clusters identified"
        )
        summary_lines.append("")

        for cluster_id, rois in sorted(clustering_results["clusters"].items()):
            summary_lines.append(f"  Cluster {cluster_id}: {len(rois)} ROIs")
            summary_lines.append(f"    ROIs: {', '.join(map(str, rois))}")

    summary_lines.append("")
    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)


def export_similarity_to_excel(file_path: str, similarity_results: Dict) -> None:
    """
    Export ROI Similarity results to Excel format.

    Args:
        file_path: Path to save Excel file
        similarity_results: Results from calculate_roi_correlation_matrix

    Creates Excel file with sheets:
    - Correlation_Matrix: Full correlation matrix
    - Pairwise_Correlations: Sorted pairwise correlations
    - Clusters: Hierarchical clustering results
    """
    import pandas as pd

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        # Sheet 1: Correlation Matrix
        corr_matrix = similarity_results.get("correlation_matrix", np.array([]))
        roi_ids = similarity_results.get("roi_ids", [])

        if len(corr_matrix) > 0:
            corr_df = pd.DataFrame(
                corr_matrix,
                index=[f"ROI {r}" for r in roi_ids],
                columns=[f"ROI {r}" for r in roi_ids],
            )
            corr_df.to_excel(writer, sheet_name="Correlation_Matrix")

        # Sheet 2: Pairwise Correlations
        pairwise_data = []
        for i, roi1 in enumerate(roi_ids):
            for j, roi2 in enumerate(roi_ids):
                if i < j:  # Upper triangle only
                    correlation = corr_matrix[i, j]
                    pairwise_data.append(
                        {
                            "ROI_1": roi1,
                            "ROI_2": roi2,
                            "Correlation": correlation,
                            "Status": (
                                "Synchronized"
                                if correlation > 0.7
                                else ("Moderate" if correlation > 0.3 else "Low")
                            ),
                        }
                    )

        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df = pairwise_df.sort_values("Correlation", ascending=False)
        pairwise_df.to_excel(writer, sheet_name="Pairwise_Correlations", index=False)

        # Sheet 3: Clustering
        if "clustering" in similarity_results:
            cluster_info = similarity_results["clustering"]
            cluster_data = []
            for cluster_id, roi_list in cluster_info.get("clusters", {}).items():
                for roi in roi_list:
                    cluster_data.append(
                        {
                            "ROI": roi,
                            "Cluster": cluster_id,
                            "Cluster_Size": len(roi_list),
                        }
                    )

            cluster_df = pd.DataFrame(cluster_data)
            cluster_df = cluster_df.sort_values(["Cluster", "ROI"])
            cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
