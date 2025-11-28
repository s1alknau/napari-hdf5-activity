"""
Test script to measure actual speedup from multiprocessing.

This script compares execution time between sequential and parallel processing.
"""

import numpy as np
import time
from typing import Dict, List, Tuple


def generate_test_data(
    num_rois: int = 10, num_frames: int = 10000
) -> Dict[int, List[Tuple[float, float]]]:
    """Generate synthetic movement data for timing test."""
    np.random.seed(42)
    test_data = {}

    for roi_id in range(num_rois):
        times = np.arange(num_frames) * 5.0
        circadian_component = 50 + 30 * np.sin(2 * np.pi * times / (24 * 3600))
        noise = np.random.randn(num_frames) * 10
        values = np.maximum(circadian_component + noise, 0)
        test_data[roi_id] = [(t, v) for t, v in zip(times, values)]

    return test_data


def run_timing_test():
    """Compare execution times between sequential and parallel."""
    print("=" * 70)
    print("Multiprocessing Performance Test")
    print("=" * 70)

    # Generate larger test data (realistic size)
    print("\nGenerating test data (10 ROIs, 10000 frames)...")
    test_data = generate_test_data(num_rois=10, num_frames=10000)
    print(f"Created {len(test_data)} ROIs with {len(test_data[0])} frames each")
    print("This simulates ~14 hours of recording at 5s intervals")

    # Import analysis function
    try:
        from src.napari_hdf5_activity._calc import run_baseline_analysis
    except ImportError as e:
        print(f"ERROR: Could not import: {e}")
        return

    # Test parameters
    params = {
        "enable_matlab_norm": True,
        "enable_detrending": True,
        "use_improved_detrending": True,
        "baseline_duration_minutes": 200.0,
        "multiplier": 1.0,
        "frame_interval": 5.0,
        "bin_size_seconds": 60,
        "quiescence_threshold": 0.5,
        "sleep_threshold_minutes": 8,
    }

    # Test 1: Sequential (num_processes=1)
    print("\n" + "-" * 70)
    print("Test 1: Sequential Processing (num_processes=1)")
    print("-" * 70)
    start_time = time.time()
    sequential_results = run_baseline_analysis(test_data, num_processes=1, **params)
    sequential_time = time.time() - start_time
    print(f"Time: {sequential_time:.2f} seconds")
    print(f"Parallel mode: {sequential_results['parameters'].get('parallel', False)}")

    # Test 2: Parallel (num_processes=2)
    print("\n" + "-" * 70)
    print("Test 2: Parallel Processing (num_processes=2)")
    print("-" * 70)
    start_time = time.time()
    parallel_2_results = run_baseline_analysis(test_data, num_processes=2, **params)
    parallel_2_time = time.time() - start_time
    speedup_2 = sequential_time / parallel_2_time
    print(f"Time: {parallel_2_time:.2f} seconds")
    print(f"Speedup: {speedup_2:.2f}x")
    print(f"Parallel mode: {parallel_2_results['parameters'].get('parallel', False)}")

    # Test 3: Parallel (num_processes=4)
    print("\n" + "-" * 70)
    print("Test 3: Parallel Processing (num_processes=4)")
    print("-" * 70)
    start_time = time.time()
    parallel_4_results = run_baseline_analysis(test_data, num_processes=4, **params)
    parallel_4_time = time.time() - start_time
    speedup_4 = sequential_time / parallel_4_time
    print(f"Time: {parallel_4_time:.2f} seconds")
    print(f"Speedup: {speedup_4:.2f}x")
    print(f"Parallel mode: {parallel_4_results['parameters'].get('parallel', False)}")

    # Test 4: Parallel (num_processes=8)
    print("\n" + "-" * 70)
    print("Test 4: Parallel Processing (num_processes=8)")
    print("-" * 70)
    start_time = time.time()
    parallel_8_results = run_baseline_analysis(test_data, num_processes=8, **params)
    parallel_8_time = time.time() - start_time
    speedup_8 = sequential_time / parallel_8_time
    print(f"Time: {parallel_8_time:.2f} seconds")
    print(f"Speedup: {speedup_8:.2f}x")
    print(f"Parallel mode: {parallel_8_results['parameters'].get('parallel', False)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Sequential (1 process):  {sequential_time:.2f}s")
    print(
        f"Parallel (2 processes):  {parallel_2_time:.2f}s  (Speedup: {speedup_2:.2f}x)"
    )
    print(
        f"Parallel (4 processes):  {parallel_4_time:.2f}s  (Speedup: {speedup_4:.2f}x)"
    )
    print(
        f"Parallel (8 processes):  {parallel_8_time:.2f}s  (Speedup: {speedup_8:.2f}x)"
    )
    print("=" * 70)

    # Verify results are identical
    print("\nVerifying result consistency...")
    seq_means = sequential_results.get("baseline_means", {})
    par_means = parallel_4_results.get("baseline_means", {})

    max_diff = 0.0
    for roi_id in seq_means.keys():
        diff = abs(seq_means[roi_id] - par_means[roi_id])
        max_diff = max(max_diff, diff)

    if max_diff < 1e-6:
        print(f"[OK] Results identical (max diff: {max_diff:.6e})")
    else:
        print(f"[WARNING] Results differ (max diff: {max_diff:.6e})")


if __name__ == "__main__":
    run_timing_test()
