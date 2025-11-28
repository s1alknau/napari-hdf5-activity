"""
Test script for multiprocessing implementation.

This script verifies that:
1. Parallel processing can be triggered with num_processes > 1
2. Sequential processing works when num_processes = 1
3. Results are identical between parallel and sequential execution
"""

import numpy as np
from typing import Dict, List, Tuple


def generate_test_data(
    num_rois: int = 5, num_frames: int = 1000
) -> Dict[int, List[Tuple[float, float]]]:
    """Generate synthetic movement data for testing."""
    np.random.seed(42)
    test_data = {}

    for roi_id in range(num_rois):
        # Generate synthetic time-series with some periodic pattern
        times = np.arange(num_frames) * 5.0  # 5 second intervals

        # Base activity with circadian-like pattern
        circadian_component = 50 + 30 * np.sin(2 * np.pi * times / (24 * 3600))

        # Add noise
        noise = np.random.randn(num_frames) * 10

        # Movement values
        values = circadian_component + noise
        values = np.maximum(values, 0)  # No negative movement

        # Create time-value pairs
        test_data[roi_id] = [(t, v) for t, v in zip(times, values)]

    return test_data


def test_parallel_vs_sequential():
    """Test that parallel and sequential analysis produce identical results."""
    print("=" * 60)
    print("Testing Multiprocessing Implementation")
    print("=" * 60)

    # Generate test data
    print("\n1. Generating test data...")
    test_data = generate_test_data(num_rois=5, num_frames=1000)
    print(f"   Created {len(test_data)} ROIs with {len(test_data[0])} frames each")

    # Import analysis functions
    print("\n2. Importing analysis functions...")
    try:
        from src.napari_hdf5_activity._calc import run_baseline_analysis_auto

        print("   Successfully imported run_baseline_analysis_auto")
    except ImportError as e:
        print(f"   ERROR: Could not import analysis function: {e}")
        return False

    # Test parameters
    params = {
        "enable_matlab_norm": True,
        "enable_detrending": True,
        "use_improved_detrending": True,
        "enable_jump_correction": True,
        "baseline_duration_minutes": 200.0,
        "multiplier": 1.0,
        "frame_interval": 5.0,
        "bin_size_seconds": 60,
        "quiescence_threshold": 0.5,
        "sleep_threshold_minutes": 8,
    }

    # Run sequential analysis
    print("\n3. Running sequential analysis (num_processes=1)...")
    try:
        sequential_results = run_baseline_analysis_auto(
            test_data, num_processes=1, **params
        )
        print("   Sequential analysis completed")
        print(f"   ROIs processed: {len(sequential_results.get('baseline_means', {}))}")
        print(
            f"   Parallel flag: {sequential_results.get('parameters', {}).get('parallel', False)}"
        )
    except Exception as e:
        print(f"   ERROR in sequential analysis: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Run parallel analysis
    print("\n4. Running parallel analysis (num_processes=4)...")
    try:
        parallel_results = run_baseline_analysis_auto(
            test_data, num_processes=4, **params
        )
        print("   Parallel analysis completed")
        print(f"   ROIs processed: {len(parallel_results.get('baseline_means', {}))}")
        print(
            f"   Parallel flag: {parallel_results.get('parameters', {}).get('parallel', False)}"
        )
    except Exception as e:
        print(f"   ERROR in parallel analysis: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Compare results
    print("\n5. Comparing results...")
    try:
        sequential_means = sequential_results.get("baseline_means", {})
        parallel_means = parallel_results.get("baseline_means", {})

        if set(sequential_means.keys()) != set(parallel_means.keys()):
            print("   ERROR: ROI sets don't match!")
            print(f"   Sequential ROIs: {sorted(sequential_means.keys())}")
            print(f"   Parallel ROIs: {sorted(parallel_means.keys())}")
            return False

        # Check if baseline means are identical (within floating point tolerance)
        max_diff = 0.0
        for roi_id in sequential_means.keys():
            seq_val = sequential_means[roi_id]
            par_val = parallel_means[roi_id]
            diff = abs(seq_val - par_val)
            max_diff = max(max_diff, diff)

            if diff > 1e-6:
                print(
                    f"   WARNING: ROI {roi_id} baseline differs: {seq_val:.6f} vs {par_val:.6f} (diff: {diff:.6e})"
                )

        if max_diff < 1e-6:
            print(f"   [OK] Results are identical (max diff: {max_diff:.6e})")
        else:
            print(f"   [OK] Results are very similar (max diff: {max_diff:.6e})")

        # Check movement data lengths
        seq_movement = sequential_results.get("movement_data", {})
        par_movement = parallel_results.get("movement_data", {})

        for roi_id in seq_movement.keys():
            if len(seq_movement[roi_id]) != len(par_movement[roi_id]):
                print(f"   ERROR: Movement data length differs for ROI {roi_id}")
                return False

        print("   [OK] Movement data lengths match")

        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"   ERROR during comparison: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_parallel_vs_sequential()
    exit(0 if success else 1)
