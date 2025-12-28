"""
Performance test with real HDF5 data.

Tests the complete processing pipeline including:
- HDF5 file reading
- Frame loading and chunking
- RGB→Grayscale conversion (if applicable)
- Movement detection
- Fraction movement calculation

Usage:
    python test_real_data_performance.py "path/to/file.h5"
"""

import sys
import time
import h5py
import numpy as np
from pathlib import Path


def get_file_info(file_path: str):
    """Extract file information."""
    print("\n" + "=" * 70)
    print("FILE INFORMATION")
    print("=" * 70)

    with h5py.File(file_path, "r") as f:
        # Detect structure
        if "frames" in f:
            dataset = f["frames"]
            structure_type = "stacked_frames"
        elif "images" in f:
            dataset = f["images"]
            structure_type = "individual_frames"
            # Get first frame for info
            first_key = sorted(dataset.keys())[0]
            first_frame = dataset[first_key]
            print("Structure: Individual frames in 'images/' group")
            print(f"Number of frames: {len(dataset.keys())}")
            print(f"Frame shape: {first_frame.shape}")
            print(f"Data type: {first_frame.dtype}")
            return len(dataset.keys()), first_frame.shape, first_frame.dtype
        else:
            print("ERROR: Unknown HDF5 structure")
            return None, None, None

        print(f"Structure: {structure_type}")
        print(f"Number of frames: {len(dataset)}")
        print(f"Frame shape: {dataset[0].shape}")
        print(f"Data type: {dataset.dtype}")

        # Calculate file size
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")

        # Estimate total data size
        frame_size = np.prod(dataset[0].shape) * dataset.dtype.itemsize
        total_size_mb = (frame_size * len(dataset)) / (1024 * 1024)
        print(f"Total data size: {total_size_mb:.1f} MB")

        # Check if RGB
        is_rgb = len(dataset[0].shape) == 3 and dataset[0].shape[2] == 3
        print(f"Color mode: {'RGB (3 channels)' if is_rgb else 'Grayscale'}")

        return len(dataset), dataset[0].shape, dataset.dtype


def run_performance_test(
    file_path: str, chunk_sizes=[20, 50, 100], process_counts=[1, 2, 4]
):
    """Run performance test with real data."""

    print("\n" + "=" * 70)
    print("REAL DATA PERFORMANCE TEST")
    print("=" * 70)
    print(f"File: {Path(file_path).name}")

    # Get file info
    num_frames, frame_shape, dtype = get_file_info(file_path)
    if num_frames is None:
        return

    # Import required modules
    try:
        from src.napari_hdf5_activity._reader import (
            process_single_file_in_parallel_dual_structure,
        )
    except ImportError as e:
        print(f"\nERROR: Could not import modules: {e}")
        print("Make sure you're in the correct directory and the package is installed")
        return

    # Create dummy masks for testing (6 ROIs)
    print("\n" + "-" * 70)
    print("Creating test ROIs...")
    print("-" * 70)

    # Simple circular masks
    if len(frame_shape) == 3:  # RGB
        height, width, _ = frame_shape
    else:  # Grayscale
        height, width = frame_shape

    masks = []
    roi_size = min(50, height // 4, width // 4)
    positions = [
        (height // 4, width // 4),
        (height // 4, 3 * width // 4),
        (height // 2, width // 2),
        (3 * height // 4, width // 4),
        (3 * height // 4, 3 * width // 4),
        (height // 2, width // 4),
    ]

    for i, (cy, cx) in enumerate(positions[:6]):
        mask = np.zeros((height, width), dtype=bool)
        y, x = np.ogrid[:height, :width]
        mask_region = (x - cx) ** 2 + (y - cy) ** 2 <= roi_size**2
        mask[mask_region] = True
        masks.append(mask)

    print(f"Created {len(masks)} ROIs (size: {roi_size}px radius)")

    # Progress callback
    def progress_callback(percent, msg):
        if percent % 10 == 0 or percent == 100:
            print(f"  Progress: {percent:.0f}% - {msg}")

    # Test different configurations
    results = []

    print("\n" + "=" * 70)
    print("PERFORMANCE TESTS")
    print("=" * 70)

    for chunk_size in chunk_sizes:
        for num_processes in process_counts:
            print("\n" + "-" * 70)
            print(f"Test: chunk_size={chunk_size}, num_processes={num_processes}")
            print("-" * 70)

            start_time = time.time()

            try:
                _, roi_changes, _ = process_single_file_in_parallel_dual_structure(
                    file_path,
                    masks,
                    chunk_size=chunk_size,
                    progress_callback=progress_callback,
                    frame_interval=5.0,
                    num_processes=num_processes,
                )

                elapsed = time.time() - start_time

                # Verify results
                total_changes = sum(len(changes) for changes in roi_changes.values())

                print(f"\n  ✓ Completed in {elapsed:.2f} seconds")
                print(f"  Total changes detected: {total_changes}")
                print(f"  Processing rate: {num_frames / elapsed:.1f} frames/sec")

                results.append(
                    {
                        "chunk_size": chunk_size,
                        "num_processes": num_processes,
                        "time": elapsed,
                        "fps": num_frames / elapsed,
                        "total_changes": total_changes,
                    }
                )

            except Exception as e:
                print(f"\n  ✗ ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"File: {Path(file_path).name}")
    print(f"Frames: {num_frames}")
    print(f"Resolution: {frame_shape}")
    print(f"ROIs: {len(masks)}")
    print("\nConfiguration                Time      FPS      Speedup")
    print("-" * 70)

    # Sort by chunk_size, then num_processes
    results.sort(key=lambda x: (x["chunk_size"], x["num_processes"]))

    # Find baseline (chunk_size=20, num_processes=1 if available)
    baseline_time = None
    for r in results:
        if r["chunk_size"] == chunk_sizes[0] and r["num_processes"] == 1:
            baseline_time = r["time"]
            break

    if baseline_time is None and results:
        baseline_time = results[0]["time"]

    for r in results:
        speedup = baseline_time / r["time"] if baseline_time else 1.0
        print(
            f"chunk={r['chunk_size']:3d}, proc={r['num_processes']:1d}    "
            f"{r['time']:6.2f}s   {r['fps']:5.1f}    {speedup:5.2f}x"
        )

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if results:
        # Find fastest configuration
        fastest = min(results, key=lambda x: x["time"])
        print("\n✓ Fastest configuration:")
        print(f"  Chunk size: {fastest['chunk_size']} frames")
        print(f"  Processes: {fastest['num_processes']}")
        print(f"  Time: {fastest['time']:.2f}s")
        print(f"  Processing rate: {fastest['fps']:.1f} frames/sec")

        # Find best chunk size for 2 processes
        best_for_2proc = min(
            [r for r in results if r["num_processes"] == 2],
            key=lambda x: x["time"],
            default=None,
        )

        if best_for_2proc:
            print("\n✓ Best for 2 processes (balanced):")
            print(f"  Chunk size: {best_for_2proc['chunk_size']} frames")
            print(f"  Time: {best_for_2proc['time']:.2f}s")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_real_data_performance.py <path_to_h5_file>")
        print("\nExample:")
        print('  python test_real_data_performance.py "C:\\Users\\...\\file.h5"')
        print("\nOptional: Customize test parameters in the script")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    # Customizable test parameters
    chunk_sizes = [20, 50, 100]  # Modify as needed
    process_counts = [1, 2, 4]  # Modify as needed

    run_performance_test(file_path, chunk_sizes, process_counts)


if __name__ == "__main__":
    main()
