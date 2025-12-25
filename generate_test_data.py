"""
Generate synthetic test data for circadian/rhythmic pattern analysis.

This script creates realistic activity data with known periods, allowing
validation of Fisher Z-transformation and FFT analysis methods.
"""

import numpy as np
import h5py
from pathlib import Path


def generate_rhythmic_signal(
    duration_hours: float,
    sampling_interval_seconds: float,
    period_hours: float,
    amplitude: float = 1.0,
    baseline: float = 0.5,
    noise_level: float = 0.1,
    phase_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic rhythmic activity signal.

    Args:
        duration_hours: Total recording duration in hours
        sampling_interval_seconds: Time between samples in seconds
        period_hours: Dominant period of the rhythm in hours
        amplitude: Amplitude of the oscillation (0-1)
        baseline: Baseline activity level (0-1)
        noise_level: Amount of random noise to add (0-1)
        phase_offset: Phase offset in radians (0-2π)

    Returns:
        Tuple of (times_seconds, activity_values)
    """
    # Calculate number of samples
    duration_seconds = duration_hours * 3600
    n_samples = int(duration_seconds / sampling_interval_seconds)

    # Generate time array
    times = np.arange(n_samples) * sampling_interval_seconds

    # Generate rhythmic signal
    period_seconds = period_hours * 3600
    omega = 2 * np.pi / period_seconds

    # Pure sinusoidal rhythm
    rhythm = amplitude * np.sin(omega * times + phase_offset) + baseline

    # Add random noise
    noise = np.random.normal(0, noise_level, n_samples)
    signal = rhythm + noise

    # Clip to [0, 1] range (realistic activity values)
    signal = np.clip(signal, 0, 1)

    return times, signal


def generate_complex_signal(
    duration_hours: float,
    sampling_interval_seconds: float,
    periods_hours: list[float],
    amplitudes: list[float],
    baseline: float = 0.5,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a signal with multiple rhythmic components.

    Args:
        duration_hours: Total recording duration in hours
        sampling_interval_seconds: Time between samples in seconds
        periods_hours: List of periods for different rhythm components
        amplitudes: List of amplitudes for each component
        baseline: Baseline activity level
        noise_level: Amount of random noise

    Returns:
        Tuple of (times_seconds, activity_values)
    """
    duration_seconds = duration_hours * 3600
    n_samples = int(duration_seconds / sampling_interval_seconds)
    times = np.arange(n_samples) * sampling_interval_seconds

    # Start with baseline
    signal = np.ones(n_samples) * baseline

    # Add each rhythmic component
    for period_hours, amplitude in zip(periods_hours, amplitudes):
        period_seconds = period_hours * 3600
        omega = 2 * np.pi / period_seconds
        signal += amplitude * np.sin(omega * times)

    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    signal += noise

    # Clip to valid range
    signal = np.clip(signal, 0, 1)

    return times, signal


def save_to_hdf5(
    filepath: str,
    roi_data: dict[int, tuple[np.ndarray, np.ndarray]],
    metadata: dict = None,
):
    """
    Save test data to HDF5 file compatible with napari-hdf5-activity plugin.

    Args:
        filepath: Path to output HDF5 file
        roi_data: Dictionary mapping ROI ID to (times, values) tuples
        metadata: Optional metadata dictionary
    """
    with h5py.File(filepath, "w") as f:
        # Get first ROI to determine dimensions
        first_times, first_values = list(roi_data.values())[0]
        n_frames = len(first_times)

        # Create realistic video data showing activity in ROIs
        # This is needed for the reader to recognize the file
        image_size = 128

        # Create ROI masks first (needed for video generation)
        masks_group = f.create_group("masks")
        roi_masks = {}
        for roi_id in roi_data.keys():
            # Create circular ROI masks arranged in a grid
            y, x = np.ogrid[:image_size, :image_size]
            row = (roi_id - 1) // 3  # 3 ROIs per row
            col = (roi_id - 1) % 3
            center_x = col * 40 + 25
            center_y = row * 40 + 25
            radius = 12
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2).astype(
                np.uint8
            )
            masks_group.create_dataset(f"roi_{roi_id}", data=mask)
            roi_masks[roi_id] = mask

        # Generate video frames based on ROI activity
        video = np.zeros((n_frames, image_size, image_size), dtype=np.uint8)

        for frame_idx in range(n_frames):
            # Create a dark background
            frame = np.ones((image_size, image_size), dtype=np.uint8) * 30

            # Add each ROI with brightness based on activity
            for roi_id, (times, values) in roi_data.items():
                activity = values[frame_idx]
                # Convert activity (0-1) to brightness (30-230)
                brightness = int(30 + activity * 200)
                mask = roi_masks[roi_id]
                frame[mask > 0] = brightness

            # Add slight noise for realism
            noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            video[frame_idx] = frame

        # Store video data
        f.create_dataset("video", data=video, compression="gzip")

        # Create groups
        results_group = f.create_group("results")

        # Store each ROI with complete analysis results
        for roi_id, (times, values) in roi_data.items():
            roi_group = results_group.create_group(f"roi_{roi_id}")

            # Store processed_data (raw signal values)
            processed_data = [(t, v) for t, v in zip(times, values)]
            processed_dtype = np.dtype([("time", "f8"), ("value", "f8")])
            processed_array = np.array(processed_data, dtype=processed_dtype)
            roi_group.create_dataset("processed_data", data=processed_array)

            # Store fraction_data (same as processed for test data)
            roi_group.create_dataset("fraction_data", data=processed_array)

            # Store movement_data (required for some analyses)
            roi_group.create_dataset("movement_data", data=processed_array)

            # Store metadata that widget might need
            roi_group.attrs["roi_id"] = roi_id
            roi_group.attrs["n_samples"] = len(times)
            roi_group.attrs["duration_seconds"] = times[-1] - times[0]

        # Store metadata
        meta_group = f.create_group("metadata")
        if metadata:
            for key, value in metadata.items():
                meta_group.attrs[key] = value
        # Add required metadata
        meta_group.attrs["frame_interval"] = metadata.get(
            "sampling_interval_seconds", 5.0
        )
        meta_group.attrs["n_frames"] = n_frames
        meta_group.attrs["n_rois"] = len(roi_data)


def create_test_dataset_1_short_cycles():
    """
    Test Dataset 1: Short activity cycles (0.5-6 hours)
    Simulates rapid activity patterns like feeding, grooming cycles.
    Duration: 8.5 hours
    """
    print("Creating Test Dataset 1: Short Cycles (8.5 hours)")

    duration = 8.5  # hours
    sampling_interval = 5.0  # seconds

    roi_data = {}

    # ROI 1: 4.2 hour period (strong rhythm)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=4.2,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.05,
    )
    roi_data[1] = (times, values)

    # ROI 2: 4.0 hour period with phase shift
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=4.0,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.05,
        phase_offset=np.pi / 4,  # 45° phase shift
    )
    roi_data[2] = (times, values)

    # ROI 3: 3.0 hour period (faster rhythm)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=3.0,
        amplitude=0.35,
        baseline=0.5,
        noise_level=0.08,
    )
    roi_data[3] = (times, values)

    # ROI 4: 4.2 hour period (same as ROI 1)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=4.2,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.05,
        phase_offset=np.pi,  # 180° phase shift from ROI 1
    )
    roi_data[4] = (times, values)

    # ROI 5: 3.0 hour period (similar to ROI 3)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=3.0,
        amplitude=0.35,
        baseline=0.5,
        noise_level=0.08,
        phase_offset=np.pi / 6,
    )
    roi_data[5] = (times, values)

    # ROI 6: 1.0 hour period (very fast rhythm)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=1.0,
        amplitude=0.25,
        baseline=0.5,
        noise_level=0.12,
    )
    roi_data[6] = (times, values)

    metadata = {
        "duration_hours": duration,
        "sampling_interval_seconds": sampling_interval,
        "description": "Short activity cycles (0.5-6h range)",
        "expected_periods": "4.2h, 4.0h, 3.0h, 4.2h, 3.0h, 1.0h",
    }

    output_path = Path("test_data_short_cycles.h5")
    save_to_hdf5(str(output_path), roi_data, metadata)
    print(f"  Saved: {output_path}")
    print(f"  ROIs: {len(roi_data)}")
    print("  Expected periods: 4.2h, 4.0h, 3.0h, 4.2h, 3.0h, 1.0h")
    print()


def create_test_dataset_2_circadian():
    """
    Test Dataset 2: Circadian rhythms (12-36 hours)
    Simulates classic 24h circadian patterns.
    Duration: 72 hours (3 days)
    """
    print("Creating Test Dataset 2: Circadian Rhythms (72 hours)")

    duration = 72  # hours
    sampling_interval = 60.0  # 1 minute (lower resolution for long recording)

    roi_data = {}

    # ROI 1: Perfect 24h circadian rhythm
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=24.0,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.05,
    )
    roi_data[1] = (times, values)

    # ROI 2: 23.5h period (slightly shorter free-running rhythm)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=23.5,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.06,
    )
    roi_data[2] = (times, values)

    # ROI 3: 24h rhythm with 12h harmonic (ultradian component)
    times, values = generate_complex_signal(
        duration,
        sampling_interval,
        periods_hours=[24.0, 12.0],
        amplitudes=[0.35, 0.15],
        baseline=0.5,
        noise_level=0.07,
    )
    roi_data[3] = (times, values)

    # ROI 4: 24h rhythm, phase delayed by 6 hours
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=24.0,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.05,
        phase_offset=np.pi / 2,  # 90° = 6h delay
    )
    roi_data[4] = (times, values)

    # ROI 5: 20h ultradian rhythm
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=20.0,
        amplitude=0.35,
        baseline=0.5,
        noise_level=0.08,
    )
    roi_data[5] = (times, values)

    # ROI 6: Noisy 24h rhythm (low amplitude)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=24.0,
        amplitude=0.2,
        baseline=0.5,
        noise_level=0.15,
    )
    roi_data[6] = (times, values)

    metadata = {
        "duration_hours": duration,
        "sampling_interval_seconds": sampling_interval,
        "description": "Circadian rhythms (12-36h range)",
        "expected_periods": "24.0h, 23.5h, 24.0h (with 12h harmonic), 24.0h, 20.0h, 24.0h",
    }

    output_path = Path("test_data_circadian.h5")
    save_to_hdf5(str(output_path), roi_data, metadata)
    print(f"  Saved: {output_path}")
    print(f"  ROIs: {len(roi_data)}")
    print("  Expected periods: 24.0h, 23.5h, 24.0h+12.0h, 24.0h, 20.0h, 24.0h")
    print()


def create_test_dataset_3_ultradian():
    """
    Test Dataset 3: Ultradian rhythms (6-18 hours)
    Simulates semi-daily patterns.
    Duration: 48 hours (2 days)
    """
    print("Creating Test Dataset 3: Ultradian Rhythms (48 hours)")

    duration = 48  # hours
    sampling_interval = 30.0  # 30 seconds

    roi_data = {}

    # ROI 1: 12h rhythm (twice daily)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=12.0,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.06,
    )
    roi_data[1] = (times, values)

    # ROI 2: 8h rhythm (three times daily)
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=8.0,
        amplitude=0.35,
        baseline=0.5,
        noise_level=0.07,
    )
    roi_data[2] = (times, values)

    # ROI 3: 16h rhythm
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=16.0,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.06,
    )
    roi_data[3] = (times, values)

    # ROI 4: 12h rhythm with 6h harmonic
    times, values = generate_complex_signal(
        duration,
        sampling_interval,
        periods_hours=[12.0, 6.0],
        amplitudes=[0.35, 0.2],
        baseline=0.5,
        noise_level=0.08,
    )
    roi_data[4] = (times, values)

    # ROI 5: 10h rhythm
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=10.0,
        amplitude=0.4,
        baseline=0.5,
        noise_level=0.06,
    )
    roi_data[5] = (times, values)

    # ROI 6: 14h rhythm
    times, values = generate_rhythmic_signal(
        duration,
        sampling_interval,
        period_hours=14.0,
        amplitude=0.35,
        baseline=0.5,
        noise_level=0.07,
    )
    roi_data[6] = (times, values)

    metadata = {
        "duration_hours": duration,
        "sampling_interval_seconds": sampling_interval,
        "description": "Ultradian rhythms (6-18h range)",
        "expected_periods": "12.0h, 8.0h, 16.0h, 12.0h (with 6h harmonic), 10.0h, 14.0h",
    }

    output_path = Path("test_data_ultradian.h5")
    save_to_hdf5(str(output_path), roi_data, metadata)
    print(f"  Saved: {output_path}")
    print(f"  ROIs: {len(roi_data)}")
    print("  Expected periods: 12.0h, 8.0h, 16.0h, 12.0h+6.0h, 10.0h, 14.0h")
    print()


def create_test_dataset_4_mixed():
    """
    Test Dataset 4: Mixed patterns for testing similarity/coherence
    Different ROIs with various relationships.
    Duration: 72 hours (3 cycles for statistical significance)
    """
    print("Creating Test Dataset 4: Mixed Patterns (72 hours)")

    duration = 72  # hours - 3 full cycles for better statistics
    sampling_interval = 30.0  # 30 seconds

    roi_data = {}

    # Group A: ROIs 1-2 with synchronized 24h rhythm (phase 0)
    for i in range(1, 3):
        times, values = generate_rhythmic_signal(
            duration,
            sampling_interval,
            period_hours=24.0,
            amplitude=0.4,
            baseline=0.5,
            noise_level=0.05,
            phase_offset=0.0,  # Same phase
        )
        roi_data[i] = (times, values)

    # Group B: ROIs 3-4 with synchronized 24h rhythm (phase π - anti-phase to Group A)
    for i in range(3, 5):
        times, values = generate_rhythmic_signal(
            duration,
            sampling_interval,
            period_hours=24.0,
            amplitude=0.4,
            baseline=0.5,
            noise_level=0.05,
            phase_offset=np.pi,  # Anti-phase to Group A
        )
        roi_data[i] = (times, values)

    # Group C: ROIs 5-6 with synchronized 20h rhythm (different period)
    for i in range(5, 7):
        times, values = generate_rhythmic_signal(
            duration,
            sampling_interval,
            period_hours=20.0,
            amplitude=0.35,
            baseline=0.5,
            noise_level=0.05,
            phase_offset=0.0,  # Same phase within group
        )
        roi_data[i] = (times, values)

    metadata = {
        "duration_hours": duration,
        "sampling_interval_seconds": sampling_interval,
        "description": "Mixed circadian patterns: Group A (ROIs 1-2: 24h, phase 0), Group B (ROIs 3-4: 24h, phase π), Group C (ROIs 5-6: 20h)",
        "expected_periods": "24.0h, 24.0h, 24.0h, 24.0h, 20.0h, 20.0h",
        "expected_similarity": "High within groups, moderate between A-B (same period, different phase), low between A/B-C (different period)",
    }

    output_path = Path("test_data_mixed.h5")
    save_to_hdf5(str(output_path), roi_data, metadata)
    print(f"  Saved: {output_path}")
    print(f"  ROIs: {len(roi_data)}")
    print("  Group A (1-2): 24h rhythm (phase 0°)")
    print("  Group B (3-4): 24h rhythm (phase 180°)")
    print("  Group C (5-6): 20h rhythm")
    print("  Duration: 72 hours (3 cycles)")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Test Datasets for Rhythmic Pattern Analysis")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create all test datasets
    create_test_dataset_1_short_cycles()
    create_test_dataset_2_circadian()
    create_test_dataset_3_ultradian()
    create_test_dataset_4_mixed()

    print("=" * 60)
    print("All test datasets created successfully!")
    print("=" * 60)
    print()
    print("Usage:")
    print("  1. Load the HDF5 files in napari-hdf5-activity plugin")
    print("  2. Run Fisher Z-Transformation analysis")
    print("  3. Run FFT Power Spectrum analysis")
    print("  4. Compare results with expected periods")
    print()
    print("Expected Results:")
    print("  - Fisher and FFT should find the same dominant periods")
    print("  - Similarity analysis should group ROIs with same periods")
    print("  - Coherence should be high for synchronized ROIs")
    print("  - Phase clustering should separate different groups")
