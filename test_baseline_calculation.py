"""
Test to verify baseline is calculated BEFORE detrending.
"""

import numpy as np
from src.napari_hdf5_activity._calc import run_baseline_analysis


# Generate synthetic data with strong linear trend
def generate_data_with_trend():
    """Generate data with strong upward trend."""
    np.random.seed(42)

    # Generate time points (0 to 5000 seconds, 5 sec intervals = 1000 frames)
    times = np.arange(0, 5000, 5)

    # Strong linear trend: starts at 100, increases to 200
    trend = np.linspace(100, 200, len(times))

    # Small random noise
    noise = np.random.normal(0, 2, len(times))

    # Combined signal
    values = trend + noise

    return list(zip(times, values))


# Create test data
print("=" * 60)
print("Testing Baseline Calculation Order")
print("=" * 60)
print()

test_data = {0: generate_data_with_trend()}

# Test 1: WITHOUT detrending
print("Test 1: Analysis WITHOUT detrending")
print("-" * 40)
results_no_detrend = run_baseline_analysis(
    test_data,
    enable_matlab_norm=False,
    enable_detrending=False,
    enable_jump_correction=False,
    baseline_duration_minutes=10,  # First 10 minutes (600 seconds)
    multiplier=2.0,
    frame_interval=5.0,
    num_processes=1,
)

baseline_no_detrend = results_no_detrend["baseline_means"][0]
upper_no_detrend = results_no_detrend["upper_thresholds"][0]
lower_no_detrend = results_no_detrend["lower_thresholds"][0]

print(f"Baseline Mean:     {baseline_no_detrend:.2f}")
print(f"Upper Threshold:   {upper_no_detrend:.2f}")
print(f"Lower Threshold:   {lower_no_detrend:.2f}")
print()

# Test 2: WITH detrending
print("Test 2: Analysis WITH detrending")
print("-" * 40)
results_with_detrend = run_baseline_analysis(
    test_data,
    enable_matlab_norm=False,
    enable_detrending=True,
    use_improved_detrending=True,
    enable_jump_correction=False,
    baseline_duration_minutes=10,  # First 10 minutes (600 seconds)
    multiplier=2.0,
    frame_interval=5.0,
    num_processes=1,
)

baseline_with_detrend = results_with_detrend["baseline_means"][0]
upper_with_detrend = results_with_detrend["upper_thresholds"][0]
lower_with_detrend = results_with_detrend["lower_thresholds"][0]

print(f"Baseline Mean:     {baseline_with_detrend:.2f}")
print(f"Upper Threshold:   {upper_with_detrend:.2f}")
print(f"Lower Threshold:   {lower_with_detrend:.2f}")
print()

# Comparison
print("=" * 60)
print("Verification Results")
print("=" * 60)
print()

# Expected: Baselines should be IDENTICAL (or very close)
# because baseline is now calculated BEFORE detrending
baseline_diff = abs(baseline_no_detrend - baseline_with_detrend)
upper_diff = abs(upper_no_detrend - upper_with_detrend)
lower_diff = abs(lower_no_detrend - lower_with_detrend)

print(f"Baseline difference:  {baseline_diff:.6f}")
print(f"Upper threshold diff: {upper_diff:.6f}")
print(f"Lower threshold diff: {lower_diff:.6f}")
print()

# Threshold for "identical" (allow small floating point errors)
tolerance = 0.001

if baseline_diff < tolerance and upper_diff < tolerance and lower_diff < tolerance:
    print("✓ SUCCESS: Baseline is calculated BEFORE detrending!")
    print("  → Baselines are identical regardless of detrending setting")
else:
    print("✗ FAILURE: Baseline appears to be affected by detrending")
    print(
        f"  → Expected difference < {tolerance}, got {max(baseline_diff, upper_diff, lower_diff):.6f}"
    )
print()

# Additional info
print("Expected behavior:")
print("  - Baseline should reflect the ORIGINAL first 10 minutes")
print("  - In this test, first 10 minutes (600 sec) have values ~100-112")
print(f"  - Actual baseline: {baseline_with_detrend:.2f}")
print()

# Check if baseline is in expected range
expected_range_min = 100
expected_range_max = 112
if expected_range_min <= baseline_with_detrend <= expected_range_max:
    print("✓ Baseline is in expected range for first 10 minutes")
else:
    print(
        f"✗ Baseline {baseline_with_detrend:.2f} is outside expected range [{expected_range_min}, {expected_range_max}]"
    )
