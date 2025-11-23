"""
_calc_calibration.py - Calibration-based threshold calculation with hysteresis

This module implements calibration-based threshold calculation using sedated animal recordings.

KEY CONCEPT:
- Sedated animal dataset = True biological "zero movement" baseline
- Identical ROI processing pipeline applied to both datasets
- COMPLETE sedated dataset used as baseline (no time selection)
- Hysteresis thresholds derived from sedated animal statistics
- Same hysteresis movement detection applied to main dataset

SCIENTIFIC ADVANTAGE:
- Higher specificity than arbitrary statistical baselines
- True biological reference for "no movement" state
- Consistent baseline across different experimental conditions
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ._calc import (
    apply_matlab_normalization_to_merged_results,
    improved_full_dataset_detrending,
    normalize_and_detrend_merged_results,
    define_movement_with_hysteresis,
    bin_fraction_movement,
    bin_quiescence,
    define_sleep_periods,
    validate_frame_difference_data,
)


def run_calibration_analysis_with_precomputed_baseline(
    merged_results: Dict[int, List[Tuple[float, float]]],
    calibration_baseline_statistics: Dict[int, Dict[str, Any]],
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
    use_improved_detrending: bool = True,
    frame_interval: float = 5.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run calibration analysis using pre-computed baseline statistics.
    """

    print("🚀 RUNNING CALIBRATION ANALYSIS WITH PRE-COMPUTED BASELINE")
    print("=" * 70)

    # VERIFY INPUT DATASET
    main_dataset_info = {}
    for roi, data in merged_results.items():
        if data:
            duration_seconds = data[-1][0] - data[0][0]
            duration_minutes = duration_seconds / 60
            main_dataset_info[roi] = {
                "points": len(data),
                "duration_minutes": duration_minutes,
                "start_time": data[0][0],
                "end_time": data[-1][0],
            }
            break

    if main_dataset_info:
        roi_id, info = next(iter(main_dataset_info.items()))
        print(f"📊 MAIN DATASET INFO:")
        print(f"   ROI {roi_id}: {info['points']} data points")
        print(f"   Duration: {info['duration_minutes']:.1f} minutes")

        if info["duration_minutes"] < 60:
            print(f"   ⚠️  WARNING: Duration seems short for main experiment data")
        else:
            print(f"   ✅ Duration appropriate for main experiment data")

    if not calibration_baseline_statistics:
        raise ValueError("No calibration baseline statistics provided")

    analysis_results = {
        "method": "calibration_precomputed",
        "parameters": {
            "enable_matlab_norm": enable_matlab_norm,
            "enable_detrending": enable_detrending,
            "use_improved_detrending": use_improved_detrending,
            "frame_interval": frame_interval,
            "uses_precomputed_calibration_baseline": True,
            "calibration_rois_processed": len(calibration_baseline_statistics),
        },
    }

    # Step 1: Preprocess main dataset
    print(f"\n📊 STEP 1: PREPROCESSING MAIN DATASET")

    if enable_matlab_norm:
        print(f"   Applying MATLAB normalization...")
        normalized_data = apply_matlab_normalization_to_merged_results(
            merged_results, enable_matlab_norm=True
        )
    else:
        normalized_data = merged_results

    if enable_detrending:
        print(f"   Applying detrending...")
        if use_improved_detrending:
            processed_data = improved_full_dataset_detrending(normalized_data)
        else:
            processed_data = normalize_and_detrend_merged_results(normalized_data)
    else:
        processed_data = normalized_data

    analysis_results["processed_data"] = processed_data

    # Step 2: Apply calibration thresholds
    print(f"\n📊 STEP 2: APPLYING CALIBRATION THRESHOLDS")

    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}

    successful_matches = 0
    missing_calibration = 0

    for roi in processed_data.keys():
        if roi in calibration_baseline_statistics:
            cal_stats = calibration_baseline_statistics[roi]

            baseline_means[roi] = cal_stats["baseline_mean"]
            upper_thresholds[roi] = cal_stats["upper_threshold"]
            lower_thresholds[roi] = cal_stats["lower_threshold"]

            roi_statistics[roi] = {
                "method": "calibration_precomputed",
                "baseline_mean": cal_stats["baseline_mean"],
                "baseline_std": cal_stats.get("baseline_std", 0.0),
                "upper_threshold": cal_stats["upper_threshold"],
                "lower_threshold": cal_stats["lower_threshold"],
                "threshold_band": cal_stats.get("threshold_band", 0.0),
                "multiplier": cal_stats.get("multiplier", 1.0),
                "uses_precomputed_baseline": True,
                "status": "success",
            }

            successful_matches += 1
            print(
                f"   ✅ ROI {roi}: Applied calibration baseline (mean={cal_stats['baseline_mean']:.1f})"
            )

        else:
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 1.0
            lower_thresholds[roi] = 0.0

            roi_statistics[roi] = {
                "method": "calibration_precomputed",
                "status": "no_calibration_data_for_roi",
                "baseline_mean": 0.0,
                "upper_threshold": 1.0,
                "lower_threshold": 0.0,
            }

            missing_calibration += 1
            print(f"   ⚠️ ROI {roi}: No calibration data - using fallback")

    print(f"   Success rate: {(successful_matches/len(processed_data)*100):.1f}%")

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
        }
    )

    # Step 3: Apply hysteresis movement detection
    print(f"\n📊 STEP 3: MOVEMENT DETECTION WITH CALIBRATION THRESHOLDS")

    movement_data = define_movement_with_hysteresis(
        processed_data, baseline_means, upper_thresholds, lower_thresholds
    )
    analysis_results["movement_data"] = movement_data

    # Step 4: Behavioral analysis
    print(f"\n📊 STEP 4: BEHAVIORAL ANALYSIS")

    successful_movement_data = {
        roi: data
        for roi, data in movement_data.items()
        if roi in roi_statistics and roi_statistics[roi]["status"] == "success"
    }

    if successful_movement_data:
        bin_size_seconds = kwargs.get("bin_size_seconds", 60)
        fraction_data = bin_fraction_movement(
            successful_movement_data, bin_size_seconds, frame_interval
        )
        analysis_results["fraction_data"] = fraction_data

        quiescence_threshold = kwargs.get("quiescence_threshold", 0.5)
        quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)
        analysis_results["quiescence_data"] = quiescence_data

        sleep_threshold_minutes = kwargs.get("sleep_threshold_minutes", 8)
        sleep_data = define_sleep_periods(
            quiescence_data, sleep_threshold_minutes, bin_size_seconds
        )
        analysis_results["sleep_data"] = sleep_data

        print(
            f"   ✅ Behavioral analysis complete for {len(successful_movement_data)} ROIs"
        )
    else:
        analysis_results["fraction_data"] = {}
        analysis_results["quiescence_data"] = {}
        analysis_results["sleep_data"] = {}

    # Add ROI colors
    try:
        from ._reader import get_roi_colors

        roi_colors = get_roi_colors(sorted(processed_data.keys()))
    except:
        roi_colors = {
            roi: f"C{i}" for i, roi in enumerate(sorted(processed_data.keys()))
        }

    analysis_results["roi_colors"] = roi_colors

    # Summary
    movement_summary = {}
    for roi, movements in movement_data.items():
        if (
            movements
            and roi in roi_statistics
            and roi_statistics[roi]["status"] == "success"
        ):
            movement_count = sum(1 for _, m in movements if m == 1)
            total_count = len(movements)
            movement_summary[roi] = (
                (movement_count / total_count * 100) if total_count > 0 else 0
            )

    avg_movement = np.mean(list(movement_summary.values())) if movement_summary else 0

    analysis_results["summary"] = {
        "total_rois": len(processed_data),
        "successful_calibration_matches": successful_matches,
        "missing_calibration_data": missing_calibration,
        "success_rate_percent": (
            (successful_matches / len(processed_data) * 100) if processed_data else 0
        ),
        "average_movement_percentage": avg_movement,
        "method_description": "Pre-computed calibration baseline applied to main experimental dataset",
    }

    print(f"\n✅ CALIBRATION ANALYSIS COMPLETE")
    print(f"   Success rate: {(successful_matches/len(processed_data)*100):.1f}%")
    print(f"   Average movement: {avg_movement:.1f}%")

    return analysis_results


def _extract_analysis_parameters(self) -> Dict[str, Any]:
    """Enhanced parameter extraction that handles calibration method specifics."""
    # Determine threshold method
    method_text = self.threshold_method.currentText()
    if "Baseline" in method_text:
        threshold_method = "baseline"
    elif "Calibration" in method_text:
        threshold_method = "calibration"
    elif "Adaptive" in method_text:
        threshold_method = "adaptive"
    else:
        threshold_method = "baseline"

    # Basic parameters
    params = {
        "threshold_method": threshold_method,
        "enable_matlab_norm": True,
        "enable_detrending": self.enable_detrending.isChecked(),
        "use_improved_detrending": True,
        "frame_interval": self.frame_interval.value(),
        "apply_hdf5_timing_correction_flag": True,
        "bin_size_seconds": self.bin_size_seconds.value(),
        "quiescence_threshold": self.quiescence_threshold.value(),
        "sleep_threshold_minutes": self.sleep_threshold_minutes.value(),
    }

    # Add method-specific parameters
    if threshold_method == "baseline":
        params.update(
            {
                "baseline_duration_minutes": self.baseline_duration_minutes.value(),
                "multiplier": self.threshold_multiplier.value(),
                "enable_jump_correction": self.enable_jump_correction.isChecked(),
            }
        )
    elif threshold_method == "calibration":
        # For the new pre-computed workflow, we don't need file path/masks
        # because they're already processed and stored in calibration_baseline_statistics
        params.update(
            {
                "calibration_multiplier": self.calibration_multiplier.value(),
                # Note: calibration_file_path and masks are handled separately
                # in the new workflow since baseline is pre-computed
            }
        )

        # Only add these for legacy workflow
        if not (
            hasattr(self, "calibration_baseline_processed")
            and self.calibration_baseline_processed
        ):
            cal_file_path = self.calibration_file_path.property("full_path")
            params.update(
                {
                    "calibration_file_path": cal_file_path,
                    "masks": self.masks,
                }
            )
    elif threshold_method == "adaptive":
        params.update(
            {
                "adaptive_duration_minutes": self.adaptive_duration_minutes.value(),
                "adaptive_multiplier": self.adaptive_base_multiplier.value(),
            }
        )

    return params


def update_calibration_workflow_status(
    self, step: str, success: bool = True, message: str = ""
):
    """Update the calibration workflow status display."""
    if not hasattr(self, "calibration_status_label"):
        return

    steps = {
        "file_selected": (
            "✅ 1. Calibration file selected"
            if success
            else "❌ 1. Select calibration file"
        ),
        "dataset_loaded": (
            "✅ 2. Calibration dataset loaded"
            if success
            else "2. Load calibration dataset"
        ),
        "rois_detected": (
            "✅ 3. Calibration ROIs detected"
            if success
            else "3. Detect ROIs (Input tab)"
        ),
        "baseline_processed": (
            "✅ 4. Calibration baseline processed" if success else "4. Process baseline"
        ),
    }

    current_status = []
    for step_key, step_text in steps.items():
        if step_key == step:
            current_status.append(step_text)
            if success and step != "baseline_processed":
                # Add next step
                next_steps = list(steps.keys())
                current_idx = next_steps.index(step_key)
                if current_idx + 1 < len(next_steps):
                    next_step_key = next_steps[current_idx + 1]
                    current_status.append(steps[next_step_key])
            break
        else:
            current_status.append(step_text)

    if message:
        current_status.append(f"\n{message}")

    self.calibration_status_label.setText("\n".join(current_status))


def process_calibration_file(
    calibration_file_path: str,
    masks: List[np.ndarray],
    frame_interval: float = 5.0,
    chunk_size: int = 50,
    progress_callback: Optional[callable] = None,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Process calibration HDF5 file with IDENTICAL pipeline as main dataset.

    This ensures the calibration baseline is calculated using exactly the same
    processing steps as the experimental data being analyzed.

    Args:
        calibration_file_path: Path to calibration HDF5 file (sedated animals)
        masks: List of ROI masks (IDENTICAL to main dataset)
        frame_interval: Time between frames
        chunk_size: Processing chunk size
        progress_callback: Optional progress callback function

    Returns:
        Dictionary mapping ROI ID to list of (time, intensity_change) tuples
    """
    print(f"\n🔬 PROCESSING CALIBRATION FILE (SEDATED ANIMALS)")
    print(f"File: {os.path.basename(calibration_file_path)}")
    print(f"Using IDENTICAL processing pipeline as main dataset")

    if not os.path.exists(calibration_file_path):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file_path}")

    try:
        # Import the same processing function used for main data
        from ._reader import process_hdf5_file

        # Process calibration file with IDENTICAL parameters
        _, calibration_roi_changes, calibration_duration = process_hdf5_file(
            file_path=calibration_file_path,
            masks=masks,  # SAME masks as main dataset
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            frame_interval=frame_interval,
        )

        # Log calibration file processing results
        print(f"✅ Calibration file processed successfully:")
        print(f"   ROIs detected: {len(calibration_roi_changes)}")
        print(f"   Duration: {calibration_duration / 60:.1f} minutes")

        # Log sample data for first few ROIs
        for roi in sorted(calibration_roi_changes.keys())[:3]:
            data_points = len(calibration_roi_changes[roi])
            if data_points > 0:
                first_time = calibration_roi_changes[roi][0][0]
                last_time = calibration_roi_changes[roi][-1][0]
                duration_min = (last_time - first_time) / 60
                sample_values = [val for _, val in calibration_roi_changes[roi][:5]]

                print(f"   ROI {roi}: {data_points} points, {duration_min:.1f}min")
                print(f"     Sample values: {[f'{v:.1f}' for v in sample_values]}")

        print(f"✅ Calibration baseline ready for threshold calculation")
        return calibration_roi_changes

    except Exception as e:
        print(f"❌ Error processing calibration file: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to process calibration file: {e}")


def compute_threshold_calibration_hysteresis(
    target_data: List[Tuple[float, float]],
    calibration_file_path: str,
    masks: List[np.ndarray],
    roi_index: int,
    calibration_multiplier: float = 1.0,
    frame_interval: float = 5.0,
    enable_matlab_norm: bool = True,
    enable_detrending: bool = True,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute calibration-based hysteresis thresholds using sedated animal baseline.

    PROCESS:
    1. Load calibration file (sedated animals) with identical ROI processing
    2. Apply SAME preprocessing pipeline (MATLAB norm + detrending)
    3. Use COMPLETE calibration dataset as baseline statistics
    4. Calculate hysteresis thresholds from sedated animal baseline
    5. Return thresholds for application to main dataset

    Args:
        target_data: Main experimental data (for quality comparison only)
        calibration_file_path: Path to sedated animal HDF5 file
        masks: ROI masks (identical to main dataset)
        roi_index: ROI index (0-based)
        calibration_multiplier: Multiplier for threshold band calculation
        frame_interval: Time between frames
        enable_matlab_norm: Apply MATLAB normalization to calibration data
        enable_detrending: Apply detrending to calibration data

    Returns:
        Tuple of (baseline_mean, upper_threshold, lower_threshold, statistics_dict)
    """
    if not target_data:
        return (
            0.0,
            0.0,
            0.0,
            {"method": "calibration_hysteresis", "status": "no_target_data"},
        )

    print(f"\n=== CALIBRATION-BASED HYSTERESIS THRESHOLD CALCULATION ===")
    print(f"🎯 Target ROI: {roi_index + 1}")
    print(f"🔬 Calibration source: Sedated animals (complete dataset)")
    print(f"⚖️  Baseline strategy: Entire calibration dataset = noise floor")

    try:
        # Step 1: Process calibration file with identical pipeline
        print(f"\n📊 Step 1: Loading calibration data...")
        calibration_raw_data = process_calibration_file(
            calibration_file_path, masks, frame_interval
        )

        # Get ROI ID from index (assuming 1-based ROI IDs)
        roi_id = roi_index + 1

        # Validate calibration data availability
        if roi_id not in calibration_raw_data:
            available_rois = list(calibration_raw_data.keys())
            return (
                0.0,
                0.0,
                0.0,
                {
                    "method": "calibration_hysteresis",
                    "status": f"roi_{roi_id}_missing_in_calibration",
                    "available_rois": available_rois,
                },
            )

        calibration_roi_raw = calibration_raw_data[roi_id]

        if not calibration_roi_raw:
            return (
                0.0,
                0.0,
                0.0,
                {
                    "method": "calibration_hysteresis",
                    "status": f"roi_{roi_id}_empty_calibration_data",
                },
            )

        print(f"✅ Calibration data loaded for ROI {roi_id}:")
        print(f"   Raw data points: {len(calibration_roi_raw)}")

        # Step 2: Apply IDENTICAL preprocessing as main dataset
        print(f"\n📊 Step 2: Applying IDENTICAL preprocessing to calibration...")
        print(f"   Same pipeline as main dataset ensures fair comparison")

        # Create single-ROI dict for preprocessing compatibility
        calibration_single_roi = {roi_id: calibration_roi_raw}

        # Apply MATLAB normalization (if enabled for main dataset)
        if enable_matlab_norm:
            calibration_normalized = apply_matlab_normalization_to_merged_results(
                calibration_single_roi, enable_matlab_norm=True
            )
            print(f"   ✅ Applied MATLAB normalization (min-subtraction)")
        else:
            calibration_normalized = calibration_single_roi
            print(f"   ⚪ Skipped MATLAB normalization")

        # Apply detrending (if enabled for main dataset)
        if enable_detrending:
            calibration_processed = improved_full_dataset_detrending(
                calibration_normalized
            )
            print(f"   ✅ Applied improved full dataset detrending")
        else:
            calibration_processed = calibration_normalized
            print(f"   ⚪ Skipped detrending")

        # Extract final processed calibration data
        calibration_final_data = calibration_processed[roi_id]

        if not calibration_final_data:
            return (
                0.0,
                0.0,
                0.0,
                {
                    "method": "calibration_hysteresis",
                    "status": f"roi_{roi_id}_empty_after_preprocessing",
                },
            )

        print(f"   ✅ Preprocessing complete: {len(calibration_final_data)} points")

        # Step 3: Calculate baseline statistics from COMPLETE sedated dataset
        print(f"\n📊 Step 3: Calculating baseline from COMPLETE sedated dataset...")
        print(
            f"   🔑 KEY: Using entire sedated dataset as 'true zero movement' baseline"
        )

        # Extract all calibration values (no time-based selection!)
        calibration_values = np.array([val for _, val in calibration_final_data])

        # Calculate comprehensive baseline statistics
        calibration_mean = np.mean(calibration_values)
        calibration_std = np.std(calibration_values)
        calibration_median = np.median(calibration_values)
        calibration_min = np.min(calibration_values)
        calibration_max = np.max(calibration_values)
        calibration_q25 = np.percentile(calibration_values, 25)
        calibration_q75 = np.percentile(calibration_values, 75)

        # Calculate calibration duration
        calibration_start_time = calibration_final_data[0][0]
        calibration_end_time = calibration_final_data[-1][0]
        calibration_duration_minutes = (
            calibration_end_time - calibration_start_time
        ) / 60

        print(f"   📈 Sedated animal baseline statistics:")
        print(f"     Data points: {len(calibration_values):,}")
        print(f"     Duration: {calibration_duration_minutes:.1f} minutes")
        print(f"     Mean: {calibration_mean:.1f}")
        print(f"     Std: {calibration_std:.1f}")
        print(f"     Median: {calibration_median:.1f}")
        print(f"     Range: {calibration_min:.1f} to {calibration_max:.1f}")
        print(f"     IQR: {calibration_q25:.1f} to {calibration_q75:.1f}")

        # Step 4: Calculate hysteresis thresholds using sedated animal baseline
        print(f"\n📊 Step 4: Computing hysteresis thresholds from sedated baseline...")

        # Use sedated animal statistics as the baseline reference
        baseline_mean = calibration_mean  # From sedated animals
        baseline_std = calibration_std  # From sedated animals

        # Calculate hysteresis band
        threshold_band = calibration_multiplier * baseline_std
        upper_threshold = baseline_mean + threshold_band
        lower_threshold = baseline_mean - threshold_band

        print(f"   🎯 Hysteresis threshold calculation:")
        print(f"     Baseline mean: {baseline_mean:.1f} (from sedated animals)")
        print(f"     Baseline std: {baseline_std:.1f} (from sedated animals)")
        print(f"     Multiplier: {calibration_multiplier}")
        print(f"     Band width: ±{threshold_band:.1f}")
        print(f"     Upper threshold: {upper_threshold:.1f} (Movement = TRUE)")
        print(f"     Lower threshold: {lower_threshold:.1f} (Movement = FALSE)")

        # Step 5: Threshold validation and safety checks
        if lower_threshold < 0:
            lower_threshold = 0
            print(f"     ⚠️ Adjusted negative lower threshold to 0")

        if np.isnan(upper_threshold) or np.isinf(upper_threshold):
            print(f"     ❌ Invalid upper threshold - applying percentile fallback")
            upper_threshold = calibration_q75
            lower_threshold = calibration_q25
            baseline_mean = calibration_median
            print(
                f"     📊 Fallback thresholds: {lower_threshold:.1f} to {upper_threshold:.1f}"
            )

        # Step 6: Quality assessment vs target data
        print(f"\n📊 Step 5: Quality assessment vs target data...")

        # Sample target data for comparison (first 1000 points)
        target_sample = target_data[: min(1000, len(target_data))]
        target_values = np.array([val for _, val in target_sample])
        target_mean = np.mean(target_values)
        target_std = np.std(target_values)
        target_median = np.median(target_values)

        # Calculate separation metrics
        signal_to_noise = (
            (target_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        )
        target_calibration_ratio = (
            target_mean / baseline_mean if baseline_mean > 0 else 1
        )
        separation_factor = (
            abs(target_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        )

        print(f"   📊 Target vs Calibration comparison:")
        print(f"     Target mean: {target_mean:.1f}")
        print(f"     Calibration mean: {baseline_mean:.1f}")
        print(f"     Signal-to-noise ratio: {signal_to_noise:.2f}")
        print(f"     Target/Calibration ratio: {target_calibration_ratio:.2f}")
        print(f"     Separation factor: {separation_factor:.2f}σ")

        # Determine calibration quality
        if signal_to_noise > 3.0 and separation_factor > 3.0:
            calibration_quality = "excellent"
            quality_desc = "Excellent separation - high confidence thresholds"
        elif signal_to_noise > 2.0 and separation_factor > 2.0:
            calibration_quality = "good"
            quality_desc = "Good separation - reliable thresholds"
        elif signal_to_noise > 1.0 and separation_factor > 1.0:
            calibration_quality = "fair"
            quality_desc = "Fair separation - moderate confidence"
        else:
            calibration_quality = "poor"
            quality_desc = (
                "Poor separation - low confidence, consider alternative method"
            )

        print(f"   🏆 Calibration quality: {calibration_quality.upper()}")
        print(f"     Assessment: {quality_desc}")

        # Step 7: Compile comprehensive statistics
        statistics = {
            # Core results
            "method": "calibration_hysteresis",
            "baseline_mean": baseline_mean,
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "threshold_band": threshold_band,
            # Compatibility fields
            "mean": baseline_mean,
            "std": baseline_std,
            "multiplier": calibration_multiplier,
            # Calibration dataset info
            "calibration_frames": len(calibration_values),
            "calibration_duration_minutes": calibration_duration_minutes,
            "calibration_file": os.path.basename(calibration_file_path),
            "uses_complete_calibration_dataset": True,
            "uses_sedated_animal_baseline": True,
            # Calibration statistics
            "calibration_mean": calibration_mean,
            "calibration_std": calibration_std,
            "calibration_median": calibration_median,
            "calibration_range": (float(calibration_min), float(calibration_max)),
            "calibration_iqr": (float(calibration_q25), float(calibration_q75)),
            # Target comparison
            "target_frames_sampled": len(target_values),
            "target_mean": target_mean,
            "target_std": target_std,
            "target_median": target_median,
            "signal_to_noise_ratio": signal_to_noise,
            "target_calibration_ratio": target_calibration_ratio,
            "separation_factor": separation_factor,
            # Quality assessment
            "calibration_quality": calibration_quality,
            "quality_description": quality_desc,
            # Processing info
            "data_preprocessing": {
                "matlab_norm_applied": enable_matlab_norm,
                "detrending_applied": enable_detrending,
                "identical_to_main_dataset": True,
            },
            # Status
            "status": "success",
        }

        print(f"\n✅ CALIBRATION-BASED HYSTERESIS THRESHOLDS COMPLETE")
        print(f"   Baseline: {baseline_mean:.1f} ± {threshold_band:.1f}")
        print(f"   Range: {lower_threshold:.1f} ≤ hysteresis ≤ {upper_threshold:.1f}")
        print(f"   Quality: {calibration_quality}")

        return baseline_mean, upper_threshold, lower_threshold, statistics

    except Exception as e:
        print(f"❌ Error in calibration threshold calculation: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return (
            0.0,
            0.0,
            0.0,
            {"method": "calibration_hysteresis", "status": f"error: {e}"},
        )


def run_calibration_analysis(
    merged_results: Dict[int, List[Tuple[float, float]]],
    calibration_file_path: str = None,
    masks: List[np.ndarray] = None,
    calibration_baseline_statistics: Dict[int, Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    UPDATED: Main calibration analysis function that routes to appropriate workflow.

    This function now supports both the old workflow (on-the-fly calibration processing)
    and the new workflow (pre-computed calibration baseline).

    Args:
        merged_results: Main experimental data from Reader
        calibration_file_path: Path to calibration file (old workflow)
        masks: ROI masks (old workflow)
        calibration_baseline_statistics: Pre-computed baseline stats (new workflow)
        **kwargs: Additional analysis parameters

    Returns:
        Complete analysis results
    """
    if calibration_baseline_statistics:
        # NEW WORKFLOW: Use pre-computed calibration baseline
        print("📊 Using PRE-COMPUTED calibration baseline workflow")
        return run_calibration_analysis_with_precomputed_baseline(
            merged_results=merged_results,
            calibration_baseline_statistics=calibration_baseline_statistics,
            **kwargs,
        )

    elif calibration_file_path and masks:
        # OLD WORKFLOW: Process calibration on-the-fly
        print("⚠️ Using LEGACY on-the-fly calibration workflow")
        print("   Consider upgrading to pre-computed workflow for better transparency")

        # Call the original function (kept for backwards compatibility)
        return run_calibration_analysis_legacy(
            merged_results=merged_results,
            calibration_file_path=calibration_file_path,
            masks=masks,
            **kwargs,
        )

    else:
        raise ValueError(
            "Either calibration_baseline_statistics (new workflow) or "
            "calibration_file_path + masks (old workflow) must be provided"
        )


def integrate_calibration_analysis_with_widget(widget) -> bool:
    """
    Integration function for calibration analysis with napari widget.

    This function extracts parameters from the widget UI, runs the complete
    calibration analysis pipeline, and updates all widget attributes with results.

    Args:
        widget: The napari widget instance containing UI parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate widget state
        if not hasattr(widget, "merged_results") or not widget.merged_results:
            widget._log_message(
                "❌ No merged_results available for calibration analysis"
            )
            return False

        # Get calibration file path from widget
        calibration_file_path = widget.calibration_file_path.property("full_path")
        if not calibration_file_path:
            widget._log_message("❌ No calibration file selected")
            return False

        if not os.path.exists(calibration_file_path):
            widget._log_message(
                f"❌ Calibration file not found: {calibration_file_path}"
            )
            return False

        # Validate ROI masks
        if not hasattr(widget, "masks") or not widget.masks:
            widget._log_message("❌ No ROI masks available - run ROI detection first")
            return False

        widget._log_message("🚀 STARTING CALIBRATION ANALYSIS INTEGRATION")
        widget._log_message(
            f"📁 Calibration file: {os.path.basename(calibration_file_path)}"
        )
        widget._log_message(f"🎯 ROIs to analyze: {len(widget.merged_results)}")
        widget._log_message(f"🔍 ROI masks: {len(widget.masks)}")

        # Extract analysis parameters from widget UI
        try:
            frame_interval = widget.frame_interval.value()
            calibration_multiplier = widget.calibration_multiplier.value()
            enable_detrending = widget.enable_detrending.isChecked()
            bin_size_seconds = widget.bin_size_seconds.value()
            quiescence_threshold = widget.quiescence_threshold.value()
            sleep_threshold_minutes = widget.sleep_threshold_minutes.value()

            widget._log_message(f"📊 Parameters extracted:")
            widget._log_message(f"   Frame interval: {frame_interval}s")
            widget._log_message(f"   Calibration multiplier: {calibration_multiplier}")
            widget._log_message(
                f"   Detrending: {'Enabled' if enable_detrending else 'Disabled'}"
            )

        except Exception as e:
            widget._log_message(f"❌ Error extracting parameters: {e}")
            return False

        # Run calibration analysis pipeline
        widget._log_message("🔬 Running calibration analysis pipeline...")

        try:
            calibration_results = run_calibration_analysis(
                merged_results=widget.merged_results,
                calibration_file_path=calibration_file_path,
                masks=widget.masks,
                enable_matlab_norm=True,  # Always use MATLAB norm for consistency
                enable_detrending=enable_detrending,
                use_improved_detrending=True,  # Use best detrending available
                calibration_multiplier=calibration_multiplier,
                frame_interval=frame_interval,
                bin_size_seconds=bin_size_seconds,
                quiescence_threshold=quiescence_threshold,
                sleep_threshold_minutes=sleep_threshold_minutes,
            )

            widget._log_message("✅ Calibration analysis completed successfully")

        except Exception as e:
            widget._log_message(f"❌ Calibration analysis failed: {e}")
            import traceback

            widget._log_message(f"Traceback: {traceback.format_exc()}")
            return False

        # Validate results
        if not calibration_results or "baseline_means" not in calibration_results:
            widget._log_message("❌ Calibration analysis returned invalid results")
            return False

        # Update widget with all calibration results
        widget._log_message("🔄 Updating widget with calibration results...")

        try:
            widget.merged_results = calibration_results.get(
                "processed_data", widget.merged_results
            )
            widget.roi_baseline_means = calibration_results.get("baseline_means", {})
            widget.roi_upper_thresholds = calibration_results.get(
                "upper_thresholds", {}
            )
            widget.roi_lower_thresholds = calibration_results.get(
                "lower_thresholds", {}
            )
            widget.roi_statistics = calibration_results.get("roi_statistics", {})
            widget.movement_data = calibration_results.get("movement_data", {})
            widget.fraction_data = calibration_results.get("fraction_data", {})
            widget.quiescence_data = calibration_results.get("quiescence_data", {})
            widget.sleep_data = calibration_results.get("sleep_data", {})

            # Set ROI colors if available
            if "roi_colors" in calibration_results:
                widget.roi_colors = calibration_results["roi_colors"]

            widget._log_message("✅ Widget attributes updated successfully")

        except Exception as e:
            widget._log_message(f"❌ Error updating widget: {e}")
            return False

        # Calculate band widths for plotting compatibility
        try:
            widget.roi_band_widths = {}
            for roi in widget.roi_baseline_means:
                if (
                    roi in widget.roi_upper_thresholds
                    and roi in widget.roi_lower_thresholds
                ):
                    upper = widget.roi_upper_thresholds[roi]
                    lower = widget.roi_lower_thresholds[roi]
                    widget.roi_band_widths[roi] = (upper - lower) / 2

            widget._log_message(
                f"✅ Calculated band widths for {len(widget.roi_band_widths)} ROIs"
            )

        except Exception as e:
            widget._log_message(f"⚠️ Could not calculate band widths: {e}")
            widget.roi_band_widths = {}

        # Log calibration analysis summary
        summary = calibration_results.get("summary", {})
        if summary:
            successful_rois = summary.get("successful_calibration_calculations", 0)
            total_rois = summary.get("total_rois", 0)
            success_rate = summary.get("success_rate_percent", 0)
            avg_movement = summary.get("average_movement_percentage", 0)

            widget._log_message("📊 CALIBRATION ANALYSIS SUMMARY:")
            widget._log_message(
                f"   Successful ROIs: {successful_rois}/{total_rois} ({success_rate:.1f}%)"
            )
            widget._log_message(f"   Average movement: {avg_movement:.1f}%")

            # Log quality distribution
            quality_dist = summary.get("quality_distribution", {})
            widget._log_message(f"   Quality distribution:")
            for quality, count in quality_dist.items():
                if count > 0:
                    widget._log_message(f"     {quality}: {count} ROIs")

            # Log quality warnings/recommendations
            if quality_dist.get("poor", 0) > 0:
                widget._log_message(
                    f"⚠️ {quality_dist['poor']} ROIs have poor calibration quality"
                )
                widget._log_message(
                    f"   Consider checking calibration file or using different method"
                )

            if (
                quality_dist.get("excellent", 0) + quality_dist.get("good", 0)
                > total_rois * 0.8
            ):
                widget._log_message(
                    "🏆 Most ROIs have good/excellent calibration quality!"
                )

        # Log sample ROI results
        widget._log_message("📋 Sample ROI results:")
        sample_rois = sorted(widget.roi_statistics.keys())[:3]
        for roi in sample_rois:
            stats = widget.roi_statistics[roi]
            if stats.get("status") == "success":
                quality = stats.get("calibration_quality", "unknown")
                snr = stats.get("signal_to_noise_ratio", 0)
                baseline = widget.roi_baseline_means.get(roi, 0)
                upper = widget.roi_upper_thresholds.get(roi, 0)
                lower = widget.roi_lower_thresholds.get(roi, 0)

                widget._log_message(f"   ROI {roi}: {quality} quality, SNR: {snr:.2f}")
                widget._log_message(
                    f"     Baseline: {baseline:.1f}, Range: {lower:.1f} - {upper:.1f}"
                )

        widget._log_message(
            "🎉 CALIBRATION ANALYSIS INTEGRATION COMPLETED SUCCESSFULLY"
        )
        return True

    except Exception as e:
        widget._log_message(f"❌ Calibration analysis integration failed: {str(e)}")
        import traceback

        widget._log_message(f"Full traceback: {traceback.format_exc()}")
        return False


# =============================================================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# =============================================================================
def run_calibration_analysis_legacy(
    merged_results: Dict[int, List[Tuple[float, float]]],
    calibration_file_path: str,
    masks: List[np.ndarray],
    **kwargs,
) -> Dict[str, Any]:
    """
    LEGACY: On-the-fly calibration processing workflow.

    This function processes calibration data during analysis rather than pre-computing it.
    Kept for backwards compatibility.

    Args:
        merged_results: Main experimental data from Reader
        calibration_file_path: Path to calibration HDF5 file
        masks: ROI masks for calibration processing
        **kwargs: Additional analysis parameters

    Returns:
        Complete analysis results using calibration-based thresholds
    """
    print("⚠️ USING LEGACY CALIBRATION WORKFLOW")
    print("=" * 50)
    print("📊 Processing calibration file on-the-fly during analysis")
    print(
        "💡 RECOMMENDATION: Use pre-computed calibration workflow for better transparency"
    )

    if not calibration_file_path or not os.path.exists(calibration_file_path):
        raise ValueError(f"Calibration file not found: {calibration_file_path}")

    if not masks:
        raise ValueError("ROI masks required for calibration processing")

    analysis_results = {"method": "calibration_legacy", "parameters": kwargs.copy()}

    # Step 1: Preprocess main dataset (same as new workflow)
    print(f"\n📊 STEP 1: PREPROCESSING MAIN DATASET")

    enable_matlab_norm = kwargs.get("enable_matlab_norm", True)
    enable_detrending = kwargs.get("enable_detrending", True)
    use_improved_detrending = kwargs.get("use_improved_detrending", True)

    if enable_matlab_norm:
        normalized_data = apply_matlab_normalization_to_merged_results(
            merged_results, enable_matlab_norm=True
        )
    else:
        normalized_data = merged_results

    if enable_detrending:
        if use_improved_detrending:
            processed_data = improved_full_dataset_detrending(normalized_data)
        else:
            processed_data = normalize_and_detrend_merged_results(normalized_data)
    else:
        processed_data = normalized_data

    analysis_results["processed_data"] = processed_data

    # Step 2: Calculate calibration-based thresholds for each ROI
    print(f"\n📊 STEP 2: CALCULATING CALIBRATION THRESHOLDS")

    baseline_means = {}
    upper_thresholds = {}
    lower_thresholds = {}
    roi_statistics = {}

    calibration_multiplier = kwargs.get("calibration_multiplier", 1.0)
    frame_interval = kwargs.get("frame_interval", 5.0)

    for roi_index, roi in enumerate(sorted(processed_data.keys())):
        roi_data = processed_data[roi]

        if not roi_data:
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {"method": "calibration_legacy", "status": "no_data"}
            continue

        try:
            # Use the existing calibration threshold calculation
            baseline_mean, upper_threshold, lower_threshold, stats = (
                compute_threshold_calibration_hysteresis(
                    target_data=roi_data,
                    calibration_file_path=calibration_file_path,
                    masks=masks,
                    roi_index=roi_index,
                    calibration_multiplier=calibration_multiplier,
                    frame_interval=frame_interval,
                    enable_matlab_norm=enable_matlab_norm,
                    enable_detrending=enable_detrending,
                )
            )

            baseline_means[roi] = baseline_mean
            upper_thresholds[roi] = upper_threshold
            lower_thresholds[roi] = lower_threshold
            roi_statistics[roi] = stats

            print(f"   ✅ ROI {roi}: Calibration thresholds calculated")

        except Exception as e:
            print(f"   ❌ ROI {roi}: Calibration failed - {e}")
            baseline_means[roi] = 0.0
            upper_thresholds[roi] = 0.0
            lower_thresholds[roi] = 0.0
            roi_statistics[roi] = {
                "method": "calibration_legacy",
                "status": f"error: {e}",
            }

    analysis_results.update(
        {
            "baseline_means": baseline_means,
            "upper_thresholds": upper_thresholds,
            "lower_thresholds": lower_thresholds,
            "roi_statistics": roi_statistics,
        }
    )

    # Step 3: Apply movement detection with calibration thresholds
    print(f"\n📊 STEP 3: APPLYING CALIBRATION-BASED MOVEMENT DETECTION")

    movement_data = define_movement_with_hysteresis(
        processed_data, baseline_means, upper_thresholds, lower_thresholds
    )
    analysis_results["movement_data"] = movement_data

    # Step 4: Behavioral analysis
    print(f"\n📊 STEP 4: BEHAVIORAL ANALYSIS")

    bin_size_seconds = kwargs.get("bin_size_seconds", 60)
    fraction_data = bin_fraction_movement(
        movement_data, bin_size_seconds, frame_interval
    )
    analysis_results["fraction_data"] = fraction_data

    quiescence_threshold = kwargs.get("quiescence_threshold", 0.5)
    quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)
    analysis_results["quiescence_data"] = quiescence_data

    sleep_threshold_minutes = kwargs.get("sleep_threshold_minutes", 8)
    sleep_data = define_sleep_periods(
        quiescence_data, sleep_threshold_minutes, bin_size_seconds
    )
    analysis_results["sleep_data"] = sleep_data

    # Add ROI colors
    try:
        from ._reader import get_roi_colors

        roi_colors = get_roi_colors(sorted(processed_data.keys()))
    except:
        roi_colors = {
            roi: f"C{i}" for i, roi in enumerate(sorted(processed_data.keys()))
        }

    analysis_results["roi_colors"] = roi_colors

    # Summary
    successful_rois = sum(
        1 for stats in roi_statistics.values() if stats.get("status") == "success"
    )

    analysis_results["summary"] = {
        "total_rois": len(processed_data),
        "successful_calibration_calculations": successful_rois,
        "calibration_file": os.path.basename(calibration_file_path),
        "method_type": "legacy_on_the_fly_processing",
    }

    print(f"\n✅ LEGACY CALIBRATION ANALYSIS COMPLETE")
    print(f"   Successful ROIs: {successful_rois}/{len(processed_data)}")
    print(f"=" * 50)

    return analysis_results


def compute_threshold_calibration(
    target_data: List[Tuple[float, float]],
    calibration_file_path: str,
    masks: List[np.ndarray],
    roi_index: int,
    percentile_threshold: float = 95.0,
    calibration_multiplier: float = 1.0,
    frame_interval: float = 5.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    BACKWARDS COMPATIBILITY: Legacy single threshold calibration calculation.

    This function maintains compatibility with older code that expects a single
    threshold value rather than hysteresis upper/lower thresholds.

    Args:
        target_data: Main experimental data for this ROI
        calibration_file_path: Path to sedated animal HDF5 file
        masks: ROI masks
        roi_index: ROI index (0-based)
        percentile_threshold: Legacy parameter (kept for compatibility)
        calibration_multiplier: Multiplier for threshold calculation
        frame_interval: Time between frames

    Returns:
        Tuple of (single_threshold, statistics_dict) - for backwards compatibility
    """
    print(f"⚠️ Using legacy single-threshold calibration function")
    print(f"   Recommend upgrading to hysteresis version for better results")

    # Use the hysteresis calculation internally
    baseline_mean, upper_threshold, lower_threshold, stats = (
        compute_threshold_calibration_hysteresis(
            target_data=target_data,
            calibration_file_path=calibration_file_path,
            masks=masks,
            roi_index=roi_index,
            calibration_multiplier=calibration_multiplier,
            frame_interval=frame_interval,
            enable_matlab_norm=True,
            enable_detrending=True,
        )
    )

    # For backwards compatibility, return the upper threshold as the "single threshold"
    single_threshold = upper_threshold

    # Add legacy compatibility fields to statistics
    legacy_stats = stats.copy()
    legacy_stats.update(
        {
            "threshold": single_threshold,
            "percentile_threshold": percentile_threshold,  # Preserved for compatibility
            "uses_hysteresis": False,  # Lie for compatibility
            "legacy_compatibility": True,
            "recommendation": "Upgrade to compute_threshold_calibration_hysteresis for better results",
        }
    )

    return single_threshold, legacy_stats
