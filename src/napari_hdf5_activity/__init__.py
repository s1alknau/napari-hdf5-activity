"""
napari-hdf5-activity - Clean and modular HDF5 analysis plugin

This plugin provides comprehensive analysis of HDF5 time-lapse data with multiple threshold methods.
"""

__version__ = "0.2.0"

# === MODULE 1: READER FUNCTIONS ===
from ._reader import (
    # Napari reader functions
    napari_get_reader,
    reader_function_dual_structure,
    # Core HDF5 processing
    process_single_file_in_parallel_dual_structure,
    process_hdf5_file_dual_structure,
    process_hdf5_files,
    # Frame and structure functions
    get_first_frame_enhanced,
    get_first_frame,
    detect_hdf5_structure_type,
    # ROI and visualization utilities
    sort_circles_left_to_right,
    get_roi_colors,
)

# === MODULE 2: CALCULATIONS (MODULAR STRUCTURE) ===

# === 2A: CORE CALCULATIONS (_calc.py) ===
from ._calc import (
    # Main baseline analysis pipeline
    run_baseline_analysis,
    integrate_baseline_analysis_with_widget,
    # Data processing
    apply_matlab_normalization_to_merged_results,
    improved_full_dataset_detrending,
    # Threshold calculation
    compute_threshold_baseline_hysteresis,
    # Movement detection
    define_movement_with_hysteresis,
    # Behavior analysis
    bin_fraction_movement,
    bin_quiescence,
    define_sleep_periods,
    bin_activity_data_for_lighting,
    # Utilities
    get_performance_metrics,
)

# === 2B: CALIBRATION CALCULATIONS (_calc_calibration.py) ===
try:
    from ._calc_calibration import (
        run_calibration_analysis_with_precomputed_baseline,
        process_calibration_baseline,
        integrate_calibration_analysis_with_widget,
        run_calibration_analysis,  # Legacy compatibility
    )
except ImportError as e:
    print(f"Warning: Calibration calculations not available: {e}")

# === 2C: INTEGRATION MODULE (_calc_integration.py) ===
try:
    from ._calc_integration import (
        run_analysis_with_method,  # Main routing function
        get_analysis_summary,  # Results summary
        validate_hdf5_timing_in_data,  # Timing validation
        export_results_for_matlab,  # MATLAB export
        quick_method_test,  # Testing utility
        integrate_analysis_with_widget,  # Widget integration
    )
except ImportError as e:
    print(f"Warning: Integration module not available: {e}")

# === MODULE 3: PLOTTING ===
try:
    from ._plot import (
        PlotGenerator,
        create_plot_config,
        create_hysteresis_kwargs,
        save_plot,
        save_all_plot_types,
    )
except ImportError as e:
    print(f"Warning: Plot functions not available: {e}")

# === MODULE 4: METADATA ===
try:
    from ._metadata import (
        extract_hdf5_metadata,
        extract_hdf5_metadata_timeseries,
        create_hdf5_metadata_timeseries_dataframe,
        write_metadata_to_csv,
        filter_hdf5_metadata_only,
        NematostellaTimeseriesAnalyzer,
        analyze_nematostella_hdf5_file,
        get_nematostella_timeseries_summary,
    )
except ImportError as e:
    print(f"Warning: Metadata functions not available: {e}")

# === MODULE 5: MAIN WIDGET ===
from ._widget import HDF5AnalysisWidget

# === BACKWARDS COMPATIBILITY ===
# Legacy aliases for backwards compatibility
try:
    # Alias integration functions for legacy code
    integrate_hdf5_analysis_with_widget = integrate_analysis_with_widget
    quick_analysis_test = quick_method_test

    # Legacy calculation aliases
    run_complete_hdf5_compatible_analysis = run_baseline_analysis

except NameError:
    print("Warning: Some legacy compatibility functions not available")

# === NAPARI PLUGIN EXPORTS ===
__all__ = [
    # Reader functions
    "napari_get_reader",
    # Main widget
    "HDF5AnalysisWidget",
    # Core analysis functions
    "run_baseline_analysis",
    "run_analysis_with_method",
    # Processing functions
    "process_single_file_in_parallel_dual_structure",
    "get_first_frame_enhanced",
    # Metadata functions
    "extract_hdf5_metadata_timeseries",
    "analyze_nematostella_hdf5_file",
    # Utility functions
    "get_roi_colors",
    "get_analysis_summary",
]
