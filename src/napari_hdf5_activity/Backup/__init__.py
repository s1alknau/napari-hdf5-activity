# napari_hdf5_activity/__init__.py

# === NAPARI PLUGIN ENTRY POINTS ===
from . import _reader, _widget
from ._reader import napari_get_reader
from ._widget import napari_provide_dock_widget
from ._metadata import (
    # Core metadata functions
    extract_hdf5_metadata,  # Main metadata extraction function
    create_metadata_summary,  # Human-readable summary generation
)

# === MODULE 1: READER (_reader.py) ===
# HDF5 file reading and data processing
from ._reader import (
    # Core data processing
    process_hdf5_files,
    merge_results,
    # Frame and ROI utilities
    get_first_frame,
    get_roi_colors,
    sort_circles_left_to_right,
)

# === MODULE 2: CALCULATIONS (MODULAR STRUCTURE) ===

# === 2A: CORE CALCULATIONS (_calc.py) ===
# Baseline method and core utilities
from ._calc import (
    # Main baseline analysis pipeline
    run_baseline_analysis,  # Primary baseline pipeline
    integrate_baseline_analysis_with_widget,  # Baseline widget integration
    # Data processing and normalization
    normalize_and_detrend_merged_results,
    apply_matlab_normalization_to_merged_results,
    # Baseline threshold calculation
    compute_threshold_baseline_hysteresis,
    # Hysteresis-based movement detection
    define_movement_with_hysteresis,
    # Behavior analysis functions
    bin_fraction_movement,
    bin_quiescence,
    define_sleep_periods,
    bin_activity_data_for_lighting,
    # Utility and diagnostic functions
    get_performance_metrics,
    # Validation functions
    validate_analysis_parameters,
    validate_frame_difference_data,
    validate_detrending_effectiveness,
    validate_matlab_normalization,
    # Detrending functions
    improved_full_dataset_detrending,
    robust_detrend_baseline,
    detect_and_remove_jumps_aggressive,
    remove_polynomial_trend,
    remove_linear_drift,
)

# === 2B: ADAPTIVE CALCULATIONS (_calc_adaptive.py) ===
# Adaptive threshold method
from ._calc_adaptive import (
    # Main adaptive analysis pipeline
    run_adaptive_analysis,  # Primary adaptive pipeline
    integrate_adaptive_analysis_with_widget,  # Adaptive widget integration
    # Adaptive threshold calculation
    compute_threshold_adaptive_hysteresis,
    # Backwards compatibility
    compute_threshold_adaptive,  # Legacy single threshold
)

# === 2C: CALIBRATION CALCULATIONS (_calc_calibration.py) ===
# Calibration-based threshold method
from ._calc_calibration import (
    # Main calibration analysis pipeline
    run_calibration_analysis,  # Primary calibration pipeline
    integrate_calibration_analysis_with_widget,  # Calibration widget integration
    # Calibration processing
    process_calibration_file,
    # Calibration threshold calculation
    compute_threshold_calibration_hysteresis,
    # Backwards compatibility
    compute_threshold_calibration,  # Legacy single threshold
)

# === 2D: UNIFIED CALCULATION INTERFACE (_calc_integration.py) ===
# Unified interface for all calculation methods
from ._calc_integration import (
    # Main unified interface
    run_analysis_with_method,  # Universal analysis function
    integrate_analysis_with_widget,  # Universal widget integration
    # Method utilities
    get_method_requirements,
    validate_method_parameters,
    get_available_methods,
    get_method_description,
    suggest_method_for_data,
    # Analysis utilities
    get_analysis_summary,
    quick_method_test,
    export_method_comparison,
    create_method_comparison_report,
    # Backwards compatibility (unified interface)
    compute_roi_thresholds_unified,
    compute_roi_thresholds_hysteresis,
)

# === MODULE 3: PLOTTING (_plot.py) ===
# All plotting, visualization, and figure generation
from ._plot import (
    # Main plotting class
    PlotGenerator,
    # Configuration utilities
    create_plot_config,
    create_hysteresis_kwargs,
    # Export utilities
    save_plot,
    save_all_plot_types,
)

# === MODULE 4: WIDGET (_widget.py) ===
# UI management and workflow coordination
from ._widget import HDF5AnalysisWidget

# === VERSION AND METADATA ===
__version__ = "2.0.0"  # Updated for modular structure
__author__ = "s1alknau"
__email__ = "alexander.knauss@leibniz-ipht.de"
__description__ = "Advanced HDF5 activity analysis with modular calculation methods and hysteresis-based movement detection"

# === COMPREHENSIVE EXPORTS FOR NAPARI PLUGIN ===
__all__ = [
    # === NAPARI ENTRY POINTS ===
    "napari_get_reader",  # File reading entry point
    "napari_provide_dock_widget",  # Widget entry point
    # === READER MODULE (_reader.py) ===
    "process_hdf5_files",  # Core HDF5 processing
    "merge_results",  # Data merging
    "get_first_frame",  # Frame extraction
    "get_roi_colors",  # ROI visualization
    "sort_circles_left_to_right",  # ROI organization
    # === CORE CALCULATION MODULE (_calc.py) ===
    # Baseline analysis
    "run_baseline_analysis",
    "integrate_baseline_analysis_with_widget",
    # Data processing
    "normalize_and_detrend_merged_results",
    "normalize_and_detrend_merged_results_with_matlab_norm",
    "apply_matlab_normalization_to_merged_results",
    # Baseline thresholds
    "compute_threshold_baseline_hysteresis",
    # Movement detection
    "define_movement_with_hysteresis",
    # Behavior analysis
    "bin_fraction_movement",
    "bin_quiescence",
    "define_sleep_periods",
    "bin_activity_data_for_lighting",
    # Utilities
    "get_performance_metrics",
    # Validation
    "validate_analysis_parameters",
    "validate_frame_difference_data",
    "validate_detrending_effectiveness",
    "validate_matlab_normalization",
    # Detrending
    "improved_full_dataset_detrending",
    "robust_detrend_baseline",
    "detect_and_remove_jumps_aggressive",
    "remove_polynomial_trend",
    "remove_linear_drift",
    # === ADAPTIVE CALCULATION MODULE (_calc_adaptive.py) ===
    "run_adaptive_analysis",
    "integrate_adaptive_analysis_with_widget",
    "compute_threshold_adaptive_hysteresis",
    "compute_threshold_adaptive",  # Legacy
    # === CALIBRATION CALCULATION MODULE (_calc_calibration.py) ===
    "run_calibration_analysis",
    "integrate_calibration_analysis_with_widget",
    "process_calibration_file",
    "compute_threshold_calibration_hysteresis",
    "compute_threshold_calibration",  # Legacy
    # === UNIFIED CALCULATION INTERFACE (_calc_integration.py) ===
    # Main interface
    "run_analysis_with_method",
    "integrate_analysis_with_widget",
    # Method utilities
    "get_method_requirements",
    "validate_method_parameters",
    "get_available_methods",
    "get_method_description",
    "suggest_method_for_data",
    # Analysis utilities
    "get_analysis_summary",
    "quick_method_test",
    "export_method_comparison",
    "create_method_comparison_report",
    # Backwards compatibility
    "compute_roi_thresholds_unified",
    "compute_roi_thresholds_hysteresis",
    # === PLOTTING MODULE (_plot.py) ===
    "PlotGenerator",  # Main plotting class
    "create_plot_config",  # Plot configuration
    "create_hysteresis_kwargs",  # Hysteresis plot setup
    "save_plot",  # Single plot export
    "save_all_plot_types",  # Batch plot export
    # === WIDGET MODULE (_widget.py) ===
    "HDF5AnalysisWidget",  # Main widget class
]


# === PLUGIN INFORMATION FOR NAPARI ===
def get_plugin_modules():
    """
    Get information about the plugin's modular architecture.

    Returns:
        dict: Information about each module and its purpose
    """
    return {
        "_reader.py": {
            "purpose": "HDF5 file reading and data processing",
            "key_functions": [
                "napari_get_reader",
                "process_hdf5_files",
                "merge_results",
                "get_first_frame",
                "get_roi_colors",
            ],
            "description": "Handles all file I/O and initial data processing",
        },
        "_calc.py": {
            "purpose": "Core calculations and baseline method",
            "key_functions": [
                "run_baseline_analysis",
                "compute_threshold_baseline_hysteresis",
                "define_movement_with_hysteresis",
                "apply_matlab_normalization_to_merged_results",
            ],
            "description": "Contains baseline method and core calculation utilities",
        },
        "_calc_adaptive.py": {
            "purpose": "Adaptive threshold calculation method",
            "key_functions": [
                "run_adaptive_analysis",
                "compute_threshold_adaptive_hysteresis",
                "integrate_adaptive_analysis_with_widget",
            ],
            "description": "Adaptive method that adjusts based on signal characteristics",
        },
        "_calc_calibration.py": {
            "purpose": "Calibration-based threshold calculation method",
            "key_functions": [
                "run_calibration_analysis",
                "compute_threshold_calibration_hysteresis",
                "process_calibration_file",
            ],
            "description": "Uses sedated animal recordings for threshold calibration",
        },
        "_calc_integration.py": {
            "purpose": "Unified interface for all calculation methods",
            "key_functions": [
                "run_analysis_with_method",
                "integrate_analysis_with_widget",
                "get_available_methods",
                "quick_method_test",
            ],
            "description": "Provides unified access to all calculation methods",
        },
        "_plot.py": {
            "purpose": "Plotting, visualization, and figure generation",
            "key_functions": [
                "PlotGenerator",
                "create_plot_config",
                "save_plot",
                "save_all_plot_types",
            ],
            "description": "Handles all visualization and plot generation",
        },
        "_widget.py": {
            "purpose": "UI management and workflow coordination",
            "key_functions": ["napari_provide_dock_widget", "HDF5AnalysisWidget"],
            "description": "Manages user interface and coordinates between modules",
        },
    }


def get_plugin_workflow():
    """
    Get information about the typical plugin workflow.

    Returns:
        list: Ordered steps of the typical analysis workflow
    """
    return [
        "1. Load HDF5 files using _reader.py",
        "2. Detect ROIs in the widget (_widget.py)",
        "3. Choose analysis method (baseline, adaptive, or calibration)",
        "4. Run analysis using unified interface (_calc_integration.py)",
        "5. Generate plots using _plot.py functions",
        "6. Export results for further analysis",
    ]


def get_available_analysis_methods():
    """
    Get information about available analysis methods.

    Returns:
        dict: Information about each analysis method
    """
    return {
        "baseline": {
            "module": "_calc.py",
            "function": "run_baseline_analysis",
            "description": "Uses first N frames to establish baseline thresholds",
            "best_for": "Stable recording conditions",
            "required_params": ["threshold_block_count", "multiplier"],
        },
        "adaptive": {
            "module": "_calc_adaptive.py",
            "function": "run_adaptive_analysis",
            "description": "Automatically adapts based on signal characteristics",
            "best_for": "Varying recording conditions or unknown signal quality",
            "required_params": ["analysis_duration_frames", "base_multiplier"],
        },
        "calibration": {
            "module": "_calc_calibration.py",
            "function": "run_calibration_analysis",
            "description": "Uses sedated animal recordings for threshold calibration",
            "best_for": "When calibration data from sedated animals is available",
            "required_params": ["calibration_file_path", "percentile_threshold"],
        },
    }


# Add plugin info functions to exports
__all__.extend(
    [
        "get_plugin_modules",
        "get_plugin_workflow",
        "get_available_analysis_methods",
    ]
)

# === BACKWARDS COMPATIBILITY ALIASES ===
# Legacy function names for existing code that might still use the old interface


def _create_legacy_alias(new_func_name, old_func_name, module_name):
    """Create a lazy-loaded legacy alias."""

    def legacy_wrapper(*args, **kwargs):
        print(
            f"Warning: {old_func_name} is deprecated. Use {new_func_name} from {module_name} instead."
        )
        # Import and call the new function
        if module_name == "_calc_integration":
            from ._calc_integration import run_analysis_with_method

            return run_analysis_with_method(*args, **kwargs)
        elif module_name == "_calc":
            from . import _calc

            return getattr(_calc, new_func_name)(*args, **kwargs)
        # Add other modules as needed

    return legacy_wrapper


# Legacy aliases (only the most critical ones)
run_complete_hdf5_compatible_analysis = _create_legacy_alias(
    "run_analysis_with_method",
    "run_complete_hdf5_compatible_analysis",
    "_calc_integration",
)

run_complete_matlab_compatible_analysis = _create_legacy_alias(
    "run_baseline_analysis", "run_complete_matlab_compatible_analysis", "_calc"
)

integrate_hdf5_analysis_with_widget = _create_legacy_alias(
    "integrate_analysis_with_widget",
    "integrate_hdf5_analysis_with_widget",
    "_calc_integration",
)


# Legacy aliases for specific threshold functions
def compute_threshold_baseline(*args, **kwargs):
    """Legacy alias for baseline threshold calculation."""
    print(
        "Warning: compute_threshold_baseline is deprecated. Use compute_threshold_baseline_hysteresis from _calc instead."
    )
    from ._calc import compute_threshold_baseline_hysteresis

    return compute_threshold_baseline_hysteresis(*args, **kwargs)


def define_movement_from_thresholds(*args, **kwargs):
    """Legacy alias for movement detection."""
    print(
        "Warning: define_movement_from_thresholds is deprecated. Use define_movement_with_hysteresis for better results."
    )
    from ._calc import define_movement_with_hysteresis

    # This would need adaptation logic here
    return define_movement_with_hysteresis(*args, **kwargs)


# Add essential legacy aliases to exports
__all__.extend(
    [
        "run_complete_hdf5_compatible_analysis",  # Legacy alias
        "run_complete_matlab_compatible_analysis",  # Legacy alias
        "integrate_hdf5_analysis_with_widget",  # Legacy alias
        "compute_threshold_baseline",  # Legacy alias
        "define_movement_from_thresholds",  # Legacy alias
    ]
)


# === MODULE VALIDATION ===
def check_plugin_modules():
    """
    Verify that all plugin modules are properly loaded.

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Test imports from each module
        from . import _calc, _calc_adaptive, _calc_calibration, _calc_integration, _plot

        # Verify key functions exist
        assert hasattr(_reader, "napari_get_reader")
        assert hasattr(_calc, "run_baseline_analysis")
        assert hasattr(_calc_adaptive, "run_adaptive_analysis")
        assert hasattr(_calc_calibration, "run_calibration_analysis")
        assert hasattr(_calc_integration, "run_analysis_with_method")
        assert hasattr(_plot, "PlotGenerator")
        assert hasattr(_widget, "napari_provide_dock_widget")

        return True, "All plugin modules loaded successfully"

    except Exception as e:
        return False, f"Module loading error: {e}"


def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        "numpy",
        "matplotlib",
        "cv2",
        "h5py",
        "psutil",
        "napari",
        "qtpy",
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"
    else:
        return True, "All dependencies satisfied"


def get_plugin_info():
    """Get comprehensive information about this plugin."""
    return {
        "name": "HDF5 Activity Analysis",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "architecture": "Modular calculation methods",
        "modules": {
            "reader": "HDF5 file reading and processing",
            "calc": "Core calculations and baseline method",
            "calc_adaptive": "Adaptive threshold calculation",
            "calc_calibration": "Calibration-based threshold calculation",
            "calc_integration": "Unified interface for all methods",
            "plot": "Plotting and visualization",
            "widget": "UI management and coordination",
        },
        "analysis_methods": {
            "baseline": "Uses first N frames for threshold calculation",
            "adaptive": "Automatically adapts based on signal characteristics",
            "calibration": "Uses sedated animal recordings for calibration",
        },
        "key_features": [
            "Modular calculation architecture",
            "Three distinct analysis methods",
            "Hysteresis-based movement detection",
            "MATLAB compatibility",
            "HDF5 timing correction",
            "Advanced plotting capabilities",
            "ROI detection and management",
            "Method comparison tools",
            "Export to multiple formats",
        ],
    }


# Add convenience functions to exports
__all__.extend(
    [
        "check_plugin_modules",
        "check_dependencies",
        "get_plugin_info",
    ]
)


# === MODULE INITIALIZATION ===
def _initialize_plugin():
    """Initialize plugin and perform any necessary setup."""
    try:
        # Check dependencies
        deps_ok, deps_msg = check_dependencies()
        if not deps_ok:
            print(f"Warning: {deps_msg}")

        # Check modules
        modules_ok, modules_msg = check_plugin_modules()
        if not modules_ok:
            print(f"Warning: {modules_msg}")
            return False

        print("HDF5 Activity Analysis Plugin (Modular v2.0) loaded successfully")
        print("Available analysis methods: baseline, adaptive, calibration")
        return True

    except Exception as e:
        print(f"Warning: Plugin initialization issue: {e}")
        return False


# Run initialization
_plugin_initialized = _initialize_plugin()


# === CONVENIENCE IMPORTS FOR COMMON USAGE ===
# Make it easy for users to access the most common functions
def get_quick_imports():
    """
    Get the most commonly used functions for quick access.

    Returns:
        dict: Dictionary of commonly used functions
    """
    return {
        # Most common analysis function
        "analyze": run_analysis_with_method,
        # Most common widget integration
        "integrate_with_widget": integrate_analysis_with_widget,
        # Method information
        "available_methods": get_available_methods,
        "method_info": get_method_description,
        # Quick testing
        "quick_test": quick_method_test,
        # Plotting
        "plot": PlotGenerator,
        # Legacy support
        "legacy_analyze": run_complete_hdf5_compatible_analysis,
    }


# Export initialization status and convenience imports
__all__.extend(["_plugin_initialized", "get_quick_imports"])

# === USAGE EXAMPLES IN DOCSTRING ===
__doc__ = f"""
HDF5 Activity Analysis Plugin v{__version__}

MODULAR ARCHITECTURE:
- _calc.py: Baseline method and core utilities
- _calc_adaptive.py: Adaptive threshold calculation
- _calc_calibration.py: Calibration-based method
- _calc_integration.py: Unified interface
- _plot.py: Visualization and plotting
- _reader.py: HDF5 file processing
- _widget.py: User interface

QUICK START:
```python
# Import the plugin
import napari_hdf5_activity as hdf5

# Load and analyze data
from napari_hdf5_activity import run_analysis_with_method

# Run baseline analysis
results = run_analysis_with_method(
    merged_results, 'baseline',
    threshold_block_count=120, multiplier=1.0
)

# Run adaptive analysis
results = run_analysis_with_method(
    merged_results, 'adaptive',
    analysis_duration_frames=180, base_multiplier=2.5
)

# Quick method comparison
comparison = hdf5.quick_method_test(merged_results)
```

WIDGET USAGE:
The plugin provides a napari dock widget for interactive analysis.
Access via: Plugins > napari-hdf5-activity: HDF5 Analysis Widget

ANALYSIS METHODS:
- Baseline: Best for stable recording conditions
- Adaptive: Best for varying conditions or unknown signal quality
- Calibration: Best when sedated animal recordings are available
"""
