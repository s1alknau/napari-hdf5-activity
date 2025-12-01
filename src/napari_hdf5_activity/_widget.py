"""
widget.py - All methods properly placed within the HDF5AnalysisWidget class
"""

from datetime import datetime
import os
import cv2
import h5py
import time
import csv
import pandas as pd
import psutil
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer, Signal, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QCheckBox,
    QSlider,
    QSplitter,
)

try:
    from ._reader import (
        detect_hdf5_structure_type,
        get_first_frame_enhanced,
        process_single_file_in_parallel_dual_structure,
        process_hdf5_file_dual_structure,
        reader_function_dual_structure,
        # Keep original imports as fallback
        napari_get_reader,
        get_first_frame,
        get_roi_colors,
        merge_results,
        process_hdf5_files,
        sort_circles_left_to_right,
        sort_circles_meandering_auto,  # New function
    )

    DUAL_STRUCTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dual structure functions not available: {e}")
    DUAL_STRUCTURE_AVAILABLE = False
    # Use original imports only
    from ._reader import (
        napari_get_reader,
        get_first_frame,  # Should be available if left_to_right is
    )
try:
    from ._metadata import (
        extract_hdf5_metadata_timeseries,
        create_hdf5_metadata_timeseries_dataframe,
        write_metadata_to_csv,
        filter_hdf5_metadata_only,
    )

    METADATA_AVAILABLE = True

    # Try to import Nematostella functions separately
    try:
        from ._metadata import (
            analyze_nematostella_hdf5_file,
            NematostellaTimeseriesAnalyzer,
        )

        nematostella_analysis_available = True
    except ImportError:
        nematostella_analysis_available = False

except ImportError as e:
    print(f"Warning: Metadata functions not available: {e}")
    METADATA_AVAILABLE = False
    nematostella_analysis_available = False


# Import calculation functions with clear fallbacks
def validate_analysis_parameters(
    frame_interval: float, chunk_size: int, baseline_duration_minutes: float
) -> Tuple[bool, str]:
    """Validate analysis parameters before starting analysis."""
    if frame_interval <= 0:
        return False, "Frame interval must be positive"
    if chunk_size <= 0:
        return False, "Chunk size must be positive"
    if baseline_duration_minutes <= 0:
        return False, "Baseline duration must be positive"
    if baseline_duration_minutes > 10000:
        return False, "Baseline duration seems unreasonably long (>10000 minutes)"
    if frame_interval > 300:
        return False, "Frame interval seems unreasonably long (>5 minutes)"
    return True, ""


# Define bin_quiescence fallback function
def bin_quiescence_fallback(fraction_data, threshold):
    """Fallback quiescence calculation when main function not available."""
    quiescence_data = {}
    for roi, data in fraction_data.items():
        quiescence_data[roi] = [
            (t, 1 if fraction < threshold else 0) for t, fraction in data
        ]
    return quiescence_data


# Try integrated system first, then legacy, then fallbacks
CALC_SYSTEM = "none"
bin_quiescence = None  # Initialize as None

try:
    from ._calc_integration import (
        run_analysis_with_method,
        get_analysis_summary,
        quick_method_test,
        validate_hdf5_timing_in_data,
        export_results_for_matlab,
    )

    # Try to get bin_quiescence from _calc since it's not in _calc_integration
    try:
        from ._calc import bin_quiescence
    except ImportError:
        bin_quiescence = bin_quiescence_fallback
        print("Warning: Using fallback bin_quiescence function")

    try:
        if CALC_SYSTEM == "integrated":
            from ._calc_integration import get_performance_metrics
        else:
            from ._calc import get_performance_metrics
    except ImportError:

        def get_performance_metrics(start_time, total_frames):
            import time

            elapsed = time.time() - start_time
            return {
                "elapsed_time": elapsed,
                "fps": total_frames / elapsed if elapsed > 0 else 0,
                "cpu_percent": 0,
                "memory_percent": 0,
                "total_frames": total_frames,
            }

    def run_complete_hdf5_compatible_analysis(merged_results, **kwargs):
        method = kwargs.pop("threshold_method", "baseline")
        return run_analysis_with_method(merged_results, method, **kwargs)

    quick_analysis_test = quick_method_test
    CALC_SYSTEM = "integrated"
    print("Using integrated calculation system")

except ImportError:
    try:
        from ._calc import (
            run_complete_hdf5_compatible_analysis,
            get_analysis_summary,
            quick_analysis_test,
            validate_hdf5_timing_in_data,
            export_results_for_matlab,
            bin_quiescence,
        )

        CALC_SYSTEM = "legacy"
        print("Using legacy calculation system")

    except ImportError as e:
        print(f"Warning: No calculation system available: {e}")
        CALC_SYSTEM = "fallback"

        # Use fallback bin_quiescence
        bin_quiescence = bin_quiescence_fallback

        def run_complete_hdf5_compatible_analysis(merged_results, **kwargs):
            return {
                "method": "fallback",
                "baseline_means": {roi: 0.0 for roi in merged_results.keys()},
                "upper_thresholds": {roi: 1.0 for roi in merged_results.keys()},
                "lower_thresholds": {roi: -1.0 for roi in merged_results.keys()},
                "movement_data": {roi: [] for roi in merged_results.keys()},
                "fraction_data": {roi: [] for roi in merged_results.keys()},
                "sleep_data": {roi: [] for roi in merged_results.keys()},
                "quiescence_data": {roi: [] for roi in merged_results.keys()},
                "roi_statistics": {roi: {} for roi in merged_results.keys()},
                "error": "No calculation system available",
            }

        def get_analysis_summary(results):
            return "No calculation system available for summary"

        def quick_analysis_test(merged_results):
            return "No calculation system available for testing"

        def validate_hdf5_timing_in_data(merged_results, frame_interval=5.0):
            return {"timing_type": "unknown", "needs_correction": False}

        def export_results_for_matlab(results, output_dir):
            print("Export function not available - no calculation system")
            return []


# Import plotting functions
try:
    from ._plot import (
        PlotGenerator,
        create_plot_config,
        create_hysteresis_kwargs,
        save_plot,
        save_all_plot_types,
    )
except ImportError as e:
    print(f"Warning: Could not import plot functions: {e}")


class HDF5AnalysisWidget(QWidget):
    """
    Simplified widget for analyzing activity in HDF5 files.
    Coordinates between _calc.py and _plot.py modules.
    Handles only UI interactions and file operations.
    """

    # Qt Signals
    progress_updated = Signal(int)
    status_updated = Signal(str)
    performance_updated = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Initialize all attributes first
        self._initialize_attributes()

        # Setup UI after all attributes are initialized
        self.setup_ui()

        # Connect signals
        self._connect_signals()

    def _initialize_attributes(self):
        """Initialize all class attributes."""
        # Performance monitoring
        self.cpu_count = psutil.cpu_count()
        # Add dataset state management
        self._initialize_dataset_state()
        # Conservative approach for Windows multiprocessing
        if os.name == "nt":  # Windows
            self.optimal_processes = max(1, min(4, int(self.cpu_count * 0.6)))
        else:  # Unix/Linux/Mac
            self.optimal_processes = max(1, int(self.cpu_count * 0.9))

        # Analysis state variables
        self.directory: Optional[str] = None
        self.file_path: Optional[str] = None
        self.masks: List[np.ndarray] = []
        self.labeled_frame: Optional[np.ndarray] = None

        # Analysis results (now populated by _calc.py)
        self.merged_results: Dict[int, List[Tuple[float, float]]] = {}
        self.roi_colors: Dict[int, str] = {}
        self.roi_thresholds: Dict[int, float] = {}
        self.roi_statistics: Dict[int, Dict[str, float]] = {}
        self.movement_data: Dict[int, List[Tuple[float, int]]] = {}
        self.fraction_data: Dict[int, List[Tuple[float, float]]] = {}
        self.quiescence_data: Dict[int, List[Tuple[float, int]]] = {}
        self.sleep_data: Dict[int, List[Tuple[float, int]]] = {}

        # Hysteresis data (populated by _calc.py)
        self.roi_baseline_means: Dict[int, float] = {}
        self.roi_upper_thresholds: Dict[int, float] = {}
        self.roi_lower_thresholds: Dict[int, float] = {}
        self.roi_band_widths: Dict[int, float] = {}

        # Worker handle for background analysis
        self.current_worker = None
        self._cancel_requested = False
        self.analysis_start_time: Optional[float] = None

        # Initialize performance timer
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_metrics)
        self.performance_timer.setInterval(1000)  # Update every second

        # Plot generator (initialized when figure is available)
        self.plot_generator = None

        # Calibration workflow state (add these lines)
        self.current_dataset_type = "main"  # "main" or "calibration"
        self.calibration_file_path_stored = None
        self.calibration_masks = []
        self.calibration_labeled_frame = None
        self.calibration_baseline_processed = False
        self.calibration_baseline_statistics = {}

        # Store main dataset state when switching to calibration
        self.main_dataset_path = None
        self.main_masks = []
        self.main_labeled_frame = None

    def _initialize_dataset_state(self):
        """Initialize dataset state management attributes."""
        self.main_dataset_path = None
        self.main_merged_results = {}
        self.main_masks = []
        self.main_labeled_frame = None
        self.main_dataset_stored = False

        self.calibration_file_path_stored = None
        self.calibration_masks = []
        self.calibration_labeled_frame = None
        self.calibration_baseline_processed = False
        self.calibration_baseline_statistics = {}

        self.current_dataset_type = "main"

    def store_main_dataset_state(self):
        """Store the current main dataset state before calibration operations."""
        try:
            self._log_message("Storing main dataset state...")

            # Check if we have valid main dataset to store
            if not hasattr(self, "file_path") or not self.file_path:
                self._log_message("WARNING: No file_path to store as main dataset")
                return False

            # IMPORTANT: Check if we have processed results OR if file is loaded
            if hasattr(self, "merged_results") and self.merged_results:
                # Case 1: We have processed data (ideal case)
                self.main_dataset_path = self.file_path
                self.main_merged_results = self.merged_results.copy()
                self.main_masks = getattr(self, "masks", []).copy()
                self.main_labeled_frame = getattr(self, "labeled_frame", None)
                self.main_dataset_stored = True

                # Verify storage
                sample_roi = list(self.main_merged_results.keys())[0]
                sample_data = self.main_merged_results[sample_roi]
                if sample_data:
                    main_duration = (sample_data[-1][0] - sample_data[0][0]) / 60
                    self._log_message(
                        "Main dataset stored successfully (PROCESSED DATA):"
                    )
                    self._log_message(
                        f"   Path: {os.path.basename(self.main_dataset_path)}"
                    )
                    self._log_message(f"   ROIs: {len(self.main_merged_results)}")
                    self._log_message(f"   Duration: {main_duration:.1f} minutes")
                    self._log_message(f"   Data points: {len(sample_data)}")
                    return True

            elif (
                hasattr(self, "file_path")
                and self.file_path
                and os.path.exists(self.file_path)
            ):
                # Case 2: We have a file loaded but no processed data yet
                self._log_message("Main dataset file loaded but not yet processed")
                self._log_message(
                    "Storing file path and current state for later processing"
                )

                self.main_dataset_path = self.file_path
                self.main_merged_results = (
                    {}
                )  # Empty for now, will be filled during analysis
                self.main_masks = getattr(self, "masks", []).copy()
                self.main_labeled_frame = getattr(self, "labeled_frame", None)
                self.main_dataset_stored = True

                self._log_message("Main dataset file stored (NOT YET PROCESSED):")
                self._log_message(
                    f"   Path: {os.path.basename(self.main_dataset_path)}"
                )
                self._log_message(f"   Masks: {len(self.main_masks)}")
                self._log_message("   Data will be processed during analysis")

                return True

            else:
                self._log_message("ERROR: No valid main dataset file or data to store")
                return False

        except Exception as e:
            self._log_message(f"ERROR storing main dataset: {e}")
            self.main_dataset_stored = False
            return False

    def restore_main_dataset_for_analysis(self):
        """Restore main dataset before running analysis."""
        self._log_message("=== RESTORING MAIN DATASET FOR ANALYSIS ===")

        # Check if we have stored main dataset
        if not hasattr(self, "main_dataset_stored") or not self.main_dataset_stored:
            self._log_message("ERROR: No main dataset was stored")
            return False

        # Check if stored data is valid
        if not hasattr(self, "main_dataset_path") or not self.main_dataset_path:
            self._log_message("ERROR: No main dataset path stored")
            return False

        try:
            # Restore main dataset state
            self.file_path = self.main_dataset_path
            self.current_dataset_type = "main"

            # Case 1: We have processed data to restore
            if hasattr(self, "main_merged_results") and self.main_merged_results:
                self.merged_results = self.main_merged_results.copy()
                sample_roi = list(self.merged_results.keys())[0]
                sample_data = self.merged_results[sample_roi]
                restored_duration = (sample_data[-1][0] - sample_data[0][0]) / 60

                self._log_message("Main dataset restored (PROCESSED DATA):")
                self._log_message(f"   Path: {os.path.basename(self.file_path)}")
                self._log_message(f"   ROIs: {len(self.merged_results)}")
                self._log_message(f"   Duration: {restored_duration:.1f} minutes")

            # Case 2: We only have file path, need to ensure data gets loaded
            else:
                self._log_message(
                    "Main dataset file restored (WILL PROCESS DURING ANALYSIS):"
                )
                self._log_message(f"   Path: {os.path.basename(self.file_path)}")
                self._log_message("   Data will be loaded/processed during analysis")

                # Ensure merged_results is available for analysis
                if not hasattr(self, "merged_results") or not self.merged_results:
                    self._log_message(
                        "   No processed data available - analysis will process from file"
                    )

            # Restore masks and labeled frame
            if hasattr(self, "main_masks"):
                self.masks = self.main_masks.copy()
            if hasattr(self, "main_labeled_frame"):
                self.labeled_frame = self.main_labeled_frame

            # Update UI to reflect main dataset
            self.lbl_file_info.setText(
                f"MAIN DATASET: {os.path.basename(self.file_path)}"
            )

            return True

        except Exception as e:
            self._log_message(f"ERROR restoring main dataset: {e}")
            import traceback

            self._log_message(f"Traceback: {traceback.format_exc()}")
            return False

    def setup_ui(self):
        """Setup the user interface with all tabs."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_input = QWidget()
        self.tab_analysis = QWidget()
        self.tab_results = QWidget()
        self.tab_extended = QWidget()
        self.tab_viewer = QWidget()

        self.tab_widget.addTab(self.tab_input, "Input")
        self.tab_widget.addTab(self.tab_analysis, "Analysis")
        self.tab_widget.addTab(self.tab_results, "Results")
        self.tab_widget.addTab(self.tab_extended, "Extended Analysis")
        self.tab_widget.addTab(self.tab_viewer, "Frame Viewer")

        layout.addWidget(self.tab_widget)

        # Setup individual tabs
        self.setup_input_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
        self.setup_extended_tab()
        self.setup_viewer_tab()

    def setup_input_tab(self):
        """Setup the input tab with file loading and ROI detection parameters."""
        layout = QVBoxLayout()
        self.tab_input.setLayout(layout)

        # File loading section
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        # Debug-Button hinzufügen
        self.btn_debug_structure = QPushButton("Debug HDF5 Structure")
        self.btn_debug_structure.setToolTip("Analyze HDF5 file structure")
        self.btn_debug_structure.clicked.connect(self.debug_current_file_structure)
        file_layout.addWidget(self.btn_debug_structure)
        # File loading buttons
        self.btn_load_file = QPushButton("Load File")
        self.btn_load_file.setToolTip("Load HDF5 file or AVI video(s) for analysis")

        self.btn_load_dir = QPushButton("Load Directory")
        self.btn_load_dir.setToolTip("Load all HDF5/AVI files from a directory")

        self.btn_detect_rois = QPushButton("Detect ROIs")
        self.btn_detect_rois.setToolTip(
            "Automatically detect circular ROIs using HoughCircles"
        )

        self.btn_clear_rois = QPushButton("Clear ROI Detection")
        self.btn_clear_rois.setToolTip("Remove ROI detection layers")

        file_layout.addWidget(self.btn_load_file)
        file_layout.addWidget(self.btn_load_dir)
        file_layout.addWidget(self.btn_detect_rois)
        file_layout.addWidget(self.btn_clear_rois)

        self.lbl_file_info = QLabel("No file loaded")
        file_layout.addWidget(self.lbl_file_info)
        layout.addWidget(file_group)

        # ROI Detection Parameters
        roi_group = QGroupBox("ROI Detection Parameters")
        roi_layout = QFormLayout()
        roi_group.setLayout(roi_layout)

        # Radius parameters
        self.min_radius = QSpinBox()
        self.min_radius.setRange(10, 1000)
        self.min_radius.setValue(380)
        self.min_radius.setToolTip("Minimum radius for circle detection")
        roi_layout.addRow("Min Radius:", self.min_radius)

        self.max_radius = QSpinBox()
        self.max_radius.setRange(10, 1000)
        self.max_radius.setValue(420)
        self.max_radius.setToolTip("Maximum radius for circle detection")
        roi_layout.addRow("Max Radius:", self.max_radius)

        # HoughCircles parameters
        self.dp_param = QDoubleSpinBox()
        self.dp_param.setRange(0.1, 5.0)
        self.dp_param.setValue(0.5)
        self.dp_param.setSingleStep(0.1)
        self.dp_param.setDecimals(1)
        self.dp_param.setToolTip(
            "Inverse ratio of accumulator resolution to image resolution"
        )
        roi_layout.addRow("DP Parameter:", self.dp_param)

        self.min_dist = QSpinBox()
        self.min_dist.setRange(10, 1000)
        self.min_dist.setValue(150)
        self.min_dist.setToolTip("Minimum distance between circle centers")
        roi_layout.addRow("Min Distance:", self.min_dist)

        self.param1 = QSpinBox()
        self.param1.setRange(10, 200)
        self.param1.setValue(40)
        self.param1.setToolTip("Upper threshold for edge detection in Canny")
        roi_layout.addRow("Param1 (Edge):", self.param1)

        self.param2 = QSpinBox()
        self.param2.setRange(10, 200)
        self.param2.setValue(40)
        self.param2.setToolTip("Accumulator threshold for center detection")
        roi_layout.addRow("Param2 (Center):", self.param2)

        # 12-Well plate preset
        self.chk_12well = QCheckBox("12-Well Plate Preset")
        self.chk_12well.setToolTip("Use preset values for 12-well plates")
        roi_layout.addRow("", self.chk_12well)

        layout.addWidget(roi_group)
        layout.addStretch()

    def setup_analysis_tab(self):
        """Setup the analysis tab with threshold calculation methods."""
        layout = QVBoxLayout()
        self.tab_analysis.setLayout(layout)

        # Basic Analysis Parameters
        analysis_group = QGroupBox("Basic Analysis Parameters")
        analysis_layout = QFormLayout()
        analysis_group.setLayout(analysis_layout)

        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(0.01, 60.0)
        self.frame_interval.setValue(5.0)
        self.frame_interval.setSingleStep(0.1)
        self.frame_interval.setToolTip("Time interval between frames in seconds")
        analysis_layout.addRow("Frame Interval (s):", self.frame_interval)

        self.time_end = QSpinBox()
        self.time_end.setRange(0, 1000000)
        self.time_end.setValue(0)
        self.time_end.setToolTip("End time for analysis (0 = use full duration)")
        analysis_layout.addRow("End Time (s):", self.time_end)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1, 10000)
        self.chunk_size.setValue(20)
        self.chunk_size.setToolTip("Number of frames to process in each chunk")
        analysis_layout.addRow("Chunk Size:", self.chunk_size)

        self.num_processes = QSpinBox()
        self.num_processes.setRange(1, self.cpu_count)
        self.num_processes.setValue(self.optimal_processes)
        self.num_processes.setToolTip(
            f"Number of parallel processes (recommended: {self.optimal_processes})"
        )
        analysis_layout.addRow("Number of Processes:", self.num_processes)

        layout.addWidget(analysis_group)

        # Threshold Calculation Methods
        threshold_group = QGroupBox("Threshold Calculation Method")
        threshold_layout = QVBoxLayout()
        threshold_group.setLayout(threshold_layout)

        # Method-specific parameters (Tabs control the method selection)
        self.threshold_params_stack = QTabWidget()
        threshold_layout.addWidget(self.threshold_params_stack)

        # === METHOD 1: BASELINE ===
        baseline_tab = QWidget()
        baseline_layout = QFormLayout()
        baseline_tab.setLayout(baseline_layout)

        self.baseline_duration_minutes = QDoubleSpinBox()
        self.baseline_duration_minutes.setRange(1.0, 10000000000.0)
        self.baseline_duration_minutes.setValue(200.0)
        self.baseline_duration_minutes.setSingleStep(1.0)
        self.baseline_duration_minutes.setDecimals(1)
        self.baseline_duration_minutes.setToolTip(
            "Duration of baseline period in minutes"
        )
        baseline_layout.addRow(
            "Baseline Duration (min):", self.baseline_duration_minutes
        )

        self.threshold_multiplier = QDoubleSpinBox()
        self.threshold_multiplier.setRange(0.0, 5.0)
        self.threshold_multiplier.setValue(0.1)
        self.threshold_multiplier.setSingleStep(0.1)
        self.threshold_multiplier.setToolTip(
            "Multiplier for hysteresis band (mean ± multiplier × std)"
        )
        baseline_layout.addRow("Threshold Multiplier:", self.threshold_multiplier)

        # Add detrending option
        self.enable_detrending = QCheckBox("Enable Detrending")
        self.enable_detrending.setChecked(False)
        self.enable_detrending.setToolTip(
            "Remove linear drift from baseline period for more accurate thresholds"
        )
        baseline_layout.addRow("", self.enable_detrending)

        # Add jump correction option
        self.enable_jump_correction = QCheckBox("Enable Jump Correction")
        self.enable_jump_correction.setChecked(False)
        self.enable_jump_correction.setToolTip(
            "Detect and correct sudden jumps/plateaus in baseline data"
        )
        baseline_layout.addRow("", self.enable_jump_correction)

        baseline_info = QLabel(
            "HYSTERESIS METHOD:\n"
            "Uses hysteresis band to prevent flicker.\n"
            "Signal > Upper → Movement = TRUE\n"
            "Signal < Lower → Movement = FALSE\n"
            "Signal between → State unchanged"
        )
        baseline_info.setStyleSheet("color: #666; font-size: 10px;")
        baseline_info.setWordWrap(True)
        baseline_layout.addRow("", baseline_info)

        self.threshold_params_stack.addTab(baseline_tab, "Baseline Method")

        # === METHOD 2: CALIBRATION ===
        calibration_tab = QWidget()
        calibration_layout = QFormLayout()
        calibration_tab.setLayout(calibration_layout)

        # Calibration file selection (existing code)
        cal_file_layout = QHBoxLayout()
        self.calibration_file_path = QLabel("No calibration file selected")
        self.calibration_file_path.setStyleSheet(
            """
            QLabel {
                border: 1px solid #ccc;
                padding: 4px;
                background: #f9f9f9;
                color: #000000;  /* Force black text */
            }
        """
        )
        self.btn_load_calibration = QPushButton("Browse...")
        cal_file_layout.addWidget(self.calibration_file_path, 3)
        cal_file_layout.addWidget(self.btn_load_calibration, 1)
        calibration_layout.addRow("Calibration File:", cal_file_layout)

        # Calibration multiplier (existing code)
        self.calibration_multiplier = QDoubleSpinBox()
        self.calibration_multiplier.setRange(0.01, 5.00)
        self.calibration_multiplier.setValue(1.00)
        self.calibration_multiplier.setSingleStep(0.01)
        self.calibration_multiplier.setDecimals(2)
        self.calibration_multiplier.setToolTip(
            "Multiplier applied to calibration std (same as baseline multiplier)"
        )
        calibration_layout.addRow(
            "Calibration Multiplier:", self.calibration_multiplier
        )

        # NEW: Calibration dataset processing controls
        cal_processing_layout = QVBoxLayout()
        # Load calibration dataset button
        self.btn_load_calibration_dataset = QPushButton("Load Calibration Dataset")
        self.btn_load_calibration_dataset.setToolTip(
            "Load selected calibration file into viewer for ROI detection"
        )
        self.btn_load_calibration_dataset.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
        )
        self.btn_load_calibration_dataset.setEnabled(
            False
        )  # Enabled when file is selected
        # Process calibration baseline button
        self.btn_process_calibration_baseline = QPushButton(
            "Process Calibration Baseline"
        )
        self.btn_process_calibration_baseline.setToolTip(
            "Process full calibration dataset to create baseline statistics"
        )
        self.btn_process_calibration_baseline.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; }"
        )
        self.btn_process_calibration_baseline.setEnabled(
            False
        )  # Enabled when calibration ROIs detected

        cal_processing_layout.addWidget(self.btn_load_calibration_dataset)
        cal_processing_layout.addWidget(self.btn_process_calibration_baseline)

        calibration_layout.addRow("Calibration Processing:", cal_processing_layout)
        # NEW: Calibration status display
        self.calibration_status_label = QLabel(
            "1. Select calibration file\n2. Load calibration dataset\n3. Detect ROIs (Input tab)\n4. Process baseline"
        )
        self.calibration_status_label.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 10px;
                color: #000000;  /* Force black text */
            }
        """
        )
        self.calibration_status_label.setWordWrap(True)
        calibration_layout.addRow("Status:", self.calibration_status_label)
        # Updated info text
        calibration_info = QLabel(
            "CALIBRATION METHOD:\n"
            "Uses sedated animals to determine noise baseline.\n"
            "Calculates: mean ± multiplier × std from complete calibration dataset.\n"
            "Same hysteresis formula as baseline method."
        )
        calibration_info.setStyleSheet("color: #666; font-size: 10px;")
        calibration_info.setWordWrap(True)
        calibration_layout.addRow("", calibration_info)

        self.threshold_params_stack.addTab(calibration_tab, "Calibration Method")

        # === METHOD 3: ADAPTIVE ===
        adaptive_tab = QWidget()
        adaptive_layout = QFormLayout()
        adaptive_tab.setLayout(adaptive_layout)

        self.adaptive_duration_minutes = QDoubleSpinBox()
        self.adaptive_duration_minutes.setRange(5.0, 120.0)
        self.adaptive_duration_minutes.setValue(15.0)
        self.adaptive_duration_minutes.setSingleStep(1.0)
        self.adaptive_duration_minutes.setDecimals(1)
        self.adaptive_duration_minutes.setToolTip(
            "Duration of initial period for adaptive analysis"
        )
        adaptive_layout.addRow(
            "Analysis Duration (min):", self.adaptive_duration_minutes
        )

        self.adaptive_base_multiplier = QDoubleSpinBox()
        self.adaptive_base_multiplier.setRange(1.0, 5.0)
        self.adaptive_base_multiplier.setValue(2.5)
        self.adaptive_base_multiplier.setSingleStep(0.1)
        self.adaptive_base_multiplier.setToolTip(
            "Base multiplier for adaptive calculation"
        )
        adaptive_layout.addRow("Base Multiplier:", self.adaptive_base_multiplier)

        adaptive_info = QLabel(
            "Automatically adapts threshold based on signal-to-noise ratio."
        )
        adaptive_info.setStyleSheet("color: #666; font-size: 10px;")
        adaptive_info.setWordWrap(True)
        adaptive_layout.addRow("", adaptive_info)

        self.threshold_params_stack.addTab(adaptive_tab, "Adaptive Method")

        layout.addWidget(threshold_group)

        # Behavior Analysis Parameters
        behavior_group = QGroupBox("Behavior Analysis Parameters")
        behavior_layout = QFormLayout()
        behavior_group.setLayout(behavior_layout)

        self.bin_size_seconds = QSpinBox()
        self.bin_size_seconds.setRange(1, 300)
        self.bin_size_seconds.setValue(60)
        self.bin_size_seconds.setToolTip(
            "Bin size for fraction movement (60s recommended)"
        )
        behavior_layout.addRow("Bin Size (seconds):", self.bin_size_seconds)

        self.quiescence_threshold = QDoubleSpinBox()
        self.quiescence_threshold.setRange(0.0, 1.0)
        self.quiescence_threshold.setValue(0.5)
        self.quiescence_threshold.setSingleStep(0.1)
        self.quiescence_threshold.setToolTip("Quiescence threshold (0.5 recommended)")
        behavior_layout.addRow("Quiescence Threshold:", self.quiescence_threshold)

        self.sleep_threshold_minutes = QSpinBox()
        self.sleep_threshold_minutes.setRange(1, 60)
        self.sleep_threshold_minutes.setValue(8)
        self.sleep_threshold_minutes.setToolTip(
            "Minimum sleep duration in minutes (8 recommended)"
        )
        behavior_layout.addRow("Sleep Threshold (min):", self.sleep_threshold_minutes)

        layout.addWidget(behavior_group)

        # Analysis Control Section
        control_group = QGroupBox("Analysis Control")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)
        reset_layout = QHBoxLayout()
        self.btn_reset_analysis = QPushButton("Reset for New Analysis")
        self.btn_reset_analysis.setToolTip(
            "Clear all data and reset for a new analysis"
        )
        self.btn_reset_analysis.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; }"
        )
        reset_layout.addWidget(self.btn_reset_analysis)
        control_layout.addLayout(reset_layout)
        # Analysis buttons
        btn_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Start Analysis")
        self.btn_analyze.setToolTip("Start the analysis with current parameters")
        self.btn_analyze.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        self.btn_stop = QPushButton("Stop Analysis")
        self.btn_stop.setToolTip("Stop the current analysis")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; }"
        )

        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_stop)
        control_layout.addLayout(btn_layout)

        # Testing and diagnostics buttons
        test_layout = QHBoxLayout()
        self.btn_quick_test = QPushButton("Quick Test")
        self.btn_quick_test.setToolTip("Run quick analysis test using _calc.py")
        self.btn_validate_timing = QPushButton("Validate HDF5 Timing")
        self.btn_validate_timing.setToolTip("Check HDF5 timing using _calc.py")

        test_layout.addWidget(self.btn_quick_test)
        test_layout.addWidget(self.btn_validate_timing)
        control_layout.addLayout(test_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        # Remove extra spacing
        control_layout.setSpacing(0)
        control_layout.setContentsMargins(0, 0, 0, 0)
        # Status label
        self.status_label = QLabel("Ready to start analysis")
        self.status_label.setStyleSheet(
            "QLabel { padding: 5px; background-color: #2b2b2b; border: 1px solid #555; color: #ffffff; }"
        )
        control_layout.addWidget(self.status_label)

        # Performance metrics label
        self.performance_label = QLabel(
            "Performance metrics will appear here during analysis"
        )
        self.performance_label.setStyleSheet(
            "QLabel { padding: 3px; font-size: 10px; color: #FFFFFF; }"
        )
        control_layout.addWidget(self.performance_label)

        layout.addWidget(control_group)

        # Analysis Log Section
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #000000;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 9px;
            }
        """
        )
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

    def setup_results_tab(self):
        """Setup the results tab with plotting and export options."""
        layout = QVBoxLayout()
        self.tab_results.setLayout(layout)

        self.results_label = QLabel("Results will be displayed here.")
        layout.addWidget(self.results_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        try:
            from ._plot import PlotGenerator

            self.plot_generator = PlotGenerator(self.figure)
            self._log_message("✅ Plot generator initialized")
        except Exception as e:
            self.plot_generator = None
            self._log_message(f"⚠️ Plot generator initialization failed: {e}")

        # Threshold Visualization Options
        viz_group = QGroupBox("Threshold Visualization (for Raw Intensity Plots)")
        viz_layout = QFormLayout()
        viz_group.setLayout(viz_layout)

        self.show_baseline_mean = QCheckBox("Show Baseline Mean Line")
        self.show_baseline_mean.setChecked(True)
        self.show_baseline_mean.setToolTip(
            "Show the baseline mean from analysis (red line)"
        )
        viz_layout.addRow("", self.show_baseline_mean)

        self.show_deviation_band = QCheckBox("Show Deviation Band (Hysteresis Zone)")
        self.show_deviation_band.setChecked(True)
        self.show_deviation_band.setToolTip(
            "Show ±σ band around baseline mean (orange area)"
        )
        viz_layout.addRow("", self.show_deviation_band)

        self.show_detection_threshold = QCheckBox("Show Detection Thresholds")
        self.show_detection_threshold.setChecked(True)
        self.show_detection_threshold.setToolTip(
            "Show upper/lower detection boundaries (dashed lines)"
        )
        viz_layout.addRow("", self.show_detection_threshold)

        self.show_threshold_stats = QCheckBox("Show Threshold Statistics")
        self.show_threshold_stats.setChecked(True)
        self.show_threshold_stats.setToolTip(
            "Show threshold calculation details on plot"
        )
        viz_layout.addRow("", self.show_threshold_stats)

        # INFO BOX - SCIENTIFIC EXPLANATION
        baseline_info = QLabel(
            "BASELINE REFERENCE:\n"
            "• Baseline Mean = Fixed reference from analysis baseline period\n"
            "• Detection Thresholds = Used in actual movement detection\n"
            "• These values NEVER change with time range selection\n"
            "• They represent the analysis parameters, not visible data statistics"
        )
        baseline_info.setStyleSheet(
            "color: #0066cc; font-size: 9px; background: #f0f8ff; "
            "padding: 8px; border: 1px solid #ccc; border-radius: 4px;"
        )
        baseline_info.setWordWrap(True)
        viz_layout.addRow("", baseline_info)

        layout.addWidget(viz_group)

        # Plot configuration
        plot_config_group = QGroupBox("Plot Configuration")
        plot_config_layout = QVBoxLayout()
        plot_config_group.setLayout(plot_config_layout)

        # Plot type and basic controls
        basic_row = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            [
                "Raw Intensity Changes",
                "Movement",
                "Fraction Movement",
                "Quiescence",
                "Sleep",
                "Lighting Conditions (dark IR)",
            ]
        )

        self.plot_dpi_spin = QSpinBox()
        self.plot_dpi_spin.setRange(50, 600)
        self.plot_dpi_spin.setValue(100)

        basic_row.addWidget(QLabel("Plot Type:"))
        basic_row.addWidget(self.plot_type_combo)
        basic_row.addWidget(QLabel("DPI:"))
        basic_row.addWidget(self.plot_dpi_spin)
        basic_row.addStretch()
        plot_config_layout.addLayout(basic_row)

        # Figure size controls
        size_row = QHBoxLayout()

        self.plot_width_spin = QDoubleSpinBox()
        self.plot_width_spin.setRange(1.0, 100.0)
        self.plot_width_spin.setValue(10.0)
        self.plot_width_spin.setSingleStep(0.5)

        self.plot_height_spin = QDoubleSpinBox()
        self.plot_height_spin.setRange(0.1, 10.0)
        self.plot_height_spin.setValue(0.6)
        self.plot_height_spin.setSingleStep(0.1)

        size_row.addWidget(QLabel("Figure Width:"))
        size_row.addWidget(self.plot_width_spin)
        size_row.addWidget(QLabel("Height Per ROI:"))
        size_row.addWidget(self.plot_height_spin)
        size_row.addStretch()
        plot_config_layout.addLayout(size_row)

        # Y-Axis scaling controls
        y_axis_group = QGroupBox("Y-Axis Scaling (Per ROI Optimization)")
        y_axis_layout = QVBoxLayout()
        y_axis_group.setLayout(y_axis_layout)

        scaling_mode_layout = QHBoxLayout()
        self.auto_scale_y = QCheckBox("Auto Scale Y-Axis (Recommended)")
        self.auto_scale_y.setChecked(True)
        self.auto_scale_y.setToolTip(
            "Automatically optimize Y-axis for each ROI individually"
        )

        self.robust_scaling = QCheckBox("Robust Scaling (Ignore Outliers)")
        self.robust_scaling.setChecked(True)
        self.robust_scaling.setToolTip(
            "Use percentile-based scaling to ignore outliers and focus on main data"
        )

        scaling_mode_layout.addWidget(self.auto_scale_y)
        scaling_mode_layout.addWidget(self.robust_scaling)
        scaling_mode_layout.addStretch()
        y_axis_layout.addLayout(scaling_mode_layout)

        # Advanced scaling options
        advanced_layout = QHBoxLayout()

        self.adaptive_scaling = QCheckBox("Adaptive Scaling")
        self.adaptive_scaling.setChecked(True)
        self.adaptive_scaling.setToolTip(
            "Automatically adjust scaling strategy based on data characteristics\n"
            "• Low variance data: Tighter scaling to show small changes\n"
            "• Outlier-heavy data: More aggressive filtering\n"
            "• Sparse data: Optimize for non-zero values"
        )

        self.center_around_zero = QCheckBox("Smart Zero Centering")
        self.center_around_zero.setChecked(True)
        self.center_around_zero.setToolTip(
            "Include zero in view when data is centered around zero"
        )

        advanced_layout.addWidget(self.adaptive_scaling)
        advanced_layout.addWidget(self.center_around_zero)
        advanced_layout.addStretch()
        y_axis_layout.addLayout(advanced_layout)

        # Manual Y-axis range controls
        manual_range_layout = QHBoxLayout()

        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1e9, 1e9)
        self.y_min_spin.setValue(0.0)
        self.y_min_spin.setEnabled(False)

        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1e9, 1e9)
        self.y_max_spin.setValue(1000.0)
        self.y_max_spin.setEnabled(False)

        self.btn_apply_y_range = QPushButton("Apply Manual Range")
        self.btn_apply_y_range.setEnabled(False)
        self.btn_apply_y_range.setToolTip(
            "Use manual Y-axis range instead of automatic optimization"
        )

        manual_range_layout.addWidget(QLabel("Manual Y Min:"))
        manual_range_layout.addWidget(self.y_min_spin)
        manual_range_layout.addWidget(QLabel("Y Max:"))
        manual_range_layout.addWidget(self.y_max_spin)
        manual_range_layout.addWidget(self.btn_apply_y_range)
        manual_range_layout.addStretch()
        y_axis_layout.addLayout(manual_range_layout)

        # Percentile controls for robust scaling
        percentile_layout = QHBoxLayout()

        self.lower_percentile_spin = QDoubleSpinBox()
        self.lower_percentile_spin.setRange(0.0, 50.0)
        self.lower_percentile_spin.setValue(5.0)
        self.lower_percentile_spin.setSingleStep(1.0)

        self.upper_percentile_spin = QDoubleSpinBox()
        self.upper_percentile_spin.setRange(50.0, 100.0)
        self.upper_percentile_spin.setValue(95.0)
        self.upper_percentile_spin.setSingleStep(1.0)

        percentile_layout.addWidget(QLabel("Lower %:"))
        percentile_layout.addWidget(self.lower_percentile_spin)
        percentile_layout.addWidget(QLabel("Upper %:"))
        percentile_layout.addWidget(self.upper_percentile_spin)
        percentile_layout.addStretch()
        y_axis_layout.addLayout(percentile_layout)

        plot_config_layout.addWidget(y_axis_group)

        # Time range selection
        time_range_group = QGroupBox("Time Range Selection")
        time_range_layout = QHBoxLayout()
        time_range_group.setLayout(time_range_layout)

        self.plot_start_time = QDoubleSpinBox()
        self.plot_start_time.setRange(0.0, 1e9)
        self.plot_start_time.setValue(0.0)
        self.plot_start_time.setSuffix(" min")
        self.plot_end_time = QDoubleSpinBox()
        self.plot_end_time.setRange(0.0, 1e9)
        self.plot_end_time.setValue(100000.0)
        self.plot_end_time.setSuffix(" min")
        self.btn_apply_time_range = QPushButton("Apply Time Range")

        time_range_layout.addWidget(QLabel("Start Time (min):"))
        time_range_layout.addWidget(self.plot_start_time)
        time_range_layout.addWidget(QLabel("End Time (min):"))
        time_range_layout.addWidget(self.plot_end_time)
        time_range_layout.addWidget(self.btn_apply_time_range)

        layout.addWidget(time_range_group)
        layout.addWidget(plot_config_group)

        # ===== SIMPLIFIED PLOT CONTROLS =====
        plot_buttons_group = QGroupBox("Plot Controls")
        plot_buttons_layout = QHBoxLayout()
        plot_buttons_group.setLayout(plot_buttons_layout)

        # Core plotting buttons
        self.btn_plot = QPushButton("Generate Plot")
        self.btn_plot.setToolTip("Generate the selected plot type")
        self.btn_plot.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        self.btn_save_plot = QPushButton("Save Current Plot")
        self.btn_save_plot.setToolTip("Save the currently displayed plot as image file")

        self.btn_save_all_plots = QPushButton("Save All Plots")
        self.btn_save_all_plots.setToolTip(
            "Save all plot types to separate image files"
        )

        # CONSOLIDATED save results button
        self.btn_save_results = QPushButton("Save Results")
        self.btn_save_results.setToolTip(
            "Save analysis results (CSV + Excel + threshold stats)"
        )
        self.btn_save_results.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
        )
        self.btn_save_with_metadata = QPushButton("Save with HDF5 Metadata")
        self.btn_save_with_metadata.setToolTip(
            "Save analysis results including comprehensive HDF5 metadata"
        )
        self.btn_save_with_metadata.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        plot_buttons_layout.addWidget(self.btn_plot)
        plot_buttons_layout.addWidget(self.btn_save_plot)
        plot_buttons_layout.addWidget(self.btn_save_all_plots)

        plot_buttons_layout.addWidget(self.btn_save_results)
        plot_buttons_layout.addWidget(self.btn_save_with_metadata)
        layout.addWidget(plot_buttons_group)

    def setup_extended_tab(self):
        """Setup the Extended Analysis tab for circadian rhythm detection."""
        layout = QVBoxLayout()
        self.tab_extended.setLayout(layout)

        # Title and description
        title_label = QLabel("Circadian Rhythm Analysis")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        desc_label = QLabel(
            "Use Fischer Z-transformation to detect recurring sleep/wake patterns "
            "and circadian rhythms in activity data."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        layout.addWidget(desc_label)

        # Fischer Z-transformation parameters
        fisher_params_group = QGroupBox("Fischer Z-Transformation Parameters")
        fisher_params_layout = QFormLayout()
        fisher_params_group.setLayout(fisher_params_layout)

        self.fisher_min_period = QDoubleSpinBox()
        self.fisher_min_period.setRange(0.0, 100.0)
        self.fisher_min_period.setValue(12.0)
        self.fisher_min_period.setSingleStep(1.0)
        self.fisher_min_period.setSuffix(" hours")
        self.fisher_min_period.setToolTip(
            "Minimum period to test (e.g., 12 hours for semi-circadian)"
        )
        fisher_params_layout.addRow("Minimum Period:", self.fisher_min_period)

        self.fisher_max_period = QDoubleSpinBox()
        self.fisher_max_period.setRange(0.0, 100.0)
        self.fisher_max_period.setValue(36.0)
        self.fisher_max_period.setSingleStep(1.0)
        self.fisher_max_period.setSuffix(" hours")
        self.fisher_max_period.setToolTip(
            "Maximum period to test (e.g., 36 hours for extended circadian)"
        )
        fisher_params_layout.addRow("Maximum Period:", self.fisher_max_period)

        self.fisher_significance = QDoubleSpinBox()
        self.fisher_significance.setRange(0.001, 0.1)
        self.fisher_significance.setValue(0.05)
        self.fisher_significance.setSingleStep(0.01)
        self.fisher_significance.setDecimals(3)
        self.fisher_significance.setToolTip(
            "Statistical significance threshold (p-value)"
        )
        fisher_params_layout.addRow("Significance Level (α):", self.fisher_significance)

        self.fisher_phase_threshold = QDoubleSpinBox()
        self.fisher_phase_threshold.setRange(0.0, 1.0)
        self.fisher_phase_threshold.setValue(0.5)
        self.fisher_phase_threshold.setSingleStep(0.05)
        self.fisher_phase_threshold.setDecimals(2)
        self.fisher_phase_threshold.setToolTip(
            "Threshold for classifying sleep vs wake phases (0-1)"
        )
        fisher_params_layout.addRow("Phase Threshold:", self.fisher_phase_threshold)

        layout.addWidget(fisher_params_group)

        # Run analysis button
        self.btn_run_fisher = QPushButton("Run Circadian Analysis")
        self.btn_run_fisher.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 10px; } QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_run_fisher.clicked.connect(self.run_fisher_analysis)
        layout.addWidget(self.btn_run_fisher)

        # Create a horizontal splitter for results and plot
        splitter = QSplitter()
        splitter.setOrientation(1)  # Horizontal

        # Results display (left side)
        self.fisher_results_text = QTextEdit()
        self.fisher_results_text.setReadOnly(True)
        self.fisher_results_text.setPlaceholderText(
            "Circadian analysis results will appear here..."
        )
        splitter.addWidget(self.fisher_results_text)

        # Plot display (right side)
        self.fisher_plot_widget = QWidget()
        fisher_plot_layout = QVBoxLayout()
        self.fisher_plot_widget.setLayout(fisher_plot_layout)

        fisher_plot_label = QLabel("Periodogram Plot")
        fisher_plot_label.setStyleSheet("font-weight: bold;")
        fisher_plot_layout.addWidget(fisher_plot_label)

        self.fisher_plot_canvas = QLabel()
        self.fisher_plot_canvas.setMinimumSize(400, 300)
        self.fisher_plot_canvas.setStyleSheet(
            "border: 1px solid #ccc; background-color: white;"
        )
        self.fisher_plot_canvas.setAlignment(Qt.AlignCenter)
        fisher_plot_layout.addWidget(self.fisher_plot_canvas)

        splitter.addWidget(self.fisher_plot_widget)

        # Set initial sizes (60% results, 40% plot)
        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

        # Export button
        self.btn_export_fisher = QPushButton("Export Circadian Results")
        self.btn_export_fisher.clicked.connect(self.export_fisher_results)
        self.btn_export_fisher.setEnabled(False)
        layout.addWidget(self.btn_export_fisher)

        layout.addStretch()

    def setup_viewer_tab(self):
        """Setup the Frame Viewer tab for browsing through dataset frames."""
        layout = QVBoxLayout()
        self.tab_viewer.setLayout(layout)

        # Title and description
        title_label = QLabel("Frame Viewer")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        desc_label = QLabel(
            "Browse through the loaded dataset frame-by-frame. "
            "Use the slider or keyboard shortcuts to navigate."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        layout.addWidget(desc_label)

        # Status label
        self.viewer_status_label = QLabel(
            "No dataset loaded. Please load a file in the Input tab first."
        )
        self.viewer_status_label.setStyleSheet("color: #999; font-style: italic;")
        layout.addWidget(self.viewer_status_label)

        # Frame navigation controls
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout()
        nav_group.setLayout(nav_layout)

        # Frame slider with current frame display
        slider_layout = QHBoxLayout()
        self.viewer_frame_slider = QSlider()
        self.viewer_frame_slider.setOrientation(1)  # Horizontal
        self.viewer_frame_slider.setMinimum(0)
        self.viewer_frame_slider.setMaximum(0)
        self.viewer_frame_slider.setValue(0)
        self.viewer_frame_slider.setEnabled(False)
        self.viewer_frame_slider.valueChanged.connect(self._on_viewer_frame_changed)

        self.viewer_frame_label = QLabel("Frame: 0 / 0")
        self.viewer_frame_label.setMinimumWidth(120)

        slider_layout.addWidget(self.viewer_frame_label)
        slider_layout.addWidget(self.viewer_frame_slider)
        nav_layout.addLayout(slider_layout)

        # Playback controls
        playback_layout = QHBoxLayout()

        self.btn_viewer_first = QPushButton("|◀")
        self.btn_viewer_first.setToolTip("First frame (Home)")
        self.btn_viewer_first.clicked.connect(lambda: self._viewer_goto_frame(0))
        self.btn_viewer_first.setEnabled(False)

        self.btn_viewer_prev = QPushButton("◀")
        self.btn_viewer_prev.setToolTip("Previous frame (←)")
        self.btn_viewer_prev.clicked.connect(lambda: self._viewer_step_frame(-1))
        self.btn_viewer_prev.setEnabled(False)

        self.btn_viewer_play = QPushButton("▶ Play")
        self.btn_viewer_play.setToolTip("Play/Pause (Space)")
        self.btn_viewer_play.setCheckable(True)
        self.btn_viewer_play.clicked.connect(self._viewer_toggle_play)
        self.btn_viewer_play.setEnabled(False)

        self.btn_viewer_next = QPushButton("▶")
        self.btn_viewer_next.setToolTip("Next frame (→)")
        self.btn_viewer_next.clicked.connect(lambda: self._viewer_step_frame(1))
        self.btn_viewer_next.setEnabled(False)

        self.btn_viewer_last = QPushButton("▶|")
        self.btn_viewer_last.setToolTip("Last frame (End)")
        self.btn_viewer_last.clicked.connect(lambda: self._viewer_goto_frame(-1))
        self.btn_viewer_last.setEnabled(False)

        playback_layout.addWidget(self.btn_viewer_first)
        playback_layout.addWidget(self.btn_viewer_prev)
        playback_layout.addWidget(self.btn_viewer_play)
        playback_layout.addWidget(self.btn_viewer_next)
        playback_layout.addWidget(self.btn_viewer_last)
        playback_layout.addStretch()

        nav_layout.addLayout(playback_layout)

        # Playback speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Playback FPS:"))

        self.viewer_fps_spin = QSpinBox()
        self.viewer_fps_spin.setRange(1, 60)
        self.viewer_fps_spin.setValue(10)
        self.viewer_fps_spin.setToolTip("Frames per second during playback")
        self.viewer_fps_spin.valueChanged.connect(self._viewer_update_timer_interval)

        speed_layout.addWidget(self.viewer_fps_spin)
        speed_layout.addStretch()
        nav_layout.addLayout(speed_layout)

        layout.addWidget(nav_group)

        # Load data button
        self.btn_viewer_load = QPushButton("Load Current Dataset into Viewer")
        self.btn_viewer_load.clicked.connect(self._viewer_load_data)
        self.btn_viewer_load.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 10px; } QPushButton:hover { background-color: #1976D2; }"
        )
        layout.addWidget(self.btn_viewer_load)

        # Info display
        self.viewer_info_text = QTextEdit()
        self.viewer_info_text.setReadOnly(True)
        self.viewer_info_text.setMaximumHeight(150)
        self.viewer_info_text.setPlaceholderText(
            "Frame information will appear here..."
        )
        layout.addWidget(self.viewer_info_text)

        layout.addStretch()

        # Timer for playback
        from qtpy.QtCore import QTimer

        self.viewer_timer = QTimer()
        self.viewer_timer.timeout.connect(self._viewer_play_next_frame)
        self.viewer_is_playing = False

    def debug_current_file_structure(self):
        """Debug the structure of the currently loaded file."""
        if not hasattr(self, "file_path") or not self.file_path:
            self._log_message("No file loaded for structure debugging")
            return

        self._log_message("=== DEBUGGING CURRENT FILE STRUCTURE ===")

        if DUAL_STRUCTURE_AVAILABLE:
            try:
                structure_info = detect_hdf5_structure_type(self.file_path)

                self._log_message(f"Structure type: {structure_info['type']}")

                if structure_info["type"] == "stacked_frames":
                    self._log_message("✅ Stacked frames detected")
                    self._log_message(f"   Dataset: '{structure_info['dataset_name']}'")
                    self._log_message(
                        f"   Frame count: {structure_info['frame_count']}"
                    )
                    self._log_message(
                        f"   Frame shape: {structure_info['frame_shape']}"
                    )
                    self._log_message(f"   Data type: {structure_info['dtype']}")

                elif structure_info["type"] == "individual_frames":
                    self._log_message("✅ Individual frames detected")
                    self._log_message(f"   Group: '{structure_info['group_name']}'")
                    self._log_message(
                        f"   Frame count: {structure_info['frame_count']}"
                    )
                    self._log_message(
                        f"   Frame shape: {structure_info['frame_shape']}"
                    )
                    self._log_message(f"   Data type: {structure_info['dtype']}")

                    # Show sample frame keys
                    if "frame_keys" in structure_info:
                        sample_keys = structure_info["frame_keys"][:10]
                        self._log_message(f"   Sample keys: {sample_keys}")
                        if len(structure_info["frame_keys"]) > 10:
                            self._log_message(
                                f"   ... and {len(structure_info['frame_keys']) - 10} more"
                            )

                elif structure_info["type"] == "error":
                    self._log_message(f"❌ Error: {structure_info['error']}")

                    # Fallback to basic structure info
                    try:
                        import h5py

                        with h5py.File(self.file_path, "r") as f:
                            self._log_message(f"Available keys: {list(f.keys())}")
                    except Exception as e2:
                        self._log_message(f"Cannot read file: {e2}")

            except Exception as e:
                self._log_message(f"Structure debugging failed: {e}")
        else:
            self._log_message(
                "Dual structure support not available - using basic debugging"
            )
            try:
                import h5py

                with h5py.File(self.file_path, "r") as f:
                    self._log_message(f"Root keys: {list(f.keys())}")

                    if "frames" in f:
                        self._log_message(
                            f"Found 'frames' dataset: shape {f['frames'].shape}"
                        )
                    if "images" in f:
                        self._log_message(
                            f"Found 'images' group with {len(f['images'].keys())} items"
                        )
                    if "timeseries" in f:
                        self._log_message(
                            f"Found 'timeseries' group with {len(f['timeseries'].keys())} items"
                        )

            except Exception as e:
                self._log_message(f"Basic debugging failed: {e}")

    def _connect_signals(self):
        """Connect all UI signals to their respective methods."""
        # Progress signals
        self.progress_updated.connect(self._on_progress_update)
        self.status_updated.connect(self._on_status_update)
        self.performance_updated.connect(self._on_performance_update)

        # File operations
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_dir.clicked.connect(self.load_directory)
        self.btn_detect_rois.clicked.connect(self.enhanced_detect_rois)
        self.btn_clear_rois.clicked.connect(self.clear_roi_detection)

        # NEW: Calibration workflow connections
        self.btn_load_calibration.clicked.connect(self.load_calibration_file)
        self.btn_load_calibration_dataset.clicked.connect(
            self.enhanced_load_calibration_dataset
        )
        self.btn_process_calibration_baseline.clicked.connect(
            self.process_calibration_baseline
        )
        # Analysis operations
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_reset_analysis.clicked.connect(self.reset_for_new_analysis)
        # Testing and diagnostics
        self.btn_quick_test.clicked.connect(self.run_quick_analysis_test)
        self.btn_validate_timing.clicked.connect(self.validate_hdf5_timing)

        # ===== SIMPLIFIED PLOTTING OPERATIONS =====
        self.btn_plot.clicked.connect(self.generate_plot)
        self.btn_save_plot.clicked.connect(self.save_current_plot)
        self.btn_save_all_plots.clicked.connect(self.save_all_plots)
        self.btn_save_results.clicked.connect(
            self.save_results_consolidated_complete
        )  # NEW CONSOLIDATED METHOD
        self.btn_save_with_metadata.clicked.connect(self.save_results_with_metadata)
        self.btn_apply_time_range.clicked.connect(self.apply_time_range)

        # Y-Axis scaling controls
        self.auto_scale_y.toggled.connect(self._on_auto_scale_toggled)
        self.robust_scaling.toggled.connect(self.generate_plot)
        self.adaptive_scaling.toggled.connect(self.generate_plot)
        self.center_around_zero.toggled.connect(self.generate_plot)
        self.lower_percentile_spin.valueChanged.connect(self.generate_plot)
        self.upper_percentile_spin.valueChanged.connect(self.generate_plot)
        self.btn_apply_y_range.clicked.connect(self.generate_plot)

        # Threshold visualization signals
        self.show_baseline_mean.toggled.connect(self.generate_plot)
        self.show_deviation_band.toggled.connect(self.generate_plot)
        self.show_detection_threshold.toggled.connect(self.generate_plot)
        self.show_threshold_stats.toggled.connect(self.generate_plot)

        # UI interactions
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.frame_interval.valueChanged.connect(self.update_end_time)
        self.threshold_params_stack.currentChanged.connect(
            self._on_threshold_tab_changed
        )
        self.chk_12well.toggled.connect(self._on_12well_toggled)

    # ===================================================================
    # FILE LOADING AND ROI DETECTION METHODS
    # ===================================================================
    def load_file(self):
        """Load HDF5 or AVI file(s) with automatic detection."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select File(s)",
            "",
            "Video Files (*.h5 *.hdf5 *.avi);;HDF5 Files (*.h5 *.hdf5);;AVI Files (*.avi);;All Files (*)",
        )
        if not file_paths:
            return

        # Handle single or multiple files
        if len(file_paths) == 1:
            file_path = file_paths[0]

            # Check if single AVI file - load as single video (not batch)
            if file_path.lower().endswith(".avi"):
                self._load_single_avi(file_path)
                return
        else:
            # Multiple files - check if they are AVIs for batch processing
            if all(f.lower().endswith(".avi") for f in file_paths):
                self._load_avi_batch(file_paths)
                return
            else:
                self._log_message(
                    "Multiple file selection only supported for AVI files"
                )
                return

        self.file_path = file_path
        basename = os.path.basename(file_path)

        # === AUTOMATISCHE LEGACY-DETECTION BEIM LADEN ===
        try:
            # Quick legacy check
            with h5py.File(file_path, "r") as f:
                is_legacy = self._quick_legacy_check(f)

            if is_legacy:
                self._log_message(f"Legacy file detected: {basename}")
                self._log_message(
                    "   Will automatically enhance with unit documentation during analysis"
                )
                self.lbl_file_info.setText(
                    f"Loaded LEGACY file: {basename} (auto-enhancement enabled)"
                )
            else:
                self._log_message(f"Modern file detected: {basename}")
                self.lbl_file_info.setText(f"Loaded file: {basename}")

        except Exception as e:
            self._log_message(f"Could not determine file type: {e}")
            self.lbl_file_info.setText(f"Loaded file: {basename}")

        # Clear any existing ROI detection
        self.masks = []

        # Enhanced structure detection and loading
        if DUAL_STRUCTURE_AVAILABLE:
            try:
                # Detect structure first
                structure_info = detect_hdf5_structure_type(file_path)
                self._log_message(f"Detected HDF5 structure: {structure_info['type']}")

                if structure_info["type"] == "error":
                    self._log_message(
                        f"Structure detection failed: {structure_info['error']}"
                    )
                    return

                self._log_message(f"Frame count: {structure_info['frame_count']}")
                self._log_message(f"Frame shape: {structure_info['frame_shape']}")
                self._log_message(f"Data location: {structure_info['data_location']}")

                # Use dual structure reader
                reader = reader_function_dual_structure
                self._log_message("Using enhanced dual structure reader")

            except Exception as e:
                self._log_message(f"Enhanced reader failed, using fallback: {e}")
                reader = napari_get_reader(file_path)
        else:
            # Use original reader
            reader = napari_get_reader(file_path)

        if reader is None:
            self._log_message("No valid HDF5 reader available.")
            return

        try:
            # Clear existing layers
            self.viewer.layers.clear()

            # Load layers from reader
            layers = reader(file_path)
            for data, meta, layer_type in layers:
                name = meta.get("name", basename)
                kwargs = {k: v for k, v in meta.items() if k not in ("name",)}

                if layer_type == "image":
                    self.viewer.add_image(data, name=name, **kwargs)
                elif layer_type == "labels":
                    self.viewer.add_labels(data, name=name, **kwargs)

            # Log structure information if available
            if layers and "metadata" in layers[0][1]:
                metadata = layers[0][1]["metadata"]
                if "structure_type" in metadata:
                    structure_type = metadata["structure_type"]
                    frame_count = metadata.get("frame_count", "unknown")
                    self._log_message(
                        f"Successfully loaded {structure_type} structure with {frame_count} frames"
                    )

        except Exception as e:
            self._log_message(f"Reader error: {e}")
            return

        # Update end time for analysis parameters
        self.update_end_time()
        self.check_hdf5_structure()

    def _load_single_avi(self, file_path: str):
        """Load a single AVI file (only first frame for ROI detection)."""
        try:
            from ._avi_reader import AVIVideoReader

            self.file_path = file_path
            basename = os.path.basename(file_path)

            self._log_message(f"Loading AVI file: {basename}")

            # Store for later processing
            self.avi_batch_paths = [file_path]  # Single file as batch
            self.avi_batch_interval = 5.0  # Default frame interval

            # Get metadata without loading all frames
            with AVIVideoReader(file_path) as reader:
                # Load ONLY first frame for ROI detection
                first_frame = reader.get_frame(0)
                if first_frame is None:
                    raise ValueError("Could not load first frame")

                # Log frame info for debugging
                self._log_message(f"First frame shape: {first_frame.shape}")
                self._log_message(f"First frame dtype: {first_frame.dtype}")
                self._log_message(
                    f"First frame value range: {first_frame.min()}-{first_frame.max()}"
                )

                # Calculate estimated frames
                video_fps = reader.fps
                target_interval = reader.metadata.get("frame_interval", 5.0)
                self.avi_batch_interval = target_interval
                frames_per_sample = max(1, int(video_fps * target_interval))
                frame_count_estimate = len(
                    range(0, reader.frame_count, frames_per_sample)
                )

                metadata = {
                    "source_type": "avi_single",
                    "fps": reader.fps,
                    "frame_interval": target_interval,
                    "frame_count": reader.frame_count,
                    "frame_count_estimate": frame_count_estimate,
                    "duration": reader.duration,
                    "resolution": {"width": reader.width, "height": reader.height},
                    "source_path": file_path,
                    "frames_per_sample": frames_per_sample,
                }

            # Clear existing layers
            self.viewer.layers.clear()

            # Add only first frame to napari
            self._log_message("Adding first frame to napari viewer...")
            layer = self.viewer.add_image(
                first_frame, name=f"{basename}_first_frame", metadata=metadata
            )
            self._log_message(f"Layer added: {layer.name}, visible: {layer.visible}")

            # Update UI
            duration_min = metadata["duration"] / 60.0
            self.lbl_file_info.setText(
                f"Loaded AVI: {basename} "
                f"({frame_count_estimate} frames estimated, {duration_min:.1f} min, {target_interval}s interval) - First frame only"
            )

            self._log_message("Loaded first frame for ROI detection")
            self._log_message(f"Frames (estimated): {frame_count_estimate}")
            self._log_message(f"Duration: {duration_min:.1f} minutes")
            self._log_message(f"Frame interval: {target_interval}s")
            self._log_message("Note: Full frames will be loaded during processing")

            # Clear any existing ROI detection
            self.masks = []

            # Update end time for analysis
            self.update_end_time()

        except ImportError:
            self._log_message(
                "Error: AVI support not available. Install opencv-python: pip install opencv-python"
            )
        except Exception as e:
            self._log_message(f"Error loading AVI file: {e}")
            import traceback

            self._log_message(traceback.format_exc())

    def _process_avi_batch_for_analysis(
        self,
        video_paths: List[str],
        masks: List[np.ndarray],
        chunk_size: int,
        progress_callback,
        frame_interval: float,
    ):
        """Process AVI batch with streaming analysis - no need to load all frames."""
        import time
        from ._avi_reader import process_avi_batch_streaming

        start_time = time.time()

        # Stream process all videos - loads and analyzes chunk by chunk
        self._log_message(
            f"Starting streaming analysis of {len(video_paths)} AVI files..."
        )
        self._log_message("Using memory-efficient streaming: load → analyze → discard")

        roi_changes, metadata = process_avi_batch_streaming(
            video_paths,
            masks,
            target_frame_interval=frame_interval,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )

        total_duration = metadata["total_duration"]
        proc_time = time.time() - start_time

        # Calculate start and end times from the data
        start_time_data = 0.0  # Start time is always 0
        end_time_data = total_duration

        self._log_message(
            f"✓ AVI batch streaming analysis complete in {proc_time:.2f}s"
        )
        self._log_message(f"  Total frames analyzed: {metadata['total_frames']}")
        self._log_message(
            f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)"
        )
        self._log_message(f"  Start time: {start_time_data:.1f}s")
        self._log_message(
            f"  End time: {end_time_data:.1f}s ({end_time_data/60:.1f}min)"
        )
        self._log_message(f"  ROIs tracked: {len(roi_changes)}")

        return video_paths[0], roi_changes, total_duration

    def _load_avi_batch(self, file_paths: List[str]):
        """Load multiple AVI files as batch timeseries (only first frame for ROI detection)."""
        self._log_message("=== _load_avi_batch() START ===")
        self._log_message(f"Received {len(file_paths)} file paths")
        for idx, path in enumerate(file_paths):
            self._log_message(f"  [{idx}] {path}")

        try:
            from ._avi_reader import AVIVideoReader

            # Get frame interval from metadata or use default
            target_interval = 5.0  # seconds (same as HDF5)

            self._log_message(f"Loading {len(file_paths)} AVI files as batch...")
            self._log_message(
                f"Target frame interval: {target_interval}s (0.2 FPS effective)"
            )

            # Store batch info for later processing
            self.avi_batch_paths = file_paths
            self.avi_batch_interval = target_interval

            # For memory efficiency: Only open first video to get basic info
            # Full metadata will be calculated during analysis
            self._log_message(
                "Getting metadata from first video only (memory efficient)..."
            )

            batch_metadata = {
                "videos": [],
                "source_type": "avi_batch",
                "target_frame_interval": target_interval,
                "video_count": len(file_paths),
            }

            # Load ONLY first frame from first video for ROI detection
            self._log_message(f"Opening first video: {file_paths[0]}")
            with AVIVideoReader(file_paths[0]) as reader:
                self._log_message("AVIVideoReader opened successfully")
                first_frame = reader.get_frame(0)
                if first_frame is None:
                    raise ValueError("Could not load first frame from first video")

                # Get basic info from first video only
                video_fps = reader.fps
                frames_per_sample = max(1, int(video_fps * target_interval))

                # Store metadata for first video
                batch_metadata["first_video_fps"] = video_fps
                batch_metadata["frames_per_sample"] = frames_per_sample
                batch_metadata["effective_fps"] = 1.0 / target_interval

                # Log frame info for debugging
                self._log_message("First frame loaded successfully")
                self._log_message(f"First frame shape: {first_frame.shape}")
                self._log_message(f"First frame dtype: {first_frame.dtype}")
                self._log_message(
                    f"First frame value range: {first_frame.min()}-{first_frame.max()}"
                )

            # Clear existing layers
            self._log_message(f"Clearing {len(self.viewer.layers)} existing layers...")
            self.viewer.layers.clear()
            self._log_message("Layers cleared")

            # Add only first frame to napari
            self._log_message("Adding first frame to napari viewer...")
            self._log_message(f"Frame data type: {type(first_frame)}")
            self._log_message(
                f"Viewer has {len(self.viewer.layers)} layers before adding"
            )

            layer = self.viewer.add_image(
                first_frame,
                name=f"batch_{len(file_paths)}_videos_first_frame",
                metadata=batch_metadata,
            )

            self._log_message("Layer added successfully!")
            self._log_message(f"Layer name: {layer.name}")
            self._log_message(f"Layer visible: {layer.visible}")
            self._log_message(f"Layer data shape: {layer.data.shape}")
            self._log_message(f"Viewer now has {len(self.viewer.layers)} layers")

            # Store file path (use first file as reference)
            self.file_path = file_paths[0]

            # Update UI - simplified message (full metadata calculated during analysis)
            self.lbl_file_info.setText(
                f"Loaded {len(file_paths)} AVI files as batch "
                f"(~{batch_metadata['effective_fps']:.2f} FPS effective) - First frame only"
            )

            self._log_message("Loaded first frame for ROI detection")
            self._log_message(f"Batch contains {len(file_paths)} video files")
            self._log_message(f"Effective FPS: {batch_metadata['effective_fps']:.2f}")
            self._log_message(f"Frame interval: {target_interval}s")
            self._log_message("Note: Full metadata will be calculated during analysis")
            self._log_message("Note: All frames will be loaded during processing")

            # Clear any existing ROI detection
            self.masks = []

            # Update end time for analysis
            self.update_end_time()

        except ImportError:
            self._log_message(
                "Error: AVI support not available. Install opencv-python: pip install opencv-python"
            )
            import traceback

            self._log_message(traceback.format_exc())
        except Exception as e:
            self._log_message(f"ERROR in _load_avi_batch: {e}")
            import traceback

            self._log_message(traceback.format_exc())
        finally:
            self._log_message("=== _load_avi_batch() END ===")

    def _quick_legacy_check(self, h5_file) -> bool:
        """Quick check if file is legacy (same logic as in _metadata.py)."""

        file_version = h5_file.attrs.get("file_version", "1.0")
        if float(file_version) < 2.2:
            return True

        if "timeseries" in h5_file:
            ts_group = h5_file["timeseries"]
            if not ts_group.attrs.get("expected_intervals_fixed", False):
                return True

        return False

    def load_directory(self):
        """Load a directory containing HDF5 or AVI files."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory Containing Video Files"
        )
        if not directory:
            return

        self.directory = directory
        self.file_path = None
        try:
            # Scan for both HDF5 and AVI files
            h5_files = [
                f for f in os.listdir(directory) if f.lower().endswith((".h5", ".hdf5"))
            ]
            avi_files = [f for f in os.listdir(directory) if f.lower().endswith(".avi")]

            total_files = len(h5_files) + len(avi_files)

            if total_files == 0:
                self.lbl_file_info.setText(
                    f"No HDF5 or AVI files found in: {directory}"
                )
                self._log_message(f"No video files found in directory: {directory}")
                return

            # Build info message
            file_info = []
            if h5_files:
                file_info.append(f"{len(h5_files)} HDF5")
            if avi_files:
                file_info.append(f"{len(avi_files)} AVI")

            files_str = ", ".join(file_info)
            self.lbl_file_info.setText(
                f"Loaded directory: {directory} ({files_str} files)"
            )
            self._log_message(f"Loaded directory with {files_str} files: {directory}")

            if h5_files:
                self._log_message(
                    f"  HDF5 files: {', '.join(h5_files[:5])}{'...' if len(h5_files) > 5 else ''}"
                )
            if avi_files:
                self._log_message(
                    f"  AVI files: {', '.join(avi_files[:5])}{'...' if len(avi_files) > 5 else ''}"
                )

        except Exception as e:
            self.lbl_file_info.setText(f"Error reading directory: {e}")
            self._log_message(f"ERROR reading directory: {e}")
            return

        # If AVI files are found, load them as batch
        if avi_files:
            self._log_message(
                f"Loading {len(avi_files)} AVI files from directory as batch..."
            )
            avi_paths = [os.path.join(directory, f) for f in sorted(avi_files)]
            self._log_message(f"AVI paths to load: {avi_paths}")
            self._log_message("Calling _load_avi_batch()...")
            self._load_avi_batch(avi_paths)
            self._log_message("_load_avi_batch() completed")
            return

        # Otherwise, use HDF5 reader for directory
        # Clear all existing layers
        self.viewer.layers.clear()

        # Use reader to load directory
        reader = napari_get_reader(directory)
        if reader is None:
            self._log_message("No valid directory for HDF5 reader.")
            return

        try:
            layers = reader(directory)
        except Exception as e:
            self._log_message(f"Reader error: {e}")
            return

        # Add each layer to viewer
        for data, meta, layer_type in layers:
            name = meta.get("name", os.path.basename(directory))
            kwargs = {k: v for k, v in meta.items() if k not in ("name",)}

            if layer_type == "image":
                self.viewer.add_image(data, name=name, **kwargs)
            elif layer_type == "labels":
                self.viewer.add_labels(data, name=name, **kwargs)

    def update_end_time(self):
        """Enhanced update_end_time method with dual structure support."""
        if self.file_path:
            try:
                # Check if this is an AVI file or AVI batch
                if hasattr(self, "avi_batch_paths") and self.avi_batch_paths:
                    # AVI batch - use metadata from viewer layer
                    if len(self.viewer.layers) > 0:
                        layer = self.viewer.layers[0]
                        if hasattr(layer, "metadata") and layer.metadata:
                            metadata = layer.metadata
                            if "total_duration" in metadata:
                                total_duration_seconds = metadata["total_duration"]
                                frame_count = metadata.get("total_frames_estimate", 0)
                            elif "duration" in metadata:
                                total_duration_seconds = metadata["duration"]
                                frame_count = metadata.get("frame_count_estimate", 0)
                            else:
                                self._log_message("No duration metadata found for AVI")
                                return
                        else:
                            self._log_message("No metadata found in layer")
                            return
                    else:
                        self._log_message("No layers found")
                        return
                elif self.file_path.lower().endswith(".avi"):
                    # Single AVI file
                    from ._avi_reader import AVIVideoReader

                    with AVIVideoReader(self.file_path) as reader:
                        total_duration_seconds = reader.duration
                        video_fps = reader.fps
                        target_interval = reader.metadata.get("frame_interval", 5.0)
                        frames_per_sample = max(1, int(video_fps * target_interval))
                        frame_count = len(
                            range(0, reader.frame_count, frames_per_sample)
                        )
                else:
                    # HDF5 file
                    if DUAL_STRUCTURE_AVAILABLE:
                        # Use structure detection to get frame count
                        structure_info = detect_hdf5_structure_type(self.file_path)
                        if structure_info["type"] != "error":
                            frame_count = structure_info["frame_count"]
                            self._log_message(
                                f"Frame count from structure detection: {frame_count}"
                            )
                        else:
                            raise Exception("Structure detection failed")
                    else:
                        # Fallback to original method
                        with h5py.File(self.file_path, "r") as f:
                            if "frames" in f:
                                frame_count = len(f["frames"])
                            else:
                                raise Exception("No 'frames' dataset found")

                    frame_interval = self.frame_interval.value()
                    total_duration_seconds = frame_count * frame_interval

                total_duration_minutes = total_duration_seconds / 60.0

                self.time_end.setValue(int(total_duration_seconds))
                self.plot_end_time.setRange(0.0, total_duration_minutes)
                self.plot_end_time.setValue(total_duration_minutes)
                self.plot_start_time.setRange(0.0, total_duration_minutes)
                self.plot_start_time.setValue(0.0)

                self._log_message(
                    f"File contains {frame_count} frames, total duration: {total_duration_minutes:.1f} min"
                )

            except Exception as e:
                self.lbl_file_info.setText(f"Error reading metadata: {str(e)}")
                self._log_message(f"ERROR reading metadata: {str(e)}")

    def enhanced_detect_rois(self):
        """Enhanced ROI detection that properly manages layers for both datasets."""

        # Determine dataset type and log clearly
        current_type = getattr(self, "current_dataset_type", "main")

        # NEW: Ensure main dataset is stored before calibration ROI detection
        if current_type == "calibration":
            if not getattr(self, "main_dataset_stored", False):
                self._log_message(
                    "WARNING: Calibration ROI detection without stored main dataset"
                )
                self._log_message("This may cause issues during analysis")

            # SET CALIBRATION VARIABLES
            current_file = self.calibration_file_path_stored
            dataset_type = "CALIBRATION"
            self._log_message(f"=== ROI DETECTION FOR {dataset_type} DATASET ===")
            self._log_message(f"File: {os.path.basename(current_file)}")
            self._log_message("NOTE: This is for calibration baseline calculation only")
        else:
            # SET MAIN DATASET VARIABLES
            current_file = self.file_path
            dataset_type = "MAIN"
            self._log_message(f"=== ROI DETECTION FOR {dataset_type} DATASET ===")
            self._log_message(f"File: {os.path.basename(current_file)}")
            self._log_message("NOTE: This is for the experimental data analysis")

        if not current_file:
            self.lbl_file_info.setText("Error: No HDF5 file loaded for ROI detection")
            return

        # Get ROI detection parameters
        if self.chk_12well.isChecked():
            params = {
                "min_radius": 100,
                "max_radius": 150,
                "dp": 1.0,
                "min_dist": 200,
                "param1": 50,
                "param2": 30,
            }
        else:
            params = {
                "min_radius": self.min_radius.value(),
                "max_radius": self.max_radius.value(),
                "dp": self.dp_param.value(),
                "min_dist": self.min_dist.value(),
                "param1": self.param1.value(),
                "param2": self.param2.value(),
            }

        try:
            # ROI detection - get first frame from viewer layer or file
            first_frame = None

            # Check if current file is HDF5 or AVI to decide source
            is_hdf5 = current_file.lower().endswith((".h5", ".hdf5"))
            is_avi = current_file.lower().endswith(".avi")

            # For HDF5 files, always read from file (not from viewer layer)
            # For AVI batch, try to use existing layer first
            if is_hdf5:
                self._log_message(
                    "HDF5 file detected - reading first frame from file..."
                )
                first_frame = get_first_frame(current_file)
            elif is_avi or (
                hasattr(self, "avi_batch_loaded") and self.avi_batch_loaded
            ):
                # Try to get frame from existing napari layer (for AVI batch)
                if len(self.viewer.layers) > 0:
                    layer = self.viewer.layers[0]
                    if hasattr(layer, "data"):
                        first_frame = layer.data
                        if len(first_frame.shape) == 3 and first_frame.shape[0] > 1:
                            # Multi-frame layer, take first frame
                            first_frame = first_frame[0]
                        self._log_message(
                            f"Using frame from viewer layer: {layer.name} (shape: {first_frame.shape})"
                        )

            # Final fallback: read from file
            if first_frame is None:
                self._log_message("Fallback: Reading first frame from file...")
                first_frame = get_first_frame(current_file)

            if first_frame is None:
                self._log_message("ERROR: Could not read first frame")
                return

            # Convert to grayscale and enhance
            if len(first_frame.shape) == 3:
                gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = first_frame.copy()

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_frame = clahe.apply(gray_frame)

            # Detect circles
            circles = cv2.HoughCircles(
                enhanced_frame,
                cv2.HOUGH_GRADIENT,
                dp=params["dp"],
                minDist=params["min_dist"],
                param1=params["param1"],
                param2=params["param2"],
                minRadius=params["min_radius"],
                maxRadius=params["max_radius"],
            )

            # Create masks and labeled frame
            masks = []
            if circles is not None:
                circles = np.uint16(np.around(circles))

                # Remove extra dimension from HoughCircles if present
                if len(circles.shape) == 3:
                    circles = circles[0]

                # Robust row-based sorting for multi-well plates
                # Detect number of rows (assuming 18 wells = 3 rows × 6 cols)
                num_circles = len(circles)
                if num_circles == 18:
                    expected_rows = 3
                elif num_circles == 12:
                    expected_rows = 3
                elif num_circles == 24:
                    expected_rows = 4
                elif num_circles == 6:
                    expected_rows = 2
                else:
                    expected_rows = int(np.sqrt(num_circles))

                # Sort all circles by Y coordinate
                y_sorted_indices = np.argsort(circles[:, 1])
                y_sorted = circles[y_sorted_indices]

                # Group into rows
                circles_per_row = num_circles // expected_rows
                sorted_circles = []

                for row_idx in range(expected_rows):
                    start_idx = row_idx * circles_per_row
                    end_idx = start_idx + circles_per_row
                    if row_idx == expected_rows - 1:
                        end_idx = num_circles  # Include remaining circles in last row

                    row_circles = y_sorted[start_idx:end_idx]

                    # Sort this row by X coordinate (left to right)
                    x_sorted_indices = np.argsort(row_circles[:, 0])
                    row_sorted = row_circles[x_sorted_indices]

                    # Reverse every odd row (0-indexed) for meandering pattern
                    # Row 0: L→R, Row 1: R→L, Row 2: L→R, etc.
                    if row_idx % 2 == 1:
                        row_sorted = row_sorted[::-1]

                    sorted_circles.extend(row_sorted)

                sorted_circles = np.array(sorted_circles, dtype=np.uint16)

                for idx, circle in enumerate(sorted_circles):
                    mask = np.zeros(gray_frame.shape, dtype=np.uint8)
                    cv2.circle(
                        mask, (circle[0], circle[1]), circle[2], 255, thickness=-1
                    )
                    masks.append(mask)

                # Create labeled frame
                if len(first_frame.shape) == 3:
                    labeled_frame = first_frame.copy()
                else:
                    labeled_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

                for idx, circle in enumerate(sorted_circles):
                    color = (
                        (255, 165, 0) if dataset_type == "CALIBRATION" else (0, 255, 0)
                    )
                    cv2.circle(
                        labeled_frame, (circle[0], circle[1]), circle[2], color, 2
                    )
                    cv2.putText(
                        labeled_frame,
                        f"{idx + 1}",
                        (circle[0] - 10, circle[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5,
                        (255, 0, 0),
                        3,
                    )

            # Store results based on dataset type
            if dataset_type == "CALIBRATION":
                # Store calibration results (for baseline processing only)
                self.calibration_masks = masks.copy()
                self.calibration_labeled_frame = labeled_frame.copy()

                # CRITICAL: Set masks for immediate calibration processing but preserve main
                self.masks = masks  # Temporary for calibration baseline processing
                self.labeled_frame = labeled_frame

                # Enable calibration baseline processing
                if hasattr(self, "btn_process_calibration_baseline"):
                    self.btn_process_calibration_baseline.setEnabled(True)

                # Update status
                if hasattr(self, "calibration_status_label"):
                    self.calibration_status_label.setText(
                        "✅ 1. Calibration file selected\n"
                        "✅ 2. Calibration first frame loaded\n"
                        "✅ 3. Calibration ROIs detected\n"
                        "4. Process baseline (Analysis tab)\n"
                        "5. Return to main dataset for analysis"
                    )

                self._log_message(
                    "Next: Go to Analysis tab and click 'Process Calibration Baseline'"
                )
                self._log_message(
                    f"Applied automatic meandering sort to {len(sorted_circles)} ROIs"
                )
            else:  # MAIN dataset
                # Store main results permanently
                self.main_masks = masks.copy()
                self.main_labeled_frame = labeled_frame.copy()
                self.masks = masks
                self.labeled_frame = labeled_frame

                # Also update stored main dataset if we're in main mode
                if hasattr(self, "main_dataset_stored") and self.main_dataset_stored:
                    self.main_masks = masks.copy()

            # Add layers to viewer
            self._add_roi_layers_to_viewer(labeled_frame, masks, dataset_type)

            result_msg = f"{dataset_type}: Detected {len(masks)} ROIs"
            self.lbl_file_info.setText(result_msg)
            self._log_message(result_msg)

        except Exception as e:
            self._log_message(f"ERROR in ROI detection: {e}")

    def reset_for_new_analysis(self):
        """Reset all analysis data and UI state for a new analysis."""
        try:
            # Clear analysis results
            self.merged_results = {}
            self.roi_baseline_means = {}
            self.roi_upper_thresholds = {}
            self.roi_lower_thresholds = {}
            self.roi_statistics = {}
            self.movement_data = {}
            self.fraction_data = {}
            self.sleep_data = {}
            self.quiescence_data = {}
            self.roi_colors = {}
            self.roi_band_widths = {}

            # Clear ROI detection
            self.masks = []
            self.labeled_frame = None

            # Clear calibration state
            self.current_dataset_type = "main"
            self.calibration_file_path_stored = None
            self.calibration_masks = []
            self.calibration_labeled_frame = None
            self.calibration_baseline_processed = False
            self.calibration_baseline_statistics = {}

            # Reset file paths
            self.file_path = None
            self.directory = None
            self.main_dataset_path = None

            # Clear viewer layers
            self.viewer.layers.clear()

            # Reset UI elements
            self.lbl_file_info.setText("No file loaded")
            self.results_label.setText("Results will be displayed here.")
            self.status_label.setText("Ready to start analysis")
            self.progress_bar.setValue(0)

            # Reset calibration UI
            if hasattr(self, "calibration_file_path"):
                self.calibration_file_path.setText("No calibration file selected")
                self.btn_load_calibration_dataset.setEnabled(False)
                self.btn_process_calibration_baseline.setEnabled(False)

            if hasattr(self, "calibration_status_label"):
                self.calibration_status_label.setText(
                    "1. Select calibration file\n"
                    "2. Load calibration dataset\n"
                    "3. Detect ROIs (Input tab)\n"
                    "4. Process baseline"
                )

            # Clear log
            self.log_text.clear()

            # Clear matplotlib figure
            if hasattr(self, "figure"):
                self.figure.clear()
                self.canvas.draw()

            self._log_message("Analysis reset complete - ready for new analysis")

        except Exception as e:
            self._log_message(f"Error during reset: {e}")

    def process_calibration_baseline(self):
        """Process calibration dataset to create baseline statistics with progress bar."""
        if not self.calibration_masks:
            self._log_message("No calibration ROIs detected")
            return

        if not self.calibration_file_path_stored:
            self._log_message("No calibration file selected")
            return

        # Start progress monitoring and disable button
        self.btn_process_calibration_baseline.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing calibration baseline...")

        @thread_worker(start_thread=False)
        def _calibration_worker():
            return self._process_calibration_baseline_worker()

        worker = _calibration_worker()
        worker.returned.connect(self._calibration_finished)
        worker.errored.connect(self._calibration_errored)
        worker.finished.connect(self._calibration_done)
        worker.start()

    def _process_calibration_baseline_worker(self):
        """Worker function for calibration baseline processing."""

        def progress_callback(percent, message):
            self.progress_updated.emit(int(percent))
            self.status_updated.emit(message)

        try:
            self._log_message("Processing calibration baseline...")
            self._log_message(
                f"File: {os.path.basename(self.calibration_file_path_stored)}"
            )
            self._log_message(f"ROIs: {len(self.calibration_masks)}")

            progress_callback(5, "Initializing calibration processing...")

            # Try to process calibration file with enhanced error handling
            calibration_roi_changes = None
            calibration_duration = 0

            try:
                from ._reader import process_hdf5_file

                progress_callback(10, "Processing calibration file (method 1)...")
                self._log_message("Attempting to process calibration file...")

                _, calibration_roi_changes, calibration_duration = process_hdf5_file(
                    file_path=self.calibration_file_path_stored,
                    masks=self.calibration_masks,
                    chunk_size=self.chunk_size.value(),
                    progress_callback=lambda p, m: progress_callback(
                        10 + (p * 0.4), f"Calibration: {m}"
                    ),
                    frame_interval=self.frame_interval.value(),
                )
                self._log_message("Successfully processed calibration file")

            except Exception as reader_error:
                self._log_message(f"Reader error encountered: {reader_error}")

                # Try alternative processing method
                try:
                    from ._reader import process_single_file_in_parallel_dual_structure

                    progress_callback(20, "Trying alternative processing method...")
                    self._log_message("Trying alternative processing method...")

                    _, calibration_roi_changes, calibration_duration = (
                        process_single_file_in_parallel_dual_structure(
                            self.calibration_file_path_stored,
                            self.calibration_masks,
                            chunk_size=self.chunk_size.value(),
                            progress_callback=lambda p, m: progress_callback(
                                20 + (p * 0.3), f"Alt method: {m}"
                            ),
                            frame_interval=self.frame_interval.value(),
                            num_processes=1,  # Use single process to avoid issues
                        )
                    )
                    self._log_message("Alternative processing successful")

                except Exception as alt_error:
                    self._log_message(
                        f"Alternative processing also failed: {alt_error}"
                    )

                    # Final fallback - try the basic reader functions
                    try:
                        from ._reader import process_hdf5_files

                        progress_callback(30, "Trying basic processing method...")
                        self._log_message("Trying basic processing method...")

                        # Use directory processing as fallback
                        cal_dir = os.path.dirname(self.calibration_file_path_stored)
                        results, durations, _, _ = process_hdf5_files(
                            cal_dir,
                            masks=self.calibration_masks,
                            num_processes=1,
                            chunk_size=self.chunk_size.value(),
                            progress_callback=lambda p, m: progress_callback(
                                30 + (p * 0.2), f"Basic: {m}"
                            ),
                            frame_interval=self.frame_interval.value(),
                        )

                        # Extract results for our specific file
                        cal_filename = os.path.basename(
                            self.calibration_file_path_stored
                        )
                        calibration_roi_changes = None
                        calibration_duration = 0

                        for file_path, roi_data in results.items():
                            if cal_filename in file_path:
                                calibration_roi_changes = roi_data
                                calibration_duration = durations.get(file_path, 0)
                                break

                        if calibration_roi_changes is None:
                            raise Exception(
                                "Could not find calibration data in results"
                            )

                        self._log_message("Basic processing successful")

                    except Exception as final_error:
                        return {
                            "success": False,
                            "error": f"All processing methods failed: {final_error}",
                        }

            # Continue with the rest of the processing if we got valid data
            if not calibration_roi_changes:
                return {
                    "success": False,
                    "error": "No calibration data obtained - processing failed",
                }

            progress_callback(60, "Applying preprocessing...")
            self._log_message(
                f"Calibration data processed: {len(calibration_roi_changes)} ROIs"
            )

            # Apply same preprocessing as main dataset
            from ._calc import (
                apply_matlab_normalization_to_merged_results,
                improved_full_dataset_detrending,
            )

            # MATLAB normalization
            progress_callback(70, "Applying MATLAB normalization...")
            self._log_message("Applying MATLAB normalization to calibration data...")
            normalized_calibration = apply_matlab_normalization_to_merged_results(
                calibration_roi_changes, enable_matlab_norm=True
            )

            # Detrending (if enabled)
            progress_callback(80, "Applying detrending...")
            if (
                hasattr(self, "enable_detrending")
                and self.enable_detrending.isChecked()
            ):
                self._log_message("Applying detrending to calibration data...")
                processed_calibration = improved_full_dataset_detrending(
                    normalized_calibration
                )
            else:
                self._log_message("Skipping detrending (disabled)")
                processed_calibration = normalized_calibration

            # Calculate baseline statistics for each ROI
            progress_callback(90, "Calculating baseline statistics...")
            self._log_message(
                "Calculating baseline statistics from COMPLETE calibration dataset..."
            )
            self._log_message(
                f"Calibration duration: {calibration_duration/60:.1f} minutes"
            )
            calibration_baseline_statistics = {}

            for roi, data in processed_calibration.items():
                if not data:
                    self._log_message(f"No data for ROI {roi}, skipping")
                    continue

                # Extract all values from complete calibration dataset
                values = np.array([val for _, val in data])

                # Calculate comprehensive statistics
                cal_mean = np.mean(values)
                cal_std = np.std(values)
                cal_multiplier = self.calibration_multiplier.value()

                # Calculate hysteresis thresholds
                threshold_band = cal_multiplier * cal_std
                upper_threshold = cal_mean + threshold_band
                lower_threshold = max(0, cal_mean - threshold_band)  # Don't go negative

                calibration_baseline_statistics[roi] = {
                    "baseline_mean": cal_mean,
                    "baseline_std": cal_std,
                    "upper_threshold": upper_threshold,
                    "lower_threshold": lower_threshold,
                    "threshold_band": threshold_band,
                    "multiplier": cal_multiplier,
                    "data_points": len(values),
                    "duration_minutes": calibration_duration / 60,
                    "data_range": (float(np.min(values)), float(np.max(values))),
                }

                self._log_message(
                    f"ROI {roi}: mean={cal_mean:.1f}, std={cal_std:.1f}, thresholds=[{lower_threshold:.1f}, {upper_threshold:.1f}], frames={len(values)}"
                )

            progress_callback(100, "Calibration baseline complete")

            return {
                "success": True,
                "statistics": calibration_baseline_statistics,
                "duration": calibration_duration,
                "roi_count": len(calibration_baseline_statistics),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calibration_finished(self, result):
        """Handle successful calibration completion."""
        if result["success"]:
            self.calibration_baseline_statistics = result["statistics"]
            self.calibration_baseline_processed = True

            # Update status
            if hasattr(self, "calibration_status_label"):
                self.calibration_status_label.setText(
                    "✅ 1. Calibration file selected\n"
                    "✅ 2. Calibration dataset loaded\n"
                    "✅ 3. Calibration ROIs detected\n"
                    "✅ 4. Calibration baseline processed\n"
                    "Ready for analysis!"
                )

            self.status_label.setText("Calibration baseline processing complete")

            # Log results
            successful_rois = result["roi_count"]
            self._log_message("Calibration baseline processing complete:")
            self._log_message(f"  ROIs processed: {successful_rois}")
            self._log_message(f"  Duration: {result['duration']/60:.1f} minutes")
            self._log_message(
                "Calibration baseline ready! Switch to main dataset and run analysis."
            )

        else:
            self._log_message(f"Calibration failed: {result['error']}")
            self.status_label.setText(f"Calibration failed: {result['error']}")

    def _calibration_errored(self, exc):
        """Handle calibration errors."""
        self.status_label.setText(f"Calibration error: {exc}")
        self._log_message(f"Calibration error: {exc}")

    def _calibration_done(self):
        """Cleanup after calibration completion."""
        self.btn_process_calibration_baseline.setEnabled(True)
        self.progress_bar.setValue(0)

    def add_calibration_layers_to_viewer(self, labeled_frame, masks):
        """Add calibration dataset layers with clear naming and organization."""
        try:
            basename = os.path.basename(self.calibration_file_path_stored)

            # Add calibration raw frame
            cal_raw_layer = self.viewer.add_image(
                labeled_frame,
                name=f"CALIBRATION - {basename} - ROI Detection",
                colormap="gray",
                visible=True,
                opacity=0.8,
            )

            # Store calibration info in metadata
            cal_raw_layer.metadata.update(
                {
                    "dataset_type": "calibration",
                    "file_path": self.calibration_file_path_stored,
                    "roi_count": len(masks),
                    "workflow_step": "roi_detection",
                }
            )

            self._log_message(
                f"Added calibration ROI detection layer: {len(masks)} ROIs"
            )

        except Exception as e:
            self._log_message(f"Error adding calibration layers: {e}")

    def add_main_dataset_layers_to_viewer(self, labeled_frame, masks):
        """Add main dataset layers with clear naming and organization."""
        try:
            basename = (
                os.path.basename(self.file_path) if self.file_path else "main_dataset"
            )

            # Add main dataset frame
            main_raw_layer = self.viewer.add_image(
                labeled_frame,
                name=f"MAIN - {basename} - ROI Detection",
                colormap="gray",
                visible=True,
                opacity=0.8,
            )

            # Store main dataset info in metadata
            main_raw_layer.metadata.update(
                {
                    "dataset_type": "main",
                    "file_path": self.file_path,
                    "roi_count": len(masks),
                    "workflow_step": "roi_detection",
                }
            )

            self._log_message(f"Added main ROI detection layer: {len(masks)} ROIs")

        except Exception as e:
            self._log_message(f"Error adding main dataset layers: {e}")

    def manage_workflow_layers(self, workflow_step):
        """
        Manage layer visibility based on workflow step.

        Args:
            workflow_step: 'main_dataset', 'calibration_setup', 'comparison', 'final_analysis'
        """
        try:
            if workflow_step == "main_dataset":
                # Show only main dataset layers
                self._set_layer_visibility_by_type("main", True)
                self._set_layer_visibility_by_type("calibration", False)
                self._set_layer_visibility_by_type("comparison", False)
                self._log_message("Switched view: Main dataset only")

            elif workflow_step == "calibration_setup":
                # Show only calibration layers
                self._set_layer_visibility_by_type("main", False)
                self._set_layer_visibility_by_type("calibration", True)
                self._set_layer_visibility_by_type("comparison", False)
                self._log_message("Switched view: Calibration dataset only")

            elif workflow_step == "comparison":
                # Show comparison view
                self.switch_to_comparison_view()

            elif workflow_step == "final_analysis":
                # Show main dataset for final analysis
                self._set_layer_visibility_by_type("main", True)
                self._set_layer_visibility_by_type("calibration", False)
                self._set_layer_visibility_by_type("comparison", False)
                self._log_message("Switched view: Main dataset for analysis")

        except Exception as e:
            self._log_message(f"Error managing workflow layers: {e}")

    def _set_layer_visibility_by_type(self, dataset_type, visible):
        """Set visibility for all layers of a specific dataset type."""
        try:
            count = 0
            for layer in self.viewer.layers:
                if (
                    hasattr(layer, "metadata")
                    and layer.metadata.get("dataset_type") == dataset_type
                ):
                    layer.visible = visible
                    count += 1

            if count > 0:
                status = "visible" if visible else "hidden"
                self._log_message(f"Set {count} {dataset_type} layers to {status}")

        except Exception as e:
            self._log_message(f"Error setting layer visibility: {e}")

    def _create_roi_comparison_image(self, cal_frame, main_frame):
        """Create side-by-side comparison of ROI detection between calibration and main datasets."""
        try:
            # Convert frames to RGB if needed
            if len(cal_frame.shape) == 3:
                cal_rgb = cal_frame.copy()
            else:
                cal_rgb = cv2.cvtColor(cal_frame, cv2.COLOR_GRAY2RGB)

            if len(main_frame.shape) == 3:
                main_rgb = main_frame.copy()
            else:
                main_rgb = cv2.cvtColor(main_frame, cv2.COLOR_GRAY2RGB)

            # Resize frames to same height for comparison
            target_height = min(
                cal_rgb.shape[0], main_rgb.shape[0], 800
            )  # Max height 800px

            # Calculate new widths maintaining aspect ratio
            cal_ratio = cal_rgb.shape[1] / cal_rgb.shape[0]
            main_ratio = main_rgb.shape[1] / main_rgb.shape[0]

            cal_width = int(target_height * cal_ratio)
            main_width = int(target_height * main_ratio)

            cal_resized = cv2.resize(cal_rgb, (cal_width, target_height))
            main_resized = cv2.resize(main_rgb, (main_width, target_height))

            # Draw ROIs on both images
            cal_with_rois = self._draw_rois_on_image(
                cal_resized, self.calibration_masks, "CAL", (255, 165, 0)
            )  # Orange
            main_with_rois = self._draw_rois_on_image(
                main_resized, self.main_masks, "MAIN", (0, 255, 0)
            )  # Green

            # Add labels at the top
            cv2.putText(
                cal_with_rois,
                "CALIBRATION DATASET",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                main_with_rois,
                "MAIN DATASET",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Create separator
            separator = (
                np.ones((target_height, 20, 3), dtype=np.uint8) * 128
            )  # Gray separator

            # Combine side by side with separator
            comparison = np.hstack([cal_with_rois, separator, main_with_rois])

            self._log_message(f"Created ROI comparison image: {comparison.shape}")
            return comparison

        except Exception as e:
            self._log_message(f"Error creating ROI comparison image: {e}")
            # Return a simple concatenation as fallback
            try:
                if cal_frame.shape == main_frame.shape:
                    return np.hstack([cal_frame, main_frame])
                else:
                    # If shapes don't match, return the calibration frame
                    return cal_frame
            except:
                return cal_frame

    def _draw_rois_on_image(self, image, masks, prefix, color):
        """Draw ROI circles and labels on image with specified color."""
        import cv2

        if not masks:
            return image.copy()

        result = image.copy()

        try:
            for i, mask in enumerate(masks):
                # Get ROI center and radius from mask
                center = self._get_roi_center(mask)
                radius = int(self._get_roi_radius(mask))

                if center[0] > 0 and center[1] > 0 and radius > 0:
                    # Scale coordinates if image was resized
                    scale_x = (
                        result.shape[1] / mask.shape[1] if mask.shape[1] > 0 else 1
                    )
                    scale_y = (
                        result.shape[0] / mask.shape[0] if mask.shape[0] > 0 else 1
                    )

                    scaled_center = (int(center[0] * scale_x), int(center[1] * scale_y))
                    scaled_radius = int(radius * min(scale_x, scale_y))

                    # Draw circle
                    cv2.circle(result, scaled_center, scaled_radius, color, 2)

                    # Draw ROI label
                    label = f"{prefix} {i+1}"
                    label_pos = (
                        scaled_center[0] - 20,
                        scaled_center[1] - scaled_radius - 10,
                    )

                    # Ensure label position is within image bounds
                    label_pos = (max(5, label_pos[0]), max(20, label_pos[1]))

                    cv2.putText(
                        result,
                        label,
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        except Exception as e:
            self._log_message(f"Error drawing ROIs on image: {e}")

        return result

    def switch_to_comparison_view(self):
        """Switch viewer to show both calibration and main datasets for comparison."""
        try:
            # Show both dataset layers
            for layer in self.viewer.layers:
                if hasattr(layer, "metadata"):
                    dataset_type = layer.metadata.get("dataset_type", "")
                    if dataset_type in ["calibration", "main"]:
                        layer.visible = True

            # Create and add comparison layer if both datasets have ROIs
            if (
                hasattr(self, "calibration_labeled_frame")
                and self.calibration_labeled_frame is not None
                and hasattr(self, "main_labeled_frame")
                and self.main_labeled_frame is not None
            ):

                self._log_message("Creating ROI correspondence comparison...")

                # Create side-by-side comparison
                comparison_image = self._create_roi_comparison_image(
                    self.calibration_labeled_frame, self.main_labeled_frame
                )

                # Remove existing comparison layer if it exists
                layers_to_remove = []
                for layer in self.viewer.layers:
                    if "Comparison" in layer.name:
                        layers_to_remove.append(layer)

                for layer in layers_to_remove:
                    self.viewer.layers.remove(layer)

                # Add new comparison layer
                comparison_layer = self.viewer.add_image(
                    comparison_image,
                    name="ROI Correspondence Comparison",
                    colormap="gray",
                    visible=True,
                )

                comparison_layer.metadata.update(
                    {"dataset_type": "comparison", "workflow_step": "roi_comparison"}
                )

                self._log_message("Added ROI comparison view")
                self._log_message(
                    "Orange circles = Calibration ROIs, Green circles = Main ROIs"
                )
                self._log_message(
                    "Verify that ROI numbers correspond to same physical locations"
                )
            else:
                self._log_message(
                    "Cannot create comparison - missing ROI data from one or both datasets"
                )

        except Exception as e:
            self._log_message(f"Error creating comparison view: {e}")

    def switch_to_main_dataset(self):
        """Switch back to main dataset for analysis."""
        if self.current_dataset_type == "calibration" and self.main_dataset_path:
            try:
                # Store calibration state
                self.calibration_masks = self.masks.copy()
                self.calibration_labeled_frame = self.labeled_frame

                # Restore main dataset
                self.current_dataset_type = "main"
                self.file_path = self.main_dataset_path
                self.masks = self.main_masks.copy()
                self.labeled_frame = self.main_labeled_frame

                # Reload main dataset in viewer
                reader = napari_get_reader(self.main_dataset_path)
                if reader:
                    self.viewer.layers.clear()
                    layers = reader(self.main_dataset_path)
                    for data, meta, layer_type in layers:
                        name = meta.get(
                            "name", os.path.basename(self.main_dataset_path)
                        )
                        kwargs = {k: v for k, v in meta.items() if k not in ("name",)}

                        if layer_type == "image":
                            self.viewer.add_image(data, name=name, **kwargs)
                        elif layer_type == "labels":
                            self.viewer.add_labels(data, name=name, **kwargs)

                    # Re-add main dataset ROIs if they exist
                    if self.masks:
                        self._add_roi_layers_to_viewer(self.labeled_frame, self.masks)

                self.lbl_file_info.setText(
                    f"MAIN DATASET: {os.path.basename(self.main_dataset_path)}"
                )
                self._log_message("Switched back to main dataset")

            except Exception as e:
                self._log_message(f"Error switching to main dataset: {e}")

    def clear_roi_detection(self):
        """Enhanced ROI detection clearing with proper event disconnection."""
        try:
            self._log_message("Clear ROI Detection button clicked")

            layers_to_remove = []
            for layer in self.viewer.layers:
                # Check if this is an ROI layer
                is_roi_layer = (
                    "ROI" in layer.name
                    or "Detected" in layer.name
                    or (
                        hasattr(layer, "metadata")
                        and layer.metadata.get("roi_type") == "circular_detection"
                    )
                )

                if is_roi_layer:
                    self._log_message(f"  Marking layer for removal: {layer.name}")
                    layers_to_remove.append(layer)

            if len(layers_to_remove) == 0:
                self._log_message("No ROI layers found to remove")
                # Still clear variables in case they exist
                self.masks = []
                self.labeled_frame = None
                return

            # Disconnect any connected events before removing layers
            for layer in layers_to_remove:
                try:
                    # Disconnect contrast events if they exist
                    if hasattr(layer, "events"):
                        layer.events.contrast_limits.disconnect()
                except Exception:
                    pass  # Event might not be connected

                self.viewer.layers.remove(layer)
                self._log_message(f"  Removed layer: {layer.name}")

            # Clear all ROI-related variables
            self.masks = []
            self.labeled_frame = None
            self.main_masks = []
            self.main_labeled_frame = None
            self.calibration_masks = []
            self.calibration_labeled_frame = None

            self._log_message(
                f"✓ Successfully removed {len(layers_to_remove)} ROI layers and cleaned up all ROI data"
            )

        except Exception as e:
            self._log_message(f"ERROR clearing ROI detection: {e}")
            import traceback

            self._log_message(traceback.format_exc())

    def _add_roi_layers_to_viewer(self, labeled_frame, masks, dataset_type):
        """Add ROI layers with clear dataset identification."""
        try:
            if dataset_type == "CALIBRATION":
                file_name = os.path.basename(self.calibration_file_path_stored)
                layer_name = f"CALIBRATION - {file_name} - ROIs ({len(masks)})"
                colormap = "gray"
            else:
                file_name = os.path.basename(self.file_path)
                layer_name = f"MAIN - {file_name} - ROIs ({len(masks)})"
                colormap = "gray"

            # Add ROI detection layer
            roi_layer = self.viewer.add_image(
                labeled_frame,
                name=layer_name,
                colormap=colormap,
                visible=True,
                opacity=0.8,
            )

            # Store metadata
            roi_layer.metadata.update(
                {
                    "dataset_type": dataset_type.lower(),
                    "file_path": (
                        self.calibration_file_path_stored
                        if dataset_type == "CALIBRATION"
                        else self.file_path
                    ),
                    "roi_count": len(masks),
                    "workflow_step": "roi_detection",
                    "analysis_ready": True,
                }
            )

            self._log_message(f"Added {dataset_type} ROI layer: {len(masks)} ROIs")

        except Exception as e:
            self._log_message(f"Error adding ROI layer: {e}")

    def _get_roi_center(self, mask):
        """Calculate the center of an ROI mask."""
        try:
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                return (int(center_x), int(center_y))
        except Exception:
            pass
        return (0, 0)

    def _get_roi_radius(self, mask):
        """Calculate the radius of an ROI mask."""
        try:
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                distances = np.sqrt(
                    (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                )
                radius = np.mean(distances)
                return radius
        except Exception:
            pass
        return 0.0

    # ===================================================================
    # SIMPLIFIED ANALYSIS EXECUTION - NOW USING _calc.py
    # ===================================================================

    def run_analysis(self):
        """Start analysis using the separated calculation module."""
        if not self.masks:
            self.status_label.setText(
                "Error: No ROIs detected. Please run ROI detection first."
            )
            self._log_message(
                "ERROR: No ROIs detected. Please run ROI detection first."
            )
            return

        # Quick validation using calc module
        is_valid, error_msg = validate_analysis_parameters(
            self.frame_interval.value(),
            self.chunk_size.value(),
            self.baseline_duration_minutes.value(),
        )

        if not is_valid:
            self.status_label.setText(f"Error: {error_msg}")
            self._log_message(f"ERROR: {error_msg}")
            return

        # UI state management
        self._cancel_requested = False
        self.analysis_start_time = time.time()
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing analysis...")
        self.performance_timer.start()

        # Log analysis start
        self._log_analysis_parameters()

        @thread_worker(start_thread=False)
        def _analysis_worker():
            return self._run_analysis_with_calc_module()

        worker_instance = _analysis_worker()
        worker_instance.returned.connect(self._analysis_finished)
        worker_instance.errored.connect(self._analysis_errored)
        worker_instance.finished.connect(self._analysis_done)
        worker_instance.start()
        self.current_worker = worker_instance

    def load_calibration_file(self):
        """Enhanced calibration file loading with workflow state management."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Calibration File",
            "",
            "Video Files (*.h5 *.hdf5 *.avi);;HDF5 Files (*.h5 *.hdf5);;AVI Files (*.avi);;All Files (*.*)",
        )
        if file_path:
            self.calibration_file_path.setText(os.path.basename(file_path))
            self.calibration_file_path.setProperty("full_path", file_path)
            self.calibration_file_path_stored = file_path

            # Enable the load calibration dataset button
            self.btn_load_calibration_dataset.setEnabled(True)

            # Update status
            self.calibration_status_label.setText(
                "✅ 1. Calibration file selected\n"
                "2. Click 'Load Calibration Dataset'\n"
                "3. Detect ROIs (Input tab)\n"
                "4. Process baseline"
            )

            # Reset calibration state
            self.calibration_baseline_processed = False
            self.calibration_baseline_statistics = {}
            self.btn_process_calibration_baseline.setEnabled(False)

            self._log_message(
                f"Calibration file selected: {os.path.basename(file_path)}"
            )
        else:
            self.calibration_file_path.setText("No calibration file selected")
            self.calibration_file_path.setProperty("full_path", None)
            self.calibration_file_path_stored = None
            self.btn_load_calibration_dataset.setEnabled(False)

            self.calibration_status_label.setText(
                "1. Select calibration file\n"
                "2. Load calibration dataset\n"
                "3. Detect ROIs (Input tab)\n"
                "4. Process baseline"
            )

    # def load_calibration_dataset(self):
    #     """Load calibration dataset into viewer for ROI detection - DEBUG VERSION."""

    #     # IMMEDIATE DEBUG OUTPUT
    #     self._log_message("=== LOAD_CALIBRATION_DATASET METHOD CALLED ===")
    #     print("LOAD_CALIBRATION_DATASET METHOD CALLED")  # Also print to console

    #     # Check prerequisites
    #     self._log_message(f"calibration_file_path_stored: {getattr(self, 'calibration_file_path_stored', 'NOT_SET')}")

    #     if not hasattr(self, 'calibration_file_path_stored') or not self.calibration_file_path_stored:
    #         self._log_message("ERROR: No calibration file selected")
    #         return

    #     self._log_message(f"Calibration file path: {self.calibration_file_path_stored}")

    #     if not os.path.exists(self.calibration_file_path_stored):
    #         self._log_message(f"ERROR: Calibration file not found: {self.calibration_file_path_stored}")
    #         return

    #     self._log_message("File exists, proceeding with calibration dataset loading...")

    #     try:
    #         # Check current state
    #         current_type = getattr(self, 'current_dataset_type', 'NOT_SET')
    #         current_file = getattr(self, 'file_path', 'NOT_SET')

    #         self._log_message(f"Before switch - current_dataset_type: {current_type}")
    #         self._log_message(f"Before switch - file_path: {current_file}")

    #         # Store main dataset state FIRST
    #         if current_type == "main" or current_type == 'NOT_SET':
    #             self.main_dataset_path = current_file
    #             self.main_masks = getattr(self, 'masks', []).copy()
    #             self.main_labeled_frame = getattr(self, 'labeled_frame', None)
    #             self._log_message("Stored main dataset state")

    #         # Switch to calibration dataset
    #         self.current_dataset_type = "calibration"
    #         self.file_path = self.calibration_file_path_stored
    #         self.directory = None

    #         self._log_message(f"After switch - current_dataset_type: {self.current_dataset_type}")
    #         self._log_message(f"After switch - file_path: {self.file_path}")

    #         # Clear current ROI detection
    #         self.masks = []
    #         self.labeled_frame = None
    #         self._log_message("Cleared ROI detection state")

    #         # Check if reader exists
    #         try:
    #             from ._reader import napari_get_reader
    #             reader = napari_get_reader(self.calibration_file_path_stored)
    #             self._log_message(f"Reader obtained: {reader is not None}")
    #         except Exception as reader_error:
    #             self._log_message(f"ERROR getting reader: {reader_error}")
    #             return

    #         if reader is None:
    #             self._log_message("ERROR: Cannot read calibration file - no valid reader")
    #             return

    #         # Clear viewer
    #         self._log_message(f"Clearing viewer - current layers: {len(self.viewer.layers)}")
    #         self.viewer.layers.clear()
    #         self._log_message("Viewer cleared")

    #         # Load calibration data
    #         self._log_message("Loading calibration layers...")
    #         try:
    #             layers = reader(self.calibration_file_path_stored)
    #             self._log_message(f"Reader returned {len(layers)} layers")

    #             for i, (data, meta, layer_type) in enumerate(layers):
    #                 name = f"CALIBRATION - {meta.get('name', os.path.basename(self.calibration_file_path_stored))}"
    #                 kwargs = {k: v for k, v in meta.items() if k not in ("name",)}

    #                 self._log_message(f"Adding layer {i}: {layer_type} - {name}")

    #                 if layer_type == "image":
    #                     layer = self.viewer.add_image(data, name=name, **kwargs)
    #                     self._log_message(f"Added image layer: {layer.name}")
    #                 elif layer_type == "labels":
    #                     layer = self.viewer.add_labels(data, name=name, **kwargs)
    #                     self._log_message(f"Added labels layer: {layer.name}")
    #                 else:
    #                     self._log_message(f"Unknown layer type: {layer_type}")

    #         except Exception as layer_error:
    #             self._log_message(f"ERROR loading layers: {layer_error}")
    #             import traceback
    #             self._log_message(f"Layer loading traceback: {traceback.format_exc()}")
    #             return

    #         # Update file info
    #         basename = os.path.basename(self.calibration_file_path_stored)
    #         self.lbl_file_info.setText(f"CALIBRATION DATASET: {basename}")
    #         self._log_message(f"Updated file info to: CALIBRATION DATASET: {basename}")

    #         # Update status
    #         if hasattr(self, 'calibration_status_label'):
    #             self.calibration_status_label.setText(
    #                 "✅ 1. Calibration file selected\n"
    #                 "✅ 2. Calibration dataset loaded\n"
    #                 "3. Detect ROIs (Input tab)\n"
    #                 "4. Process baseline"
    #             )
    #             self._log_message("Updated calibration status label")
    #         else:
    #             self._log_message("WARNING: No calibration_status_label found")

    #         self._log_message("=== CALIBRATION DATASET LOADING COMPLETE ===")
    #         self._log_message("Next step: Go to Input tab and click 'Detect ROIs'")

    #     except Exception as e:
    #         self._log_message(f"CRITICAL ERROR in load_calibration_dataset: {e}")
    #         import traceback
    #         self._log_message(f"Full traceback: {traceback.format_exc()}")
    #         # Reset to main dataset on error
    #         self.current_dataset_type = "main"
    def enhanced_load_calibration_dataset(self):
        """Load calibration dataset while preserving main dataset."""
        self._log_message("=== LOADING CALIBRATION DATASET ===")

        if (
            not hasattr(self, "calibration_file_path_stored")
            or not self.calibration_file_path_stored
        ):
            self._log_message("ERROR: No calibration file selected")
            return

        try:
            # CRITICAL: Store main dataset BEFORE any calibration operations
            if not getattr(self, "main_dataset_stored", False):
                if not self.store_main_dataset_state():
                    self._log_message("ERROR: Failed to store main dataset state")
                    self._log_message(
                        "Cannot proceed with calibration without preserving main dataset"
                    )
                    return

            # Switch to calibration mode
            self.current_dataset_type = "calibration"

            # Load calibration first frame (don't change self.file_path)
            if DUAL_STRUCTURE_AVAILABLE:
                from ._reader import get_first_frame_enhanced

                first_frame, structure_info = get_first_frame_enhanced(
                    self.calibration_file_path_stored
                )
            else:
                from ._reader import get_first_frame

                first_frame = get_first_frame(self.calibration_file_path_stored)

            if first_frame is None:
                self._log_message("ERROR: Could not read calibration first frame")
                return

            self._log_message(f"Loaded calibration first frame: {first_frame.shape}")

            # Add calibration layer
            basename = os.path.basename(self.calibration_file_path_stored)
            layer_name = f"CALIBRATION - {basename} (First Frame)"

            cal_layer = self.viewer.add_image(
                first_frame, name=layer_name, colormap="plasma", visible=True
            )

            cal_layer.metadata.update(
                {
                    "dataset_type": "calibration",
                    "file_path": self.calibration_file_path_stored,
                    "workflow_step": "first_frame_loaded",
                }
            )

            # Update UI
            self.lbl_file_info.setText(
                f"CALIBRATION: {basename} (Main dataset preserved)"
            )

            # Update status
            if hasattr(self, "calibration_status_label"):
                self.calibration_status_label.setText(
                    "✅ 1. Calibration file selected\n"
                    "✅ 2. Calibration first frame loaded\n"
                    "3. Detect ROIs (Input tab)\n"
                    "4. Process baseline\n"
                    "5. Run analysis on main dataset"
                )

            self._log_message("Calibration first frame loaded (main dataset preserved)")

        except Exception as e:
            self._log_message(f"ERROR loading calibration dataset: {e}")
            self.current_dataset_type = "main"

    def restore_main_dataset_for_analysis(self):
        """
        CRITICAL: Restore main dataset before running analysis.
        This ensures analysis runs on the correct (main) dataset.
        """
        self._log_message("=== RESTORING MAIN DATASET FOR ANALYSIS ===")

        if not hasattr(self, "main_dataset_stored") or not self.main_dataset_stored:
            self._log_message("WARNING: No main dataset stored - analysis may fail")
            return False

        try:
            # Restore main dataset state
            self.file_path = self.main_dataset_path
            self.merged_results = self.main_merged_results.copy()
            self.masks = self.main_masks.copy()
            self.labeled_frame = self.main_labeled_frame
            self.current_dataset_type = "main"

            # Verify restoration
            if self.merged_results:
                sample_roi = list(self.merged_results.keys())[0]
                sample_data = self.merged_results[sample_roi]
                if sample_data:
                    restored_duration = (sample_data[-1][0] - sample_data[0][0]) / 60
                    self._log_message("✅ Main dataset restored:")
                    self._log_message(f"   Path: {os.path.basename(self.file_path)}")
                    self._log_message(f"   ROIs: {len(self.merged_results)}")
                    self._log_message(f"   Duration: {restored_duration:.1f} minutes")
                    self._log_message(f"   Data points: {len(sample_data)}")
                    return True

            self._log_message("ERROR: Restored dataset appears empty")
            return False

        except Exception as e:
            self._log_message(f"ERROR restoring main dataset: {e}")
            return False

    def _apply_automatic_timing_fix(
        self, merged_results: Dict[int, List[Tuple[float, float]]]
    ) -> Tuple[Dict, bool]:
        """
        Automatically detect and fix timing issues in HDF5 data.

        Returns:
            (corrected_merged_results, was_corrected)
        """
        if not merged_results:
            return merged_results, False

        # Get sample data to analyze timing
        sample_roi = list(merged_results.keys())[0]
        sample_data = merged_results[sample_roi]

        if len(sample_data) < 3:
            self._log_message("Insufficient data for timing analysis")
            return merged_results, False

        # Calculate actual intervals from timestamps
        times = [t for t, _ in sample_data[:20]]  # Use first 20 points
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        actual_interval = np.median(intervals)
        expected_interval = self.frame_interval.value()
        interval_std = np.std(intervals)

        self._log_message("🔍 AUTOMATIC TIMING ANALYSIS:")
        self._log_message(f"  Expected interval: {expected_interval:.1f}s")
        self._log_message(f"  Detected interval: {actual_interval:.1f}s")
        self._log_message(f"  Interval std: {interval_std:.2f}s")

        # Check if correction is needed (tolerance: 1 second or 10% of expected)
        tolerance = max(1.0, expected_interval * 0.1)
        needs_correction = abs(actual_interval - expected_interval) > tolerance

        if not needs_correction:
            self._log_message("✅ Timing is correct - no correction needed")
            return merged_results, False

        # AUTOMATIC CORRECTION
        correction_factor = actual_interval / expected_interval
        self._log_message("⚠️  TIMING MISMATCH DETECTED - Applying automatic fix")
        self._log_message(f"  Correction factor: {correction_factor:.2f}x")

        # 1. Update frame interval
        self.frame_interval.setValue(actual_interval)
        self._log_message(
            f"✅ Updated frame interval: {expected_interval:.1f}s → {actual_interval:.1f}s"
        )

        # 2. Adjust baseline duration (keep same number of frames)
        original_baseline_min = self.baseline_duration_minutes.value()
        baseline_frames = int((original_baseline_min * 60) / expected_interval)
        corrected_baseline_min = (baseline_frames * actual_interval) / 60
        self.baseline_duration_minutes.setValue(corrected_baseline_min)
        self._log_message(
            f"✅ Adjusted baseline: {original_baseline_min:.1f}min → {corrected_baseline_min:.1f}min ({baseline_frames} frames)"
        )

        # 3. Adjust bin size for fraction movement
        original_bin = self.bin_size_seconds.value()
        frames_per_bin = max(1, round(original_bin / actual_interval))
        corrected_bin = frames_per_bin * actual_interval
        self.bin_size_seconds.setValue(int(corrected_bin))
        self._log_message(
            f"✅ Adjusted bin size: {original_bin}s → {corrected_bin:.0f}s ({frames_per_bin} frames/bin)"
        )

        # 4. Update plot time ranges if they exist
        if hasattr(self, "plot_end_time"):
            self.update_end_time()  # This will recalculate based on new frame interval

        self._log_message("🎉 AUTOMATIC TIMING CORRECTION COMPLETE!")

        return merged_results, True

    def _extract_analysis_parameters(self) -> Dict[str, Any]:
        """Enhanced parameter extraction that handles calibration method specifics."""
        # Determine threshold method from active tab
        threshold_method = self._get_current_threshold_method()

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
            "num_processes": self.num_processes.value(),
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
                "✅ 4. Calibration baseline processed"
                if success
                else "4. Process baseline"
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

    def _run_analysis_with_calc_module(self):
        """Simplified analysis using integration system."""
        try:
            # Determine method
            method_text = self._get_current_threshold_method_display()
            method = (
                "calibration"
                if "Calibration" in method_text
                else "baseline" if "Baseline" in method_text else "adaptive"
            )

            # Determine file and masks to use
            if (
                method == "calibration"
                and hasattr(self, "main_dataset_path")
                and self.main_dataset_path
            ):
                file_to_process = self.main_dataset_path
                masks_to_use = getattr(self, "main_masks", [])
            else:
                file_to_process = self.file_path
                masks_to_use = self.masks

            if not file_to_process or not masks_to_use:
                raise RuntimeError("No file or masks available for processing")

            # Progress callback
            def progress_callback(percent, msg):
                if self._cancel_requested:
                    raise RuntimeError("Analysis canceled")
                self.progress_updated.emit(int(percent))
                self.status_updated.emit(msg)

            # Check if this is an AVI batch
            if hasattr(self, "avi_batch_paths") and self.avi_batch_paths:
                self._log_message("Processing AVI batch - loading all frames...")
                _, merged_results, _ = self._process_avi_batch_for_analysis(
                    self.avi_batch_paths,
                    masks_to_use,
                    self.chunk_size.value(),
                    progress_callback,
                    self.avi_batch_interval,
                )
            else:
                # Process complete dataset using reader (HDF5)
                _, merged_results, _ = process_single_file_in_parallel_dual_structure(
                    file_to_process,
                    masks_to_use,
                    self.chunk_size.value(),
                    progress_callback,
                    self.frame_interval.value(),
                    self.num_processes.value(),
                )

            self.merged_results = merged_results

            # Apply timing correction
            merged_results, _ = self._apply_automatic_timing_fix(merged_results)

            # Get analysis parameters
            analysis_params = self._extract_analysis_parameters()
            if method == "calibration":
                analysis_params["calibration_baseline_statistics"] = (
                    self.calibration_baseline_statistics
                )

            # Use integration system
            return run_analysis_with_method(merged_results, method, **analysis_params)

        except Exception as e:
            self._log_message(f"Analysis error: {e}")
            raise

    def _log_analysis_parameters(self):
        """Log analysis parameters for debugging."""
        self._log_message("=" * 50)
        self._log_message("STARTING ANALYSIS WITH _calc.py MODULE")
        self._log_message(f"ROIs: {len(self.masks)}")
        self._log_message(f"Processes: {self.num_processes.value()}")
        self._log_message(f"Chunk size: {self.chunk_size.value()}")
        self._log_message(f"Method: {self._get_current_threshold_method_display()}")
        self._log_message(f"Frame interval: {self.frame_interval.value()}s")
        self._log_message(
            f"Baseline duration: {self.baseline_duration_minutes.value()} min"
        )
        self._log_message(f"Threshold multiplier: {self.threshold_multiplier.value()}")
        self._log_message("MATLAB normalization: Enabled")
        self._log_message("HDF5 timing correction: Enabled")
        self._log_message("=" * 50)

    def _log_timing_analysis_with_units(self, timing_info):
        """Log timing analysis with clear unit display."""
        self._log_message("TIMING ANALYSIS (units clarified):")

        if "mean_frame_drift" in timing_info:
            drift_s = timing_info["mean_frame_drift"]
            drift_ms = drift_s * 1000
            self._log_message(f"  Mean Frame Drift: {drift_s:.3f}s ({drift_ms:.1f}ms)")

        if "max_frame_drift" in timing_info:
            max_drift_s = timing_info["max_frame_drift"]
            max_drift_ms = max_drift_s * 1000
            self._log_message(
                f"  Max Frame Drift: {max_drift_s:.3f}s ({max_drift_ms:.1f}ms)"
            )

        if "std_frame_drift" in timing_info:
            std_drift_s = timing_info["std_frame_drift"]
            std_drift_ms = std_drift_s * 1000
            self._log_message(
                f"  Drift Std Dev: {std_drift_s:.3f}s ({std_drift_ms:.1f}ms)"
            )

    def stop_analysis(self):
        """Stop analysis with proper cleanup."""
        self._cancel_requested = True
        self.status_label.setText("Stopping analysis...")
        self._log_message("STOP requested by user")

        # Stop performance monitoring
        self.performance_timer.stop()

        # Reset UI state
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _analysis_finished(self, result: Dict[str, Any]):
        """Handle successful analysis completion using results from _calc.py."""
        try:
            # Capture start time immediately before it can be cleared by _analysis_done()
            start_time = self.analysis_start_time

            # Store results from _calc.py - USE CORRECT KEY
            self.merged_results = result.get(
                "processed_data", {}
            )  # Changed from 'detrended_data'
            self.roi_baseline_means = result.get("baseline_means", {})
            self.roi_upper_thresholds = result.get("upper_thresholds", {})
            self.roi_lower_thresholds = result.get("lower_thresholds", {})
            self.roi_statistics = result.get("roi_statistics", {})
            self.movement_data = result.get("movement_data", {})
            self.fraction_data = result.get("fraction_data", {})
            self.sleep_data = result.get("sleep_data", {})

            # Get ROI colors from calc results
            self.roi_colors = result.get("roi_colors", {})

            # Fallback if no ROI colors provided
            if not self.roi_colors and self.merged_results:
                self.roi_colors = {
                    roi: f"C{i}"
                    for i, roi in enumerate(sorted(self.merged_results.keys()))
                }

            self._log_message(f"✅ ROI colors set: {self.roi_colors}")

            # Calculate quiescence data using _calc.py functions
            if self.fraction_data:
                self.quiescence_data = bin_quiescence(
                    self.fraction_data, self.quiescence_threshold.value()
                )

            # Calculate band widths for plotting compatibility
            self.roi_band_widths = {}
            for roi in self.roi_baseline_means:
                if (
                    roi in self.roi_upper_thresholds
                    and roi in self.roi_lower_thresholds
                ):
                    upper = self.roi_upper_thresholds[roi]
                    lower = self.roi_lower_thresholds[roi]
                    self.roi_band_widths[roi] = (upper - lower) / 2

            # Calculate performance metrics using _calc.py
            total_frames = (
                sum(len(data) for data in self.merged_results.values())
                if self.merged_results
                else 0
            )
            performance_metrics = get_performance_metrics(start_time, total_frames)

            # Generate summary using _calc.py
            summary = get_analysis_summary(result)
            self.status_label.setText(
                f"Analysis completed: {performance_metrics['fps']:.1f} fps"
            )
            self.results_label.setText("Analysis completed successfully")

            # Log completion with summary
            self._log_message("=" * 60)
            self._log_message("ANALYSIS COMPLETED SUCCESSFULLY")
            self._log_message(f"Processing rate: {performance_metrics['fps']:.1f} fps")
            self._log_message(f"ROIs processed: {len(self.merged_results)}")
            self._log_message(f"Total data points: {total_frames}")
            self._log_message("ANALYSIS SUMMARY:")
            for line in summary.split("\n"):
                if line.strip():
                    self._log_message(line)

            # Log timing diagnostics if available
            timing_info = result.get("timing_diagnostics", {})
            if timing_info:
                self._log_message(
                    f"HDF5 Timing: {timing_info.get('timing_type', 'unknown')}"
                )
                self._log_message(
                    f"Timing correction: {timing_info.get('needs_hdf5_correction', False)}"
                )

        except Exception as e:
            self._log_message(f"Error in analysis completion: {str(e)}")
            import traceback

            self._log_message(f"Full traceback: {traceback.format_exc()}")

    def _analysis_errored(self, exc):
        """Handle analysis errors."""
        self.performance_timer.stop()
        if "canceled" in str(exc).lower():
            self.status_label.setText("Analysis canceled by user.")
            self._log_message("Analysis CANCELED by user")
        else:
            self.status_label.setText(f"Analysis error: {exc}")
            self._log_message(f"ERROR: {exc}")

        # Reset UI state
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)

    def _analysis_done(self):
        """Cleanup after analysis completion or cancellation."""
        self.performance_timer.stop()
        self._cancel_requested = False
        self.analysis_start_time = None

        # Reset UI state
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ===================================================================
    # TESTING AND DIAGNOSTICS USING _calc.py
    # ===================================================================

    def run_quick_analysis_test(self):
        """Run quick analysis test using _calc.py module."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("No data available for testing")
            return

        try:
            test_summary = quick_method_test(self.merged_results)
            self._log_message("QUICK ANALYSIS TEST RESULTS:")
            for line in test_summary.split("\n"):
                if line.strip():
                    self._log_message(line)
        except Exception as e:
            self._log_message(f"Quick test failed: {e}")

    def validate_hdf5_timing(self):
        """Validate HDF5 timing using _calc.py module."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("No data available for timing validation")
            return

        try:
            timing_diagnostics = validate_hdf5_timing_in_data(
                self.merged_results, self.frame_interval.value()
            )

            self._log_message("HDF5 TIMING DIAGNOSTICS:")
            self._log_message(f"Timing type: {timing_diagnostics['timing_type']}")
            self._log_message(
                f"First timestamp: {timing_diagnostics['first_time']:.1f}s"
            )
            self._log_message(
                f"Average interval: {timing_diagnostics['avg_interval']:.1f}s"
            )
            self._log_message(
                f"Expected interval: {timing_diagnostics['expected_interval']:.1f}s"
            )
            self._log_message(
                f"Interval consistent: {timing_diagnostics['interval_consistent']}"
            )
            self._log_message(
                f"Needs correction: {timing_diagnostics['needs_hdf5_correction']}"
            )
            self._log_message(
                f"Recommendation: {timing_diagnostics['recommended_action']}"
            )

        except Exception as e:
            self._log_message(f"Timing validation failed: {e}")

    # ===================================================================
    # PLOTTING METHODS - NOW USING _plot.py MODULE
    # ===================================================================

    def generate_plot(self):
        """Generate plot using PlotGenerator."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.results_label.setText("No analysis results to plot.")
            return

        # ROI colors should now come from _calc.py, but add safety check
        if not hasattr(self, "roi_colors") or not self.roi_colors:
            self._log_message("No ROI colors from analysis, creating fallback")
            self.roi_colors = {
                roi: f"C{i}" for i, roi in enumerate(sorted(self.merged_results.keys()))
            }

        # Force clear the canvas to prevent artifacts
        try:
            if hasattr(self, "canvas") and self.canvas:
                self.canvas.clear()

                # Clear any cached renderers
                if hasattr(self.canvas.figure, "_cachedRenderer"):
                    self.canvas.figure._cachedRenderer = None

                # Force immediate refresh
                self.canvas.draw_idle()
                self.canvas.flush_events()
        except Exception as e:
            self._log_message(f"Canvas cleanup warning: {e}")

        # Initialize plot generator if needed
        if self.plot_generator is None:
            try:
                from ._plot import PlotGenerator

                self.plot_generator = PlotGenerator(self.figure)
            except Exception as e:
                self._log_message(f"Plot generator init failed: {e}")
                self.results_label.setText("Plot system initialization failed.")
                return

        plot_type = self.plot_type_combo.currentText()

        try:
            # Get data based on plot type
            if plot_type == "Raw Intensity Changes":
                data_dict = self.merged_results
                from ._plot import create_hysteresis_kwargs

                kwargs = create_hysteresis_kwargs(widget_instance=self)
                # Remove merged_results from kwargs to avoid duplicate argument
                kwargs.pop("merged_results", None)

            elif plot_type == "Movement":
                data_dict = getattr(self, "movement_data", {})
                kwargs = {}

            elif plot_type == "Fraction Movement":
                data_dict = getattr(self, "fraction_data", {})
                kwargs = {}

            elif plot_type == "Quiescence":
                data_dict = getattr(self, "quiescence_data", {})
                kwargs = {}

            elif plot_type == "Sleep":
                data_dict = getattr(self, "sleep_data", {})
                kwargs = {}

            elif plot_type == "Lighting Conditions (dark IR)":
                data_dict = getattr(self, "fraction_data", {})
                from ._plot import create_hysteresis_kwargs

                kwargs = create_hysteresis_kwargs(
                    widget_instance=self
                )  # Keep merged_results for lighting
                kwargs.update({"bin_minutes": 10})  # Smaller bins for smoother curves

                # Extract LED data from HDF5 if available
                led_data = self._extract_led_data_from_hdf5()
                if led_data:
                    kwargs["led_data"] = led_data
                    self._log_message(
                        f"Using LED data from HDF5: {len(led_data.get('times', []))} data points"
                    )

            else:
                self.results_label.setText(f"Unknown plot type: {plot_type}")
                return

            if not data_dict:
                self.results_label.setText(f"No {plot_type.lower()} data available.")
                return

            # Create plot config
            from ._plot import create_plot_config

            plot_config = create_plot_config(self)

            # Generate plot
            success = self.plot_generator.generate_plot(
                plot_type, data_dict, self.roi_colors, plot_config, **kwargs
            )

            if success:
                # Force complete canvas refresh
                self.canvas.draw()
                self.canvas.flush_events()
                self.results_label.setText(f"Generated {plot_type} plot successfully.")
                self._log_message(f"Generated {plot_type} plot")
            else:
                self.results_label.setText(f"Failed to generate {plot_type} plot.")
                self._log_message("Plot generation returned False")
                # Add debugging info
                self._log_message(f"Debug - Plot type: {plot_type}")
                self._log_message(
                    f"Debug - Data dict keys: {list(data_dict.keys()) if data_dict else 'None'}"
                )
                self._log_message(
                    f"Debug - ROI colors: {len(self.roi_colors) if self.roi_colors else 0} colors"
                )

        except Exception as e:
            self._log_message(f"Plot error: {e}")
            self.results_label.setText(f"Plot generation failed: {str(e)}")
            import traceback

            self._log_message(f"Traceback: {traceback.format_exc()}")

    def _extract_led_data_from_hdf5(self):
        """Extract LED power timeseries from HDF5 or AVI file.

        Light phase = white LED ON (alone or with IR LED)
        Dark phase = only IR LED ON (white LED OFF)

        Returns:
            dict or None: Dictionary with 'times', 'white_powers', and 'ir_powers' keys if LED data exists,
                         None if no LED data is available
        """
        try:
            if not hasattr(self, "file_path") or not self.file_path:
                return None

            # Check if this is an AVI file
            if self.file_path.lower().endswith(".avi"):
                return self._extract_led_data_from_avi()

            # HDF5 file processing
            with h5py.File(self.file_path, "r") as f:
                if "timeseries" not in f:
                    return None

                timeseries = f["timeseries"]

                # Try to find white LED data (various possible names)
                white_led = None
                white_led_names = [
                    "led_white_power_percent",
                    "white_led_power",
                    "led_white_power",
                    "white_led_power_percent",
                    "led_power_percent",
                ]
                for name in white_led_names:
                    if name in timeseries:
                        white_led = timeseries[name][:]
                        self._log_message(f"Found white LED data: {name}")
                        break

                # Try to find IR LED data (various possible names)
                ir_led = None
                ir_led_names = [
                    "led_ir_power_percent",
                    "ir_led_power",
                    "led_ir_power",
                    "ir_led_power_percent",
                ]
                for name in ir_led_names:
                    if name in timeseries:
                        ir_led = timeseries[name][:]
                        self._log_message(f"Found IR LED data: {name}")
                        break

                # If no separate LED channels found, return None
                if white_led is None:
                    self._log_message("No white LED data found in HDF5 timeseries")
                    return None

                # Get timestamps (try capture_timestamps first, fallback to calculated times)
                if "capture_timestamps" in timeseries:
                    times = timeseries["capture_timestamps"][:]
                else:
                    # Fallback: use frame interval to calculate times
                    frame_interval = self.frame_interval.value()
                    times = np.arange(len(white_led)) * frame_interval

                result = {"times": times.tolist(), "white_powers": white_led.tolist()}

                if ir_led is not None:
                    result["ir_powers"] = ir_led.tolist()

                return result

        except Exception as e:
            self._log_message(f"Could not extract LED data from HDF5: {e}")
            return None

    def _extract_led_data_from_avi(self):
        """AVI files don't contain LED data.

        LED data is only available in HDF5 files.
        For AVIs, plots will not show lighting conditions.

        Returns:
            None (AVIs don't have LED data)
        """
        self._log_message(
            "AVI files don't contain LED data - lighting conditions will not be shown in plots"
        )
        return None

    def apply_time_range(self):
        """Apply time range and regenerate plot."""
        self.generate_plot()

    def save_current_plot(self):
        """Save the current plot using _plot.py module."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )

        if file_path:
            dpi = self.plot_dpi_spin.value()
            success = save_plot(self.figure, file_path, dpi)

            if success:
                self._log_message(f"Plot saved: {os.path.basename(file_path)}")
                self.results_label.setText(f"Plot saved: {os.path.basename(file_path)}")
            else:
                error_msg = "Failed to save plot"
                self.results_label.setText(error_msg)
                self._log_message(f"ERROR: {error_msg}")

    def save_all_plots(self):
        """Save all plot types using _plot.py module."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save All Plots"
        )

        if not directory:
            return

        try:
            # Prepare all data sets
            data_sets = {
                "merged_results": getattr(self, "merged_results", {}),
                "movement_data": getattr(self, "movement_data", {}),
                "fraction_data": getattr(self, "fraction_data", {}),
                "quiescence_data": getattr(self, "quiescence_data", {}),
                "sleep_data": getattr(self, "sleep_data", {}),
            }

            # Create plot configuration
            plot_config = create_plot_config(self)

            # Generate timestamp
            timestamp = str(int(time.time()))

            # Save all plots using the separated module
            saved_files = save_all_plot_types(
                self.plot_generator,
                data_sets,
                self.roi_colors,
                plot_config,
                directory,
                timestamp,
            )

            if saved_files:
                self.results_label.setText(
                    f"Saved {len(saved_files)} plots to {directory}"
                )
                self._log_message(f"Saved {len(saved_files)} plots successfully")

                # Restore original plot type
                self.generate_plot()
            else:
                self.results_label.setText("No plots were saved")
                self._log_message("WARNING: No plots were saved")

        except Exception as e:
            error_msg = f"Error saving plots: {str(e)}"
            self.results_label.setText(error_msg)
            self._log_message(f"ERROR: {error_msg}")

    # ===================================================================
    # EXPORT AND RESULTS METHODS
    # ===================================================================

    def check_hdf5_structure(self):
        """Enhanced HDF5 structure checking with dual structure support."""
        if not self.file_path:
            self._log_message("No file loaded")
            return

        # Skip structure check for AVI files
        if self.file_path.lower().endswith(".avi") or (
            hasattr(self, "avi_batch_paths") and self.avi_batch_paths
        ):
            self._log_message("AVI file(s) loaded - skipping HDF5 structure check")
            return

        import h5py

        try:
            if DUAL_STRUCTURE_AVAILABLE:
                # Use enhanced structure detection
                structure_info = detect_hdf5_structure_type(self.file_path)

                self._log_message("=== ENHANCED HDF5 FILE STRUCTURE ANALYSIS ===")
                self._log_message(f"Structure type: {structure_info['type']}")

                if structure_info["type"] == "stacked_frames":
                    self._log_message("✅ Stacked frames structure")
                    self._log_message(f"   Dataset: {structure_info['dataset_name']}")
                    self._log_message(
                        f"   Frame count: {structure_info['frame_count']}"
                    )
                    self._log_message(
                        f"   Frame shape: {structure_info['frame_shape']}"
                    )

                elif structure_info["type"] == "individual_frames":
                    self._log_message("✅ Individual frames structure")
                    self._log_message(f"   Group: {structure_info['group_name']}")
                    self._log_message(
                        f"   Frame count: {structure_info['frame_count']}"
                    )
                    self._log_message(
                        f"   Frame shape: {structure_info['frame_shape']}"
                    )
                    self._log_message(
                        f"   Sample keys: {structure_info['frame_keys'][:5]}..."
                    )

                elif structure_info["type"] == "alternative_dataset":
                    self._log_message("✅ Alternative dataset structure")
                    self._log_message(f"   Dataset: {structure_info['dataset_name']}")
                    self._log_message(
                        f"   Frame count: {structure_info['frame_count']}"
                    )

                elif structure_info["type"] == "error":
                    self._log_message(
                        f"❌ Structure detection failed: {structure_info['error']}"
                    )

            else:
                # Fallback to original method
                with h5py.File(self.file_path, "r") as f:
                    self._log_message("=== BASIC HDF5 FILE STRUCTURE ===")
                    self._log_message(f"Root keys: {list(f.keys())}")

                    def print_structure(name, obj):
                        if isinstance(obj, h5py.Group):
                            self._log_message(
                                f"Group: {name} - keys: {list(obj.keys())}"
                            )
                        elif isinstance(obj, h5py.Dataset):
                            self._log_message(f"Dataset: {name} - shape: {obj.shape}")

                    f.visititems(print_structure)

        except Exception as e:
            self._log_message(f"HDF5 structure check failed: {e}")

    def _create_basic_matlab_export(
        self, analysis_results: Dict[str, Any], output_dir: str
    ) -> List[str]:
        """Fallback MATLAB export when modern system not available."""
        import csv
        import json
        from datetime import datetime

        created_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Export basic CSV for MATLAB
            csv_file = os.path.join(output_dir, f"matlab_export_{timestamp}.csv")

            with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # Header
                writer.writerow(["# MATLAB Export"])
                writer.writerow(
                    [f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
                )
                writer.writerow([])

                # ROI summary
                writer.writerow(
                    ["ROI_ID", "Baseline_Mean", "Upper_Threshold", "Lower_Threshold"]
                )

                baseline_means = analysis_results.get("baseline_means", {})
                upper_thresholds = analysis_results.get("upper_thresholds", {})
                lower_thresholds = analysis_results.get("lower_thresholds", {})

                for roi in sorted(baseline_means.keys()):
                    writer.writerow(
                        [
                            roi,
                            baseline_means.get(roi, 0),
                            upper_thresholds.get(roi, 0),
                            lower_thresholds.get(roi, 0),
                        ]
                    )

            created_files.append(csv_file)

            # Export parameters as JSON
            json_file = os.path.join(output_dir, f"matlab_parameters_{timestamp}.json")
            with open(json_file, "w") as f:
                json.dump(
                    analysis_results.get("parameters", {}), f, indent=2, default=str
                )

            created_files.append(json_file)

        except Exception as e:
            print(f"Error in basic MATLAB export: {e}")

        return created_files

    def export_results_for_matlab_compatibility(self):
        """Export analysis results for MATLAB compatibility."""
        # 1) Check preconditions
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.results_label.setText("No results to export.")
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory for MATLAB Export"
        )
        if not directory:
            return

        try:
            # 2) Prepare analysis object
            analysis_results = {
                "merged_results": getattr(self, "merged_results", {}),
                "baseline_means": getattr(self, "roi_baseline_means", {}),
                "upper_thresholds": getattr(self, "roi_upper_thresholds", {}),
                "lower_thresholds": getattr(self, "roi_lower_thresholds", {}),
                "movement_data": getattr(self, "movement_data", {}),
                "fraction_data": getattr(self, "fraction_data", {}),
                "sleep_data": getattr(self, "sleep_data", {}),
                "parameters": {
                    "threshold_method": self._get_current_threshold_method_display(),
                    "frame_interval": self.frame_interval.value(),
                    "enable_matlab_norm": True,
                    "enable_detrending": self.enable_detrending.isChecked(),
                },
            }

            # Add method-specific parameters
            if hasattr(self, "baseline_duration_minutes"):
                analysis_results["parameters"][
                    "baseline_duration_minutes"
                ] = self.baseline_duration_minutes.value()
            if hasattr(self, "threshold_multiplier"):
                analysis_results["parameters"][
                    "threshold_multiplier"
                ] = self.threshold_multiplier.value()
            if hasattr(self, "calibration_multiplier"):
                analysis_results["parameters"][
                    "calibration_multiplier"
                ] = self.calibration_multiplier.value()

            # 3) Try modern export
            created_files = []
            try:
                from ._calc_integration import export_results_for_matlab as _export_fn

                created_files = _export_fn(analysis_results, directory)
            except Exception:
                _export_fn = None

            # 4) Fallback export if modern system not available
            if not created_files:
                created_files = self._create_basic_matlab_export(
                    analysis_results, directory
                )

            # 5) UI feedback
            if created_files:
                self.results_label.setText(
                    f"Exported {len(created_files)} files for MATLAB"
                )
                self._log_message(
                    f"MATLAB export completed: {len(created_files)} files"
                )
                for p in created_files:
                    self._log_message(f"  Created: {os.path.basename(p)}")
            else:
                self.results_label.setText("Export failed")
                self._log_message("MATLAB export failed")

        except Exception as e:
            err = f"Error exporting for MATLAB: {e}"
            self.results_label.setText(err)
            self._log_message(f"ERROR: {err}")

    def save_results_consolidated_complete(self):
        """
        UPDATED VERSION: Complete consolidated save function with all sheets.
        This replaces the previous save_results_consolidated method.
        """
        import os
        import time

        # Check if analysis results are available
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.results_label.setText(
                "❌ No analysis results to save. Run analysis first."
            )
            self._log_message("Save failed: No analysis results available")
            return

        # Check if we have behavioral analysis data
        has_behavioral_data = (
            hasattr(self, "movement_data")
            and self.movement_data
            and hasattr(self, "fraction_data")
            and self.fraction_data
        )

        if not has_behavioral_data:
            self.results_label.setText("⚠️ Incomplete analysis data detected.")
            self._log_message("Warning: Saving with incomplete behavioral analysis")

        # Get base filename from user
        from qtpy.QtWidgets import QFileDialog

        base_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Results",
            f"analysis_results_{int(time.time())}",  # No extension - we'll add them
            "All Files (*)",
        )

        if not base_path:
            self._log_message("Save cancelled by user")
            return

        # Remove any extension from base_path to ensure clean naming
        base_path = os.path.splitext(base_path)[0]

        saved_files = []
        sheets_created = []

        try:
            # === SAVE CSV VERSION ===
            csv_path = f"{base_path}.csv"
            self._log_message(f"Saving CSV version: {csv_path}")

            try:
                self._save_results_csv(csv_path)
                saved_files.append(("CSV", csv_path))
                self._log_message("✅ CSV saved successfully")
            except Exception as e:
                self._log_message(f"❌ CSV save failed: {e}")

            # === SAVE COMPLETE EXCEL VERSION (if possible) ===
            try:
                import pandas as pd
                import openpyxl

                excel_path = f"{base_path}.xlsx"
                self._log_message(
                    f"Saving complete Excel version with all sheets: {excel_path}"
                )

                # Use the complete Excel save method
                self._save_results_excel_to_path(excel_path)
                saved_files.append(("Excel", excel_path))

                # Count sheets created
                wb = openpyxl.load_workbook(excel_path)
                sheets_created = wb.sheetnames
                wb.close()

                self._log_message(
                    f"✅ Excel saved with {len(sheets_created)} sheets: {', '.join(sheets_created)}"
                )

            except ImportError:
                self._log_message(
                    "⚠️ Excel export not available (missing pandas/openpyxl)"
                )
                self._log_message("   Install with: pip install pandas openpyxl")
            except Exception as e:
                self._log_message(f"❌ Excel save failed: {e}")
                import traceback

                self._log_message(f"Traceback: {traceback.format_exc()}")

            # === SHOW THRESHOLD STATS IN LOG ===
            if hasattr(self, "roi_baseline_means") and self.roi_baseline_means:
                self._log_message("\n" + "=" * 50)
                self._log_message("THRESHOLD STATISTICS (included with save)")
                self._log_message("=" * 50)
                self._show_threshold_stats_in_log()

            # === UPDATE UI WITH RESULTS ===
            if saved_files:
                if sheets_created:
                    file_list = f"CSV + Excel ({len(sheets_created)} sheets: {', '.join(sheets_created)})"
                else:
                    file_list = ", ".join(
                        [
                            f"{fmt} ({os.path.basename(path)})"
                            for fmt, path in saved_files
                        ]
                    )

                self.results_label.setText(f"✅ Saved: {file_list}")
                self._log_message(
                    f"\n🎉 SAVE COMPLETE: {len(saved_files)} files created"
                )

                # Show success dialog with file details
                self._show_save_success_dialog_complete(saved_files, sheets_created)
            else:
                self.results_label.setText(
                    "❌ All save attempts failed - check log for details"
                )
                self._log_message("❌ No files were saved successfully")

        except Exception as e:
            error_msg = f"Save operation failed: {e}"
            self.results_label.setText(error_msg)
            self._log_message(f"❌ {error_msg}")
            import traceback

            self._log_message(f"Traceback: {traceback.format_exc()}")

    def _show_save_success_dialog_complete(self, saved_files, sheets_created):
        """Show success dialog with complete file and sheet details."""
        from qtpy.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("Save Complete")
        msg.setText(
            f"Successfully saved analysis results in {len(saved_files)} format(s):"
        )

        file_details = []
        for file_format, file_path in saved_files:
            filename = os.path.basename(file_path)
            try:
                size_bytes = os.path.getsize(file_path)
                if size_bytes > 1024 * 1024:
                    file_size = f"{size_bytes/(1024*1024):.1f} MB"
                elif size_bytes > 1024:
                    file_size = f"{size_bytes/1024:.1f} KB"
                else:
                    file_size = f"{size_bytes} bytes"
            except:
                file_size = "Unknown size"

            file_details.append(f"• {file_format}: {filename} ({file_size})")

        if sheets_created:
            file_details.append("")
            file_details.append("Excel Sheets Created:")
            for sheet in sheets_created:
                file_details.append(f"  - {sheet}")

        msg.setDetailedText("\n".join(file_details))
        msg.setInformativeText("Analysis results saved successfully")
        msg.exec_()

    def add_nematostella_analysis_to_widget(widget_instance):
        """
        Add Nematostella-specific analysis capabilities to the existing widget.
        This function can be called from the widget to enable enhanced analysis.

        Args:
            widget_instance: Instance of HDF5AnalysisWidget
        """
        # Add new button to widget if it doesn't exist
        if not hasattr(widget_instance, "btn_nematostella_analysis"):
            from qtpy.QtWidgets import QPushButton

            widget_instance.btn_nematostella_analysis = QPushButton(
                "Nematostella Timeseries Analysis"
            )
            widget_instance.btn_nematostella_analysis.setToolTip(
                "Run specialized Nematostella timeseries analysis"
            )
            widget_instance.btn_nematostella_analysis.setStyleSheet(
                "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; }"
            )

            # Add to the existing results tab layout
            if hasattr(widget_instance, "tab_results"):
                layout = widget_instance.tab_results.layout()
                if layout:
                    # Find the plot buttons group and add after it
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if (
                            item
                            and hasattr(item.widget(), "title")
                            and "Controls" in item.widget().title()
                        ):
                            # Insert after plot controls group
                            layout.insertWidget(
                                i + 1, widget_instance.btn_nematostella_analysis
                            )
                            break
                    else:
                        # Fallback: add at the end
                        layout.addWidget(widget_instance.btn_nematostella_analysis)

            # Connect the button
            widget_instance.btn_nematostella_analysis.clicked.connect(
                lambda: run_nematostella_analysis_from_widget(widget_instance)
            )

    def run_nematostella_analysis_from_widget(widget_instance):
        """
        Run Nematostella analysis from within the napari widget.

        Args:
            widget_instance: Instance of HDF5AnalysisWidget
        """
        if not hasattr(widget_instance, "file_path") or not widget_instance.file_path:
            widget_instance._log_message(
                "No HDF5 file loaded for Nematostella analysis"
            )
            widget_instance.results_label.setText("Error: No HDF5 file loaded")
            return

        try:
            widget_instance._log_message("Starting Nematostella timeseries analysis...")
            widget_instance.results_label.setText("Running Nematostella analysis...")

            # Get quick summary first
            summary = get_nematostella_timeseries_summary(widget_instance.file_path)
            widget_instance._log_message("Timeseries summary:")
            for line in summary.split("\n"):
                if line.strip():
                    widget_instance._log_message(f"  {line}")

            # Run full analysis
            results = analyze_nematostella_hdf5_file(widget_instance.file_path)

            if results["success"]:
                widget_instance._log_message(
                    "Nematostella analysis completed successfully!"
                )
                widget_instance._log_message(
                    f"Excel file created: {results['excel_file']}"
                )
                widget_instance._log_message(
                    f"Report file created: {results['report_file']}"
                )
                widget_instance._log_message(
                    f"Sheets created: {', '.join(results['sheets_created'])}"
                )

                # Update results display
                widget_instance.results_label.setText(
                    f"Nematostella analysis complete: {len(results['sheets_created'])} Excel sheets created"
                )

                # Log key findings from report
                widget_instance._log_message("Key Analysis Results:")
                report_lines = results["report"].split("\n")
                in_important_section = False
                for line in report_lines:
                    if any(
                        section in line
                        for section in [
                            "## Timing Analysis",
                            "## LED System Analysis",
                            "## Environmental Conditions",
                        ]
                    ):
                        in_important_section = True
                        widget_instance._log_message(line)
                    elif line.startswith("##") and in_important_section:
                        in_important_section = False
                    elif in_important_section and line.strip().startswith("-"):
                        widget_instance._log_message(f"  {line.strip()}")

            else:
                widget_instance._log_message(
                    f"Nematostella analysis failed: {results['error']}"
                )
                widget_instance.results_label.setText(
                    f"Analysis failed: {results['error']}"
                )

        except Exception as e:
            error_msg = f"Nematostella analysis error: {e}"
            widget_instance._log_message(error_msg)
            widget_instance.results_label.setText(error_msg)

    def _show_save_success_dialog_with_metadata(
        self, saved_files, metadata_dict, nematostella_results=None
    ):
        """Show success dialog with metadata details and optional Nematostella analysis."""
        from qtpy.QtWidgets import QMessageBox

        msg = QMessageBox(self)

        # Adjust title and message based on whether Nematostella analysis was performed
        if nematostella_results and nematostella_results["success"]:
            msg.setWindowTitle("Save Complete: Metadata + Nematostella Analysis")
            msg.setText(
                f"Successfully saved metadata with specialized Nematostella timeseries analysis in {len(saved_files)} format(s):"
            )
        else:
            msg.setWindowTitle("Save with Metadata Complete")
            msg.setText(
                f"Successfully saved analysis results with metadata in {len(saved_files)} format(s):"
            )

        # Count metadata statistics
        total_static_params = 0
        total_timeseries_params = 0
        total_timeseries_points = 0

        for source_name, metadata in metadata_dict.items():
            # Skip Nematostella analysis entry for counting (it's not traditional metadata)
            if source_name == "nematostella_analysis":
                continue

            # Count static parameters
            static_data = {k: v for k, v in metadata.items() if k != "timeseries_data"}
            total_static_params += len(static_data)

            # Count time-series parameters
            if "timeseries_data" in metadata and metadata["timeseries_data"]:
                ts_data = metadata["timeseries_data"]
                total_timeseries_params += len(ts_data)
                for param_data in ts_data.values():
                    if hasattr(param_data, "__len__"):
                        total_timeseries_points = max(
                            total_timeseries_points, len(param_data)
                        )

        file_details = []
        for file_format, file_path in saved_files:
            filename = os.path.basename(file_path)
            file_size = "Unknown size"
            try:
                size_bytes = os.path.getsize(file_path)
                if size_bytes > 1024 * 1024:  # > 1MB
                    file_size = f"{size_bytes/(1024*1024):.1f} MB"
                elif size_bytes > 1024:  # > 1KB
                    file_size = f"{size_bytes/1024:.1f} KB"
                else:
                    file_size = f"{size_bytes} bytes"
            except:
                pass

            file_details.append(f"• {file_format}: {filename} ({file_size})")

        # Add Nematostella analysis files if available
        if nematostella_results and nematostella_results["success"]:
            file_details.append("")
            file_details.append("Nematostella Analysis Files:")

            # Add Excel analysis file
            excel_file = nematostella_results["excel_file"]
            try:
                excel_size_bytes = os.path.getsize(excel_file)
                if excel_size_bytes > 1024 * 1024:
                    excel_size = f"{excel_size_bytes/(1024*1024):.1f} MB"
                elif excel_size_bytes > 1024:
                    excel_size = f"{excel_size_bytes/1024:.1f} KB"
                else:
                    excel_size = f"{excel_size_bytes} bytes"
            except:
                excel_size = "Unknown size"

            file_details.append(
                f"• Excel Analysis: {os.path.basename(excel_file)} ({excel_size})"
            )

            # Add text report file
            report_file = nematostella_results["report_file"]
            try:
                report_size_bytes = os.path.getsize(report_file)
                if report_size_bytes > 1024:
                    report_size = f"{report_size_bytes/1024:.1f} KB"
                else:
                    report_size = f"{report_size_bytes} bytes"
            except:
                report_size = "Unknown size"

            file_details.append(
                f"• Text Report: {os.path.basename(report_file)} ({report_size})"
            )
            file_details.append(
                f"• Analysis Sheets: {len(nematostella_results['sheets_created'])}"
            )

        # Add metadata summary
        file_details.append("")
        file_details.append("Metadata Summary:")
        file_details.append(f"• Static parameters: {total_static_params}")
        file_details.append(f"• Time-series parameters: {total_timeseries_params}")
        if total_timeseries_points > 0:
            file_details.append(
                f"• Time-series length: {total_timeseries_points} time points"
            )
            duration_min = (
                total_timeseries_points * self.frame_interval.value()
            ) / 60.0
            file_details.append(f"• Total duration: {duration_min:.1f} minutes")

        # Add Nematostella analysis summary if available
        if nematostella_results and nematostella_results["success"]:
            file_details.append("")
            file_details.append("Nematostella Analysis Summary:")

            # Extract key metrics from the analysis results
            if (
                "analysis_results" in nematostella_results
                and nematostella_results["analysis_results"]
            ):
                analysis_results = nematostella_results["analysis_results"]

                # Timing analysis summary
                if "timing_analysis" in analysis_results:
                    timing = analysis_results["timing_analysis"]
                    if "timing" in timing:
                        accuracy = timing["timing"]["timing_accuracy"]
                        file_details.append(f"• Timing Accuracy: {accuracy:.1%}")

                # Environmental stability
                if (
                    "env_analysis" in analysis_results
                    and analysis_results["env_analysis"]
                ):
                    env = analysis_results["env_analysis"]["environment"]
                    if "temperature" in env:
                        temp_range = env["temperature"]["range"]
                        file_details.append(
                            f"• Temperature Stability: ±{temp_range/2:.1f}°C"
                        )

                # LED system performance
                if (
                    "led_analysis" in analysis_results
                    and analysis_results["led_analysis"]
                ):
                    led = analysis_results["led_analysis"]
                    if "led_sync" in led:
                        sync_rate = led["led_sync"]["success_rate"]
                        file_details.append(f"• LED Sync Success: {sync_rate:.1%}")

        msg.setDetailedText("\n".join(file_details))

        # Adjust informative text based on analysis type
        if nematostella_results and nematostella_results["success"]:
            msg.setInformativeText(
                "Files include comprehensive HDF5 metadata AND specialized Nematostella timeseries analysis with timing, environmental, and LED system evaluation."
            )
        else:
            msg.setInformativeText(
                "Files include comprehensive HDF5 metadata in time-series format matching analysis data structure."
            )

        msg.exec_()
        # Adjust informative text based on analysis type
        if nematostella_results and nematostella_results["success"]:
            msg.setInformativeText(
                "Files include comprehensive HDF5 metadata AND specialized Nematostella timeseries analysis with timing, environmental, and LED system evaluation."
            )
        else:
            msg.setInformativeText(
                "Files include comprehensive HDF5 metadata in time-series format matching analysis data structure."
            )

        msg.exec_()

    def _show_threshold_stats_in_log(self):
        """
        HELPER METHOD: Show threshold statistics in the log.
        This replaces the separate "Show Threshold Stats" button functionality.
        """
        if not hasattr(self, "roi_baseline_means") or not self.roi_baseline_means:
            self._log_message("No threshold statistics available")
            return

        # Generate threshold statistics for log
        method = self._get_current_threshold_method_display()
        self._log_message(f"Method: {method}")

        if hasattr(self, "baseline_duration_minutes"):
            self._log_message(
                f"Baseline Duration: {self.baseline_duration_minutes.value():.1f} minutes"
            )
        if hasattr(self, "threshold_multiplier"):
            self._log_message(
                f"Hysteresis Multiplier: {self.threshold_multiplier.value():.2f}"
            )
        elif hasattr(self, "calibration_multiplier"):
            self._log_message(
                f"Calibration Multiplier: {self.calibration_multiplier.value():.2f}"
            )

        self._log_message(
            f"Detrending: {'Enabled' if getattr(self, 'enable_detrending', False) and self.enable_detrending.isChecked() else 'Disabled'}"
        )

        roi_band_widths = getattr(self, "roi_band_widths", {})
        roi_upper_thresholds = getattr(self, "roi_upper_thresholds", {})
        roi_lower_thresholds = getattr(self, "roi_lower_thresholds", {})

        # Show statistics for first 5 ROIs to avoid log spam
        rois_to_show = sorted(self.roi_baseline_means.keys())[:5]

        for roi in rois_to_show:
            baseline_mean = self.roi_baseline_means[roi]
            band_width = roi_band_widths.get(roi, 0)
            upper_threshold = roi_upper_thresholds.get(roi, baseline_mean + band_width)
            lower_threshold = roi_lower_thresholds.get(roi, baseline_mean - band_width)

            self._log_message(f"\nROI {roi} HYSTERESIS SYSTEM:")
            self._log_message(f"  Baseline Mean: {baseline_mean:.3f}")
            self._log_message(f"  Band Width: ±{band_width:.3f}")
            self._log_message(
                f"  Upper Threshold: {upper_threshold:.3f} (Movement = TRUE when above)"
            )
            self._log_message(
                f"  Lower Threshold: {lower_threshold:.3f} (Movement = FALSE when below)"
            )
            self._log_message(
                f"  Hysteresis Zone: {lower_threshold:.3f} to {upper_threshold:.3f} (State unchanged)"
            )

        if len(self.roi_baseline_means) > 5:
            self._log_message(f"\n... and {len(self.roi_baseline_means) - 5} more ROIs")

    def _show_save_success_dialog(self, saved_files):
        """
        HELPER METHOD: Show success dialog with list of saved files.
        """
        from qtpy.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("Save Results Complete")
        msg.setText(
            f"Successfully saved analysis results in {len(saved_files)} format(s):"
        )

        file_details = []
        for file_format, file_path in saved_files:
            filename = os.path.basename(file_path)
            file_size = "Unknown size"
            try:
                size_bytes = os.path.getsize(file_path)
                if size_bytes > 1024 * 1024:  # > 1MB
                    file_size = f"{size_bytes/(1024*1024):.1f} MB"
                elif size_bytes > 1024:  # > 1KB
                    file_size = f"{size_bytes/1024:.1f} KB"
                else:
                    file_size = f"{size_bytes} bytes"
            except:
                pass

            file_details.append(f"• {file_format}: {filename} ({file_size})")

        msg.setDetailedText("\n".join(file_details))
        msg.setInformativeText(
            "Files can be opened in Excel, analyzed further, or imported into other analysis software."
        )
        msg.exec_()

    def _save_results_excel_to_path(self, excel_path: str):
        """
        COMPLETE METHOD: Save Excel results with ALL sheets to a specific file path.
        Creates the same multi-sheet structure as shown in the screenshot.
        """
        try:
            import pandas as pd

            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

                # === SHEET 1: SUMMARY ===
                summary_data = []
                sorted_rois = sorted(self.merged_results.keys())

                for roi in sorted_rois:
                    row_data = {
                        "ROI": roi,
                        "Baseline Mean": getattr(self, "roi_baseline_means", {}).get(
                            roi, 0
                        ),
                        "Upper Threshold": getattr(
                            self, "roi_upper_thresholds", {}
                        ).get(roi, 0),
                        "Lower Threshold": getattr(
                            self, "roi_lower_thresholds", {}
                        ).get(roi, 0),
                        "Threshold Band Width": getattr(
                            self, "roi_band_widths", {}
                        ).get(roi, 0),
                    }

                    # Calculate movement statistics
                    movement_data = getattr(self, "movement_data", {})
                    if roi in movement_data and movement_data[roi]:
                        movement_values = [m for _, m in movement_data[roi]]
                        row_data["Total Movement Events"] = sum(movement_values)
                        row_data["Movement Percentage"] = (
                            (sum(movement_values) / len(movement_values) * 100)
                            if movement_values
                            else 0
                        )

                    # Calculate sleep statistics
                    sleep_data = getattr(self, "sleep_data", {})
                    if roi in sleep_data and sleep_data[roi]:
                        sleep_values = [s for _, s in sleep_data[roi]]
                        total_sleep_bins = sum(sleep_values)
                        row_data["Total Sleep Bins"] = total_sleep_bins
                        row_data["Sleep Time (min)"] = (
                            total_sleep_bins * self.bin_size_seconds.value()
                        ) / 60

                    summary_data.append(row_data)

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # === SHEET 2: RAW INTENSITY ===
                if hasattr(self, "merged_results") and self.merged_results:
                    intensity_df = self._create_time_series_dataframe(
                        self.merged_results,
                        sorted_rois,
                        "Intensity",
                        convert_to_minutes=True,
                    )
                    intensity_df.to_excel(
                        writer, sheet_name="Raw_Intensity", index=False
                    )

                # === SHEET 3: MOVEMENT ===
                if hasattr(self, "movement_data") and self.movement_data:
                    movement_df = self._create_time_series_dataframe(
                        self.movement_data,
                        sorted_rois,
                        "Movement",
                        convert_to_minutes=True,
                    )
                    movement_df.to_excel(writer, sheet_name="Movement", index=False)

                # === SHEET 4: FRACTION MOVEMENT ===
                if hasattr(self, "fraction_data") and self.fraction_data:
                    fraction_df = self._create_time_series_dataframe(
                        self.fraction_data,
                        sorted_rois,
                        "Fraction",
                        convert_to_minutes=True,
                    )
                    fraction_df.to_excel(
                        writer, sheet_name="Fraction_Movement", index=False
                    )

                # === SHEET 5: SLEEP ===
                if hasattr(self, "sleep_data") and self.sleep_data:
                    sleep_df = self._create_time_series_dataframe(
                        self.sleep_data, sorted_rois, "Sleep", convert_to_minutes=True
                    )
                    sleep_df.to_excel(writer, sheet_name="Sleep", index=False)

                # === SHEET 6: LIGHTING CONDITIONS ===
                if hasattr(self, "fraction_data") and self.fraction_data:
                    # Create lighting conditions data (binned activity for circadian analysis)
                    try:
                        # Use 30-minute bins for lighting analysis
                        from ._calc import bin_activity_data_for_lighting

                        lighting_data = bin_activity_data_for_lighting(
                            self.fraction_data, bin_minutes=30
                        )

                        if lighting_data:
                            lighting_df = self._create_time_series_dataframe(
                                lighting_data,
                                sorted_rois,
                                "Activity_30min_bins",
                                convert_to_minutes=True,
                            )
                            lighting_df.to_excel(
                                writer, sheet_name="Lighting_Conditions", index=False
                            )
                    except Exception as e:
                        self._log_message(
                            f"Warning: Could not create lighting conditions sheet: {e}"
                        )

                # === SHEET 7: PARAMETERS ===
                # Determine source type (HDF5 or AVI)
                is_avi = hasattr(self, "avi_batch_paths") and self.avi_batch_paths
                source_label = "AVI Batch" if is_avi else "HDF5"

                params_data = {
                    "Parameter": [
                        "Data Source Type",
                        "Analysis Method",
                        "Frame Interval (s)",
                        "Baseline Duration (min)",
                        "Threshold Multiplier",
                        "Detrending Enabled",
                        "Jump Correction Enabled",
                        "Bin Size (s)",
                        "Quiescence Threshold",
                        "Sleep Threshold (min)",
                        "Number of ROIs",
                        "Total Analysis Time (min)",
                        "Generated Date",
                        "Software Version",
                    ],
                    "Value": [
                        source_label,
                        self._get_current_threshold_method_display(),
                        self.frame_interval.value(),
                        (
                            getattr(self, "baseline_duration_minutes", {}).value()
                            if hasattr(self, "baseline_duration_minutes")
                            else "N/A"
                        ),
                        (
                            getattr(self, "threshold_multiplier", {}).value()
                            if hasattr(self, "threshold_multiplier")
                            else "N/A"
                        ),
                        (
                            getattr(self, "enable_detrending", {}).isChecked()
                            if hasattr(self, "enable_detrending")
                            else "N/A"
                        ),
                        (
                            getattr(self, "enable_jump_correction", {}).isChecked()
                            if hasattr(self, "enable_jump_correction")
                            else "N/A"
                        ),
                        self.bin_size_seconds.value(),
                        self.quiescence_threshold.value(),
                        self.sleep_threshold_minutes.value(),
                        len(sorted_rois),
                        (
                            self.plot_end_time.value()
                            if hasattr(self, "plot_end_time")
                            else "N/A"
                        ),
                        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "HDF5 Analysis Widget v1.0",
                    ],
                    "Description": [
                        "Type of data source (HDF5 file or AVI batch)",
                        "Threshold calculation method used",
                        "Time interval between frames",
                        "Duration of baseline period for threshold calculation",
                        "Multiplier for hysteresis band width",
                        "Whether detrending was applied to remove drift",
                        "Whether jump correction was applied",
                        "Time bin size for fraction movement calculation",
                        "Threshold below which animal is considered quiescent",
                        "Minimum continuous quiescence duration for sleep",
                        "Total number of ROIs analyzed",
                        "Total duration of analysis",
                        "When this file was generated",
                        "Software version and name",
                    ],
                }

                # Add AVI-specific parameters if applicable
                if is_avi and hasattr(self, "avi_batch_paths"):
                    params_data["Parameter"].extend(
                        [
                            "Number of AVI Files",
                            "AVI Start Time (s)",
                            "AVI End Time (s)",
                            "Total Duration (min)",
                        ]
                    )

                    # Calculate total duration from merged_results
                    total_duration_s = 0
                    start_time_s = 0
                    if self.merged_results:
                        first_roi = next(iter(self.merged_results.values()))
                        if first_roi:
                            start_time_s = first_roi[0][0]
                            end_time_s = first_roi[-1][0]
                            total_duration_s = end_time_s - start_time_s

                    params_data["Value"].extend(
                        [
                            len(self.avi_batch_paths),
                            f"{start_time_s:.1f}",
                            f"{end_time_s:.1f}",
                            f"{total_duration_s / 60:.1f}",
                        ]
                    )

                    params_data["Description"].extend(
                        [
                            "Number of AVI video files processed",
                            "Start time of the analysis (in seconds)",
                            "End time of the analysis (in seconds)",
                            "Total duration of the video sequence (in minutes)",
                        ]
                    )

                params_df = pd.DataFrame(params_data)
                params_df.to_excel(writer, sheet_name="Parameters", index=False)

        except Exception as e:
            raise Exception(f"Complete Excel save error: {e}")

    def _save_results_csv(self, file_path: str):
        """Save results in clear CSV format."""
        try:
            sorted_rois = sorted(self.merged_results.keys())

            # Create main data CSV
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # === HEADER SECTION ===
                # Determine source type (HDF5 or AVI)
                is_avi = hasattr(self, "avi_batch_paths") and self.avi_batch_paths
                source_label = "AVI" if is_avi else "HDF5"
                writer.writerow([f"{source_label} Analysis Results"])
                writer.writerow(
                    [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
                )
                writer.writerow(
                    [f"Analysis Method: {self._get_current_threshold_method_display()}"]
                )
                writer.writerow(
                    [f"Frame Interval: {self.frame_interval.value()} seconds"]
                )
                writer.writerow([f"Number of ROIs: {len(sorted_rois)}"])
                writer.writerow([])  # Empty row

                # === ROI SUMMARY TABLE ===
                writer.writerow(["ROI SUMMARY"])
                summary_headers = [
                    "ROI",
                    "Baseline Mean",
                    "Upper Threshold",
                    "Lower Threshold",
                    "Movement %",
                    "Sleep Time (min)",
                ]
                writer.writerow(summary_headers)

                roi_baseline_means = getattr(self, "roi_baseline_means", {})
                roi_upper_thresholds = getattr(self, "roi_upper_thresholds", {})
                roi_lower_thresholds = getattr(self, "roi_lower_thresholds", {})
                movement_data = getattr(self, "movement_data", {})
                sleep_data = getattr(self, "sleep_data", {})

                for roi in sorted_rois:
                    # Calculate statistics
                    movement_pct = 0
                    if roi in movement_data and movement_data[roi]:
                        movement_values = [m for _, m in movement_data[roi]]
                        movement_pct = (
                            (sum(movement_values) / len(movement_values) * 100)
                            if movement_values
                            else 0
                        )

                    sleep_minutes = 0
                    if roi in sleep_data and sleep_data[roi]:
                        sleep_values = [s for _, s in sleep_data[roi]]
                        total_sleep_bins = sum(sleep_values)
                        sleep_minutes = (
                            total_sleep_bins * self.bin_size_seconds.value()
                        ) / 60

                    writer.writerow(
                        [
                            roi,
                            f"{roi_baseline_means.get(roi, 0):.3f}",
                            f"{roi_upper_thresholds.get(roi, 0):.3f}",
                            f"{roi_lower_thresholds.get(roi, 0):.3f}",
                            f"{movement_pct:.1f}",
                            f"{sleep_minutes:.1f}",
                        ]
                    )

                writer.writerow([])  # Empty row
                writer.writerow([])  # Empty row

                # === TIME SERIES DATA ===
                writer.writerow(["RAW INTENSITY DATA (Time in minutes)"])

                # Create time-aligned data
                all_times = set()
                for roi_data in self.merged_results.values():
                    for time_point, _ in roi_data:
                        all_times.add(round(time_point / 60.0, 2))  # Convert to minutes

                sorted_times = sorted(all_times)

                # Header row: Time, ROI1, ROI2, ROI3, ...
                header = ["Time (min)"] + [f"ROI_{roi}" for roi in sorted_rois]
                writer.writerow(header)

                # Create data rows
                for time_min in sorted_times:
                    row = [f"{time_min:.2f}"]

                    for roi in sorted_rois:
                        # Find value at this time point
                        value = None
                        if roi in self.merged_results:
                            for t, v in self.merged_results[roi]:
                                if abs(t / 60.0 - time_min) < 0.01:  # Within tolerance
                                    value = v
                                    break

                        row.append(f"{value:.6f}" if value is not None else "")

                    writer.writerow(row)

            self._log_message(f"CSV saved: {file_path}")

        except Exception as e:
            self._log_message(f"Error in CSV export: {e}")
            raise

    def _create_time_series_dataframe(
        self,
        data_dict: Dict,
        sorted_rois: List[int],
        data_type: str,
        convert_to_minutes: bool = True,
    ):
        """
        Create a pandas DataFrame with clear time-series structure for Excel export.
        This method already exists in your code but here's the complete version to ensure compatibility.
        """

        # Collect all unique time points
        all_times = set()
        for roi_data in data_dict.values():
            for time_point, _ in roi_data:
                if convert_to_minutes:
                    all_times.add(round(time_point / 60.0, 2))  # Round to 2 decimals
                else:
                    all_times.add(round(time_point, 2))

        sorted_times = sorted(all_times)

        # Create DataFrame structure
        df_data = {"Time (min)" if convert_to_minutes else "Time (s)": sorted_times}

        # Add data for each ROI
        for roi in sorted_rois:
            roi_values = []

            # Create time->value mapping for this ROI
            time_value_map = {}
            if roi in data_dict:
                for time_point, value in data_dict[roi]:
                    if convert_to_minutes:
                        time_key = round(time_point / 60.0, 2)
                    else:
                        time_key = round(time_point, 2)
                    time_value_map[time_key] = value

            # Fill values for all time points
            for time_point in sorted_times:
                if time_point in time_value_map:
                    roi_values.append(time_value_map[time_point])
                else:
                    roi_values.append(None)  # Missing data as None

            # Column name format: ROI_1, ROI_2, etc.
            df_data[f"ROI_{roi}"] = roi_values

        return pd.DataFrame(df_data)

    def save_results(self):
        """Save analysis results - let user choose format."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.results_label.setText("No results to save.")
            return

        # Ask user for format
        file_path, file_type = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".xlsx") or "Excel" in file_type:
                self.save_results_excel_format()
            else:
                # Default to CSV
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                self._save_results_csv(file_path)
                self.results_label.setText(
                    f"Results saved to {os.path.basename(file_path)}"
                )
                self._log_message(f"Results saved: {file_path}")

        except Exception as e:
            self.results_label.setText(f"Error saving results: {str(e)}")
            self._log_message(f"ERROR saving results: {str(e)}")

    def show_threshold_statistics(self):
        """Show detailed hysteresis statistics."""
        if not hasattr(self, "roi_baseline_means") or not self.roi_baseline_means:
            self._log_message("No hysteresis statistics available")
            return

        # Create statistics summary
        stats_text = "HYSTERESIS DETECTION SYSTEM STATISTICS\n"
        stats_text += "=" * 50 + "\n\n"

        # Add method parameters
        stats_text += f"Method: {self._get_current_threshold_method_display()}\n"
        if hasattr(self, "baseline_duration_minutes"):
            stats_text += f"Baseline Duration: {self.baseline_duration_minutes.value():.1f} minutes\n"
        if hasattr(self, "threshold_multiplier"):
            stats_text += (
                f"Hysteresis Multiplier: {self.threshold_multiplier.value():.2f}\n"
            )
        elif hasattr(self, "calibration_multiplier"):
            stats_text += (
                f"Calibration Multiplier: {self.calibration_multiplier.value():.2f}\n"
            )
        stats_text += f"Detrending: {'Enabled' if self.enable_detrending.isChecked() else 'Disabled'}\n\n"

        roi_band_widths = getattr(self, "roi_band_widths", {})
        roi_upper_thresholds = getattr(self, "roi_upper_thresholds", {})
        roi_lower_thresholds = getattr(self, "roi_lower_thresholds", {})
        roi_statistics = getattr(self, "roi_statistics", {})

        for roi in sorted(self.roi_baseline_means.keys()):
            baseline_mean = self.roi_baseline_means[roi]
            band_width = roi_band_widths.get(roi, 0)
            stats = roi_statistics.get(roi, {})

            upper_threshold = roi_upper_thresholds.get(roi, baseline_mean + band_width)
            lower_threshold = roi_lower_thresholds.get(roi, baseline_mean - band_width)

            stats_text += f"ROI {roi} - HYSTERESIS SYSTEM:\n"
            stats_text += f"  Baseline Mean: {baseline_mean:.3f}\n"
            stats_text += f"  Band Width: ±{band_width:.3f}\n"
            stats_text += f"  Upper Threshold: {upper_threshold:.3f} (Movement = TRUE when above)\n"
            stats_text += f"  Lower Threshold: {lower_threshold:.3f} (Movement = FALSE when below)\n"
            stats_text += f"  Hysteresis Zone: {lower_threshold:.3f} to {upper_threshold:.3f} (State unchanged)\n"

            if stats.get("was_detrended", False):
                stats_text += "  Detrending: Applied\n"

            # Add method-specific information
            method = stats.get("method", "unknown")
            if "calibration" in method:
                snr = stats.get("signal_to_noise_ratio", 0)
                quality = stats.get("calibration_quality", "unknown")
                stats_text += f"  Calibration Quality: {quality}\n"
                stats_text += f"  Signal-to-Noise Ratio: {snr:.2f}\n"

            stats_text += "\n"

        # Display in log
        self._log_message("DETAILED HYSTERESIS STATISTICS:")
        for line in stats_text.split("\n"):
            if line.strip():
                self._log_message(line)

    # def save_results_with_metadata(self):
    #     """
    #     Save HDF5 metadata with optional Nematostella timeseries analysis.
    #     Enhanced to automatically detect and analyze Nematostella experiments.
    #     """

    #     # Check if we have a file loaded
    #     if not hasattr(self, 'file_path') or not self.file_path:
    #         self.results_label.setText("No HDF5 file loaded. Load a file first.")
    #         self._log_message("Save failed: No HDF5 file loaded")
    #         return

    #     # Analysis results are optional for metadata extraction
    #     has_analysis_results = (hasattr(self, "merged_results") and self.merged_results)

    #     if has_analysis_results:
    #         self._log_message("Saving analysis results with HDF5 metadata...")
    #     else:
    #         self._log_message("Saving HDF5 metadata only (no analysis results available)...")

    #     # NEW: Check for Nematostella timeseries data
    #     nematostella_results = None
    #     # Direkte Prüfung statt globaler Variable
    #     try:
    #         from ._metadata import analyze_nematostella_hdf5_file
    #         nematostella_available = True
    #     except ImportError:
    #         nematostella_available = False

    #     if nematostella_available:
    #         try:
    #             self._log_message("Checking for Nematostella timeseries data...")

    #             # Quick check if this is a Nematostella experiment
    #             with h5py.File(self.file_path, 'r') as h5_file:
    #                 if 'timeseries' in h5_file:
    #                     ts_group = h5_file['timeseries']
    #                     # Check for typical Nematostella parameters
    #                     nematostella_indicators = [
    #                         'actual_intervals', 'expected_intervals', 'frame_drift',
    #                         'temperature', 'humidity', 'led_power_percent'
    #                     ]

    #                     found_indicators = [key for key in ts_group.keys() if key in nematostella_indicators]

    #                     if len(found_indicators) >= 2:  # At least 2 indicators found
    #                         self._log_message(f"Nematostella experiment detected! Found: {', '.join(found_indicators)}")
    #                         self._log_message("Running specialized Nematostella timeseries analysis...")

    #                         # Run Nematostella analysis
    #                         nematostella_results = analyze_nematostella_hdf5_file(self.file_path)

    #                         if nematostella_results['success']:
    #                             self._log_message(f"Nematostella analysis completed: {len(nematostella_results['sheets_created'])} sheets")
    #                         else:
    #                             self._log_message(f"Nematostella analysis failed: {nematostella_results['error']}")
    #                     else:
    #                         self._log_message("No Nematostella-specific timeseries detected")
    #         except Exception as e:
    #             self._log_message(f"Nematostella detection failed: {e}")

    #     # Get base filename from user
    #     from qtpy.QtWidgets import QFileDialog

    #     if nematostella_results and nematostella_results['success']:
    #         dialog_title = "Save HDF5 Metadata with Nematostella Analysis"
    #         default_name = f"nematostella_metadata_{int(time.time())}"
    #     else:
    #         dialog_title = "Save HDF5 Metadata" + (" with Analysis Results" if has_analysis_results else "")
    #         default_name = f"hdf5_metadata_{int(time.time())}"

    #     base_path, _ = QFileDialog.getSaveFileName(
    #         self, dialog_title, default_name, "All Files (*)"
    #     )

    #     if not base_path:
    #         self._log_message("Save cancelled by user")
    #         return

    #     base_path = os.path.splitext(base_path)[0]
    #     saved_files = []

    #     try:
    #         # Extract regular metadata
    #         self._log_message("Extracting HDF5 metadata with time-series support...")
    #         metadata_dict = {}

    #         # Extract from main file with time-series capability
    #         if hasattr(self, 'file_path') and self.file_path:
    #             self._log_message(f"   Extracting from main file: {os.path.basename(self.file_path)}")
    #             try:
    #                 main_metadata = extract_hdf5_metadata_timeseries(self.file_path)
    #                 metadata_dict['main_file'] = main_metadata

    #                 if 'timeseries_data' in main_metadata and main_metadata['timeseries_data']:
    #                     ts_data = main_metadata['timeseries_data']
    #                     self._log_message(f"     Found {len(ts_data)} time-series parameters")

    #             except Exception as e:
    #                 self._log_message(f"     Main file metadata extraction failed: {e}")
    #                 metadata_dict['main_file'] = {'error': str(e), 'timeseries_data': {}}

    #         # Add analysis metadata (only if we have analysis results)
    #         if has_analysis_results:
    #             metadata_dict['analysis_info'] = {
    #                 'analysis_method': self._get_current_threshold_method_display(),
    #                 'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    #                 'frame_interval': self.frame_interval.value(),
    #                 'rois_analyzed': len(self.merged_results),
    #                 'software_version': 'HDF5 Activity Analysis Widget v1.0',
    #                 'parameters': self._get_analysis_parameters_for_metadata(),
    #                 'timeseries_data': {}
    #             }
    #         else:
    #             metadata_dict['file_info_only'] = {
    #                 'extraction_type': 'HDF5 metadata only',
    #                 'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    #                 'source_file': os.path.basename(self.file_path),
    #                 'software_version': 'HDF5 Activity Analysis Widget v1.0',
    #                 'timeseries_data': {}
    #             }

    #         # NEW: Add Nematostella analysis results if available
    #         if nematostella_results and nematostella_results['success']:
    #             metadata_dict['nematostella_analysis'] = {
    #                 'analysis_type': 'Nematostella Timeseries Analysis',
    #                 'excel_file': nematostella_results['excel_file'],
    #                 'report_file': nematostella_results['report_file'],
    #                 'sheets_created': nematostella_results['sheets_created'],
    #                 'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    #                 'timeseries_data': {}
    #             }

    #         self._log_message("Metadata extraction completed")

    #         # Save CSV with metadata
    #         csv_path = f"{base_path}_metadata.csv"
    #         self._log_message(f"Saving CSV with metadata: {os.path.basename(csv_path)}")

    #         try:
    #             self._save_results_csv_with_metadata(csv_path, metadata_dict, has_analysis_results)
    #             saved_files.append(("CSV with Metadata", csv_path))
    #             self._log_message("CSV with metadata saved successfully")
    #         except Exception as e:
    #             self._log_message(f"CSV save failed: {e}")

    #         # Save Excel with metadata (if pandas available)
    #         try:
    #             import pandas as pd
    #             excel_path = f"{base_path}_metadata.xlsx"
    #             self._log_message(f"Saving Excel with metadata: {os.path.basename(excel_path)}")

    #             self._save_results_excel_with_metadata(excel_path, metadata_dict, has_analysis_results)
    #             saved_files.append(("Excel with Metadata", excel_path))
    #             self._log_message("Excel with metadata saved successfully")

    #         except ImportError:
    #             self._log_message("Excel export not available (missing pandas/openpyxl)")
    #         except Exception as e:
    #             self._log_message(f"Excel save failed: {e}")

    #         # Update UI
    #         if saved_files:
    #             file_list = ", ".join([f"{fmt} ({os.path.basename(path)})" for fmt, path in saved_files])

    #             if nematostella_results and nematostella_results['success']:
    #                 result_msg = f"Saved metadata + Nematostella analysis: {file_list}"
    #                 result_msg += f" + {nematostella_results['excel_file']}"
    #             else:
    #                 result_msg = f"Saved metadata: {file_list}"

    #             self.results_label.setText(result_msg)
    #             self._log_message(f"Save with metadata complete: {len(saved_files)} files created")

    #             # Log Nematostella results if available
    #             if nematostella_results and nematostella_results['success']:
    #                 self._log_message("Nematostella Analysis Summary:")
    #                 report_lines = nematostella_results['report'].split('\n')
    #                 for line in report_lines:
    #                     if any(section in line for section in ['## Timing Analysis', '## Environmental Conditions', '## LED System']):
    #                         self._log_message(line)
    #                     elif line.strip().startswith('-') and any(keyword in line for keyword in ['Mean', 'Accuracy', 'Success Rate']):
    #                         self._log_message(f"  {line.strip()}")

    #             # Check if method supports nematostella_results parameter
    #             try:
    #                 self._show_save_success_dialog_with_metadata(saved_files, metadata_dict, nematostella_results)
    #             except TypeError:
    #                 # Fallback to old method signature
    #                 self._show_save_success_dialog_with_metadata(saved_files, metadata_dict)
    #         else:
    #             self.results_label.setText("All save attempts failed - check log")

    #     except Exception as e:
    #         error_msg = f"Save with metadata failed: {e}"
    #         self.results_label.setText(error_msg)
    #         self._log_message(error_msg)
    #         import traceback
    #         self._log_message(f"Traceback: {traceback.format_exc()}")
    def save_results_with_metadata(self):
        """
        Save HDF5 metadata with automatic legacy enhancement and optional Nematostella analysis.
        Enhanced to automatically detect legacy files and add unit documentation.
        """

        # Check if we have a file loaded
        if not hasattr(self, "file_path") or not self.file_path:
            self.results_label.setText("No HDF5 file loaded. Load a file first.")
            self._log_message("Save failed: No HDF5 file loaded")
            return

        # Analysis results are optional for metadata extraction
        has_analysis_results = hasattr(self, "merged_results") and self.merged_results

        if has_analysis_results:
            self._log_message("Saving analysis results with HDF5 metadata...")
        else:
            self._log_message(
                "Saving HDF5 metadata only (no analysis results available)..."
            )

        # NEW: Check for Nematostella timeseries data
        nematostella_results = None
        # Direkte Prüfung statt globaler Variable
        try:
            from ._metadata import analyze_nematostella_hdf5_file

            nematostella_available = True
        except ImportError:
            nematostella_available = False

        if nematostella_available:
            try:
                self._log_message("Checking for Nematostella timeseries data...")

                # Quick check if this is a Nematostella experiment
                with h5py.File(self.file_path, "r") as h5_file:
                    if "timeseries" in h5_file:
                        ts_group = h5_file["timeseries"]
                        # Check for typical Nematostella parameters
                        nematostella_indicators = [
                            "actual_intervals",
                            "expected_intervals",
                            "frame_drift",
                            "temperature",
                            "humidity",
                            "led_power_percent",
                        ]

                        found_indicators = [
                            key
                            for key in ts_group.keys()
                            if key in nematostella_indicators
                        ]

                        if len(found_indicators) >= 2:  # At least 2 indicators found
                            self._log_message(
                                f"Nematostella experiment detected! Found: {', '.join(found_indicators)}"
                            )
                            self._log_message(
                                "Running specialized Nematostella timeseries analysis..."
                            )

                            # Run Nematostella analysis
                            nematostella_results = analyze_nematostella_hdf5_file(
                                self.file_path
                            )

                            if nematostella_results["success"]:
                                self._log_message(
                                    f"Nematostella analysis completed: {len(nematostella_results['sheets_created'])} sheets"
                                )
                            else:
                                self._log_message(
                                    f"Nematostella analysis failed: {nematostella_results['error']}"
                                )
                        else:
                            self._log_message(
                                "No Nematostella-specific timeseries detected"
                            )
            except Exception as e:
                self._log_message(f"Nematostella detection failed: {e}")

        # Get base filename from user
        from qtpy.QtWidgets import QFileDialog

        if nematostella_results and nematostella_results["success"]:
            dialog_title = "Save HDF5 Metadata with Nematostella Analysis"
            default_name = f"nematostella_metadata_{int(time.time())}"
        else:
            dialog_title = "Save HDF5 Metadata" + (
                " with Analysis Results" if has_analysis_results else ""
            )
            default_name = f"hdf5_metadata_{int(time.time())}"

        base_path, _ = QFileDialog.getSaveFileName(
            self, dialog_title, default_name, "All Files (*)"
        )

        if not base_path:
            self._log_message("Save cancelled by user")
            return

        base_path = os.path.splitext(base_path)[0]
        saved_files = []

        try:
            # === AUTOMATIC LEGACY ENHANCEMENT INTEGRATION ===
            self._log_message(
                "Extracting HDF5 metadata with automatic legacy enhancement..."
            )
            metadata_dict = {}

            # Extract from main file with automatic legacy enhancement
            if hasattr(self, "file_path") and self.file_path:
                self._log_message(
                    f"   Extracting from main file: {os.path.basename(self.file_path)}"
                )
                try:
                    # This function now automatically enhances legacy files
                    main_metadata = extract_hdf5_metadata_timeseries(self.file_path)
                    metadata_dict["main_file"] = main_metadata

                    # Log automatic legacy enhancement results
                    if main_metadata.get("legacy_enhanced", False):
                        enhancement_info = main_metadata.get("_enhancement_summary", {})
                        enhanced_params = enhancement_info.get("parameters_enhanced", 0)
                        self._log_message("     ✅ Legacy file automatically enhanced!")
                        self._log_message(
                            f"     📏 Unit documentation added for {enhanced_params} parameters"
                        )
                        self._log_message(
                            f"     🕒 Enhancement timestamp: {main_metadata.get('enhancement_timestamp', 'unknown')}"
                        )
                        self._log_message(
                            "     📊 Unit standard: seconds for timing, celsius for temp, percent for humidity"
                        )
                    elif main_metadata.get("modern_file", False):
                        self._log_message(
                            "     ✅ Modern file with existing unit documentation detected"
                        )
                    else:
                        self._log_message("     ⚠️ File type could not be determined")

                    # Log timeseries data found
                    if (
                        "timeseries_data" in main_metadata
                        and main_metadata["timeseries_data"]
                    ):
                        ts_data = main_metadata["timeseries_data"]
                        # Count non-metadata parameters
                        param_count = len(
                            [k for k in ts_data.keys() if not k.startswith("_")]
                        )
                        self._log_message(
                            f"     📈 Found {param_count} time-series parameters"
                        )

                        # Log unit enhancement details if available
                        unit_info = ts_data.get("_unit_info", {})
                        if unit_info:
                            timing_params = [
                                k
                                for k, v in unit_info.items()
                                if v.get("units") == "seconds"
                            ]
                            environmental_params = [
                                k
                                for k, v in unit_info.items()
                                if v.get("units") in ["celsius", "percent"]
                            ]
                            if timing_params:
                                self._log_message(
                                    f"       ⏱️ Timing parameters: {', '.join(timing_params[:3])}{'...' if len(timing_params) > 3 else ''}"
                                )
                            if environmental_params:
                                self._log_message(
                                    f"       🌡️ Environmental parameters: {', '.join(environmental_params)}"
                                )

                except Exception as e:  # <-- JETZT KORREKT EINGERÜCKT
                    self._log_message(f"     Main file metadata extraction failed: {e}")
                    metadata_dict["main_file"] = {
                        "error": str(e),
                        "timeseries_data": {},
                    }

            # Add analysis metadata (only if we have analysis results)
            if has_analysis_results:
                metadata_dict["analysis_info"] = {
                    "analysis_method": self._get_current_threshold_method_display(),
                    "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_interval": self.frame_interval.value(),
                    "rois_analyzed": len(self.merged_results),
                    "software_version": "HDF5 Activity Analysis Widget v1.0 (Legacy Enhanced)",
                    "parameters": self._get_analysis_parameters_for_metadata(),
                    "timeseries_data": {},
                    "legacy_compatibility": True,  # Mark as legacy-compatible
                }
            else:
                metadata_dict["file_info_only"] = {
                    "extraction_type": "HDF5 metadata only (Legacy Enhanced)",
                    "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_file": os.path.basename(self.file_path),
                    "software_version": "HDF5 Activity Analysis Widget v1.0 (Legacy Enhanced)",
                    "timeseries_data": {},
                    "legacy_compatibility": True,
                }

            # NEW: Add Nematostella analysis results if available
            if nematostella_results and nematostella_results["success"]:
                metadata_dict["nematostella_analysis"] = {
                    "analysis_type": "Nematostella Timeseries Analysis (Legacy Enhanced)",
                    "excel_file": nematostella_results["excel_file"],
                    "report_file": nematostella_results["report_file"],
                    "sheets_created": nematostella_results["sheets_created"],
                    "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timeseries_data": {},
                    "legacy_enhanced": metadata_dict["main_file"].get(
                        "legacy_enhanced", False
                    ),
                }

            self._log_message("Metadata extraction with legacy enhancement completed")

            # Save CSV with enhanced metadata
            csv_path = f"{base_path}_metadata.csv"
            self._log_message(
                f"Saving enhanced CSV with metadata: {os.path.basename(csv_path)}"
            )

            try:
                self._save_results_csv_with_metadata(
                    csv_path, metadata_dict, has_analysis_results
                )
                saved_files.append(("Enhanced CSV with Metadata", csv_path))
                self._log_message("Enhanced CSV with metadata saved successfully")
            except Exception as e:
                self._log_message(f"CSV save failed: {e}")

            # Save Excel with enhanced metadata (if pandas available)
            try:
                import pandas as pd

                excel_path = f"{base_path}_metadata.xlsx"
                self._log_message(
                    f"Saving enhanced Excel with metadata: {os.path.basename(excel_path)}"
                )

                self._save_results_excel_with_metadata(
                    excel_path, metadata_dict, has_analysis_results
                )
                saved_files.append(("Enhanced Excel with Metadata", excel_path))
                self._log_message("Enhanced Excel with metadata saved successfully")

            except ImportError:
                self._log_message(
                    "Excel export not available (missing pandas/openpyxl)"
                )
            except Exception as e:
                self._log_message(f"Excel save failed: {e}")

            # Update UI with legacy enhancement information
            if saved_files:
                file_list = ", ".join(
                    [f"{fmt} ({os.path.basename(path)})" for fmt, path in saved_files]
                )

                # Enhanced result message
                is_legacy = metadata_dict.get("main_file", {}).get(
                    "legacy_enhanced", False
                )
                legacy_suffix = " (Legacy Enhanced)" if is_legacy else ""

                if nematostella_results and nematostella_results["success"]:
                    result_msg = f"Saved metadata + Nematostella analysis{legacy_suffix}: {file_list}"
                    result_msg += (
                        f" + {os.path.basename(nematostella_results['excel_file'])}"
                    )
                else:
                    result_msg = f"Saved metadata{legacy_suffix}: {file_list}"

                self.results_label.setText(result_msg)
                self._log_message(
                    f"Save with metadata complete: {len(saved_files)} files created"
                )

                # Log enhancement summary
                if is_legacy:
                    enhancement_summary = metadata_dict["main_file"].get(
                        "_enhancement_summary", {}
                    )
                    enhanced_count = enhancement_summary.get("parameters_enhanced", 0)
                    self._log_message("📋 Legacy Enhancement Summary:")
                    self._log_message(f"   Parameters enhanced: {enhanced_count}")
                    self._log_message(
                        f"   Unit standard applied: {enhancement_summary.get('unit_standard', 'Unknown')}"
                    )
                    self._log_message(
                        "   Files include comprehensive unit documentation"
                    )

                # Log Nematostella results if available
                if nematostella_results and nematostella_results["success"]:
                    self._log_message("Nematostella Analysis Summary:")
                    report_lines = nematostella_results["report"].split("\n")
                    for line in report_lines:
                        if any(
                            section in line
                            for section in [
                                "## Timing Analysis",
                                "## Environmental Conditions",
                                "## LED System",
                            ]
                        ):
                            self._log_message(line)
                        elif line.strip().startswith("-") and any(
                            keyword in line
                            for keyword in ["Mean", "Accuracy", "Success Rate"]
                        ):
                            self._log_message(f"  {line.strip()}")

                # Show enhanced success dialog
                try:
                    self._show_save_success_dialog_with_metadata(
                        saved_files, metadata_dict, nematostella_results
                    )
                except TypeError:
                    # Fallback to old method signature
                    self._show_save_success_dialog_with_metadata(
                        saved_files, metadata_dict
                    )
            else:
                self.results_label.setText("All save attempts failed - check log")

        except Exception as e:
            error_msg = f"Save with metadata failed: {e}"
            self.results_label.setText(error_msg)
            self._log_message(error_msg)
            import traceback

            self._log_message(f"Traceback: {traceback.format_exc()}")

    def _create_legacy_enhanced_sheets(
        self, writer, ts_data: dict, unit_info: dict, source_name: str
    ):
        """Create enhanced sheets for legacy data with automatic unit documentation."""

        # Erstelle DataFrame mit Unit-erweiterten Spalten-Namen
        enhanced_columns = {}

        for param_name, param_data in ts_data.items():
            if param_name.startswith("_"):
                continue  # Skip metadata

            unit = unit_info.get(param_name, {}).get("units", "unknown")

            if unit == "seconds" and "drift" in param_name.lower():
                # Für Timing-Daten: beide Einheiten
                enhanced_columns[f"{param_name}_sec"] = param_data
                enhanced_columns[f"{param_name}_ms"] = [
                    d * 1000 if d else 0 for d in param_data
                ]
            else:
                # Standard Parameter mit Unit-Suffix
                enhanced_columns[f"{param_name}_{unit}"] = param_data

        if enhanced_columns:
            # Frame index hinzufügen
            max_length = max(
                len(data)
                for data in enhanced_columns.values()
                if isinstance(data, (list, tuple))
            )
            enhanced_columns["frame_index"] = list(range(max_length))

            df = pd.DataFrame(enhanced_columns)
            sheet_name = f"Enhanced_{source_name}"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _create_automatic_units_reference_sheet(self, writer, metadata_dict: dict):
        """Automatically create units reference sheet for legacy files."""

        units_found = set()

        # Sammle alle gefundenen Parameter und ihre Units
        for metadata in metadata_dict.values():
            if "timeseries_data" in metadata:
                unit_info = metadata["timeseries_data"].get("_unit_info", {})
                for param, info in unit_info.items():
                    units_found.add(
                        (param, info["units"], info.get("display_hint", ""))
                    )

        if units_found:
            units_data = []
            for param, unit, hint in sorted(units_found):
                units_data.append(
                    {
                        "Parameter": param,
                        "Units": unit,
                        "Display_Hint": hint,
                        "Enhancement": "Automatically added for legacy compatibility",
                    }
                )

            units_df = pd.DataFrame(units_data)
            units_df.to_excel(writer, sheet_name="Auto_Units_Reference", index=False)

    def _save_results_csv_with_metadata(
        self, file_path: str, metadata_dict: dict, has_analysis_results: bool = True
    ):
        """
        Save CSV with HDF5 metadata time-series (with optional analysis results).
        """
        import csv
        from datetime import datetime

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # === HEADER SECTION ===
            if has_analysis_results:
                writer.writerow(["HDF5 Analysis Results with Time-Series Metadata"])
                sorted_rois = sorted(self.merged_results.keys())
                writer.writerow([f"Number of ROIs: {len(sorted_rois)}"])
            else:
                writer.writerow(["HDF5 Time-Series Metadata Only"])
                writer.writerow(
                    ["No analysis results available - metadata extraction only"]
                )

            writer.writerow(
                [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
            )
            writer.writerow([f"Source file: {os.path.basename(self.file_path)}"])
            writer.writerow([])

            # === ANALYSIS RESULTS SUMMARY (only if available) ===
            if has_analysis_results:
                writer.writerow(["=== ANALYSIS RESULTS SUMMARY ==="])
                writer.writerow(
                    [
                        "ROI",
                        "Baseline Mean",
                        "Upper Threshold",
                        "Lower Threshold",
                        "Movement %",
                        "Sleep Time (min)",
                    ]
                )

                # Get analysis data
                roi_baseline_means = getattr(self, "roi_baseline_means", {})
                roi_upper_thresholds = getattr(self, "roi_upper_thresholds", {})
                roi_lower_thresholds = getattr(self, "roi_lower_thresholds", {})
                movement_data = getattr(self, "movement_data", {})
                sleep_data = getattr(self, "sleep_data", {})

                for roi in sorted_rois:
                    # Calculate statistics
                    movement_pct = 0
                    if roi in movement_data and movement_data[roi]:
                        movement_values = [m for _, m in movement_data[roi]]
                        movement_pct = (
                            (sum(movement_values) / len(movement_values) * 100)
                            if movement_values
                            else 0
                        )

                    sleep_minutes = 0
                    if roi in sleep_data and sleep_data[roi]:
                        sleep_values = [s for _, s in sleep_data[roi]]
                        total_sleep_bins = sum(sleep_values)
                        sleep_minutes = (
                            total_sleep_bins * self.bin_size_seconds.value()
                        ) / 60

                    writer.writerow(
                        [
                            roi,
                            f"{roi_baseline_means.get(roi, 0):.3f}",
                            f"{roi_upper_thresholds.get(roi, 0):.3f}",
                            f"{roi_lower_thresholds.get(roi, 0):.3f}",
                            f"{movement_pct:.1f}",
                            f"{sleep_minutes:.1f}",
                        ]
                    )

                writer.writerow([])
                writer.writerow([])

            # === STATIC HDF5 METADATA SECTIONS ===
            for source_name, metadata in metadata_dict.items():
                if source_name in ["analysis_info", "file_info_only"]:
                    continue  # Handle separately

                writer.writerow(
                    [f"=== {source_name.upper().replace('_', ' ')} STATIC METADATA ==="]
                )

                # Write static metadata (excluding timeseries_data)
                static_metadata = {
                    k: v for k, v in metadata.items() if k != "timeseries_data"
                }
                if static_metadata:
                    from ._metadata import write_metadata_to_csv

                    write_metadata_to_csv(writer, static_metadata, source_name.upper())
                else:
                    writer.writerow(["No static metadata available"])

                writer.writerow([])

            # === HDF5 TIME-SERIES METADATA SECTIONS ===
            has_hdf5_timeseries = False
            for source_name, metadata in metadata_dict.items():
                if "timeseries_data" in metadata and metadata["timeseries_data"]:
                    has_hdf5_timeseries = True
                    writer.writerow(
                        [
                            f"=== {source_name.upper().replace('_', ' ')} HDF5 TIME-SERIES METADATA ==="
                        ]
                    )

                    ts_data = metadata["timeseries_data"]

                    # Filter out analysis-related data - only keep actual HDF5 metadata
                    from ._metadata import filter_hdf5_metadata_only

                    hdf5_metadata_only = filter_hdf5_metadata_only(ts_data)

                    if hdf5_metadata_only:
                        max_length = max(
                            len(data) for data in hdf5_metadata_only.values()
                        )

                        # Log what we're including
                        param_names = list(hdf5_metadata_only.keys())
                        self._log_message(
                            f"   Including HDF5 time-series: {param_names}"
                        )

                        # Align time with analysis data (or use generic timing)
                        frame_interval = (
                            self.frame_interval.value() if has_analysis_results else 5.0
                        )

                        # Header: Time (min), parameters...
                        header = ["Time (min)"] + param_names
                        writer.writerow(header)

                        # Data rows
                        for i in range(max_length):
                            time_min = (i * frame_interval) / 60.0
                            row = [f"{time_min:.2f}"]

                            for param_name in param_names:
                                param_data = hdf5_metadata_only[param_name]
                                if i < len(param_data):
                                    value = param_data[i]
                                    if isinstance(value, (int, float)):
                                        row.append(f"{value:.6f}")
                                    else:
                                        row.append(str(value))
                                else:
                                    row.append("")  # Missing data

                            writer.writerow(row)

                        writer.writerow([])
                        writer.writerow(
                            [
                                f"HDF5 time-series metadata: {len(hdf5_metadata_only)} parameters, {max_length} time points"
                            ]
                        )

                        # List all parameters
                        writer.writerow(["Parameters included:"])
                        for param in param_names:
                            writer.writerow([f"  - {param}"])

                    else:
                        writer.writerow(["No HDF5 time-series metadata found"])

                    writer.writerow([])
                    writer.writerow([])

            if not has_hdf5_timeseries:
                writer.writerow(["=== NO HDF5 TIME-SERIES METADATA FOUND ==="])
                writer.writerow(
                    ["Your HDF5 file may not contain time-series metadata."]
                )
                writer.writerow([])

            # === ANALYSIS/FILE INFO PARAMETERS ===
            info_section = metadata_dict.get("analysis_info") or metadata_dict.get(
                "file_info_only"
            )
            if info_section:
                section_name = (
                    "ANALYSIS PARAMETERS"
                    if has_analysis_results
                    else "FILE INFORMATION"
                )
                writer.writerow([f"=== {section_name} ==="])

                # Write parameters
                writer.writerow(["Parameter", "Value", "Description"])

                param_descriptions = {
                    "analysis_method": "Threshold calculation method used",
                    "analysis_timestamp": "When analysis was performed",
                    "extraction_timestamp": "When metadata was extracted",
                    "frame_interval": "Time interval between frames (seconds)",
                    "rois_analyzed": "Total number of ROIs analyzed",
                    "extraction_type": "Type of extraction performed",
                    "source_file": "Source HDF5 file name",
                    "software_version": "Analysis software version",
                }

                for param, value in info_section.items():
                    if param != "parameters":
                        description = param_descriptions.get(param, "Parameter")
                        writer.writerow([param, str(value), description])

                # Write nested parameters (if analysis results available)
                if "parameters" in info_section:
                    writer.writerow([])
                    writer.writerow(["Detailed Parameters:"])
                    writer.writerow(["Parameter", "Value"])

                    for param, value in info_section["parameters"].items():
                        writer.writerow([param, str(value)])

            # === FOOTER ===
            writer.writerow([])
            writer.writerow(["=== END OF FILE ==="])
            writer.writerow(
                [f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
            )

    def _show_save_success_dialog_with_metadata(self, saved_files, metadata_dict):
        """Show success dialog with metadata details."""
        from qtpy.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("Save with Metadata Complete")
        msg.setText(
            f"Successfully saved analysis results with metadata in {len(saved_files)} format(s):"
        )

        # Count metadata statistics
        total_static_params = 0
        total_timeseries_params = 0
        total_timeseries_points = 0

        for source_name, metadata in metadata_dict.items():
            # Count static parameters
            static_data = {k: v for k, v in metadata.items() if k != "timeseries_data"}
            total_static_params += len(static_data)

            # Count time-series parameters
            if "timeseries_data" in metadata and metadata["timeseries_data"]:
                ts_data = metadata["timeseries_data"]
                total_timeseries_params += len(ts_data)
                for param_data in ts_data.values():
                    if hasattr(param_data, "__len__"):
                        total_timeseries_points = max(
                            total_timeseries_points, len(param_data)
                        )

        file_details = []
        for file_format, file_path in saved_files:
            filename = os.path.basename(file_path)
            file_size = "Unknown size"
            try:
                size_bytes = os.path.getsize(file_path)
                if size_bytes > 1024 * 1024:  # > 1MB
                    file_size = f"{size_bytes/(1024*1024):.1f} MB"
                elif size_bytes > 1024:  # > 1KB
                    file_size = f"{size_bytes/1024:.1f} KB"
                else:
                    file_size = f"{size_bytes} bytes"
            except:
                pass

            file_details.append(f"• {file_format}: {filename} ({file_size})")

        # Add metadata summary
        file_details.append("")
        file_details.append("Metadata Summary:")
        file_details.append(f"• Static parameters: {total_static_params}")
        file_details.append(f"• Time-series parameters: {total_timeseries_params}")
        if total_timeseries_points > 0:
            file_details.append(
                f"• Time-series length: {total_timeseries_points} time points"
            )
            duration_min = (
                total_timeseries_points * self.frame_interval.value()
            ) / 60.0
            file_details.append(f"• Total duration: {duration_min:.1f} minutes")

        msg.setDetailedText("\n".join(file_details))
        msg.setInformativeText(
            "Files include comprehensive HDF5 metadata in time-series format matching analysis data structure."
        )
        msg.exec_()

    def _save_results_excel_with_metadata(
        self, excel_path: str, metadata_dict: dict, has_analysis_results: bool = True
    ):
        """
        Save Excel with unit-enhanced headers for legacy files and individual HDF5 sheets.
        """
        import pandas as pd

        # Check if this is a legacy enhanced file
        is_legacy_enhanced = metadata_dict.get("main_file", {}).get(
            "legacy_enhanced", False
        )

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

            # === SUMMARY SHEET (with unit-aware columns if legacy enhanced) ===
            if has_analysis_results:
                sorted_rois = sorted(self.merged_results.keys())

                summary_data = []
                for roi in sorted_rois:
                    if is_legacy_enhanced:
                        # Enhanced column names with units
                        row_data = {
                            "ROI": roi,
                            "Baseline_Mean_intensity": getattr(
                                self, "roi_baseline_means", {}
                            ).get(roi, 0),
                            "Upper_Threshold_intensity": getattr(
                                self, "roi_upper_thresholds", {}
                            ).get(roi, 0),
                            "Lower_Threshold_intensity": getattr(
                                self, "roi_lower_thresholds", {}
                            ).get(roi, 0),
                        }
                    else:
                        # Traditional column names
                        row_data = {
                            "ROI": roi,
                            "Baseline Mean": getattr(
                                self, "roi_baseline_means", {}
                            ).get(roi, 0),
                            "Upper Threshold": getattr(
                                self, "roi_upper_thresholds", {}
                            ).get(roi, 0),
                            "Lower Threshold": getattr(
                                self, "roi_lower_thresholds", {}
                            ).get(roi, 0),
                        }

                    # Add movement and sleep statistics with unit-aware names
                    movement_data = getattr(self, "movement_data", {})
                    if roi in movement_data and movement_data[roi]:
                        movement_values = [m for _, m in movement_data[roi]]
                        movement_pct = (
                            (sum(movement_values) / len(movement_values) * 100)
                            if movement_values
                            else 0
                        )

                        if is_legacy_enhanced:
                            row_data["Movement_Percentage_0to100"] = movement_pct
                        else:
                            row_data["Movement Percentage"] = movement_pct

                    sleep_data = getattr(self, "sleep_data", {})
                    if roi in sleep_data and sleep_data[roi]:
                        sleep_values = [s for _, s in sleep_data[roi]]
                        total_sleep_bins = sum(sleep_values)
                        sleep_minutes = (
                            total_sleep_bins * self.bin_size_seconds.value()
                        ) / 60

                        if is_legacy_enhanced:
                            row_data["Sleep_Time_minutes"] = sleep_minutes
                        else:
                            row_data["Sleep Time (min)"] = sleep_minutes

                    summary_data.append(row_data)

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                all_sheets_created = ["Summary"]
            else:
                # Create info sheet with legacy enhancement info
                info_data = [
                    {
                        "Property": "Extraction Type",
                        "Value": (
                            "HDF5 Metadata Only (Legacy Enhanced)"
                            if is_legacy_enhanced
                            else "HDF5 Metadata Only"
                        ),
                        "Description": "No analysis results available",
                    },
                    {
                        "Property": "Source File",
                        "Value": os.path.basename(self.file_path),
                        "Description": "HDF5 file analyzed",
                    },
                    {
                        "Property": "Extraction Date",
                        "Value": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Description": "When metadata was extracted",
                    },
                ]

                if is_legacy_enhanced:
                    info_data.append(
                        {
                            "Property": "Legacy Enhancement",
                            "Value": "Applied",
                            "Description": "Unit documentation added automatically",
                        }
                    )

                info_df = pd.DataFrame(info_data)
                info_df.to_excel(writer, sheet_name="File_Info", index=False)
                all_sheets_created = ["File_Info"]

            # === PROCESS HDF5 TIME-SERIES METADATA WITH UNIT ENHANCEMENT ===
            for source_name, metadata in metadata_dict.items():
                if source_name in [
                    "analysis_info",
                    "file_info_only",
                    "nematostella_analysis",
                ]:
                    continue

                # Process HDF5 time-series metadata
                if "timeseries_data" in metadata and metadata["timeseries_data"]:
                    ts_data = metadata["timeseries_data"]

                    # Get unit information if available
                    unit_info = ts_data.get("_unit_info", {})

                    # Filter to get only HDF5 metadata
                    hdf5_metadata_only = self._filter_hdf5_metadata_only(ts_data)

                    if hdf5_metadata_only:
                        self._log_message(
                            f"Creating unit-enhanced Excel sheets for {len(hdf5_metadata_only)} HDF5 parameters from {source_name}"
                        )

                        # Use frame interval from analysis if available, otherwise default
                        frame_interval = (
                            self.frame_interval.value() if has_analysis_results else 5.0
                        )

                        # CREATE UNIT-ENHANCED SHEETS
                        if is_legacy_enhanced and unit_info:
                            # Use enhanced unit-aware sheet creation
                            created_sheets = (
                                self._create_unit_enhanced_timeseries_sheets(
                                    writer,
                                    hdf5_metadata_only,
                                    unit_info,
                                    frame_interval,
                                    source_name,
                                )
                            )
                            all_sheets_created.extend(created_sheets)

                            self._log_message(
                                f"   ✅ Created {len(created_sheets)} unit-enhanced sheets"
                            )
                            for sheet_name in created_sheets:
                                self._log_message(f"   - {sheet_name}")

                        else:
                            # Fallback to regular sheet creation
                            try:
                                from ._metadata import (
                                    create_individual_timeseries_sheets,
                                    create_combined_timeseries_sheet,
                                )

                                individual_sheets = create_individual_timeseries_sheets(
                                    writer, hdf5_metadata_only, frame_interval
                                )
                                all_sheets_created.extend(individual_sheets)

                                for sheet_name in individual_sheets:
                                    self._log_message(
                                        f"   ✓ Created sheet '{sheet_name}'"
                                    )

                            except ImportError:
                                created_sheets = (
                                    self._create_timeseries_sheets_manually(
                                        writer, hdf5_metadata_only, source_name
                                    )
                                )
                                all_sheets_created.extend(created_sheets)
                            except Exception as e:
                                self._log_message(
                                    f"   Error with metadata functions: {e}"
                                )
                                created_sheets = (
                                    self._create_timeseries_sheets_manually(
                                        writer, hdf5_metadata_only, source_name
                                    )
                                )
                                all_sheets_created.extend(created_sheets)

                # Static HDF5 metadata sheet
                static_metadata = {
                    k: v for k, v in metadata.items() if k != "timeseries_data"
                }
                if static_metadata:
                    try:
                        try:
                            from ._metadata import create_metadata_dataframe

                            meta_df = create_metadata_dataframe(
                                static_metadata, source_name
                            )
                        except ImportError:
                            meta_df = self._create_metadata_dataframe_manually(
                                static_metadata, source_name
                            )

                        sheet_name = f"Static_{source_name}"[:31]
                        meta_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        all_sheets_created.append(sheet_name)
                        self._log_message(
                            f"   ✓ Created static metadata sheet '{sheet_name}'"
                        )
                    except Exception as e:
                        self._log_message(
                            f"   Warning: Could not create static sheet: {e}"
                        )

            # === PARAMETERS SHEET (only if analysis available) ===
            if has_analysis_results and "analysis_info" in metadata_dict:
                try:
                    params_data = []
                    analysis_info = metadata_dict["analysis_info"]

                    for key, value in analysis_info.items():
                        if key != "parameters":
                            params_data.append(
                                {
                                    "Parameter": key,
                                    "Value": str(value),
                                    "Category": "Analysis Info",
                                }
                            )

                    if "parameters" in analysis_info:
                        for key, value in analysis_info["parameters"].items():
                            params_data.append(
                                {
                                    "Parameter": key,
                                    "Value": str(value),
                                    "Category": "Analysis Parameters",
                                }
                            )

                    if params_data:
                        params_df = pd.DataFrame(params_data)
                        params_df.to_excel(
                            writer, sheet_name="Analysis_Parameters", index=False
                        )
                        all_sheets_created.append("Analysis_Parameters")
                except Exception as e:
                    self._log_message(
                        f"   Warning: Could not create parameters sheet: {e}"
                    )

            # Log final summary with enhancement info
            enhancement_info = " (with unit enhancement)" if is_legacy_enhanced else ""
            self._log_message(
                f"Excel file created{enhancement_info} with {len(all_sheets_created)} sheets:"
            )
            for sheet in all_sheets_created:
                self._log_message(f"   - {sheet}")

    def _create_unit_enhanced_timeseries_sheets(
        self,
        writer,
        hdf5_metadata: dict,
        unit_info: dict,
        frame_interval: float,
        source_name: str,
    ):
        """Create individual timeseries sheets with unit-enhanced column headers."""
        sheets_created = []

        # Create individual sheet for each parameter with unit-enhanced names
        for param_name, param_data in hdf5_metadata.items():
            if (
                not isinstance(param_data, (list, tuple, np.ndarray))
                or len(param_data) == 0
            ):
                continue

            try:
                max_length = len(param_data)
                time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

                # Get unit information
                unit = unit_info.get(param_name, {}).get("units", "unknown")

                # Create unit-enhanced column names
                if unit == "seconds" and "drift" in param_name.lower():
                    # For timing parameters: create both seconds and milliseconds columns
                    df_data = {
                        "Time_minutes": time_minutes,
                        f"{param_name}_seconds": param_data,
                        f"{param_name}_milliseconds": [
                            d * 1000 if d else 0 for d in param_data
                        ],
                    }
                elif unit == "celsius":
                    df_data = {
                        "Time_minutes": time_minutes,
                        f"{param_name}_celsius": param_data,
                    }
                elif unit == "percent":
                    df_data = {
                        "Time_minutes": time_minutes,
                        f"{param_name}_percent": param_data,
                    }
                elif unit == "milliseconds":
                    df_data = {
                        "Time_minutes": time_minutes,
                        f"{param_name}_milliseconds": param_data,
                    }
                else:
                    df_data = {
                        "Time_minutes": time_minutes,
                        f"{param_name}_{unit}": param_data,
                    }

                # Create DataFrame and sheet
                param_df = pd.DataFrame(df_data)
                clean_name = self._clean_sheet_name(f"{param_name}_Enhanced")

                # Ensure unique sheet name
                original_clean_name = clean_name
                counter = 1
                while clean_name in sheets_created:
                    clean_name = f"{original_clean_name[:28]}_{counter}"
                    counter += 1

                param_df.to_excel(writer, sheet_name=clean_name, index=False)
                sheets_created.append(clean_name)

            except Exception as e:
                self._log_message(
                    f"   Warning: Could not create enhanced sheet for {param_name}: {e}"
                )
                continue

        # Create combined sheet with all enhanced parameters
        if len(hdf5_metadata) > 1:
            try:
                max_length = max(len(data) for data in hdf5_metadata.values())
                time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

                combined_data = {"Time_minutes": time_minutes}

                for param_name, param_data in hdf5_metadata.items():
                    unit = unit_info.get(param_name, {}).get("units", "unknown")

                    # Pad data if necessary
                    if len(param_data) < max_length:
                        padded_data = list(param_data) + [np.nan] * (
                            max_length - len(param_data)
                        )
                    else:
                        padded_data = param_data

                    # Add with unit-enhanced name
                    if unit == "seconds" and "drift" in param_name.lower():
                        combined_data[f"{param_name}_seconds"] = padded_data
                        combined_data[f"{param_name}_ms"] = [
                            d * 1000 if d and not np.isnan(d) else np.nan
                            for d in padded_data
                        ]
                    else:
                        combined_data[f"{param_name}_{unit}"] = padded_data

                combined_df = pd.DataFrame(combined_data)
                combined_name = f"Enhanced_All_{source_name}"[:31]
                combined_df.to_excel(writer, sheet_name=combined_name, index=False)
                sheets_created.append(combined_name)

            except Exception as e:
                self._log_message(
                    f"   Warning: Could not create enhanced combined sheet: {e}"
                )

        return sheets_created

    def _filter_hdf5_metadata_only(self, ts_data: dict) -> dict:
        """Filter to keep only actual HDF5 metadata, excluding analysis results."""
        hdf5_metadata_only = {}

        # Only exclude specific analysis result patterns
        analysis_result_patterns = [
            "roi_",
            "baseline_",
            "threshold_",
            "upper_threshold",
            "lower_threshold",
            "movement_data",
            "fraction_data",
            "sleep_data",
            "quiescence_data",
            "intensity_roi_",
            "analysis_",
            "processed_",
            "calculated_",
        ]

        for param_name, param_data in ts_data.items():
            param_lower = param_name.lower()

            # Keep the parameter unless it matches specific analysis result patterns
            is_analysis_result = any(
                pattern in param_lower for pattern in analysis_result_patterns
            )

            if not is_analysis_result:
                hdf5_metadata_only[param_name] = param_data

        return hdf5_metadata_only

    def _create_timeseries_sheets_manually(
        self, writer, hdf5_metadata: dict, source_name: str
    ):
        """Manual fallback for creating time-series sheets."""
        import pandas as pd
        import numpy as np

        sheets_created = []

        for param_name, param_data in hdf5_metadata.items():
            try:
                if not hasattr(param_data, "__len__") or len(param_data) == 0:
                    continue

                # Create DataFrame with this parameter
                max_length = len(param_data)
                frame_interval = self.frame_interval.value()

                # Create time column aligned with analysis
                time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

                # Create DataFrame
                df_data = {"Time (min)": time_minutes, param_name: param_data}

                param_df = pd.DataFrame(df_data)

                # Clean parameter name for Excel sheet (max 31 chars)
                clean_name = self._clean_sheet_name(param_name)

                # Ensure unique sheet name
                original_clean_name = clean_name
                counter = 1
                while clean_name in sheets_created:
                    clean_name = f"{original_clean_name[:28]}_{counter}"
                    counter += 1

                # Create the sheet
                param_df.to_excel(writer, sheet_name=clean_name, index=False)
                sheets_created.append(clean_name)
                self._log_message(
                    f"   ✓ Created manual sheet '{clean_name}' for {param_name}"
                )

            except Exception as e:
                self._log_message(
                    f"   Warning: Could not create sheet for {param_name}: {e}"
                )
                continue

        # Also create a combined sheet if we have multiple parameters
        if len(hdf5_metadata) > 1:
            try:
                # Find the maximum length across all parameters
                max_length = max(len(data) for data in hdf5_metadata.values())
                frame_interval = self.frame_interval.value()
                time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

                # Create combined DataFrame
                df_data = {"Time (min)": time_minutes}

                for param_name, param_data in hdf5_metadata.items():
                    # Pad shorter series with NaN
                    if len(param_data) < max_length:
                        padded_data = list(param_data) + [np.nan] * (
                            max_length - len(param_data)
                        )
                    else:
                        padded_data = param_data
                    df_data[param_name] = padded_data

                combined_df = pd.DataFrame(df_data)
                combined_name = f"All_HDF5_{source_name}"[:31]
                combined_df.to_excel(writer, sheet_name=combined_name, index=False)
                sheets_created.append(combined_name)
                self._log_message(
                    f"   ✓ Created combined manual sheet '{combined_name}'"
                )

            except Exception as e:
                self._log_message(f"   Warning: Could not create combined sheet: {e}")

        return sheets_created

    def _clean_sheet_name(self, param_name: str) -> str:
        """Clean parameter name to be valid Excel sheet name."""
        # Remove or replace invalid characters
        clean = param_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        clean = (
            clean.replace("*", "star")
            .replace("?", "q")
            .replace("[", "")
            .replace("]", "")
        )
        clean = clean.replace("<", "lt").replace(">", "gt").replace("|", "_")

        # Truncate to 31 characters max
        if len(clean) > 31:
            # Try to keep meaningful parts
            if "_" in clean:
                parts = clean.split("_")
                if len(parts[0]) <= 25:
                    clean = parts[0] + "_" + "".join(p[0] for p in parts[1:] if p)[:5]
                else:
                    clean = clean[:31]
            else:
                clean = clean[:31]

        # Remove trailing underscores and ensure not empty
        clean = clean.rstrip("_")
        if not clean:
            clean = "unnamed_param"

        return clean

    def _create_metadata_dataframe_manually(self, metadata: dict, source_name: str):
        """Manual fallback for creating metadata DataFrame."""
        import pandas as pd

        rows = []

        def flatten_metadata(d, prefix=""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    flatten_metadata(value, full_key)
                else:
                    try:
                        str_value = str(value)
                        data_type = type(value).__name__
                    except:
                        str_value = f"<{type(value).__name__}>"
                        data_type = "Complex"

                    rows.append(
                        {
                            "Category": prefix if prefix else "Root",
                            "Parameter": key,
                            "Value": str_value,
                            "Data_Type": data_type,
                            "Source": source_name,
                        }
                    )

        flatten_metadata(metadata)
        return pd.DataFrame(rows)

    def _create_individual_sheets_fallback(
        self, writer, hdf5_metadata: dict, sheets_created: list
    ):
        """Fallback method using new module functions if available."""
        try:
            from ._metadata import create_individual_timeseries_sheets

            new_sheets = create_individual_timeseries_sheets(
                writer, hdf5_metadata, self.frame_interval.value()
            )
            sheets_created.extend(new_sheets)
            self._log_message(f"   ✓ Created {len(new_sheets)} fallback sheets")
        except Exception as e:
            self._log_message(f"   ✗ Fallback method failed: {e}")

    def _create_metadata_sheet_fallback(
        self, writer, hdf5_metadata: dict, source_name: str
    ):
        """
        Fallback method to create HDF5 metadata sheet when import fails.
        """
        import pandas as pd
        import numpy as np

        try:
            if not hdf5_metadata:
                return

            # Find the maximum length
            max_length = max(
                len(data) if hasattr(data, "__len__") else 1
                for data in hdf5_metadata.values()
            )

            # Create time column
            frame_interval = self.frame_interval.value()
            time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

            # Build DataFrame
            df_data = {"Time (min)": time_minutes}

            for param_name, param_data in hdf5_metadata.items():
                if hasattr(param_data, "__len__") and len(param_data) > 0:
                    # Pad shorter series with NaN
                    padded_data = list(param_data) + [np.nan] * (
                        max_length - len(param_data)
                    )
                    df_data[param_name] = padded_data
                else:
                    # Single value or empty data
                    df_data[param_name] = (
                        [param_data] * max_length if max_length > 0 else []
                    )

            df = pd.DataFrame(df_data)
            sheet_name = f"HDF5_{source_name}"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            self._log_message(f"   Created fallback metadata sheet '{sheet_name}'")

        except Exception as e:
            self._log_message(f"   Error in fallback metadata sheet creation: {e}")

    def _create_static_metadata_sheet_fallback(
        self, writer, static_metadata: dict, source_name: str
    ):
        """
        Fallback method to create static metadata sheet when import fails.
        """
        import pandas as pd

        try:
            rows = []

            def flatten_metadata(d, prefix=""):
                for key, value in d.items():
                    full_key = f"{prefix}.{key}" if prefix else key

                    if isinstance(value, dict):
                        flatten_metadata(value, full_key)
                    else:
                        rows.append(
                            {
                                "Category": prefix if prefix else "Root",
                                "Parameter": key,
                                "Value": str(value),
                                "Data_Type": type(value).__name__,
                                "Source": source_name,
                            }
                        )

            flatten_metadata(static_metadata)

            df = pd.DataFrame(rows)
            sheet_name = f"Static_{source_name}"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            self._log_message(f"   Created fallback static sheet '{sheet_name}'")

        except Exception as e:
            self._log_message(f"   Error in fallback static sheet creation: {e}")

    # def _create_metadata_dataframe(self, metadata: dict, source_name: str):
    #     """Helper to create DataFrame from metadata."""

    #     rows = []

    #     def flatten_metadata(d, prefix=""):
    #         for key, value in d.items():
    #             full_key = f"{prefix}.{key}" if prefix else key

    #             if isinstance(value, dict):
    #                 flatten_metadata(value, full_key)
    #             else:
    #                 try:
    #                     str_value = str(value)
    #                     data_type = type(value).__name__
    #                 except:
    #                     str_value = f"<{type(value).__name__}>"
    #                     data_type = 'Complex'

    #                 rows.append({
    #                     'Category': prefix if prefix else 'Root',
    #                     'Parameter': key,
    #                     'Value': str_value,
    #                     'Data_Type': data_type,
    #                     'Source': source_name
    #                 })

    #     flatten_metadata(metadata)
    #     return pd.DataFrame(rows)

    def _get_analysis_parameters_for_metadata(self) -> dict:
        """Extract analysis parameters for metadata."""
        params = {
            "method": self._get_current_threshold_method_display(),
            "frame_interval_seconds": self.frame_interval.value(),
            "bin_size_seconds": self.bin_size_seconds.value(),
            "quiescence_threshold": self.quiescence_threshold.value(),
            "sleep_threshold_minutes": self.sleep_threshold_minutes.value(),
        }

        # Method-specific parameters
        if hasattr(self, "baseline_duration_minutes"):
            params["baseline_duration_minutes"] = self.baseline_duration_minutes.value()
        if hasattr(self, "threshold_multiplier"):
            params["threshold_multiplier"] = self.threshold_multiplier.value()
        if hasattr(self, "calibration_multiplier"):
            params["calibration_multiplier"] = self.calibration_multiplier.value()
        if hasattr(self, "enable_detrending"):
            params["detrending_enabled"] = self.enable_detrending.isChecked()

        return params

    # ===================================================================
    # UI EVENT HANDLERS
    # ===================================================================

    def _on_progress_update(self, percent: int):
        """Update progress bar."""
        self.progress_bar.setValue(percent)

    def _on_status_update(self, message: str):
        """Update status label."""
        self.status_label.setText(message)
        self._log_message(message)

    def _on_performance_update(self, metrics: str):
        """Update performance metrics."""
        self.performance_label.setText(metrics)

    def _update_performance_metrics(self):
        """Update real-time performance metrics during analysis."""
        if self.analysis_start_time and not self._cancel_requested:
            elapsed = time.time() - self.analysis_start_time
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()

            metrics = (
                f"Elapsed: {elapsed:.1f}s | "
                f"CPU: {cpu_percent:.1f}% | "
                f"Memory: {memory_info.percent:.1f}% | "
                f"Processes: {self.num_processes.value()}"
            )

            self.performance_updated.emit(metrics)

    def _on_auto_scale_toggled(self, checked: bool):
        """Enable/disable manual Y-axis controls and advanced options based on auto scale setting."""
        # Manual controls
        self.y_min_spin.setEnabled(not checked)
        self.y_max_spin.setEnabled(not checked)
        self.btn_apply_y_range.setEnabled(not checked)

        # Advanced auto-scaling options
        self.robust_scaling.setEnabled(checked)
        self.adaptive_scaling.setEnabled(checked)
        self.center_around_zero.setEnabled(checked)
        self.lower_percentile_spin.setEnabled(
            checked and self.robust_scaling.isChecked()
        )
        self.upper_percentile_spin.setEnabled(
            checked and self.robust_scaling.isChecked()
        )

        # Regenerate plot with new scaling
        if hasattr(self, "merged_results") and self.merged_results:
            self.generate_plot()

    def _on_threshold_tab_changed(self, tab_index: int):
        """Handle threshold method tab changes."""
        # Tab index directly determines the method
        # 0 = Baseline, 1 = Calibration, 2 = Adaptive
        method_names = ["Baseline", "Calibration", "Adaptive"]
        if 0 <= tab_index < len(method_names):
            self._log_message(f"Threshold method changed to: {method_names[tab_index]}")

    def _get_current_threshold_method(self) -> str:
        """Get current threshold method based on active tab."""
        tab_index = self.threshold_params_stack.currentIndex()
        method_map = {0: "baseline", 1: "calibration", 2: "adaptive"}
        return method_map.get(tab_index, "baseline")

    def _get_current_threshold_method_display(self) -> str:
        """Get current threshold method display name based on active tab."""
        tab_index = self.threshold_params_stack.currentIndex()
        method_map = {
            0: "Baseline (First Frames)",
            1: "Calibration (Sedated Animals)",
            2: "Adaptive (Smart Detection)",
        }
        return method_map.get(tab_index, "Baseline (First Frames)")

    def load_calibration_file(self):
        """Enhanced calibration file loading with robust UI updates."""

        self._log_message("=== LOAD_CALIBRATION_FILE METHOD CALLED ===")

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Calibration File",
                "",
                "Video Files (*.h5 *.hdf5 *.avi);;HDF5 Files (*.h5 *.hdf5);;AVI Files (*.avi);;All Files (*.*)",
            )

            self._log_message(f"File dialog returned: '{file_path}'")

            if file_path and os.path.exists(file_path):
                basename = os.path.basename(file_path)

                # Store the path first
                self.calibration_file_path_stored = file_path
                self._log_message(f"Stored calibration file path: {file_path}")

                # Force UI update with multiple methods
                self.calibration_file_path.setText(basename)
                self.calibration_file_path.setProperty("full_path", file_path)

                # Force the widget to process events and update
                from qtpy.QtCore import QCoreApplication

                QCoreApplication.processEvents()

                # Verify the text was set
                current_text = self.calibration_file_path.text()
                self._log_message(f"UI text field now shows: '{current_text}'")

                if current_text != basename:
                    self._log_message(
                        f"WARNING: UI text mismatch! Expected '{basename}', got '{current_text}'"
                    )
                    # Try setting it again
                    self.calibration_file_path.setText(basename)
                    QCoreApplication.processEvents()
                    self._log_message(
                        f"After retry, UI shows: '{self.calibration_file_path.text()}'"
                    )

                # Enable the load dataset button
                if hasattr(self, "btn_load_calibration_dataset"):
                    self.btn_load_calibration_dataset.setEnabled(True)
                    self._log_message("Enabled 'Load Calibration Dataset' button")

                # Update status
                if hasattr(self, "calibration_status_label"):
                    self.calibration_status_label.setText(
                        "✅ 1. Calibration file selected\n"
                        "2. Click 'Load Calibration Dataset'\n"
                        "3. Detect ROIs (Input tab)\n"
                        "4. Process baseline"
                    )

                # Reset calibration processing state
                self.calibration_baseline_processed = False
                self.calibration_baseline_statistics = {}
                if hasattr(self, "btn_process_calibration_baseline"):
                    self.btn_process_calibration_baseline.setEnabled(False)

                self._log_message(f"Calibration file selection complete: {basename}")

            else:
                self._log_message("No valid file selected or file doesn't exist")

                # Reset UI
                self.calibration_file_path.setText("No calibration file selected")
                self.calibration_file_path_stored = None

                if hasattr(self, "btn_load_calibration_dataset"):
                    self.btn_load_calibration_dataset.setEnabled(False)

        except Exception as e:
            self._log_message(f"ERROR in load_calibration_file: {e}")
            import traceback

            self._log_message(f"Traceback: {traceback.format_exc()}")

    def _on_12well_toggled(self, checked: bool):
        """Enable/disable and populate ROI controls when preset is toggled."""
        # 12-well plate preset values
        preset = {
            "min_radius": 40,
            "max_radius": 75,
            "dp": 1.0,
            "min_dist": 75,
            "param1": 50,
            "param2": 30,
        }

        if checked:
            # Push the preset into the spin-boxes
            self.min_radius.setValue(preset["min_radius"])
            self.max_radius.setValue(preset["max_radius"])
            self.dp_param.setValue(preset["dp"])
            self.min_dist.setValue(preset["min_dist"])
            self.param1.setValue(preset["param1"])
            self.param2.setValue(preset["param2"])

        # Disable or re-enable editing
        for widget in (
            self.min_radius,
            self.max_radius,
            self.dp_param,
            self.min_dist,
            self.param1,
            self.param2,
        ):
            widget.setEnabled(not checked)

    def on_tab_changed(self, index: int):
        """Handle tab changes."""
        # Add specific logic for when tabs are changed if needed
        pass

    # ===================================================================
    # ROI MANAGEMENT METHODS
    # ===================================================================

    def _toggle_all_roi_visibility(self):
        """Toggle visibility of all ROI mask layers."""
        roi_layers = [
            layer
            for layer in self.viewer.layers
            if hasattr(layer, "metadata")
            and layer.metadata.get("roi_type") == "circular_detection"
        ]

        if not roi_layers:
            self._log_message("No ROI layers found")
            return

        # Check current state and toggle
        any_visible = any(layer.visible for layer in roi_layers)
        new_visibility = not any_visible

        for layer in roi_layers:
            layer.visible = new_visibility

        status = "visible" if new_visibility else "hidden"
        self._log_message(f"All ROI layers are now {status}")

    def _show_only_selected_roi(self):
        """Show only the currently selected ROI layer."""
        selected_layers = list(self.viewer.layers.selection)
        roi_layers = [
            layer
            for layer in self.viewer.layers
            if hasattr(layer, "metadata")
            and layer.metadata.get("roi_type") == "circular_detection"
        ]

        if not roi_layers:
            self._log_message("No ROI layers found")
            return

        selected_roi_layers = [
            layer for layer in selected_layers if layer in roi_layers
        ]

        if not selected_roi_layers:
            self._log_message("No ROI layer selected")
            return

        # Hide all ROI layers first
        for layer in roi_layers:
            layer.visible = False

        # Show only selected ROI layers
        for layer in selected_roi_layers:
            layer.visible = True
            roi_id = layer.metadata.get("roi_id", "unknown")
            self._log_message(f"Showing only ROI {roi_id}")

    def _reset_roi_visibility(self):
        """Reset ROI layer visibility to default state."""
        roi_layers = [
            layer
            for layer in self.viewer.layers
            if hasattr(layer, "metadata")
            and layer.metadata.get("roi_type") == "circular_detection"
        ]

        for layer in roi_layers:
            layer.visible = False  # Default state
            layer.opacity = 0.6  # Reset opacity
            layer.blending = "additive"  # Reset blending

        self._log_message(f"Reset visibility for {len(roi_layers)} ROI layers")

    # ===================================================================
    # EXTENDED ANALYSIS METHODS (FISCHER Z-TRANSFORMATION)
    # ===================================================================

    def run_fisher_analysis(self):
        """Run Fischer Z-transformation circadian analysis on movement data."""
        # Check if we have analysis results
        if not hasattr(self, "fraction_data") or not self.fraction_data:
            self.fisher_results_text.setPlainText(
                "ERROR: No analysis results available.\n\n"
                "Please run the main analysis first (Analysis tab) before "
                "attempting circadian pattern detection."
            )
            self._log_message(
                "⚠️ Fisher analysis requires movement data from main analysis"
            )
            return

        self._log_message("Starting Fischer Z-transformation circadian analysis...")
        self.fisher_results_text.setPlainText("Running analysis...\n")

        try:
            from ._fisher_analysis import (
                analyze_roi_circadian_patterns,
                generate_circadian_summary,
            )

            # Get parameters
            min_period = self.fisher_min_period.value()
            max_period = self.fisher_max_period.value()
            significance = self.fisher_significance.value()
            phase_threshold = self.fisher_phase_threshold.value()
            sampling_interval = self.frame_interval.value()

            # Run analysis
            self._log_message(
                f"  Period range: {min_period:.1f} - {max_period:.1f} hours"
            )
            self._log_message(f"  Significance level: {significance:.3f}")
            self._log_message(f"  Phase threshold: {phase_threshold:.2f}")

            fisher_results = analyze_roi_circadian_patterns(
                self.fraction_data,
                sampling_interval=sampling_interval,
                min_period_hours=min_period,
                max_period_hours=max_period,
                significance_level=significance,
                phase_threshold=phase_threshold,
            )

            # Store results
            self.fisher_analysis_results = fisher_results

            # Generate summary
            summary = generate_circadian_summary(fisher_results)

            # Display results
            self.fisher_results_text.setPlainText(summary)

            # Create and display plot
            self._create_fisher_plot(fisher_results)

            # Enable export button
            self.btn_export_fisher.setEnabled(True)

            # Count significant results
            n_significant = sum(
                1
                for r in fisher_results.values()
                if r.get("periodogram", {}).get("is_significant", False)
            )

            self._log_message(
                f"✓ Fisher analysis complete: {n_significant}/{len(fisher_results)} ROIs show significant rhythms"
            )

        except Exception as e:
            error_msg = f"ERROR during Fischer analysis:\n\n{str(e)}\n\nPlease check the console for details."
            self.fisher_results_text.setPlainText(error_msg)
            self._log_message(f"❌ Fisher analysis failed: {e}")
            import traceback

            traceback.print_exc()

    def export_fisher_results(self):
        """Export Fischer Z-transformation analysis results to Excel/CSV."""
        if not hasattr(self, "fisher_analysis_results"):
            self._log_message("⚠️ No Fisher analysis results to export")
            return

        from qtpy.QtWidgets import QFileDialog
        import os

        # Get save location
        default_name = "circadian_analysis_results"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Circadian Analysis Results",
            default_name,
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*.*)",
        )

        if not file_path:
            self._log_message("Export cancelled by user")
            return

        # Remove extension for consistent naming
        base_path = os.path.splitext(file_path)[0]

        try:
            # Export to CSV
            csv_path = f"{base_path}.csv"
            self._export_fisher_to_csv(csv_path)
            self._log_message(f"✓ Exported circadian results to CSV: {csv_path}")

            # Try to export to Excel if pandas is available
            try:
                import pandas as pd

                excel_path = f"{base_path}.xlsx"
                self._export_fisher_to_excel(excel_path)
                self._log_message(
                    f"✓ Exported circadian results to Excel: {excel_path}"
                )
            except ImportError:
                self._log_message("⚠️ Excel export not available (pandas not installed)")

        except Exception as e:
            self._log_message(f"❌ Export failed: {e}")
            import traceback

            traceback.print_exc()

    def _export_fisher_to_csv(self, file_path: str):
        """Export Fisher results to CSV format."""
        import csv

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(
                ["Circadian Rhythm Analysis Results (Fischer Z-transformation)"]
            )
            writer.writerow(
                [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
            )
            writer.writerow([])

            # Summary table
            writer.writerow(["ROI Summary"])
            writer.writerow(
                [
                    "ROI",
                    "Significant Rhythm",
                    "Dominant Period (hours)",
                    "Z-Score",
                    "P-Value",
                    "Wake Phases",
                    "Sleep Phases",
                    "Wake Fraction (%)",
                ]
            )

            for roi_id, result in sorted(self.fisher_analysis_results.items()):
                if "error" in result:
                    writer.writerow(
                        [roi_id, "Error", result["error"], "", "", "", "", ""]
                    )
                    continue

                periodogram = result.get("periodogram", {})
                phase_analysis = result.get("phase_analysis", {})

                is_sig = periodogram.get("is_significant", False)
                period = periodogram.get("dominant_period", 0)
                z_score = periodogram.get("dominant_z_score", 0)
                p_value = periodogram.get("p_value", 1.0)

                n_wake = len(phase_analysis.get("wake_phases", []))
                n_sleep = len(phase_analysis.get("sleep_phases", []))
                wake_frac = phase_analysis.get("wake_fraction", 0) * 100

                writer.writerow(
                    [
                        roi_id,
                        "Yes" if is_sig else "No",
                        f"{period:.2f}",
                        f"{z_score:.2f}",
                        f"{p_value:.4f}",
                        n_wake,
                        n_sleep,
                        f"{wake_frac:.1f}",
                    ]
                )

    def _export_fisher_to_excel(self, file_path: str):
        """Export Fisher results to Excel format with multiple sheets."""
        import pandas as pd

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary_data = []
            for roi_id, result in sorted(self.fisher_analysis_results.items()):
                if "error" in result:
                    continue

                periodogram = result.get("periodogram", {})
                phase_analysis = result.get("phase_analysis", {})

                summary_data.append(
                    {
                        "ROI": roi_id,
                        "Significant Rhythm": periodogram.get("is_significant", False),
                        "Dominant Period (hours)": periodogram.get(
                            "dominant_period", 0
                        ),
                        "Z-Score": periodogram.get("dominant_z_score", 0),
                        "P-Value": periodogram.get("p_value", 1.0),
                        "Wake Phases": len(phase_analysis.get("wake_phases", [])),
                        "Sleep Phases": len(phase_analysis.get("sleep_phases", [])),
                        "Wake Fraction (%)": phase_analysis.get("wake_fraction", 0)
                        * 100,
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 2: Parameters
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Minimum Period",
                        "Maximum Period",
                        "Significance Level",
                        "Phase Threshold",
                        "Sampling Interval",
                    ],
                    "Value": [
                        f"{self.fisher_min_period.value():.1f} hours",
                        f"{self.fisher_max_period.value():.1f} hours",
                        f"{self.fisher_significance.value():.3f}",
                        f"{self.fisher_phase_threshold.value():.2f}",
                        f"{self.frame_interval.value():.1f} seconds",
                    ],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

    def _create_fisher_plot(self, fisher_results: Dict[int, Dict]):
        """Create and display periodogram plot for Fisher analysis results."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from qtpy.QtGui import QPixmap
            import io

            # Create figure with subplots
            n_rois = len(fisher_results)
            n_significant = sum(
                1
                for r in fisher_results.values()
                if r.get("periodogram", {}).get("is_significant", False)
            )

            # Determine layout - max 3 columns
            n_cols = min(3, n_rois)
            n_rows = (n_rois + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False
            )
            fig.suptitle(
                f"Fischer Z-Transformation Periodogram\n{n_significant}/{n_rois} ROIs with Significant Rhythms",
                fontsize=14,
                fontweight="bold",
            )

            # Plot each ROI
            for idx, (roi_id, result) in enumerate(sorted(fisher_results.items())):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                periodogram = result.get("periodogram", {})

                if "error" in result or "error" in periodogram:
                    # Show error message
                    ax.text(
                        0.5,
                        0.5,
                        f"ROI {roi_id}\nInsufficient data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    # Plot periodogram
                    periods = periodogram.get("periods", [])
                    z_scores = periodogram.get("z_scores", [])
                    critical_z = periodogram.get("critical_z", 0)
                    is_significant = periodogram.get("is_significant", False)

                    # Plot Z-scores
                    ax.plot(periods, z_scores, "b-", linewidth=1.5, label="Z-score")

                    # Plot significance threshold
                    if critical_z > 0:
                        ax.axhline(
                            y=critical_z,
                            color="r",
                            linestyle="--",
                            linewidth=1,
                            label="Significance",
                        )

                    # Mark dominant period
                    if is_significant:
                        dominant_period = periodogram.get("dominant_period", 0)
                        dominant_z = periodogram.get("dominant_z_score", 0)
                        ax.plot(
                            dominant_period,
                            dominant_z,
                            "ro",
                            markersize=8,
                            label=f"Peak: {dominant_period:.1f}h",
                        )

                    # Styling
                    ax.set_xlabel("Period (hours)", fontsize=9)
                    ax.set_ylabel("Z-score", fontsize=9)
                    title_color = "green" if is_significant else "black"
                    ax.set_title(f"ROI {roi_id}", fontsize=10, color=title_color)
                    ax.legend(fontsize=7, loc="best")
                    ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(n_rois, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis("off")

            plt.tight_layout()

            # Convert to QPixmap and display
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self.fisher_plot_canvas.setPixmap(
                pixmap.scaled(
                    self.fisher_plot_canvas.size(),
                    1,  # KeepAspectRatio
                    1,  # SmoothTransformation
                )
            )

            plt.close(fig)

        except Exception as e:
            self._log_message(f"⚠️ Could not create Fisher plot: {e}")
            import traceback

            traceback.print_exc()

    # ===================================================================
    # FRAME VIEWER METHODS
    # ===================================================================

    def _viewer_load_data(self):
        """Load the current dataset into the frame viewer."""
        # Check if we have a loaded file
        if not hasattr(self, "file_path") or not self.file_path:
            self.viewer_status_label.setText(
                "⚠️ No file loaded. Please load a file in the Input tab first."
            )
            self._log_message("⚠️ Frame viewer: No file loaded")
            return

        try:

            # Check if HDF5 or AVI
            is_hdf5 = self.file_path.lower().endswith((".h5", ".hdf5"))
            is_avi = hasattr(self, "avi_batch_paths") and self.avi_batch_paths

            if is_avi:
                # Load AVI batch
                self._viewer_load_avi_batch()
            elif is_hdf5:
                # Load HDF5
                self._viewer_load_hdf5()
            else:
                self.viewer_status_label.setText("⚠️ Unsupported file format")
                self._log_message("⚠️ Frame viewer: Unsupported file format")

        except Exception as e:
            self.viewer_status_label.setText(f"❌ Error loading data: {e}")
            self._log_message(f"❌ Frame viewer error: {e}")
            import traceback

            traceback.print_exc()

    def _viewer_load_hdf5(self):
        """Load HDF5 file frames into viewer."""
        import h5py

        self._log_message(f"Loading HDF5 file into frame viewer: {self.file_path}")

        with h5py.File(self.file_path, "r") as f:
            # Get frame interval from metadata
            if "metadata" in f.attrs:
                import json

                metadata = json.loads(f.attrs["metadata"])
                self.viewer_frame_interval = metadata.get("frame_interval", 5.0)
            else:
                # Default to 5 seconds
                self.viewer_frame_interval = 5.0

            # Find the dataset - try multiple common names
            dataset_found = False

            # Try common dataset names in order
            for dataset_name in ["frames", "images", "data"]:
                if dataset_name in f:
                    data_obj = f[dataset_name]

                    if isinstance(data_obj, h5py.Dataset):
                        # Stacked frames format: (N, H, W) or (N, H, W, C)
                        self._log_message(
                            f"Found stacked dataset: {dataset_name} with shape {data_obj.shape}"
                        )
                        self.viewer_frames = data_obj
                        self.viewer_n_frames = (
                            data_obj.shape[0] if data_obj.ndim >= 3 else 1
                        )
                        self.viewer_file_handle = h5py.File(self.file_path, "r")
                        self.viewer_dataset_name = dataset_name
                        self.viewer_is_sequence = False
                        dataset_found = True
                        break
                    elif isinstance(data_obj, h5py.Group):
                        # Individual frames format: group with frame_XXXXXX datasets
                        frame_names = sorted(
                            [k for k in data_obj.keys() if k.startswith("frame_")]
                        )
                        if frame_names:
                            self._log_message(
                                f"Found individual frames in group: {dataset_name} ({len(frame_names)} frames)"
                            )
                            self.viewer_frames = None
                            self.viewer_frame_names = frame_names
                            self.viewer_n_frames = len(frame_names)
                            self.viewer_file_handle = h5py.File(self.file_path, "r")
                            self.viewer_dataset_name = dataset_name
                            self.viewer_is_sequence = True
                            dataset_found = True
                            break

            if not dataset_found:
                # List available keys for debugging
                available_keys = list(f.keys())
                self._log_message(f"Available HDF5 keys: {available_keys}")
                raise ValueError(
                    f"No 'frames', 'images', or 'data' dataset found in HDF5 file. "
                    f"Available keys: {available_keys}"
                )

        # Update UI
        self.viewer_current_frame = 0
        self.viewer_frame_slider.setMaximum(self.viewer_n_frames - 1)
        self.viewer_frame_slider.setValue(0)
        self.viewer_frame_slider.setEnabled(True)

        # Enable controls
        self.btn_viewer_first.setEnabled(True)
        self.btn_viewer_prev.setEnabled(True)
        self.btn_viewer_play.setEnabled(True)
        self.btn_viewer_next.setEnabled(True)
        self.btn_viewer_last.setEnabled(True)

        self.viewer_status_label.setText(
            f"✓ Loaded HDF5: {self.viewer_n_frames} frames (Interval: {self.viewer_frame_interval}s)"
        )
        self._log_message(
            f"✓ Frame viewer: Loaded {self.viewer_n_frames} frames from HDF5 (interval: {self.viewer_frame_interval}s)"
        )

        # Display first frame
        self._viewer_show_frame(0)

    def _viewer_load_avi_batch(self):
        """Load AVI batch frames into viewer."""

        self._log_message(
            f"Loading AVI batch into frame viewer: {len(self.avi_batch_paths)} files"
        )

        # Get frame interval from AVI metadata if available
        if hasattr(self, "avi_metadata") and self.avi_metadata:
            self.viewer_frame_interval = self.avi_metadata.get("frame_interval", 5.0)
        else:
            self.viewer_frame_interval = 5.0

        # For AVI, we'll use napari layers if available
        if len(self.viewer.layers) > 0:
            layer = self.viewer.layers[0]
            if hasattr(layer, "data"):
                self.viewer_frames = layer.data
                self.viewer_n_frames = (
                    layer.data.shape[0] if layer.data.ndim >= 3 else 1
                )
                self.viewer_file_handle = None
                self.viewer_is_sequence = False

                # Update UI
                self.viewer_current_frame = 0
                self.viewer_frame_slider.setMaximum(self.viewer_n_frames - 1)
                self.viewer_frame_slider.setValue(0)
                self.viewer_frame_slider.setEnabled(True)

                # Enable controls
                self.btn_viewer_first.setEnabled(True)
                self.btn_viewer_prev.setEnabled(True)
                self.btn_viewer_play.setEnabled(True)
                self.btn_viewer_next.setEnabled(True)
                self.btn_viewer_last.setEnabled(True)

                self.viewer_status_label.setText(
                    f"✓ Loaded AVI: {self.viewer_n_frames} frames (Interval: {self.viewer_frame_interval}s)"
                )
                self._log_message(
                    f"✓ Frame viewer: Loaded {self.viewer_n_frames} frames from AVI batch (interval: {self.viewer_frame_interval}s)"
                )

                # Display first frame
                self._viewer_show_frame(0)
            else:
                raise ValueError("No image data in viewer layer")
        else:
            raise ValueError("No layers in viewer. Please load AVI batch first.")

    def _viewer_show_frame(self, frame_idx):
        """Display a specific frame in the napari viewer."""
        try:
            if frame_idx < 0 or frame_idx >= self.viewer_n_frames:
                return

            # Get frame data
            if hasattr(self, "viewer_file_handle") and self.viewer_file_handle:
                # HDF5 file
                if hasattr(self, "viewer_frame_names"):
                    # Individual frame datasets
                    frame_name = self.viewer_frame_names[frame_idx]
                    frame_data = self.viewer_file_handle[
                        f"{self.viewer_dataset_name}/{frame_name}"
                    ][()]
                else:
                    # Single dataset
                    frame_data = self.viewer_file_handle[self.viewer_dataset_name][
                        frame_idx
                    ]
            else:
                # From napari layer (AVI or pre-loaded)
                frame_data = self.viewer_frames[frame_idx]

            # Calculate time from frame index
            frame_time_seconds = frame_idx * self.viewer_frame_interval
            frame_time_minutes = frame_time_seconds / 60.0
            frame_time_hours = frame_time_minutes / 60.0

            # Copy frame data and draw time text on it
            import numpy as np
            import cv2

            # Make a writable copy
            frame_with_text = np.array(frame_data, copy=True)

            # Ensure it's uint8 for cv2.putText
            if frame_with_text.dtype != np.uint8:
                # Normalize to 0-255 range
                frame_min = frame_with_text.min()
                frame_max = frame_with_text.max()
                if frame_max > frame_min:
                    frame_with_text = (
                        (frame_with_text - frame_min) / (frame_max - frame_min) * 255
                    ).astype(np.uint8)
                else:
                    frame_with_text = np.zeros_like(frame_with_text, dtype=np.uint8)

            # Ensure 2D shape for text overlay
            if frame_with_text.ndim == 3 and frame_with_text.shape[2] == 1:
                frame_with_text = frame_with_text[:, :, 0]

            # Convert to 3-channel BGR for colored text
            if len(frame_with_text.shape) == 2:
                frame_with_text = cv2.cvtColor(frame_with_text, cv2.COLOR_GRAY2BGR)

            # Prepare time text
            time_text = f"t = {frame_time_seconds:.1f}s ({frame_time_minutes:.2f}min)"

            # Position: lower left (10 pixels from left, 30 pixels from bottom)
            height, width = frame_with_text.shape[:2]
            text_position = (10, height - 10)

            # Draw text in red (BGR: 0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 0, 255)  # Red in BGR
            thickness = 2

            cv2.putText(
                frame_with_text,
                time_text,
                text_position,
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            # Create or update napari layer
            layer_name = "Frame Viewer"

            # Check if layer exists
            if layer_name in self.viewer.layers:
                # Update existing layer
                self.viewer.layers[layer_name].data = frame_with_text
            else:
                # Create new layer
                self.viewer.add_image(
                    frame_with_text,
                    name=layer_name,
                    colormap="gray",
                )

            # Update frame label with time
            self.viewer_frame_label.setText(
                f"Frame: {frame_idx + 1} / {self.viewer_n_frames} | "
                f"Time: {frame_time_seconds:.1f}s ({frame_time_minutes:.2f}min / {frame_time_hours:.3f}h)"
            )
            self.viewer_current_frame = frame_idx

            # Update info (use original frame_data, not the one with text)
            info_lines = [
                f"Frame: {frame_idx + 1} / {self.viewer_n_frames}",
                f"Time: {frame_time_seconds:.1f}s ({frame_time_minutes:.2f} min)",
                f"Hours: {frame_time_hours:.3f} h",
                f"Shape: {frame_data.shape}",
                f"Dtype: {frame_data.dtype}",
                f"Min/Max: {np.min(frame_data):.2f} / {np.max(frame_data):.2f}",
                f"Mean: {np.mean(frame_data):.2f}",
            ]
            self.viewer_info_text.setPlainText("\n".join(info_lines))

        except Exception as e:
            self._log_message(f"❌ Error showing frame {frame_idx}: {e}")

    def _on_viewer_frame_changed(self, value):
        """Handle slider value change."""
        self._viewer_show_frame(value)

    def _viewer_goto_frame(self, frame_idx):
        """Go to specific frame."""
        if frame_idx < 0:
            frame_idx = self.viewer_n_frames - 1
        frame_idx = max(0, min(self.viewer_n_frames - 1, frame_idx))
        self.viewer_frame_slider.setValue(frame_idx)

    def _viewer_step_frame(self, step):
        """Step forward or backward by n frames."""
        new_idx = self.viewer_current_frame + step
        self._viewer_goto_frame(new_idx)

    def _viewer_toggle_play(self):
        """Toggle playback on/off."""
        if self.btn_viewer_play.isChecked():
            # Start playing
            self.viewer_is_playing = True
            self.btn_viewer_play.setText("⏸ Pause")
            interval = int(1000 / self.viewer_fps_spin.value())
            self.viewer_timer.start(interval)
            self._log_message(f"▶ Playing at {self.viewer_fps_spin.value()} FPS")
        else:
            # Stop playing
            self.viewer_is_playing = False
            self.btn_viewer_play.setText("▶ Play")
            self.viewer_timer.stop()
            self._log_message("⏸ Paused")

    def _viewer_play_next_frame(self):
        """Advance to next frame during playback."""
        if self.viewer_is_playing:
            next_idx = self.viewer_current_frame + 1
            if next_idx >= self.viewer_n_frames:
                # Loop back to start
                next_idx = 0
            self._viewer_goto_frame(next_idx)

    def _viewer_update_timer_interval(self):
        """Update playback timer interval when FPS changes."""
        if self.viewer_is_playing:
            interval = int(1000 / self.viewer_fps_spin.value())
            self.viewer_timer.setInterval(interval)

    # ===================================================================
    # UTILITY METHODS
    # ===================================================================

    def _log_message(self, message: str):
        """Add message to analysis log with proper Qt handling."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Use QTimer.singleShot to ensure this runs in the main thread
        from qtpy.QtCore import QTimer

        def append_to_log():
            try:
                self.log_text.append(formatted_message)
                # Auto-scroll to bottom - use moveCursor instead of setTextCursor
                cursor = self.log_text.textCursor()
                cursor.movePosition(cursor.End)
                # Don't connect the cursor, just move to end
                self.log_text.moveCursor(cursor.End)
                self.log_text.ensureCursorVisible()
            except Exception as e:
                print(f"Logging error: {e}")

        # Execute in main thread
        QTimer.singleShot(0, append_to_log)

    def cleanup_resources(self):
        """Clean up resources when widget is destroyed."""
        if self.current_worker:
            self._cancel_requested = True

        if hasattr(self, "performance_timer"):
            self.performance_timer.stop()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup_resources()


# ===================================================================
# HELPER FUNCTIONS (MOVED FROM OUTSIDE CLASS)
# ===================================================================


def prepare_analysis_parameters(widget, method):
    """
    Prepare parameters for analysis based on widget state and method.
    """
    base_params = {
        "enable_matlab_norm": True,
        "enable_detrending": getattr(widget, "enable_detrending", None),
        "frame_interval": getattr(widget, "frame_interval", None),
    }

    # Extract values safely
    if hasattr(base_params["enable_detrending"], "isChecked"):
        base_params["enable_detrending"] = base_params["enable_detrending"].isChecked()
    else:
        base_params["enable_detrending"] = True

    if hasattr(base_params["frame_interval"], "value"):
        base_params["frame_interval"] = base_params["frame_interval"].value()
    else:
        base_params["frame_interval"] = 5.0

    # Method-specific parameters
    if method == "baseline":
        try:
            baseline_duration_minutes = widget.baseline_duration_minutes.value()
            multiplier = widget.threshold_multiplier.value()
            frame_interval = base_params["frame_interval"]

            base_params.update(
                {
                    "threshold_block_count": int(
                        (baseline_duration_minutes * 60) / frame_interval
                    ),
                    "multiplier": multiplier,
                    "enable_jump_correction": getattr(
                        widget, "enable_jump_correction", None
                    ),
                }
            )

            if hasattr(base_params["enable_jump_correction"], "isChecked"):
                base_params["enable_jump_correction"] = base_params[
                    "enable_jump_correction"
                ].isChecked()
            else:
                base_params["enable_jump_correction"] = True

        except Exception as e:
            print(f"Warning: Could not extract baseline parameters: {e}")

    elif method == "adaptive":
        try:
            duration_minutes = widget.adaptive_duration_minutes.value()
            frame_interval = base_params["frame_interval"]

            base_params.update(
                {
                    "analysis_duration_frames": int(
                        (duration_minutes * 60) / frame_interval
                    ),
                    "base_multiplier": widget.adaptive_base_multiplier.value(),
                }
            )
        except Exception as e:
            print(f"Warning: Could not extract adaptive parameters: {e}")

    elif method == "calibration":
        try:
            calibration_file = widget.calibration_file_path.property("full_path")

            base_params.update(
                {
                    "calibration_file_path": calibration_file,
                    "masks": getattr(widget, "masks", []),
                    "calibration_multiplier": widget.calibration_multiplier.value(),
                }
            )
        except Exception as e:
            print(f"Warning: Could not extract calibration parameters: {e}")

    return base_params


# Provide the dock widget to Napari
def napari_provide_dock_widget(viewer):
    return HDF5AnalysisWidget(viewer)
