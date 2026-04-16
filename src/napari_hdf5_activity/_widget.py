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
from qtpy.QtCore import QTimer, Signal, Qt, QSettings
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
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QCheckBox,
    QSlider,
    QSplitter,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
)

class _ScaledPixmapLabel(QLabel):
    """QLabel that scales its pixmap to fit while preserving aspect ratio."""

    def __init__(self):
        super().__init__()
        self._raw_pixmap = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._raw_pixmap = pixmap
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._raw_pixmap and not self._raw_pixmap.isNull():
            scaled = self._raw_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)


try:
    from ._reader import (
        detect_hdf5_structure_type,
        detect_file_structure_type,
        get_first_frame_enhanced,
        get_frame_norm_factor,
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
    from ._io_abstraction import open_file_reader, is_supported_file

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


def _fmt_p(p: float) -> str:
    """Format a p-value for display.

    Handles float underflow (p == 0.0) which occurs when N is very large and the
    test statistic exceeds the range of float64 survival functions.
    """
    if p == 0.0:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _parse_recording_start_datetime(file_path: str):
    """Extract recording start datetime from filename pattern YYYYMMDD_HHMMSS.

    Returns a datetime object or None if the pattern is not found.
    Example: 'nematostella_timelapse_20260126_181210.h5' → datetime(2026,1,26,18,12,10)
    """
    import re
    from datetime import datetime as _dt
    name = os.path.basename(file_path)
    m = re.search(r'(\d{8})_(\d{6})', name)
    if m:
        try:
            return _dt.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None
    return None


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
    if baseline_duration_minutes > 100000:
        return False, "Baseline duration seems unreasonably long (>100000 minutes)"
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


# ---------------------------------------------------------------------------
# Mixin imports — the widget class is split across several files to keep
# each file manageable.  All mixins share ``self`` through Python's MRO.
# ---------------------------------------------------------------------------
try:
    from ._widget_telemetry import TelemetryMixin
except ImportError as e:
    print(f"Warning: TelemetryMixin not available: {e}")
    class TelemetryMixin: pass  # noqa: E701

try:
    from ._widget_export import ExportMixin
except ImportError as e:
    print(f"Warning: ExportMixin not available: {e}")
    class ExportMixin: pass  # noqa: E701

try:
    from ._widget_frame_viewer import FrameViewerMixin
except ImportError as e:
    print(f"Warning: FrameViewerMixin not available: {e}")
    class FrameViewerMixin: pass  # noqa: E701

try:
    from ._widget_circadian import CircadianMixin
except ImportError as e:
    print(f"Warning: CircadianMixin not available: {e}")
    class CircadianMixin: pass  # noqa: E701


class HDF5AnalysisWidget(TelemetryMixin, ExportMixin, FrameViewerMixin, CircadianMixin, QWidget):
    """
    Widget for analyzing activity in HDF5/Zarr files.

    Core UI coordination and file I/O logic lives here.  Functionality is
    split across mixin classes (same ``self`` namespace via MRO):

    - :class:`._widget_telemetry.TelemetryMixin`   — HDF5/Zarr telemetry tab
    - :class:`._widget_export.ExportMixin`          — results save/export
    - :class:`._widget_frame_viewer.FrameViewerMixin` — frame viewer + video export
    - :class:`._widget_circadian.CircadianMixin`    — circadian rhythm analysis
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

        # Restore last ROI detection settings from previous session
        self._load_roi_settings()

    def _initialize_attributes(self):
        """Initialize all class attributes."""
        # Performance monitoring
        self.cpu_count = psutil.cpu_count()
        # Add dataset state management
        self._initialize_dataset_state()
        # Memory-aware process count for Windows (workers import napari/numba ~600MB each)
        if os.name == "nt":  # Windows
            try:
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                usable_mb = max(0, available_mb - 2048)  # Reserve 2GB for main process
                max_from_memory = max(1, int(usable_mb / 600))
                cpu_based = max(1, min(4, int(self.cpu_count * 0.6)))
                self.optimal_processes = min(cpu_based, max_from_memory)
            except Exception:
                self.optimal_processes = max(1, min(2, int(self.cpu_count * 0.5)))
        else:  # Unix/Linux/Mac
            self.optimal_processes = max(1, int(self.cpu_count * 0.9))

        # Analysis state variables
        self.directory: Optional[str] = None
        self.file_path: Optional[str] = None
        self.recording_start_datetime = None  # parsed from filename YYYYMMDD_HHMMSS
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
        self._analysis_generation = 0  # incremented on each start/stop to invalidate stale callbacks

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
        # Create main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create scroll area to ensure GUI fits in window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create container widget for tab widget
        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_input = QWidget()
        self.tab_analysis = QWidget()
        self.tab_results = QWidget()
        self.tab_extended = QWidget()
        self.tab_viewer = QWidget()
        self.tab_telemetry = QWidget()

        self.tab_widget.addTab(self.tab_input, "Input")
        self.tab_widget.addTab(self.tab_analysis, "Analysis")
        self.tab_widget.addTab(self.tab_results, "Results")
        self.tab_widget.addTab(self.tab_extended, "Extended Analysis")
        self.tab_widget.addTab(self.tab_viewer, "Frame Viewer")
        self.tab_widget.addTab(self.tab_telemetry, "HDF5 Telemetry")

        container_layout.addWidget(self.tab_widget)

        # Add container to scroll area
        scroll_area.setWidget(container)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        # Setup individual tabs
        self.setup_input_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
        self.setup_extended_tab()
        self.setup_viewer_tab()
        self.setup_telemetry_tab()

    def setup_input_tab(self):
        """Setup the input tab with file loading and ROI detection parameters."""
        layout = QVBoxLayout()
        self.tab_input.setLayout(layout)

        # File loading section
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        # Debug-Button hinzufügen
        self.btn_debug_structure = QPushButton("Debug File Structure")
        self.btn_debug_structure.setToolTip("Analyze HDF5 or Zarr file structure")
        self.btn_debug_structure.clicked.connect(self.debug_current_file_structure)
        file_layout.addWidget(self.btn_debug_structure)
        # File loading buttons
        self.btn_load_file = QPushButton("Load File")
        self.btn_load_file.setToolTip("Load HDF5 file or AVI video(s) for analysis")

        self.btn_load_zarr = QPushButton("Load Zarr")
        self.btn_load_zarr.setToolTip(
            "Load a Zarr store for analysis.\n"
            "• Zip store (.zarr file): click and select the .zarr file\n"
            "• Directory store (.zarr folder): use 'Load Directory' instead"
        )

        self.btn_load_dir = QPushButton("Load Directory")
        self.btn_load_dir.setToolTip("Load all HDF5/AVI files from a directory")

        self.btn_detect_rois = QPushButton("Detect ROIs")
        self.btn_detect_rois.setToolTip(
            "Automatically detect circular ROIs using HoughCircles"
        )

        self.btn_clear_rois = QPushButton("Clear ROI Detection")
        self.btn_clear_rois.setToolTip("Remove ROI detection layers")

        file_layout.addWidget(self.btn_load_file)
        file_layout.addWidget(self.btn_load_zarr)
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

        # Radius parameters (defaults for 6-well plate)
        self.min_radius = QSpinBox()
        self.min_radius.setRange(10, 1000)
        self.min_radius.setValue(100)
        self.min_radius.setToolTip("Minimum radius for circle detection (pixels)")
        roi_layout.addRow("Min Radius:", self.min_radius)

        self.max_radius = QSpinBox()
        self.max_radius.setRange(10, 1000)
        self.max_radius.setValue(145)
        self.max_radius.setToolTip("Maximum radius for circle detection (pixels)")
        roi_layout.addRow("Max Radius:", self.max_radius)

        # HoughCircles parameters
        self.dp_param = QDoubleSpinBox()
        self.dp_param.setRange(0.1, 5.0)
        self.dp_param.setValue(1.0)
        self.dp_param.setSingleStep(0.1)
        self.dp_param.setDecimals(1)
        self.dp_param.setToolTip(
            "Inverse ratio of accumulator resolution (1.0 recommended)"
        )
        roi_layout.addRow("DP Parameter:", self.dp_param)

        self.min_dist = QSpinBox()
        self.min_dist.setRange(10, 1000)
        self.min_dist.setValue(300)
        self.min_dist.setToolTip("Minimum distance between circle centers (pixels)")
        roi_layout.addRow("Min Distance:", self.min_dist)

        self.param1 = QSpinBox()
        self.param1.setRange(10, 300)
        self.param1.setValue(30)
        self.param1.setToolTip("Canny edge detection threshold (higher = fewer edges)")
        roi_layout.addRow("Param1 (Edge):", self.param1)

        self.param2 = QSpinBox()
        self.param2.setRange(5, 200)
        self.param2.setValue(60)
        self.param2.setToolTip("Circle detection sensitivity (lower = more circles)")
        roi_layout.addRow("Param2 (Center):", self.param2)

        # Plate presets
        self.chk_6well = QCheckBox("6-Well Plate Preset")
        self.chk_6well.setToolTip(
            "Use preset values for 6-well plates (radius 100-145)"
        )
        self.chk_6well.setChecked(True)
        roi_layout.addRow("", self.chk_6well)

        self.chk_12well = QCheckBox("12-Well Plate Preset")
        self.chk_12well.setToolTip(
            "Use preset values for 12-well plates (radius 70-120)"
        )
        roi_layout.addRow("", self.chk_12well)

        # ROI Scale slider - scale detected ROIs from center
        self.roi_scale = QDoubleSpinBox()
        self.roi_scale.setRange(0.1, 2.0)
        self.roi_scale.setValue(1.0)
        self.roi_scale.setSingleStep(0.05)
        self.roi_scale.setToolTip(
            "Scale ROIs from center (1.0 = original size, 0.8 = 80%)"
        )
        roi_layout.addRow("ROI Scale:", self.roi_scale)

        # Apply scale button
        self.btn_apply_scale = QPushButton("Apply Scale")
        self.btn_apply_scale.setToolTip("Re-scale detected ROIs without re-detecting")
        self.btn_apply_scale.setEnabled(False)
        roi_layout.addRow("", self.btn_apply_scale)

        # Manual ROI editing
        edit_btn_layout = QHBoxLayout()
        self.btn_edit_rois = QPushButton("Edit ROI Circles")
        self.btn_edit_rois.setToolTip(
            "Add draggable circles to the napari viewer.\n"
            "Move/resize them, then click 'Apply Edits' to update masks."
        )
        self.btn_edit_rois.setEnabled(False)
        self.btn_apply_roi_edits = QPushButton("Apply Edits")
        self.btn_apply_roi_edits.setToolTip("Convert the edited circles into new ROI masks")
        self.btn_apply_roi_edits.setEnabled(False)
        self.btn_apply_roi_edits.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        edit_btn_layout.addWidget(self.btn_edit_rois)
        edit_btn_layout.addWidget(self.btn_apply_roi_edits)
        roi_layout.addRow("", edit_btn_layout)

        layout.addWidget(roi_group)

        # ROI Selection (exclude/include)
        roi_select_group = QGroupBox("ROI Selection (exclude empty wells)")
        self.roi_select_layout = QVBoxLayout()
        roi_select_group.setLayout(self.roi_select_layout)

        self.roi_select_info = QLabel("Run ROI detection first")
        self.roi_select_info.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        self.roi_select_layout.addWidget(self.roi_select_info)

        # Buttons for quick select/deselect
        roi_btn_layout = QHBoxLayout()
        self.btn_select_all_rois = QPushButton("Select All")
        self.btn_deselect_all_rois = QPushButton("Deselect All")
        self.btn_select_all_rois.clicked.connect(lambda: self._set_all_roi_checkboxes(True))
        self.btn_deselect_all_rois.clicked.connect(lambda: self._set_all_roi_checkboxes(False))
        roi_btn_layout.addWidget(self.btn_select_all_rois)
        roi_btn_layout.addWidget(self.btn_deselect_all_rois)
        roi_btn_layout.addStretch()
        self.roi_select_layout.addLayout(roi_btn_layout)

        # Container for dynamic checkboxes
        self.roi_checkboxes_layout = QHBoxLayout()
        self.roi_select_layout.addLayout(self.roi_checkboxes_layout)
        self.roi_checkboxes: list = []

        layout.addWidget(roi_select_group)
        layout.addStretch()

    def _populate_roi_checkboxes(self, n_rois: int):
        """Create checkboxes for each detected ROI."""
        # Clear existing
        for cb in self.roi_checkboxes:
            cb.setParent(None)
        self.roi_checkboxes.clear()

        # Create new checkboxes
        for i in range(n_rois):
            cb = QCheckBox(f"ROI {i + 1}")
            cb.setChecked(True)
            cb.setToolTip(f"Include ROI {i + 1} in analysis")
            self.roi_checkboxes_layout.addWidget(cb)
            self.roi_checkboxes.append(cb)

        self.roi_select_info.setText(f"{n_rois} ROIs detected — uncheck to exclude from analysis")

    def _set_all_roi_checkboxes(self, checked: bool):
        """Select or deselect all ROI checkboxes."""
        for cb in self.roi_checkboxes:
            cb.setChecked(checked)

    def _get_excluded_roi_indices(self) -> list:
        """Return list of ROI indices (0-based) that are unchecked."""
        return [i for i, cb in enumerate(self.roi_checkboxes) if not cb.isChecked()]

    def _get_active_masks(self) -> list:
        """Return only the masks for checked (included) ROIs."""
        if not self.roi_checkboxes:
            return self.masks  # No checkboxes = use all
        return [m for i, m in enumerate(self.masks) if i < len(self.roi_checkboxes) and self.roi_checkboxes[i].isChecked()]

    def _get_roi_index_mapping(self) -> dict:
        """Return mapping from sequential reader ROI index (1-based) to original ROI index.

        When ROIs are excluded, the reader assigns sequential indices 1..N to the
        N active masks.  This mapping restores the original ROI numbering.
        Example: if ROI 1 is excluded from 6 ROIs, returns {1:2, 2:3, 3:4, 4:5, 5:6}.
        If no ROIs are excluded, returns identity mapping {1:1, 2:2, ...}.
        """
        if not self.roi_checkboxes:
            return {i + 1: i + 1 for i in range(len(self.masks))}
        active_original_indices = [
            i for i, cb in enumerate(self.roi_checkboxes) if cb.isChecked()
        ]
        # reader_idx is 1-based, original is also 1-based (i+1)
        return {
            reader_idx: orig_0based + 1
            for reader_idx, orig_0based in enumerate(active_original_indices, start=1)
        }

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
        self.chunk_size.setValue(10)
        self.chunk_size.setToolTip("Number of frames to process in each chunk (lower = less RAM)")
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

        # === SHARED PREPROCESSING (applies to all threshold methods) ===
        preprocessing_layout = QHBoxLayout()

        self.enable_detrending = QCheckBox("Enable Detrending")
        self.enable_detrending.setChecked(False)
        self.enable_detrending.setToolTip(
            "Remove linear drift from the signal for more accurate thresholds.\n"
            "Applies to all threshold methods."
        )
        preprocessing_layout.addWidget(self.enable_detrending)

        self.enable_jump_correction = QCheckBox("Jump Correction (frame mean)")
        self.enable_jump_correction.setChecked(False)
        self.enable_jump_correction.setToolTip(
            "Detect and correct sudden illumination jumps.\n"
            "Uses HDF5 frame_mean telemetry when available (recommended);\n"
            "falls back to signal-based detection otherwise.\n"
            "Applies to all threshold methods."
        )
        preprocessing_layout.addWidget(self.enable_jump_correction)

        self.adaptive_illumination_baseline = QCheckBox("Adaptive Illumination Baseline")
        self.adaptive_illumination_baseline.setChecked(False)
        self.adaptive_illumination_baseline.setToolTip(
            "Re-compute baseline per light/dark period from HDF5 LED data.\n"
            "Falls back to global baseline when no LED data is available.\n"
            "Applies to all threshold methods."
        )
        preprocessing_layout.addWidget(self.adaptive_illumination_baseline)
        preprocessing_layout.addStretch()
        threshold_layout.addLayout(preprocessing_layout)

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

        # === METHOD 4: FIXED THRESHOLD ===
        fixed_tab = QWidget()
        fixed_layout = QFormLayout()
        fixed_tab.setLayout(fixed_layout)

        self.fixed_threshold_value = QDoubleSpinBox()
        self.fixed_threshold_value.setRange(0.0001, 1.0)
        self.fixed_threshold_value.setValue(0.05)
        self.fixed_threshold_value.setSingleStep(0.005)
        self.fixed_threshold_value.setDecimals(4)
        self.fixed_threshold_value.setToolTip(
            "Fixed threshold in normalized signal units [0-1].\n"
            "Signal > threshold → Movement = TRUE\n"
            "Signal < threshold × hysteresis_ratio → Movement = FALSE\n\n"
            "Read the suggested value from 'Signal stats' below.\n"
            "Typical range: 0.02 – 0.15 depending on recording."
        )
        fixed_layout.addRow("Threshold (norm. 0-1):", self.fixed_threshold_value)

        self.fixed_threshold_hysteresis = QDoubleSpinBox()
        self.fixed_threshold_hysteresis.setRange(0.1, 1.0)
        self.fixed_threshold_hysteresis.setValue(0.8)
        self.fixed_threshold_hysteresis.setSingleStep(0.05)
        self.fixed_threshold_hysteresis.setDecimals(2)
        self.fixed_threshold_hysteresis.setToolTip(
            "Lower threshold = fixed value × this ratio.\n"
            "0.8 → lower = 80% of upper (20% hysteresis band)."
        )
        fixed_layout.addRow("Hysteresis ratio:", self.fixed_threshold_hysteresis)

        fixed_info = QLabel(
            "FIXED THRESHOLD:\n"
            "Uses a paper-defined absolute pixel value.\n"
            "Upper = threshold value\n"
            "Lower = threshold × hysteresis ratio\n"
            "No baseline period needed."
        )
        fixed_info.setStyleSheet("color: #666; font-size: 10px;")
        fixed_info.setWordWrap(True)
        fixed_layout.addRow("", fixed_info)

        self.fixed_signal_stats_label = QLabel("Signal range: run analysis first")
        self.fixed_signal_stats_label.setStyleSheet(
            "color: #4af; font-size: 10px; font-family: monospace;"
        )
        self.fixed_signal_stats_label.setWordWrap(True)
        fixed_layout.addRow("Signal stats:", self.fixed_signal_stats_label)

        self.btn_apply_fixed_threshold = QPushButton("Apply Fixed Threshold")
        self.btn_apply_fixed_threshold.setEnabled(False)
        self.btn_apply_fixed_threshold.setToolTip(
            "Re-run only movement detection and downstream analysis with the current\n"
            "fixed threshold — no need to redo preprocessing."
        )
        self.btn_apply_fixed_threshold.setStyleSheet(
            "QPushButton { background-color: #2a6496; color: white; font-weight: bold; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        fixed_layout.addRow("", self.btn_apply_fixed_threshold)

        self.threshold_params_stack.addTab(fixed_tab, "Fixed Threshold")

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
        self.quiescence_threshold.setToolTip(
            "Quiescence threshold for fraction movement:\n"
            "• fraction_movement < threshold → Quiescence = YES (resting)\n"
            "• fraction_movement ≥ threshold → Quiescence = NO (active)\n"
            "Default: 0.5 (less than 50% movement = quiescent)"
        )
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
        self.btn_quick_test.setToolTip("Run quick analysis test on loaded data (HDF5 and Zarr)")
        self.btn_validate_timing = QPushButton("Validate Timing")
        self.btn_validate_timing.setToolTip("Check recording timing quality (HDF5 and Zarr)")

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

        # Matplotlib figure — use a modest initial size; the canvas expands to fill
        # the tab and tight_layout() is called after each draw to use available space
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
                "Sleep Quality",
                "Lighting Conditions (dark IR)",
            ]
        )

        self.plot_dpi_spin = QSpinBox()
        self.plot_dpi_spin.setRange(50, 1200)
        self.plot_dpi_spin.setValue(600)

        basic_row.addWidget(QLabel("Plot Type:"))
        basic_row.addWidget(self.plot_type_combo)

        # Sleep Quality metric sub-selector (visible only when Sleep Quality selected)
        self.sleep_quality_metric_combo = QComboBox()
        self.sleep_quality_metric_combo.addItems([
            "Sleep min/h",
            "Transitions/h",
            "Bout Length/h",
            "Sleep h/day",
        ])
        self.sleep_quality_metric_combo.setVisible(False)
        self.sleep_quality_metric_combo.currentIndexChanged.connect(lambda _: self.generate_plot())
        basic_row.addWidget(self.sleep_quality_metric_combo)

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

        # Amplitude mode toggle
        self.show_real_amplitude = QCheckBox("Show Real Amplitude (instead of 0-1 normalized)")
        self.show_real_amplitude.setChecked(False)
        self.show_real_amplitude.setToolTip(
            "Toggle between MinMax-normalized [0,1] view and real amplitude values.\n"
            "Real amplitude shows sum(|Δpixel|) per ROI in raw pixel counts (MATLAB-style)."
        )
        plot_config_layout.addWidget(self.show_real_amplitude)

        self.chk_divide_by_pixels = QCheckBox("÷ ROI pixel count (per-pixel mean)")
        self.chk_divide_by_pixels.setChecked(False)   # default: pixel sum (MATLAB-style)
        self.chk_divide_by_pixels.setEnabled(False)   # enabled only when Real Amplitude is on
        self.chk_divide_by_pixels.setToolTip(
            "Unchecked (default): pixel sum — sum(|Δpixel|) per ROI, MATLAB-equivalent\n"
            "Checked: per-pixel mean — divide by ROI pixel count, ROI-size independent"
        )
        self.chk_divide_by_pixels.toggled.connect(self.generate_plot)
        # Enable/disable together with Real Amplitude toggle
        self.show_real_amplitude.toggled.connect(self.chk_divide_by_pixels.setEnabled)
        plot_config_layout.addWidget(self.chk_divide_by_pixels)

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

        # Per-ROI Y-axis limits (Raw Intensity only)
        self.per_roi_y_group = QGroupBox("Per-ROI Y-Axis Limits (Raw Intensity only)")
        self.per_roi_y_group.setCheckable(True)
        self.per_roi_y_group.setChecked(False)
        self.per_roi_y_group.setToolTip(
            "Set individual Y-axis limits for each ROI subplot.\n"
            "Enable the group, then click 'Read from Plot' to pre-fill\n"
            "current limits, adjust, and click Apply per ROI."
        )
        per_roi_outer_layout = QVBoxLayout()
        self.per_roi_y_group.setLayout(per_roi_outer_layout)

        # Scrollable area for the per-ROI rows
        self.per_roi_scroll = QScrollArea()
        self.per_roi_scroll.setWidgetResizable(True)
        self.per_roi_scroll.setMaximumHeight(180)
        self.per_roi_scroll.setFrameShape(self.per_roi_scroll.NoFrame)
        self.per_roi_inner = QWidget()
        self.per_roi_inner_layout = QVBoxLayout()
        self.per_roi_inner_layout.setSpacing(2)
        self.per_roi_inner_layout.setContentsMargins(0, 0, 0, 0)
        self.per_roi_inner.setLayout(self.per_roi_inner_layout)
        self.per_roi_scroll.setWidget(self.per_roi_inner)
        per_roi_outer_layout.addWidget(self.per_roi_scroll)

        per_roi_btn_row = QHBoxLayout()
        self.btn_per_roi_read = QPushButton("Read from Plot")
        self.btn_per_roi_read.setToolTip(
            "Copy the current Y-axis limits from each ROI subplot into the fields below."
        )
        self.btn_per_roi_read.clicked.connect(self._read_current_ylimits)
        self.btn_per_roi_reset = QPushButton("Reset All")
        self.btn_per_roi_reset.setToolTip("Clear all per-ROI limits (revert to auto-scaling).")
        self.btn_per_roi_reset.clicked.connect(self._reset_per_roi_ylimits)
        self.btn_per_roi_refresh = QPushButton("Refresh ROI List")
        self.btn_per_roi_refresh.setToolTip(
            "Rebuild the ROI list from the currently loaded data."
        )
        self.btn_per_roi_refresh.clicked.connect(self._refresh_per_roi_controls)
        per_roi_btn_row.addWidget(self.btn_per_roi_read)
        per_roi_btn_row.addWidget(self.btn_per_roi_reset)
        per_roi_btn_row.addWidget(self.btn_per_roi_refresh)
        per_roi_btn_row.addStretch()
        per_roi_outer_layout.addLayout(per_roi_btn_row)

        plot_config_layout.addWidget(self.per_roi_y_group)

        # Internal state for per-ROI limits
        self.per_roi_y_limits: dict = {}        # {roi_id: (y_min, y_max)}
        self._per_roi_y_widgets: dict = {}      # {roi_id: (y_min_spin, y_max_spin)}

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

        # Plot binning configuration (separate from analysis binning)
        plot_binning_group = QGroupBox("Plot Binning (Quiescence & Sleep re-derived from rebinned data)")
        plot_binning_layout = QHBoxLayout()
        plot_binning_group.setLayout(plot_binning_layout)

        self.plot_bin_minutes = QSpinBox()
        self.plot_bin_minutes.setRange(0, 240)  # 0 = no rebinning, 1-240 minutes
        self.plot_bin_minutes.setValue(10)  # Default 10 minutes
        self.plot_bin_minutes.setSuffix(" min")
        self.plot_bin_minutes.setSpecialValueText("Original (1 min)")
        self.plot_bin_minutes.setToolTip(
            "Re-bin data for visualization purposes.\n"
            "0 = Use original binning from analysis (typically 1 min bins)\n"
            "This does NOT affect analysis calculations.\n"
            "Useful for publications (e.g., 60 min for circadian plots).\n"
            "Original analysis uses 'Bin Size' from Behavior Analysis section."
        )

        # Preset buttons for common binning intervals
        self.btn_plot_bin_original = QPushButton("Original")
        self.btn_plot_bin_original.setToolTip("Use original binning (no re-binning)")
        self.btn_plot_bin_10min = QPushButton("10 min")
        self.btn_plot_bin_10min.setToolTip("Set plot binning to 10 minutes")
        self.btn_plot_bin_30min = QPushButton("30 min")
        self.btn_plot_bin_30min.setToolTip("Set plot binning to 30 minutes")
        self.btn_plot_bin_60min = QPushButton("60 min")
        self.btn_plot_bin_60min.setToolTip("Set plot binning to 60 minutes (1 hour)")

        # Connect preset buttons
        self.btn_plot_bin_original.clicked.connect(
            lambda: self.plot_bin_minutes.setValue(0)
        )
        self.btn_plot_bin_10min.clicked.connect(
            lambda: self.plot_bin_minutes.setValue(10)
        )
        self.btn_plot_bin_30min.clicked.connect(
            lambda: self.plot_bin_minutes.setValue(30)
        )
        self.btn_plot_bin_60min.clicked.connect(
            lambda: self.plot_bin_minutes.setValue(60)
        )

        # Auto-refresh plot when bin size changes
        self.plot_bin_minutes.valueChanged.connect(self._on_plot_bin_changed)

        plot_binning_layout.addWidget(QLabel("Plot Bin Size:"))
        plot_binning_layout.addWidget(self.plot_bin_minutes)
        plot_binning_layout.addWidget(QLabel("Presets:"))
        plot_binning_layout.addWidget(self.btn_plot_bin_original)
        plot_binning_layout.addWidget(self.btn_plot_bin_10min)
        plot_binning_layout.addWidget(self.btn_plot_bin_30min)
        plot_binning_layout.addWidget(self.btn_plot_bin_60min)
        plot_binning_layout.addStretch()

        # ZT time axis and lighting overlay options
        self.chk_zt_axis = QCheckBox("ZT time (h)")
        self.chk_zt_axis.setToolTip("X-axis in hours from recording start (ZT 0 = start)")
        self.chk_zt_axis.toggled.connect(lambda _: self.generate_plot())

        self.chk_show_lighting = QCheckBox("Light/Dark")
        self.chk_show_lighting.setToolTip("Overlay day/night shading from HDF5 LED data")
        self.chk_show_lighting.toggled.connect(lambda _: self.generate_plot())

        plot_binning_layout.addWidget(self.chk_zt_axis)
        plot_binning_layout.addWidget(self.chk_show_lighting)

        layout.addWidget(time_range_group)
        layout.addWidget(plot_binning_group)
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

        self.btn_save_individual_rois = QPushButton("Save ROIs individually...")
        self.btn_save_individual_rois.setToolTip(
            "Save one PNG per ROI for the current plot type into a folder"
        )

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
        plot_buttons_layout.addWidget(self.btn_save_individual_rois)
        plot_buttons_layout.addWidget(self.btn_save_all_plots)

        plot_buttons_layout.addWidget(self.btn_save_results)
        plot_buttons_layout.addWidget(self.btn_save_with_metadata)
        layout.addWidget(plot_buttons_group)

    def setup_extended_tab(self):
        """Setup the Extended Analysis tab for rhythmic pattern detection."""
        layout = QVBoxLayout()
        self.tab_extended.setLayout(layout)

        # Title and description
        title_label = QLabel("Rhythmic Pattern Analysis")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        desc_label = QLabel(
            "Detect any recurring periodic patterns in activity data: "
            "circadian rhythms (24h), ultradian rhythms (<20h), short activity cycles, "
            "or custom periods using Fischer Z-transformation, FFT, and correlation methods."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        layout.addWidget(desc_label)

        # Period range parameters
        fisher_params_group = QGroupBox("Period Range Parameters")
        fisher_params_layout = QFormLayout()
        fisher_params_group.setLayout(fisher_params_layout)

        self.fisher_min_period = QDoubleSpinBox()
        self.fisher_min_period.setRange(0.1, 100.0)
        self.fisher_min_period.setValue(0.5)
        self.fisher_min_period.setSingleStep(0.5)
        self.fisher_min_period.setDecimals(1)
        self.fisher_min_period.setSuffix(" hours")
        self.fisher_min_period.setToolTip(
            "Minimum period to test:\n"
            "• 0.5-2h: Very fast ultradian rhythms\n"
            "• 2-8h: Standard ultradian rhythms\n"
            "• 8-12h: Extended ultradian/tidal rhythms\n"
            "• 20-22h: Circadian rhythms (24h cycles)"
        )
        fisher_params_layout.addRow("Minimum Period:", self.fisher_min_period)

        self.fisher_max_period = QDoubleSpinBox()
        self.fisher_max_period.setRange(0.1, 200.0)
        self.fisher_max_period.setValue(8.0)
        self.fisher_max_period.setSingleStep(1.0)
        self.fisher_max_period.setDecimals(1)
        self.fisher_max_period.setSuffix(" hours")
        self.fisher_max_period.setToolTip(
            "Maximum period to test:\n"
            "• 2-8h: Fast ultradian rhythms only\n"
            "• 8-20h: Include infradian/extended rhythms\n"
            "• 20-28h: Include circadian (24h) rhythms\n"
            "• >28h: Longer multi-day cycles (requires 72h+ data)"
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

        layout.addWidget(fisher_params_group)

        # Target period for Similarity / Coherence / Phase Clustering
        target_period_group = QGroupBox("Target Period (Similarity / Coherence / Phase)")
        target_period_layout = QFormLayout()
        target_period_group.setLayout(target_period_layout)

        self.target_period = QDoubleSpinBox()
        self.target_period.setRange(0.5, 200.0)
        self.target_period.setValue(24.0)
        self.target_period.setSingleStep(1.0)
        self.target_period.setDecimals(1)
        self.target_period.setSuffix(" hours")
        self.target_period.setToolTip(
            "Expected rhythm period for Similarity Matrix, Coherence Analysis\n"
            "and Phase Clustering.\n\n"
            "• 24 h — circadian (default)\n"
            "• 12 h — semidiurnal / tidal\n"
            "• 8 h — ultradian\n\n"
            "This is independent of the chi² period search range above.\n"
            "It controls the cross-correlation lag window (±T/2) and the\n"
            "Welch segment length used for coherence estimation."
        )
        target_period_layout.addRow("Target Period:", self.target_period)

        layout.addWidget(target_period_group)

        # Quick preset buttons
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout()
        preset_group.setLayout(preset_layout)

        from qtpy.QtWidgets import QPushButton

        btn_preset_short = QPushButton("Ultradian\n(0.5-8h)")
        btn_preset_short.setToolTip(
            "Ultradian rhythms: Periods shorter than ~12 hours\n"
            "Good for: Fast oscillations, feeding cycles, short activity bouts\n"
            "Requires: Any recording length"
        )
        btn_preset_short.clicked.connect(lambda: self._set_period_preset(0.5, 8.0))
        preset_layout.addWidget(btn_preset_short)

        btn_preset_medium = QPushButton("Infradian\n(8-20h)")
        btn_preset_medium.setToolTip(
            "Infradian rhythms: Periods between ultradian and circadian\n"
            "Good for: Extended activity/rest cycles, tidal rhythms\n"
            "Requires: 24+ hour recordings"
        )
        btn_preset_medium.clicked.connect(lambda: self._set_period_preset(8.0, 20.0))
        preset_layout.addWidget(btn_preset_medium)

        btn_preset_circadian = QPushButton("Circadian\n(20-28h)")
        btn_preset_circadian.setToolTip(
            "Circadian rhythms: ~24-hour day/night cycles\n"
            "Good for: Daily activity patterns, sleep/wake cycles\n"
            "Requires: 48+ hour recordings for reliable detection"
        )
        btn_preset_circadian.clicked.connect(
            lambda: self._set_period_preset(20.0, 28.0)
        )
        preset_layout.addWidget(btn_preset_circadian)

        btn_preset_auto = QPushButton("Auto-Detect\nRange")
        btn_preset_auto.setToolTip(
            "Automatically determine optimal period range based on recording duration"
        )
        btn_preset_auto.clicked.connect(self._auto_detect_period_range)
        preset_layout.addWidget(btn_preset_auto)

        layout.addWidget(preset_group)

        # Cosinor-specific option
        self.chk_cosinor_population = QCheckBox("Show population mean")
        self.chk_cosinor_population.setToolTip(
            "Add a full-width subplot below the per-ROI panels showing\n"
            "the population-level summary across all ROIs.\n\n"
            "• Cosinor: population cosinor fit (mean MESOR, amplitude, phase)\n"
            "• FFT: mean power ± SEM across all ROIs\n"
            "• Chi² Periodogram: mean Z-score ± SEM across all ROIs\n"
            "• Phase Clustering: mean resultant vector (circular mean + R)"
        )
        self.chk_cosinor_population.toggled.connect(self._rerender_current_fisher_plot)

        self.population_peak_mode = QComboBox()
        self.population_peak_mode.addItems(["Median", "Mean"])
        self.population_peak_mode.setToolTip(
            "How the vertical peak marker in the population mean panel is computed.\n"
            "• Median: median of dominant periods from significant ROIs\n"
            "• Mean: peak (argmax) of the averaged power/Z-score spectrum"
        )
        self.population_peak_mode.currentIndexChanged.connect(self._rerender_current_fisher_plot)

        pop_row = QHBoxLayout()
        pop_row.setContentsMargins(0, 0, 0, 0)
        pop_row.addWidget(self.chk_cosinor_population)
        pop_row.addWidget(QLabel("Peak:"))
        pop_row.addWidget(self.population_peak_mode)
        pop_row.addStretch()
        layout.addLayout(pop_row)

        # Analysis method selection
        method_group = QGroupBox("Analysis Method")
        method_layout = QFormLayout()
        method_group.setLayout(method_layout)

        self.fisher_method_combo = QComboBox()
        self.fisher_method_combo.addItems(
            [
                "Chi² Periodogram",
                "FFT Power Spectrum",
                "Cosinor Analysis",
                "ROI Similarity Matrix",
                "Coherence Analysis",
                "Phase Clustering",
            ]
        )
        self.fisher_method_combo.setToolTip(
            "Select the rhythmic pattern analysis method:\n\n"
            "• Chi² Periodogram: Most robust. Tests all periods in range using chi-square\n"
            "  statistic. Recommended default. Use Fraction Movement.\n\n"
            "• FFT Power Spectrum: Good for noisy data. Permutation-based significance\n"
            "  (1000 permutations). Use Fraction Movement.\n\n"
            "• Cosinor Analysis: Fits cosine curve — gives MESOR, Amplitude, Acrophase.\n"
            "  Tests specific periods (12, 24, 30 h + min/mid/max of range).\n"
            "  Use Fraction or Normalized Movement. Needs ≥ 2 complete cycles.\n\n"
            "• ROI Similarity Matrix: Pairwise cross-correlation between all ROIs at\n"
            "  optimal time lag (≤ max_period / 2). Hierarchical clustering at r = 0.5.\n"
            "  Period range does not affect result. Use Fraction Movement.\n\n"
            "• Coherence Analysis: Welch magnitude-squared coherence between ROI pairs\n"
            "  at one target period = midpoint of period range.\n"
            "  ⚠ Set period range so that (min + max) / 2 = target period\n"
            "  (e.g. 20–28 h for circadian → target = 24 h).\n"
            "  Needs long recordings (≥ 3× target period) for good frequency resolution.\n"
            "  Recommended bin size: 30–60 min.\n\n"
            "• Phase Clustering: Hilbert-transform phase per ROI → peak activity time.\n"
            "  Bins ROIs into 4 groups: 0–6 h / 6–12 h / 12–18 h / 18–24 h.\n"
            "  ⚠ Set period range so midpoint = target period (same as Coherence).\n"
            "  Reliable only when the target rhythm is strong and dominant in the signal.\n"
            "  Phase is relative to recording start (ZT 0 = start of recording)."
        )
        self.fisher_method_combo.currentIndexChanged.connect(
            self._on_fisher_method_changed
        )
        method_layout.addRow("Method:", self.fisher_method_combo)

        layout.addWidget(method_group)

        # Cluster threshold slider — only visible for ROI Similarity Matrix (method index 3)
        self.similarity_threshold_group = QGroupBox("Cluster Threshold")
        similarity_thresh_layout = QHBoxLayout()
        self.similarity_threshold_group.setLayout(similarity_thresh_layout)

        similarity_thresh_layout.addWidget(QLabel("r ="))

        self.similarity_threshold_slider = QSlider(Qt.Horizontal)
        self.similarity_threshold_slider.setRange(0, 100)
        self.similarity_threshold_slider.setValue(50)  # default r = 0.50
        self.similarity_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.similarity_threshold_slider.setTickInterval(10)
        self.similarity_threshold_slider.setToolTip(
            "Minimum Pearson r to merge ROIs into the same cluster.\n"
            "Higher = fewer, tighter clusters.\n"
            "Default: 0.50  (distance cut = 0.50)"
        )
        self.similarity_threshold_slider.valueChanged.connect(
            self._on_similarity_threshold_changed
        )
        similarity_thresh_layout.addWidget(self.similarity_threshold_slider)

        self.similarity_threshold_label = QLabel("0.50")
        self.similarity_threshold_label.setMinimumWidth(35)
        similarity_thresh_layout.addWidget(self.similarity_threshold_label)

        self.similarity_threshold_group.setVisible(False)  # shown only for Similarity method
        layout.addWidget(self.similarity_threshold_group)

        # Actogram settings group — only visible for Actogram method (index 6)
        self.actogram_settings_group = QGroupBox("Actogram Settings")
        actogram_settings_layout = QFormLayout()
        self.actogram_settings_group.setLayout(actogram_settings_layout)

        self.actogram_period = QDoubleSpinBox()
        self.actogram_period.setRange(1.0, 72.0)
        self.actogram_period.setValue(24.0)
        self.actogram_period.setSingleStep(0.5)
        self.actogram_period.setDecimals(1)
        self.actogram_period.setSuffix(" h")
        self.actogram_period.setToolTip(
            "Row period τ for the actogram (hours).\n"
            "Each row covers one τ of time; data is plotted twice (double-plot).\n"
            "Use 24 h for entrained rhythms, or set to τ (free-running period) from Chi²."
        )
        actogram_settings_layout.addRow("Row period τ:", self.actogram_period)

        self.actogram_chk_zt_axis = QCheckBox("ZT time (h)")
        self.actogram_chk_zt_axis.setToolTip(
            "Label X-axis as ZT (Zeitgeber Time).\n"
            "ZT 0 = recording start (= lights-on if started at ZT 0)."
        )
        self.actogram_chk_zt_axis.stateChanged.connect(self._rerender_actogram)
        actogram_settings_layout.addRow("", self.actogram_chk_zt_axis)

        self.actogram_chk_show_lighting = QCheckBox("Light/Dark shading")
        self.actogram_chk_show_lighting.setChecked(True)
        self.actogram_chk_show_lighting.setToolTip(
            "Overlay yellow light-phase shading from HDF5 LED data.\n"
            "Only available when LED data is present in the HDF5 file."
        )
        self.actogram_chk_show_lighting.stateChanged.connect(self._rerender_actogram)
        actogram_settings_layout.addRow("", self.actogram_chk_show_lighting)

        # Hide τ spinbox — Cosinor always auto-detects τ from the fit
        tau_label = actogram_settings_layout.labelForField(self.actogram_period)
        if tau_label:
            tau_label.setVisible(False)
        self.actogram_period.setVisible(False)

        self.actogram_settings_group.setVisible(False)  # shown when checkbox is checked

        # Re-Binning Options for Extended Analysis
        rebinning_group = QGroupBox("Analysis Binning (Optional)")
        rebinning_layout = QVBoxLayout()
        rebinning_group.setLayout(rebinning_layout)

        # Info label
        rebinning_info = QLabel(
            "Re-bin fraction data for extended analysis. "
            "Larger bins = smoother data, better for long periods. "
            "Smaller bins = more detail, better for short periods."
        )
        rebinning_info.setWordWrap(True)
        rebinning_info.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        rebinning_layout.addWidget(rebinning_info)

        # Binning controls layout
        binning_controls = QHBoxLayout()

        binning_controls.addWidget(QLabel("Analysis Bin Size:"))

        # Decrease button
        self.btn_decrease_bin = QPushButton("−")
        self.btn_decrease_bin.setMaximumWidth(30)
        self.btn_decrease_bin.setToolTip("Decrease bin size")
        self.btn_decrease_bin.clicked.connect(
            lambda: self._adjust_analysis_bin_size(-1)
        )
        binning_controls.addWidget(self.btn_decrease_bin)

        # Bin size spinbox
        self.analysis_bin_size = QSpinBox()
        self.analysis_bin_size.setRange(60, 7200)  # 1 min to 2 h
        self.analysis_bin_size.setValue(
            1800
        )  # Default: 30 min — recommended for chi²/FFT periodogram
        self.analysis_bin_size.setSingleStep(10)
        self.analysis_bin_size.setSuffix(" sec")
        self.analysis_bin_size.setToolTip(
            "Bin size for extended analysis (1 min – 2 h).\n"
            "IMPORTANT: The chi² Z-score scales with n (number of bins).\n"
            "Smaller bins → larger n → inflated Z-scores → false positives.\n\n"
            "Recommendations by method:\n"
            "• Chi² Periodogram: 1800 s (30 min) — standard for circadian analysis\n"
            "• FFT Power Spectrum: 1800 s (30 min)\n"
            "• Cosinor: 300–1800 s (5–30 min)\n"
            "• Similarity / Coherence: 1800–3600 s (30–60 min)\n"
            "• Phase Clustering: 300–1800 s (5–30 min)\n\n"
            "Data will be automatically re-binned if different from original."
        )
        self.analysis_bin_size.setMinimumWidth(100)
        self.analysis_bin_size.valueChanged.connect(self._update_bin_size_info)
        binning_controls.addWidget(self.analysis_bin_size)

        # Increase button
        self.btn_increase_bin = QPushButton("+")
        self.btn_increase_bin.setMaximumWidth(30)
        self.btn_increase_bin.setToolTip("Increase bin size")
        self.btn_increase_bin.clicked.connect(lambda: self._adjust_analysis_bin_size(1))
        binning_controls.addWidget(self.btn_increase_bin)

        binning_controls.addWidget(QLabel("Presets:"))

        # Preset buttons
        self.btn_bin_original = QPushButton("Original")
        self.btn_bin_original.setToolTip(
            "Use original bin size from main analysis (typically 60s)"
        )
        self.btn_bin_original.clicked.connect(
            lambda: self._set_analysis_bin_preset("original")
        )
        binning_controls.addWidget(self.btn_bin_original)

        self.btn_bin_30s = QPushButton("30 sec")
        self.btn_bin_30s.setToolTip(
            "30 second bins - high resolution for short periods"
        )
        self.btn_bin_30s.clicked.connect(lambda: self._set_analysis_bin_preset(30))
        binning_controls.addWidget(self.btn_bin_30s)

        self.btn_bin_1min = QPushButton("1 min")
        self.btn_bin_1min.setToolTip("1 minute bins - standard resolution")
        self.btn_bin_1min.clicked.connect(lambda: self._set_analysis_bin_preset(60))
        binning_controls.addWidget(self.btn_bin_1min)

        self.btn_bin_5min = QPushButton("5 min")
        self.btn_bin_5min.setToolTip("5 minute bins - smooth data for long periods")
        self.btn_bin_5min.clicked.connect(lambda: self._set_analysis_bin_preset(300))
        binning_controls.addWidget(self.btn_bin_5min)

        self.btn_bin_10min = QPushButton("10 min")
        self.btn_bin_10min.setToolTip(
            "10 minute bins - very smooth data for circadian analysis"
        )
        self.btn_bin_10min.clicked.connect(lambda: self._set_analysis_bin_preset(600))
        binning_controls.addWidget(self.btn_bin_10min)

        self.btn_bin_30min = QPushButton("30 min")
        self.btn_bin_30min.setToolTip(
            "30 minute bins - reduces saturation for Cosinor analysis"
        )
        self.btn_bin_30min.clicked.connect(lambda: self._set_analysis_bin_preset(1800))
        binning_controls.addWidget(self.btn_bin_30min)

        self.btn_bin_60min = QPushButton("60 min")
        self.btn_bin_60min.setToolTip(
            "60 minute bins - recommended for Coherence and PLV analysis"
        )
        self.btn_bin_60min.clicked.connect(lambda: self._set_analysis_bin_preset(3600))
        binning_controls.addWidget(self.btn_bin_60min)

        binning_controls.addStretch()

        rebinning_layout.addLayout(binning_controls)

        # Display current vs original bin size
        self.bin_size_info_label = QLabel("")
        self.bin_size_info_label.setStyleSheet(
            "color: #27ae60; font-size: 10px; font-style: italic;"
        )
        rebinning_layout.addWidget(self.bin_size_info_label)

        # Data source selection (fraction vs raw movement)
        data_source_layout = QHBoxLayout()
        data_source_layout.addWidget(QLabel("Data Source:"))

        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "Fraction Movement (0-1)",
            "Raw Intensity (continuous)",
            "Normalized Movement (0-1)",
        ])
        self.data_source_combo.setCurrentIndex(0)  # Default: fraction movement
        self.data_source_combo.setToolTip(
            "Choose data source for extended analysis:\n\n"
            "• Fraction Movement (0-1): Proportion of time in movement state per bin.\n"
            "  Recommended for Chi², FFT, Similarity, Coherence, Phase Clustering.\n\n"
            "• Raw Intensity (continuous): Per-pixel intensity changes, MinMax normalized.\n"
            "  Use for Cosinor when a sinusoidal waveform is expected.\n\n"
            "• Normalized Movement (0-1): Re-binned & min/max normalized per ROI.\n"
            "  Comparable to Aguillon et al. 2023 'Normalized Movement (a.u.)'.\n"
            "  Best for Cosinor and cross-study comparison.\n\n"
            "Recommendation by method:\n"
            "  Chi² / FFT / Similarity / Coherence / Phase Clustering → Fraction Movement\n"
            "  Cosinor → Fraction Movement or Normalized Movement"
        )
        self.data_source_combo.currentIndexChanged.connect(self._on_data_source_changed)
        data_source_layout.addWidget(self.data_source_combo)

        data_source_layout.addStretch()

        rebinning_layout.addLayout(data_source_layout)

        layout.addWidget(rebinning_group)

        # Cycle/Period Selection for Post-Hoc Analysis
        cycle_selection_group = QGroupBox("Time Range Selection (Optional)")
        cycle_selection_layout = QFormLayout()
        cycle_selection_group.setLayout(cycle_selection_layout)

        # Enable cycle selection checkbox
        self.enable_cycle_selection = QCheckBox("Analyze specific time range only")
        self.enable_cycle_selection.setToolTip(
            "Enable this to analyze only a specific portion of your recording.\n"
            "Useful for cycle-specific analysis or comparing different time periods."
        )
        self.enable_cycle_selection.stateChanged.connect(
            self._on_cycle_selection_toggled
        )
        cycle_selection_layout.addRow("", self.enable_cycle_selection)

        # Start time selection
        self.cycle_start_time = QDoubleSpinBox()
        self.cycle_start_time.setRange(0.0, 10000.0)
        self.cycle_start_time.setValue(0.0)
        self.cycle_start_time.setSingleStep(1.0)
        self.cycle_start_time.setDecimals(1)
        self.cycle_start_time.setSuffix(" hours")
        self.cycle_start_time.setEnabled(False)
        self.cycle_start_time.setToolTip(
            "Start time of the analysis window (hours from recording start)"
        )
        cycle_selection_layout.addRow("Start Time:", self.cycle_start_time)

        # End time selection
        self.cycle_end_time = QDoubleSpinBox()
        self.cycle_end_time.setRange(0.0, 10000.0)
        self.cycle_end_time.setValue(24.0)
        self.cycle_end_time.setSingleStep(1.0)
        self.cycle_end_time.setDecimals(1)
        self.cycle_end_time.setSuffix(" hours")
        self.cycle_end_time.setEnabled(False)
        self.cycle_end_time.setToolTip(
            "End time of the analysis window (hours from recording start)"
        )
        cycle_selection_layout.addRow("End Time:", self.cycle_end_time)

        # Quick cycle selection buttons
        cycle_buttons_layout = QHBoxLayout()

        self.btn_cycle_first24 = QPushButton("First 24h")
        self.btn_cycle_first24.setToolTip("Analyze first 24 hours of recording")
        self.btn_cycle_first24.clicked.connect(lambda: self._set_cycle_range(0, 24))
        self.btn_cycle_first24.setEnabled(False)
        cycle_buttons_layout.addWidget(self.btn_cycle_first24)

        self.btn_cycle_second24 = QPushButton("Second 24h")
        self.btn_cycle_second24.setToolTip("Analyze hours 24-48 of recording")
        self.btn_cycle_second24.clicked.connect(lambda: self._set_cycle_range(24, 48))
        self.btn_cycle_second24.setEnabled(False)
        cycle_buttons_layout.addWidget(self.btn_cycle_second24)

        self.btn_cycle_last24 = QPushButton("Last 24h")
        self.btn_cycle_last24.setToolTip("Analyze last 24 hours of recording")
        self.btn_cycle_last24.clicked.connect(self._set_cycle_last_24h)
        self.btn_cycle_last24.setEnabled(False)
        cycle_buttons_layout.addWidget(self.btn_cycle_last24)

        self.btn_cycle_reset = QPushButton("Full Recording")
        self.btn_cycle_reset.setToolTip("Reset to analyze entire recording")
        self.btn_cycle_reset.clicked.connect(self._reset_cycle_range)
        self.btn_cycle_reset.setEnabled(False)
        cycle_buttons_layout.addWidget(self.btn_cycle_reset)

        cycle_selection_layout.addRow("", cycle_buttons_layout)

        layout.addWidget(cycle_selection_group)

        # Show Actogram checkbox — only visible when Cosinor is selected
        self.chk_show_actogram = QCheckBox("Show Actogram alongside result")
        self.chk_show_actogram.setToolTip(
            "After running Cosinor analysis, also open a double-plotted actogram.\n"
            "τ is taken automatically from the fitted Cosinor period.\n\n"
            "The actogram shows the time series split into τ-length rows so\n"
            "rhythm drift or masking becomes immediately visible."
        )
        self.chk_show_actogram.stateChanged.connect(
            lambda: self.actogram_settings_group.setVisible(
                self.chk_show_actogram.isChecked()
            )
        )
        self.chk_show_actogram.setVisible(False)  # shown only for Cosinor method
        layout.addWidget(self.chk_show_actogram)
        layout.addWidget(self.actogram_settings_group)  # directly below its checkbox

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Run analysis button
        self.btn_run_fisher = QPushButton("Run Rhythmic Pattern Analysis")
        self.btn_run_fisher.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 10px; } QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_run_fisher.clicked.connect(self.run_fisher_analysis)
        buttons_layout.addWidget(self.btn_run_fisher)

        # Load results from HDF5 button (for test data)
        self.btn_load_hdf5_results = QPushButton("Load Results from HDF5")
        self.btn_load_hdf5_results.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 10px; } QPushButton:hover { background-color: #0b7dda; }"
        )
        self.btn_load_hdf5_results.setToolTip(
            "Load pre-computed results directly from HDF5 file.\n"
            "Use this for test data that already contains analysis results."
        )
        self.btn_load_hdf5_results.clicked.connect(self._load_results_from_hdf5)
        buttons_layout.addWidget(self.btn_load_hdf5_results)

        # Save results to HDF5 button
        self.btn_save_hdf5_results = QPushButton("Save Results to HDF5")
        self.btn_save_hdf5_results.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; "
            "padding: 10px; } QPushButton:hover { background-color: #e68900; }"
        )
        self.btn_save_hdf5_results.setToolTip(
            "Save current analysis results to HDF5 file.\n"
            "This allows you to reload results later without re-running analysis."
        )
        self.btn_save_hdf5_results.clicked.connect(self._save_results_to_hdf5)
        buttons_layout.addWidget(self.btn_save_hdf5_results)

        layout.addLayout(buttons_layout)

        # Create a horizontal splitter for results and plot
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)

        # Results display (left side)
        self.fisher_results_text = QTextEdit()
        self.fisher_results_text.setReadOnly(True)
        self.fisher_results_text.setPlaceholderText(
            "Rhythmic pattern analysis results will appear here...\n\n"
            "Tip: Use 'Auto-Detect Range' to automatically set optimal period range for your recording duration."
        )
        splitter.addWidget(self.fisher_results_text)

        # Plot display (right side)
        self.fisher_plot_widget = QWidget()
        fisher_plot_layout = QVBoxLayout()
        self.fisher_plot_widget.setLayout(fisher_plot_layout)

        # Plot header with buttons
        plot_header_layout = QHBoxLayout()
        plot_header_layout.addStretch()

        # Button to save the current periodogram plot as image
        self.btn_save_fisher_plot = QPushButton("Save Plot...")
        self.btn_save_fisher_plot.setToolTip("Save the current plot as PNG / PDF / SVG")
        self.btn_save_fisher_plot.setMaximumWidth(110)
        self.btn_save_fisher_plot.clicked.connect(self._save_fisher_plot)
        self.btn_save_fisher_plot.setEnabled(False)
        plot_header_layout.addWidget(self.btn_save_fisher_plot)

        # Button to open plot in separate window
        self.btn_popout_plot = QPushButton("🗖 Open in Separate Window")
        self.btn_popout_plot.setToolTip("Open plot in a larger, resizable window")
        self.btn_popout_plot.setMaximumWidth(200)
        self.btn_popout_plot.clicked.connect(self._open_plot_window)
        self.btn_popout_plot.setEnabled(False)  # Enable after first plot
        plot_header_layout.addWidget(self.btn_popout_plot)

        # Button to open actogram (enabled after analysis when checkbox is checked)
        self.btn_show_actogram = QPushButton("📅 Show Actogram")
        self.btn_show_actogram.setToolTip(
            "Open the double-plotted actogram in a separate window.\n"
            "Enable 'Show Actogram alongside result' and run analysis first."
        )
        self.btn_show_actogram.setMaximumWidth(160)
        self.btn_show_actogram.clicked.connect(self._open_stored_actogram)
        self.btn_show_actogram.setEnabled(False)
        plot_header_layout.addWidget(self.btn_show_actogram)

        fisher_plot_layout.addLayout(plot_header_layout)

        self.fisher_plot_canvas = _ScaledPixmapLabel()
        self.fisher_plot_canvas.setMinimumSize(400, 300)
        self.fisher_plot_canvas.setStyleSheet(
            "border: 1px solid #ccc; background-color: white;"
        )
        fisher_plot_layout.addWidget(self.fisher_plot_canvas, 1)  # Allow expansion

        splitter.addWidget(self.fisher_plot_widget)

        # Set initial sizes (60% results, 40% plot)
        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

        # Export buttons
        self.btn_export_fisher = QPushButton("Export Current Analysis (Excel)")
        self.btn_export_fisher.clicked.connect(self.export_fisher_results)
        self.btn_export_fisher.setEnabled(False)
        layout.addWidget(self.btn_export_fisher)

        self.btn_export_all_circadian = QPushButton("Export All Analyses (Excel)")
        self.btn_export_all_circadian.clicked.connect(self.export_all_circadian_results)
        self.btn_export_all_circadian.setEnabled(False)
        layout.addWidget(self.btn_export_all_circadian)

        layout.addStretch()
    def _connect_signals(self):
        """Connect all UI signals to their respective methods."""
        # Progress signals
        self.progress_updated.connect(self._on_progress_update)
        self.status_updated.connect(self._on_status_update)
        self.performance_updated.connect(self._on_performance_update)

        # File operations
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_zarr.clicked.connect(self._load_zarr_dialog)
        self.btn_load_dir.clicked.connect(self.load_directory)
        self.btn_detect_rois.clicked.connect(self.enhanced_detect_rois)
        self.btn_clear_rois.clicked.connect(self.clear_roi_detection)

        # Plate preset checkboxes - make mutually exclusive
        self.chk_6well.stateChanged.connect(self._on_6well_preset_changed)
        self.chk_12well.stateChanged.connect(self._on_12well_preset_changed)

        # ROI scale button
        self.btn_apply_scale.clicked.connect(self._apply_roi_scale)
        self.btn_edit_rois.clicked.connect(self._open_roi_editor)
        self.btn_apply_roi_edits.clicked.connect(self._apply_roi_edits)

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
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)
        self.btn_plot.clicked.connect(self.generate_plot)
        self.btn_save_plot.clicked.connect(self.save_current_plot)
        self.btn_save_individual_rois.clicked.connect(self.save_individual_roi_plots)
        self.btn_save_all_plots.clicked.connect(self.save_all_plots)
        self.btn_save_results.clicked.connect(
            self.save_results_consolidated_complete
        )  # NEW CONSOLIDATED METHOD
        self.btn_save_with_metadata.clicked.connect(self.save_results_with_metadata)
        self.btn_apply_time_range.clicked.connect(self.apply_time_range)

        # Amplitude mode toggle
        self.show_real_amplitude.toggled.connect(self.generate_plot)
        self.show_real_amplitude.toggled.connect(self._update_fixed_signal_stats)
        self.chk_divide_by_pixels.toggled.connect(self._update_fixed_signal_stats)
        self.threshold_params_stack.currentChanged.connect(self._update_fixed_signal_stats)
        self.fixed_threshold_value.valueChanged.connect(self._preview_fixed_threshold)
        self.fixed_threshold_hysteresis.valueChanged.connect(self._preview_fixed_threshold)
        self.btn_apply_fixed_threshold.clicked.connect(self._apply_fixed_threshold)

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
        self.chk_6well.toggled.connect(self._on_6well_toggled)
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
            "Video Files (*.h5 *.hdf5 *.avi *.mp4);;HDF5 Files (*.h5 *.hdf5);;Video Files (*.avi *.mp4);;All Files (*)",
        )
        if not file_paths:
            return

        # Handle single or multiple files
        if len(file_paths) == 1:
            file_path = file_paths[0]

            # Check if single video file - load as single video (not batch)
            if file_path.lower().endswith((".avi", ".mp4")):
                self._load_single_avi(file_path)
                return
        else:
            # Multiple files - check if they are videos for batch processing
            if all(f.lower().endswith((".avi", ".mp4")) for f in file_paths):
                self._load_avi_batch(file_paths)
                return
            else:
                self._log_message(
                    "Multiple file selection only supported for AVI files"
                )
                return

        self.file_path = file_path
        self.recording_start_datetime = _parse_recording_start_datetime(file_path)
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

        # Update frame viewer file selector
        self._update_viewer_file_combo()

    def _load_single_avi(self, file_path: str):
        """Load a single AVI file (only first frame for ROI detection)."""
        try:
            from ._avi_reader import AVIVideoReader

            self.file_path = file_path
            self.recording_start_datetime = _parse_recording_start_datetime(file_path)
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

            # Update frame viewer file selector
            self._update_viewer_file_combo()

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
            self.recording_start_datetime = _parse_recording_start_datetime(file_paths[0])

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

            # Mark batch as loaded so downstream code picks the right source
            self.avi_batch_loaded = True

            # Estimate total duration by scanning all video files (reads only
            # fps + frame_count via cv2 properties — no frame decoding, very fast).
            try:
                estimated_total_seconds = 0.0
                for vp in file_paths:
                    with AVIVideoReader(vp) as _r:
                        estimated_total_seconds += _r.duration
                batch_metadata["total_duration"] = estimated_total_seconds
                layer.metadata["total_duration"] = estimated_total_seconds
                self._log_message(
                    f"Estimated total duration: {estimated_total_seconds / 60:.1f} min"
                )
            except Exception as _e:
                self._log_message(f"Duration estimation failed: {_e}")

            # Update end time / baseline spinbox now that we have the duration
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
        """Load a directory containing HDF5/AVI files, or a Zarr store directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory or Zarr Store"
        )
        if not directory:
            return

        # Check if the selected directory is itself a Zarr store
        zarr_markers = (".zgroup", ".zarray", ".zmetadata")
        if any(os.path.exists(os.path.join(directory, m)) for m in zarr_markers):
            self._load_zarr_file(directory)
            return

        self.directory = directory
        self.file_path = None
        try:
            # Scan for both HDF5 and AVI files
            h5_files = [
                f for f in os.listdir(directory) if f.lower().endswith((".h5", ".hdf5"))
            ]
            avi_files = [f for f in os.listdir(directory) if f.lower().endswith((".avi", ".mp4"))]

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
            # Update frame viewer file selector
            self._update_viewer_file_combo()
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

        # Update frame viewer file selector for directory
        self._update_viewer_file_combo()

    def _load_zarr_dialog(self):
        """Open a picker for a Zarr zip store (file) or directory store."""
        from qtpy.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("Select Zarr Store Type")
        msg.setText("Which type of Zarr store do you want to load?")
        btn_file = msg.addButton("Zip store (.zarr file)", QMessageBox.ButtonRole.AcceptRole)
        btn_dir  = msg.addButton("Directory store (.zarr folder)", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is btn_file:
            zarr_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Zarr Zip Store",
                "",
                "Zarr Files (*.zarr);;All Files (*)",
            )
            if zarr_path:
                self._load_zarr_file(zarr_path)
        elif clicked is btn_dir:
            zarr_path = QFileDialog.getExistingDirectory(
                self,
                "Select Zarr Directory Store",
                "",
            )
            if zarr_path:
                zarr_markers = (".zgroup", ".zarray", ".zmetadata")
                if not any(os.path.exists(os.path.join(zarr_path, m)) for m in zarr_markers):
                    self._log_message(
                        f"Selected directory does not appear to be a Zarr store: {zarr_path}"
                    )
                    return
                self._load_zarr_file(zarr_path)

    def _load_zarr_file(self, zarr_path: str):
        """Load a Zarr store directory as an activity recording."""
        self.file_path = zarr_path
        self.recording_start_datetime = _parse_recording_start_datetime(zarr_path)
        basename = os.path.basename(zarr_path.rstrip("/\\"))

        self._log_message(f"Loading Zarr store: {basename}")
        self.lbl_file_info.setText(f"Loaded Zarr store: {basename}")

        self.masks = []

        # Structure detection (format-agnostic)
        if DUAL_STRUCTURE_AVAILABLE:
            try:
                structure_info = detect_file_structure_type(zarr_path)
                fmt = structure_info.get("file_format", "zarr")
                self._log_message(
                    f"Detected {fmt} structure: {structure_info['type']} "
                    f"({structure_info.get('frame_count', '?')} frames, "
                    f"shape {structure_info.get('frame_shape', '?')})"
                )
                if structure_info["type"] in ("error", "unknown"):
                    self._log_message(
                        f"Structure detection failed: {structure_info.get('error', 'unknown structure')}"
                    )
                    return
                reader = reader_function_dual_structure
            except Exception as e:
                self._log_message(f"Structure detection error: {e}")
                return
        else:
            self._log_message("Dual structure reader not available — cannot load Zarr.")
            return

        try:
            self.viewer.layers.clear()
            layers = reader(zarr_path)
            for data, meta, layer_type in layers:
                name = meta.get("name", basename)
                kwargs = {k: v for k, v in meta.items() if k != "name"}
                if layer_type == "image":
                    self.viewer.add_image(data, name=name, **kwargs)
                elif layer_type == "labels":
                    self.viewer.add_labels(data, name=name, **kwargs)
        except Exception as e:
            self._log_message(f"Zarr reader error: {e}")
            return

        self.update_end_time()
        self.check_hdf5_structure()
        self._update_viewer_file_combo()

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
                elif self.file_path.lower().endswith((".avi", ".mp4")):
                    # Single video file
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
                    # HDF5 or Zarr file
                    if DUAL_STRUCTURE_AVAILABLE:
                        # Use format-agnostic detection (handles both HDF5 and Zarr)
                        structure_info = detect_file_structure_type(self.file_path)
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

                # Initialize baseline duration spinbox to recording length at load time
                if hasattr(self, "baseline_duration_minutes") and total_duration_minutes > 0:
                    self.baseline_duration_minutes.setMaximum(total_duration_minutes)
                    self.baseline_duration_minutes.setValue(total_duration_minutes)

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
            self.lbl_file_info.setText("Error: No file loaded for ROI detection")
            return

        # Get ROI detection parameters based on preset or manual values
        # Always use current spinbox values (presets just set the spinbox values)
        params = {
            "min_radius": self.min_radius.value(),
            "max_radius": self.max_radius.value(),
            "dp": self.dp_param.value(),
            "min_dist": self.min_dist.value(),
            "param1": self.param1.value(),
            "param2": self.param2.value(),
        }

        # Log which preset is active (if any)
        if self.chk_6well.isChecked():
            self._log_message("6-well preset active (values can be adjusted)")
        elif self.chk_12well.isChecked():
            self._log_message("12-well preset active (values can be adjusted)")

        # Log the actual parameters being used
        self._log_message(
            f"ROI Detection: radius={params['min_radius']}-{params['max_radius']}px, "
            f"dp={params['dp']}, minDist={params['min_dist']}, "
            f"param1={params['param1']}, param2={params['param2']}"
        )

        try:
            # ROI detection - get first frame from viewer layer or file
            first_frame = None

            # Check if current file is HDF5/Zarr or AVI to decide source
            is_avi = current_file.lower().endswith((".avi", ".mp4"))
            is_file_based = not is_avi and not (
                hasattr(self, "avi_batch_loaded") and self.avi_batch_loaded
            )

            # For HDF5/Zarr files, always read from file using format-agnostic reader
            # For AVI batch, try to use existing viewer layer first
            if is_file_based:
                self._log_message(
                    "Reading first frame from file..."
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

            # Ensure uint8 for CLAHE (required by OpenCV)
            if gray_frame.dtype != np.uint8:
                # Normalize to 0-255 range and convert to uint8
                gray_float = gray_frame.astype(np.float32)
                gray_min, gray_max = gray_float.min(), gray_float.max()
                if gray_max > gray_min:
                    gray_norm = (gray_float - gray_min) / (gray_max - gray_min)
                else:
                    gray_norm = gray_float
                gray_frame = (gray_norm * 255).astype(np.uint8)
                self._log_message(
                    f"Converted frame from {first_frame.dtype} to uint8 for CLAHE"
                )

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

                # Store original circles for later scaling
                self._original_circles = sorted_circles.copy()
                self._original_frame_shape = gray_frame.shape
                self._original_first_frame = first_frame.copy()
                self.btn_apply_scale.setEnabled(True)

                # Apply current scale factor
                scale = self.roi_scale.value()
                scaled_circles = sorted_circles.copy().astype(np.float32)
                scaled_circles[:, 2] = scaled_circles[:, 2] * scale  # Scale radius only
                scaled_circles = np.round(scaled_circles).astype(np.uint16)

                for idx, circle in enumerate(scaled_circles):
                    mask = np.zeros(gray_frame.shape, dtype=np.uint8)
                    cv2.circle(
                        mask, (circle[0], circle[1]), circle[2], 255, thickness=-1
                    )
                    masks.append(mask)

                # Create labeled frame with scaled circles.
                # Always build from the uint8 normalized gray_frame so that the
                # drawn circle colors (0-255 range) are visible regardless of
                # whether the source file is HDF5 (possibly uint16) or Zarr.
                labeled_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

                for idx, circle in enumerate(scaled_circles):
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
            self._log_message("💡 Use 'Edit ROI Circles' to manually adjust positions in the viewer")

            if dataset_type == "MAIN":
                self.btn_edit_rois.setEnabled(True)
                self.btn_apply_scale.setEnabled(True)

            # Persist the parameters that led to a successful detection
            self._save_roi_settings()

            # Populate ROI selection checkboxes
            if dataset_type == "MAIN":
                self._populate_roi_checkboxes(len(masks))

        except Exception as e:
            self._log_message(f"ERROR in ROI detection: {e}")

    # ------------------------------------------------------------------
    # ROI detection settings persistence (QSettings → Windows registry)
    # ------------------------------------------------------------------
    _ROI_SETTINGS_ORG = "napari-hdf5-activity"
    _ROI_SETTINGS_APP = "HDF5AnalysisWidget"

    def _save_roi_settings(self):
        """Persist current ROI detection parameters via QSettings."""
        s = QSettings(self._ROI_SETTINGS_ORG, self._ROI_SETTINGS_APP)
        s.setValue("roi/min_radius", self.min_radius.value())
        s.setValue("roi/max_radius", self.max_radius.value())
        s.setValue("roi/dp_param", self.dp_param.value())
        s.setValue("roi/min_dist", self.min_dist.value())
        s.setValue("roi/param1", self.param1.value())
        s.setValue("roi/param2", self.param2.value())
        s.setValue("roi/roi_scale", self.roi_scale.value())
        preset = "6well" if self.chk_6well.isChecked() else (
            "12well" if self.chk_12well.isChecked() else "none"
        )
        s.setValue("roi/preset", preset)
        s.sync()

    def _load_roi_settings(self):
        """Restore ROI detection parameters saved by a previous session."""
        s = QSettings(self._ROI_SETTINGS_ORG, self._ROI_SETTINGS_APP)
        if not s.contains("roi/min_radius"):
            return  # No saved settings yet → keep widget defaults
        self.min_radius.setValue(int(s.value("roi/min_radius", 100)))
        self.max_radius.setValue(int(s.value("roi/max_radius", 145)))
        self.dp_param.setValue(float(s.value("roi/dp_param", 1.0)))
        self.min_dist.setValue(int(s.value("roi/min_dist", 300)))
        self.param1.setValue(int(s.value("roi/param1", 30)))
        self.param2.setValue(int(s.value("roi/param2", 60)))
        self.roi_scale.setValue(float(s.value("roi/roi_scale", 1.0)))
        preset = s.value("roi/preset", "6well")
        self.chk_6well.setChecked(preset == "6well")
        self.chk_12well.setChecked(preset == "12well")

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

    def _on_6well_preset_changed(self, state):
        """Handle 6-well preset checkbox change - make mutually exclusive."""
        if state == 2:  # Checked
            self.chk_12well.blockSignals(True)
            self.chk_12well.setChecked(False)
            self.chk_12well.blockSignals(False)
            # Update spinbox values to show preset (tested optimal for 6-well)
            self.min_radius.setValue(100)
            self.max_radius.setValue(145)
            self.dp_param.setValue(1.0)
            self.min_dist.setValue(300)
            self.param1.setValue(30)
            self.param2.setValue(60)

    def _on_12well_preset_changed(self, state):
        """Handle 12-well preset checkbox change - make mutually exclusive."""
        if state == 2:  # Checked
            self.chk_6well.blockSignals(True)
            self.chk_6well.setChecked(False)
            self.chk_6well.blockSignals(False)
            # Update spinbox values to show preset
            self.min_radius.setValue(70)
            self.max_radius.setValue(120)
            self.dp_param.setValue(1.0)
            self.min_dist.setValue(150)
            self.param1.setValue(50)
            self.param2.setValue(25)

    def _apply_roi_scale(self):
        """Apply scale factor to detected ROIs without re-detecting."""
        if not hasattr(self, "_original_circles") or self._original_circles is None:
            self._log_message("ERROR: No ROIs detected yet. Run detection first.")
            return

        try:
            scale = self.roi_scale.value()
            self._log_message(f"Applying ROI scale: {scale:.2f}")

            # Get original data
            original_circles = self._original_circles
            frame_shape = self._original_frame_shape
            first_frame = self._original_first_frame

            # Scale the radii (keep centers unchanged)
            scaled_circles = original_circles.copy().astype(np.float32)
            scaled_circles[:, 2] = scaled_circles[:, 2] * scale
            scaled_circles = np.round(scaled_circles).astype(np.uint16)

            # Recreate masks with scaled radii
            masks = []
            for circle in scaled_circles:
                mask = np.zeros(frame_shape, dtype=np.uint8)
                cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)
                masks.append(mask)

            # Recreate labeled frame
            if len(first_frame.shape) == 3:
                labeled_frame = first_frame.copy()
            else:
                labeled_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB)

            current_type = getattr(self, "current_dataset_type", "main")
            dataset_type = "CALIBRATION" if current_type == "calibration" else "MAIN"

            for idx, circle in enumerate(scaled_circles):
                color = (255, 165, 0) if dataset_type == "CALIBRATION" else (0, 255, 0)
                cv2.circle(labeled_frame, (circle[0], circle[1]), circle[2], color, 2)
                cv2.putText(
                    labeled_frame,
                    f"{idx + 1}",
                    (circle[0] - 10, circle[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (255, 0, 0),
                    3,
                )

            # Update stored masks
            if dataset_type == "CALIBRATION":
                self.calibration_masks = masks.copy()
                self.calibration_labeled_frame = labeled_frame.copy()
                self.masks = masks
                self.labeled_frame = labeled_frame
            else:
                self.main_masks = masks.copy()
                self.main_labeled_frame = labeled_frame.copy()
                self.masks = masks
                self.labeled_frame = labeled_frame

            # Update viewer layer
            self._add_roi_layers_to_viewer(labeled_frame, masks, dataset_type)

            avg_radius = np.mean(scaled_circles[:, 2])
            self._log_message(
                f"Scaled {len(masks)} ROIs to {scale:.0%} (avg radius: {avg_radius:.0f}px)"
            )
            self.lbl_file_info.setText(
                f"{dataset_type}: {len(masks)} ROIs (scale: {scale:.0%})"
            )

        except Exception as e:
            self._log_message(f"ERROR applying ROI scale: {e}")

    def _open_roi_editor(self):
        """Add a draggable Points layer (one point per circle centre) for manual repositioning."""
        if not hasattr(self, "_original_circles") or self._original_circles is None:
            self._log_message("⚠️ No ROIs detected yet — run detection first.")
            return

        try:
            circles = self._original_circles
            # Points layer uses (row, col) = (y, x) order
            centres = np.array([[float(c[1]), float(c[0])] for c in circles])

            # Remove any previous edit layer
            for layer in list(self.viewer.layers):
                if getattr(layer, "name", "").startswith("ROI Centres (edit)"):
                    self.viewer.layers.remove(layer)

            # Ensure the raw first-frame image is visible as background reference
            for layer in self.viewer.layers:
                name = getattr(layer, "name", "")
                if "first_frame" in name or "frame" in name.lower():
                    if "ROI" not in name and "edit" not in name:
                        layer.visible = True

            # Hide ROI annotation layer(s) to avoid double-circle overlap
            for layer in self.viewer.layers:
                name = getattr(layer, "name", "")
                if "ROI" in name and "edit" not in name and hasattr(layer, "visible"):
                    layer.visible = False

            scale = self.roi_scale.value()
            avg_r = float(np.mean(circles[:, 2])) * scale

            self._roi_edit_layer = self.viewer.add_points(
                centres,
                name="ROI Centres (edit)",
                size=avg_r * 2,
                symbol="ring",
                face_color="transparent",
                border_color="lime",
                border_width=0.05,
            )
            self._roi_edit_layer.visible = True

            # Select the edit layer so the user can immediately drag points
            try:
                self.viewer.layers.selection.active = self._roi_edit_layer
            except Exception:
                pass

            # Set select mode so individual points can be dragged
            try:
                self._roi_edit_layer.mode = "select"
            except Exception:
                pass

            self.btn_apply_roi_edits.setEnabled(True)
            self._log_message(
                f"✏️ {len(centres)} ROI centres loaded into edit layer. "
                "Press S (select mode) then drag individual points. "
                "Click 'Apply Edits' when done."
            )

        except Exception as e:
            self._log_message(f"ERROR opening ROI editor: {e}")
            import traceback; traceback.print_exc()

    def _apply_roi_edits(self):
        """Read the moved point positions and rebuild masks with original radii."""
        if not hasattr(self, "_roi_edit_layer"):
            self._log_message("⚠️ No edit layer — click 'Edit ROI Circles' first.")
            return

        try:
            new_centres = self._roi_edit_layer.data  # shape (N, 2) in (row, col) order
            if new_centres is None or len(new_centres) == 0:
                self._log_message("⚠️ No points in editor layer.")
                return

            original_radii = self._original_circles[:, 2].astype(int)
            scale = self.roi_scale.value()
            frame_shape = self._original_frame_shape

            # Build new circles array: col=cx, row=cy, keep original radius × scale
            new_circles = np.array([
                [int(round(pt[1])), int(round(pt[0])),
                 int(round(original_radii[i] * scale))]
                for i, pt in enumerate(new_centres)
            ], dtype=np.uint16)

            # Update stored circles and reset scale (baked in)
            self._original_circles = new_circles
            self.roi_scale.setValue(1.0)

            # Recreate masks
            masks = []
            for c in new_circles:
                mask = np.zeros(frame_shape, dtype=np.uint8)
                cv2.circle(mask, (int(c[0]), int(c[1])), int(c[2]), 255, thickness=-1)
                masks.append(mask)

            # Recreate labeled frame
            first_frame = self._original_first_frame
            labeled_frame = (first_frame.copy() if len(first_frame.shape) == 3
                             else cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB))
            for idx, c in enumerate(new_circles):
                cv2.circle(labeled_frame, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 2)
                cv2.putText(labeled_frame, str(idx + 1),
                            (int(c[0]) - 10, int(c[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 3)

            self.masks = masks
            self.main_masks = masks.copy()
            self.labeled_frame = labeled_frame
            self.main_labeled_frame = labeled_frame.copy()

            self.viewer.layers.remove(self._roi_edit_layer)
            del self._roi_edit_layer
            self.btn_apply_roi_edits.setEnabled(False)

            self._add_roi_layers_to_viewer(labeled_frame, masks, "MAIN")
            self._populate_roi_checkboxes(len(masks))

            self._log_message(f"✅ Applied {len(masks)} repositioned ROIs.")
            self.lbl_file_info.setText(f"MAIN: {len(masks)} ROIs (manually adjusted)")

        except Exception as e:
            self._log_message(f"ERROR applying ROI edits: {e}")
            import traceback; traceback.print_exc()

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
        # Check if analysis is already running
        if hasattr(self, "current_worker") and self.current_worker is not None:
            self._log_message(
                "⚠️ Analysis already running! Please wait or stop current analysis first."
            )
            self.status_label.setText("Analysis already running!")
            return

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

        # Stop any previously running worker before starting new one
        if hasattr(self, "current_worker") and self.current_worker is not None:
            self._cancel_requested = True
            try:
                self.current_worker.returned.disconnect()
                self.current_worker.errored.disconnect()
                self.current_worker.finished.disconnect()
            except Exception:
                pass
            self.current_worker = None

        # Increment generation so any still-running old worker's callbacks are ignored
        self._analysis_generation += 1
        my_generation = self._analysis_generation

        # UI state management
        self._cancel_requested = False
        self.analysis_start_time = time.time()
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_plot.setEnabled(False)  # Disable until analysis fully completes
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing analysis...")
        self.performance_timer.start()

        # Log analysis start
        self._log_analysis_parameters()

        @thread_worker(start_thread=False)
        def _analysis_worker():
            return self._run_analysis_with_calc_module()

        def _on_finished(result):
            if self._analysis_generation == my_generation:
                self._analysis_finished(result)

        def _on_errored(exc):
            if self._analysis_generation == my_generation:
                self._analysis_errored(exc)

        def _on_done():
            if self._analysis_generation == my_generation:
                self._analysis_done()

        worker_instance = _analysis_worker()
        worker_instance.returned.connect(_on_finished)
        worker_instance.errored.connect(_on_errored)
        worker_instance.finished.connect(_on_done)
        worker_instance.start()
        self.current_worker = worker_instance

    def load_calibration_file(self):
        """Enhanced calibration file loading with workflow state management."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Calibration File",
            "",
            "Video Files (*.h5 *.hdf5 *.avi *.mp4);;HDF5 Files (*.h5 *.hdf5);;Video Files (*.avi *.mp4);;All Files (*.*)",
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

        # Preprocessing params shared by all threshold methods
        jump_correction = self.enable_jump_correction.isChecked()
        aib = self.adaptive_illumination_baseline.isChecked()
        led_data = None
        if aib:
            led_data = self._extract_led_data_from_hdf5()
            if led_data is None:
                self._log_message(
                    "⚠️ Adaptive Illumination Baseline: checkbox is ON but no white LED "
                    "channel found in file (need 'led_white_power_percent' or similar). "
                    "Adaptive baseline will NOT be applied."
                )
            else:
                self._log_message(
                    f"✓ Adaptive Illumination Baseline: LED data extracted "
                    f"({len(led_data.get('times', []))} time points)."
                )
        params.update(
            {
                "enable_jump_correction": jump_correction,
                "frame_mean_data": self._extract_frame_mean_from_hdf5() if jump_correction else None,
                "adaptive_illumination_baseline": aib,
                "led_data": led_data,
            }
        )

        # Add method-specific parameters
        if threshold_method == "baseline":
            params.update(
                {
                    "baseline_duration_minutes": self.baseline_duration_minutes.value(),
                    "multiplier": self.threshold_multiplier.value(),
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
        elif threshold_method == "fixed":
            params.update(
                {
                    "fixed_threshold_value": self.fixed_threshold_value.value(),
                    "fixed_threshold_hysteresis": self.fixed_threshold_hysteresis.value(),
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
                masks_to_use = self._get_active_masks()

            # Log excluded ROIs
            excluded = self._get_excluded_roi_indices()
            if excluded:
                excluded_names = [f"ROI {i+1}" for i in excluded]
                self._log_message(f"Excluding {len(excluded)} ROI(s): {', '.join(excluded_names)}")

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
                # Recalculate safe process count from current available RAM
                try:
                    available_mb = psutil.virtual_memory().available / (1024 * 1024)
                    usable_mb = max(0, available_mb - 1024)  # keep 1 GB headroom
                    safe_processes = max(1, min(self.num_processes.value(), int(usable_mb / 800)))
                    if safe_processes < self.num_processes.value():
                        self._log_message(
                            f"⚠️ Low RAM ({available_mb:.0f} MB free) — reducing workers "
                            f"from {self.num_processes.value()} to {safe_processes}"
                        )
                except Exception:
                    safe_processes = self.num_processes.value()

                # Process complete dataset using reader (HDF5)
                _, merged_results, _ = process_single_file_in_parallel_dual_structure(
                    file_to_process,
                    masks_to_use,
                    self.chunk_size.value(),
                    progress_callback,
                    self.frame_interval.value(),
                    safe_processes,
                )

            # Remap ROI indices to preserve original numbering when ROIs are excluded
            roi_mapping = self._get_roi_index_mapping()
            if any(k != v for k, v in roi_mapping.items()):
                merged_results = {roi_mapping.get(k, k): v for k, v in merged_results.items()}
                self._log_message(f"ROI index mapping applied: {roi_mapping}")

            # Store raw data as fallback (will be overwritten with normalized
            # data in _analysis_finished if the calc step succeeds)
            self.merged_results = merged_results

            # Signal that reader is done but calc is still running
            self.status_updated.emit("Computing thresholds & movement detection...")
            self.progress_updated.emit(95)  # Not 100% yet - calc step still pending

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
        # Bump generation so any pending callbacks from the old worker are ignored
        self._analysis_generation += 1
        self._log_message("STOP requested by user")

        # Stop performance monitoring
        self.performance_timer.stop()

        # Ask the worker thread to exit as soon as possible
        if hasattr(self, "current_worker") and self.current_worker is not None:
            try:
                self.current_worker.quit()
            except Exception:
                pass
            self.current_worker = None

        # Reset UI state fully so a new analysis can start cleanly
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis stopped. Ready.")
        self.analysis_start_time = None

    def _auto_adjust_time_range(self):
        """Set plot_start_time / plot_end_time from the actual recording duration."""
        data = getattr(self, "merged_results", None) or getattr(self, "movement_data", None)
        if not data:
            return

        max_time_seconds = 0.0
        for roi_data in data.values():
            if roi_data:
                times = [t for t, _ in roi_data]
                if times:
                    max_time_seconds = max(max_time_seconds, max(times))

        max_time_minutes = max_time_seconds / 60.0
        if max_time_minutes <= 0:
            return

        if hasattr(self, "plot_end_time"):
            self.plot_end_time.setRange(0.0, max_time_minutes * 1.05)
            self.plot_end_time.setValue(max_time_minutes)
            self.plot_start_time.setRange(0.0, max_time_minutes * 1.05)
            self.plot_start_time.setValue(0.0)
            self._log_message(
                f"Plot time range set: 0 – {max_time_minutes:.1f} min "
                f"({max_time_minutes / 60:.2f} h)"
            )

        # Cap baseline duration spinbox to recording length
        if hasattr(self, "baseline_duration_minutes"):
            self.baseline_duration_minutes.setMaximum(max_time_minutes)
            if self.baseline_duration_minutes.value() > max_time_minutes:
                self.baseline_duration_minutes.setValue(max_time_minutes)

        # Update Extended Analysis time range (hours axis)
        max_time_hours = max_time_seconds / 3600.0
        if hasattr(self, "cycle_end_time") and max_time_hours > 0:
            self.cycle_end_time.setRange(0.0, max_time_hours * 1.05)
            self.cycle_end_time.setValue(max_time_hours)
            self.cycle_start_time.setRange(0.0, max_time_hours * 1.05)
            self.cycle_start_time.setValue(0.0)
            self._log_message(
                f"Extended Analysis time range set: 0 – {max_time_hours:.2f} h"
            )

    def _analysis_finished(self, result: Dict[str, Any]):
        """Handle successful analysis completion using results from _calc.py."""
        try:
            # Capture start time immediately before it can be cleared by _analysis_done()
            start_time = self.analysis_start_time

            # Store normalized [0,1] data (default for display)
            self.merged_results = result.get("processed_data", {})
            self.roi_baseline_means = result.get("baseline_means", {})
            self.roi_upper_thresholds = result.get("upper_thresholds", {})
            self.roi_lower_thresholds = result.get("lower_thresholds", {})
            # Store raw amplitude data (pre-MinMax, for real amplitude view)
            self.merged_results_raw = result.get("processed_data_raw", {})
            self.roi_baseline_means_raw = result.get("baseline_means_raw", {})
            self.roi_upper_thresholds_raw = result.get("upper_thresholds_raw", {})
            self.roi_lower_thresholds_raw = result.get("lower_thresholds_raw", {})
            self.roi_statistics = result.get("roi_statistics", {})
            self.movement_data = result.get("movement_data", {})
            self.fraction_data = result.get("fraction_data", {})
            self.sleep_data = result.get("sleep_data", {})
            # Cache LED data from HDF5 for lighting overlay in plots
            self.led_data = self._extract_led_data_from_hdf5() or {}
            # Update Real Amplitude checkbox availability now that raw data is loaded
            self._update_real_amplitude_controls()
            # Pixel count per ROI for sum-mode amplitude display (sum = mean × n_pixels)
            masks = getattr(self, "masks", [])
            self.roi_pixel_counts = {
                i + 1: int(np.sum(m > 0)) for i, m in enumerate(masks)
            }
            # Normalization factor: 255 for uint8, 65535 for uint16 cameras
            # Used to convert back to raw pixel units (MATLAB-equivalent values)
            if DUAL_STRUCTURE_AVAILABLE and getattr(self, "file_path", None):
                self.frame_norm_factor = get_frame_norm_factor(self.file_path)
            else:
                self.frame_norm_factor = 1.0
            self._log_message(f"  Frame normalization factor: {self.frame_norm_factor:.0f} (bit depth)")
            self._update_fixed_signal_stats()
            if hasattr(self, "btn_apply_fixed_threshold"):
                self.btn_apply_fixed_threshold.setEnabled(bool(self.merged_results_raw))

            # Log inactive ROIs (MinMax normalization is now applied in _calc.py)
            roi_active = result.get("roi_active", {})
            inactive_rois = [roi for roi, active in roi_active.items() if not active]
            if inactive_rois:
                self._log_message(f"⚠️ {len(inactive_rois)} inactive ROI(s) detected (low amplitude): {inactive_rois}")
            self.roi_active = roi_active

            # Get ROI colors from calc results
            self.roi_colors = result.get("roi_colors", {})

            # Fallback if no ROI colors provided
            if not self.roi_colors and self.merged_results:
                self.roi_colors = {
                    roi: f"C{(roi - 1) % 10}"
                    for roi in sorted(self.merged_results.keys())
                }

            self._log_message(f"✅ ROI colors set: {self.roi_colors}")

            # Calculate quiescence data using _calc.py functions
            if self.fraction_data:
                self.quiescence_data = bin_quiescence(
                    self.fraction_data, self.quiescence_threshold.value()
                )

            # Calculate sleep quality metrics (MATLAB-compatible hourly analysis)
            if self.sleep_data:
                from ._calc import calculate_sleep_quality_hourly
                self.sleep_quality_data = calculate_sleep_quality_hourly(
                    self.sleep_data
                )
                self._log_message("✅ Sleep quality metrics calculated (min/h, transitions/h, bout length/h)")

            # Calculate band widths for plotting compatibility (normalized)
            self.roi_band_widths = {}
            for roi in self.roi_baseline_means:
                if (
                    roi in self.roi_upper_thresholds
                    and roi in self.roi_lower_thresholds
                ):
                    upper = self.roi_upper_thresholds[roi]
                    lower = self.roi_lower_thresholds[roi]
                    self.roi_band_widths[roi] = (upper - lower) / 2
            # Also for raw amplitude mode
            self.roi_band_widths_raw = {}
            for roi in self.roi_baseline_means_raw:
                if (
                    roi in self.roi_upper_thresholds_raw
                    and roi in self.roi_lower_thresholds_raw
                ):
                    upper = self.roi_upper_thresholds_raw[roi]
                    lower = self.roi_lower_thresholds_raw[roi]
                    self.roi_band_widths_raw[roi] = (upper - lower) / 2

            # Update plot time range based on actual data duration
            self._auto_adjust_time_range()

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

            # Re-enable Generate Plot button and set progress to 100%
            self.btn_plot.setEnabled(True)
            self.progress_bar.setValue(100)

            # Auto-generate plot so user doesn't have to click manually
            self.generate_plot()

        except Exception as e:
            self._log_message(f"Error in analysis completion: {str(e)}")
            import traceback

            self._log_message(f"Full traceback: {traceback.format_exc()}")
        finally:
            # Always re-enable Generate Plot button
            self.btn_plot.setEnabled(True)
            self.progress_bar.setValue(100)

    def _analysis_errored(self, exc):
        """Handle analysis errors."""
        self.performance_timer.stop()
        import traceback
        self._log_message(f"ANALYSIS ERROR: {exc}")
        self._log_message(f"Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
        if "canceled" in str(exc).lower():
            self.status_label.setText("Analysis canceled by user.")
            self._log_message("Analysis CANCELED by user")
        else:
            self.status_label.setText(f"Analysis error: {exc}")
            self._log_message(f"ERROR: {exc}")

        # Reset UI state
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_plot.setEnabled(True)  # Re-enable even on error
        self.progress_bar.setValue(0)

    def _analysis_done(self):
        """Cleanup after analysis completion or cancellation."""
        self.performance_timer.stop()
        self._cancel_requested = False
        self.analysis_start_time = None
        self.current_worker = None  # Clear worker reference

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
        """Validate recording timing using _calc.py module (HDF5 and Zarr)."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("No data available for timing validation")
            return

        # Detect file format for the header label
        fmt_label = "HDF5"
        if hasattr(self, "file_path") and self.file_path:
            try:
                from ._io_abstraction import detect_file_format
                fmt_label = detect_file_format(self.file_path).upper()
            except Exception:
                pass

        try:
            timing_diagnostics = validate_hdf5_timing_in_data(
                self.merged_results, self.frame_interval.value()
            )

            self._log_message(f"=== {fmt_label} TIMING DIAGNOSTICS ===")
            self._log_message(f"Timing type     : {timing_diagnostics['timing_type']}")
            self._log_message(
                f"First timestamp : {timing_diagnostics['first_time']:.1f}s"
            )
            self._log_message(
                f"Average interval: {timing_diagnostics['avg_interval']:.1f}s"
            )
            self._log_message(
                f"Expected interval: {timing_diagnostics['expected_interval']:.1f}s"
            )
            self._log_message(
                f"Interval quality: {timing_diagnostics.get('timing_quality', 'n/a')}"
            )
            self._log_message(
                f"Consistent      : {timing_diagnostics['interval_consistent']}"
            )
            self._log_message(
                f"Needs correction: {timing_diagnostics.get('needs_hdf5_correction', timing_diagnostics.get('needs_correction', 'n/a'))}"
            )
            self._log_message(
                f"Recommendation  : {timing_diagnostics['recommended_action']}"
            )

        except Exception as e:
            self._log_message(f"Timing validation failed: {e}")

    # ===================================================================
    # PLOTTING METHODS - NOW USING _plot.py MODULE
    # ===================================================================

    def _on_plot_bin_changed(self):
        """Refresh plot when plot bin size changes."""
        if hasattr(self, "merged_results") and self.merged_results:
            self.generate_plot()

    def _on_plot_type_changed(self):
        """Handle plot type dropdown change - show/hide sub-selectors."""
        plot_type = self.plot_type_combo.currentText()
        # Show sleep quality metric selector only for Sleep Quality
        if hasattr(self, "sleep_quality_metric_combo"):
            self.sleep_quality_metric_combo.setVisible(plot_type == "Sleep Quality")
        # Real Amplitude checkboxes only apply to Raw Intensity Changes
        self._update_real_amplitude_controls()
        # Generate plot
        self.generate_plot()

    def _preview_fixed_threshold(self, *_):
        """Update threshold lines on the Raw Intensity plot instantly — no recomputation."""
        if not getattr(self, "merged_results", {}):
            return
        upper = self.fixed_threshold_value.value()
        lower = upper * self.fixed_threshold_hysteresis.value()
        mid = (upper + lower) / 2.0
        self.roi_upper_thresholds = {roi: upper for roi in self.merged_results}
        self.roi_lower_thresholds = {roi: lower for roi in self.merged_results}
        self.roi_baseline_means = {roi: mid for roi in self.merged_results}
        self.roi_band_widths = {roi: (upper - lower) / 2.0 for roi in self.merged_results}
        # Same for raw amplitude view
        self.roi_upper_thresholds_raw = dict(self.roi_upper_thresholds)
        self.roi_lower_thresholds_raw = dict(self.roi_lower_thresholds)
        self.roi_baseline_means_raw = dict(self.roi_baseline_means)
        self.roi_band_widths_raw = dict(self.roi_band_widths)
        if (
            hasattr(self, "plot_type_combo")
            and self.plot_type_combo.currentText() == "Raw Intensity Changes"
            and self.threshold_params_stack.tabText(
                self.threshold_params_stack.currentIndex()
            ) == "Fixed Threshold"
        ):
            self.generate_plot()

    def _apply_fixed_threshold(self):
        """Re-run only Phase 2 (movement detection → sleep) with the current fixed threshold.
        Phase 1 (preprocessing, MinMax) is already cached in merged_results_raw."""
        import numpy as np
        from ._calc import (
            define_movement_with_hysteresis,
            bin_fraction_movement,
            bin_quiescence,
            define_sleep_periods,
            calculate_sleep_quality_hourly,
        )

        raw = getattr(self, "merged_results_raw", {})
        if not raw:
            self._log_message("⚠️ No preprocessed data cached — run a full analysis first.")
            return

        self._log_message("Applying fixed threshold (Phase 2 only)…")
        upper_norm = self.fixed_threshold_value.value()
        ratio = self.fixed_threshold_hysteresis.value()
        lower_norm = upper_norm * ratio

        # De-normalise per ROI so movement detection is in pre-MinMax signal space
        upper_thresholds, lower_thresholds, baseline_means = {}, {}, {}
        norm_upper, norm_lower, norm_baseline = {}, {}, {}
        for roi, pts in raw.items():
            if not pts:
                continue
            vals = np.array([v for _, v in pts])
            min_v, max_v = float(np.min(vals)), float(np.max(vals))
            rng = max_v - min_v
            if rng > 0:
                u = upper_norm * rng + min_v
                l = lower_norm * rng + min_v
            else:
                u, l = upper_norm, lower_norm
            upper_thresholds[roi] = u
            lower_thresholds[roi] = l
            baseline_means[roi] = (u + l) / 2.0
            # Normalized display values are identical for all ROIs
            norm_upper[roi] = upper_norm
            norm_lower[roi] = lower_norm
            norm_baseline[roi] = (upper_norm + lower_norm) / 2.0

        frame_interval = getattr(self, "frame_interval_seconds", 5.0)
        bin_size_s = self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60

        movement_data = define_movement_with_hysteresis(raw, baseline_means, upper_thresholds, lower_thresholds)
        fraction_data = bin_fraction_movement(movement_data, bin_size_s, frame_interval)
        quiescence_threshold = 0.5
        quiescence_data = bin_quiescence(fraction_data, quiescence_threshold)

        sleep_threshold_minutes = self.sleep_threshold_minutes.value() if hasattr(self, "sleep_threshold_minutes") else 8
        sleep_data = define_sleep_periods(quiescence_data, sleep_threshold_minutes, bin_size_s)

        sleep_quality_data = calculate_sleep_quality_hourly(sleep_data, data_bin_seconds=bin_size_s)

        # Update widget state
        self.movement_data = movement_data
        self.fraction_data = fraction_data
        self.quiescence_data = quiescence_data
        self.sleep_data = sleep_data
        self.sleep_quality_data = sleep_quality_data
        self.roi_upper_thresholds = norm_upper
        self.roi_lower_thresholds = norm_lower
        self.roi_baseline_means = norm_baseline
        self.roi_band_widths = {roi: (upper_norm - lower_norm) / 2.0 for roi in norm_upper}
        self.roi_upper_thresholds_raw = dict(norm_upper)
        self.roi_lower_thresholds_raw = dict(norm_lower)
        self.roi_baseline_means_raw = dict(norm_baseline)
        self.roi_band_widths_raw = dict(self.roi_band_widths)

        self._log_message(f"Fixed threshold applied: upper={upper_norm:.4f}, lower={lower_norm:.4f}")
        self.generate_plot()

    def _update_fixed_signal_stats(self, *_):
        """Update the signal range label in the Fixed Threshold tab."""
        if not hasattr(self, "fixed_signal_stats_label"):
            return
        import numpy as np

        use_real = (
            hasattr(self, "show_real_amplitude") and self.show_real_amplitude.isChecked()
        )
        divide = (
            hasattr(self, "chk_divide_by_pixels") and self.chk_divide_by_pixels.isChecked()
        )

        if use_real and getattr(self, "merged_results_raw", {}):
            raw_data = self.merged_results_raw
            pixel_counts = getattr(self, "roi_pixel_counts", {})
            norm_factor = getattr(self, "frame_norm_factor", 1.0)
            if not divide and pixel_counts:
                scale = {roi: pixel_counts.get(roi, 1) * norm_factor for roi in raw_data}
                all_vals = [v * scale.get(roi, 1.0) for roi, pts in raw_data.items() for _, v in pts]
                unit = "pixel sum (MATLAB)"
            else:
                all_vals = [v for pts in raw_data.values() for _, v in pts]
                unit = "per-pixel mean"
        elif getattr(self, "merged_results", {}):
            all_vals = [v for pts in self.merged_results.values() for _, v in pts]
            unit = "normalized [0-1]"
        else:
            self.fixed_signal_stats_label.setText("Signal range: run analysis first")
            return

        all_vals = np.array(all_vals)
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            self.fixed_signal_stats_label.setText("Signal range: no data")
            return

        def fmt(v):
            if abs(v) >= 1000:
                return f"{v:.0f}"
            elif abs(v) >= 1:
                return f"{v:.3f}"
            else:
                return f"{v:.3e}"

        mean_v = float(np.mean(all_vals))
        p95_v = float(np.percentile(all_vals, 95))
        max_v = float(np.max(all_vals))
        self.fixed_signal_stats_label.setText(
            f"[{unit}]\n"
            f"Mean: {fmt(mean_v)}   P95: {fmt(p95_v)}   Max: {fmt(max_v)}\n"
            f"→ Threshold suggestion: ~{fmt(p95_v * 1.5)}"
        )

    def _update_real_amplitude_controls(self):
        """Enable/disable Real Amplitude checkboxes based on plot type and data availability."""
        if not hasattr(self, "show_real_amplitude"):
            return
        plot_type = self.plot_type_combo.currentText() if hasattr(self, "plot_type_combo") else ""
        is_raw_plot = (plot_type == "Raw Intensity Changes")
        has_raw_data = bool(getattr(self, "merged_results_raw", {}))
        available = is_raw_plot and has_raw_data
        self.show_real_amplitude.setEnabled(available)
        if not available:
            self.show_real_amplitude.setToolTip(
                "Only available for 'Raw Intensity Changes' plot type\n"
                "and requires raw (pre-MinMax) data from the current analysis session."
            )
            if hasattr(self, "chk_divide_by_pixels"):
                self.chk_divide_by_pixels.setEnabled(False)
        else:
            self.show_real_amplitude.setToolTip(
                "Toggle between MinMax-normalized [0,1] view and real amplitude values.\n"
                "Real amplitude shows sum(|Δpixel|) per ROI in raw pixel counts (MATLAB-style)."
            )
            if hasattr(self, "chk_divide_by_pixels"):
                self.chk_divide_by_pixels.setEnabled(self.show_real_amplitude.isChecked())

    # ------------------------------------------------------------------
    # Per-ROI Y-axis limit helpers
    # ------------------------------------------------------------------

    def _refresh_per_roi_controls(self):
        """Rebuild per-ROI Y-limit rows from the currently loaded ROI data."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("No data loaded — run analysis first.")
            return
        roi_ids = sorted(self.merged_results.keys())
        self._rebuild_per_roi_y_controls(roi_ids)

    def _rebuild_per_roi_y_controls(self, roi_ids):
        """Create one Min/Max row per ROI inside the scrollable area."""
        from qtpy.QtWidgets import QHBoxLayout, QDoubleSpinBox, QLabel, QPushButton, QWidget

        # Clear previous rows
        while self.per_roi_inner_layout.count():
            item = self.per_roi_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._per_roi_y_widgets = {}

        for roi_id in roi_ids:
            color = self.roi_colors.get(roi_id, "#888888") if hasattr(self, "roi_colors") else "#888888"

            row_widget = QWidget()
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row_widget.setLayout(row)

            lbl = QLabel(f"ROI {roi_id}")
            lbl.setStyleSheet(
                f"color: {color}; font-weight: bold; min-width: 50px;"
            )

            y_min_spin = QDoubleSpinBox()
            y_min_spin.setRange(-1e9, 1e9)
            y_min_spin.setDecimals(5)
            y_min_spin.setSingleStep(0.0001)
            y_min_spin.setValue(self.per_roi_y_limits.get(roi_id, (0.0, 1.0))[0])
            y_min_spin.setFixedWidth(95)

            y_max_spin = QDoubleSpinBox()
            y_max_spin.setRange(-1e9, 1e9)
            y_max_spin.setDecimals(5)
            y_max_spin.setSingleStep(0.0001)
            y_max_spin.setValue(self.per_roi_y_limits.get(roi_id, (0.0, 1.0))[1])
            y_max_spin.setFixedWidth(95)

            btn_apply = QPushButton("Apply")
            btn_apply.setFixedWidth(52)
            btn_apply.clicked.connect(
                lambda _, r=roi_id, mn=y_min_spin, mx=y_max_spin:
                    self._apply_single_roi_ylimit(r, mn, mx)
            )

            row.addWidget(lbl)
            row.addWidget(QLabel("Min:"))
            row.addWidget(y_min_spin)
            row.addWidget(QLabel("Max:"))
            row.addWidget(y_max_spin)
            row.addWidget(btn_apply)
            row.addStretch()

            self.per_roi_inner_layout.addWidget(row_widget)
            self._per_roi_y_widgets[roi_id] = (y_min_spin, y_max_spin)

    def _apply_single_roi_ylimit(self, roi_id, y_min_spin, y_max_spin):
        """Store the limit for one ROI and regenerate the plot."""
        lo, hi = y_min_spin.value(), y_max_spin.value()
        if lo >= hi:
            self._log_message(f"⚠️ ROI {roi_id}: Y Min must be < Y Max")
            return
        self.per_roi_y_limits[roi_id] = (lo, hi)
        self.generate_plot()

    def _read_current_ylimits(self):
        """Populate per-ROI spinboxes from the current y-axis limits in the plot."""
        if not hasattr(self, "_per_roi_y_widgets") or not self._per_roi_y_widgets:
            self._refresh_per_roi_controls()
            return
        roi_ids = sorted(self._per_roi_y_widgets.keys())
        # Collect only non-colorbar, non-polar visible axes in draw order
        axes = [
            ax for ax in self.figure.get_axes()
            if ax.get_visible() and ax.get_label() != "<colorbar>"
        ]
        for i, roi_id in enumerate(roi_ids):
            if i >= len(axes):
                break
            ymin, ymax = axes[i].get_ylim()
            y_min_spin, y_max_spin = self._per_roi_y_widgets[roi_id]
            y_min_spin.setValue(ymin)
            y_max_spin.setValue(ymax)
        self._log_message(
            f"Read Y-limits from plot for {min(len(roi_ids), len(axes))} ROI(s)"
        )

    def _reset_per_roi_ylimits(self):
        """Clear all per-ROI limits and regenerate with auto-scaling."""
        self.per_roi_y_limits = {}
        # Reset spinbox display values
        for (y_min_spin, y_max_spin) in self._per_roi_y_widgets.values():
            y_min_spin.setValue(0.0)
            y_max_spin.setValue(1.0)
        self.generate_plot()

    def generate_plot(self):
        """Generate plot using PlotGenerator."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.results_label.setText("No analysis results to plot.")
            return

        # ROI colors should now come from _calc.py, but add safety check
        if not hasattr(self, "roi_colors") or not self.roi_colors:
            self._log_message("No ROI colors from analysis, creating fallback")
            self.roi_colors = {
                roi: f"C{(roi - 1) % 10}" for roi in sorted(self.merged_results.keys())
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
            # Check amplitude mode toggle
            use_real_amplitude = (
                hasattr(self, "show_real_amplitude")
                and self.show_real_amplitude.isChecked()
            )

            # ZT mode and lighting overlay settings
            zt_mode = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
            show_lighting = self.chk_show_lighting.isChecked() if hasattr(self, "chk_show_lighting") else False
            if show_lighting:
                _raw_led = getattr(self, "led_data", None)
                if _raw_led and isinstance(_raw_led, dict) and _raw_led.get("times") and _raw_led.get("white_powers"):
                    led_data_for_plot = _raw_led
                    self._log_message(
                        f"Light/Dark overlay: {len(_raw_led['times'])} LED data points"
                    )
                else:
                    # No white LED channel in file — pass empty dict to trigger legacy 12h fallback
                    led_data_for_plot = {}
                    self._log_message(
                        "Light/Dark: no white LED data in file — using legacy 12h cycle (lights-on at ZT 0)"
                    )
            else:
                led_data_for_plot = None

            # Get data based on plot type
            if plot_type == "Raw Intensity Changes":
                pixel_sum_scales = {}  # {roi: scale_factor} — used to scale thresholds too
                if use_real_amplitude and not getattr(self, "merged_results_raw", {}):
                    self._log_message(
                        "⚠️ Real Amplitude: no raw data available (not saved in HDF5 results). "
                        "Run a fresh analysis to use this mode."
                    )
                if use_real_amplitude and getattr(self, "merged_results_raw", {}):
                    divide_by_pixels = (
                        hasattr(self, "chk_divide_by_pixels")
                        and self.chk_divide_by_pixels.isChecked()
                    )
                    pixel_counts = getattr(self, "roi_pixel_counts", {})
                    norm_factor = getattr(self, "frame_norm_factor", 1.0)
                    if not divide_by_pixels and pixel_counts:
                        # MATLAB-equivalent pixel sum:
                        # mean × n_pixels × norm_factor = Σ|ΔPixel_raw| per ROI
                        pixel_sum_scales = {
                            roi: pixel_counts.get(roi, 1) * norm_factor
                            for roi in self.merged_results_raw
                        }
                        data_dict = {
                            roi: [(t, v * pixel_sum_scales[roi]) for t, v in data]
                            for roi, data in self.merged_results_raw.items()
                        }
                        self._log_message(f"Plot mode: Pixel Sum MATLAB (×{norm_factor:.0f})")
                    else:
                        # Per-pixel mean mode: sum(|Δpixel|) ÷ n_pixels
                        data_dict = self.merged_results_raw
                        self._log_message("Plot mode: Per-pixel mean")
                else:
                    data_dict = self.merged_results
                    self._log_message("Plot mode: Normalized [0-1]")

                from ._plot import create_hysteresis_kwargs

                kwargs = create_hysteresis_kwargs(
                    widget_instance=self, use_real_amplitude=use_real_amplitude
                )
                # Remove merged_results from kwargs to avoid duplicate argument
                kwargs.pop("merged_results", None)

                # Scale threshold lines to match pixel sum mode
                if pixel_sum_scales:
                    for thresh_key in ("roi_baseline_means", "roi_upper_thresholds",
                                       "roi_lower_thresholds", "roi_band_widths"):
                        if thresh_key in kwargs and kwargs[thresh_key]:
                            kwargs[thresh_key] = {
                                roi: val * pixel_sum_scales.get(roi, 1.0)
                                for roi, val in kwargs[thresh_key].items()
                            }

                # ZT mode and lighting overlay
                kwargs["zt_mode"] = zt_mode
                kwargs["led_data"] = led_data_for_plot

                # Per-ROI Y-axis limits (only when the group is enabled)
                if (
                    hasattr(self, "per_roi_y_group")
                    and self.per_roi_y_group.isChecked()
                    and self.per_roi_y_limits
                ):
                    kwargs["per_roi_y_limits"] = self.per_roi_y_limits
                    # Auto-rebuild the ROI list if not yet done
                    if not self._per_roi_y_widgets:
                        self._rebuild_per_roi_y_controls(sorted(data_dict.keys()))

            elif plot_type == "Movement":
                data_dict = getattr(self, "movement_data", {})
                kwargs = {"zt_mode": zt_mode, "led_data": led_data_for_plot}

            elif plot_type == "Fraction Movement":
                plot_bin_value = getattr(self, "plot_bin_minutes", None)
                bin_minutes = plot_bin_value.value() if plot_bin_value else 0
                data_dict, _, _ = self._get_rebinned_behavioral_data(bin_minutes)
                if bin_minutes > 0:
                    original_bin = self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60
                    new_bin_seconds = bin_minutes * 60
                    if new_bin_seconds > original_bin:
                        self._log_message(f"Fraction Movement re-binned: {original_bin}s → {new_bin_seconds}s")
                kwargs = {"zt_mode": zt_mode, "led_data": led_data_for_plot}

            elif plot_type == "Quiescence":
                plot_bin_value = getattr(self, "plot_bin_minutes", None)
                bin_minutes = plot_bin_value.value() if plot_bin_value else 0
                _, data_dict, _ = self._get_rebinned_behavioral_data(bin_minutes)
                if bin_minutes > 0:
                    original_bin = self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60
                    new_bin_seconds = bin_minutes * 60
                    if new_bin_seconds > original_bin:
                        self._log_message(f"Quiescence re-derived from {new_bin_seconds}s rebinned fraction data")
                kwargs = {"zt_mode": zt_mode, "led_data": led_data_for_plot}

            elif plot_type == "Sleep":
                plot_bin_value = getattr(self, "plot_bin_minutes", None)
                bin_minutes = plot_bin_value.value() if plot_bin_value else 0
                _, _, data_dict = self._get_rebinned_behavioral_data(bin_minutes)
                if bin_minutes > 0:
                    original_bin = self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60
                    new_bin_seconds = bin_minutes * 60
                    if new_bin_seconds > original_bin:
                        self._log_message(f"Sleep re-derived from {new_bin_seconds}s rebinned fraction data")
                kwargs = {"zt_mode": zt_mode, "led_data": led_data_for_plot}

            elif plot_type == "Sleep Quality":
                data_dict = getattr(self, "sleep_quality_data", {})
                # Get metric from sub-selector if available
                sleep_metric = "sleep_minutes"
                if hasattr(self, "sleep_quality_metric_combo"):
                    metric_text = self.sleep_quality_metric_combo.currentText()
                    if "Transitions" in metric_text:
                        sleep_metric = "transitions"
                    elif "Bout" in metric_text:
                        sleep_metric = "bout_length"
                    elif "day" in metric_text:
                        sleep_metric = "sleep_hours_per_day"
                kwargs = {"sleep_metric": sleep_metric, "zt_mode": zt_mode, "led_data": led_data_for_plot}

            elif plot_type == "Lighting Conditions (dark IR)":
                data_dict = getattr(self, "fraction_data", {})
                from ._plot import create_hysteresis_kwargs

                kwargs = create_hysteresis_kwargs(
                    widget_instance=self
                )  # Keep merged_results for lighting

                # Use plot_bin_minutes from GUI (default 10 if not set)
                plot_bin_value = getattr(self, "plot_bin_minutes", None)
                bin_minutes = plot_bin_value.value() if plot_bin_value else 10
                kwargs.update({"bin_minutes": bin_minutes})
                self._log_message(f"Using plot binning: {bin_minutes} minutes")

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

            # Create plot config — screen rendering, publication settings only at export
            from ._plot import create_plot_config

            plot_config = create_plot_config(self)
            plot_config["export_mode"] = False

            # Generate plot
            success = self.plot_generator.generate_plot(
                plot_type, data_dict, self.roi_colors, plot_config, **kwargs
            )

            if success:
                # Force complete canvas refresh; suppress the UserWarning that
                # tight_layout emits when subplots_adjust was already applied
                import warnings as _warnings
                try:
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore", UserWarning)
                        self.figure.tight_layout()
                except Exception:
                    pass
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
            if self.file_path.lower().endswith((".avi", ".mp4")):
                return self._extract_led_data_from_avi()

            # HDF5 or Zarr file processing via format-agnostic reader
            with open_file_reader(self.file_path) as r:
                root_keys = r.keys("/")
                if "timeseries" not in root_keys:
                    return None

                ts_keys = r.keys("timeseries")

                # ── Step 1: Try to find white LED data ────────────────────────────
                # New format (v2.1.0+): separate white_led_power / ir_led_power keys.
                # Old format (pre-v2.1.0): single led_power key shared between both
                # LEDs; led_type_str ("white"/"ir") or phase_str ("light"/"dark")
                # indicates which LED was active per frame.
                white_led = None
                ir_led = None

                # New-format: explicit white LED key
                white_led_names = [
                    "led_white_power_percent",
                    "white_led_power_percent",
                    "led_white_power",
                    "white_led_power",
                    "white_led",
                ]
                for name in white_led_names:
                    if name in ts_keys:
                        white_led = r.read_all(f"timeseries/{name}").astype(float)
                        self._log_message(f"Found white LED data: {name}")
                        break

                # Old-format: single led_power + type discriminator
                if white_led is None and "led_power" in ts_keys:
                    led_power = r.read_all("timeseries/led_power").astype(float)
                    self._log_message("Found legacy led_power key — reconstructing white/IR from type discriminator")

                    if "led_type_str" in ts_keys:
                        # led_type_str values: "white" → light phase, "ir" → dark phase
                        led_type_raw = r.read_all("timeseries/led_type_str")
                        led_type_str = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in led_type_raw])
                        white_mask = led_type_str == "white"
                        white_led = np.where(white_mask, led_power, 0.0)
                        ir_led = np.where(~white_mask, led_power, 0.0)
                        self._log_message("Reconstructed white/IR LED from led_type_str")
                    elif "phase_str" in ts_keys:
                        # phase_str values: "light" → white LED on, "dark"/"continuous" → IR only
                        phase_raw = r.read_all("timeseries/phase_str")
                        phase_str = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in phase_raw])
                        light_mask = phase_str == "light"
                        white_led = np.where(light_mask, led_power, 0.0)
                        ir_led = np.where(~light_mask, led_power, 0.0)
                        self._log_message("Reconstructed white/IR LED from phase_str")
                    else:
                        # No type info — treat led_power as white LED (best guess)
                        white_led = led_power
                        self._log_message("No LED type discriminator found — treating led_power as white LED")

                # Special case: "led_power_percent" without specific white/IR separation
                if white_led is None and "led_power_percent" in ts_keys:
                    self._log_message(
                        "Found generic 'led_power_percent' but no white LED channel - likely IR-only system"
                    )
                    self._log_message(
                        "→ Using legacy 12h light/dark cycles for visualization"
                    )
                    return None

                # Phase-string fallback: no LED power data at all, but phase_str exists
                if white_led is None and "phase_str" in ts_keys:
                    phase_raw = r.read_all("timeseries/phase_str")
                    phase_str = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in phase_raw])
                    # Binary 1/0 is sufficient for extract_illumination_periods (checks > 0)
                    white_led = (phase_str == "light").astype(float)
                    self._log_message("No LED power data — using phase_str to derive light/dark phases")

                if white_led is None:
                    self._log_message("No white LED data found in timeseries")
                    return None

                # ── Step 2: IR LED (new format, if not already set above) ─────────
                if ir_led is None:
                    ir_led_names = [
                        "led_ir_power_percent",
                        "ir_led_power",
                        "led_ir_power",
                        "ir_led_power_percent",
                    ]
                    for name in ir_led_names:
                        if name in ts_keys:
                            ir_led = r.read_all(f"timeseries/{name}").astype(float)
                            self._log_message(f"Found IR LED data: {name}")
                            break

                # ── Step 3: Timestamps ────────────────────────────────────────────
                # Old format writes both absolute `timestamps` and `recording_elapsed_sec`.
                # New format writes only `recording_elapsed_sec` (and optionally
                # `capture_timestamps` in COMPREHENSIVE mode).
                frame_interval = self.frame_interval.value()
                if "recording_elapsed_sec" in ts_keys:
                    times = r.read_all("timeseries/recording_elapsed_sec").astype(float)
                elif "capture_timestamps" in ts_keys:
                    times = r.read_all("timeseries/capture_timestamps").astype(float)
                elif "timestamps" in ts_keys:
                    ts_raw = r.read_all("timeseries/timestamps").astype(float)
                    times = ts_raw - ts_raw[0] if len(ts_raw) > 0 else ts_raw
                else:
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

    def _extract_frame_mean_from_hdf5(self):
        """Extract frame_mean timeseries from HDF5 for jump correction.

        Returns:
            dict with 'times' (list of floats, seconds) and 'values' (list of floats),
            or None if not available.
        """
        try:
            if not hasattr(self, "file_path") or not self.file_path:
                return None
            if self.file_path.lower().endswith((".avi", ".mp4")):
                return None

            with open_file_reader(self.file_path) as r:
                root_keys = r.keys("/")
                if "timeseries" not in root_keys:
                    return None
                ts_keys = r.keys("timeseries")
                # Accept both HDF5 name (frame_mean) and Zarr name (frame_mean_intensity)
                fm_key = None
                for candidate in ("frame_mean", "frame_mean_intensity"):
                    if candidate in ts_keys:
                        fm_key = candidate
                        break
                if fm_key is None:
                    return None

                fm = r.read_all(f"timeseries/{fm_key}").astype(float)

                frame_interval = self.frame_interval.value()
                if "capture_timestamps" in ts_keys:
                    times = r.read_all("timeseries/capture_timestamps").astype(float)
                elif "recording_elapsed_sec" in ts_keys:
                    times = r.read_all("timeseries/recording_elapsed_sec").astype(float)
                else:
                    times = np.arange(len(fm), dtype=float) * frame_interval

                return {"times": times.tolist(), "values": fm.tolist()}

        except Exception as e:
            self._log_message(f"Could not extract frame_mean: {e}")
            return None

    def apply_time_range(self):
        """Apply time range and regenerate plot."""
        self.generate_plot()

    def save_current_plot(self):
        """Save the current plot using _plot.py module."""
        # Build a meaningful default filename
        try:
            base = os.path.splitext(os.path.basename(self.file_path))[0]
        except Exception:
            base = "analysis"
        plot_type = (
            self.plot_type_combo.currentText().replace(" ", "_").lower()
            if hasattr(self, "plot_type_combo")
            else "plot"
        )
        default_name = f"{base}_{plot_type}.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            default_name,
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )

        if file_path:
            dpi = self.plot_dpi_spin.value()
            success = save_plot(self.figure, file_path, dpi, publication_style=(dpi >= 300))

            if success:
                self._log_message(f"Plot saved: {os.path.basename(file_path)}")
                self.results_label.setText(f"Plot saved: {os.path.basename(file_path)}")
            else:
                error_msg = "Failed to save plot"
                self.results_label.setText(error_msg)
                self._log_message(f"ERROR: {error_msg}")

    def save_individual_roi_plots(self):
        """Save one PNG per ROI for the currently selected plot type."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.results_label.setText("No analysis results — run analysis first.")
            return

        plot_type = self.plot_type_combo.currentText() if hasattr(self, "plot_type_combo") else ""

        # Get the data dict for the current plot type
        if plot_type == "Raw Intensity Changes":
            data_dict = getattr(self, "merged_results", {})
        elif plot_type == "Movement":
            data_dict = getattr(self, "movement_data", {})
        elif plot_type in ("Fraction Movement", "Quiescence", "Sleep"):
            bin_minutes = self.plot_bin_minutes.value() if hasattr(self, "plot_bin_minutes") else 0
            fd, qd, sd = self._get_rebinned_behavioral_data(bin_minutes)
            data_dict = {"Fraction Movement": fd, "Quiescence": qd, "Sleep": sd}[plot_type]
        elif plot_type == "Sleep Quality":
            data_dict = getattr(self, "sleep_quality_data", {})
        else:
            self.results_label.setText(f"Individual save not supported for '{plot_type}'.")
            return

        if not data_dict:
            self.results_label.setText(f"No {plot_type} data available.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for individual ROI plots")
        if not out_dir:
            return

        try:
            base = os.path.splitext(os.path.basename(self.file_path))[0]
        except Exception:
            base = "analysis"
        plot_slug = plot_type.replace(" ", "_").lower()

        from ._plot import PlotGenerator, create_plot_config
        plot_config = create_plot_config(self)
        plot_config["export_mode"] = True  # publication DPI + dimensions for file output
        zt_mode = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
        show_lighting = self.chk_show_lighting.isChecked() if hasattr(self, "chk_show_lighting") else False
        led_data_for_plot = (getattr(self, "led_data", None) or {}) if show_lighting else None
        kwargs = {"zt_mode": zt_mode, "led_data": led_data_for_plot}

        dpi = self.plot_dpi_spin.value() if hasattr(self, "plot_dpi_spin") else 300
        saved = 0
        roi_colors = getattr(self, "roi_colors", {})

        for roi_id in sorted(data_dict.keys()):
            single_data = {roi_id: data_dict[roi_id]}
            single_colors = {roi_id: roi_colors.get(roi_id, f"C{(roi_id - 1) % 10}")}

            from matplotlib.figure import Figure as _Figure
            from ._plot import JOURNAL_SINGLE_COL_IN, apply_publication_style
            fig = _Figure(figsize=(JOURNAL_SINGLE_COL_IN, 2.5))
            gen = PlotGenerator(fig)
            gen.generate_plot(plot_type, single_data, single_colors, plot_config, **kwargs)

            out_path = os.path.join(out_dir, f"{base}_{plot_slug}_ROI{roi_id}.png")
            prev_rc = apply_publication_style()
            try:
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            finally:
                import matplotlib.pyplot as _plt
                _plt.rcParams.update(prev_rc)
            saved += 1

        self._log_message(f"✅ Saved {saved} individual ROI plot(s) to {out_dir}")
        self.results_label.setText(f"Saved {saved} individual ROI plots to folder.")

    def save_all_plots(self):
        """Save all plot types using _plot.py module."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save All Plots"
        )

        if not directory:
            return

        try:
            # Extract current display settings so saved plots match what is shown
            zt_mode = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
            show_lighting = self.chk_show_lighting.isChecked() if hasattr(self, "chk_show_lighting") else False
            if show_lighting:
                led_data_for_plot = getattr(self, "led_data", None) or {}
            else:
                led_data_for_plot = None

            # Apply pixel-sum mode to raw intensity data if the toggle is off (= sum mode)
            divide_by_pixels = hasattr(self, "chk_divide_by_pixels") and self.chk_divide_by_pixels.isChecked()
            pixel_counts = getattr(self, "roi_pixel_counts", {})
            norm_factor = getattr(self, "frame_norm_factor", 1.0)
            raw_results = getattr(self, "merged_results_raw", {})
            if not divide_by_pixels and pixel_counts and raw_results:
                merged_for_plot = {
                    roi: [(t, v * pixel_counts.get(roi, 1) * norm_factor) for t, v in data]
                    for roi, data in raw_results.items()
                }
            else:
                merged_for_plot = getattr(self, "merged_results", {})

            # Prepare all data sets
            data_sets = {
                "merged_results": merged_for_plot,
                "movement_data": getattr(self, "movement_data", {}),
                "fraction_data": getattr(self, "fraction_data", {}),
                "quiescence_data": getattr(self, "quiescence_data", {}),
                "sleep_data": getattr(self, "sleep_data", {}),
            }

            # Create plot configuration — export uses publication DPI + dimensions
            plot_config = create_plot_config(self)
            plot_config["export_mode"] = True

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
                zt_mode=zt_mode,
                led_data=led_data_for_plot,
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
        if self.file_path.lower().endswith((".avi", ".mp4")) or (
            hasattr(self, "avi_batch_paths") and self.avi_batch_paths
        ):
            self._log_message("Video file(s) loaded - skipping HDF5 structure check")
            return

        import h5py

        # Determine whether this is HDF5 or Zarr so we use the right detector
        _zarr_markers = (".zgroup", ".zarray", ".zmetadata")
        _is_zarr = os.path.isdir(self.file_path) and any(
            os.path.exists(os.path.join(self.file_path, m)) for m in _zarr_markers
        )

        try:
            if DUAL_STRUCTURE_AVAILABLE:
                # Use format-agnostic detection for Zarr, HDF5-specific for HDF5
                if _is_zarr:
                    structure_info = detect_file_structure_type(self.file_path)
                else:
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
                    if structure_info.get("key_template"):
                        self._log_message(
                            f"   Key template: {structure_info['key_template']}"
                        )
                    elif structure_info.get("frame_keys"):
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
                # Fallback to original method — skip for Zarr stores
                if _is_zarr:
                    self._log_message("Zarr store loaded — basic HDF5 fallback skipped")
                else:
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
        method_map = {0: "baseline", 1: "calibration", 2: "adaptive", 3: "fixed"}
        return method_map.get(tab_index, "baseline")

    def _get_current_threshold_method_display(self) -> str:
        """Get current threshold method display name based on active tab."""
        tab_index = self.threshold_params_stack.currentIndex()
        method_map = {
            0: "Baseline (First Frames)",
            1: "Calibration (Sedated Animals)",
            2: "Adaptive (Smart Detection)",
            3: "Fixed Threshold (Paper Value)",
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
                "Video Files (*.h5 *.hdf5 *.avi *.mp4);;HDF5 Files (*.h5 *.hdf5);;Video Files (*.avi *.mp4);;All Files (*.*)",
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

    def _on_6well_toggled(self, checked: bool):
        """Enable/disable ROI spinboxes when 6-well preset is toggled."""
        if checked:
            self.min_radius.setValue(100)
            self.max_radius.setValue(145)
            self.dp_param.setValue(1.0)
            self.min_dist.setValue(300)
            self.param1.setValue(30)
            self.param2.setValue(60)

        for widget in (
            self.min_radius,
            self.max_radius,
            self.dp_param,
            self.min_dist,
            self.param1,
            self.param2,
        ):
            widget.setEnabled(not checked)

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
                    "adaptive_illumination_baseline": getattr(
                        widget, "adaptive_illumination_baseline", None
                    ),
                }
            )

            aib = base_params["adaptive_illumination_baseline"]
            if hasattr(aib, "isChecked"):
                base_params["adaptive_illumination_baseline"] = aib.isChecked()
            else:
                base_params["adaptive_illumination_baseline"] = False

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
