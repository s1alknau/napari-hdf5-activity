"""_widget_circadian.py — CircadianMixin for HDF5AnalysisWidget.

Provides the Extended Analysis tab (circadian rhythmicity): Chi-square
periodogram (Fisher exact), FFT power spectrum, cosinor analysis, ROI
similarity matrix, coherence analysis, phase clustering, actogram, and all
associated plots and exports.  Mixed into HDF5AnalysisWidget so all methods
share the same ``self`` namespace.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


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

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import QTimer, Qt, QSettings
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CircadianMixin:
    """Mixin providing all circadian / rhythmicity analysis functionality.

    Requires that the host class (HDF5AnalysisWidget) provides:
    - self.merged_results, self.fraction_data, self.movement_data
    - self.quiescence_data, self.sleep_data
    - self.frame_interval, self.bin_size_seconds
    - self.viewer (napari.Viewer)
    - self._log_message(msg: str)
    """

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

    def _set_period_preset(self, min_period, max_period):
        """Set period range from preset."""
        self.fisher_min_period.setValue(min_period)
        self.fisher_max_period.setValue(max_period)
        self._log_message(
            f"Period range set to: {min_period:.1f} - {max_period:.1f} hours"
        )

    def _auto_detect_period_range(self):
        """Automatically detect optimal period range based on recording duration."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("⚠️ No data loaded. Please run analysis first.")
            return

        # Get recording duration from first ROI
        first_roi_data = next(iter(self.merged_results.values()))
        if not first_roi_data:
            return

        times = [t for t, _ in first_roi_data]
        duration_seconds = max(times) - min(times)
        duration_hours = duration_seconds / 3600.0

        # Determine optimal range based on duration
        # Rule: max_period should be at most 1/2 of recording duration
        if duration_hours < 12:
            # Short recording: 0.5h - duration/2
            min_period = 0.5
            max_period = max(3.0, duration_hours / 2)
            preset_name = "Short Cycles"
        elif duration_hours < 48:
            # Medium recording: 1h - duration/2
            min_period = 1.0
            max_period = min(18.0, duration_hours / 2)
            preset_name = "Ultradian/Semi-Daily"
        else:
            # Long recording: circadian analysis
            min_period = 12.0
            max_period = min(36.0, duration_hours / 2)
            preset_name = "Circadian"

        self.fisher_min_period.setValue(min_period)
        self.fisher_max_period.setValue(max_period)

        self._log_message(
            f"Auto-detected: Recording duration = {duration_hours:.1f}h\n"
            f"  → Using '{preset_name}' preset: {min_period:.1f} - {max_period:.1f} hours"
        )

    def _on_fisher_method_changed(self, index):
        """Handle change in analysis method selection."""
        methods = [
            "Chi² Periodogram",
            "FFT Power Spectrum",
            "Cosinor Analysis",
            "ROI Similarity Matrix",
            "Coherence Analysis",
            "Phase Clustering",
        ]
        self._log_message(f"Analysis method changed to: {methods[index]}")

        # Show cluster threshold slider only for ROI Similarity Matrix
        if hasattr(self, "similarity_threshold_group"):
            self.similarity_threshold_group.setVisible(index == 3)

        # Show actogram checkbox only for Cosinor (index 2)
        if hasattr(self, "chk_show_actogram"):
            self.chk_show_actogram.setVisible(index == 2)
            if index != 2:
                # Hide settings group when leaving Cosinor
                if hasattr(self, "actogram_settings_group"):
                    self.actogram_settings_group.setVisible(False)

        # Update data source dropdown based on method
        self._update_data_source_for_method(index)

    def _rerender_current_fisher_plot(self):
        """Re-render the current extended analysis plot (e.g. when population checkbox is toggled)."""
        if (
            hasattr(self, "fisher_analysis_results")
            and self.fisher_analysis_results
            and hasattr(self, "current_fisher_method")
        ):
            self._create_circadian_plot(
                self.fisher_analysis_results, self.current_fisher_method
            )

    def _on_similarity_threshold_changed(self, value):
        """Update label and redraw dendrogram when the cluster threshold slider moves."""
        r = value / 100.0
        if hasattr(self, "similarity_threshold_label"):
            self.similarity_threshold_label.setText(f"{r:.2f}")
        # Live-redraw only if similarity results are already shown
        if (
            hasattr(self, "fisher_analysis_results")
            and self.fisher_analysis_results
            and getattr(self, "current_fisher_method", -1) == 3
        ):
            self._create_similarity_plot(self.fisher_analysis_results)

    def _update_data_source_for_method(self, method_index):
        """Update data source dropdown based on selected analysis method.

        All methods (Fisher, FFT, Cosinor, etc.) can operate on either
        Fraction Movement or Raw Intensity data.
        """
        if not hasattr(self, "data_source_combo"):
            return

        self.data_source_combo.blockSignals(True)

        current_index = self.data_source_combo.currentIndex()
        self.data_source_combo.clear()
        self.data_source_combo.addItems(
            ["Fraction Movement (0-1)", "Raw Intensity (continuous)"]
        )
        if current_index < self.data_source_combo.count():
            self.data_source_combo.setCurrentIndex(current_index)
        else:
            self.data_source_combo.setCurrentIndex(0)
        self.data_source_combo.setEnabled(True)
        self.data_source_combo.setToolTip(
            "Choose data source for analysis:\n"
            "• Fraction Movement: Binned activity ratio (0-1), smoother signal\n"
            "• Raw Intensity: Continuous per-pixel intensity changes (MinMax 0-1)"
        )

        self.data_source_combo.blockSignals(False)

    def _on_cycle_selection_toggled(self, state):
        """Handle toggling of cycle selection checkbox."""
        enabled = state == 2  # Qt.Checked = 2

        # Enable/disable cycle selection controls
        self.cycle_start_time.setEnabled(enabled)
        self.cycle_end_time.setEnabled(enabled)
        self.btn_cycle_first24.setEnabled(enabled)
        self.btn_cycle_second24.setEnabled(enabled)
        self.btn_cycle_last24.setEnabled(enabled)
        self.btn_cycle_reset.setEnabled(enabled)

        if enabled:
            self._log_message(
                "Time range selection enabled - analysis will use specified window"
            )
        else:
            self._log_message(
                "Time range selection disabled - analysis will use full recording"
            )

    def _set_cycle_range(self, start_hours, end_hours):
        """Set the cycle selection range to specific hours."""
        self.cycle_start_time.setValue(start_hours)
        self.cycle_end_time.setValue(end_hours)
        self._log_message(f"Time range set to {start_hours:.1f}h - {end_hours:.1f}h")

    def _set_cycle_last_24h(self):
        """Set the cycle selection to the last 24 hours of the recording."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("⚠️ No data loaded - cannot determine recording duration")
            return

        # Get recording duration from first ROI
        first_roi = list(self.merged_results.keys())[0]
        data = self.merged_results[first_roi]

        if not data:
            self._log_message("⚠️ No data available")
            return

        # Calculate duration in hours
        duration_seconds = data[-1][0] - data[0][0]
        duration_hours = duration_seconds / 3600.0

        if duration_hours < 24:
            self._log_message(
                f"⚠️ Recording is only {duration_hours:.1f}h - using full recording"
            )
            self._set_cycle_range(0, duration_hours)
        else:
            start_hours = duration_hours - 24
            self._set_cycle_range(start_hours, duration_hours)

    def _reset_cycle_range(self):
        """Reset cycle selection to full recording."""
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("⚠️ No data loaded")
            return

        # Get recording duration from first ROI
        first_roi = list(self.merged_results.keys())[0]
        data = self.merged_results[first_roi]

        if not data:
            self._log_message("⚠️ No data available")
            return

        # Calculate duration in hours
        duration_seconds = data[-1][0] - data[0][0]
        duration_hours = duration_seconds / 3600.0

        self._set_cycle_range(0, duration_hours)

    def _adjust_analysis_bin_size(self, direction):
        """Adjust analysis bin size by one step in the given direction."""
        current = self.analysis_bin_size.value()
        step = self.analysis_bin_size.singleStep()
        new_value = current + (direction * step)

        # Clamp to valid range
        new_value = max(
            self.analysis_bin_size.minimum(),
            min(self.analysis_bin_size.maximum(), new_value),
        )
        self.analysis_bin_size.setValue(new_value)

    def _set_analysis_bin_preset(self, preset):
        """Set analysis bin size to a preset value."""
        if preset == "original":
            # Use original bin size from main analysis
            original_bin = (
                self.bin_size_seconds.value()
                if hasattr(self, "bin_size_seconds")
                else 60
            )
            self.analysis_bin_size.setValue(original_bin)
            self._log_message(f"Analysis bin size set to original: {original_bin}s")
        else:
            # Numeric preset
            self.analysis_bin_size.setValue(preset)
            self._log_message(f"Analysis bin size set to: {preset}s")

    def _update_bin_size_info(self):
        """Update the info label showing bin size comparison."""
        analysis_bin = self.analysis_bin_size.value()
        original_bin = (
            self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60
        )

        if analysis_bin == original_bin:
            self.bin_size_info_label.setText(
                f"✓ Using original bin size ({original_bin}s) - no re-binning needed"
            )
            self.bin_size_info_label.setStyleSheet(
                "color: #27ae60; font-size: 10px; font-style: italic;"
            )
        elif analysis_bin > original_bin:
            factor = analysis_bin / original_bin
            self.bin_size_info_label.setText(
                f"⚠ Data will be re-binned: {original_bin}s → {analysis_bin}s ({factor:.1f}x larger bins)"
            )
            self.bin_size_info_label.setStyleSheet(
                "color: #f39c12; font-size: 10px; font-style: italic;"
            )
        else:
            self.bin_size_info_label.setText(
                f"⚠ Cannot re-bin to smaller size: {analysis_bin}s < {original_bin}s (original). Using original."
            )
            self.bin_size_info_label.setStyleSheet(
                "color: #e74c3c; font-size: 10px; font-style: italic;"
            )

    def _on_data_source_changed(self, index):
        """Handle change in data source selection."""
        source_names = [
            "Fraction Movement (0-1)",
            "Raw Intensity (continuous)",
            "Normalized Movement (0-1)",
        ]
        if index < len(source_names):
            self._log_message(f"Data source changed to: {source_names[index]}")

    def _rebin_timeseries_data(
        self,
        data_dict: Dict[int, List[Tuple[float, float]]],
        new_bin_size: int,
        original_bin_size: int,
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        Re-bin timeseries data to a larger bin size.

        Args:
            data_dict: Dictionary of {roi_id: [(time, value), ...]}
            new_bin_size: New bin size in seconds
            original_bin_size: Original bin size in seconds

        Returns:
            Dictionary with re-binned data
        """
        if new_bin_size <= original_bin_size:
            # Cannot re-bin to smaller size, return original
            return data_dict

        # Calculate binning factor (must be integer for proper binning)
        factor = int(round(new_bin_size / original_bin_size))

        rebinned_data = {}

        for roi_id, timeseries in data_dict.items():
            if not timeseries:
                rebinned_data[roi_id] = []
                continue

            # Group data points into larger bins
            rebinned = []
            for i in range(0, len(timeseries), factor):
                bin_data = timeseries[i : i + factor]

                if not bin_data:
                    continue

                # Use middle timepoint of the bin
                times = [t for t, _ in bin_data]
                values = [v for _, v in bin_data]

                avg_time = sum(times) / len(times)
                avg_value = sum(values) / len(values)

                rebinned.append((avg_time, avg_value))

            rebinned_data[roi_id] = rebinned

        return rebinned_data

    def _get_rebinned_behavioral_data(self, bin_minutes: int):
        """Rebin fraction data and recompute quiescence and sleep consistently.

        Returns (fraction_data, quiescence_data, sleep_data) all at the same
        bin resolution. When bin_minutes=0 the stored originals are returned.
        """
        from ._calc import bin_quiescence, define_sleep_periods

        fraction = getattr(self, "fraction_data", {})
        original_bin = self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60

        if bin_minutes > 0:
            new_bin_seconds = bin_minutes * 60
            if new_bin_seconds > original_bin:
                fraction = self._rebin_timeseries_data(fraction, new_bin_seconds, original_bin)
                bin_seconds = new_bin_seconds
            else:
                bin_seconds = original_bin
        else:
            bin_seconds = original_bin

        quiescence = bin_quiescence(fraction, self.quiescence_threshold.value())
        sleep = define_sleep_periods(
            quiescence,
            self.sleep_threshold_minutes.value(),
            bin_seconds,
        )
        return fraction, quiescence, sleep

    def _load_results_from_hdf5(self):
        """Load ALL analysis results from HDF5 file (core + extended analysis)."""
        from qtpy.QtWidgets import QFileDialog
        from ._results_io import load_comprehensive_results

        # Open file dialog to select results file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Results from HDF5",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not file_path:
            self._log_message("Load cancelled by user")
            return

        try:
            self._log_message(f"Loading comprehensive results from: {file_path}")

            # Load all results using comprehensive load function
            loaded_data = load_comprehensive_results(file_path)

            # Restore core analysis results
            if "core_analysis" in loaded_data:
                core = loaded_data["core_analysis"]

                # Restore raw signal (merged_results) — new key name
                if "merged_results" in core:
                    self.merged_results = core["merged_results"]
                    self._log_message(
                        f"  ✓ Loaded raw signal for {len(self.merged_results)} ROIs"
                    )
                elif "movement_data" in core:  # backward compat with old saves
                    self.merged_results = core["movement_data"]
                    self._log_message(
                        f"  ✓ Loaded raw signal (legacy key) for {len(self.merged_results)} ROIs"
                    )

                # Restore binary movement data
                if "movement_data" in core:
                    self.movement_data = core["movement_data"]
                    self._log_message(
                        f"  ✓ Loaded binary movement data for {len(self.movement_data)} ROIs"
                    )

                # Restore fraction data
                if "fraction_data" in core:
                    self.fraction_data = core["fraction_data"]
                    self._log_message(
                        f"  ✓ Loaded fraction data for {len(self.fraction_data)} ROIs"
                    )

                # Restore quiescence data
                if "quiescence_data" in core:
                    self.quiescence_data = core["quiescence_data"]
                    self._log_message(
                        f"  ✓ Loaded quiescence data for {len(self.quiescence_data)} ROIs"
                    )

                # Restore sleep data
                if "sleep_data" in core:
                    self.sleep_data = core["sleep_data"]
                    self._log_message(
                        f"  ✓ Loaded sleep data for {len(self.sleep_data)} ROIs"
                    )
                    # Recompute sleep quality metrics from restored sleep data
                    try:
                        from ._calc import calculate_sleep_quality_hourly
                        self.sleep_quality_data = calculate_sleep_quality_hourly(
                            self.sleep_data
                        )
                        self._log_message("  ✓ Sleep quality metrics recomputed")
                    except Exception:
                        pass

                # Restore ROI colors
                if "roi_colors" in core:
                    self.roi_colors = core["roi_colors"]
                    self._log_message(
                        f"  ✓ Loaded ROI colors for {len(self.roi_colors)} ROIs"
                    )

                # Restore ROI statistics
                if "roi_statistics" in core:
                    self.roi_statistics = core["roi_statistics"]
                    self._log_message(
                        f"  ✓ Loaded ROI statistics for {len(self.roi_statistics)} ROIs"
                    )

                # Restore thresholds
                if "thresholds" in core:
                    self.roi_baseline_means = {}
                    self.roi_upper_thresholds = {}
                    self.roi_lower_thresholds = {}

                    for roi_id, thresh_data in core["thresholds"].items():
                        self.roi_baseline_means[roi_id] = thresh_data.get(
                            "baseline_mean", 0.0
                        )
                        self.roi_upper_thresholds[roi_id] = thresh_data.get(
                            "upper_threshold", 0.0
                        )
                        self.roi_lower_thresholds[roi_id] = thresh_data.get(
                            "lower_threshold", 0.0
                        )

                    self._log_message(
                        f"  ✓ Loaded thresholds for {len(self.roi_baseline_means)} ROIs"
                    )

            # Restore extended analysis results
            if "extended_analysis" in loaded_data and loaded_data["extended_analysis"]:
                extended = loaded_data["extended_analysis"]

                # Restore fisher analysis results
                if "results" in extended:
                    self.fisher_analysis_results = extended["results"]
                    self._log_message("  ✓ Loaded extended analysis results")

                # Restore method selection
                if "method_index" in extended:
                    method_idx = extended["method_index"]
                    self.current_fisher_method = method_idx
                    if hasattr(self, "fisher_method_combo"):
                        self.fisher_method_combo.setCurrentIndex(method_idx)
                    self._log_message(
                        f"  ✓ Restored analysis method: {extended.get('method_name', 'Unknown')}"
                    )

                    # Re-create plot for extended analysis
                    if hasattr(self, "fisher_analysis_results"):
                        self._create_circadian_plot(
                            self.fisher_analysis_results, method_idx
                        )

                    # Generate and display summary
                    if method_idx == 0:  # Chi² Periodogram
                        from ._fisher_analysis import generate_circadian_summary

                        summary = generate_circadian_summary(
                            self.fisher_analysis_results
                        )
                    elif method_idx == 1:  # FFT Power Spectrum
                        from ._circadian_fft import generate_fft_summary

                        summary = generate_fft_summary(self.fisher_analysis_results)
                    elif method_idx == 2:  # Cosinor Analysis
                        summary = self._generate_cosinor_summary(
                            self.fisher_analysis_results
                        )
                    elif method_idx == 3:  # ROI Similarity Matrix
                        summary = self._generate_similarity_summary(
                            self.fisher_analysis_results
                        )
                    elif method_idx == 4:  # Coherence Analysis
                        summary = self._generate_coherence_summary(
                            self.fisher_analysis_results
                        )
                    elif method_idx == 5:  # Phase Clustering
                        summary = self._generate_phase_clustering_summary(
                            self.fisher_analysis_results
                        )
                    else:
                        summary = "Extended analysis results loaded successfully."

                    self.fisher_results_text.setPlainText(summary)

            # Restore analysis parameters
            if "analysis_parameters" in loaded_data:
                params = loaded_data["analysis_parameters"]

                # Restore core parameters
                if "core" in params:
                    core_params = params["core"]
                    if (
                        "frame_interval" in core_params
                        and core_params["frame_interval"] is not None
                    ):
                        if hasattr(self, "frame_interval"):
                            self.frame_interval.setValue(core_params["frame_interval"])
                    if (
                        "bin_size_seconds" in core_params
                        and core_params["bin_size_seconds"] is not None
                    ):
                        if hasattr(self, "bin_size_seconds"):
                            self.bin_size_seconds.setValue(
                                core_params["bin_size_seconds"]
                            )

                # Restore extended parameters
                if "extended" in params:
                    ext_params = params["extended"]
                    if (
                        "min_period_hours" in ext_params
                        and ext_params["min_period_hours"] is not None
                    ):
                        if hasattr(self, "fisher_min_period"):
                            self.fisher_min_period.setValue(
                                ext_params["min_period_hours"]
                            )
                    if (
                        "max_period_hours" in ext_params
                        and ext_params["max_period_hours"] is not None
                    ):
                        if hasattr(self, "fisher_max_period"):
                            self.fisher_max_period.setValue(
                                ext_params["max_period_hours"]
                            )
                    if (
                        "significance_level" in ext_params
                        and ext_params["significance_level"] is not None
                    ):
                        if hasattr(self, "fisher_significance"):
                            self.fisher_significance.setValue(
                                ext_params["significance_level"]
                            )

                self._log_message("  ✓ Restored analysis parameters")

            # Build summary message
            summary_lines = [
                "Successfully loaded comprehensive results from HDF5!",
                "",
                f"File: {file_path}",
            ]

            if hasattr(self, "merged_results") and self.merged_results:
                summary_lines.append(f"ROIs loaded: {len(self.merged_results)}")
                summary_lines.append(f"ROI IDs: {sorted(self.merged_results.keys())}")

            summary_lines.append("")
            summary_lines.append("Loaded Data:")

            if hasattr(self, "merged_results") and self.merged_results:
                summary_lines.append(
                    f"  • Movement data: {len(self.merged_results)} ROIs"
                )

            if hasattr(self, "fraction_data") and self.fraction_data:
                summary_lines.append(
                    f"  • Fraction data: {len(self.fraction_data)} ROIs"
                )

            if hasattr(self, "roi_baseline_means") and self.roi_baseline_means:
                summary_lines.append(
                    f"  • Thresholds: {len(self.roi_baseline_means)} ROIs"
                )

            if (
                hasattr(self, "fisher_analysis_results")
                and self.fisher_analysis_results
            ):
                method_name = loaded_data.get("extended_analysis", {}).get(
                    "method_name", "Unknown"
                )
                summary_lines.append(f"  • Extended analysis: {method_name}")

            summary_lines.extend(
                [
                    "",
                    "All analysis results have been restored.",
                    "You can now perform post-hoc cycle/period analysis or export results.",
                ]
            )

            self._log_message("✓ Successfully loaded comprehensive results")

            # Auto-adjust time range spinboxes to match loaded recording duration
            self._auto_adjust_time_range()
            # Raw data is not saved in HDF5 results — disable Real Amplitude controls
            self._update_real_amplitude_controls()

            # Compute pixel counts per ROI for Real Amplitude (sum) display mode.
            # Uses current self.masks if they are loaded (from ROI detection step).
            masks = getattr(self, "masks", [])
            if masks:
                self.roi_pixel_counts = {
                    i + 1: int(np.sum(m > 0)) for i, m in enumerate(masks)
                }
                self._log_message(
                    f"  ✓ Pixel counts computed for {len(self.roi_pixel_counts)} ROIs"
                )
            else:
                self.roi_pixel_counts = {}
                self._log_message(
                    "  ⚠ No masks available — Real Amplitude (pixel sum) mode not available"
                )

            # Only show summary in text widget if no extended analysis was loaded
            # (if extended analysis was loaded, its summary is already shown above)
            if not (
                "extended_analysis" in loaded_data and loaded_data["extended_analysis"]
            ):
                self.fisher_results_text.setPlainText("\n".join(summary_lines))

            # Enable export button if we have extended results
            if (
                hasattr(self, "fisher_analysis_results")
                and self.fisher_analysis_results
            ):
                if hasattr(self, "btn_export_fisher"):
                    self.btn_export_fisher.setEnabled(True)

        except Exception as e:
            self.fisher_results_text.setPlainText(
                f"ERROR loading results from HDF5:\n\n{str(e)}"
            )
            self._log_message(f"ERROR loading results from HDF5: {e}")
            import traceback

            traceback.print_exc()

    def _save_results_to_hdf5(self):
        """Save ALL analysis results to HDF5 file (core + extended analysis)."""
        from qtpy.QtWidgets import QFileDialog
        import os
        from ._results_io import save_comprehensive_results

        # Check if we have results to save
        if not hasattr(self, "merged_results") or not self.merged_results:
            self.fisher_results_text.setPlainText(
                "ERROR: No analysis results to save.\n\n"
                "Please run analysis first or load data before saving."
            )
            self._log_message("⚠️ Cannot save: No analysis results available")
            return

        # Open file dialog to select save location
        default_name = "comprehensive_analysis_results.h5"
        if hasattr(self, "file_path") and self.file_path:
            # Suggest name based on current file
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            default_name = f"{base_name}_comprehensive_results.h5"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results to HDF5",
            default_name,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not file_path:
            self._log_message("Save cancelled by user")
            return

        try:
            self._log_message(f"Saving comprehensive results to: {file_path}")

            # Collect core analysis results
            core_results = {
                "merged_results": self.merged_results,        # raw frame differences
                "movement_data_binary": getattr(self, "movement_data", {}),  # binary per bin
                "fraction_data": getattr(self, "fraction_data", {}),
                "quiescence_data": getattr(self, "quiescence_data", {}),
                "sleep_data": getattr(self, "sleep_data", {}),
                "roi_colors": getattr(self, "roi_colors", {}),
                "roi_statistics": getattr(self, "roi_statistics", {}),
                "roi_summary": {},
                "thresholds": {},
            }

            # Add threshold information (core values + extra stats from roi_statistics)
            if hasattr(self, "roi_baseline_means"):
                for roi_id in self.roi_baseline_means:
                    thresh_dict = {
                        "baseline_mean": self.roi_baseline_means.get(roi_id, 0.0),
                        "upper_threshold": self.roi_upper_thresholds.get(roi_id, 0.0),
                        "lower_threshold": self.roi_lower_thresholds.get(roi_id, 0.0),
                    }
                    # Append extra stats if available
                    if hasattr(self, "roi_statistics") and roi_id in self.roi_statistics:
                        stats = self.roi_statistics[roi_id]
                        for key in ("std", "threshold_band", "min_baseline",
                                    "multiplier", "mean"):
                            if key in stats:
                                thresh_dict[key] = float(stats[key])
                    core_results["thresholds"][roi_id] = thresh_dict

            # Collect extended analysis results if available
            extended_results = None
            if (
                hasattr(self, "fisher_analysis_results")
                and self.fisher_analysis_results
            ):
                method_index = getattr(self, "current_fisher_method", 0)
                method_name = (
                    self.fisher_method_combo.currentText()
                    if hasattr(self, "fisher_method_combo")
                    else "Unknown"
                )

                # Format extended results to match expected structure
                # The save function expects keys like 'fisher', 'fft', 'cosinor', etc.
                extended_results = {
                    "method_index": method_index,
                    "method_name": method_name,
                }

                # Map method index to expected key names
                results_data = self.fisher_analysis_results
                if method_index == 0:  # Chi² Periodogram
                    # Filter to ROI data only (integers), format for save
                    fisher_data = {}
                    for roi_id, roi_result in results_data.items():
                        if isinstance(roi_id, int):
                            periodogram = roi_result.get("periodogram", {})
                            fisher_data[roi_id] = {
                                "periods": periodogram.get("periods", []),
                                "z_scores": periodogram.get("z_scores", []),
                                "dominant_period": periodogram.get(
                                    "dominant_period", 0
                                ),
                                "dominant_z_score": periodogram.get(
                                    "dominant_z_score", 0
                                ),
                                "is_significant": periodogram.get(
                                    "is_significant", False
                                ),
                            }
                    extended_results["fisher"] = fisher_data
                    # Save sleep phase results if available
                    if "sleep_phase_results" in results_data:
                        extended_results["sleep_phase_results"] = results_data[
                            "sleep_phase_results"
                        ]

                elif method_index == 1:  # FFT Power Spectrum
                    fft_data = {}
                    for roi_id, roi_result in results_data.items():
                        if isinstance(roi_id, int):
                            fft_data[roi_id] = {
                                "periods": roi_result.get("relevant_periods", []),
                                "power": roi_result.get("relevant_power", []),
                                "dominant_period": roi_result.get("dominant_period", 0),
                                "dominant_power": roi_result.get("dominant_power", 0),
                            }
                    extended_results["fft"] = fft_data
                    if "sleep_phase_results" in results_data:
                        extended_results["sleep_phase_results"] = results_data[
                            "sleep_phase_results"
                        ]

                elif method_index == 2:  # Cosinor Analysis
                    cosinor_data = {}
                    roi_results = results_data.get("roi_results", {})
                    for roi_id, roi_result in roi_results.items():
                        if isinstance(roi_id, int):
                            best = roi_result.get("best_result", {})
                            period = roi_result.get("best_period", 24)
                            cosinor_data[roi_id] = {
                                period: {
                                    "mesor": best.get("mesor", 0),
                                    "amplitude": best.get("amplitude", 0),
                                    "peak_time": best.get("peak_time", 0),
                                    "r_squared": best.get("r_squared", 0),
                                    "p_value": best.get("p_value", 1),
                                    "significant": best.get("significant", False),
                                }
                            }
                    extended_results["cosinor"] = cosinor_data
                    if "sleep_phase_results" in results_data:
                        extended_results["sleep_phase_results"] = results_data[
                            "sleep_phase_results"
                        ]

                elif method_index == 3:  # ROI Similarity
                    extended_results["similarity"] = {
                        "matrix": results_data.get("correlation_matrix", []),
                        "roi_ids": results_data.get("roi_ids", []),
                    }

                elif method_index == 4:  # Coherence
                    extended_results["coherence_matrix"] = results_data.get(
                        "coherence_matrix", []
                    )
                    extended_results["coherence_roi_ids"] = results_data.get(
                        "roi_ids", []
                    )

                elif method_index == 5:  # Phase Clustering
                    phase_data = {}
                    roi_phases = results_data.get("roi_phases", {})
                    for roi_id, phase_info in roi_phases.items():
                        if isinstance(roi_id, int):
                            phase_data[roi_id] = {
                                "phase": phase_info.get("phase_radians", 0),
                                "amplitude": phase_info.get("amplitude", 0),
                            }
                    extended_results["phase"] = phase_data

            # Collect analysis parameters
            analysis_params = {
                "core": {
                    "frame_interval": (
                        self.frame_interval.value()
                        if hasattr(self, "frame_interval")
                        else None
                    ),
                    "bin_size_seconds": (
                        self.bin_size_seconds.value()
                        if hasattr(self, "bin_size_seconds")
                        else None
                    ),
                },
                "extended": {
                    "min_period_hours": (
                        self.fisher_min_period.value()
                        if hasattr(self, "fisher_min_period")
                        else None
                    ),
                    "max_period_hours": (
                        self.fisher_max_period.value()
                        if hasattr(self, "fisher_max_period")
                        else None
                    ),
                    "significance_level": (
                        self.fisher_significance.value()
                        if hasattr(self, "fisher_significance")
                        else None
                    ),
                    "analysis_method": (
                        self.fisher_method_combo.currentText()
                        if hasattr(self, "fisher_method_combo")
                        else None
                    ),
                },
            }

            # Collect metadata
            metadata = {
                "saved_from": "napari-hdf5-activity comprehensive analysis",
                "save_timestamp": datetime.now().isoformat(),
                "n_rois": len(self.merged_results),
                "source_file": (
                    os.path.basename(self.file_path)
                    if hasattr(self, "file_path") and self.file_path
                    else None
                ),
            }

            # Save using comprehensive save function
            success = save_comprehensive_results(
                file_path=file_path,
                core_results=core_results,
                extended_results=extended_results,
                analysis_params=analysis_params,
                metadata=metadata,
            )

            if success:
                # Build summary message
                summary_lines = [
                    "Successfully saved comprehensive results to HDF5!",
                    "",
                    f"File: {file_path}",
                    f"ROIs saved: {len(self.merged_results)}",
                    f"ROI IDs: {sorted(self.merged_results.keys())}",
                    "",
                    "Saved Data:",
                    f"  • Movement data: {len(self.merged_results)} ROIs",
                ]

                if hasattr(self, "fraction_data") and self.fraction_data:
                    summary_lines.append(
                        f"  • Fraction data: {len(self.fraction_data)} ROIs"
                    )

                if hasattr(self, "roi_baseline_means") and self.roi_baseline_means:
                    summary_lines.append(
                        f"  • Thresholds: {len(self.roi_baseline_means)} ROIs"
                    )

                if extended_results:
                    summary_lines.append(
                        f"  • Extended analysis: {extended_results['method_name']}"
                    )

                summary_lines.extend(
                    [
                        "",
                        "You can reload these results anytime using 'Load Results from HDF5'.",
                        "All data is available for post-hoc cycle/period analysis.",
                    ]
                )

                self._log_message(
                    f"✓ Successfully saved comprehensive results for {len(self.merged_results)} ROIs"
                )
                self.fisher_results_text.setPlainText("\n".join(summary_lines))
            else:
                raise Exception("Save operation returned failure status")

        except Exception as e:
            self.fisher_results_text.setPlainText(
                f"ERROR saving results to HDF5:\n\n{str(e)}"
            )
            self._log_message(f"ERROR saving results: {e}")
            import traceback

            traceback.print_exc()

    def run_fisher_analysis(self):
        """Run rhythmic pattern analysis on movement data using selected method."""
        # Check if we have analysis results
        if not hasattr(self, "fraction_data") or not self.fraction_data:
            self.fisher_results_text.setPlainText(
                "ERROR: No analysis results available.\n\n"
                "Please run the main analysis first (Analysis tab) before "
                "attempting rhythmic pattern detection."
            )
            self._log_message(
                "⚠️ Rhythmic pattern analysis requires movement data from main analysis"
            )
            return

        # Get selected method
        method_index = self.fisher_method_combo.currentIndex()
        method_name = self.fisher_method_combo.currentText()

        self._log_message(f"Starting {method_name} analysis...")
        self.fisher_results_text.setPlainText("Running analysis...\n")

        try:
            # Get parameters
            min_period = self.fisher_min_period.value()
            max_period = self.fisher_max_period.value()
            significance = self.fisher_significance.value()
            sampling_interval = self.frame_interval.value()

            # Select data source based on user choice
            data_source_index = self.data_source_combo.currentIndex()
            if data_source_index == 0:
                # Fraction movement (0-1)
                if not hasattr(self, "fraction_data") or not self.fraction_data:
                    self.fisher_results_text.setPlainText(
                        "ERROR: No fraction movement data available.\n\n"
                        "Please run the main analysis first."
                    )
                    self._log_message(
                        "⚠️ No fraction movement data available for rhythmic pattern analysis"
                    )
                    return
                source_data = self.fraction_data
                data_type_name = "Fraction Movement (0-1)"
            elif data_source_index == 1:
                # Raw movement data
                if not hasattr(self, "merged_results") or not self.merged_results:
                    self.fisher_results_text.setPlainText(
                        "ERROR: No movement data available.\n\n"
                        "Please run the main analysis first."
                    )
                    self._log_message(
                        "⚠️ No movement data available for rhythmic pattern analysis"
                    )
                    return
                source_data = self.merged_results
                data_type_name = "Raw Intensity (continuous)"
            else:
                # Normalized movement (min/max per ROI, comparable to literature)
                if not hasattr(self, "merged_results") or not self.merged_results:
                    self.fisher_results_text.setPlainText(
                        "ERROR: No movement data available.\n\n"
                        "Please run the main analysis first."
                    )
                    self._log_message(
                        "⚠️ No movement data available for rhythmic pattern analysis"
                    )
                    return
                from ._calc import bin_and_normalize_movement

                norm_bin_size = (
                    self.analysis_bin_size.value()
                    if hasattr(self, "analysis_bin_size")
                    else self.bin_size_seconds.value()
                )
                source_data = bin_and_normalize_movement(
                    self.merged_results, norm_bin_size
                )
                data_type_name = "Normalized Movement (min/max, 0-1)"

            # Get bin sizes
            original_bin_size = self.bin_size_seconds.value()
            analysis_bin_size = self.analysis_bin_size.value()

            self._log_message(f"  Data source: {data_type_name}")
            self._log_message(f"  Original bin size: {original_bin_size}s")
            self._log_message(f"  Analysis bin size: {analysis_bin_size}s")
            self._log_message(
                f"  Detecting periods: {min_period:.1f} - {max_period:.1f} hours"
            )

            # Apply re-binning if needed
            # (Normalized Movement already binned at analysis_bin_size)
            if data_source_index == 2:
                bin_size = norm_bin_size
                self._log_message(
                    f"  Normalized Movement binned at {bin_size}s"
                )
            elif analysis_bin_size > original_bin_size:
                self._log_message(
                    f"  Re-binning data: {original_bin_size}s → {analysis_bin_size}s"
                )
                source_data = self._rebin_timeseries_data(
                    source_data, analysis_bin_size, original_bin_size
                )
                bin_size = analysis_bin_size
            else:
                bin_size = original_bin_size

            # Check if cycle/time range selection is enabled
            analysis_data = source_data
            if (
                hasattr(self, "enable_cycle_selection")
                and self.enable_cycle_selection.isChecked()
            ):
                start_hours = self.cycle_start_time.value()
                end_hours = self.cycle_end_time.value()

                # Convert hours to seconds for filtering
                start_time = start_hours * 3600.0
                end_time = end_hours * 3600.0

                # Filter data to time range
                filtered_data = {}
                for roi_id, data in source_data.items():
                    filtered = [
                        (t, v) for t, v in data
                        if start_time <= t <= end_time
                    ]
                    if filtered:
                        filtered_data[roi_id] = filtered
                analysis_data = filtered_data

                self._log_message(
                    f"  ✓ Time range filter applied: {start_hours:.1f}h - {end_hours:.1f}h"
                )
                self._log_message(
                    f"  Analyzing {len(analysis_data)} ROIs in selected time window"
                )
            else:
                self._log_message("  Using full recording for analysis")

            # Route to appropriate analysis method
            if method_index == 0:  # Chi² Periodogram
                results, summary = self._run_fisher_method(
                    min_period,
                    max_period,
                    significance,
                    sampling_interval,
                    bin_size,
                    analysis_data,
                )
            elif method_index == 1:  # FFT Power Spectrum
                results, summary = self._run_fft_method(
                    min_period,
                    max_period,
                    significance,
                    sampling_interval,
                    bin_size,
                    analysis_data,
                )
            elif method_index == 2:  # Cosinor Analysis
                self._log_message(
                    f"  Using {data_type_name} (with rebinning/time range if set)"
                )
                results, summary = self._run_cosinor_method(
                    min_period,
                    max_period,
                    significance,
                    sampling_interval,
                    bin_size,
                    analysis_data,
                )
            elif method_index == 3:  # ROI Similarity Matrix
                results, summary = self._run_similarity_method(
                    sampling_interval, bin_size, analysis_data
                )
            elif method_index == 4:  # Coherence Analysis
                results, summary = self._run_coherence_method(
                    sampling_interval, bin_size, analysis_data
                )
            elif method_index == 5:  # Phase Clustering
                results, summary = self._run_phase_clustering_method(
                    sampling_interval, bin_size, analysis_data
                )
            else:
                raise ValueError(f"Unknown method index: {method_index}")

            # Calculate sleep phase if enabled (for methods that support it)
            # Note: Cosinor (method_index=2) excluded because it requires continuous
            # sinusoidal data, not binary sleep states
            sleep_results = None
            sleep_summary = ""
            sleep_source_name = ""
            if self.chk_calculate_sleep_phase.isChecked() and method_index in [
                0,
                1,
            ]:  # Fisher, FFT only (not Cosinor)
                # Determine sleep data source from user selection
                sleep_source_index = (
                    self.sleep_source_combo.currentIndex()
                    if hasattr(self, "sleep_source_combo")
                    else 1
                )

                sleep_analysis_data = None
                if sleep_source_index == 0:
                    # Quiescence data (same temporal resolution as activity)
                    if hasattr(self, "quiescence_data") and self.quiescence_data:
                        total_points = sum(
                            sum(1 for t, v in data if v == 1)
                            for data in self.quiescence_data.values()
                        )
                        sleep_source_name = "quiescence_data (binary rest state)"
                        self._log_message(
                            f"  Analyzing sleep rhythms from quiescence_data "
                            f"({total_points} quiescence points, same resolution as activity)..."
                        )
                        sleep_analysis_data = self.quiescence_data
                    else:
                        self._log_message(
                            "  WARNING: No quiescence data available. "
                            "Run main analysis first."
                        )
                else:
                    # Sleep data (≥8min sustained quiescence)
                    if hasattr(self, "sleep_data") and self.sleep_data:
                        sleep_threshold = self.sleep_threshold_minutes.value()
                        total_points = sum(
                            sum(1 for t, v in data if v == 1)
                            for data in self.sleep_data.values()
                        )
                        sleep_source_name = (
                            f"sleep_data (≥{sleep_threshold}min sustained quiescence)"
                        )
                        self._log_message(
                            f"  Analyzing sleep rhythms from sleep_data "
                            f"(≥{sleep_threshold}min, {total_points} sleep points, "
                            f"low-pass filtered)..."
                        )
                        sleep_analysis_data = self.sleep_data
                    elif hasattr(self, "quiescence_data") and self.quiescence_data:
                        # Fallback to quiescence data
                        total_points = sum(
                            sum(1 for t, v in data if v == 1)
                            for data in self.quiescence_data.values()
                        )
                        sleep_source_name = "quiescence_data (fallback)"
                        self._log_message(
                            f"  No sleep_data available, falling back to quiescence_data "
                            f"({total_points} quiescence points)..."
                        )
                        sleep_analysis_data = self.quiescence_data
                    else:
                        self._log_message(
                            "  WARNING: No sleep/quiescence data available. "
                            "Run main analysis first."
                        )

                if sleep_analysis_data:
                    # Apply same time range filter if enabled
                    if (
                        hasattr(self, "enable_cycle_selection")
                        and self.enable_cycle_selection.isChecked()
                    ):
                        start_hours = self.cycle_start_time.value()
                        end_hours = self.cycle_end_time.value()
                        start_time = start_hours * 3600.0
                        end_time = end_hours * 3600.0
                        filtered_sleep = {}
                        for roi_id, data in sleep_analysis_data.items():
                            filtered = [
                                (t, v) for t, v in data
                                if start_time <= t <= end_time
                            ]
                            if filtered:
                                filtered_sleep[roi_id] = filtered
                        sleep_analysis_data = filtered_sleep

                    # Run the same analysis on sleep data
                    if method_index == 0:  # Chi² Periodogram
                        sleep_results, sleep_summary = self._run_fisher_method(
                            min_period,
                            max_period,
                            significance,
                            sampling_interval,
                            bin_size,
                            sleep_analysis_data,
                        )
                    elif method_index == 1:  # FFT Power Spectrum
                        sleep_results, sleep_summary = self._run_fft_method(
                            min_period,
                            max_period,
                            significance,
                            sampling_interval,
                            bin_size,
                            sleep_analysis_data,
                        )

                    # Log sleep analysis results
                    if sleep_results:
                        # Handle both structures: direct ROI keys (Fisher/FFT) or nested roi_results (Cosinor)
                        if "roi_results" in sleep_results:
                            sleep_roi_count = len(
                                [
                                    k
                                    for k in sleep_results["roi_results"].keys()
                                    if isinstance(k, int)
                                ]
                            )
                        else:
                            sleep_roi_count = len(
                                [k for k in sleep_results.keys() if isinstance(k, int)]
                            )
                        self._log_message(
                            f"  ✓ Sleep rhythm analysis complete: {sleep_roi_count} ROIs"
                        )
                    else:
                        self._log_message(
                            "  ⚠️ Sleep rhythm analysis returned no results"
                        )

                    # Combine results
                    results["sleep_phase_results"] = sleep_results
                    summary = self._combine_activity_sleep_summary(
                        summary, sleep_summary, results, sleep_results,
                        sleep_source_name=sleep_source_name,
                    )

            # Store results + data context for plot functions
            self.fisher_analysis_results = results
            self.current_fisher_method = method_index
            self.fisher_data_type_name = data_type_name    # e.g. "Fraction Movement (0-1)"
            self.fisher_analysis_data = analysis_data      # the actual data that was analysed

            # Warn if period range was capped by recording duration
            if method_index in [0, 1]:
                for roi_id, roi_result in results.items():
                    if not isinstance(roi_id, int):
                        continue
                    peri = roi_result.get("periodogram", roi_result)
                    if peri.get("period_capped", False):
                        actual_max = peri.get("actual_max_period", max_period)
                        self._log_message(
                            f"  ⚠️ Max period capped at {actual_max:.1f}h "
                            f"(recording duration / 2). Your setting of {max_period:.1f}h "
                            f"requires a recording ≥ {max_period * 2:.0f}h."
                        )
                        break  # one warning is enough

            # Display results
            self.fisher_results_text.setPlainText(summary)

            # Create and display plot
            self._create_circadian_plot(results, method_index)

            # Optionally open actogram in a separate window
            if hasattr(self, "chk_show_actogram") and self.chk_show_actogram.isChecked():
                tau_hours = self._get_actogram_tau(results, method_index)
                self._run_actogram(analysis_data, bin_size, tau_hours)

            # Enable export button
            self.btn_export_fisher.setEnabled(True)

            self._log_message(f"✓ Rhythmic pattern analysis complete ({method_name})")

        except Exception as e:
            error_msg = f"ERROR during rhythmic pattern analysis:\n\n{str(e)}\n\nPlease check the console for details."
            self.fisher_results_text.setPlainText(error_msg)
            self._log_message(f"❌ Rhythmic pattern analysis failed: {e}")
            import traceback

            traceback.print_exc()

    # ------------------------------------------------------------------
    # Actogram
    # ------------------------------------------------------------------

    def _run_actogram(self, analysis_data: dict, bin_size_s: float, tau_hours: float):
        """Create a double-plotted actogram from fraction movement data.

        Each row covers one period τ.  Data is plotted twice so that a
        drifting (free-running) rhythm appears as a diagonal band.

        Parameters
        ----------
        analysis_data : dict
            {roi_id: [(time_s, value), ...]}  – fraction movement data.
        bin_size_s : float
            Bin duration in seconds (spacing between successive data points).
        tau_hours : float
            Row period in hours (τ).  24 h for entrained, free-running τ from Chi².
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import io
            from qtpy.QtGui import QPixmap

            if not analysis_data:
                self.fisher_results_text.setPlainText("No data available for actogram.")
                return

            tau_s = tau_hours * 3600.0
            show_lighting = (
                hasattr(self, "actogram_chk_show_lighting")
                and self.actogram_chk_show_lighting.isChecked()
            )
            zt_axis = (
                hasattr(self, "actogram_chk_zt_axis")
                and self.actogram_chk_zt_axis.isChecked()
            )

            # ---- derive light-period rectangles from LED data ----
            light_periods = []  # list of (t_on_s, t_off_s) absolute seconds
            led_data = getattr(self, "led_data", None)
            if show_lighting and led_data and isinstance(led_data, dict) and "times" in led_data and "white_powers" in led_data:
                times_led = np.array(led_data["times"], dtype=float)
                wpow = np.array(led_data["white_powers"], dtype=float)
                light_on = wpow > 0
                # Detect rising / falling edges
                edges = np.diff(light_on.astype(int))
                rising  = np.where(edges ==  1)[0] + 1
                falling = np.where(edges == -1)[0] + 1
                # Handle starts/ends in light
                if light_on[0]:
                    rising = np.concatenate([[0], rising])
                if light_on[-1]:
                    falling = np.concatenate([falling, [len(light_on)]])
                for r, f in zip(rising, falling):
                    light_periods.append((float(times_led[r]), float(times_led[min(f, len(times_led)-1)])))

            # ---- integer ROIs only ----
            roi_ids = sorted(k for k in analysis_data.keys() if isinstance(k, int))
            n_rois = len(roi_ids)
            if n_rois == 0:
                self.fisher_results_text.setPlainText("No ROI data for actogram.")
                return

            roi_colors = getattr(self, "roi_colors", {})

            # ---- layout: up to 3 columns ----
            n_cols = min(3, n_rois)
            n_rows_fig = (n_rois + n_cols - 1) // n_cols

            fig_w = max(6, n_cols * 5)
            fig_h = max(4, n_rows_fig * 6)
            from matplotlib.figure import Figure as _Figure
            fig = _Figure(figsize=(fig_w, fig_h), layout="constrained")
            axes_grid = fig.subplots(n_rows_fig, n_cols, squeeze=False)
            fig.suptitle(f"Double-Plotted Actogram  (τ = {tau_hours:.1f} h)", fontsize=13, fontweight="bold")

            summary_lines = [
                "ACTOGRAM",
                "=" * 50,
                f"Row period τ: {tau_hours:.1f} h",
                f"Bin size:     {bin_size_s:.0f} s",
                f"ROIs:         {n_rois}",
                f"Light periods detected: {len(light_periods)}",
                "",
            ]

            for roi_idx, roi_id in enumerate(roi_ids):
                row_fig = roi_idx // n_cols
                col_fig = roi_idx % n_cols
                ax = axes_grid[row_fig][col_fig]

                data = analysis_data[roi_id]
                if not data:
                    ax.axis("off")
                    continue

                times_s = np.array([t for t, _ in data], dtype=float)
                values  = np.array([v for _, v in data], dtype=float)

                # Normalise values to [0, 1] per ROI for consistent row height
                vmax = values.max()
                if vmax > 0:
                    values_norm = values / vmax
                else:
                    values_norm = values

                t_start = times_s[0]
                t_end   = times_s[-1]
                total_duration_s = t_end - t_start

                n_day_rows = max(1, int(np.ceil(total_duration_s / tau_s)))

                # ROI colour for activity bars
                c = roi_colors.get(roi_id, "#2c3e50")

                for r in range(n_day_rows):
                    row_start_s = t_start + r * tau_s
                    row_end_s   = row_start_s + 2.0 * tau_s

                    y_bottom = n_day_rows - r - 1  # inverted: row 0 → top
                    y_top    = y_bottom + 1

                    # --- light/dark background (same style as results tab) ---
                    if light_periods:
                        # Dark phase: gray across full row
                        ax.add_patch(mpatches.Rectangle(
                            (0.0, y_bottom), 2.0 * tau_hours, 1.0,
                            color="gray", alpha=0.2, zorder=0, linewidth=0,
                        ))
                        # Light phase: yellow overlay
                        for t_on, t_off in light_periods:
                            x0 = max((t_on  - row_start_s) / 3600.0, 0.0)
                            x1 = min((t_off - row_start_s) / 3600.0, 2.0 * tau_hours)
                            if x1 > x0:
                                ax.add_patch(mpatches.Rectangle(
                                    (x0, y_bottom), x1 - x0, 1.0,
                                    color="yellow", alpha=0.2, zorder=0, linewidth=0,
                                ))

                    # --- activity fill_between ---
                    mask = (times_s >= row_start_s) & (times_s < row_end_s)
                    if mask.any():
                        x_vals = (times_s[mask] - row_start_s) / 3600.0
                        y_vals = y_bottom + values_norm[mask]
                        ax.fill_between(
                            x_vals, y_bottom, y_vals,
                            color=c, linewidth=0, alpha=0.85, zorder=2,
                        )

                    # --- row separator line ---
                    ax.axhline(y_bottom, color="#cccccc", lw=0.4, zorder=1)

                # ---- τ midline (divides the two halves) ----
                ax.axvline(tau_hours, color="#888888", lw=0.8, ls="--", zorder=3)

                ax.set_xlim(0, 2.0 * tau_hours)
                ax.set_ylim(0, n_day_rows)

                # Y-axis: day numbers (inverted display — row 0 at top)
                tick_positions = [n_day_rows - r - 0.5 for r in range(n_day_rows)]
                tick_labels    = [f"Day {r + 1}" for r in range(n_day_rows)]
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=7)

                # X-axis
                x_ticks = np.arange(0, 2.0 * tau_hours + 0.01, max(tau_hours / 4, 1.0))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([f"{v:.0f}" for v in x_ticks], fontsize=7)
                ax.set_xlabel("ZT (h)" if zt_axis else "Time (h)", fontsize=8)

                ax.set_title(f"ROI {roi_id}", fontsize=9, fontweight="bold")
                ax.tick_params(axis="both", labelsize=7)

                # Summary per ROI
                summary_lines.append(f"ROI {roi_id}: {n_day_rows} row(s), {len(times_s)} data points")

            # Hide unused axes
            for idx in range(n_rois, n_rows_fig * n_cols):
                r, c = divmod(idx, n_cols)
                axes_grid[r][c].axis("off")

            # ---- store figure; user opens it via "Show Actogram" button ----
            self.fisher_actogram_figure = fig
            self.fisher_actogram_tau = tau_hours
            if hasattr(self, "btn_show_actogram"):
                self.btn_show_actogram.setEnabled(True)
            self._log_message(f"✓ Actogram ready — click 'Show Actogram' to view ({n_rois} ROI(s), τ = {tau_hours:.1f} h)")

        except Exception as e:
            self._log_message(f"❌ Actogram failed: {e}")
            import traceback
            traceback.print_exc()

    def _get_actogram_tau(self, results: dict, method_index: int) -> float:
        """Determine actogram row period τ.

        For Cosinor (method_index=2): use the median best-fit period across all ROIs.
        For all other methods: use the actogram_period spinbox value.
        """
        if method_index == 2:
            # Cosinor results: roi_results[roi_id]['best_period']
            roi_results = results.get("roi_results", {})
            periods = [
                v.get("best_period")
                for v in roi_results.values()
                if isinstance(v, dict) and v.get("best_period") is not None
            ]
            if periods:
                import numpy as np
                tau = float(np.median(periods))
                self._log_message(f"  Actogram τ auto-detected from Cosinor fit: {tau:.2f} h")
                return tau
        return self.actogram_period.value() if hasattr(self, "actogram_period") else 24.0

    def _open_stored_actogram(self):
        """Open the previously computed actogram figure in a popup window."""
        fig = getattr(self, "fisher_actogram_figure", None)
        tau = getattr(self, "fisher_actogram_tau", 24.0)
        if fig is None:
            self._log_message("⚠️ No actogram available — run analysis with 'Show Actogram' checked first.")
            return
        self._open_actogram_window(fig, tau)

    def _open_actogram_window(self, fig, tau_hours: float):
        """Open an actogram figure in a resizable popup dialog."""
        try:
            from qtpy.QtWidgets import (
                QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            )
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
                NavigationToolbar2QT as NavigationToolbar,
            )

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Actogram  (τ = {tau_hours:.1f} h)")
            dialog.resize(1200, 800)

            layout = QVBoxLayout()
            dialog.setLayout(layout)

            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(600, 400)

            toolbar = NavigationToolbar(canvas, dialog)
            toolbar.setStyleSheet(
                "QToolBar { background-color: #f0f0f0; border: none; padding: 2px; }"
                "QToolButton { background-color: #f0f0f0; border: 1px solid #ccc;"
                "  border-radius: 3px; padding: 3px; margin: 1px; color: #222; }"
                "QToolButton:hover { background-color: #dde8f5; border-color: #4a90d9; }"
                "QToolButton:checked { background-color: #c8ddf5; border-color: #4a90d9; }"
            )
            layout.addWidget(toolbar)
            layout.addWidget(canvas)

            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_save = QPushButton("Save Plot...")
            btn_save.clicked.connect(lambda: self._save_plot_from_dialog(canvas))
            btn_layout.addWidget(btn_save)
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.close)
            btn_layout.addWidget(btn_close)
            layout.addLayout(btn_layout)

            dialog.exec_()

        except Exception as e:
            self._log_message(f"⚠️ Could not open actogram window: {e}")

    def _run_fisher_method(
        self,
        min_period,
        max_period,
        significance,
        sampling_interval,
        bin_size,
        fraction_data=None,
    ):
        """Run Chi² Periodogram analysis."""
        from ._fisher_analysis import (
            analyze_roi_circadian_patterns,
            generate_circadian_summary,
        )

        # Use provided fraction_data or default to self.fraction_data
        data_to_analyze = (
            fraction_data if fraction_data is not None else self.fraction_data
        )

        results = analyze_roi_circadian_patterns(
            data_to_analyze,  # Use fraction_data (proportion 0-1)
            sampling_interval=sampling_interval,
            min_period_hours=min_period,
            max_period_hours=max_period,
            significance_level=significance,
            phase_threshold=0.5,  # Keep for backward compatibility, unused
            bin_size_seconds=bin_size,
        )

        summary = generate_circadian_summary(results)
        return results, summary

    def _create_inverted_activity_data(self, activity_data):
        """
        Create inverted activity data for sleep phase analysis.
        For fraction movement (0-1): inverted = 1 - value
        For raw movement: inverted = max - value
        """
        inverted_data = {}

        for roi, data_list in activity_data.items():
            if not data_list:
                inverted_data[roi] = []
                continue

            # Extract values to determine if this is fraction (0-1) or raw data
            values = [v for t, v in data_list]
            max_val = max(values) if values else 1.0

            inverted_list = []
            for timestamp, value in data_list:
                if max_val <= 1.0:
                    # Fraction movement (0-1): invert as 1 - value
                    inverted_value = 1.0 - value
                else:
                    # Raw movement: invert as max - value
                    inverted_value = max_val - value
                inverted_list.append((timestamp, inverted_value))

            inverted_data[roi] = inverted_list

        return inverted_data

    def _combine_activity_sleep_summary(
        self, activity_summary, sleep_summary, activity_results, sleep_results,
        sleep_source_name="",
    ):
        """Combine activity and sleep phase summaries into one comprehensive report."""
        combined = []

        # Get data source name from combo box
        data_source_index = self.data_source_combo.currentIndex()
        if data_source_index == 0:
            data_source_name = "Fraction Movement"
        elif data_source_index == 1:
            data_source_name = "Raw Intensity"
        else:
            data_source_name = "Normalized Movement (min/max)"

        combined.append("=" * 60)
        combined.append("ACTIVITY & SLEEP RHYTHM ANALYSIS RESULTS")
        combined.append("=" * 60)
        combined.append("")

        # Activity Phase Section
        combined.append("─" * 40)
        combined.append(f"ACTIVITY RHYTHMS (from {data_source_name})")
        combined.append("─" * 40)

        # Extract key info from activity results
        # Handle both structures: direct ROI keys (Fisher) or nested "roi_results" (Cosinor)
        if "roi_results" in activity_results:
            act_roi_data = activity_results["roi_results"]
        else:
            # Fisher results have ROI IDs as direct keys
            act_roi_data = {
                k: v for k, v in activity_results.items() if isinstance(k, int)
            }

        if act_roi_data:
            roi_items = {k: v for k, v in act_roi_data.items() if isinstance(k, int)}
            for roi_id, roi_data in sorted(roi_items.items()):
                # Handle nested structure for different analysis methods
                if "periodogram" in roi_data:
                    # Fisher/FFT structure
                    periodogram = roi_data.get("periodogram", {})
                    period = periodogram.get("dominant_period", "N/A")
                    significant = periodogram.get("is_significant", False)
                    z_score = periodogram.get(
                        "dominant_z_score", periodogram.get("max_power", "N/A")
                    )
                    peak_time = (
                        "N/A (use Cosinor)"  # Fisher doesn't calculate peak time
                    )
                elif "best_result" in roi_data:
                    # Cosinor structure: best_period at top, details in best_result
                    period = roi_data.get("best_period", "N/A")
                    best_result = roi_data.get("best_result", {})
                    peak_time = best_result.get("peak_time", "N/A")
                    significant = best_result.get("significant", False)
                    z_score = None
                elif "dominant_period" in roi_data:
                    # FFT direct structure
                    period = roi_data.get("dominant_period", "N/A")
                    peak_time = "N/A"
                    significant = roi_data.get("is_significant", False)
                    z_score = roi_data.get("dominant_power", None)
                else:
                    continue

                sig_str = "✓" if significant else "✗"

                if isinstance(period, (int, float)):
                    period_str = f"{period:.1f}h"
                else:
                    period_str = str(period)

                if isinstance(peak_time, (int, float)):
                    peak_time_str = f"{peak_time:.1f}h"
                else:
                    peak_time_str = str(peak_time)

                if z_score is not None and isinstance(z_score, (int, float)):
                    combined.append(
                        f"  ROI {roi_id}: Period={period_str}, Z={z_score:.1f} {sig_str}"
                    )
                else:
                    combined.append(
                        f"  ROI {roi_id}: Period={period_str}, Peak Time={peak_time_str} {sig_str}"
                    )

        combined.append("")

        # Sleep Phase Section
        if not sleep_source_name:
            sleep_threshold = (
                self.sleep_threshold_minutes.value()
                if hasattr(self, "sleep_threshold_minutes")
                else 8
            )
            sleep_source_name = f"sleep_data (≥{sleep_threshold}min quiescence)"
        combined.append("─" * 40)
        combined.append(f"SLEEP RHYTHMS (from {sleep_source_name})")
        combined.append("─" * 40)

        # Handle both structures for sleep results
        if sleep_results:
            if "roi_results" in sleep_results:
                slp_roi_data = sleep_results["roi_results"]
            else:
                slp_roi_data = {
                    k: v for k, v in sleep_results.items() if isinstance(k, int)
                }

            sleep_roi_items = {
                k: v for k, v in slp_roi_data.items() if isinstance(k, int)
            }
            for roi_id, roi_data in sorted(sleep_roi_items.items()):
                # Handle nested structure for different analysis methods
                if "periodogram" in roi_data:
                    # Fisher/FFT structure
                    periodogram = roi_data.get("periodogram", {})
                    period = periodogram.get("dominant_period", "N/A")
                    significant = periodogram.get("is_significant", False)
                    z_score = periodogram.get(
                        "dominant_z_score", periodogram.get("max_power", "N/A")
                    )
                    sleep_phase = "N/A"
                elif "best_result" in roi_data:
                    # Cosinor structure: best_period at top, details in best_result
                    period = roi_data.get("best_period", "N/A")
                    best_result = roi_data.get("best_result", {})
                    sleep_phase = best_result.get("peak_time", "N/A")
                    significant = best_result.get("significant", False)
                    z_score = None
                elif "dominant_period" in roi_data:
                    # FFT direct structure
                    period = roi_data.get("dominant_period", "N/A")
                    sleep_phase = "N/A"
                    significant = roi_data.get("is_significant", False)
                    z_score = roi_data.get("dominant_power", None)
                else:
                    continue

                sig_str = "✓" if significant else "✗"

                if isinstance(period, (int, float)):
                    period_str = f"{period:.1f}h"
                else:
                    period_str = str(period)

                if z_score is not None and isinstance(z_score, (int, float)):
                    combined.append(
                        f"  ROI {roi_id}: Period={period_str}, Z={z_score:.1f} {sig_str}"
                    )
                else:
                    if isinstance(sleep_phase, (int, float)):
                        sleep_str = f"{sleep_phase:.1f}h"
                    else:
                        sleep_str = str(sleep_phase)
                    combined.append(
                        f"  ROI {roi_id}: Period={period_str}, Sleep Phase={sleep_str} {sig_str}"
                    )

        combined.append("")

        # Summary comparison
        combined.append("─" * 40)
        combined.append("PERIOD COMPARISON (Activity vs Sleep)")
        combined.append("─" * 40)

        # Use previously extracted ROI data
        if act_roi_data and sleep_results:
            # Get sleep ROI data (handle both structures)
            if "roi_results" in sleep_results:
                slp_compare_data = sleep_results["roi_results"]
            else:
                slp_compare_data = {
                    k: v for k, v in sleep_results.items() if isinstance(k, int)
                }

            act_roi_keys = [k for k in act_roi_data.keys() if isinstance(k, int)]
            for roi_id in sorted(act_roi_keys):
                act_data = act_roi_data.get(roi_id, {})
                slp_data = slp_compare_data.get(roi_id, {})

                # Handle nested structure for Fisher/FFT
                if "periodogram" in act_data:
                    act_period = act_data.get("periodogram", {}).get("dominant_period")
                else:
                    act_period = act_data.get("dominant_period")

                if "periodogram" in slp_data:
                    slp_period = slp_data.get("periodogram", {}).get("dominant_period")
                else:
                    slp_period = slp_data.get("dominant_period")

                # Also try to get peak_time for Cosinor
                act_phase = act_data.get("peak_time")
                slp_phase = slp_data.get("peak_time")

                if act_period is not None and slp_period is not None:
                    if act_phase is not None and slp_phase is not None:
                        # Cosinor: show fitted peak time comparison
                        diff = abs(act_phase - slp_phase)
                        if diff > 12:
                            diff = 24 - diff
                        combined.append(
                            f"  ROI {roi_id}: Act Peak={act_phase:.1f}h, "
                            f"Sleep Peak={slp_phase:.1f}h, Δ={diff:.1f}h"
                        )
                    else:
                        # Fisher/FFT: show period comparison
                        combined.append(
                            f"  ROI {roi_id}: Act Period={act_period:.1f}h, "
                            f"Sleep Period={slp_period:.1f}h"
                        )

        combined.append("")
        combined.append("=" * 60)
        combined.append("Legend: ✓ = significant rhythm, ✗ = not significant")
        combined.append("=" * 60)

        return "\n".join(combined)

    def _run_fft_method(
        self,
        min_period,
        max_period,
        significance,
        sampling_interval,
        bin_size,
        fraction_data=None,
    ):
        """Run FFT Power Spectrum analysis with significance testing."""
        from ._circadian_fft import analyze_roi_fft_patterns, generate_fft_summary

        # Use provided fraction_data or default to self.fraction_data
        data_to_analyze = (
            fraction_data if fraction_data is not None else self.fraction_data
        )

        results = analyze_roi_fft_patterns(
            data_to_analyze,  # Use fraction_data (proportion 0-1)
            sampling_interval=sampling_interval,
            min_period_hours=min_period,
            max_period_hours=max_period,
            bin_size_seconds=bin_size,
            window="hann",
            significance_level=significance,
            n_permutations=1000,  # Use 1000 permutations for good statistical power
        )

        summary = generate_fft_summary(results)
        return results, summary

    def _run_cosinor_method(
        self,
        min_period,
        max_period,
        significance,
        sampling_interval,
        bin_size,
        data=None,
    ):
        """Run Cosinor analysis on any time-series data (fraction movement or raw intensity)."""
        from ._cosinor_analysis import (
            multi_period_cosinor,
            population_cosinor,
        )

        # Prepare test periods - test common periods in the specified range
        test_periods = []
        if min_period <= 12 and max_period >= 12:
            test_periods.append(12.0)  # 12h ultradian
        if min_period <= 24 and max_period >= 24:
            test_periods.append(24.0)  # 24h circadian
        if min_period <= 30 and max_period >= 30:
            test_periods.append(30.0)  # 30h infradian

        # Also add min, max, and midpoint
        midpoint = (min_period + max_period) / 2
        for p in [min_period, midpoint, max_period]:
            if p not in test_periods:
                test_periods.append(p)

        test_periods = sorted(test_periods)

        # Use provided data or fall back to fraction_data
        data_to_analyze = (
            data if data is not None else self.fraction_data
        )

        # Compute recording duration for cycle-count warnings / plot annotation
        recording_duration_h = 0.0
        for _dl in data_to_analyze.values():
            if _dl:
                _ts = [t for t, _ in _dl]
                recording_duration_h = max(
                    recording_duration_h, (max(_ts) - min(_ts)) / 3600.0
                )

        # Warn for every test period that yields < 2 complete cycles
        for _tp in test_periods:
            _n_cyc = recording_duration_h / _tp if _tp > 0 else 0
            if _n_cyc < 2.0:
                self._log_message(
                    f"⚠️ Cosinor: period {_tp:.1f} h → only {_n_cyc:.1f} complete cycle(s) "
                    f"in {recording_duration_h:.1f} h recording — result unreliable (need ≥ 2 cycles)"
                )

        # Analyze each ROI with cosinor
        roi_results = {}
        population_time_series = []
        population_timestamps = []

        for roi_id, data_list in data_to_analyze.items():
            if not data_list:
                continue

            # Use actual timestamps from the data — critical for correct period detection.
            # Passing timestamps=None would cause the cosinor to invent synthetic timestamps
            # using bin_size, which is wrong for raw intensity data (5s intervals ≠ 60s bins).
            timestamps_s = np.array([t for t, _ in data_list])  # seconds
            values = np.array([v for _, v in data_list])

            multi_result = multi_period_cosinor(
                time_series=values,
                timestamps=timestamps_s,
                test_periods=test_periods,
                sampling_interval=bin_size,
                alpha=significance,
            )

            roi_results[roi_id] = multi_result
            population_time_series.append(values)
            population_timestamps.append(timestamps_s)

        # Population-level cosinor for best period across all ROIs
        if population_time_series:
            # Find the most common best period
            best_periods = [
                r["best_period"] for r in roi_results.values() if "best_period" in r
            ]
            if best_periods:
                # Use median of best periods as population period
                population_period = float(np.median(best_periods))

                pop_result = population_cosinor(
                    time_series_list=population_time_series,
                    timestamps_list=population_timestamps,
                    period_hours=population_period,
                    sampling_interval=bin_size,
                    alpha=significance,
                )
            else:
                pop_result = {"error": "No valid ROI results"}
        else:
            pop_result = {"error": "No time series data"}

        # Combine results
        results = {
            "roi_results": roi_results,
            "population_result": pop_result,
            "test_periods": test_periods,
            "sampling_interval": sampling_interval,
            "recording_duration_h": recording_duration_h,
        }

        # Generate summary
        summary = self._generate_cosinor_summary(results)

        return results, summary

    def _generate_cosinor_summary(self, results):
        """Generate text summary for cosinor analysis results."""
        data_type_name = getattr(self, "fisher_data_type_name", "Fraction Movement (0-1)")
        lines = [
            "=" * 70,
            "COSINOR ANALYSIS - Circadian Rhythm Quantification",
            "=" * 70,
            "",
            f"Data source: {data_type_name}",
            "",
            "Cosinor analysis fits a cosine curve to the signal:",
            "  y(t) = MESOR + Amplitude × cos(2πt/τ + φ)",
            "",
            "Parameters:",
            "  • MESOR: Midline Estimating Statistic of Rhythm (mean level)",
            "  • Amplitude: Half the difference between peak and trough",
            "  • φ (phase angle): phase offset of the fitted cosine (radians)",
            "  • Peak Time: time from recording start to first fitted cosine peak",
            "    NOTE: not the biological acrophase without a known ZT reference",
            "  • Period (τ): Duration of one complete cycle",
            "",
        ]

        roi_results = results.get("roi_results", {})
        test_periods = results.get("test_periods", [])

        # Diagnostic checks for period range issues
        warnings = []
        boundary_count = 0
        best_periods = []

        for roi_id, roi_data in roi_results.items():
            best_result = roi_data.get("best_result", {})
            if "error" in best_result or not best_result.get("significant", False):
                continue

            best_period = best_result.get("period", 0)
            if best_period > 0:
                best_periods.append(best_period)

            # Check if best period is at boundary of test_periods
            if len(test_periods) > 0:
                min_test = min(test_periods)
                max_test = max(test_periods)

                # Check if best period matches boundary (exactly or very close)
                if best_period == min_test or best_period == max_test:
                    boundary_count += 1

        # Generate warnings
        n_significant = len(best_periods)
        if (
            boundary_count > n_significant * 0.3 and n_significant > 0
        ):  # More than 30% at boundaries
            warnings.append(
                f"⚠️  WARNING: {boundary_count}/{n_significant} ROIs have best-fit periods at test range boundaries.\n"
                f"   This suggests the period range may be too narrow.\n"
                f"   Consider expanding the test period range to capture true rhythms."
            )

        # Check if best periods cluster at extremes
        if len(best_periods) >= 3:
            best_periods_array = np.array(best_periods)
            min_best = best_periods_array.min()
            max_best = best_periods_array.max()

            if len(test_periods) > 0:
                test_min = min(test_periods)
                test_max = max(test_periods)

                # Suggest expanding range if periods cluster near boundaries
                if max_best > 12.0 and test_max < 24.0:
                    warnings.append(
                        f"ℹ️  INFO: Some best-fit periods exceed 12h (max: {max_best:.1f}h).\n"
                        f"   Consider testing 24h period for circadian analysis."
                    )
                elif min_best < 2.0 and test_min > 1.0:
                    warnings.append(
                        f"ℹ️  INFO: Some best-fit periods below 2h (min: {min_best:.1f}h).\n"
                        f"   Consider testing shorter periods (0.5-1h) for ultradian analysis."
                    )

        if warnings:
            lines.extend(warnings)
            lines.append("")

        lines.extend(
            [
                "=" * 70,
                "INDIVIDUAL ROI RESULTS",
                "=" * 70,
                "",
            ]
        )

        for roi_id in sorted(roi_results.keys()):
            roi_data = roi_results[roi_id]
            best_result = roi_data.get("best_result", {})

            if "error" in best_result:
                lines.extend([f"ROI {roi_id}:", f"  Error: {best_result['error']}", ""])
                continue

            # Check if period is at boundary
            best_period = best_result.get("period", 0)
            boundary_marker = ""
            if len(test_periods) > 0:
                min_test = min(test_periods)
                max_test = max(test_periods)
                if best_period == max_test:
                    boundary_marker = f" ⚠️ (at upper test boundary {max_test:.2f}h)"
                elif best_period == min_test:
                    boundary_marker = f" ⚠️ (at lower test boundary {min_test:.2f}h)"

            lines.extend(
                [
                    f"ROI {roi_id}:",
                    f"  Best-fit period: {best_period:.2f} hours{boundary_marker}",
                    f"  MESOR (mean level): {best_result.get('mesor', 0):.4f}",
                    f"  Amplitude: {best_result.get('amplitude', 0):.4f}",
                    f"  Peak Time (fitted cosine peak): {best_result.get('peak_time', 0):.2f} h from start",
                    f"  R²: {best_result.get('r_squared', 0):.3f}",
                    f"  p-value: {_fmt_p(best_result.get('p_value', 1))} {'***' if best_result.get('p_value', 1) < 0.001 else '**' if best_result.get('p_value', 1) < 0.01 else '*' if best_result.get('p_value', 1) < 0.05 else 'ns'}",
                    f"  Significant: {'YES' if best_result.get('significant', False) else 'NO'}",
                ]
            )

            # Add confidence intervals if available
            if "ci_amplitude" in best_result:
                ci_amp = best_result["ci_amplitude"]
                ci_pt = best_result.get("ci_peak_time", (float("nan"), float("nan")))
                lines.extend(
                    [
                        f"  95% CI Amplitude: [{ci_amp[0]:.4f}, {ci_amp[1]:.4f}]",
                        f"  95% CI Peak Time: [{ci_pt[0]:.2f}h, {ci_pt[1]:.2f}h]",
                    ]
                )

            lines.append("")

            # Show all tested periods
            all_results = roi_data.get("all_results", [])
            if all_results:
                lines.append("  Tested periods:")
                for res in all_results:
                    sig_marker = "✓" if res.get("significant", False) else " "
                    lines.append(
                        f"    [{sig_marker}] {res.get('test_period', 0):.1f}h: "
                        f"R²={res.get('r_squared', 0):.3f}, "
                        f"Amp={res.get('amplitude', 0):.4f}, "
                        f"p={_fmt_p(res.get('p_value', 1))}"
                    )
                lines.append("")

        # Population-level results
        pop_result = results.get("population_result", {})
        if "error" not in pop_result:
            lines.extend(
                [
                    "=" * 70,
                    "POPULATION-LEVEL COSINOR",
                    "=" * 70,
                    "",
                    f"Population MESOR: {pop_result.get('population_mesor', 0):.4f}",
                    f"Population Amplitude: {pop_result.get('population_amplitude', 0):.4f}",
                    f"Population Peak Time (circular mean): {pop_result.get('population_peak_time', 0):.2f} h from start",
                    f"Test period: {pop_result.get('period', 0):.2f} hours",
                    f"p-value: {_fmt_p(pop_result.get('p_value', 1))}",
                    f"Significant rhythm: {'YES' if pop_result.get('significant', False) else 'NO'}",
                    "",
                    f"Individual ROIs analyzed: {pop_result.get('n_individuals', 0)}",
                    f"ROIs with significant rhythm: {pop_result.get('n_significant', 0)} "
                    f"({pop_result.get('proportion_significant', 0)*100:.1f}%)",
                    "",
                ]
            )

        lines.extend(
            [
                "=" * 70,
                "INTERPRETATION GUIDE",
                "=" * 70,
                "",
                "MESOR:",
                "  The rhythm-adjusted mean activity level (baseline activity)",
                "",
                "Amplitude:",
                "  Larger amplitude = stronger rhythm (higher variation from mean)",
                "  Small amplitude suggests weak or no rhythmic pattern",
                "",
                "Acrophase:",
                "  Time of peak activity in the cycle",
                "  For 24h rhythm: 0h=start of recording, 12h=halfway through",
                "",
                "Significance (p-value):",
                "  p < 0.05: Significant rhythm detected",
                "  p < 0.01: Highly significant rhythm (**)",
                "  p < 0.001: Very highly significant rhythm (***)",
                "",
                "R² (goodness of fit):",
                "  >0.3: Strong rhythmic pattern",
                "  0.1-0.3: Moderate rhythm",
                "  <0.1: Weak or no rhythm",
                "",
            ]
        )

        return "\n".join(lines)

    def _run_similarity_method(self, sampling_interval, bin_size, fraction_data=None):
        """Run ROI Similarity analysis."""
        from ._circadian_similarity import (
            calculate_roi_correlation_matrix,
            hierarchical_clustering,
            generate_similarity_summary,
        )

        # Use provided fraction_data or default to self.fraction_data
        data_to_analyze = (
            fraction_data if fraction_data is not None else self.fraction_data
        )

        # Use half of max period as max lag (adaptive to period range)
        max_lag = self.fisher_max_period.value() / 2

        # Get significance level from UI
        significance = self.fisher_significance.value()

        correlation_results = calculate_roi_correlation_matrix(
            data_to_analyze,  # Use fraction_data (proportion 0-1)
            sampling_interval=sampling_interval,
            bin_size_seconds=bin_size,
            max_lag_hours=max_lag,
            significance_level=significance,
        )

        clustering_results = hierarchical_clustering(
            correlation_results["correlation_matrix"],
            correlation_results["roi_ids"],
            method="average",
        )

        correlation_results["clustering"] = clustering_results
        summary = generate_similarity_summary(correlation_results, clustering_results)

        return correlation_results, summary

    def _run_coherence_method(self, sampling_interval, bin_size, fraction_data=None):
        """Run Coherence analysis."""
        from ._circadian_coherence import (
            calculate_coherence_matrix,
            generate_coherence_summary,
        )

        # Use provided fraction_data or default to self.fraction_data
        data_to_analyze = (
            fraction_data if fraction_data is not None else self.fraction_data
        )

        # Use midpoint of period range from GUI instead of hardcoded 24h
        midpoint_period = (
            self.fisher_min_period.value() + self.fisher_max_period.value()
        ) / 2

        # Get significance level from UI
        significance = self.fisher_significance.value()

        results = calculate_coherence_matrix(
            data_to_analyze,  # Use fraction_data (proportion 0-1)
            sampling_interval=sampling_interval,
            bin_size_seconds=bin_size,
            target_period_hours=midpoint_period,
            significance_level=significance,
        )

        summary = generate_coherence_summary(results)
        return results, summary

    def _run_phase_clustering_method(
        self, sampling_interval, bin_size, fraction_data=None
    ):
        """Run Phase Clustering analysis."""
        from ._circadian_coherence import detect_phase_clusters

        # Use provided fraction_data or default to self.fraction_data
        data_to_analyze = (
            fraction_data if fraction_data is not None else self.fraction_data
        )

        # Use midpoint of period range from GUI instead of hardcoded 24h
        midpoint_period = (
            self.fisher_min_period.value() + self.fisher_max_period.value()
        ) / 2

        results = detect_phase_clusters(
            data_to_analyze,  # Use fraction_data (proportion 0-1)
            dominant_period_hours=midpoint_period,
            sampling_interval=sampling_interval,
            bin_size_seconds=bin_size,
        )

        # Generate summary
        summary_lines = [
            "=" * 60,
            "PHASE CLUSTERING ANALYSIS",
            "=" * 60,
            "",
            f"Total ROIs analyzed: {results['n_rois']}",
            f"Dominant period: {results['dominant_period_hours']:.1f} hours",
            "",
            "ROI Phase Clusters:",
            "",
        ]

        for cluster_name, roi_list in results["phase_clusters"].items():
            if roi_list:
                summary_lines.append(
                    f"  {cluster_name.replace('_', ' ').title()}: {len(roi_list)} ROIs"
                )
                summary_lines.append(f"    ROIs: {', '.join(map(str, roi_list))}")
                summary_lines.append("")

        summary_lines.append("")
        summary_lines.append("Individual ROI Phases:")
        summary_lines.append("")

        for roi_id, phase_info in sorted(results["roi_phases"].items()):
            summary_lines.append(
                f"  ROI {roi_id}: Peak activity at {phase_info['phase_hours']:.1f}h "
                f"(amplitude: {phase_info['amplitude']:.2f})"
            )

        summary_lines.append("")
        summary_lines.append("=" * 60)

        summary = "\n".join(summary_lines)
        return results, summary

    def _create_circadian_plot(self, results: Dict, method_index: int):
        """Create and display plot based on selected analysis method."""
        if method_index == 0:  # Chi² Periodogram
            self._create_fisher_plot(results)
        elif method_index == 1:  # FFT Power Spectrum
            self._create_fft_plot(results)
        elif method_index == 2:  # Cosinor Analysis
            self._create_cosinor_plot(results)
        elif method_index == 3:  # ROI Similarity Matrix
            self._create_similarity_plot(results)
        elif method_index == 4:  # Coherence Analysis
            self._create_coherence_plot(results)
        elif method_index == 5:  # Phase Clustering
            self._create_phase_cluster_plot(results)

    def _create_fft_plot(self, fft_results: Dict[int, Dict]):
        """Create FFT power spectrum plots."""
        try:
            import matplotlib.pyplot as plt
            from qtpy.QtGui import QPixmap
            import io

            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                self.fisher_plot_figure = None

            # Get data source for plot title
            data_source_index = self.data_source_combo.currentIndex()
            data_source = (
                "Fraction Movement" if data_source_index == 0 else "Raw Intensity"
            )

            # Check if sleep phase results are available
            sleep_results = fft_results.get("sleep_phase_results", None)
            has_sleep = sleep_results is not None and len(sleep_results) > 0

            # Filter to only ROI keys (integers)
            roi_only_results = {
                k: v for k, v in fft_results.items() if isinstance(k, int)
            }
            n_rois = len(roi_only_results)
            n_cols = min(3, n_rois)
            n_rows_per_section = (n_rois + n_cols - 1) // n_cols

            has_population = (
                n_rois >= 2
                and hasattr(self, "chk_cosinor_population")
                and self.chk_cosinor_population.isChecked()
            )

            # If we have sleep data, double the rows
            if has_sleep:
                total_rows = n_rows_per_section * 2
                fig_height = 3.5 * total_rows + 1.5
            else:
                total_rows = n_rows_per_section
                fig_height = 3.5 * total_rows + 0.5

            if has_population:
                fig_height += 4.0

            fig_width = 4 * n_cols
            from matplotlib.figure import Figure as _Figure
            from matplotlib.gridspec import GridSpec as _GridSpec
            fig = _Figure(figsize=(fig_width, fig_height))

            if has_population:
                _pop_h   = 2.8
                _gap_h   = 0.7
                _bot_pad = 0.25
                _pop_bot = _bot_pad / fig_height
                _pop_top = (_pop_h + _bot_pad) / fig_height
                _dat_bot = (_pop_h + _bot_pad + _gap_h) / fig_height
                gs = _GridSpec(total_rows, n_cols, figure=fig,
                               left=0.08, right=0.97,
                               top=0.96, bottom=_dat_bot,
                               hspace=0.55, wspace=0.35)
                gs_pop = _GridSpec(1, n_cols, figure=fig,
                                   left=0.08, right=0.97,
                                   top=_pop_top, bottom=_pop_bot,
                                   wspace=0.35)
            else:
                gs = _GridSpec(total_rows, n_cols, figure=fig,
                               left=0.08, right=0.97,
                               top=0.96, bottom=0.05,
                               hspace=0.55, wspace=0.35)
                gs_pop = None

            axes = np.array(
                [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                 for r in range(total_rows)]
            )
            ax_pop = fig.add_subplot(gs_pop[0, :]) if has_population else None

            # Check if max period was capped for any ROI
            cap_note = ""
            requested_max = self.fisher_max_period.value()
            for _roi_id, _roi_res in roi_only_results.items():
                _periods = _roi_res.get("relevant_periods", [])
                if len(_periods) > 0:
                    _actual_max = max(_periods)
                    if _actual_max < requested_max * 0.95:  # capped by >5%
                        cap_note = f"  (max period capped at {_actual_max:.1f}h = recording / 2)"
                    break

            if has_sleep:
                fig.suptitle(
                    f"FFT Power Spectrum  —  Data source: {data_source}{cap_note}{self._get_recording_start_str()}",
                    fontsize=11,
                    fontweight="bold",
                    y=0.99,
                )
            else:
                fig.suptitle(
                    f"FFT Power Spectrum  —  Activity from {data_source}{cap_note}{self._get_recording_start_str()}",
                    fontsize=11,
                    fontweight="bold",
                    y=0.99,
                )

            # Helper function to plot a section
            def plot_section(results_dict, start_row, section_label):
                roi_items = {
                    k: v for k, v in results_dict.items() if isinstance(k, int)
                }

                for idx, (roi_id, result) in enumerate(sorted(roi_items.items())):
                    row = start_row + (idx // n_cols)
                    col = idx % n_cols
                    ax = axes[row, col]

                    roi_color = (
                        self.roi_colors.get(roi_id, f"C{idx}")
                        if hasattr(self, "roi_colors")
                        else f"C{idx}"
                    )

                    if "error" in result:
                        ax.text(
                            0.5,
                            0.5,
                            f"ROI {roi_id}\n{result['error']}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        periods = result.get("relevant_periods", [])
                        power = result.get("relevant_power", [])

                        ax.plot(periods, power, color=roi_color, linewidth=1.5)
                        ax.set_xlabel("Period (h)", fontsize=9)
                        ax.set_ylabel("Power (a.u.)", fontsize=9)
                        ax.set_title(
                            f"ROI {roi_id} - {section_label}",
                            fontsize=9,
                            color=roi_color,
                            fontweight="bold",
                            pad=4,
                            loc="left",
                        )
                        ax.tick_params(axis="both", labelsize=8)
                        ax.grid(True, alpha=0.3)

                        # Per-ROI Y-axis scaling
                        if len(power) > 0:
                            roi_y_max = max(power) * 1.1
                            ax.set_ylim(0, roi_y_max)

                        if "dominant_period" in result:
                            dominant_period = result["dominant_period"]
                            dominant_power = result.get("dominant_power", 0)
                            ax.axvline(
                                x=dominant_period,
                                color=roi_color,
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.5,
                            )
                            ax.plot(
                                dominant_period,
                                dominant_power,
                                "o",
                                color=roi_color,
                                markersize=8,
                                markeredgecolor="black",
                                markeredgewidth=1,
                                label=f"Peak: {dominant_period:.1f}h",
                            )
                            ax.legend(fontsize=7)

                            # Stats box: period, peak time (phase), p-value
                            peak_t = result.get("dominant_peak_time_hours", None)
                            p_val  = result.get("p_value", 1.0)
                            sig_marker = " *" if result.get("is_significant", False) else ""
                            if peak_t is not None and not np.isnan(peak_t):
                                stats_text = (
                                    f"Period: {dominant_period:.2f}h\n"
                                    f"Peak time: {peak_t:.1f}h\n"
                                    f"p: {_fmt_p(p_val)}{sig_marker}"
                                )
                            else:
                                stats_text = (
                                    f"Period: {dominant_period:.2f}h\n"
                                    f"p: {_fmt_p(p_val)}{sig_marker}"
                                )
                            ax.text(
                                0.97, 0.97, stats_text,
                                transform=ax.transAxes, fontsize=8,
                                verticalalignment="top",
                                horizontalalignment="right",
                                bbox=dict(boxstyle="round", facecolor="wheat",
                                          alpha=0.5, pad=0.5),
                            )

                        # Set x-axis to actual data range (analysis caps max at duration/2)
                        if len(periods) > 0:
                            ax.set_xlim(min(periods), max(periods))

                # Hide unused subplots in this section
                for idx in range(n_rois, n_rows_per_section * n_cols):
                    row = start_row + (idx // n_cols)
                    col = idx % n_cols
                    if row < start_row + n_rows_per_section:
                        axes[row, col].axis("off")

            # Plot Activity section
            plot_section(roi_only_results, 0, "Activity")

            # Plot Sleep section if available
            if has_sleep:
                plot_section(sleep_results, n_rows_per_section, "Sleep")

            # --- Population mean panel ---
            if has_population and ax_pop is not None:
                all_p, all_pow = [], []
                for res in roi_only_results.values():
                    if "error" not in res:
                        p = np.array(res.get("relevant_periods", []))
                        pw = np.array(res.get("relevant_power", []))
                        if len(p) > 1 and len(p) == len(pw):
                            all_p.append(p)
                            all_pow.append(pw)

                if len(all_p) >= 2:
                    # relevant_periods is in descending order (high→low period)
                    # because it comes from 1/rfftfreq. Sort ascending for interp/linspace.
                    sorted_pairs = [
                        (np.sort(p), pw[np.argsort(p)])
                        for p, pw in zip(all_p, all_pow)
                    ]
                    all_p_s  = [sp for sp, _ in sorted_pairs]
                    all_pw_s = [spw for _, spw in sorted_pairs]
                    p_min = max(a[0]  for a in all_p_s)   # largest common lower bound
                    p_max = min(a[-1] for a in all_p_s)   # smallest common upper bound
                    if p_max > p_min:
                        grid = np.linspace(p_min, p_max, 300)
                        interp = np.array([np.interp(grid, p, pw) for p, pw in zip(all_p_s, all_pw_s)])
                        mean_pw = interp.mean(axis=0)
                        sem_pw = interp.std(axis=0) / np.sqrt(len(interp))

                        ax_pop.plot(grid, mean_pw, color="black", linewidth=2,
                                    label=f"Mean (n={len(interp)})")
                        ax_pop.fill_between(grid, mean_pw - sem_pw, mean_pw + sem_pw,
                                            alpha=0.25, color="gray", label="±SEM")

                        sig_periods = [
                            res.get("dominant_period")
                            for res in roi_only_results.values()
                            if res.get("is_significant") and res.get("dominant_period") is not None
                        ]
                        use_mean_peak = (
                            hasattr(self, "population_peak_mode")
                            and self.population_peak_mode.currentText() == "Mean"
                        )
                        if use_mean_peak:
                            peak_p = float(grid[np.argmax(mean_pw)])
                            ax_pop.axvline(peak_p, color="red", linestyle="--", linewidth=1.5,
                                           label=f"Mean peak: {peak_p:.1f}h")
                        elif sig_periods:
                            med_p = float(np.median(sig_periods))
                            ax_pop.axvline(med_p, color="red", linestyle="--", linewidth=1.5,
                                           label=f"Median peak: {med_p:.1f}h")
                        if sig_periods:
                            n_sig = len(sig_periods)
                            ax_pop.text(0.97, 0.95,
                                        f"Significant: {n_sig}/{len(all_p)}",
                                        transform=ax_pop.transAxes, fontsize=8,
                                        va="top", ha="right",
                                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

                        ax_pop.set_xlabel("Period (h)", fontsize=9)
                        ax_pop.set_ylabel("Power (a.u.)", fontsize=9)
                        ax_pop.set_title("Population Mean Power Spectrum", fontsize=10,
                                         fontweight="bold")
                        ax_pop.legend(fontsize=8, loc="upper left")
                        ax_pop.grid(True, alpha=0.3)
                        ax_pop.tick_params(axis="both", labelsize=8)
                else:
                    ax_pop.text(0.5, 0.5, "Not enough valid ROIs for population mean",
                                ha="center", va="center", transform=ax_pop.transAxes)
                    ax_pop.axis("off")

            self.fisher_plot_figure = fig

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())

            # Scale pixmap to fit canvas while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.fisher_plot_canvas.size(),
                1,  # Qt.KeepAspectRatio
                1,  # Qt.SmoothTransformation
            )
            self.fisher_plot_canvas.setPixmap(scaled_pixmap)

            # Enable pop-out button after successful plot creation
            if hasattr(self, "btn_popout_plot"):
                self.btn_popout_plot.setEnabled(True)
                if hasattr(self, "btn_save_fisher_plot"):
                    self.btn_save_fisher_plot.setEnabled(True)

        except Exception as e:
            self._log_message(f"⚠️ Could not create FFT plot: {e}")
            import traceback

            traceback.print_exc()

    def _create_cosinor_plot(self, cosinor_results: Dict):
        """Create cosinor analysis plots showing data and fitted curves.

        Cosinor analysis only uses raw movement data (continuous values),
        not fraction movement or binary sleep data.
        """
        try:
            import matplotlib.pyplot as plt
            from qtpy.QtGui import QPixmap
            import io

            # Close the previous figure and disconnect all Qt event callbacks
            # before creating a new one to prevent NavigationToolbar2QT dangling refs.
            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                try:
                    self.fisher_plot_figure = None
                except Exception:
                    pass
                self.fisher_plot_figure = None

            roi_results = cosinor_results.get("roi_results", {})
            # Filter to only integer ROI keys
            roi_results = {k: v for k, v in roi_results.items() if isinstance(k, int)}
            n_rois = len(roi_results)

            # Handle case with no ROIs
            if n_rois == 0:
                self._log_message("⚠️ No ROI results found for Cosinor plot")
                return

            n_cols = min(3, n_rois)
            n_rows_per_section = (n_rois + n_cols - 1) // n_cols

            show_population = (
                hasattr(self, "chk_cosinor_population")
                and self.chk_cosinor_population.isChecked()
            )
            pop_result = cosinor_results.get("population_result", {})
            has_population = show_population and "error" not in pop_result

            fig_height = 4.5 * n_rows_per_section + 1 + (3.5 if has_population else 0)
            from matplotlib.figure import Figure as _Figure
            fig = _Figure(figsize=(13, fig_height))

            if has_population:
                gs = fig.add_gridspec(
                    n_rows_per_section + 1, n_cols,
                    hspace=0.45,
                    height_ratios=[1] * n_rows_per_section + [0.8],
                )
                axes = np.array(
                    [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                     for r in range(n_rows_per_section)]
                )
                ax_pop = fig.add_subplot(gs[n_rows_per_section, :])
            else:
                gs = fig.add_gridspec(n_rows_per_section, n_cols, hspace=0.45)
                axes = np.array(
                    [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                     for r in range(n_rows_per_section)]
                )
                ax_pop = None

            data_type_name = getattr(self, "fisher_data_type_name", "Fraction Movement (0-1)")
            fig.suptitle(
                f"Cosinor Analysis  —  {data_type_name}{self._get_recording_start_str()}",
                fontsize=13,
                fontweight="bold",
                y=0.99,
            )

            # Recording duration (hours) — needed for figure-level warning and per-ROI annotation
            _recording_dur_h = cosinor_results.get("recording_duration_h", 0.0)

            # Figure-level warning if recording < 2× any test period
            test_periods_plot = cosinor_results.get("test_periods", [])
            short_periods = [
                tp for tp in test_periods_plot
                if _recording_dur_h > 0 and _recording_dur_h / tp < 2.0
            ]
            if short_periods:
                warn_txt = (
                    f"⚠  Recording ({_recording_dur_h:.1f} h) < 2× period — "
                    f"result(s) unreliable for: "
                    + ", ".join(f"{p:.1f} h" for p in short_periods)
                )
                fig.text(
                    0.5, 0.005, warn_txt,
                    ha="center", va="bottom", fontsize=8,
                    color="#B71C1C",
                    bbox=dict(boxstyle="round", fc="#FFEBEE", ec="#EF9A9A", alpha=0.9),
                )

            # Use the data that was actually analysed (stored by run_fisher_analysis)
            activity_data_dict = getattr(self, "fisher_analysis_data", None)
            if not activity_data_dict:
                activity_data_dict = self.fraction_data if hasattr(self, "fraction_data") else {}
            activity_y_label = data_type_name

            # Lighting overlay settings (shared by per-ROI and population panels)
            _show_lighting = (
                hasattr(self, "chk_show_lighting")
                and self.chk_show_lighting.isChecked()
            )
            _led_data = getattr(self, "led_data", None) if _show_lighting else None

            def _overlay_lighting(ax, t_max, add_legend=False):
                """Draw light (yellow) / dark (gray) bands on a ZT-hours axis."""
                if not _show_lighting:
                    return
                led = _led_data
                if led and isinstance(led, dict) and led.get("times") and led.get("white_powers"):
                    # HDF5 LED data available — reuse the same logic as _plot.py
                    import numpy as _np
                    times_h = _np.array(led["times"]) / 3600.0
                    wp = _np.array(led["white_powers"])
                    is_light = wp > 0.5
                    transitions = _np.diff(is_light.astype(int))
                    starts = _np.where(transitions == 1)[0] + 1
                    ends = _np.where(transitions == -1)[0] + 1
                    if is_light[0]:
                        starts = _np.concatenate([[0], starts])
                    if is_light[-1]:
                        ends = _np.concatenate([ends, [len(is_light) - 1]])
                    for i, (s, e) in enumerate(zip(starts, ends)):
                        ax.axvspan(times_h[s], times_h[e], alpha=0.2, color="yellow",
                                   zorder=0, label="Light" if i == 0 and add_legend else "")
                    # Dark periods: between light phases + leading/trailing dark
                    for i in range(len(ends)):
                        dark_s = times_h[ends[i]]
                        dark_e = times_h[starts[i + 1]] if i < len(starts) - 1 else t_max
                        ax.axvspan(dark_s, dark_e, alpha=0.15, color="gray", zorder=0,
                                   label="Dark" if i == 0 and add_legend else "")
                    # Leading dark before first light
                    if len(starts) > 0 and times_h[starts[0]] > 0:
                        ax.axvspan(0, times_h[starts[0]], alpha=0.15, color="gray",
                                   zorder=0)
                # else: no LED data → no overlay (AVI or other sources without LED metadata)

            # Helper function to plot ROI results
            def plot_section(
                results_dict, raw_data_dict, start_row, section_label, y_label
            ):
                roi_items = {
                    k: v for k, v in results_dict.items() if isinstance(k, int)
                }

                for idx, (roi_id, roi_data) in enumerate(sorted(roi_items.items())):
                    row = start_row + (idx // n_cols)
                    col = idx % n_cols
                    ax = axes[row, col]

                    roi_color = (
                        self.roi_colors.get(roi_id, f"C{idx}")
                        if hasattr(self, "roi_colors")
                        else f"C{idx}"
                    )

                    best_result = roi_data.get("best_result", {})

                    if "error" in best_result:
                        ax.text(
                            0.5,
                            0.5,
                            f"ROI {roi_id}\n{best_result['error']}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Get data for this ROI
                    if roi_id in raw_data_dict:
                        times_hours = np.array(
                            [t / 3600 for t, _ in raw_data_dict[roi_id]]
                        )
                        values = np.array([v for _, v in raw_data_dict[roi_id]])

                        # Plot actual data as a continuous line
                        ax.plot(
                            times_hours,
                            values,
                            alpha=0.6,
                            linewidth=0.8,
                            color=roi_color,
                            label=f"{section_label} data",
                        )

                        # Reconstruct fitted curve at raw data time points
                        # (avoids length mismatch when analysis used rebinned data)
                        period = best_result.get("period", 0)
                        amplitude = best_result.get("amplitude", 0)
                        mesor = best_result.get("mesor", 0)
                        peak_time = best_result.get("peak_time", 0)
                        phase_angle_rad = best_result.get("phase_angle_rad", 0)

                        if period > 0 and not np.isnan(period):
                            omega = 2 * np.pi / period
                            # Reconstruct using the phase angle φ:
                            # y(t) = MESOR + A·cos(ω·t + φ)
                            fitted_curve = mesor + amplitude * np.cos(
                                omega * times_hours + phase_angle_rad
                            )
                            ax.plot(
                                times_hours,
                                fitted_curve,
                                color="black",
                                linewidth=2,
                                label=f"Fitted curve ({period:.1f}h)",
                            )

                            # MESOR line
                            ax.axhline(
                                y=mesor,
                                color="gray",
                                linestyle="--",
                                linewidth=1,
                                alpha=0.5,
                                label=f"MESOR={mesor:.3f}",
                            )

                            # Peak time marker
                            peak_value = mesor + amplitude
                            if (
                                isinstance(peak_time, (int, float))
                                and peak_time < times_hours[-1]
                            ):
                                ax.plot(
                                    peak_time,
                                    peak_value,
                                    "^",
                                    color="red",
                                    markersize=10,
                                    markeredgecolor="black",
                                    markeredgewidth=0.5,
                                    label=f"Peak Time={peak_time:.1f}h",
                                )

                        textstr = f"MESOR: {best_result.get('mesor', 0):.3f}\n"
                        textstr += f"Amplitude: {best_result.get('amplitude', 0):.3f}\n"
                        textstr += f"R²: {best_result.get('r_squared', 0):.3f}\n"
                        textstr += f"p: {_fmt_p(best_result.get('p_value', 1))}"
                        if best_result.get("significant", False):
                            textstr += " *"
                        if period > 0 and _recording_dur_h > 0:
                            n_cycles = _recording_dur_h / period
                            warn_flag = "⚠ " if n_cycles < 2.0 else ""
                            textstr += f"\n{warn_flag}cycles: {n_cycles:.1f}"

                        ax.text(
                            0.95,
                            0.05,
                            textstr,
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment="bottom",
                            horizontalalignment="right",
                            bbox=dict(
                                boxstyle="round", facecolor="wheat", alpha=0.5, pad=0.5
                            ),
                        )

                    ax.set_xlabel("Time (h)", fontsize=9)
                    ax.set_ylabel(y_label, fontsize=9)

                    # Per-ROI Y-axis scaling - include fitted curve range
                    if roi_id in raw_data_dict and len(values) > 0:
                        data_max = max(values)
                        data_min = min(values)
                        # Also consider fitted curve range if it exists
                        if period > 0 and not np.isnan(period):
                            fitted_max = mesor + amplitude
                            fitted_min = mesor - amplitude
                            roi_y_max = max(data_max, fitted_max) * 1.1
                            roi_y_min = (
                                min(0, data_min, fitted_min) * 1.1
                                if min(0, data_min, fitted_min) < 0
                                else min(0, data_min)
                            )
                        else:
                            roi_y_max = data_max * 1.1
                            roi_y_min = min(0, data_min)
                        ax.set_ylim(roi_y_min, roi_y_max)
                    ax.tick_params(axis="both", labelsize=8)
                    is_significant = best_result.get("significant", False)
                    title_weight = "bold" if is_significant else "normal"
                    ax.set_title(
                        f"ROI {roi_id} - {section_label}",
                        fontsize=9,
                        color=roi_color,
                        fontweight=title_weight,
                        pad=4,
                        loc="left",
                    )
                    # Lighting overlay (drawn before legend so bands stay in background)
                    if roi_id in raw_data_dict and len(raw_data_dict[roi_id]) > 0:
                        _t_max_roi = raw_data_dict[roi_id][-1][0] / 3600.0
                        _t_min_roi = raw_data_dict[roi_id][0][0] / 3600.0
                        _overlay_lighting(ax, _t_max_roi, add_legend=(idx == 0))
                        # Restore data-range x-limits after axvspan (which can extend them)
                        ax.set_xlim(_t_min_roi, _t_max_roi)
                    ax.legend(fontsize=7, loc="upper left")
                    ax.grid(True, alpha=0.3)

                # Hide unused subplots in this section
                for idx in range(n_rois, n_rows_per_section * n_cols):
                    row = start_row + (idx // n_cols)
                    col = idx % n_cols
                    if row < start_row + n_rows_per_section:
                        axes[row, col].axis("off")

            # Plot Activity section — use a short label derived from data_type_name
            # e.g. "Fraction Movement (0-1)" → "Fraction Movement"
            section_label = data_type_name.split("(")[0].strip()
            plot_section(
                roi_results,
                activity_data_dict,
                0,
                section_label,
                activity_y_label,
            )

            # --- Optional population mean subplot ---
            if has_population and ax_pop is not None:
                # Build common time grid from union of all ROI time ranges
                all_times_h = []
                for data_list in activity_data_dict.values():
                    if data_list:
                        all_times_h.extend(t / 3600.0 for t, _ in data_list)
                t_min = min(all_times_h) if all_times_h else 0.0
                t_max = max(all_times_h) if all_times_h else 24.0
                t_grid = np.linspace(t_min, t_max, 500)

                # Faint individual fitted curves
                for idx, (roi_id, roi_data) in enumerate(sorted(roi_results.items())):
                    best = roi_data.get("best_result", {})
                    period = best.get("period", 0)
                    amplitude = best.get("amplitude", 0)
                    mesor = best.get("mesor", 0)
                    phi = best.get("phase_angle_rad", 0)
                    if period > 0 and not np.isnan(period):
                        omega = 2 * np.pi / period
                        fitted = mesor + amplitude * np.cos(omega * t_grid + phi)
                        roi_color = (
                            self.roi_colors.get(roi_id, f"C{idx}")
                            if hasattr(self, "roi_colors") else f"C{idx}"
                        )
                        ax_pop.plot(t_grid, fitted, color=roi_color,
                                    alpha=0.25, linewidth=1)

                # Population mean fit
                pop_period = pop_result.get("period", 24.0)
                pop_amplitude = pop_result.get("population_amplitude", 0)
                pop_mesor = pop_result.get("population_mesor", 0)
                pop_peak_time = pop_result.get("population_peak_time", 0)

                if pop_period > 0:
                    omega_pop = 2 * np.pi / pop_period
                    phase_pop = -omega_pop * pop_peak_time
                    pop_curve = pop_mesor + pop_amplitude * np.cos(
                        omega_pop * t_grid + phase_pop
                    )
                    ax_pop.plot(t_grid, pop_curve, color="black", linewidth=2.5,
                                label=f"Population mean ({pop_period:.1f}h)", zorder=5)
                    ax_pop.axhline(y=pop_mesor, color="gray", linestyle="--",
                                   linewidth=1, alpha=0.5,
                                   label=f"Pop. MESOR={pop_mesor:.3f}")

                p_val = pop_result.get("p_value", 1.0)
                sig_marker = " *" if pop_result.get("significant", False) else ""
                n_sig = pop_result.get("n_significant", 0)
                n_ind = pop_result.get("n_individuals", 0)
                stats_text = (
                    f"MESOR: {pop_mesor:.3f}\n"
                    f"Amplitude: {pop_amplitude:.3f}\n"
                    f"Peak time: {pop_peak_time:.1f}h\n"
                    f"p: {_fmt_p(p_val)}{sig_marker}\n"
                    f"n={n_ind} ROIs  ({n_sig} significant)"
                )
                ax_pop.text(
                    0.97, 0.05, stats_text, transform=ax_pop.transAxes,
                    fontsize=8, verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat",
                              alpha=0.5, pad=0.5),
                )
                ax_pop.set_title("Population Mean", fontsize=10,
                                 fontweight="bold", loc="left")
                ax_pop.set_xlabel("Time (h)", fontsize=9)
                ax_pop.set_ylabel(activity_y_label, fontsize=9)
                _overlay_lighting(ax_pop, t_max, add_legend=False)
                ax_pop.legend(fontsize=7, loc="upper left")
                ax_pop.grid(True, alpha=0.3)
                ax_pop.tick_params(axis="both", labelsize=8)

            # subplots_adjust instead of tight_layout: the latter does not
            # support axes that span multiple columns (population subplot).
            fig.subplots_adjust(top=0.94, left=0.07, right=0.97,
                                bottom=0.04, hspace=0.50, wspace=0.30)
            self.fisher_plot_figure = fig

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self.fisher_plot_canvas.setPixmap(
                pixmap.scaled(self.fisher_plot_canvas.size(), 1, 1)
            )

            # Enable pop-out button after successful plot creation
            if hasattr(self, "btn_popout_plot"):
                self.btn_popout_plot.setEnabled(True)
                if hasattr(self, "btn_save_fisher_plot"):
                    self.btn_save_fisher_plot.setEnabled(True)

        except Exception as e:
            self._log_message(f"⚠️ Could not create Cosinor plot: {e}")
            import traceback

            traceback.print_exc()

    def _create_similarity_plot(self, similarity_results: Dict):
        """Create correlation matrix heatmap and dendrogram."""
        try:
            import matplotlib.pyplot as plt
            from qtpy.QtGui import QPixmap
            import io
            from scipy.cluster import hierarchy

            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                self.fisher_plot_figure = None

            from matplotlib.figure import Figure as _Figure
            fig = _Figure(figsize=(14, 6))

            # Get data source for plot title
            data_source_index = self.data_source_combo.currentIndex()
            data_source = (
                "Fraction Movement" if data_source_index == 0 else "Raw Intensity"
            )
            fig.suptitle(
                f"ROI Similarity Analysis\n(Activity from {data_source})",
                fontsize=14,
                fontweight="bold",
            )

            # Correlation heatmap
            ax1 = fig.add_subplot(1, 2, 1)
            corr_matrix = similarity_results["correlation_matrix"]
            roi_ids = similarity_results["roi_ids"]

            im = ax1.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax1.set_xticks(range(len(roi_ids)))
            ax1.set_yticks(range(len(roi_ids)))
            ax1.set_xticklabels(roi_ids, rotation=45)
            ax1.set_yticklabels(roi_ids)
            ax1.set_title("ROI Correlation Matrix")
            fig.colorbar(im, ax=ax1, label="Correlation")

            # Dendrogram
            ax2 = fig.add_subplot(1, 2, 2)
            if "clustering" in similarity_results:
                linkage = similarity_results["clustering"]["linkage_matrix"]

                # Cluster cut: distance = 1 − r  (slider in [0,100] → r in [0,1])
                r_threshold = (
                    self.similarity_threshold_slider.value() / 100.0
                    if hasattr(self, "similarity_threshold_slider")
                    else 0.5
                )
                color_threshold = 1.0 - r_threshold

                # Create dendrogram with colors
                hierarchy.dendrogram(
                    linkage,
                    labels=[str(r) for r in roi_ids],
                    ax=ax2,
                    color_threshold=color_threshold,
                    above_threshold_color="gray",
                )
                ax2.tick_params(axis="x", labelsize=9)

                ax2.set_title("Hierarchical Clustering", fontsize=12, fontweight="bold")
                ax2.set_xlabel("ROI", fontsize=11)
                ax2.set_ylabel("Distance (1 - correlation)", fontsize=11)
                ax2.tick_params(axis="both", labelsize=10)

                # Add horizontal line at threshold
                ax2.axhline(
                    y=color_threshold,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Cluster cut  r={r_threshold:.2f}  (d={color_threshold:.2f})",
                )
                ax2.legend(fontsize=9)

                # Add grid for better readability
                ax2.grid(True, alpha=0.3, axis="y")

            fig.tight_layout()
            self.fisher_plot_figure = fig

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self.fisher_plot_canvas.setPixmap(
                pixmap.scaled(self.fisher_plot_canvas.size(), 1, 1)
            )

            # Enable pop-out button after successful plot creation
            if hasattr(self, "btn_popout_plot"):
                self.btn_popout_plot.setEnabled(True)
                if hasattr(self, "btn_save_fisher_plot"):
                    self.btn_save_fisher_plot.setEnabled(True)

        except Exception as e:
            self._log_message(f"⚠️ Could not create similarity plot: {e}")
            import traceback

            traceback.print_exc()

    def _create_coherence_plot(self, coherence_results: Dict):
        """Create coherence matrix heatmap."""
        try:
            import matplotlib.pyplot as plt
            from qtpy.QtGui import QPixmap
            import io

            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                self.fisher_plot_figure = None

            from matplotlib.figure import Figure as _Figure
            fig = _Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)

            # Get data source for plot title
            data_source_index = self.data_source_combo.currentIndex()
            data_source = (
                "Fraction Movement" if data_source_index == 0 else "Raw Intensity"
            )

            coherence_matrix = coherence_results["coherence_matrix"]
            roi_ids = coherence_results["roi_ids"]

            im = ax.imshow(coherence_matrix, cmap="hot", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(roi_ids)))
            ax.set_yticks(range(len(roi_ids)))
            ax.set_xticklabels(roi_ids, rotation=45)
            ax.set_yticklabels(roi_ids)
            ax.set_title(
                f"ROI Coherence at ~{coherence_results['target_period_hours']:.0f}h Period\n(Activity from {data_source})"
            )
            fig.colorbar(im, ax=ax, label="Coherence")

            fig.tight_layout()
            self.fisher_plot_figure = fig

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self.fisher_plot_canvas.setPixmap(
                pixmap.scaled(self.fisher_plot_canvas.size(), 1, 1)
            )

            # Enable pop-out button after successful plot creation
            if hasattr(self, "btn_popout_plot"):
                self.btn_popout_plot.setEnabled(True)
                if hasattr(self, "btn_save_fisher_plot"):
                    self.btn_save_fisher_plot.setEnabled(True)

        except Exception as e:
            self._log_message(f"⚠️ Could not create coherence plot: {e}")
            import traceback

            traceback.print_exc()

    def _create_phase_cluster_plot(self, phase_results: Dict):
        """Create polar plot showing ROI phases."""
        try:
            import matplotlib.pyplot as plt
            from qtpy.QtGui import QPixmap
            import io

            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                self.fisher_plot_figure = None

            from matplotlib.figure import Figure as _Figure
            fig = _Figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="polar")

            roi_phases = phase_results["roi_phases"]
            # Filter to only integer ROI keys
            roi_phases_filtered = {
                k: v for k, v in roi_phases.items() if isinstance(k, int)
            }

            for idx, (roi_id, phase_info) in enumerate(
                sorted(roi_phases_filtered.items())
            ):
                # Get ROI-specific color
                roi_color = (
                    self.roi_colors.get(roi_id, f"C{idx}")
                    if hasattr(self, "roi_colors")
                    else f"C{idx}"
                )

                theta = (phase_info["phase_radians"] + np.pi) % (
                    2 * np.pi
                )  # Adjust for polar plot
                r = phase_info["amplitude"]
                ax.plot(
                    [theta, theta],
                    [0, r],
                    color=roi_color,
                    linewidth=2,
                    label=f"ROI {roi_id}",
                )
                ax.scatter(
                    [theta],
                    [r],
                    color=roi_color,
                    s=100,
                    edgecolor="black",
                    linewidth=1,
                    zorder=5,
                )

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            # --- Population mean resultant vector ---
            has_population = (
                hasattr(self, "chk_cosinor_population")
                and self.chk_cosinor_population.isChecked()
            )
            if has_population and len(roi_phases_filtered) >= 2:
                thetas = np.array([
                    (v["phase_radians"] + np.pi) % (2 * np.pi)
                    for v in roi_phases_filtered.values()
                ])
                amplitudes = np.array([v["amplitude"] for v in roi_phases_filtered.values()])
                sin_mean = np.mean(np.sin(thetas))
                cos_mean = np.mean(np.cos(thetas))
                mean_theta = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
                R = float(np.sqrt(sin_mean ** 2 + cos_mean ** 2))  # synchrony index 0–1
                mean_amp = float(np.mean(amplitudes)) * R

                ax.plot([mean_theta, mean_theta], [0, mean_amp],
                        color="black", linewidth=3.5, zorder=10,
                        label=f"Population mean (R={R:.2f})")
                ax.scatter([mean_theta], [mean_amp], color="black", s=200,
                           edgecolor="white", linewidth=1.5, zorder=11)

                # Convert mean phase back to hours
                target_period = (self.fisher_min_period.value() + self.fisher_max_period.value()) / 2.0
                mean_phase_h = (mean_theta / (2 * np.pi)) * target_period
                ax.text(mean_theta, mean_amp * 1.12,
                        f"R={R:.2f}\n{mean_phase_h:.1f}h",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="black")

            # Get data source for plot title
            data_source_index = self.data_source_combo.currentIndex()
            data_source = (
                "Fraction Movement" if data_source_index == 0 else "Raw Intensity"
            )
            ax.set_title(
                f"ROI Activity Phases\n(Activity from {data_source})",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=8)

            # Polar axes are incompatible with tight_layout — use subplots_adjust instead
            fig.subplots_adjust(left=0.05, right=0.82, bottom=0.05, top=0.90)
            self.fisher_plot_figure = fig

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self.fisher_plot_canvas.setPixmap(
                pixmap.scaled(self.fisher_plot_canvas.size(), 1, 1)
            )

            # Enable pop-out button after successful plot creation
            if hasattr(self, "btn_popout_plot"):
                self.btn_popout_plot.setEnabled(True)
                if hasattr(self, "btn_save_fisher_plot"):
                    self.btn_save_fisher_plot.setEnabled(True)

        except Exception as e:
            self._log_message(f"⚠️ Could not create phase plot: {e}")
            import traceback

            traceback.print_exc()

    def export_fisher_results(self):
        """Export rhythmic pattern analysis results to Excel/CSV."""
        if not hasattr(self, "fisher_analysis_results"):
            self._log_message("⚠️ No analysis results to export")
            return

        if not hasattr(self, "current_fisher_method"):
            self._log_message("⚠️ Cannot determine analysis method")
            return

        from qtpy.QtWidgets import QFileDialog
        import os

        # Determine method name for file naming
        method_names = {
            0: "fisher_z",
            1: "fft_spectrum",
            2: "cosinor",
            3: "roi_similarity",
            4: "coherence_analysis",
            5: "phase_clustering",
        }
        method_name = method_names.get(self.current_fisher_method, "rhythmic_analysis")

        # Get save location
        default_name = f"rhythmic_{method_name}_results"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Rhythmic Pattern Analysis Results",
            default_name,
            "Excel Files (*.xlsx);;All Files (*.*)",
        )

        if not file_path:
            self._log_message("Export cancelled by user")
            return

        # Ensure .xlsx extension
        if not file_path.endswith(".xlsx"):
            file_path = f"{file_path}.xlsx"

        # Remove extension for consistent naming
        base_path = os.path.splitext(file_path)[0]

        try:
            # Route to appropriate export method based on analysis type
            if self.current_fisher_method == 0:  # Chi² Periodogram
                self._export_fisher_method_results(base_path)
            elif self.current_fisher_method == 1:  # FFT Power Spectrum
                self._export_fft_method_results(base_path)
            elif self.current_fisher_method == 2:  # Cosinor Analysis
                self._export_cosinor_method_results(base_path)
            elif self.current_fisher_method == 3:  # ROI Similarity Matrix
                self._export_similarity_method_results(file_path)
            elif self.current_fisher_method == 4:  # Coherence Analysis
                self._export_coherence_method_results(file_path)
            elif self.current_fisher_method == 5:  # Phase Clustering
                self._export_phase_clustering_method_results(file_path)
            else:
                self._log_message(
                    f"❌ Unknown analysis method: {self.current_fisher_method}"
                )
                return

            # Export plot if available
            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                plot_path = f"{base_path}_plot.png"
                self.fisher_plot_figure.savefig(
                    plot_path, dpi=300, bbox_inches="tight", facecolor="white"
                )
                self._log_message(f"✓ Exported plot to: {plot_path}")

            self._log_message("✓ Export complete!")

        except Exception as e:
            self._log_message(f"❌ Export failed: {e}")
            import traceback

            traceback.print_exc()

    def _export_fisher_method_results(self, base_path):
        """Export Chi² Periodogram results to CSV and Excel."""
        # Export to CSV
        csv_path = f"{base_path}.csv"
        self._export_fisher_to_csv(csv_path)
        self._log_message(f"✓ Exported Fisher results to CSV: {csv_path}")

        # Export to Excel
        try:
            import pandas as pd

            excel_path = f"{base_path}.xlsx"
            self._export_fisher_to_excel(excel_path)
            self._log_message(f"✓ Exported Fisher results to Excel: {excel_path}")
        except ImportError:
            self._log_message("⚠️ Excel export not available (pandas not installed)")

    def _export_fft_method_results(self, base_path):
        """Export FFT Power Spectrum results to CSV and Excel."""
        # Export to CSV (similar to Fisher)
        csv_path = f"{base_path}.csv"
        self._export_fft_to_csv(csv_path)
        self._log_message(f"✓ Exported FFT results to CSV: {csv_path}")

        # Export to Excel
        try:
            import pandas as pd

            excel_path = f"{base_path}.xlsx"
            self._export_fft_to_excel(excel_path)
            self._log_message(f"✓ Exported FFT results to Excel: {excel_path}")
        except ImportError:
            self._log_message("⚠️ Excel export not available (pandas not installed)")

    def _export_cosinor_method_results(self, base_path):
        """Export Cosinor Analysis results to CSV and Excel."""
        csv_path = f"{base_path}.csv"
        self._export_cosinor_to_csv(csv_path)
        self._log_message(f"✓ Exported Cosinor results to CSV: {csv_path}")

        try:
            import pandas as pd

            excel_path = f"{base_path}.xlsx"
            self._export_cosinor_to_excel(excel_path)
            self._log_message(f"✓ Exported Cosinor results to Excel: {excel_path}")
        except ImportError:
            self._log_message("⚠️ Excel export not available (pandas not installed)")
        except Exception as e:
            self._log_message(f"⚠️ Cosinor export error: {e}")

    def _export_cosinor_to_excel(self, file_path: str):
        """Export Cosinor analysis results to Excel format."""
        import pandas as pd

        results = self.fisher_analysis_results
        roi_results = results.get("roi_results", {})
        sleep_results = results.get("sleep_phase_results", {})
        sleep_roi_results = (
            sleep_results.get("roi_results", {}) if sleep_results else {}
        )

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Activity Summary
            activity_data = []
            for roi_id, roi_data in sorted(
                {k: v for k, v in roi_results.items() if isinstance(k, int)}.items()
            ):
                best_result = roi_data.get("best_result", {})
                if "error" in best_result:
                    continue

                activity_data.append(
                    {
                        "ROI": roi_id,
                        "Best Period (hours)": roi_data.get("best_period", "N/A"),
                        "Peak Time (hours from start)": best_result.get("peak_time", "N/A"),
                        "Phase Angle (rad)": best_result.get("phase_angle_rad", "N/A"),
                        "MESOR": best_result.get("mesor", "N/A"),
                        "Amplitude": best_result.get("amplitude", "N/A"),
                        "R-squared": best_result.get("r_squared", "N/A"),
                        "P-value": best_result.get("p_value", "N/A"),
                        "Significant": best_result.get("significant", False),
                    }
                )

            if activity_data:
                activity_df = pd.DataFrame(activity_data)
                activity_df.to_excel(writer, sheet_name="Activity_Summary", index=False)

            # Sheet 2: Sleep Summary (if available)
            if sleep_roi_results:
                sleep_data = []
                for roi_id, roi_data in sorted(
                    {
                        k: v for k, v in sleep_roi_results.items() if isinstance(k, int)
                    }.items()
                ):
                    best_result = roi_data.get("best_result", {})
                    if "error" in best_result:
                        continue

                    sleep_data.append(
                        {
                            "ROI": roi_id,
                            "Best Period (hours)": roi_data.get("best_period", "N/A"),
                            "Sleep Peak Time (hours from start)": best_result.get("peak_time", "N/A"),
                            "MESOR": best_result.get("mesor", "N/A"),
                            "Amplitude": best_result.get("amplitude", "N/A"),
                            "R-squared": best_result.get("r_squared", "N/A"),
                            "P-value": best_result.get("p_value", "N/A"),
                            "Significant": best_result.get("significant", False),
                        }
                    )

                if sleep_data:
                    sleep_df = pd.DataFrame(sleep_data)
                    sleep_df.to_excel(writer, sheet_name="Sleep_Summary", index=False)

            # Sheet 3: Combined comparison (Activity vs Sleep)
            if activity_data and sleep_roi_results:
                comparison_data = []
                for roi_id in sorted(
                    set([d["ROI"] for d in activity_data]).intersection(
                        [k for k in sleep_roi_results.keys() if isinstance(k, int)]
                    )
                ):
                    act_row = next((d for d in activity_data if d["ROI"] == roi_id), {})
                    slp_roi = sleep_roi_results.get(roi_id, {})
                    slp_best = slp_roi.get("best_result", {})

                    act_peak = act_row.get("Peak Time (hours from start)", 0)
                    slp_peak = slp_best.get("peak_time", 0)

                    # Calculate peak time difference
                    if isinstance(act_peak, (int, float)) and isinstance(
                        slp_peak, (int, float)
                    ):
                        phase_diff = abs(act_peak - slp_peak)
                        if phase_diff > 12:
                            phase_diff = 24 - phase_diff
                    else:
                        phase_diff = "N/A"

                    comparison_data.append(
                        {
                            "ROI": roi_id,
                            "Activity Peak Time (h)": act_peak,
                            "Sleep Peak Time (h)": slp_peak,
                            "Peak Time Difference (h)": phase_diff,
                            "Activity Period (h)": act_row.get(
                                "Best Period (hours)", "N/A"
                            ),
                            "Sleep Period (h)": slp_roi.get("best_period", "N/A"),
                        }
                    )

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df.to_excel(
                        writer, sheet_name="Activity_vs_Sleep", index=False
                    )

            # Sheet 4: Parameters
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Minimum Period",
                        "Maximum Period",
                        "Significance Level",
                        "Sampling Interval",
                        "Sleep Phase Calculated",
                    ],
                    "Value": [
                        f"{self.fisher_min_period.value():.1f} hours",
                        f"{self.fisher_max_period.value():.1f} hours",
                        f"{self.fisher_significance.value():.3f}",
                        f"{self.frame_interval.value():.1f} seconds",
                        "Yes" if sleep_roi_results else "No",
                    ],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

    def _export_similarity_method_results(self, base_path):
        """Export ROI Similarity results to CSV and Excel."""
        csv_path = f"{base_path}.csv"
        self._export_similarity_to_csv(csv_path)
        self._log_message(f"✓ Exported Similarity results to CSV: {csv_path}")

        try:
            from ._circadian_similarity import export_similarity_to_excel
            excel_path = f"{base_path}.xlsx"
            export_similarity_to_excel(excel_path, self.fisher_analysis_results)
            self._log_message(f"✓ Exported Similarity results to Excel: {excel_path}")
        except Exception as e:
            self._log_message(f"⚠️ Similarity Excel export error: {e}")

    def _export_coherence_method_results(self, base_path):
        """Export Coherence Analysis results to CSV and Excel."""
        csv_path = f"{base_path}.csv"
        self._export_coherence_to_csv(csv_path)
        self._log_message(f"✓ Exported Coherence results to CSV: {csv_path}")

        try:
            from ._circadian_coherence import export_coherence_to_excel
            excel_path = f"{base_path}.xlsx"
            export_coherence_to_excel(excel_path, self.fisher_analysis_results)
            self._log_message(f"✓ Exported Coherence results to Excel: {excel_path}")
        except Exception as e:
            self._log_message(f"⚠️ Coherence Excel export error: {e}")

    def _export_phase_clustering_method_results(self, base_path):
        """Export Phase Clustering results to CSV and Excel."""
        csv_path = f"{base_path}.csv"
        self._export_phase_clustering_to_csv(csv_path)
        self._log_message(f"✓ Exported Phase Clustering results to CSV: {csv_path}")

        try:
            import pandas as pd
            excel_path = f"{base_path}.xlsx"
            self._export_phase_clustering_to_excel(excel_path)
            self._log_message(f"✓ Exported Phase Clustering results to Excel: {excel_path}")
        except Exception as e:
            self._log_message(f"⚠️ Phase Clustering Excel export error: {e}")

    def _export_phase_clustering_to_excel(self, file_path):
        """Export Phase Clustering results to Excel."""
        import pandas as pd

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Phase Clusters
            if "phase_clusters" in self.fisher_analysis_results:
                cluster_data = []
                for cluster_name, roi_list in self.fisher_analysis_results[
                    "phase_clusters"
                ].items():
                    for roi in roi_list:
                        cluster_data.append(
                            {
                                "ROI": roi,
                                "Cluster": cluster_name.replace("_", " ").title(),
                                "Cluster_Size": len(roi_list),
                            }
                        )

                if cluster_data:
                    cluster_df = pd.DataFrame(cluster_data)
                    cluster_df = cluster_df.sort_values(["Cluster", "ROI"])
                    cluster_df.to_excel(
                        writer, sheet_name="Phase_Clusters", index=False
                    )

            # Sheet 2: Phase Values
            if "roi_phases" in self.fisher_analysis_results:
                phase_data = []
                for roi_id, phase_info in self.fisher_analysis_results[
                    "roi_phases"
                ].items():
                    phase_data.append(
                        {
                            "ROI": roi_id,
                            "Phase_Radians": phase_info["phase_radians"],
                            "Phase_Degrees": np.degrees(phase_info["phase_radians"]),
                            "Phase_Hours": phase_info["phase_hours"],
                            "Amplitude": phase_info["amplitude"],
                        }
                    )

                phase_df = pd.DataFrame(phase_data)
                phase_df = phase_df.sort_values("ROI")
                phase_df.to_excel(writer, sheet_name="Phase_Values", index=False)

            # Sheet 3: Parameters
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Number of Clusters",
                        "Number of ROIs",
                        "Dominant Period (hours)",
                    ],
                    "Value": [
                        str(
                            len(self.fisher_analysis_results.get("phase_clusters", {}))
                        ),
                        str(self.fisher_analysis_results.get("n_rois", 0)),
                        f"{self.fisher_analysis_results.get('dominant_period_hours', 0):.1f}",
                    ],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

        self._log_message(f"✓ Exported Phase Clustering results to Excel: {file_path}")

    def _export_cosinor_to_csv(self, file_path: str):
        """Export Cosinor Analysis results to CSV format."""
        import csv

        results = self.fisher_analysis_results
        roi_results = results.get("roi_results", {})
        sleep_results = results.get("sleep_phase_results", {})
        sleep_roi_results = sleep_results.get("roi_results", {}) if sleep_results else {}

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["Cosinor Analysis Results"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([])

            # Activity Summary
            writer.writerow(["Activity Summary"])
            writer.writerow(["ROI", "Best Period (h)", "Peak Time (h)", "Phase Angle (rad)",
                             "MESOR", "Amplitude", "R-squared", "P-value", "Significant"])
            for roi_id, roi_data in sorted(
                {k: v for k, v in roi_results.items() if isinstance(k, int)}.items()
            ):
                best = roi_data.get("best_result", {})
                if "error" in best:
                    writer.writerow([roi_id, "Error", best["error"]])
                    continue
                writer.writerow([
                    roi_id,
                    f"{roi_data.get('best_period', 'N/A')}",
                    f"{best.get('peak_time', 'N/A')}",
                    f"{best.get('phase_angle_rad', 'N/A')}",
                    f"{best.get('mesor', 'N/A')}",
                    f"{best.get('amplitude', 'N/A')}",
                    f"{best.get('r_squared', 'N/A')}",
                    f"{best.get('p_value', 'N/A')}",
                    str(best.get("significant", False)),
                ])

            if sleep_roi_results:
                writer.writerow([])
                writer.writerow(["Sleep Summary"])
                writer.writerow(["ROI", "Best Period (h)", "Sleep Peak Time (h)",
                                 "MESOR", "Amplitude", "R-squared", "P-value", "Significant"])
                for roi_id, roi_data in sorted(
                    {k: v for k, v in sleep_roi_results.items() if isinstance(k, int)}.items()
                ):
                    best = roi_data.get("best_result", {})
                    if "error" in best:
                        writer.writerow([roi_id, "Error", best["error"]])
                        continue
                    writer.writerow([
                        roi_id,
                        f"{roi_data.get('best_period', 'N/A')}",
                        f"{best.get('peak_time', 'N/A')}",
                        f"{best.get('mesor', 'N/A')}",
                        f"{best.get('amplitude', 'N/A')}",
                        f"{best.get('r_squared', 'N/A')}",
                        f"{best.get('p_value', 'N/A')}",
                        str(best.get("significant", False)),
                    ])

            writer.writerow([])
            writer.writerow(["Parameters"])
            writer.writerow(["Parameter", "Value"])
            writer.writerow(["Minimum Period", f"{self.fisher_min_period.value():.1f} hours"])
            writer.writerow(["Maximum Period", f"{self.fisher_max_period.value():.1f} hours"])
            writer.writerow(["Significance Level", f"{self.fisher_significance.value():.3f}"])
            writer.writerow(["Sampling Interval", f"{self.frame_interval.value():.1f} seconds"])
            writer.writerow(["Sleep Phase Calculated", "Yes" if sleep_roi_results else "No"])

    def _export_similarity_to_csv(self, file_path: str):
        """Export ROI Similarity (correlation matrix) results to CSV format."""
        import csv

        results = self.fisher_analysis_results

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["ROI Similarity Analysis Results"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([])

            # Correlation matrix
            corr_matrix = results.get("correlation_matrix", None)
            roi_ids = results.get("roi_ids", [])
            if corr_matrix is not None and len(roi_ids) > 0:
                writer.writerow(["Correlation Matrix"])
                writer.writerow(["ROI"] + [f"ROI {r}" for r in roi_ids])
                for i, roi_id in enumerate(roi_ids):
                    row = [f"ROI {roi_id}"]
                    for j in range(len(roi_ids)):
                        row.append(f"{corr_matrix[i, j]:.4f}")
                    writer.writerow(row)
            else:
                writer.writerow(["No correlation matrix available"])

            # Lag matrix
            lag_matrix = results.get("lag_matrix", None)
            if lag_matrix is not None and len(roi_ids) > 0:
                writer.writerow([])
                writer.writerow(["Lag Matrix (hours)"])
                writer.writerow(["ROI"] + [f"ROI {r}" for r in roi_ids])
                for i, roi_id in enumerate(roi_ids):
                    row = [f"ROI {roi_id}"]
                    for j in range(len(roi_ids)):
                        row.append(f"{lag_matrix[i, j]:.2f}")
                    writer.writerow(row)

            # Cluster assignments
            clusters = results.get("clusters", None)
            if clusters is not None and len(roi_ids) > 0:
                writer.writerow([])
                writer.writerow(["Cluster Assignments"])
                writer.writerow(["ROI", "Cluster"])
                for roi_id, cluster in zip(roi_ids, clusters):
                    writer.writerow([roi_id, cluster])

    def _export_coherence_to_csv(self, file_path: str):
        """Export Coherence Analysis results to CSV format."""
        import csv

        results = self.fisher_analysis_results

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["Coherence Analysis Results"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([])

            # Pairwise coherence summary
            coherence_data = results.get("coherence_results", {})
            if coherence_data:
                writer.writerow(["Pairwise Coherence Summary"])
                writer.writerow(["ROI Pair", "Mean Coherence", "Peak Frequency (Hz)",
                                 "Peak Period (h)"])
                for pair_key, pair_data in coherence_data.items():
                    if isinstance(pair_data, dict):
                        freqs = pair_data.get("frequencies", [])
                        coh = pair_data.get("coherence", [])
                        mean_coh = float(np.mean(coh)) if len(coh) > 0 else 0.0
                        if len(freqs) > 0 and len(coh) > 0:
                            peak_idx = int(np.argmax(coh))
                            peak_freq = freqs[peak_idx]
                            peak_period = (1.0 / peak_freq / 3600.0) if peak_freq > 0 else 0.0
                        else:
                            peak_freq = 0.0
                            peak_period = 0.0
                        writer.writerow([pair_key, f"{mean_coh:.4f}",
                                         f"{peak_freq:.6f}", f"{peak_period:.2f}"])

                # Detailed frequency-by-frequency data per pair
                writer.writerow([])
                writer.writerow(["Detailed Coherence per Pair"])
                for pair_key, pair_data in coherence_data.items():
                    if isinstance(pair_data, dict):
                        freqs = pair_data.get("frequencies", [])
                        coh = pair_data.get("coherence", [])
                        if len(freqs) > 0:
                            writer.writerow([])
                            writer.writerow([f"Pair: {pair_key}"])
                            writer.writerow(["Frequency (Hz)", "Period (h)", "Coherence"])
                            for f, c in zip(freqs, coh):
                                period_h = (1.0 / f / 3600.0) if f > 0 else 0.0
                                writer.writerow([f"{f:.6f}", f"{period_h:.2f}", f"{c:.4f}"])
            else:
                writer.writerow(["No coherence data available"])

    def _export_phase_clustering_to_csv(self, file_path: str):
        """Export Phase Clustering results to CSV format."""
        import csv

        results = self.fisher_analysis_results

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["Phase Clustering Results"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([])

            # Parameters
            writer.writerow(["Parameters"])
            writer.writerow(["Number of Clusters",
                             len(results.get("phase_clusters", {}))])
            writer.writerow(["Number of ROIs", results.get("n_rois", 0)])
            writer.writerow(["Dominant Period (h)",
                             f"{results.get('dominant_period_hours', 0):.1f}"])
            writer.writerow([])

            # Cluster assignments
            phase_clusters = results.get("phase_clusters", {})
            if phase_clusters:
                writer.writerow(["Cluster Assignments"])
                writer.writerow(["ROI", "Cluster", "Cluster Size"])
                for cluster_name, roi_list in phase_clusters.items():
                    label = cluster_name.replace("_", " ").title()
                    for roi in sorted(roi_list):
                        writer.writerow([roi, label, len(roi_list)])

            # Phase values per ROI
            roi_phases = results.get("roi_phases", {})
            if roi_phases:
                writer.writerow([])
                writer.writerow(["Phase Values per ROI"])
                writer.writerow(["ROI", "Phase (rad)", "Phase (deg)", "Phase (h)", "Amplitude"])
                for roi_id, phase_info in sorted(roi_phases.items()):
                    writer.writerow([
                        roi_id,
                        f"{phase_info.get('phase_radians', 0):.4f}",
                        f"{np.degrees(phase_info.get('phase_radians', 0)):.2f}",
                        f"{phase_info.get('phase_hours', 0):.2f}",
                        f"{phase_info.get('amplitude', 0):.4f}",
                    ])

    def _export_fft_to_csv(self, file_path: str):
        """Export FFT Power Spectrum results to CSV format."""
        import csv

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow(["FFT Power Spectrum Analysis Results"])
            writer.writerow(
                [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
            )
            writer.writerow([])

            # Summary table
            writer.writerow(["ROI Summary"])
            writer.writerow(
                [
                    "ROI",
                    "Dominant Period (hours)",
                    "Dominant Frequency (Hz)",
                    "Spectral Power",
                    "Number of Peaks",
                    "Mean Activity",
                    "Std Activity",
                ]
            )

            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
                if "error" in result:
                    writer.writerow([roi_id, "Error", result["error"], "", "", "", ""])
                    continue

                dominant_period = result.get("dominant_period", 0)
                dominant_freq = result.get("dominant_frequency", 0)
                dominant_power = result.get("dominant_power", 0)
                n_peaks = len(result.get("frequency_peaks", []))
                mean_activity = result.get("mean_activity", 0)
                std_activity = result.get("std_activity", 0)

                writer.writerow(
                    [
                        roi_id,
                        f"{dominant_period:.2f}",
                        f"{dominant_freq:.6f}",
                        f"{dominant_power:.2e}",
                        n_peaks,
                        f"{mean_activity:.4f}",
                        f"{std_activity:.4f}",
                    ]
                )

            # Detailed peak information
            writer.writerow([])
            writer.writerow(["Detailed Peak Information"])
            writer.writerow(
                [
                    "ROI",
                    "Peak #",
                    "Period (hours)",
                    "Frequency (Hz)",
                    "Power",
                    "Prominence",
                ]
            )

            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
                if "error" in result:
                    continue

                peaks = result.get("frequency_peaks", [])
                for idx, peak in enumerate(peaks[:5], 1):  # Top 5 peaks
                    writer.writerow(
                        [
                            roi_id,
                            idx,
                            f"{peak['period_hours']:.2f}",
                            f"{peak['frequency_hz']:.6f}",
                            f"{peak['power']:.2e}",
                            f"{peak['prominence']:.2e}",
                        ]
                    )

    def _export_fft_to_excel(self, file_path: str):
        """Export FFT Power Spectrum results to Excel format."""
        import pandas as pd

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary_data = []
            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
                if "error" in result:
                    summary_data.append(
                        {
                            "ROI": roi_id,
                            "Error": result["error"],
                            "Dominant_Period_h": None,
                            "Dominant_Frequency_Hz": None,
                            "Spectral_Power": None,
                            "N_Peaks": None,
                            "Mean_Activity": None,
                            "Std_Activity": None,
                        }
                    )
                    continue

                summary_data.append(
                    {
                        "ROI": roi_id,
                        "Error": None,
                        "Dominant_Period_h": result.get("dominant_period", 0),
                        "Dominant_Frequency_Hz": result.get("dominant_frequency", 0),
                        "Spectral_Power": result.get("dominant_power", 0),
                        "N_Peaks": len(result.get("frequency_peaks", [])),
                        "Mean_Activity": result.get("mean_activity", 0),
                        "Std_Activity": result.get("std_activity", 0),
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 2: All Peaks
            all_peaks_data = []
            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
                if "error" in result:
                    continue

                peaks = result.get("frequency_peaks", [])
                for idx, peak in enumerate(peaks[:10], 1):  # Top 10 peaks
                    all_peaks_data.append(
                        {
                            "ROI": roi_id,
                            "Peak_Number": idx,
                            "Period_hours": peak["period_hours"],
                            "Frequency_Hz": peak["frequency_hz"],
                            "Power": peak["power"],
                            "Prominence": peak["prominence"],
                        }
                    )

            if all_peaks_data:
                peaks_df = pd.DataFrame(all_peaks_data)
                peaks_df.to_excel(writer, sheet_name="Frequency_Peaks", index=False)

            # Sheet 3: Full Power Spectra (for plotting)
            # Export complete power spectrum data for each ROI
            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
                if "error" in result:
                    continue

                # Get relevant period range data
                relevant_periods = result.get("relevant_periods", [])
                relevant_power = result.get("relevant_power", [])

                if len(relevant_periods) > 0 and len(relevant_power) > 0:
                    spectrum_data = pd.DataFrame(
                        {
                            "Period_hours": relevant_periods,
                            "Power": relevant_power,
                        }
                    )
                    # Limit to reasonable number of points for Excel (max 10000)
                    if len(spectrum_data) > 10000:
                        # Downsample by taking every nth point
                        step = len(spectrum_data) // 10000 + 1
                        spectrum_data = spectrum_data.iloc[::step]

                    spectrum_data.to_excel(
                        writer, sheet_name=f"ROI_{roi_id}_Spectrum", index=False
                    )

            # Sheet 4: Sleep Phase Results (if available)
            sleep_results = self.fisher_analysis_results.get("sleep_phase_results", {})
            if sleep_results:
                sleep_roi_data = {
                    k: v for k, v in sleep_results.items() if isinstance(k, int)
                }
                if sleep_roi_data:
                    sleep_summary_data = []
                    for roi_id, result in sorted(sleep_roi_data.items()):
                        if "error" in result:
                            continue
                        sleep_summary_data.append(
                            {
                                "ROI": roi_id,
                                "Dominant_Period_h": result.get("dominant_period", 0),
                                "Dominant_Power": result.get("dominant_power", 0),
                                "Significant": result.get("is_significant", False),
                            }
                        )
                    if sleep_summary_data:
                        sleep_df = pd.DataFrame(sleep_summary_data)
                        sleep_df.to_excel(
                            writer, sheet_name="Sleep_Phase_Summary", index=False
                        )

            # Sheet 5: Parameters
            has_sleep = bool(sleep_results)
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Analysis Method",
                        "Generated",
                        "Sleep Phase Calculated",
                    ],
                    "Value": [
                        "FFT Power Spectrum",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Yes" if has_sleep else "No",
                    ],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

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

            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
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
            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
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

            # Sheet 2: Full Periodograms (for plotting)
            # Export complete periodogram data for each ROI
            for roi_id, result in sorted(
                {
                    k: v
                    for k, v in self.fisher_analysis_results.items()
                    if isinstance(k, int)
                }.items()
            ):
                if "error" in result:
                    continue

                periodogram = result.get("periodogram", {})
                periods = periodogram.get("periods", [])
                z_scores = periodogram.get("z_scores", [])

                if len(periods) > 0 and len(z_scores) > 0:
                    periodogram_data = pd.DataFrame(
                        {
                            "Period_hours": periods,
                            "Z_Score": z_scores,
                        }
                    )
                    # Limit to reasonable number of points for Excel (max 10000)
                    if len(periodogram_data) > 10000:
                        # Downsample by taking every nth point
                        step = len(periodogram_data) // 10000 + 1
                        periodogram_data = periodogram_data.iloc[::step]

                    periodogram_data.to_excel(
                        writer, sheet_name=f"ROI_{roi_id}_Periodogram", index=False
                    )

            # Sheet 3: Sleep Phase Results (if available)
            sleep_results = self.fisher_analysis_results.get("sleep_phase_results", {})
            if sleep_results:
                sleep_roi_data = {
                    k: v for k, v in sleep_results.items() if isinstance(k, int)
                }
                if sleep_roi_data:
                    sleep_summary_data = []
                    for roi_id, result in sorted(sleep_roi_data.items()):
                        if "error" in result:
                            continue
                        periodogram = result.get("periodogram", {})
                        sleep_summary_data.append(
                            {
                                "ROI": roi_id,
                                "Significant Rhythm": periodogram.get(
                                    "is_significant", False
                                ),
                                "Dominant Period (hours)": periodogram.get(
                                    "dominant_period", 0
                                ),
                                "Z-Score": periodogram.get("dominant_z_score", 0),
                                "P-Value": periodogram.get("p_value", 1.0),
                            }
                        )
                    if sleep_summary_data:
                        sleep_df = pd.DataFrame(sleep_summary_data)
                        sleep_df.to_excel(
                            writer, sheet_name="Sleep_Phase_Summary", index=False
                        )

            # Sheet 4: Parameters
            has_sleep = bool(sleep_results)
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Minimum Period",
                        "Maximum Period",
                        "Significance Level",
                        "Sampling Interval",
                        "Sleep Phase Calculated",
                    ],
                    "Value": [
                        f"{self.fisher_min_period.value():.1f} hours",
                        f"{self.fisher_max_period.value():.1f} hours",
                        f"{self.fisher_significance.value():.3f}",
                        f"{self.frame_interval.value():.1f} seconds",
                        "Yes" if has_sleep else "No",
                    ],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

    def _export_similarity_to_excel(self, file_path: str, similarity_results: Dict):
        """Export ROI Similarity results to Excel format."""
        import pandas as pd
        import numpy as np

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Correlation Matrix
            corr_matrix = similarity_results.get("correlation_matrix", np.array([]))
            roi_ids = similarity_results.get("roi_ids", [])

            if len(corr_matrix) > 0:
                corr_df = pd.DataFrame(
                    corr_matrix,
                    index=[f"ROI {r}" for r in roi_ids],
                    columns=[f"ROI {r}" for r in roi_ids],
                )
                corr_df.to_excel(writer, sheet_name="Correlation_Matrix")

            # Sheet 2: Pairwise Correlations
            pairwise_data = []
            for i, roi1 in enumerate(roi_ids):
                for j, roi2 in enumerate(roi_ids):
                    if i < j:  # Upper triangle only
                        correlation = corr_matrix[i, j]
                        pairwise_data.append(
                            {
                                "ROI_1": roi1,
                                "ROI_2": roi2,
                                "Correlation": correlation,
                                "Status": (
                                    "Synchronized"
                                    if correlation > 0.7
                                    else ("Moderate" if correlation > 0.3 else "Low")
                                ),
                            }
                        )

            pairwise_df = pd.DataFrame(pairwise_data)
            pairwise_df = pairwise_df.sort_values("Correlation", ascending=False)
            pairwise_df.to_excel(
                writer, sheet_name="Pairwise_Correlations", index=False
            )

            # Sheet 3: Clustering
            if "clustering" in similarity_results:
                cluster_info = similarity_results["clustering"]
                cluster_data = []
                for cluster_id, roi_list in cluster_info.get("clusters", {}).items():
                    for roi in roi_list:
                        cluster_data.append(
                            {
                                "ROI": roi,
                                "Cluster": cluster_id,
                                "Cluster_Size": len(roi_list),
                            }
                        )

                cluster_df = pd.DataFrame(cluster_data)
                cluster_df = cluster_df.sort_values(["Cluster", "ROI"])
                cluster_df.to_excel(writer, sheet_name="Clusters", index=False)

    def _export_coherence_to_excel(self, file_path: str, coherence_results: Dict):
        """Export Coherence Analysis results to Excel format."""
        import pandas as pd
        import numpy as np

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Coherence Matrix
            coherence_matrix = coherence_results.get("coherence_matrix", np.array([]))
            roi_ids = coherence_results.get("roi_ids", [])
            target_period = coherence_results.get("target_period_hours", 24.0)

            if len(coherence_matrix) > 0:
                coherence_df = pd.DataFrame(
                    coherence_matrix,
                    index=[f"ROI {r}" for r in roi_ids],
                    columns=[f"ROI {r}" for r in roi_ids],
                )
                coherence_df.to_excel(writer, sheet_name="Coherence_Matrix")

            # Sheet 2: Pairwise Coherence
            pairwise_coherence = coherence_results.get("pairwise_coherence", {})
            pairwise_data = []

            for (roi1, roi2), result in pairwise_coherence.items():
                pairwise_data.append(
                    {
                        "ROI_1": roi1,
                        "ROI_2": roi2,
                        "Coherence": result.get("circadian_coherence", 0),
                        "Coherence_Period": result.get(
                            "circadian_period", target_period
                        ),
                        "Max_Coherence": result.get("max_coherence", 0),
                        "Max_Coherence_Period": result.get("max_coherence_period", 0),
                        "Synchronized": (
                            "Yes" if result.get("is_synchronized", False) else "No"
                        ),
                    }
                )

            pairwise_df = pd.DataFrame(pairwise_data)
            pairwise_df = pairwise_df.sort_values("Coherence", ascending=False)
            pairwise_df.to_excel(writer, sheet_name="Pairwise_Coherence", index=False)

            # Sheet 3: Parameters
            params_df = pd.DataFrame(
                {
                    "Parameter": ["Target Period", "Number of ROIs"],
                    "Value": [f"{target_period:.1f} hours", str(len(roi_ids))],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

    def _export_phase_clustering_to_excel(self, file_path: str, phase_results: Dict):
        """Export Phase Clustering results to Excel format."""
        import pandas as pd

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Sheet 1: Phase Clusters
            clusters = phase_results.get("clusters", {})
            cluster_data = []

            for cluster_name, cluster_info in clusters.items():
                roi_list = cluster_info.get("rois", [])
                for roi in roi_list:
                    cluster_data.append(
                        {
                            "Cluster": cluster_name,
                            "ROI": roi,
                            "Cluster_Size": len(roi_list),
                        }
                    )

            cluster_df = pd.DataFrame(cluster_data)
            cluster_df = cluster_df.sort_values(["Cluster", "ROI"])
            cluster_df.to_excel(writer, sheet_name="Phase_Clusters", index=False)

            # Sheet 2: Individual ROI Phases
            roi_phases = phase_results.get("roi_phases", {})
            phase_data = []

            for roi_id, phase_info in sorted(roi_phases.items()):
                phase_data.append(
                    {
                        "ROI": roi_id,
                        "Peak_Time_Hours": phase_info.get("peak_time_hours", 0),
                        "Phase_Radians": phase_info.get("phase_radians", 0),
                        "Amplitude": phase_info.get("amplitude", 0),
                        "Mean_Activity": phase_info.get("mean_activity", 0),
                    }
                )

            phase_df = pd.DataFrame(phase_data)
            phase_df.to_excel(writer, sheet_name="ROI_Phases", index=False)

            # Sheet 3: Parameters
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Dominant Period",
                        "Total ROIs",
                        "Number of Clusters",
                    ],
                    "Value": [
                        f"{phase_results.get('dominant_period_hours', 0):.1f} hours",
                        str(phase_results.get("n_rois", 0)),
                        str(len(clusters)),
                    ],
                }
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

    def _create_fisher_plot(self, fisher_results: Dict[int, Dict]):
        """Create and display periodogram plot for Fisher analysis results."""
        try:
            import matplotlib.pyplot as plt
            from qtpy.QtGui import QPixmap
            import io

            # Close old figure if it exists to free memory
            if hasattr(self, "fisher_plot_figure") and self.fisher_plot_figure:
                self.fisher_plot_figure = None
                self.fisher_plot_figure = None

            # Get data source name for titles
            data_source_index = self.data_source_combo.currentIndex()
            data_source = (
                "Fraction Movement" if data_source_index == 0 else "Raw Intensity"
            )

            # Check if sleep phase results are available
            sleep_results = fisher_results.get("sleep_phase_results", None)
            has_sleep = sleep_results is not None and len(sleep_results) > 0

            # Create figure with subplots (only count integer ROI keys)
            roi_only_results = {
                k: v for k, v in fisher_results.items() if isinstance(k, int)
            }
            n_rois = len(roi_only_results)

            # Determine layout - max 3 columns
            n_cols = min(3, n_rois)
            n_rows_per_section = (n_rois + n_cols - 1) // n_cols

            has_population = (
                n_rois >= 2
                and hasattr(self, "chk_cosinor_population")
                and self.chk_cosinor_population.isChecked()
            )

            # If we have sleep data, double the rows (Activity on top, Sleep on bottom)
            if has_sleep:
                total_rows = n_rows_per_section * 2
                fig_height = 3.5 * total_rows + 1.5
            else:
                total_rows = n_rows_per_section
                fig_height = 3.5 * total_rows + 0.5

            if has_population:
                fig_height += 4.0

            fig_width = 4 * n_cols
            from matplotlib.figure import Figure as _Figure
            from matplotlib.gridspec import GridSpec as _GridSpec
            fig = _Figure(figsize=(fig_width, fig_height))

            # Two-GridSpec layout: data rows on top, population row at bottom
            # with an explicit gap so the two sections never overlap.
            if has_population:
                _pop_h   = 2.8  # inches reserved for population axes
                _gap_h   = 0.7  # inches gap between data section and population
                _bot_pad = 0.25 # inches bottom padding
                _pop_bot = _bot_pad / fig_height
                _pop_top = (_pop_h + _bot_pad) / fig_height
                _dat_bot = (_pop_h + _bot_pad + _gap_h) / fig_height
                gs = _GridSpec(total_rows, n_cols, figure=fig,
                               left=0.08, right=0.97,
                               top=0.96, bottom=_dat_bot,
                               hspace=0.55, wspace=0.35)
                gs_pop = _GridSpec(1, n_cols, figure=fig,
                                   left=0.08, right=0.97,
                                   top=_pop_top, bottom=_pop_bot,
                                   wspace=0.35)
            else:
                gs = _GridSpec(total_rows, n_cols, figure=fig,
                               left=0.08, right=0.97,
                               top=0.96, bottom=0.05,
                               hspace=0.55, wspace=0.35)
                gs_pop = None

            axes = np.array(
                [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                 for r in range(total_rows)]
            )

            # Population row: split into Activity + Sleep panels when sleep is present
            if has_population:
                if has_sleep and n_cols >= 2:
                    mid = n_cols // 2
                    ax_pop = fig.add_subplot(gs_pop[0, :mid])
                    ax_pop_sleep = fig.add_subplot(gs_pop[0, mid:])
                else:
                    ax_pop = fig.add_subplot(gs_pop[0, :])
                    ax_pop_sleep = None
            else:
                ax_pop = None
                ax_pop_sleep = None

            # Check if max period was capped for any ROI
            cap_note = ""
            for _roi_id, _roi_res in roi_only_results.items():
                _peri = _roi_res.get("periodogram", {})
                if _peri.get("period_capped", False):
                    _actual = _peri.get("actual_max_period", 0)
                    cap_note = f"  (max period capped at {_actual:.1f}h = recording / 2)"
                    break

            if has_sleep:
                fig.suptitle(
                    f"Chi² Periodogram  —  {data_source}{cap_note}{self._get_recording_start_str()}",
                    fontsize=11,
                    fontweight="bold",
                    y=0.99,
                )
            else:
                fig.suptitle(
                    f"Chi² Periodogram  —  Activity from {data_source}{cap_note}{self._get_recording_start_str()}",
                    fontsize=11,
                    fontweight="bold",
                    y=0.99,
                )

            # Helper function to plot a section (Activity or Sleep)
            def plot_section(results_dict, start_row, section_label):
                # Filter to integer keys only
                roi_items = {
                    k: v for k, v in results_dict.items() if isinstance(k, int)
                }

                for idx, (roi_id, result) in enumerate(sorted(roi_items.items())):
                    row = start_row + (idx // n_cols)
                    col = idx % n_cols
                    ax = axes[row, col]

                    periodogram = result.get("periodogram", {})

                    # Get ROI-specific color
                    roi_color = (
                        self.roi_colors.get(roi_id, f"C{idx}")
                        if hasattr(self, "roi_colors")
                        else f"C{idx}"
                    )

                    if "error" in result or "error" in periodogram:
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
                        periods = periodogram.get("periods", [])
                        z_scores = periodogram.get("z_scores", [])
                        critical_z = periodogram.get("critical_z", 0)
                        is_significant = periodogram.get("is_significant", False)

                        ax.plot(
                            periods,
                            z_scores,
                            color=roi_color,
                            linewidth=1.5,
                            label="Z-score",
                        )

                        if critical_z > 0:
                            ax.axhline(
                                y=critical_z,
                                color="gray",
                                linestyle="--",
                                linewidth=1,
                                label="Significance",
                            )

                        if is_significant:
                            dominant_period = periodogram.get("dominant_period", 0)
                            dominant_z = periodogram.get("dominant_z_score", 0)
                            ax.axvline(
                                x=dominant_period,
                                color=roi_color,
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.5,
                            )
                            ax.plot(
                                dominant_period,
                                dominant_z,
                                "o",
                                color=roi_color,
                                markersize=8,
                                markeredgecolor="black",
                                markeredgewidth=1,
                                label=f"Peak: {dominant_period:.1f}h",
                            )

                        ax.set_xlabel("Period (h)", fontsize=9)
                        ax.set_ylabel("Z-score", fontsize=9)
                        ax.tick_params(axis="both", labelsize=8)
                        title_weight = "bold" if is_significant else "normal"
                        # Include section label in ROI title
                        ax.set_title(
                            f"ROI {roi_id} - {section_label}",
                            fontsize=9,
                            color=roi_color,
                            fontweight=title_weight,
                            pad=4,
                            loc="left",
                        )
                        ax.legend(fontsize=7, loc="best")
                        ax.grid(True, alpha=0.3)

                        # Per-ROI Y-axis scaling
                        if len(z_scores) > 0:
                            roi_y_max = max(z_scores) * 1.1
                            ax.set_ylim(
                                0,
                                max(
                                    roi_y_max,
                                    critical_z * 1.2 if critical_z > 0 else roi_y_max,
                                ),
                            )

                        # Set x-axis to actual data range (analysis caps max at duration/2)
                        if len(periods) > 0:
                            ax.set_xlim(min(periods), max(periods))

                # Hide unused subplots in this section
                for idx in range(n_rois, n_rows_per_section * n_cols):
                    row = start_row + (idx // n_cols)
                    col = idx % n_cols
                    if row < start_row + n_rows_per_section:
                        axes[row, col].axis("off")

            # Plot Activity section
            plot_section(roi_only_results, 0, "Activity")

            # Plot Sleep section if available
            if has_sleep:
                plot_section(sleep_results, n_rows_per_section, "Sleep")

            # --- Population mean panel helper ---
            def plot_population_mean(ax, results_dict, title="Population Mean Periodogram"):
                all_p, all_z, all_crit = [], [], []
                for res in results_dict.values():
                    if not isinstance(res, dict):
                        continue
                    peri = res.get("periodogram", {})
                    if "error" not in res and "error" not in peri:
                        p = np.array(peri.get("periods", []))
                        z = np.array(peri.get("z_scores", []))
                        cz = peri.get("critical_z", 0)
                        if len(p) > 1 and len(p) == len(z):
                            all_p.append(p)
                            all_z.append(z)
                            if cz > 0:
                                all_crit.append(cz)

                if len(all_p) >= 2:
                    # Sort ascending for correct interp (periods come in descending order)
                    sorted_pairs = [
                        (np.sort(p), z[np.argsort(p)])
                        for p, z in zip(all_p, all_z)
                    ]
                    all_p_s = [sp for sp, _ in sorted_pairs]
                    all_z_s = [sz for _, sz in sorted_pairs]
                    p_min = max(a[0]  for a in all_p_s)
                    p_max = min(a[-1] for a in all_p_s)
                    if p_max > p_min:
                        grid = np.linspace(p_min, p_max, 300)
                        interp = np.array([np.interp(grid, p, z) for p, z in zip(all_p_s, all_z_s)])
                        mean_z = interp.mean(axis=0)
                        sem_z = interp.std(axis=0) / np.sqrt(len(interp))

                        ax.plot(grid, mean_z, color="black", linewidth=2,
                                label=f"Mean Z (n={len(interp)})")
                        ax.fill_between(grid, mean_z - sem_z, mean_z + sem_z,
                                        alpha=0.25, color="gray", label="±SEM")

                        if all_crit:
                            mean_crit = float(np.mean(all_crit))
                            ax.axhline(mean_crit, color="red", linestyle="--",
                                       linewidth=1.2, label=f"Mean threshold ({mean_crit:.2f})")

                        sig_periods = [
                            res.get("periodogram", {}).get("dominant_period")
                            for res in results_dict.values()
                            if isinstance(res, dict)
                            and res.get("periodogram", {}).get("is_significant")
                            and res.get("periodogram", {}).get("dominant_period") is not None
                        ]
                        use_mean_peak = (
                            hasattr(self, "population_peak_mode")
                            and self.population_peak_mode.currentText() == "Mean"
                        )
                        if use_mean_peak:
                            peak_p = float(grid[np.argmax(mean_z)])
                            ax.axvline(peak_p, color="steelblue", linestyle="--",
                                       linewidth=1.5, label=f"Mean peak: {peak_p:.1f}h")
                        elif sig_periods:
                            med_p = float(np.median(sig_periods))
                            ax.axvline(med_p, color="steelblue", linestyle="--",
                                       linewidth=1.5, label=f"Median peak: {med_p:.1f}h")
                        if sig_periods:
                            n_sig = len(sig_periods)
                            ax.text(0.97, 0.95,
                                    f"Significant: {n_sig}/{len(all_p)}",
                                    transform=ax.transAxes, fontsize=8,
                                    va="top", ha="right",
                                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

                        ax.set_xlabel("Period (h)", fontsize=9)
                        ax.set_ylabel("Z-score", fontsize=9)
                        ax.set_title(title, fontsize=10, fontweight="bold")
                        ax.legend(fontsize=8, loc="upper left")
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis="both", labelsize=8)
                        ax.set_ylim(bottom=0)
                        return True
                ax.text(0.5, 0.5, "Not enough valid ROIs for population mean",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.axis("off")
                return False

            # --- Draw population mean panels ---
            if has_population and ax_pop is not None:
                pop_title = "Population Mean — Activity" if has_sleep else "Population Mean Periodogram"
                plot_population_mean(ax_pop, roi_only_results, pop_title)

            if has_sleep and ax_pop_sleep is not None:
                sleep_roi_results = {k: v for k, v in sleep_results.items() if isinstance(k, int)}
                plot_population_mean(ax_pop_sleep, sleep_roi_results, "Population Mean — Sleep")

            # Store figure for later export
            self.fisher_plot_figure = fig

            # Convert to QPixmap and display with higher DPI
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())

            # Scale pixmap to fit canvas while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.fisher_plot_canvas.size(),
                1,  # Qt.KeepAspectRatio
                1,  # Qt.SmoothTransformation
            )
            self.fisher_plot_canvas.setPixmap(scaled_pixmap)

            # Enable pop-out button after successful plot creation
            if hasattr(self, "btn_popout_plot"):
                self.btn_popout_plot.setEnabled(True)
                if hasattr(self, "btn_save_fisher_plot"):
                    self.btn_save_fisher_plot.setEnabled(True)

            # Note: Don't close fig - we need it for export

        except Exception as e:
            self._log_message(f"⚠️ Could not create Fisher plot: {e}")
            import traceback

            traceback.print_exc()

    def _open_plot_window(self):
        """Open the current plot in a separate, resizable window."""
        if not hasattr(self, "fisher_plot_figure") or self.fisher_plot_figure is None:
            self._log_message("⚠️ No plot available to display")
            return

        try:
            from qtpy.QtWidgets import (
                QDialog,
                QVBoxLayout,
                QHBoxLayout,
                QPushButton,
                QScrollArea,
            )
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.backends.backend_qt5agg import (
                NavigationToolbar2QT as NavigationToolbar,
            )

            # Create dialog window
            dialog = QDialog(self)
            dialog.setWindowTitle("Rhythmic Pattern Analysis - Plot View")
            dialog.resize(1400, 900)  # Larger default size

            layout = QVBoxLayout()
            dialog.setLayout(layout)

            # Create matplotlib canvas
            canvas = FigureCanvas(self.fisher_plot_figure)
            canvas.setMinimumSize(800, 600)

            # Add navigation toolbar for zoom/pan
            toolbar = NavigationToolbar(canvas, dialog)
            toolbar.setStyleSheet(
                "QToolBar { background-color: #f0f0f0; border: none; padding: 2px; }"
                "QToolButton { background-color: #f0f0f0; border: 1px solid #ccc;"
                "  border-radius: 3px; padding: 3px; margin: 1px; color: #222; }"
                "QToolButton:hover { background-color: #dde8f5; border-color: #4a90d9; }"
                "QToolButton:checked { background-color: #c8ddf5; border-color: #4a90d9; }"
            )
            layout.addWidget(toolbar)

            # Add canvas in scroll area for very large plots
            scroll_area = QScrollArea()
            scroll_area.setWidget(canvas)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)

            # Add toggle buttons + close button
            button_layout = QHBoxLayout()

            # Y-axis toggle
            btn_toggle_yaxis = QPushButton("Hide Y-Axis")
            btn_toggle_yaxis._yaxis_visible = True

            def toggle_yaxis():
                btn_toggle_yaxis._yaxis_visible = not btn_toggle_yaxis._yaxis_visible
                visible = btn_toggle_yaxis._yaxis_visible
                for ax in self.fisher_plot_figure.get_axes():
                    ax.yaxis.set_visible(visible)
                    ax.set_ylabel(ax.get_ylabel() if visible else "")
                btn_toggle_yaxis.setText("Hide Y-Axis" if visible else "Show Y-Axis")
                canvas.draw_idle()

            btn_toggle_yaxis.clicked.connect(toggle_yaxis)
            button_layout.addWidget(btn_toggle_yaxis)

            # Legend toggle
            btn_toggle_legend = QPushButton("Hide Legend")
            btn_toggle_legend._legend_visible = True

            def toggle_legend():
                btn_toggle_legend._legend_visible = not btn_toggle_legend._legend_visible
                visible = btn_toggle_legend._legend_visible
                for ax in self.fisher_plot_figure.get_axes():
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.set_visible(visible)
                btn_toggle_legend.setText("Hide Legend" if visible else "Show Legend")
                canvas.draw_idle()

            btn_toggle_legend.clicked.connect(toggle_legend)
            button_layout.addWidget(btn_toggle_legend)

            button_layout.addStretch()

            btn_save = QPushButton("Save Plot...")
            btn_save.clicked.connect(lambda: self._save_plot_from_dialog(canvas))
            button_layout.addWidget(btn_save)

            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.close)
            button_layout.addWidget(btn_close)

            layout.addLayout(button_layout)

            # Show dialog
            dialog.exec_()

        except Exception as e:
            self._log_message(f"⚠️ Could not open plot window: {e}")
            import traceback

            traceback.print_exc()

    def _save_plot_from_dialog(self, canvas):
        """Save plot from dialog window."""
        try:
            from qtpy.QtWidgets import QFileDialog
            import os

            # Get save location
            default_name = "rhythmic_pattern_plot.png"
            if hasattr(self, "current_fisher_method"):
                method_names = [
                    "fisher",
                    "fft",
                    "cosinor",
                    "similarity",
                    "coherence",
                    "phase",
                ]
                method_name = (
                    method_names[self.current_fisher_method]
                    if self.current_fisher_method < len(method_names)
                    else "plot"
                )
                default_name = f"rhythmic_pattern_{method_name}.png"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                default_name,
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
            )

            if file_path:
                # Determine format from extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf":
                    canvas.figure.savefig(
                        file_path, format="pdf", dpi=300, bbox_inches="tight"
                    )
                elif ext == ".svg":
                    canvas.figure.savefig(file_path, format="svg", bbox_inches="tight")
                else:
                    canvas.figure.savefig(
                        file_path, format="png", dpi=300, bbox_inches="tight"
                    )

                self._log_message(f"✓ Plot saved to: {file_path}")

        except Exception as e:
            self._log_message(f"⚠️ Could not save plot: {e}")
            import traceback

            traceback.print_exc()

    def _save_fisher_plot(self):
        """Save the currently displayed periodogram / analysis plot as an image file."""
        if not hasattr(self, "fisher_plot_figure") or self.fisher_plot_figure is None:
            self._log_message("⚠️ No plot to save — run an analysis first.")
            return

        # Build a default filename from the HDF5 file and current method
        try:
            base = os.path.splitext(os.path.basename(self.file_path))[0]
        except Exception:
            base = "analysis"
        method_names = {0: "fisher", 1: "fft", 2: "cosinor", 3: "similarity",
                        4: "coherence", 5: "phase_clustering"}
        method_idx = getattr(self, "current_fisher_method", 0)
        method_tag = method_names.get(method_idx, "plot")
        default_name = f"{base}_{method_tag}_plot.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            default_name,
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )
        if not file_path:
            return

        try:
            dpi = self.plot_dpi_spin.value() if hasattr(self, "plot_dpi_spin") else 150
            self.fisher_plot_figure.savefig(
                file_path, dpi=dpi, bbox_inches="tight", facecolor="white"
            )
            self._log_message(f"✓ Plot saved: {os.path.basename(file_path)}")
        except Exception as e:
            self._log_message(f"⚠️ Could not save plot: {e}")

    # ===================================================================
    # FRAME VIEWER METHODS
    # ===================================================================

