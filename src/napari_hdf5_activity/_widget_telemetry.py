"""_widget_telemetry.py — TelemetryMixin for HDF5AnalysisWidget.

Provides the HDF5 Telemetry tab: metadata tree display, timeseries
selection and plotting, and plot export.  Mixed into HDF5AnalysisWidget
so all methods share the same ``self`` namespace.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

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

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from ._io_abstraction import open_file_reader
    IO_ABSTRACTION_AVAILABLE = True
except ImportError:
    IO_ABSTRACTION_AVAILABLE = False


class TelemetryMixin:
    """Mixin providing all Telemetry-tab functionality.

    Requires that the host class (HDF5AnalysisWidget) provides:
    - self.file_path (str)
    - self.tab_telemetry (QWidget)
    - self._log_message(msg: str)
    """

    # Timeseries categories and units for telemetry display
    TELEMETRY_CATEGORIES = {
        "Timing": [
            "frame_drift", "cumulative_drift", "actual_intervals", "expected_intervals",
        ],
        "Environment": ["temperature", "humidity"],
        "LED": [
            "led_white_power_percent", "led_ir_power_percent", "led_power_percent",
            "led_duration_ms", "led_sync_success",
        ],
        "Frame Stats": ["frame_mean", "frame_max", "frame_min", "frame_std"],
    }

    TIMESERIES_UNITS = {
        "frame_drift": "s", "cumulative_drift": "s", "actual_intervals": "s",
        "expected_intervals": "s", "temperature": "\u00b0C", "humidity": "%",
        "led_white_power_percent": "%", "led_ir_power_percent": "%",
        "led_power_percent": "%", "led_duration_ms": "ms",
        "led_sync_success": "bool", "frame_mean": "px intensity",
        "frame_max": "px intensity", "frame_min": "px intensity",
        "frame_std": "px intensity",
    }


    # --- paste methods here ---

    def setup_telemetry_tab(self):
        """Setup the HDF5 Telemetry tab for viewing file metadata and timeseries."""
        layout = QVBoxLayout()
        self.tab_telemetry.setLayout(layout)

        # --- Section 1: File & Frame Metadata ---
        meta_group = QGroupBox("File & Frame Metadata")
        meta_layout = QVBoxLayout()
        meta_group.setLayout(meta_layout)

        # Load button
        btn_row = QHBoxLayout()
        self.btn_load_telemetry = QPushButton("Load Metadata")
        self.btn_load_telemetry.setToolTip("Read metadata and timeseries info from the loaded HDF5 file")
        self.btn_load_telemetry.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.btn_load_telemetry.clicked.connect(self._load_telemetry_metadata)
        btn_row.addWidget(self.btn_load_telemetry)
        btn_row.addStretch()
        meta_layout.addLayout(btn_row)

        # Tree widget for metadata display
        self.telemetry_tree = QTreeWidget()
        self.telemetry_tree.setHeaderLabels(["Property", "Value"])
        self.telemetry_tree.setAlternatingRowColors(True)
        self.telemetry_tree.setMinimumHeight(200)
        self.telemetry_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.telemetry_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        meta_layout.addWidget(self.telemetry_tree)

        layout.addWidget(meta_group)

        # --- Section 2: Timeseries Plots ---
        ts_group = QGroupBox("Timeseries Data")
        ts_layout = QVBoxLayout()
        ts_group.setLayout(ts_layout)

        # Top row: dataset selector (left) + controls (right)
        selector_and_controls = QHBoxLayout()

        # Left: multi-select list of individual timeseries
        select_panel = QVBoxLayout()
        select_panel.addWidget(QLabel("Select datasets to plot:"))

        self.telemetry_list = QListWidget()
        self.telemetry_list.setSelectionMode(QListWidget.MultiSelection)
        self.telemetry_list.setMaximumHeight(160)
        self.telemetry_list.setToolTip(
            "Click to select/deselect individual timeseries. "
            "Use the category buttons for quick selection."
        )
        select_panel.addWidget(self.telemetry_list)

        # Quick-select buttons for categories
        cat_btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("All")
        btn_sel_all.setToolTip("Select all timeseries")
        btn_sel_all.clicked.connect(lambda: self._select_telemetry_items(all_=True))
        btn_sel_none = QPushButton("None")
        btn_sel_none.setToolTip("Deselect all timeseries")
        btn_sel_none.clicked.connect(lambda: self._select_telemetry_items(all_=False))
        cat_btn_row.addWidget(btn_sel_all)
        cat_btn_row.addWidget(btn_sel_none)

        for cat_name in list(self.TELEMETRY_CATEGORIES.keys()) + ["Other"]:
            btn = QPushButton(cat_name)
            btn.setToolTip(f"Select only {cat_name} timeseries")
            btn.clicked.connect(lambda checked, c=cat_name: self._select_telemetry_category(c))
            cat_btn_row.addWidget(btn)
        cat_btn_row.addStretch()
        select_panel.addLayout(cat_btn_row)

        selector_and_controls.addLayout(select_panel, stretch=1)

        # Right: x-axis control
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("X-Axis:"))
        self.telemetry_xaxis_combo = QComboBox()
        self.telemetry_xaxis_combo.addItems(["Frame Number", "Time (minutes)", "Time (hours)"])
        self.telemetry_xaxis_combo.setCurrentIndex(1)
        right_panel.addWidget(self.telemetry_xaxis_combo)
        right_panel.addStretch()
        selector_and_controls.addLayout(right_panel)

        ts_layout.addLayout(selector_and_controls)

        # Buttons row: Plot + Export
        btn_row2 = QHBoxLayout()
        self.btn_plot_telemetry = QPushButton("Plot Telemetry")
        self.btn_plot_telemetry.setToolTip("Plot selected timeseries")
        self.btn_plot_telemetry.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388E3C; }"
        )
        self.btn_plot_telemetry.clicked.connect(self._plot_telemetry)
        btn_row2.addWidget(self.btn_plot_telemetry)

        self.btn_save_telemetry_plot = QPushButton("Save Current Plot")
        self.btn_save_telemetry_plot.setToolTip("Save the current telemetry plot as PNG/PDF/SVG")
        self.btn_save_telemetry_plot.clicked.connect(self._save_telemetry_plot)
        btn_row2.addWidget(self.btn_save_telemetry_plot)

        self.btn_export_all_telemetry = QPushButton("Export All Plots")
        self.btn_export_all_telemetry.setToolTip(
            "Save each timeseries as individual plot + one combined overview"
        )
        self.btn_export_all_telemetry.clicked.connect(self._export_all_telemetry_plots)
        btn_row2.addWidget(self.btn_export_all_telemetry)

        btn_row2.addStretch()
        ts_layout.addLayout(btn_row2)

        # Matplotlib canvas for timeseries plots
        self.telemetry_figure = Figure(figsize=(10, 8), dpi=100)
        self.telemetry_canvas = FigureCanvas(self.telemetry_figure)
        self.telemetry_canvas.setMinimumHeight(500)
        ts_layout.addWidget(self.telemetry_canvas)

        layout.addWidget(ts_group)

        # Storage for loaded telemetry data
        self.telemetry_timeseries = {}
        self.telemetry_frame_interval = None

    def _load_telemetry_metadata(self):
        """Load and display HDF5 file metadata in the telemetry tree."""
        file_path = getattr(self, "file_path", None)
        if not file_path:
            self._log_message("No HDF5 file loaded. Please load a file first.")
            return

        try:
            self.telemetry_tree.clear()
            self._log_message(f"Loading telemetry from: {os.path.basename(file_path)}")

            # Use open_file_reader for format-agnostic attribute/key reading
            if IO_ABSTRACTION_AVAILABLE:
                from ._io_abstraction import open_file_reader
                with open_file_reader(file_path) as reader:
                    # --- File Info ---
                    file_item = QTreeWidgetItem(self.telemetry_tree, ["File Info", ""])
                    file_item.setExpanded(True)
                    file_size = os.path.getsize(file_path)
                    QTreeWidgetItem(file_item, ["Filename", os.path.basename(file_path)])
                    QTreeWidgetItem(file_item, ["Size", f"{file_size / (1024*1024):.1f} MB"])

                    # --- Root Attributes ---
                    root_attrs = reader.get_attrs("/")
                    if root_attrs:
                        attr_item = QTreeWidgetItem(self.telemetry_tree, ["Root Attributes", f"({len(root_attrs)} entries)"])
                        attr_item.setExpanded(True)
                        for key in sorted(root_attrs.keys()):
                            val = root_attrs[key]
                            if isinstance(val, bytes):
                                val = val.decode("utf-8", errors="replace")
                            QTreeWidgetItem(attr_item, [str(key), str(val)])

                    # --- Timeseries: read arrays for plotting ---
                    self.telemetry_timeseries = {}
                    self.telemetry_frame_interval = None

                    root_keys = reader.keys("/")

                    if "timeseries" in root_keys:
                        ts_keys = reader.keys("timeseries")
                        ts_tree_item = QTreeWidgetItem(self.telemetry_tree, [
                            "Timeseries", f"({len(ts_keys)} datasets)"
                        ])
                        ts_tree_item.setExpanded(True)

                        # Read frame interval from root attrs
                        for attr_name in ["frame_interval", "interval", "fps"]:
                            if attr_name in root_attrs:
                                val = float(root_attrs[attr_name])
                                if attr_name == "fps" and val > 0:
                                    self.telemetry_frame_interval = 1.0 / val
                                else:
                                    self.telemetry_frame_interval = val
                                break

                        for ds_name in sorted(ts_keys):
                            ds_path = f"timeseries/{ds_name}"
                            if reader.is_array(ds_path):
                                # Store array data for plotting
                                try:
                                    self.telemetry_timeseries[ds_name] = reader.read_all(ds_path)
                                except Exception:
                                    pass

                                try:
                                    shp = reader.shape(ds_path)
                                    dtp = reader.dtype(ds_path)
                                    shape_str = f"shape={shp}, dtype={dtp}"
                                except Exception:
                                    shape_str = ""

                                unit = self.TIMESERIES_UNITS.get(ds_name, "")
                                unit_str = f" [{unit}]" if unit else ""
                                QTreeWidgetItem(ts_tree_item, [
                                    f"{ds_name}{unit_str}", shape_str
                                ])

                                # Show attributes of timeseries datasets
                                ds_attrs = reader.get_attrs(ds_path)
                                if ds_attrs:
                                    ds_tree = ts_tree_item.child(ts_tree_item.childCount() - 1)
                                    for ak in ds_attrs:
                                        av = ds_attrs[ak]
                                        if isinstance(av, bytes):
                                            av = av.decode("utf-8", errors="replace")
                                        QTreeWidgetItem(ds_tree, [f"  @{ak}", str(av)])

            else:
                # Fallback: h5py only
                import h5py
                with h5py.File(file_path, "r") as f:
                    # --- File Info ---
                    file_item = QTreeWidgetItem(self.telemetry_tree, ["File Info", ""])
                    file_item.setExpanded(True)
                    file_size = os.path.getsize(file_path)
                    QTreeWidgetItem(file_item, ["Filename", os.path.basename(file_path)])
                    QTreeWidgetItem(file_item, ["Size", f"{file_size / (1024*1024):.1f} MB"])
                    QTreeWidgetItem(file_item, ["HDF5 Driver", str(f.driver)])

                    # --- Root Attributes ---
                    if f.attrs:
                        attr_item = QTreeWidgetItem(self.telemetry_tree, ["Root Attributes", f"({len(f.attrs)} entries)"])
                        attr_item.setExpanded(True)
                        for key in sorted(f.attrs.keys()):
                            val = f.attrs[key]
                            if isinstance(val, bytes):
                                val = val.decode("utf-8", errors="replace")
                            QTreeWidgetItem(attr_item, [str(key), str(val)])

                    # --- Timeseries: read arrays for plotting ---
                    self.telemetry_timeseries = {}
                    self.telemetry_frame_interval = None

                    if "timeseries" in f:
                        ts_group = f["timeseries"]
                        ts_tree_item = QTreeWidgetItem(self.telemetry_tree, [
                            "Timeseries", f"({len(ts_group)} datasets)"
                        ])
                        ts_tree_item.setExpanded(True)

                        # Read frame interval from root attrs
                        for attr_name in ["frame_interval", "interval", "fps"]:
                            if attr_name in f.attrs:
                                val = float(f.attrs[attr_name])
                                if attr_name == "fps" and val > 0:
                                    self.telemetry_frame_interval = 1.0 / val
                                else:
                                    self.telemetry_frame_interval = val
                                break

                        for ds_name in sorted(ts_group.keys()):
                            ds = ts_group[ds_name]
                            if hasattr(ds, "shape"):
                                # Store array data for plotting
                                try:
                                    self.telemetry_timeseries[ds_name] = ds[:]
                                except Exception:
                                    pass

                                unit = self.TIMESERIES_UNITS.get(ds_name, "")
                                unit_str = f" [{unit}]" if unit else ""
                                shape_str = f"shape={ds.shape}, dtype={ds.dtype}"
                                QTreeWidgetItem(ts_tree_item, [
                                    f"{ds_name}{unit_str}", shape_str
                                ])

                                # Show attributes of timeseries datasets
                                if ds.attrs:
                                    ds_tree = ts_tree_item.child(ts_tree_item.childCount() - 1)
                                    for ak in ds.attrs:
                                        av = ds.attrs[ak]
                                        if isinstance(av, bytes):
                                            av = av.decode("utf-8", errors="replace")
                                        QTreeWidgetItem(ds_tree, [f"  @{ak}", str(av)])

            # Try to populate the HDF5 datasets tree (HDF5-only, with fallback)
            try:
                import h5py
                with h5py.File(file_path, "r") as f:
                    datasets_item = QTreeWidgetItem(self.telemetry_tree, ["Datasets", ""])
                    datasets_item.setExpanded(True)
                    self._add_hdf5_items_to_tree(f, datasets_item)
            except Exception:
                pass  # Not an HDF5 file or h5py not available — skip recursive tree

            n_ts = len(self.telemetry_timeseries)
            self._log_message(f"Telemetry loaded: {n_ts} timeseries datasets found")

            # Populate the dataset selector list
            self._populate_telemetry_list()

            # Auto-plot all loaded timeseries
            if self.telemetry_timeseries:
                self._plot_telemetry()

        except Exception as e:
            self._log_message(f"Error loading telemetry: {e}")
            import traceback
            self._log_message(traceback.format_exc())

    def _add_hdf5_items_to_tree(self, h5_group, parent_item):
        """Recursively add HDF5 groups and datasets to the tree widget."""
        import h5py

        for key in sorted(h5_group.keys()):
            item = h5_group[key]
            if isinstance(item, h5py.Group):
                if key == "timeseries":
                    continue  # Shown separately
                group_node = QTreeWidgetItem(parent_item, [
                    f"{key}/", f"(Group, {len(item)} items)"
                ])
                self._add_hdf5_items_to_tree(item, group_node)
            elif isinstance(item, h5py.Dataset):
                info_parts = [f"shape={item.shape}", f"dtype={item.dtype}"]
                if item.compression:
                    info_parts.append(f"compression={item.compression}")
                QTreeWidgetItem(parent_item, [key, ", ".join(info_parts)])

    def _get_category_for_dataset(self, ds_name: str) -> str:
        """Return the category name for a given timeseries dataset name."""
        for cat_name, ds_list in self.TELEMETRY_CATEGORIES.items():
            if ds_name in ds_list:
                return cat_name
        return "Other"

    def _populate_telemetry_list(self):
        """Populate the dataset selector list with all available timeseries."""
        self.telemetry_list.clear()
        if not self.telemetry_timeseries:
            return

        # Group by category for ordered display
        categorized = {}
        for ds_name in self.telemetry_timeseries:
            cat = self._get_category_for_dataset(ds_name)
            categorized.setdefault(cat, []).append(ds_name)

        # Add items in category order
        cat_order = list(self.TELEMETRY_CATEGORIES.keys()) + ["Other"]
        for cat_name in cat_order:
            if cat_name not in categorized:
                continue
            for ds_name in categorized[cat_name]:
                unit = self.TIMESERIES_UNITS.get(ds_name, "")
                unit_str = f" [{unit}]" if unit else ""
                item = QListWidgetItem(f"[{cat_name}]  {ds_name}{unit_str}")
                item.setData(Qt.UserRole, ds_name)  # store raw name for lookup
                item.setSelected(True)
                self.telemetry_list.addItem(item)

    def _select_telemetry_items(self, all_: bool):
        """Select or deselect all items in the telemetry list."""
        for i in range(self.telemetry_list.count()):
            self.telemetry_list.item(i).setSelected(all_)

    def _select_telemetry_category(self, category: str):
        """Select only items belonging to the given category."""
        for i in range(self.telemetry_list.count()):
            item = self.telemetry_list.item(i)
            ds_name = item.data(Qt.UserRole)
            cat = self._get_category_for_dataset(ds_name)
            item.setSelected(cat == category)

    def _get_active_telemetry_series(self):
        """Return list of (category, dataset_name) for all selected items in the list."""
        active_series = []
        for item in self.telemetry_list.selectedItems():
            ds_name = item.data(Qt.UserRole)
            if ds_name in self.telemetry_timeseries:
                cat = self._get_category_for_dataset(ds_name)
                active_series.append((cat, ds_name))
        return active_series

    def _build_telemetry_xaxis(self, n_points):
        """Build x-axis array and label based on current x-axis mode selection."""
        import numpy as np

        xaxis_mode = self.telemetry_xaxis_combo.currentText()
        if xaxis_mode == "Frame Number":
            return np.arange(n_points), "Frame Number"
        elif xaxis_mode == "Time (hours)":
            interval = self.telemetry_frame_interval or 5.0
            return np.arange(n_points) * interval / 3600.0, "Time (hours)"
        else:  # Time (minutes)
            interval = self.telemetry_frame_interval or 5.0
            return np.arange(n_points) * interval / 60.0, "Time (minutes)"

    # Category colors for telemetry plots
    TELEMETRY_CAT_COLORS = {
        "Timing": "#2196F3",
        "Environment": "#FF9800",
        "LED": "#9C27B0",
        "Frame Stats": "#4CAF50",
        "Other": "#795548",
    }

    def _plot_telemetry(self):
        """Plot selected timeseries categories from the loaded HDF5 telemetry data."""
        if not self.telemetry_timeseries:
            self._log_message("No telemetry data loaded. Click 'Load Metadata' first.")
            return

        active_series = self._get_active_telemetry_series()
        if not active_series:
            self._log_message("No timeseries data available for selected categories.")
            return

        n_plots = len(active_series)

        # Clear and create subplots
        self.telemetry_figure.clear()

        # Dynamically adjust figure height based on number of plots
        height_per_plot = 1.5
        min_height = 4.0
        fig_height = max(min_height, n_plots * height_per_plot)
        self.telemetry_figure.set_size_inches(10, fig_height)
        self.telemetry_canvas.setMinimumHeight(int(fig_height * 100))

        axes = self.telemetry_figure.subplots(n_plots, 1, sharex=True, squeeze=False)

        x_label = "Frame Number"
        for idx, (cat_name, ds_name) in enumerate(active_series):
            ax = axes[idx, 0]
            data = self.telemetry_timeseries[ds_name]
            x, x_label = self._build_telemetry_xaxis(len(data))

            color = self.TELEMETRY_CAT_COLORS.get(cat_name, "#666666")
            unit = self.TIMESERIES_UNITS.get(ds_name, "")
            unit_str = f" [{unit}]" if unit else ""

            mean_val = float(np.mean(data))
            std_val = float(np.std(data))
            stat_label = f"μ = {mean_val:.3g}{' ' + unit if unit else ''},  σ = {std_val:.3g}"

            ax.plot(x, data, color=color, linewidth=0.8, alpha=0.9, label=stat_label)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

            ax.set_ylabel(f"{ds_name}{unit_str}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            # Category label on right side
            ax.text(
                1.01, 0.5, cat_name, transform=ax.transAxes,
                fontsize=7, color=color, alpha=0.7, va="center", rotation=270,
            )

        # X-axis label on bottom plot only
        if n_plots > 0:
            axes[-1, 0].set_xlabel(x_label, fontsize=9)

        self.telemetry_figure.suptitle(
            f"HDF5 Telemetry — {os.path.basename(getattr(self, 'file_path', '') or '')}",
            fontsize=10, y=1.0,
        )
        self.telemetry_figure.tight_layout()
        self.telemetry_canvas.draw()
        self._log_message(f"Plotted {n_plots} telemetry timeseries ({x_label})")

    def _save_telemetry_plot(self):
        """Save the current telemetry plot to a file."""
        if not self.telemetry_timeseries:
            self._log_message("No telemetry plot to save.")
            return

        try:
            base_name = os.path.splitext(
                os.path.basename(getattr(self, "file_path", "") or "telemetry")
            )[0]
            default_name = f"{base_name}_telemetry.png"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Telemetry Plot", default_name,
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
            )
            if file_path:
                ext = os.path.splitext(file_path)[1].lower()
                fmt = "pdf" if ext == ".pdf" else ("svg" if ext == ".svg" else "png")
                self.telemetry_figure.savefig(
                    file_path, format=fmt, dpi=300, bbox_inches="tight"
                )
                self._log_message(f"Telemetry plot saved: {file_path}")
        except Exception as e:
            self._log_message(f"Error saving telemetry plot: {e}")

    def _export_all_telemetry_plots(self):
        """Export individual plots for each timeseries + one combined overview."""
        if not self.telemetry_timeseries:
            self._log_message("No telemetry data loaded. Click 'Load Metadata' first.")
            return

        import numpy as np

        # Ask for output directory
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Telemetry Plots"
        )
        if not out_dir:
            return

        base_name = os.path.splitext(
            os.path.basename(getattr(self, "file_path", "") or "telemetry")
        )[0]

        saved_count = 0

        try:
            # Get all available series (ignore category filters for full export)
            all_known = set()
            for ds_list in self.TELEMETRY_CATEGORIES.values():
                all_known.update(ds_list)

            all_series = []
            for cat_name, ds_names in self.TELEMETRY_CATEGORIES.items():
                for ds_name in ds_names:
                    if ds_name in self.telemetry_timeseries:
                        all_series.append((cat_name, ds_name))
            for ds_name in sorted(self.telemetry_timeseries.keys()):
                if ds_name not in all_known:
                    all_series.append(("Other", ds_name))

            if not all_series:
                self._log_message("No timeseries data to export.")
                return

            # 1. Individual plots for each timeseries
            for cat_name, ds_name in all_series:
                fig, ax = Figure(figsize=(10, 3), dpi=150), None
                ax = fig.add_subplot(111)
                data = self.telemetry_timeseries[ds_name]
                x, x_label = self._build_telemetry_xaxis(len(data))

                color = self.TELEMETRY_CAT_COLORS.get(cat_name, "#666666")
                ax.plot(x, data, color=color, linewidth=0.8, alpha=0.9)

                unit = self.TIMESERIES_UNITS.get(ds_name, "")
                unit_str = f" [{unit}]" if unit else ""
                ax.set_ylabel(f"{ds_name}{unit_str}", fontsize=9)
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_title(f"{ds_name} ({cat_name})", fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                out_path = os.path.join(out_dir, f"{base_name}_{ds_name}.png")
                fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
                fig.clear()
                import matplotlib.pyplot as plt
                plt.close(fig)
                saved_count += 1

            # 2. Combined overview plot with all timeseries
            n_plots = len(all_series)
            height_per = 1.8
            fig_h = max(6.0, n_plots * height_per)
            fig_combined = Figure(figsize=(12, fig_h), dpi=150)
            axes = fig_combined.subplots(n_plots, 1, sharex=True, squeeze=False)

            x_label = "Frame Number"
            for idx, (cat_name, ds_name) in enumerate(all_series):
                ax = axes[idx, 0]
                data = self.telemetry_timeseries[ds_name]
                x, x_label = self._build_telemetry_xaxis(len(data))

                color = self.TELEMETRY_CAT_COLORS.get(cat_name, "#666666")
                ax.plot(x, data, color=color, linewidth=0.8, alpha=0.9)

                unit = self.TIMESERIES_UNITS.get(ds_name, "")
                unit_str = f" [{unit}]" if unit else ""
                ax.set_ylabel(f"{ds_name}{unit_str}", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)
                ax.text(
                    1.01, 0.5, cat_name, transform=ax.transAxes,
                    fontsize=6, color=color, alpha=0.7, va="center", rotation=270,
                )

            if n_plots > 0:
                axes[-1, 0].set_xlabel(x_label, fontsize=9)
            fig_combined.suptitle(
                f"HDF5 Telemetry Overview — {base_name}", fontsize=11, y=1.0,
            )
            fig_combined.tight_layout()

            combined_path = os.path.join(out_dir, f"{base_name}_telemetry_all.png")
            fig_combined.savefig(
                combined_path, format="png", dpi=150, bbox_inches="tight"
            )
            fig_combined.clear()
            import matplotlib.pyplot as plt
            plt.close(fig_combined)
            saved_count += 1

            self._log_message(
                f"Exported {saved_count} telemetry plots to: {out_dir}"
            )

        except Exception as e:
            self._log_message(f"Error exporting telemetry plots: {e}")
            import traceback
            self._log_message(traceback.format_exc())

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
