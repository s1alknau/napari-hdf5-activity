"""_widget_frame_viewer.py — FrameViewerMixin for HDF5AnalysisWidget.

Provides the Frame Viewer tab: HDF5/AVI/Zarr frame loading, playback,
ROI overlay, video/GIF export, and synchronized activity plots.
Mixed into HDF5AnalysisWidget so all methods share the same ``self`` namespace.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
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
    from ._io_abstraction import open_file_reader, detect_file_format
    IO_ABSTRACTION_AVAILABLE = True
except ImportError:
    IO_ABSTRACTION_AVAILABLE = False


class FrameViewerMixin:
    """Mixin providing all Frame Viewer tab functionality.

    Requires that the host class provides:
    - self.file_path (str)
    - self.masks (list)
    - self._log_message(msg: str)
    - self.viewer (napari.Viewer)
    """

    def setup_viewer_tab(self):
        """Setup the Frame Viewer tab for browsing through dataset frames."""
        from qtpy.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QSlider, QPushButton,
            QSpinBox, QComboBox, QSizePolicy, QCheckBox, QTextEdit,
        )
        from qtpy.QtCore import Qt, QTimer

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
        self.viewer_frame_slider = QSlider(Qt.Horizontal)
        self.viewer_frame_slider.setMinimum(0)
        self.viewer_frame_slider.setMaximum(0)
        self.viewer_frame_slider.setValue(0)
        self.viewer_frame_slider.setEnabled(False)

        # Debounce timer: load frame 60 ms after the last keyboard/programmatic change.
        # During mouse drag the timer is suppressed — the frame loads on sliderReleased.
        self._viewer_slider_debounce = QTimer()
        self._viewer_slider_debounce.setSingleShot(True)
        self._viewer_slider_debounce.setInterval(60)
        self._viewer_slider_debounce.timeout.connect(self._on_viewer_frame_debounced)
        self._viewer_slider_dragging = False

        self.viewer_frame_slider.sliderPressed.connect(self._on_viewer_slider_pressed)
        self.viewer_frame_slider.sliderReleased.connect(self._on_viewer_slider_released)
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

        # File selector for multiple loaded files
        file_select_layout = QHBoxLayout()
        file_select_layout.addWidget(QLabel("File:"))
        self.viewer_file_combo = QComboBox()
        self.viewer_file_combo.setToolTip("Select which file to view in the Frame Viewer")
        self.viewer_file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_select_layout.addWidget(self.viewer_file_combo)
        layout.addLayout(file_select_layout)

        # Load data button
        self.btn_viewer_load = QPushButton("Load Selected File into Viewer")
        self.btn_viewer_load.clicked.connect(self._viewer_load_data)
        self.btn_viewer_load.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 10px; } QPushButton:hover { background-color: #1976D2; }"
        )
        layout.addWidget(self.btn_viewer_load)

        # ROI overlay option
        overlay_row = QHBoxLayout()
        self.viewer_show_roi_overlay = QCheckBox("Show ROI overlay (circles + numbers)")
        self.viewer_show_roi_overlay.setChecked(True)
        self.viewer_show_roi_overlay.setToolTip(
            "Draw detected ROI circles and numbers on each frame "
            "so you can identify which well corresponds to which ROI"
        )
        self.viewer_show_roi_overlay.toggled.connect(self._on_viewer_roi_overlay_changed)
        overlay_row.addWidget(self.viewer_show_roi_overlay)
        overlay_row.addStretch()
        layout.addLayout(overlay_row)

        # Export video/GIF section
        export_group = QGroupBox("Export Video/GIF")
        export_layout = QVBoxLayout()
        export_group.setLayout(export_layout)

        # Time range selection
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Time Range:"))

        self.export_start_frame = QSpinBox()
        self.export_start_frame.setPrefix("Start Frame: ")
        self.export_start_frame.setMinimum(0)
        self.export_start_frame.setMaximum(0)
        self.export_start_frame.setValue(0)
        self.export_start_frame.setEnabled(False)
        range_layout.addWidget(self.export_start_frame)

        self.export_end_frame = QSpinBox()
        self.export_end_frame.setPrefix("End Frame: ")
        self.export_end_frame.setMinimum(0)
        self.export_end_frame.setMaximum(0)
        self.export_end_frame.setValue(0)
        self.export_end_frame.setEnabled(False)
        range_layout.addWidget(self.export_end_frame)

        range_layout.addStretch()
        export_layout.addLayout(range_layout)

        # Export buttons
        export_buttons_layout = QHBoxLayout()

        self.btn_export_video = QPushButton("Export as Video (MP4)")
        self.btn_export_video.setToolTip("Export selected frame range as MP4 video")
        self.btn_export_video.clicked.connect(self._export_video)
        self.btn_export_video.setEnabled(False)
        export_buttons_layout.addWidget(self.btn_export_video)

        self.btn_export_gif = QPushButton("Export as GIF")
        self.btn_export_gif.setToolTip("Export selected frame range as animated GIF")
        self.btn_export_gif.clicked.connect(self._export_gif)
        self.btn_export_gif.setEnabled(False)
        export_buttons_layout.addWidget(self.btn_export_gif)

        export_layout.addLayout(export_buttons_layout)

        # Export FPS control
        export_fps_layout = QHBoxLayout()
        export_fps_layout.addWidget(QLabel("Export FPS:"))

        self.export_fps_spin = QSpinBox()
        self.export_fps_spin.setRange(1, 60)
        self.export_fps_spin.setValue(10)
        self.export_fps_spin.setToolTip("Frames per second for exported video/GIF")
        export_fps_layout.addWidget(self.export_fps_spin)
        export_fps_layout.addStretch()

        export_layout.addLayout(export_fps_layout)

        layout.addWidget(export_group)

        # Info display
        self.viewer_info_text = QTextEdit()
        self.viewer_info_text.setReadOnly(True)
        self.viewer_info_text.setMaximumHeight(100)
        self.viewer_info_text.setPlaceholderText(
            "Frame information will appear here..."
        )
        layout.addWidget(self.viewer_info_text)

        # === SYNCHRONIZED ANALYSIS PLOTS ===
        sync_plot_group = QGroupBox("Synchronized Analysis Plots")
        sync_plot_layout = QVBoxLayout()
        sync_plot_group.setLayout(sync_plot_layout)

        # Enable checkbox
        self.sync_plots_enabled = QCheckBox("Show synchronized plots with time marker")
        self.sync_plots_enabled.setChecked(False)
        self.sync_plots_enabled.setToolTip(
            "Display analysis results below with a time marker "
            "synchronized to the current frame position"
        )
        self.sync_plots_enabled.toggled.connect(self._toggle_sync_plots)
        sync_plot_layout.addWidget(self.sync_plots_enabled)

        # Plot type selector
        sync_type_layout = QHBoxLayout()
        sync_type_layout.addWidget(QLabel("Plot Type:"))
        self.sync_plot_type = QComboBox()
        self.sync_plot_type.addItems([
            "Raw Intensity (0-1)",
            "Movement (binary)",
            "Fraction Movement",
            "Quiescence",
            "Sleep",
            "Sleep Quality",
        ])
        self.sync_plot_type.currentIndexChanged.connect(self._update_sync_plot)
        self.sync_plot_type.setCurrentIndex(2)  # Default to Fraction Movement
        sync_type_layout.addWidget(self.sync_plot_type)

        # Binning control for synchronized plots
        sync_type_layout.addWidget(QLabel("Bin:"))
        self.sync_plot_bin = QSpinBox()
        self.sync_plot_bin.setRange(0, 240)
        self.sync_plot_bin.setValue(0)
        self.sync_plot_bin.setSuffix(" min")
        self.sync_plot_bin.setSpecialValueText("Original")
        self.sync_plot_bin.setToolTip("Re-bin data for visualization (0 = original binning)")
        self.sync_plot_bin.valueChanged.connect(self._update_sync_plot)
        sync_type_layout.addWidget(self.sync_plot_bin)

        sync_type_layout.addStretch()
        sync_plot_layout.addLayout(sync_type_layout)

        # Matplotlib canvas for synchronized plots
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.sync_figure = Figure(figsize=(12, 6), dpi=100)
        self.sync_canvas = FigureCanvasQTAgg(self.sync_figure)
        self.sync_canvas.setMinimumHeight(350)
        self.sync_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sync_canvas.setVisible(False)  # Hidden by default
        sync_plot_layout.addWidget(self.sync_canvas, stretch=1)

        # Store time marker line references
        self.sync_time_markers = []

        # Export buttons for video/GIF with synchronized plots
        export_sync_layout = QHBoxLayout()
        self.btn_export_sync_video = QPushButton("Export Video with Plots (MP4)")
        self.btn_export_sync_video.setToolTip(
            "Export video showing frame + synchronized analysis plots side by side"
        )
        self.btn_export_sync_video.clicked.connect(self._export_video_with_plots)
        self.btn_export_sync_video.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        export_sync_layout.addWidget(self.btn_export_sync_video)

        self.btn_export_sync_gif = QPushButton("Export GIF with Plots")
        self.btn_export_sync_gif.setToolTip(
            "Export animated GIF showing frame + synchronized analysis plots"
        )
        self.btn_export_sync_gif.clicked.connect(self._export_gif_with_plots)
        self.btn_export_sync_gif.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        export_sync_layout.addWidget(self.btn_export_sync_gif)

        export_sync_layout.addStretch()
        sync_plot_layout.addLayout(export_sync_layout)

        layout.addWidget(sync_plot_group)

        layout.addStretch()

        # Timer for playback
        from qtpy.QtCore import QTimer

        self.viewer_timer = QTimer()
        self.viewer_timer.timeout.connect(self._viewer_play_next_frame)
        self.viewer_is_playing = False

    def _update_viewer_file_combo(self):
        """Update the file selector combo box with all available files."""
        if not hasattr(self, "viewer_file_combo"):
            return

        self.viewer_file_combo.blockSignals(True)
        self.viewer_file_combo.clear()

        files_added = set()

        # Add current HDF5/Zarr file
        if hasattr(self, "file_path") and self.file_path and self.file_path not in files_added:
            self.viewer_file_combo.addItem(
                os.path.basename(self.file_path), self.file_path
            )
            files_added.add(self.file_path)

        # Add AVI batch as ONE entry (not individual files)
        if hasattr(self, "avi_batch_paths") and self.avi_batch_paths:
            n = len(self.avi_batch_paths)
            interval = getattr(self, "avi_batch_interval", 5.0)
            label = f"AVI Batch ({n} videos, interval={interval:.0f}s)"
            self.viewer_file_combo.addItem(label, "__avi_batch__")

        # Add HDF5 and Zarr files from directory
        if hasattr(self, "directory") and self.directory:
            try:
                for f in sorted(os.listdir(self.directory)):
                    full_path = os.path.join(self.directory, f)
                    is_zarr_dir = f.lower().endswith(".zarr") and os.path.isdir(full_path)
                    if (f.lower().endswith((".h5", ".hdf5")) or is_zarr_dir) and full_path not in files_added:
                        self.viewer_file_combo.addItem(f, full_path)
                        files_added.add(full_path)
            except Exception:
                pass

        self.viewer_file_combo.blockSignals(False)

        if self.viewer_file_combo.count() > 0:
            self._log_message(
                f"Frame Viewer: {self.viewer_file_combo.count()} source(s) available"
            )

    def _viewer_load_data(self):
        """Load the selected file into the frame viewer."""
        # Use file from combo box if available
        selected_path = None
        if hasattr(self, "viewer_file_combo") and self.viewer_file_combo.count() > 0:
            selected_path = self.viewer_file_combo.currentData()

        # AVI batch sentinel — load all batch videos as one concatenated stream
        if selected_path == "__avi_batch__":
            try:
                self._viewer_load_avi_batch()
            except Exception as e:
                self.viewer_status_label.setText(f"❌ Error loading AVI batch: {e}")
                self._log_message(f"❌ Frame viewer AVI batch error: {e}")
                import traceback
                traceback.print_exc()
            return

        # Fallback to self.file_path
        if not selected_path:
            selected_path = getattr(self, "file_path", None)

        if not selected_path:
            self.viewer_status_label.setText(
                "⚠️ No file loaded. Please load a file in the Input tab first."
            )
            self._log_message("⚠️ Frame viewer: No file loaded")
            return

        # Set file_path to the selected file for viewer loading
        self.file_path = selected_path

        try:
            # Normalize path (strip trailing separators/spaces from directory pickers)
            selected_path = selected_path.rstrip("/\\").strip()
            self.file_path = selected_path

            path_lower = selected_path.lower()
            is_avi = path_lower.endswith((".avi", ".mp4"))
            # Any directory is treated as a Zarr store (HDF5/AVI are never directories)
            is_directory = os.path.isdir(selected_path)
            is_zarr = path_lower.endswith(".zarr") or is_directory
            is_hdf5 = not is_zarr and path_lower.endswith((".h5", ".hdf5"))

            # Content-based fallback for files with unexpected extensions
            if not is_avi and not is_hdf5 and not is_zarr and IO_ABSTRACTION_AVAILABLE:
                try:
                    fmt = detect_file_format(selected_path)
                    is_hdf5 = fmt == "hdf5"
                    is_zarr = fmt == "zarr"
                except Exception:
                    pass

            self._log_message(
                f"Frame viewer: loading {repr(selected_path)} "
                f"[dir={is_directory}, zarr={is_zarr}, hdf5={is_hdf5}, avi={is_avi}]"
            )

            if is_avi:
                self._viewer_load_avi_batch()
            elif is_hdf5 or is_zarr:
                self._viewer_load_hdf5()
            else:
                self.viewer_status_label.setText("⚠️ Unsupported file format")
                self._log_message(
                    f"⚠️ Frame viewer: Unsupported file format — path: {repr(selected_path)}"
                )

        except Exception as e:
            self.viewer_status_label.setText(f"❌ Error loading data: {e}")
            self._log_message(f"❌ Frame viewer error: {e}")
            import traceback

            traceback.print_exc()

    def _viewer_load_hdf5(self):
        """Load HDF5 or Zarr file frames into the viewer.

        Uses detect_file_structure_type to handle all supported layouts
        (stacked /frames, /images/frames, individual /images/frame_XXXXXX)
        for both HDF5 and Zarr stores.  A persistent FileReader handle is
        opened and stored in self.viewer_file_handle so frames can be
        accessed during playback without re-opening the file.
        """
        import json

        # Leaving AVI batch mode
        self._viewer_release_avi_cap()
        self._viewer_is_avi_batch = False

        self._log_message(f"Loading file into frame viewer: {self.file_path}")

        # ---- open the recording -----------------------------------------
        # FrameSource resolves the layout (stacked / individual, HDF5 / Zarr)
        # and keeps one open handle for the whole viewing session.
        from ._frame_source import FrameSource, FrameSourceError

        self._viewer_close_source()
        try:
            source = FrameSource(self.file_path).open()
        except FrameSourceError as exc:
            self._log_message(f"Frame viewer: {exc}")
            raise ValueError(str(exc)) from exc

        self.viewer_file_handle = source
        self.viewer_frames = None
        self.viewer_n_frames = source.n_frames
        self.viewer_dataset_name = source.dataset_name
        self.viewer_is_sequence = source.layout == "individual"
        self._log_message(f"Frame viewer: {source.describe()}")

        # ---- frame interval from the root metadata attribute --------------
        self.viewer_frame_interval = 5.0
        try:
            root_attrs = source.read_attrs("/")
            if "metadata" in root_attrs:
                meta = json.loads(root_attrs["metadata"])
                self.viewer_frame_interval = meta.get("frame_interval", 5.0)
            elif "frame_interval" in root_attrs:
                self.viewer_frame_interval = float(root_attrs["frame_interval"])
        except Exception:
            pass

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

        # Enable export controls
        self.export_start_frame.setEnabled(True)
        self.export_start_frame.setMaximum(self.viewer_n_frames - 1)
        self.export_start_frame.setValue(0)

        self.export_end_frame.setEnabled(True)
        self.export_end_frame.setMaximum(self.viewer_n_frames - 1)
        self.export_end_frame.setValue(self.viewer_n_frames - 1)  # Default to all frames

        self.btn_export_video.setEnabled(True)
        self.btn_export_gif.setEnabled(True)

        self.viewer_status_label.setText(
            f"✓ Loaded HDF5: {self.viewer_n_frames} frames (Interval: {self.viewer_frame_interval}s)"
        )
        self._log_message(
            f"✓ Frame viewer: Loaded {self.viewer_n_frames} frames from HDF5 (interval: {self.viewer_frame_interval}s)"
        )

        # Pre-load frames into cache for smooth playback
        self._viewer_preload_frames()

        # Display first frame
        self._viewer_show_frame(0)

    def _viewer_load_avi_batch(self):
        """Set up AVI batch for streaming display — no frames loaded into RAM.

        Scans metadata from each video (fast, no frame reading), then stores
        a lookup table so any global frame index can be resolved to (video,
        local_frame) on demand.  Frames are fetched one at a time when the
        slider moves, keeping memory usage at ≈ one frame.
        """
        import cv2

        avi_paths = getattr(self, "avi_batch_paths", [])
        if not avi_paths:
            # Single-file AVI selected directly — treat as a batch of one
            fp = getattr(self, "file_path", None)
            if fp and fp.lower().endswith((".avi", ".mp4")):
                avi_paths = [fp]
            else:
                raise ValueError("No AVI batch paths available.")

        target_interval = getattr(self, "avi_batch_interval", 5.0)
        self.viewer_frame_interval = target_interval

        self._log_message(
            f"Setting up AVI batch viewer: {len(avi_paths)} files, "
            f"sampling interval={target_interval}s (streaming, no RAM preload)"
        )

        # Release any previously open VideoCapture handle
        if getattr(self, "_viewer_avi_cap", None) is not None:
            try:
                self._viewer_avi_cap.release()
            except Exception:
                pass
            self._viewer_avi_cap = None
            self._viewer_avi_cap_path = None

        # Scan metadata — opens each cap briefly, reads properties, closes
        video_infos: List[Dict] = []
        cumulative = 0
        for path in avi_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self._log_message(f"  ⚠️  Cannot open: {os.path.basename(path)}")
                cap.release()
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps <= 0 or n_frames == 0:
                self._log_message(f"  ⚠️  Bad metadata: {os.path.basename(path)}")
                continue
            frames_per_sample = max(1, int(fps * target_interval))
            sampled = len(range(0, n_frames, frames_per_sample))
            video_infos.append({
                "path": path,
                "fps": fps,
                "n_frames": n_frames,
                "frames_per_sample": frames_per_sample,
                "sampled": sampled,
                "start_idx": cumulative,  # global index of first sampled frame
            })
            cumulative += sampled

        if not video_infos:
            raise ValueError("No valid AVI files found in batch.")

        self._log_message(
            f"  → {len(video_infos)} valid videos, {cumulative} sampled frames total"
        )
        for info in video_infos:
            self._log_message(
                f"    {os.path.basename(info['path'])}: "
                f"{info['sampled']} frames (every {info['frames_per_sample']}th)"
            )

        # Store batch info for on-demand frame reads
        self._viewer_avi_batch_info = video_infos
        self._viewer_is_avi_batch = True
        self._viewer_avi_cap = None       # opened lazily in _viewer_read_raw_frame_avi
        self._viewer_avi_cap_path = None  # path currently open in _viewer_avi_cap

        # Viewer state
        self.viewer_n_frames = cumulative
        self.viewer_frames = None
        self._viewer_close_source()   # release any HDF5/Zarr file first
        self.viewer_frame_cache = None    # no preloading for streaming
        self.viewer_is_sequence = False

        # Per-video boundaries for the status line
        self._viewer_avi_video_boundaries = [
            (info["start_idx"],
             info["start_idx"] + info["sampled"] - 1,
             os.path.basename(info["path"]))
            for info in video_infos
        ]

        # Update UI controls
        self.viewer_current_frame = 0
        self.viewer_frame_slider.setMaximum(self.viewer_n_frames - 1)
        self.viewer_frame_slider.setValue(0)
        self.viewer_frame_slider.setEnabled(True)

        for btn in (self.btn_viewer_first, self.btn_viewer_prev,
                    self.btn_viewer_play, self.btn_viewer_next,
                    self.btn_viewer_last):
            btn.setEnabled(True)

        self.export_start_frame.setEnabled(True)
        self.export_start_frame.setMaximum(self.viewer_n_frames - 1)
        self.export_start_frame.setValue(0)
        self.export_end_frame.setEnabled(True)
        self.export_end_frame.setMaximum(self.viewer_n_frames - 1)
        self.export_end_frame.setValue(self.viewer_n_frames - 1)
        self.btn_export_video.setEnabled(True)
        self.btn_export_gif.setEnabled(True)

        total_duration = cumulative * target_interval
        self.viewer_status_label.setText(
            f"AVI Batch (streaming): {cumulative} frames from {len(video_infos)} videos "
            f"({total_duration/60:.1f} min, interval={target_interval}s)"
        )

        # Display first frame
        self._viewer_show_frame(0)

    def _viewer_read_raw_frame_avi(self, frame_idx: int) -> "np.ndarray":
        """Read one grayscale frame from the AVI batch by global frame index.

        Finds which video contains the requested global index, keeps a cached
        VideoCapture for consecutive reads within the same file, and seeks
        directly to the correct video frame via cap.set().
        """
        import cv2

        batch_info = getattr(self, "_viewer_avi_batch_info", [])
        if not batch_info:
            return np.zeros((100, 100), dtype=np.uint8)

        # Locate which video owns this global frame index
        info = None
        for v in batch_info:
            if frame_idx < v["start_idx"] + v["sampled"]:
                info = v
                break

        if info is None:
            return np.zeros((100, 100), dtype=np.uint8)

        local_sampled_idx = frame_idx - info["start_idx"]
        actual_frame_idx  = local_sampled_idx * info["frames_per_sample"]

        # Reuse cap if same video, otherwise open the new one
        if getattr(self, "_viewer_avi_cap_path", None) != info["path"]:
            if getattr(self, "_viewer_avi_cap", None) is not None:
                try:
                    self._viewer_avi_cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(info["path"])
            self._viewer_avi_cap = cap
            self._viewer_avi_cap_path = info["path"]
        else:
            cap = self._viewer_avi_cap

        if cap is None or not cap.isOpened():
            return np.zeros((100, 100), dtype=np.uint8)

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return np.zeros((100, 100), dtype=np.uint8)

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def _viewer_read_raw_frame(self, frame_idx: int) -> "np.ndarray":
        """Read one raw frame from whatever source is currently active.

        Handles AVI batch streaming, HDF5 stacked/individual, and Zarr.
        """
        # AVI batch — on-demand streaming from disk
        if getattr(self, "_viewer_is_avi_batch", False):
            return self._viewer_read_raw_frame_avi(frame_idx)

        # FrameSource knows the layout and the format; nothing to branch on.
        return self.viewer_file_handle.read_frame(frame_idx)

    def _viewer_close_source(self):
        """Close the open recording, if any.

        The previous code only set the attribute to None, so on a 130 GB file
        the handle survived until the garbage collector happened to run.
        """
        source = getattr(self, "viewer_file_handle", None)
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        self.viewer_file_handle = None

    def _viewer_release_avi_cap(self):
        """Release a cached AVI VideoCapture handle if one is open."""
        if getattr(self, "_viewer_avi_cap", None) is not None:
            try:
                self._viewer_avi_cap.release()
            except Exception:
                pass
            self._viewer_avi_cap = None
            self._viewer_avi_cap_path = None

    def _viewer_preload_frames(self):
        """Pre-load all frames into memory cache for smooth playback."""
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt
        import psutil

        # AVI batch uses streaming — frames are read on demand, no preload needed
        if getattr(self, "_viewer_is_avi_batch", False):
            self.viewer_frame_cache = None
            return

        # Get available system memory
        available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

        # Estimate memory usage from the first frame
        sample_frame = self._viewer_read_raw_frame(0)

        # Calculate memory (3 channels for BGR, uint8)
        frame_size_mb = (sample_frame.shape[0] * sample_frame.shape[1] * 3) / (
            1024 * 1024
        )
        total_size_mb = frame_size_mb * self.viewer_n_frames

        # Safety limits:
        # 1. Don't use more than 50% of available RAM
        # 2. Don't use more than 4GB total
        # 3. Don't cache more than 10000 frames
        max_ram_mb = min(available_ram_mb * 0.5, 4096)  # 50% of available or 4GB max
        max_frames = min(10000, int(max_ram_mb / frame_size_mb))

        self._log_message(
            f"System RAM: {available_ram_mb:.0f} MB available, "
            f"Frame size: {frame_size_mb:.2f} MB, "
            f"Total needed: {total_size_mb:.1f} MB"
        )

        # Check if caching is feasible
        if total_size_mb > max_ram_mb:
            self._log_message(
                f"⚠ Skipping frame cache: {total_size_mb:.0f} MB needed exceeds "
                f"safe limit ({max_ram_mb:.0f} MB). Playback may be slower."
            )
            self.viewer_frame_cache = None
            return

        if self.viewer_n_frames > max_frames:
            self._log_message(
                f"⚠ Skipping frame cache: {self.viewer_n_frames} frames exceeds "
                f"limit ({max_frames}). Playback may be slower."
            )
            self.viewer_frame_cache = None
            return

        self._log_message(
            f"Pre-loading {self.viewer_n_frames} frames into cache "
            f"(~{total_size_mb:.1f} MB)..."
        )

        # Show progress dialog
        progress = QProgressDialog(
            f"Loading {self.viewer_n_frames} frames into memory...",
            "Cancel",
            0,
            self.viewer_n_frames,
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)  # Show after 500ms

        # Pre-load and pre-process all frames
        self.viewer_frame_cache = []

        try:
            for i in range(self.viewer_n_frames):
                # Update progress
                if i % 10 == 0:  # Update every 10 frames
                    progress.setValue(i)
                    if progress.wasCanceled():
                        self._log_message("Frame cache loading canceled by user")
                        self.viewer_frame_cache = None
                        return

                # Load frame via unified helper (handles AVI batch, HDF5, Zarr)
                frame_data = self._viewer_read_raw_frame(i)

                # Pre-process frame to display format
                frame_processed = self._prepare_frame_for_display(frame_data)
                self.viewer_frame_cache.append(frame_processed)

            progress.setValue(self.viewer_n_frames)
            self._log_message(
                f"✓ Frame cache ready: {self.viewer_n_frames} frames "
                f"({total_size_mb:.1f} MB in RAM)"
            )

        except Exception as e:
            self._log_message(f"❌ Frame cache loading failed: {e}")
            self.viewer_frame_cache = None
            import traceback

            traceback.print_exc()

    # Maximum dimension (width or height) for display. Larger frames are
    # downsampled proportionally before uploading to the GPU as a texture.
    # Prevents GL_OUT_OF_MEMORY on high-resolution cameras (HIK 4K+).
    VIEWER_MAX_DISPLAY_DIM = 1920

    def _prepare_frame_for_display(self, frame_data):
        """Pre-process a frame to display format (BGR uint8, max 1920px)."""
        import numpy as np
        import cv2

        # Make a writable copy
        frame = np.array(frame_data, copy=True)

        # Ensure it's uint8
        if frame.dtype != np.uint8:
            # Normalize to 0-255 range
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_max > frame_min:
                frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(
                    np.uint8
                )
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)

        # Ensure 2D shape
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        # Convert to 3-channel BGR for colored text overlay
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Downsample if too large for GPU texture (prevents GL_OUT_OF_MEMORY).
        # Only affects display — analysis uses the full-resolution data.
        h, w = frame.shape[:2]
        max_dim = self.VIEWER_MAX_DISPLAY_DIM
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def _draw_roi_overlay(self, frame):
        """Draw ROI circles and numbers on a frame (in-place).

        Uses the stored circle positions from ROI detection. Falls back to
        computing circle approximations from masks if circles aren't available.

        Circle coordinates are in the detection frame's coordinate space
        (_original_frame_shape).  The display frame may have been downsampled
        by _prepare_frame_for_display, so we scale all coordinates to match.
        """
        import cv2
        import numpy as np

        disp_h, disp_w = frame.shape[:2]

        # Get ROI exclusion info
        excluded = set()
        if hasattr(self, "roi_checkboxes") and self.roi_checkboxes:
            excluded = {
                i for i, cb in enumerate(self.roi_checkboxes) if not cb.isChecked()
            }

        # Try to use stored circle positions (most accurate)
        circles = getattr(self, "_original_circles", None)
        radius_scale = getattr(self, "roi_scale", None)
        radius_scale_val = radius_scale.value() if radius_scale is not None else 1.0

        if circles is not None:
            # Scale from detection-frame space to display-frame space
            orig_shape = getattr(self, "_original_frame_shape", None)
            if orig_shape is not None and (orig_shape[0] != disp_h or orig_shape[1] != disp_w):
                sx = disp_w / orig_shape[1]
                sy = disp_h / orig_shape[0]
            else:
                sx = sy = 1.0

            for idx, circle in enumerate(circles):
                cx = int(circle[0] * sx)
                cy = int(circle[1] * sy)
                r  = max(1, int(circle[2] * ((sx + sy) / 2) * radius_scale_val))
                is_excluded = idx in excluded

                color = (0, 0, 255) if is_excluded else (0, 255, 0)
                cv2.circle(frame, (cx, cy), r, color, 2)

                label = f"{idx + 1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.8
                thickness = 3
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                offset = int(r / 1.414) + 4
                text_x = min(cx + offset, disp_w - tw - 8)
                text_y = max(cy - offset, th + 8)
                cv2.rectangle(
                    frame,
                    (text_x - 4, text_y - th - 4),
                    (text_x + tw + 4, text_y + 4),
                    (0, 0, 0),
                    -1,
                )
                text_color = (100, 100, 255) if is_excluded else (255, 100, 100)
                cv2.putText(
                    frame, label, (text_x, text_y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA,
                )
            return

        # Fallback: derive circles from masks (masks are in detection space too)
        masks = getattr(self, "masks", [])
        if not masks:
            return

        orig_shape = getattr(self, "_original_frame_shape", None)
        if orig_shape is not None and (orig_shape[0] != disp_h or orig_shape[1] != disp_w):
            sx = disp_w / orig_shape[1]
            sy = disp_h / orig_shape[0]
        else:
            sx = sy = 1.0

        for idx, mask in enumerate(masks):
            if mask is None or mask.size == 0:
                continue
            is_excluded = idx in excluded
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            cx = int(np.mean(xs) * sx)
            cy = int(np.mean(ys) * sy)
            r  = max(1, int(np.sqrt(len(xs) / np.pi) * ((sx + sy) / 2)))

            color = (0, 0, 255) if is_excluded else (0, 255, 0)
            cv2.circle(frame, (cx, cy), r, color, 2)

            label = f"{idx + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.8
            thickness = 3
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            offset = int(r / 1.414) + 4
            text_x = min(cx + offset, disp_w - tw - 8)
            text_y = max(cy - offset, th + 8)
            cv2.rectangle(
                frame,
                (text_x - 4, text_y - th - 4),
                (text_x + tw + 4, text_y + 4),
                (0, 0, 0),
                -1,
            )
            text_color = (100, 100, 255) if is_excluded else (255, 100, 100)
            cv2.putText(
                frame, label, (text_x, text_y),
                font, font_scale, text_color, thickness, cv2.LINE_AA,
            )

    def _on_viewer_roi_overlay_changed(self, enabled):
        """Re-render the current frame when the ROI overlay toggle changes."""
        if hasattr(self, "viewer_current_frame"):
            self._viewer_show_frame(self.viewer_current_frame)

    def _viewer_show_frame(self, frame_idx):
        """Display a specific frame in the napari viewer."""
        try:
            if frame_idx < 0 or frame_idx >= self.viewer_n_frames:
                return

            # Calculate time from frame index
            frame_time_seconds = frame_idx * self.viewer_frame_interval
            frame_time_minutes = frame_time_seconds / 60.0
            frame_time_hours = frame_time_minutes / 60.0

            import numpy as np
            import cv2

            # Use cached frame if available (FAST!)
            if hasattr(self, "viewer_frame_cache") and self.viewer_frame_cache:
                # Get pre-processed frame from cache (already BGR uint8)
                frame_with_text = self.viewer_frame_cache[frame_idx].copy()
                frame_data = frame_with_text  # For info display
            else:
                # Load on-the-fly: AVI batch streaming, HDF5/Zarr, or in-memory array
                if getattr(self, "_viewer_is_avi_batch", False):
                    frame_data = self._viewer_read_raw_frame_avi(frame_idx)
                elif hasattr(self, "viewer_file_handle") and self.viewer_file_handle:
                    frame_data = self._viewer_read_raw_frame(frame_idx)
                else:
                    frame_data = self.viewer_frames[frame_idx]

                frame_with_text = self._prepare_frame_for_display(frame_data)

            # Prepare time text
            time_text = f"t = {frame_time_seconds:.1f}s ({frame_time_minutes:.2f}min)"

            # Position: lower left (10 pixels from left, 30 pixels from bottom)
            height, width = frame_with_text.shape[:2]
            text_position = (10, height - 10)

            # Draw text in white (BGR: 255, 255, 255), 50% larger
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2  # Increased from 0.8 (50% larger)
            color = (255, 255, 255)  # White in BGR
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

            # Draw ROI overlay if enabled
            if (
                hasattr(self, "viewer_show_roi_overlay")
                and self.viewer_show_roi_overlay.isChecked()
            ):
                self._draw_roi_overlay(frame_with_text)

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

    def _on_viewer_slider_pressed(self):
        """Mouse button down on slider — suppress frame loads until release."""
        self._viewer_slider_dragging = True
        self._viewer_slider_debounce.stop()

    def _on_viewer_slider_released(self):
        """Mouse button released — load the frame at the final drag position."""
        self._viewer_slider_dragging = False
        value = self.viewer_frame_slider.value()
        self._viewer_debounced_frame_idx = value
        self._viewer_show_frame(value)
        if hasattr(self, "sync_plots_enabled") and self.sync_plots_enabled.isChecked():
            self._update_sync_time_marker(value)

    def _on_viewer_frame_changed(self, value):
        """Handle slider value change — update label instantly.

        While dragging: only the label is refreshed (no frame load, no debounce).
        For keyboard / programmatic changes: start the debounce timer.
        """
        self._viewer_debounced_frame_idx = value
        n = getattr(self, "viewer_n_frames", 1)
        interval = getattr(self, "viewer_frame_interval", 5.0)
        t_sec = value * interval
        self.viewer_frame_label.setText(
            f"Frame: {value + 1} / {n} | Time: {t_sec:.1f}s ({t_sec/60:.2f}min)"
        )
        if not self._viewer_slider_dragging:
            self._viewer_slider_debounce.start()

    def _on_viewer_frame_debounced(self):
        """Load the frame after keyboard/programmatic slider idle for 60 ms."""
        value = getattr(self, "_viewer_debounced_frame_idx", 0)
        self._viewer_show_frame(value)
        if hasattr(self, "sync_plots_enabled") and self.sync_plots_enabled.isChecked():
            self._update_sync_time_marker(value)

    def _toggle_sync_plots(self, enabled):
        """Toggle synchronized plots visibility."""
        self.sync_canvas.setVisible(enabled)
        if enabled:
            self._update_sync_plot()

    def _get_sync_plot_data(self):
        """Return (plot_type, data_dict) for the current sync plot settings, with rebinning applied."""
        sync_type = self.sync_plot_type.currentText()
        sync_bin = self.sync_plot_bin.value() if hasattr(self, "sync_plot_bin") else 0
        original_bin = self.bin_size_seconds.value() if hasattr(self, "bin_size_seconds") else 60
        new_bin_seconds = sync_bin * 60

        def maybe_rebin(dd):
            if dd and new_bin_seconds > original_bin:
                return self._rebin_timeseries_data(dd, new_bin_seconds, original_bin)
            return dd

        if "Raw Intensity" in sync_type:
            return "Raw Intensity Changes", self.merged_results
        elif "Movement (binary)" in sync_type:
            return "Movement", maybe_rebin(getattr(self, "movement_data", {}))
        elif "Fraction Movement" in sync_type:
            return "Fraction Movement", maybe_rebin(getattr(self, "fraction_data", {}))
        elif "Quiescence" in sync_type:
            return "Quiescence", maybe_rebin(getattr(self, "quiescence_data", {}))
        elif "Sleep Quality" in sync_type:
            return "Sleep Quality", getattr(self, "sleep_quality_data", {})
        elif "Sleep" in sync_type:
            return "Sleep", maybe_rebin(getattr(self, "sleep_data", {}))
        else:
            return "Raw Intensity Changes", self.merged_results

    def _update_sync_plot(self):
        """Update the synchronized analysis plot (same style as Analysis tab)."""
        if not hasattr(self, "sync_plots_enabled") or not self.sync_plots_enabled.isChecked():
            return

        # Check if we have analysis data
        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("⚠️ No analysis data for synchronized plots — run Main Analysis first")
            self.sync_figure.clear()
            ax = self.sync_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No analysis data.\nRun Main Analysis first.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.axis("off")
            self.sync_canvas.draw()
            return

        from ._plot import PlotGenerator, create_plot_config, create_hysteresis_kwargs

        plot_type, data_dict = self._get_sync_plot_data()

        if not data_dict:
            msg = f"No {self.sync_plot_type.currentText()} data available"
            self._log_message(f"⚠️ {msg}")
            self.sync_figure.clear()
            ax = self.sync_figure.add_subplot(111)
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.axis("off")
            self.sync_canvas.draw()
            return

        # Get ROI colors
        roi_colors = getattr(self, "roi_colors", {})
        n_rois = len(data_dict)
        if n_rois == 0:
            return

        # Clear figure
        self.sync_figure.clear()

        # Dynamically adjust figure size based on canvas size and number of ROIs
        canvas_height = self.sync_canvas.height()
        canvas_width = self.sync_canvas.width()
        dpi = self.sync_figure.get_dpi()

        # Set figure size to match canvas
        fig_width = max(10, canvas_width / dpi)
        fig_height = max(4, canvas_height / dpi)
        self.sync_figure.set_size_inches(fig_width, fig_height)

        # Create plot configuration from widget settings (same as Analysis tab)
        plot_config = create_plot_config(self)
        # Fit all ROIs exactly into the available canvas height
        plot_config["height_per_roi"] = max(0.6, fig_height / n_rois) if n_rois > 0 else 0.8
        plot_config["fig_width"] = fig_width
        plot_config["fig_height"] = fig_height  # prevent generate_plot from overriding

        # Resolve ZT mode and lighting overlay (same logic as Analysis tab)
        zt_mode = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
        show_lighting = self.chk_show_lighting.isChecked() if hasattr(self, "chk_show_lighting") else False
        if show_lighting:
            _raw_led = getattr(self, "led_data", None)
            if _raw_led and isinstance(_raw_led, dict) and _raw_led.get("times") and _raw_led.get("white_powers"):
                led_data_for_plot = _raw_led
            else:
                led_data_for_plot = {}  # empty dict → legacy 12h fallback
        else:
            led_data_for_plot = None

        # Get kwargs for specific plot types
        if plot_type == "Raw Intensity Changes":
            hysteresis_kwargs = create_hysteresis_kwargs(widget_instance=self)
            hysteresis_kwargs.pop("merged_results", None)
        elif plot_type == "Sleep Quality":
            hysteresis_kwargs = {"sleep_metric": "sleep_minutes"}
        else:
            hysteresis_kwargs = {}

        # Always include ZT mode and lighting in kwargs
        hysteresis_kwargs["zt_mode"] = zt_mode
        hysteresis_kwargs["led_data"] = led_data_for_plot

        # Use PlotGenerator (same as Analysis tab)
        plot_generator = PlotGenerator(self.sync_figure)
        try:
            plot_generator.generate_plot(
                plot_type, data_dict, roi_colors, plot_config, **hysteresis_kwargs
            )
        except Exception as e:
            self._log_message(f"ERROR in sync plot: {e}")
            self.sync_figure.clear()
            ax = self.sync_figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="red", wrap=True)
            ax.axis("off")
            self.sync_canvas.draw()
            return

        # Store time markers for each subplot
        self.sync_time_markers = []
        for ax in self.sync_figure.get_axes():
            marker = ax.axvline(x=0, color="red", linewidth=2, linestyle="--", alpha=0.9, zorder=100)
            self.sync_time_markers.append(marker)

        # Adjust layout to maximize plot area (suppress warning for incompatible axes)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use subplots_adjust for better space usage
                self.sync_figure.subplots_adjust(
                    left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.15
                )
        except Exception:
            pass

        # Draw canvas
        self.sync_canvas.draw()

        # Update marker to current position
        if hasattr(self, "viewer_current_frame"):
            self._update_sync_time_marker(self.viewer_current_frame)

    def _update_sync_time_marker(self, frame_idx):
        """Update the time marker position in synchronized plots."""
        if not hasattr(self, "sync_time_markers") or not self.sync_time_markers:
            return

        if not hasattr(self, "viewer_frame_interval"):
            return

        # Calculate time in the same unit as the x-axis (hours if ZT mode, else minutes)
        zt_mode = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
        time_seconds = frame_idx * self.viewer_frame_interval
        time_x = time_seconds / 3600.0 if zt_mode else time_seconds / 60.0

        # Update all markers
        for marker in self.sync_time_markers:
            marker.set_xdata([time_x, time_x])

        # Redraw canvas (use blit for performance if available)
        try:
            self.sync_canvas.draw_idle()
        except Exception:
            pass

    def _export_video_with_plots(self):
        """Export video with synchronized analysis plots (same style as Analysis tab)."""
        import cv2
        import numpy as np
        from qtpy.QtWidgets import QFileDialog, QProgressDialog
        from qtpy.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from ._plot import PlotGenerator, create_plot_config, create_hysteresis_kwargs

        # Check prerequisites
        if not hasattr(self, "viewer_n_frames") or self.viewer_n_frames == 0:
            self._log_message("❌ No frames loaded. Load data into Frame Viewer first.")
            return

        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("❌ No analysis data. Run Main Analysis first.")
            return

        # Get frame range from export controls
        start_frame = self.export_start_frame.value()
        end_frame = self.export_end_frame.value()

        if start_frame >= end_frame:
            self._log_message("❌ Start frame must be less than end frame")
            return

        # Ask for save location
        default_name = f"video_with_plots_{start_frame}-{end_frame}.mp4"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video with Plots",
            default_name,
            "MP4 Video (*.mp4);;All Files (*)",
        )

        if not file_path:
            return

        export_fps = self.export_fps_spin.value()
        n_frames = end_frame - start_frame

        self._log_message(f"🎬 Exporting {n_frames} frames with plots at {export_fps} FPS...")

        # Setup progress dialog
        progress = QProgressDialog("Exporting video with plots...", "Cancel", 0, n_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Export Progress")
        progress.show()

        # Process events to show dialog
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()

        plot_type, data_dict = self._get_sync_plot_data()

        if not data_dict:
            self._log_message(f"❌ No {self.sync_plot_type.currentText()} data available")
            return

        roi_colors = getattr(self, "roi_colors", {})
        n_rois = len(data_dict)

        # Get first frame to determine dimensions
        first_frame = self._get_viewer_frame(start_frame)
        frame_height, frame_width = first_frame.shape[:2]

        # Create plot figure dimensions matching Analysis tab style
        plot_dpi = 100
        plot_width = frame_width
        height_per_roi = 60  # pixels per ROI
        plot_height = max(200, height_per_roi * n_rois)
        fig_width = plot_width / plot_dpi
        fig_height = plot_height / plot_dpi

        # Total output dimensions: frame on top, plots below
        output_width = frame_width
        output_height = frame_height + plot_height

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(file_path, fourcc, export_fps, (output_width, output_height))

        # Create plot configuration from widget settings (same as Analysis tab)
        base_plot_config = create_plot_config(self)
        base_plot_config["fig_width"] = fig_width
        base_plot_config["height_per_roi"] = fig_height / n_rois if n_rois > 0 else 0.6
        base_plot_config["dpi"] = plot_dpi

        # Get extra kwargs for specific plot types
        if plot_type == "Raw Intensity Changes":
            hysteresis_kwargs = create_hysteresis_kwargs(widget_instance=self)
            hysteresis_kwargs.pop("merged_results", None)
        elif plot_type == "Sleep Quality":
            sleep_metric = "sleep_minutes"
            if hasattr(self, "sleep_quality_metric_combo"):
                metric_text = self.sleep_quality_metric_combo.currentText()
                if "Transitions" in metric_text:
                    sleep_metric = "transitions"
                elif "Bout" in metric_text:
                    sleep_metric = "bout_length"
            hysteresis_kwargs = {"sleep_metric": sleep_metric}
        else:
            hysteresis_kwargs = {}

        # Resolve ZT mode and lighting overlay (same logic as Analysis tab)
        zt_mode_export = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
        show_lighting_export = self.chk_show_lighting.isChecked() if hasattr(self, "chk_show_lighting") else False
        if show_lighting_export:
            _raw_led = getattr(self, "led_data", None)
            if _raw_led and isinstance(_raw_led, dict) and _raw_led.get("times") and _raw_led.get("white_powers"):
                led_data_export = _raw_led
            else:
                led_data_export = {}  # empty dict → legacy 12h fallback
        else:
            led_data_export = None
        hysteresis_kwargs["zt_mode"] = zt_mode_export
        hysteresis_kwargs["led_data"] = led_data_export

        try:
            for i, frame_idx in enumerate(range(start_frame, end_frame)):
                if progress.wasCanceled():
                    self._log_message("⚠️ Export cancelled by user")
                    break

                progress.setValue(i)

                # Log progress every 10 frames
                if i % 10 == 0:
                    self._log_message(f"   Processing frame {i + 1}/{n_frames}...")
                    QApplication.processEvents()

                # Get frame
                frame = self._get_viewer_frame(frame_idx)

                # Add time text to frame
                time_seconds = frame_idx * self.viewer_frame_interval
                time_minutes = time_seconds / 60.0
                time_text = f"t = {time_seconds:.1f}s ({time_minutes:.2f}min)"

                frame_bgr = frame.copy()
                if len(frame_bgr.shape) == 2:
                    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
                elif len(frame_bgr.shape) == 3:
                    if frame_bgr.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGBA2BGR)
                    elif frame_bgr.shape[2] == 1:
                        frame_bgr = cv2.cvtColor(frame_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)
                    # else: assume already BGR with 3 channels

                cv2.putText(
                    frame_bgr, time_text, (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                )

                # Draw ROI overlay (circles + numbers) if enabled
                if hasattr(self, "viewer_show_roi_overlay") and self.viewer_show_roi_overlay.isChecked():
                    self._draw_roi_overlay(frame_bgr)

                # Create plot using PlotGenerator (same as Analysis tab)
                plot_fig = Figure(figsize=(fig_width, fig_height), dpi=plot_dpi)
                plot_generator = PlotGenerator(plot_fig)

                # Generate plot with same settings as Analysis tab
                plot_generator.generate_plot(
                    plot_type, data_dict, roi_colors, base_plot_config, **hysteresis_kwargs
                )

                # Add red time marker to all subplots (use hours if ZT mode, else minutes)
                time_x = time_seconds / 3600.0 if zt_mode_export else time_minutes
                for ax in plot_fig.get_axes():
                    ax.axvline(
                        x=time_x, color="red", linewidth=2,
                        linestyle="--", alpha=0.9, zorder=100
                    )

                # Render plot to image (suppress tight_layout warning)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plot_fig.tight_layout()
                plot_canvas = FigureCanvasAgg(plot_fig)
                plot_canvas.draw()

                # Get image from canvas
                buf = plot_canvas.buffer_rgba()
                plot_img = np.asarray(buf)
                plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

                # Resize plot to match frame width if needed
                if plot_img_bgr.shape[1] != frame_width or plot_img_bgr.shape[0] != plot_height:
                    plot_img_bgr = cv2.resize(plot_img_bgr, (frame_width, plot_height))

                # Close figure to free memory
                import matplotlib.pyplot as plt
                plt.close(plot_fig)

                # Combine frame and plot vertically
                combined = np.vstack([frame_bgr, plot_img_bgr])

                # Write frame
                out.write(combined)

            progress.setValue(n_frames)
            out.release()

            self._log_message(f"✅ Video with plots exported: {file_path}")
            self._log_message(f"   Dimensions: {output_width}x{output_height}, {n_frames} frames at {export_fps} FPS")

        except Exception as e:
            out.release()
            self._log_message(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()

    def _get_viewer_frame(self, frame_idx):
        """Get a single frame from the viewer data source."""
        import numpy as np

        if hasattr(self, "viewer_frame_cache") and self.viewer_frame_cache:
            return self.viewer_frame_cache[frame_idx].copy()

        # Use unified reader (handles AVI batch, HDF5, Zarr)
        frame = self._viewer_read_raw_frame(frame_idx)

        # Normalize to uint8 if needed.
        # Done with float32 + in-place arithmetic to minimise peak memory:
        # the previous version `((f - f.min()) / (f.max() - f.min()) * 255)`
        # produced three float64 temporaries (~30 MiB for a 1024x1224 frame),
        # which triggered MemoryError on memory-fragmented worker processes.
        # float32 halves the temporary, and `out=` removes the extra copies.
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            fmin = float(frame.min())
            fmax = float(frame.max())
            if fmax > fmin:
                scale = 255.0 / (fmax - fmin)
                buf = np.subtract(frame, fmin, dtype=np.float32)
                np.multiply(buf, scale, out=buf)
                np.clip(buf, 0.0, 255.0, out=buf)
                frame = buf.astype(np.uint8)
                del buf
            else:
                frame = np.zeros(frame.shape, dtype=np.uint8)

        return frame

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
                next_idx = 0
            # Show frame directly — do NOT go through slider → debounce chain.
            # At ≥17 FPS the 60ms debounce timer would never expire and rendering
            # would stall completely.
            self._viewer_show_frame(next_idx)
            # Keep slider in sync visually without re-triggering signals
            self.viewer_frame_slider.blockSignals(True)
            self.viewer_frame_slider.setValue(next_idx)
            self.viewer_frame_slider.blockSignals(False)

    def _viewer_update_timer_interval(self):
        """Update playback timer interval when FPS changes."""
        if self.viewer_is_playing:
            interval = int(1000 / self.viewer_fps_spin.value())
            self.viewer_timer.setInterval(interval)

    def _export_video(self):
        """Export selected frame range as MP4 video."""
        try:
            import cv2
            import numpy as np
            from qtpy.QtWidgets import QFileDialog

            # Get frame range
            start_frame = self.export_start_frame.value()
            end_frame = self.export_end_frame.value()

            if start_frame >= end_frame:
                self._log_message("❌ Start frame must be less than end frame")
                return

            # Get export FPS
            export_fps = self.export_fps_spin.value()

            # Ask user for save location
            default_name = f"export_frames_{start_frame}-{end_frame}.mp4"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Video",
                default_name,
                "MP4 Video (*.mp4);;All Files (*)",
            )

            if not file_path:
                return  # User cancelled

            self._log_message(
                f"🎬 Exporting frames {start_frame}-{end_frame} as video ({export_fps} FPS)..."
            )

            n_frames = end_frame - start_frame + 1
            from qtpy.QtWidgets import QProgressDialog
            from qtpy.QtCore import Qt
            progress_dlg = QProgressDialog(
                f"Exporting video ({n_frames} frames)...", "Cancel", 0, n_frames, self
            )
            progress_dlg.setWindowModality(Qt.WindowModal)
            progress_dlg.setWindowTitle("Export Progress")
            progress_dlg.setMinimumDuration(0)
            progress_dlg.setValue(0)
            progress_dlg.show()

            # Get first frame to determine video dimensions
            first_frame = self._prepare_frame_for_display(
                self._viewer_read_raw_frame(start_frame)
            )
            height, width = first_frame.shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(file_path, fourcc, export_fps, (width, height))

            if not out.isOpened():
                self._log_message(f"❌ Failed to create video file: {file_path}")
                return

            # Process each frame
            cancelled = False
            for idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if progress_dlg.wasCanceled():
                    cancelled = True
                    break

                frame = self._prepare_frame_for_display(
                    self._viewer_read_raw_frame(frame_idx)
                )

                # Add time text
                frame_time_seconds = frame_idx * self.viewer_frame_interval
                frame_time_minutes = frame_time_seconds / 60.0
                time_text = (
                    f"t = {frame_time_seconds:.1f}s ({frame_time_minutes:.2f}min)"
                )

                text_position = (10, height - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                color = (255, 255, 255)  # White
                thickness = 2

                cv2.putText(
                    frame,
                    time_text,
                    text_position,
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

                # Write frame
                out.write(frame)
                progress_dlg.setValue(idx + 1)

            # Finalize
            progress_dlg.setValue(n_frames)
            out.release()

            if cancelled:
                import os as _os
                try:
                    _os.remove(file_path)
                except OSError:
                    pass
                self._log_message("⚠️ Video export cancelled.")
                return

            self._log_message(f"✅ Video exported successfully: {file_path}")
            self._log_message(
                f"   {n_frames} frames at {export_fps} FPS ({n_frames/export_fps:.1f}s duration)"
            )

        except Exception as e:
            self._log_message(f"❌ Video export failed: {e}")
            import traceback

            traceback.print_exc()

    def _export_gif(self):
        """Export selected frame range as animated GIF."""
        try:
            import cv2
            import numpy as np
            from qtpy.QtWidgets import QFileDialog
            from PIL import Image

            # Get frame range
            start_frame = self.export_start_frame.value()
            end_frame = self.export_end_frame.value()

            if start_frame >= end_frame:
                self._log_message("❌ Start frame must be less than end frame")
                return

            # Pre-flight: GIF stores all frames in RAM simultaneously.
            # 1024×1224 RGB uint8 ≈ 3.6 MB per frame → 200 frames ≈ 720 MB.
            MAX_GIF_FRAMES = 200
            n_frames_requested = end_frame - start_frame + 1
            if n_frames_requested > MAX_GIF_FRAMES:
                self._log_message(
                    f"⚠️ GIF export blocked: {n_frames_requested} frames requested, "
                    f"maximum is {MAX_GIF_FRAMES} (GIF loads all frames into RAM at once). "
                    f"Use 'Export as Video (MP4)' for long sequences."
                )
                return

            # Get export FPS
            export_fps = self.export_fps_spin.value()
            frame_duration = int(1000 / export_fps)  # Duration in milliseconds

            # Ask user for save location
            default_name = f"export_frames_{start_frame}-{end_frame}.gif"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export GIF",
                default_name,
                "GIF Animation (*.gif);;All Files (*)",
            )

            if not file_path:
                return  # User cancelled

            n_frames = end_frame - start_frame + 1
            self._log_message(
                f"🎞️ Exporting frames {start_frame}-{end_frame} as GIF ({export_fps} FPS)..."
            )

            from qtpy.QtWidgets import QProgressDialog
            from qtpy.QtCore import Qt
            progress_dlg = QProgressDialog(
                f"Exporting GIF ({n_frames} frames)...", "Cancel", 0, n_frames, self
            )
            progress_dlg.setWindowModality(Qt.WindowModal)
            progress_dlg.setWindowTitle("Export Progress")
            progress_dlg.setMinimumDuration(0)
            progress_dlg.setValue(0)
            progress_dlg.show()

            frames_for_gif = []
            cancelled = False

            # Process each frame
            for idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if progress_dlg.wasCanceled():
                    cancelled = True
                    break

                try:
                    frame = self._prepare_frame_for_display(
                        self._viewer_read_raw_frame(frame_idx)
                    )
                except MemoryError:
                    self._log_message(
                        f"⚠️ GIF export ran out of memory at frame {idx + 1}/{n_frames}. "
                        f"Use 'Export as Video (MP4)' instead."
                    )
                    return

                # Add time text
                frame_time_seconds = frame_idx * self.viewer_frame_interval
                frame_time_minutes = frame_time_seconds / 60.0
                time_text = (
                    f"t = {frame_time_seconds:.1f}s ({frame_time_minutes:.2f}min)"
                )

                height, width = frame.shape[:2]
                text_position = (10, height - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                color = (255, 255, 255)  # White
                thickness = 2

                cv2.putText(
                    frame,
                    time_text,
                    text_position,
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    pil_image = Image.fromarray(frame_rgb)
                    frames_for_gif.append(pil_image)
                except MemoryError:
                    self._log_message(
                        f"⚠️ GIF export ran out of memory at frame {idx + 1}/{n_frames}. "
                        f"Use 'Export as Video (MP4)' instead."
                    )
                    return
                progress_dlg.setValue(idx + 1)

            progress_dlg.setValue(n_frames)

            if cancelled:
                self._log_message("⚠️ GIF export cancelled.")
                return

            # Save as GIF
            self._log_message("   Saving GIF file...")
            frames_for_gif[0].save(
                file_path,
                save_all=True,
                append_images=frames_for_gif[1:],
                duration=frame_duration,
                loop=0,
                optimize=False,
            )

            self._log_message(f"✅ GIF exported successfully: {file_path}")
            self._log_message(
                f"   {n_frames} frames at {export_fps} FPS ({n_frames/export_fps:.1f}s duration)"
            )

        except ImportError:
            self._log_message(
                "❌ PIL (Pillow) is required for GIF export. Install with: pip install Pillow"
            )
        except Exception as e:
            self._log_message(f"❌ GIF export failed: {e}")
            import traceback

            traceback.print_exc()

    def _export_gif_with_plots(self):
        """Export animated GIF with synchronized analysis plots (same style as Analysis tab)."""
        import cv2
        import numpy as np
        from qtpy.QtWidgets import QFileDialog, QProgressDialog
        from qtpy.QtCore import Qt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from ._plot import PlotGenerator, create_plot_config, create_hysteresis_kwargs
        from PIL import Image

        # Check prerequisites
        if not hasattr(self, "viewer_n_frames") or self.viewer_n_frames == 0:
            self._log_message("❌ No frames loaded. Load data into Frame Viewer first.")
            return

        if not hasattr(self, "merged_results") or not self.merged_results:
            self._log_message("❌ No analysis data. Run Main Analysis first.")
            return

        # Get frame range from export controls
        start_frame = self.export_start_frame.value()
        end_frame = self.export_end_frame.value()

        if start_frame >= end_frame:
            self._log_message("❌ Start frame must be less than end frame")
            return

        # Ask for save location
        default_name = f"gif_with_plots_{start_frame}-{end_frame}.gif"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export GIF with Plots",
            default_name,
            "GIF Animation (*.gif);;All Files (*)",
        )

        if not file_path:
            return

        export_fps = self.export_fps_spin.value()
        frame_duration = int(1000 / export_fps)  # Duration in milliseconds
        n_frames = end_frame - start_frame

        self._log_message(f"🎞️ Exporting {n_frames} frames with plots as GIF at {export_fps} FPS...")

        # Setup progress dialog
        progress = QProgressDialog("Exporting GIF with plots...", "Cancel", 0, n_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Export Progress")
        progress.show()

        # Process events to show dialog
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()

        plot_type, data_dict = self._get_sync_plot_data()

        if not data_dict:
            self._log_message(f"❌ No {self.sync_plot_type.currentText()} data available")
            return

        roi_colors = getattr(self, "roi_colors", {})
        n_rois = len(data_dict)

        # Get first frame to determine dimensions
        first_frame = self._get_viewer_frame(start_frame)
        frame_height, frame_width = first_frame.shape[:2]

        # Create plot figure dimensions matching Analysis tab style
        plot_dpi = 100
        plot_width = frame_width
        height_per_roi = 60  # pixels per ROI
        plot_height = max(200, height_per_roi * n_rois)
        fig_width = plot_width / plot_dpi
        fig_height = plot_height / plot_dpi

        # Total output dimensions: frame on top, plots below
        output_width = frame_width
        output_height = frame_height + plot_height

        # Create plot configuration from widget settings (same as Analysis tab)
        base_plot_config = create_plot_config(self)
        base_plot_config["fig_width"] = fig_width
        base_plot_config["height_per_roi"] = fig_height / n_rois if n_rois > 0 else 0.6
        base_plot_config["dpi"] = plot_dpi

        # Get extra kwargs for specific plot types
        if plot_type == "Raw Intensity Changes":
            hysteresis_kwargs = create_hysteresis_kwargs(widget_instance=self)
            hysteresis_kwargs.pop("merged_results", None)
        elif plot_type == "Sleep Quality":
            sleep_metric = "sleep_minutes"
            if hasattr(self, "sleep_quality_metric_combo"):
                metric_text = self.sleep_quality_metric_combo.currentText()
                if "Transitions" in metric_text:
                    sleep_metric = "transitions"
                elif "Bout" in metric_text:
                    sleep_metric = "bout_length"
            hysteresis_kwargs = {"sleep_metric": sleep_metric}
        else:
            hysteresis_kwargs = {}

        # Resolve ZT mode and lighting overlay (same logic as Analysis tab)
        zt_mode_gif = self.chk_zt_axis.isChecked() if hasattr(self, "chk_zt_axis") else False
        show_lighting_gif = self.chk_show_lighting.isChecked() if hasattr(self, "chk_show_lighting") else False
        if show_lighting_gif:
            _raw_led = getattr(self, "led_data", None)
            if _raw_led and isinstance(_raw_led, dict) and _raw_led.get("times") and _raw_led.get("white_powers"):
                led_data_gif = _raw_led
            else:
                led_data_gif = {}  # empty dict → legacy 12h fallback
        else:
            led_data_gif = None
        hysteresis_kwargs["zt_mode"] = zt_mode_gif
        hysteresis_kwargs["led_data"] = led_data_gif

        frames_for_gif = []

        try:
            for i, frame_idx in enumerate(range(start_frame, end_frame)):
                if progress.wasCanceled():
                    self._log_message("⚠️ Export cancelled by user")
                    return

                progress.setValue(i)

                # Log progress every 10 frames
                if i % 10 == 0:
                    self._log_message(f"   Processing frame {i + 1}/{n_frames}...")
                    QApplication.processEvents()

                # Get frame
                frame = self._get_viewer_frame(frame_idx)

                # Add time text to frame
                time_seconds = frame_idx * self.viewer_frame_interval
                time_minutes = time_seconds / 60.0
                time_text = f"t = {time_seconds:.1f}s ({time_minutes:.2f}min)"

                frame_bgr = frame.copy()
                if len(frame_bgr.shape) == 2:
                    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
                elif len(frame_bgr.shape) == 3:
                    if frame_bgr.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGBA2BGR)
                    elif frame_bgr.shape[2] == 1:
                        frame_bgr = cv2.cvtColor(frame_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)
                    # else: assume already BGR with 3 channels

                cv2.putText(
                    frame_bgr, time_text, (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                )

                # Draw ROI overlay (circles + numbers) if enabled
                if hasattr(self, "viewer_show_roi_overlay") and self.viewer_show_roi_overlay.isChecked():
                    self._draw_roi_overlay(frame_bgr)

                # Create plot using PlotGenerator (same as Analysis tab)
                plot_fig = Figure(figsize=(fig_width, fig_height), dpi=plot_dpi)
                plot_generator = PlotGenerator(plot_fig)

                # Generate plot with same settings as Analysis tab
                plot_generator.generate_plot(
                    plot_type, data_dict, roi_colors, base_plot_config, **hysteresis_kwargs
                )

                # Add red time marker to all subplots (use hours if ZT mode, else minutes)
                time_x_gif = time_seconds / 3600.0 if zt_mode_gif else time_minutes
                for ax in plot_fig.get_axes():
                    ax.axvline(
                        x=time_x_gif, color="red", linewidth=2,
                        linestyle="--", alpha=0.9, zorder=100
                    )

                # Render plot to image (suppress tight_layout warning)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plot_fig.tight_layout()
                plot_canvas = FigureCanvasAgg(plot_fig)
                plot_canvas.draw()

                # Get image from canvas
                buf = plot_canvas.buffer_rgba()
                plot_img = np.asarray(buf)
                plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

                # Resize plot to match frame width if needed
                if plot_img_bgr.shape[1] != frame_width or plot_img_bgr.shape[0] != plot_height:
                    plot_img_bgr = cv2.resize(plot_img_bgr, (frame_width, plot_height))

                # Close figure to free memory
                import matplotlib.pyplot as plt
                plt.close(plot_fig)

                # Combine frame and plot vertically
                combined = np.vstack([frame_bgr, plot_img_bgr])

                # Convert BGR to RGB for PIL
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(combined_rgb)
                frames_for_gif.append(pil_image)

            progress.setValue(n_frames)

            # Save as GIF
            self._log_message("   Saving GIF file...")
            frames_for_gif[0].save(
                file_path,
                save_all=True,
                append_images=frames_for_gif[1:],
                duration=frame_duration,
                loop=0,
                optimize=False,
            )

            self._log_message(f"✅ GIF with plots exported: {file_path}")
            self._log_message(f"   Dimensions: {output_width}x{output_height}, {n_frames} frames at {export_fps} FPS")

        except ImportError as e:
            if "PIL" in str(e) or "Pillow" in str(e):
                self._log_message("❌ PIL (Pillow) is required for GIF export. Install with: pip install Pillow")
            else:
                self._log_message(f"❌ Import error: {e}")
        except Exception as e:
            self._log_message(f"❌ GIF export failed: {e}")
            import traceback
            traceback.print_exc()

    def _get_recording_start_str(self) -> str:
        """Return formatted recording start datetime for plot titles, or empty string."""
        dt = getattr(self, "recording_start_datetime", None)
        if dt is None:
            return ""
        return f"  [{dt.strftime('%d.%m.%Y  %H:%M')}]"
