"""
_plot.py - Plotting module for HDF5 Analysis

This module handles all plotting functionality for the HDF5 analysis widget.
It contains methods for generating different types of plots including:
- Raw intensity changes with hysteresis visualization
- Movement data
- Fraction movement
- Quiescence
- Sleep
- Daylight cycle (dark IR)

All plotting functions are designed to work with matplotlib figures and can be
used independently of the UI widget.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.colors as mcolors

try:
    from ._calc import bin_activity_data_for_lighting

    CALC_AVAILABLE = True
except ImportError:
    CALC_AVAILABLE = False
    print("Warning: _calc module not available for lighting plot binning")


class PlotGenerator:
    """
    Class to handle all plotting functionality for HDF5 analysis.
    This class is UI-independent and can generate plots based on data and parameters.
    """

    def __init__(self, figure: Figure):
        """
        Initialize the plot generator with a matplotlib figure.

        Args:
            figure: matplotlib Figure object to draw plots on
        """
        self.figure = figure

    def generate_plot(
        self,
        plot_type: str,
        data_dict: Dict,
        roi_colors: Dict,
        plot_config: Dict,
        **kwargs,
    ) -> bool:
        """
        Generate a plot based on the specified type and configuration.

        Args:
            plot_type: Type of plot to generate
            data_dict: Dictionary containing the data to plot
            roi_colors: Dictionary mapping ROI IDs to colors
            plot_config: Dictionary containing plot configuration parameters
            **kwargs: Additional arguments specific to plot types

        Returns:
            bool: True if plot was generated successfully, False otherwise
        """
        try:
            # Thorough figure cleanup to prevent artifacts
            self.figure.clear()

            # Just use figure.clear() - it handles cleanup properly
            self.figure.clear()

            # Reset figure state
            self.figure.patch.set_visible(True)

            # Clear any cached renderers
            self.figure._cachedRenderer = None

            # Set figure properties
            dpi = plot_config.get("dpi", 100)
            fig_width = plot_config.get("fig_width", 10.0)
            height_per_roi = plot_config.get("height_per_roi", 0.6)

            self.figure.set_dpi(dpi)

            # Calculate figure height based on number of ROIs
            num_rois = len(data_dict)
            fig_height = max(1.0, height_per_roi * num_rois)
            self.figure.set_size_inches(fig_width, fig_height)

            # Route to appropriate plotting method
            if plot_type == "Raw Intensity Changes":
                return self._plot_raw_intensity_enhanced(
                    data_dict, roi_colors, plot_config, **kwargs
                )
            elif plot_type == "Movement":
                return self._plot_movement(data_dict, roi_colors, plot_config)
            elif plot_type == "Fraction Movement":
                return self._plot_fraction_movement(data_dict, roi_colors, plot_config)
            elif plot_type == "Quiescence":
                return self._plot_quiescence(data_dict, roi_colors, plot_config)
            elif plot_type == "Sleep":
                return self._plot_sleep(data_dict, roi_colors, plot_config)
            elif plot_type == "Lighting Conditions (dark IR)":
                return self._plot_lighting_conditions(
                    data_dict, roi_colors, plot_config, **kwargs
                )
            else:
                print(f"Unsupported plot type: {plot_type}")
                return False

        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            return False

    def _plot_raw_intensity_enhanced(
        self, merged_results: Dict, roi_colors: Dict, plot_config: Dict, **kwargs
    ) -> bool:
        """
        Plot raw intensity changes with hysteresis visualization.

        Args:
            merged_results: Dictionary of ROI intensity data
            roi_colors: Dictionary mapping ROI IDs to colors
            plot_config: Plot configuration parameters
            **kwargs: Additional parameters including hysteresis data
        """
        # Extract time range (convert from minutes to seconds)
        start_t_minutes = plot_config.get("start_time", 0.0)
        end_t_minutes = plot_config.get("end_time", 1000.0)
        start_t = start_t_minutes * 60.0
        end_t = end_t_minutes * 60.0

        # Extract hysteresis data from kwargs
        roi_baseline_means = kwargs.get("roi_baseline_means", {})
        roi_band_widths = kwargs.get("roi_band_widths", {})
        roi_upper_thresholds = kwargs.get("roi_upper_thresholds", {})
        roi_lower_thresholds = kwargs.get("roi_lower_thresholds", {})

        # Visualization options
        show_baseline_mean = kwargs.get("show_baseline_mean", True)
        show_deviation_band = kwargs.get("show_deviation_band", True)
        show_detection_threshold = kwargs.get("show_detection_threshold", True)
        show_threshold_stats = kwargs.get("show_threshold_stats", True)

        # Option to recalculate baseline for visible range
        recalculate_baseline_for_range = kwargs.get(
            "recalculate_baseline_for_range", False
        )

        sorted_rois = sorted(merged_results.keys())
        n_rois = len(sorted_rois)

        if n_rois == 0:
            self.figure.suptitle("No intensity data available", fontsize=14)
            return False

        # Create subplot grid
        gs = self.figure.add_gridspec(n_rois, 1, hspace=0.4)
        self.figure.subplots_adjust(left=0.18)
        axes = []

        for i, roi in enumerate(sorted_rois):
            # Create subplot
            if i == 0:
                ax_roi = self.figure.add_subplot(gs[i, 0])
                title = "ROI Intensity with Hysteresis Detection System"
                ax_roi.set_title(title, fontsize=12)
            else:
                ax_roi = self.figure.add_subplot(gs[i, 0], sharex=axes[0])

            axes.append(ax_roi)

            # Filter data to time range
            data = merged_results[roi]
            data_in_range = [(t, c) for (t, c) in data if start_t <= t <= end_t]

            if not data_in_range:
                ax_roi.text(
                    0.5,
                    0.5,
                    f"No data for ROI {roi} in selected time range",
                    ha="center",
                    va="center",
                    transform=ax_roi.transAxes,
                )
                ax_roi.set_xlim(start_t_minutes, end_t_minutes)
                continue

            # Extract and convert time to minutes
            times, changes = zip(*data_in_range)
            times = np.array(times, dtype=float)
            changes = np.array(changes, dtype=float)
            times_minutes = times / 60.0

            # Plot the intensity changes
            color = roi_colors.get(roi, f"C{i}")
            ax_roi.plot(
                times_minutes, changes, color=color, linewidth=1.0, alpha=0.8, zorder=3
            )
            ax_roi.set_xlim(start_t_minutes, end_t_minutes)

            # Add hysteresis visualization
            if roi in roi_baseline_means:
                baseline_mean = roi_baseline_means[roi]

                # Determine thresholds - use explicit values if available, otherwise calculate
                if roi in roi_upper_thresholds and roi in roi_lower_thresholds:
                    # Use explicit threshold values from analysis
                    upper_threshold = roi_upper_thresholds[roi]
                    lower_threshold = roi_lower_thresholds[roi]
                    band_width = (upper_threshold - lower_threshold) / 2.0

                    baseline_label = "Baseline Mean (Analysis)"
                    threshold_label = "Detection Thresholds (Analysis)"
                    band_label = "Hysteresis Band (Analysis)"

                elif roi in roi_band_widths:
                    # Calculate thresholds from baseline + band width
                    band_width = roi_band_widths[roi]
                    upper_threshold = baseline_mean + band_width
                    lower_threshold = baseline_mean - band_width

                    baseline_label = "Baseline Mean (Calculated)"
                    threshold_label = "Detection Thresholds (Calculated)"
                    band_label = "Hysteresis Band (Calculated)"

                else:
                    # No threshold data available - skip visualization
                    print(f"Warning: No threshold data available for ROI {roi}")
                    # Format subplot without thresholds
                    self._format_subplot_enhanced(
                        ax_roi, roi, i, n_rois, color, merged_results, plot_config
                    )
                    continue

                # OPTION: Recalculate baseline for visible time range (if requested)
                if recalculate_baseline_for_range and len(data_in_range) > 10:
                    # Calculate baseline from visible data
                    visible_values = np.array(changes)
                    visible_baseline_mean = np.mean(visible_values)
                    visible_baseline_std = np.std(visible_values)

                    # Use original multiplier if available, otherwise estimate
                    multiplier = kwargs.get("threshold_multiplier", 1.0)
                    visible_band_width = multiplier * visible_baseline_std

                    # Override with recalculated values
                    baseline_mean = visible_baseline_mean
                    band_width = visible_band_width
                    upper_threshold = baseline_mean + band_width
                    lower_threshold = baseline_mean - band_width

                    # Update labels
                    baseline_label = f"Baseline Mean (Range: {start_t_minutes:.0f}-{end_t_minutes:.0f}min)"
                    threshold_label = f"Detection Thresholds (Range-specific)"
                    band_label = f"Hysteresis Band (Range-specific)"

                # Check if baseline is visible in current range
                if len(changes) > 0:
                    y_min_plot = np.min(changes)
                    y_max_plot = np.max(changes)
                    baseline_visible = y_min_plot <= baseline_mean <= y_max_plot
                else:
                    baseline_visible = False

                # Plot baseline mean line
                if show_baseline_mean:
                    linestyle = "-" if baseline_visible else "--"
                    alpha = 0.8 if baseline_visible else 0.5
                    ax_roi.axhline(
                        y=baseline_mean,
                        linestyle=linestyle,
                        color="red",
                        alpha=alpha,
                        linewidth=2.0,
                        zorder=4,
                        label=baseline_label,
                    )

                # Plot hysteresis band
                if show_deviation_band:
                    alpha_band = 0.2 if baseline_visible else 0.1
                    ax_roi.fill_between(
                        times_minutes,
                        lower_threshold,
                        upper_threshold,
                        alpha=alpha_band,
                        color="orange",
                        zorder=2,
                        label=band_label,
                    )

                # Plot detection thresholds
                if show_detection_threshold:
                    linestyle = "--" if baseline_visible else ":"
                    alpha_thresh = 0.9 if baseline_visible else 0.5
                    ax_roi.axhline(
                        y=upper_threshold,
                        linestyle=linestyle,
                        color="darkred",
                        alpha=alpha_thresh,
                        linewidth=2.0,
                        zorder=5,
                        label=threshold_label,
                    )
                    ax_roi.axhline(
                        y=lower_threshold,
                        linestyle=linestyle,
                        color="darkred",
                        alpha=alpha_thresh,
                        linewidth=2.0,
                        zorder=5,
                    )

                # Add statistics text if requested
                if show_threshold_stats:
                    stats_text = (
                        f"Baseline: {baseline_mean:.1f}\n"
                        f"Upper: {upper_threshold:.1f}\n"
                        f"Lower: {lower_threshold:.1f}\n"
                        f"Band: ±{band_width:.1f}"
                    )
                    ax_roi.text(
                        0.02,
                        0.98,
                        stats_text,
                        transform=ax_roi.transAxes,
                        verticalalignment="top",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )

            # Format subplot
            self._format_subplot_enhanced(
                ax_roi, roi, i, n_rois, color, merged_results, plot_config
            )

        # Format shared axes
        self._format_shared_axes_minutes(axes, start_t_minutes, end_t_minutes)
        self.figure.text(
            0.01,
            0.5,
            "Normalized Intensity Change",
            va="center",
            rotation="vertical",
            fontsize=11,
        )

        return True

    def _plot_movement(
        self, movement_data: Dict, roi_colors: Dict, plot_config: Dict
    ) -> bool:
        """Plot movement data with separate subplots for each ROI."""
        return self._plot_binary_data(
            movement_data,
            roi_colors,
            plot_config,
            "Movement Data",
            ["No", "Yes"],
            "Movement",
        )

    def _plot_fraction_movement(
        self, fraction_data: Dict, roi_colors: Dict, plot_config: Dict
    ) -> bool:
        """Plot fraction movement data."""
        return self._plot_continuous_data(
            fraction_data,
            roi_colors,
            plot_config,
            "Fraction Movement",
            "Fraction Movement",
            y_range=(0, 1.05),
        )

    def _plot_quiescence(
        self, quiescence_data: Dict, roi_colors: Dict, plot_config: Dict
    ) -> bool:
        """Plot quiescence data."""
        return self._plot_binary_data(
            quiescence_data,
            roi_colors,
            plot_config,
            "Quiescence",
            ["No", "Yes"],
            "Quiescence",
        )

    def _plot_sleep(
        self, sleep_data: Dict, roi_colors: Dict, plot_config: Dict
    ) -> bool:
        """Plot sleep data."""
        return self._plot_binary_data(
            sleep_data,
            roi_colors,
            plot_config,
            "Sleep",
            ["Awake", "Sleep"],
            "Sleep State",
        )

    def _plot_lighting_conditions(
        self, fraction_data: Dict, roi_colors: Dict, plot_config: Dict, **kwargs
    ) -> bool:
        """Generate lighting conditions plot with separate subplots for each ROI."""
        try:
            # Get parameters
            bin_minutes = kwargs.get("bin_minutes", 30)

            # Prepare data using calc module or fallback
            if CALC_AVAILABLE:
                from ._calc import bin_activity_data_for_lighting

                binned_data = bin_activity_data_for_lighting(fraction_data, bin_minutes)
            else:
                # Fallback: use fraction_data directly
                print("Using fraction_data directly for lighting plot")
                binned_data = fraction_data

            if not binned_data:
                print("No binned data available for lighting plot")
                return False

            # Extract time range and convert to hours
            start_t_minutes = plot_config.get("start_time", 0.0)
            end_t_minutes = plot_config.get("end_time", 1000.0)
            start_t = start_t_minutes * 60.0  # Convert to seconds
            end_t = end_t_minutes * 60.0
            start_hours = start_t / 3600
            end_hours = end_t / 3600

            sorted_rois = sorted(binned_data.keys())
            n_rois = len(sorted_rois)

            if n_rois == 0:
                self.figure.suptitle("No lighting data available", fontsize=14)
                return False

            # Create gridspec for subplots
            gs = self.figure.add_gridspec(n_rois, 1, hspace=0.4)
            self.figure.subplots_adjust(left=0.18)
            axes = []

            for i, roi in enumerate(sorted_rois):
                # Create subplot with shared x-axis
                if i == 0:
                    ax_roi = self.figure.add_subplot(gs[i, 0])
                    title = f"Activity Pattern - Lighting Conditions (dark IR) ({bin_minutes}min bins)"
                    ax_roi.set_title(title, fontsize=12)
                else:
                    ax_roi = self.figure.add_subplot(gs[i, 0], sharex=axes[0])

                axes.append(ax_roi)

                # Get data for this ROI
                data = binned_data[roi]
                if not data:
                    ax_roi.text(
                        0.5,
                        0.5,
                        f"No data for ROI {roi} in selected time range",
                        ha="center",
                        va="center",
                        transform=ax_roi.transAxes,
                    )
                    ax_roi.set_xlim(start_hours, end_hours)
                    continue

                # Filter data to time range and convert to hours
                times_seconds, activities = zip(*data)
                data_in_range = [
                    (t / 3600, a)
                    for t, a in zip(times_seconds, activities)
                    if start_t <= t <= end_t
                ]

                if not data_in_range:
                    ax_roi.text(
                        0.5,
                        0.5,
                        f"No data for ROI {roi} in selected time range",
                        ha="center",
                        va="center",
                        transform=ax_roi.transAxes,
                    )
                    ax_roi.set_xlim(start_hours, end_hours)
                    continue

                times_hours, activities = zip(*data_in_range)
                times_hours = np.array(times_hours)
                activities = np.array(activities)

                # Get ROI color
                color = roi_colors.get(roi, f"C{i}")

                # Plot activity data
                ax_roi.plot(
                    times_hours,
                    activities,
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    marker="o",
                    markersize=3,
                )
                ax_roi.fill_between(times_hours, activities, 0, alpha=0.3, color=color)

                # Add lighting period indicators
                self._add_lighting_periods(ax_roi, start_hours, end_hours, i == 0)

                # Set axis limits and formatting
                ax_roi.set_xlim(start_hours, end_hours)

                # Y-axis scaling
                self._apply_y_axis_scaling(ax_roi, activities, plot_config)

                # Add ROI label
                ax_roi.text(
                    1.01,
                    0.5,
                    f"ROI {roi}",
                    transform=ax_roi.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    ha="left",
                    va="center",
                )

                # X-axis handling
                if i < n_rois - 1:
                    ax_roi.set_xticklabels([])
                    ax_roi.set_xlabel("")
                    ax_roi.tick_params(
                        axis="x",
                        which="both",
                        bottom=True,
                        top=False,
                        labelbottom=False,
                    )
                else:
                    ax_roi.set_xlabel("Time (hours)")
                    ax_roi.xaxis.label.set_fontsize(11)
                    ax_roi.tick_params(axis="x", labelsize=10)

                # Add gridlines and clean up spines
                ax_roi.grid(True, alpha=0.3)
                ax_roi.spines["top"].set_visible(False)
                ax_roi.spines["right"].set_visible(False)

                # Add legend only to first subplot
                if i == 0:
                    ax_roi.legend(loc="upper right", fontsize=8)

            # Format shared x-axis for hours
            self._format_shared_axes_hours(axes, start_hours, end_hours)

            # Add Y-axis label
            self.figure.text(
                0.01,
                0.5,
                "Activity Level",
                va="center",
                rotation="vertical",
                fontsize=11,
            )

            return True

        except Exception as e:
            print(f"Error generating lighting plot: {e}")
            return False

    def _plot_binary_data(
        self,
        data_dict: Dict,
        roi_colors: Dict,
        plot_config: Dict,
        title: str,
        y_labels: List[str],
        y_axis_label: str,
    ) -> bool:
        """Generic method for plotting binary data with minutes axis."""
        # Convert time range
        start_t_minutes = plot_config.get("start_time", 0.0)
        end_t_minutes = plot_config.get("end_time", 1000.0)
        start_t = start_t_minutes * 60.0
        end_t = end_t_minutes * 60.0

        sorted_rois = sorted(data_dict.keys())
        n_rois = len(sorted_rois)

        if n_rois == 0:
            self.figure.suptitle(f"No {title.lower()} available", fontsize=14)
            return False

        gs = self.figure.add_gridspec(n_rois, 1, hspace=0.3)
        self.figure.subplots_adjust(left=0.18)
        axes = []

        for i, roi in enumerate(sorted_rois):
            if i == 0:
                ax_roi = self.figure.add_subplot(gs[i, 0])
                ax_roi.set_title(
                    f"{title} from {start_t_minutes:.1f} to {end_t_minutes:.1f} min"
                )
            else:
                ax_roi = self.figure.add_subplot(gs[i, 0], sharex=axes[0])

            axes.append(ax_roi)
            data = data_dict[roi]
            data_in_range = [(t, s) for (t, s) in data if start_t <= t <= end_t]

            ax_roi.set_xlim(start_t_minutes, end_t_minutes)

            if not data_in_range:
                ax_roi.text(
                    0.5,
                    0.5,
                    f"No data for ROI {roi} in selected time range",
                    ha="center",
                    va="center",
                    transform=ax_roi.transAxes,
                )
                continue

            times, states = zip(*data_in_range)
            times_minutes = np.array(times) / 60.0
            color = roi_colors.get(roi, f"C{i}")
            ax_roi.step(times_minutes, states, where="mid", color=color, linewidth=1.2)
            ax_roi.fill_between(
                times_minutes, states, 0, step="mid", alpha=0.3, color=color
            )

            ax_roi.set_ylim(-0.1, 1.1)
            ax_roi.set_yticks([0, 1])
            ax_roi.set_yticklabels(y_labels)

            # Add ROI label
            ax_roi.text(
                1.01,
                0.5,
                f"ROI {roi}",
                transform=ax_roi.transAxes,
                fontsize=10,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
            )

            # X-axis handling
            if i < n_rois - 1:
                ax_roi.set_xticklabels([])
                ax_roi.set_xlabel("")
            else:
                ax_roi.set_xlabel("Time (min)")

            ax_roi.grid(True, alpha=0.3)

        self._format_shared_axes_minutes(axes, start_t_minutes, end_t_minutes)
        self.figure.text(
            0.02, 0.5, y_axis_label, va="center", rotation="vertical", fontsize=12
        )

        return True

    def _plot_continuous_data(
        self,
        data_dict: Dict,
        roi_colors: Dict,
        plot_config: Dict,
        title: str,
        y_axis_label: str,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """Generic method for plotting continuous data with minutes axis."""
        # Convert time range
        start_t_minutes = plot_config.get("start_time", 0.0)
        end_t_minutes = plot_config.get("end_time", 1000.0)
        start_t = start_t_minutes * 60.0
        end_t = end_t_minutes * 60.0

        sorted_rois = sorted(data_dict.keys())
        n_rois = len(sorted_rois)

        if n_rois == 0:
            self.figure.suptitle(f"No {title.lower()} available", fontsize=14)
            return False

        gs = self.figure.add_gridspec(n_rois, 1, hspace=0.3)
        self.figure.subplots_adjust(left=0.18)
        axes = []

        for i, roi in enumerate(sorted_rois):
            if i == 0:
                ax_roi = self.figure.add_subplot(gs[i, 0])
                ax_roi.set_title(
                    f"{title} from {start_t_minutes:.1f} to {end_t_minutes:.1f} min"
                )
            else:
                ax_roi = self.figure.add_subplot(gs[i, 0], sharex=axes[0])

            axes.append(ax_roi)
            data = data_dict[roi]
            data_in_range = [(t, f) for (t, f) in data if start_t <= t <= end_t]

            ax_roi.set_xlim(start_t_minutes, end_t_minutes)

            if not data_in_range:
                ax_roi.text(
                    0.5,
                    0.5,
                    f"No data for ROI {roi} in selected time range",
                    ha="center",
                    va="center",
                    transform=ax_roi.transAxes,
                )
                continue

            times, values = zip(*data_in_range)
            times_minutes = np.array(times) / 60.0
            color = roi_colors.get(roi, f"C{i}")
            ax_roi.plot(
                times_minutes,
                values,
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
            )
            ax_roi.fill_between(times_minutes, values, 0, alpha=0.2, color=color)

            if y_range:
                ax_roi.set_ylim(*y_range)
                if title == "Fraction Movement":
                    ax_roi.axhline(
                        y=0.5, linestyle=":", color="gray", linewidth=0.8, alpha=0.7
                    )
                    ax_roi.set_yticks([0, 0.5, 1.0])

            # Add ROI label
            ax_roi.text(
                1.01,
                0.5,
                f"ROI {roi}",
                transform=ax_roi.transAxes,
                fontsize=10,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
            )

            # X-axis handling
            if i < n_rois - 1:
                ax_roi.set_xticklabels([])
                ax_roi.set_xlabel("")
            else:
                ax_roi.set_xlabel("Time (min)")

            ax_roi.grid(True, alpha=0.3)

        self._format_shared_axes_minutes(axes, start_t_minutes, end_t_minutes)
        self.figure.text(
            0.02, 0.5, y_axis_label, va="center", rotation="vertical", fontsize=12
        )

        return True

    def _format_subplot_enhanced(
        self,
        ax_roi,
        roi: int,
        index: int,
        total_rois: int,
        color: str,
        data_dict: Dict,
        plot_config: Dict,
    ):
        """Enhanced subplot formatting with improved Y-axis scaling."""
        # Apply scientific notation
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax_roi.yaxis.set_major_formatter(formatter)
        ax_roi.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

        # Apply Y-axis scaling
        auto_scale_y = plot_config.get("auto_scale_y", True)
        if auto_scale_y:
            # Calculate robust limits for this specific ROI
            roi_data = data_dict.get(roi, [])
            start_t = plot_config.get("start_time", 0.0) * 60.0  # Convert to seconds
            end_t = plot_config.get("end_time", 1000.0) * 60.0
            roi_data_in_range = [
                (t, val) for (t, val) in roi_data if start_t <= t <= end_t
            ]

            if roi_data_in_range:
                values = np.array([val for (_, val) in roi_data_in_range])

                robust_scaling = plot_config.get("robust_scaling", True)
                if robust_scaling:
                    lower_percentile = plot_config.get("lower_percentile", 5.0)
                    upper_percentile = plot_config.get("upper_percentile", 95.0)
                    y_min = np.percentile(values, lower_percentile)
                    y_max = np.percentile(values, upper_percentile)
                else:
                    y_min = np.min(values)
                    y_max = np.max(values)

                # Add margin
                margin = (y_max - y_min) * 0.05
                ax_roi.set_ylim(y_min - margin, y_max + margin)
        else:
            # Use manual Y-axis range
            y_min = plot_config.get("y_min", 0.0)
            y_max = plot_config.get("y_max", 1000.0)
            ax_roi.set_ylim(y_min, y_max)

        # Add ROI label
        ax_roi.text(
            1.01,
            0.5,
            f"ROI {roi}",
            transform=ax_roi.transAxes,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )

        # X-axis handling
        if index < total_rois - 1:
            ax_roi.set_xticklabels([])
            ax_roi.set_xlabel("")
            ax_roi.tick_params(
                axis="x", which="both", bottom=True, top=False, labelbottom=False
            )
        else:
            ax_roi.set_xlabel("Time (min)")
            ax_roi.xaxis.label.set_fontsize(11)
            ax_roi.tick_params(axis="x", labelsize=10)

        # Add gridlines and clean up spines
        ax_roi.grid(True, alpha=0.3)
        ax_roi.spines["top"].set_visible(False)
        ax_roi.spines["right"].set_visible(False)

    def _format_shared_axes_minutes(
        self, axes: List, start_t_minutes: float, end_t_minutes: float
    ):
        """Format shared x-axis for all subplots with minutes scale."""
        if not axes:
            return

        # Calculate optimal tick spacing in MINUTES
        time_range_minutes = end_t_minutes - start_t_minutes
        if time_range_minutes > 2000:  # > ~33 hours
            interval_minutes = 500  # 500-minute ticks
        elif time_range_minutes > 1000:  # > ~16 hours
            interval_minutes = 200  # 200-minute ticks
        elif time_range_minutes > 500:  # > ~8 hours
            interval_minutes = 100  # 100-minute ticks
        elif time_range_minutes > 200:  # > ~3 hours
            interval_minutes = 50  # 50-minute ticks
        elif time_range_minutes > 100:  # > ~1.5 hours
            interval_minutes = 20  # 20-minute ticks
        else:
            interval_minutes = 10  # 10-minute ticks

        start_tick = (start_t_minutes // interval_minutes) * interval_minutes
        if start_tick < start_t_minutes:
            start_tick += interval_minutes

        ticks = np.arange(start_tick, end_t_minutes + 1, interval_minutes)
        if len(ticks) < 2:
            ticks = np.linspace(start_t_minutes, end_t_minutes, 5)

        # Apply to all subplots
        for ax_roi in axes:
            ax_roi.set_xticks(ticks)
            ax_roi.set_xlim(start_t_minutes, end_t_minutes)

        # Format bottom axis with minutes
        axes[-1].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f"{int(x)}")
        )
        axes[-1].set_xlabel("Time (min)")

    def _format_shared_axes_hours(
        self, axes: List, start_hours: float, end_hours: float
    ):
        """Format shared x-axis for all subplots with hours scale."""
        if not axes:
            return

        # Calculate optimal tick spacing in hours
        time_range_hours = end_hours - start_hours
        if time_range_hours > 120:  # > 5 days
            interval_hours = 24  # Daily ticks
        elif time_range_hours > 48:  # > 2 days
            interval_hours = 12  # Twice daily
        elif time_range_hours > 24:  # > 1 day
            interval_hours = 6  # 4 times daily
        else:
            interval_hours = 3  # Every 3 hours

        start_tick = (start_hours // interval_hours) * interval_hours
        if start_tick < start_hours:
            start_tick += interval_hours

        ticks = np.arange(start_tick, end_hours + 1, interval_hours)
        if len(ticks) < 2:
            ticks = np.linspace(start_hours, end_hours, 5)

        # Apply to all subplots
        for ax_roi in axes:
            ax_roi.set_xticks(ticks)
            ax_roi.set_xlim(start_hours, end_hours)

        # Format the bottom axis
        axes[-1].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f"{int(x)}h")
        )

    def _apply_y_axis_scaling(self, ax_roi, activities: np.ndarray, plot_config: Dict):
        """Apply Y-axis scaling based on configuration."""
        auto_scale_y = plot_config.get("auto_scale_y", True)

        # Ensure activities is a numpy array and has data
        if not isinstance(activities, np.ndarray):
            activities = np.array(activities)

        if len(activities) == 0:
            # No data, use default range
            ax_roi.set_ylim(0, 1)
            return

        if auto_scale_y:
            robust_scaling = plot_config.get("robust_scaling", True)
            if robust_scaling:
                lower_percentile = plot_config.get("lower_percentile", 5.0)
                upper_percentile = plot_config.get("upper_percentile", 95.0)

                # Handle edge cases
                if len(activities) < 2:
                    y_min = float(np.min(activities))
                    y_max = float(np.max(activities))
                else:
                    y_min = float(np.percentile(activities, lower_percentile))
                    y_max = float(np.percentile(activities, upper_percentile))
            else:
                y_min = float(np.min(activities))
                y_max = float(np.max(activities))

            # Add margin and handle edge cases
            if y_max == y_min:
                margin = abs(y_max) * 0.1 if y_max != 0 else 1.0
                y_min -= margin
                y_max += margin
            else:
                margin = (y_max - y_min) * 0.05
                y_min -= margin
                y_max += margin

            ax_roi.set_ylim(max(0, y_min), y_max)
        else:
            # Use manual Y-axis range
            y_min = plot_config.get("y_min", 0.0)
            y_max = plot_config.get("y_max", 1000.0)
            ax_roi.set_ylim(y_min, y_max)

    def _add_lighting_periods(
        self, ax_roi, start_hours: float, end_hours: float, add_legend: bool = False
    ):
        """Add lighting period indicators to the plot."""
        # Light periods (assuming lights on from 7:00 to 19:00)
        light_start_hour = 7
        light_end_hour = 19

        # Calculate light/dark periods within the plot range
        plot_start_day = int(start_hours // 24)
        plot_end_day = int(end_hours // 24) + 1

        for day in range(plot_start_day, plot_end_day + 1):
            day_start = day * 24
            light_start = day_start + light_start_hour
            light_end = day_start + light_end_hour
            dark_start = light_end
            dark_end = day_start + 24 + light_start_hour

            # Light period (yellow background)
            if light_start <= end_hours and light_end >= start_hours:
                light_plot_start = max(light_start, start_hours)
                light_plot_end = min(light_end, end_hours)
                ax_roi.axvspan(
                    light_plot_start,
                    light_plot_end,
                    alpha=0.2,
                    color="yellow",
                    zorder=0,
                    label="Light" if day == plot_start_day and add_legend else "",
                )

            # Dark period (gray background)
            if dark_start <= end_hours and dark_end >= start_hours:
                dark_plot_start = max(dark_start, start_hours)
                dark_plot_end = min(dark_end, end_hours)
                ax_roi.axvspan(
                    dark_plot_start,
                    dark_plot_end,
                    alpha=0.2,
                    color="gray",
                    zorder=0,
                    label="Dark" if day == plot_start_day and add_legend else "",
                )


# Utility functions for plot configuration
def create_plot_config(widget_instance=None, **kwargs) -> Dict:
    """
    Create a plot configuration dictionary from widget parameters or kwargs.

    Args:
        widget_instance: Optional widget instance to extract parameters from
        **kwargs: Override parameters

    Returns:
        Dictionary containing plot configuration
    """
    config = {
        "dpi": 100,
        "fig_width": 10.0,
        "height_per_roi": 0.6,
        "start_time": 0.0,  # in minutes
        "end_time": 1000.0,  # in minutes
        "auto_scale_y": True,
        "robust_scaling": True,
        "adaptive_scaling": True,  # NEW
        "center_around_zero": True,  # NEW
        "lower_percentile": 5.0,
        "upper_percentile": 95.0,
        "y_min": 0.0,
        "y_max": 1000.0,
    }

    # Extract from widget if provided
    if widget_instance is not None:
        try:
            config.update(
                {
                    "dpi": widget_instance.plot_dpi_spin.value(),
                    "fig_width": widget_instance.plot_width_spin.value(),
                    "height_per_roi": widget_instance.plot_height_spin.value(),
                    "start_time": widget_instance.plot_start_time.value(),
                    "end_time": widget_instance.plot_end_time.value(),
                    "auto_scale_y": widget_instance.auto_scale_y.isChecked(),
                    "robust_scaling": widget_instance.robust_scaling.isChecked(),
                    "lower_percentile": widget_instance.lower_percentile_spin.value(),
                    "upper_percentile": widget_instance.upper_percentile_spin.value(),
                    "y_min": widget_instance.y_min_spin.value(),
                    "y_max": widget_instance.y_max_spin.value(),
                }
            )

            # Add new scaling options if available
            if hasattr(widget_instance, "adaptive_scaling"):
                config["adaptive_scaling"] = (
                    widget_instance.adaptive_scaling.isChecked()
                )
            if hasattr(widget_instance, "center_around_zero"):
                config["center_around_zero"] = (
                    widget_instance.center_around_zero.isChecked()
                )

        except AttributeError as e:
            print(f"Warning: Could not extract all parameters from widget: {e}")

    # Override with kwargs
    config.update(kwargs)

    return config


def create_hysteresis_kwargs(widget_instance=None, **kwargs) -> Dict:
    """
    Create hysteresis-specific kwargs for raw intensity plotting.
    Updated to handle calibration method parameters.
    """
    hysteresis_kwargs = {
        "roi_baseline_means": {},
        "roi_band_widths": {},
        "roi_upper_thresholds": {},
        "roi_lower_thresholds": {},
        "show_baseline_mean": True,
        "show_deviation_band": True,
        "show_detection_threshold": True,
        "show_threshold_stats": True,
        "threshold_multiplier": 1.0,
    }

    # Extract from widget if provided
    if widget_instance is not None:
        try:
            hysteresis_kwargs.update(
                {
                    "roi_baseline_means": getattr(
                        widget_instance, "roi_baseline_means", {}
                    ),
                    "roi_band_widths": getattr(widget_instance, "roi_band_widths", {}),
                    "roi_upper_thresholds": getattr(
                        widget_instance, "roi_upper_thresholds", {}
                    ),
                    "roi_lower_thresholds": getattr(
                        widget_instance, "roi_lower_thresholds", {}
                    ),
                    "show_baseline_mean": widget_instance.show_baseline_mean.isChecked(),
                    "show_deviation_band": widget_instance.show_deviation_band.isChecked(),
                    "show_detection_threshold": widget_instance.show_detection_threshold.isChecked(),
                    "show_threshold_stats": widget_instance.show_threshold_stats.isChecked(),
                }
            )

            # Handle different multiplier names based on method
            method_text = widget_instance.threshold_method.currentText()
            if "Calibration" in method_text:
                # Use calibration multiplier
                if hasattr(widget_instance, "calibration_multiplier"):
                    hysteresis_kwargs["threshold_multiplier"] = (
                        widget_instance.calibration_multiplier.value()
                    )
            elif "Adaptive" in method_text:
                # Use adaptive multiplier
                if hasattr(widget_instance, "adaptive_base_multiplier"):
                    hysteresis_kwargs["threshold_multiplier"] = (
                        widget_instance.adaptive_base_multiplier.value()
                    )
            else:
                # Use baseline multiplier (default)
                if hasattr(widget_instance, "threshold_multiplier"):
                    hysteresis_kwargs["threshold_multiplier"] = (
                        widget_instance.threshold_multiplier.value()
                    )

        except AttributeError as e:
            print(
                f"Warning: Could not extract all hysteresis parameters from widget: {e}"
            )

    # Override with kwargs
    hysteresis_kwargs.update(kwargs)

    return hysteresis_kwargs


def save_plot(figure: Figure, file_path: str, dpi: int = 100) -> bool:
    """
    Save a matplotlib figure to file.

    Args:
        figure: matplotlib Figure to save
        file_path: Path where to save the file
        dpi: DPI for saving

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        figure.savefig(file_path, dpi=dpi, bbox_inches="tight")
        return True
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
        return False


def save_all_plot_types(
    plot_generator: PlotGenerator,
    data_sets: Dict,
    roi_colors: Dict,
    plot_config: Dict,
    output_directory: str,
    timestamp: str = None,
) -> List[str]:
    """
    Save all available plot types to files.

    Args:
        plot_generator: PlotGenerator instance
        data_sets: Dictionary containing all data types
        roi_colors: ROI color mapping
        plot_config: Plot configuration
        output_directory: Directory to save plots
        timestamp: Optional timestamp for filename

    Returns:
        List of saved file paths
    """
    import os
    import time

    if timestamp is None:
        timestamp = str(int(time.time()))

    saved_files = []

    # Define available plot types and their corresponding data
    plot_types = [
        ("Raw Intensity Changes", "merged_results"),
        ("Movement", "movement_data"),
        ("Fraction Movement", "fraction_data"),
        ("Quiescence", "quiescence_data"),
        ("Sleep", "sleep_data"),
        ("Lighting Conditions (dark IR)", "fraction_data"),  # Uses fraction_data
    ]

    for plot_type, data_key in plot_types:
        if data_key not in data_sets or not data_sets[data_key]:
            print(f"Skipping {plot_type}: no data available")
            continue

        try:
            # Prepare kwargs for special plot types
            kwargs = {}
            if plot_type == "Raw Intensity Changes":
                kwargs = create_hysteresis_kwargs()
            elif plot_type == "Lighting Conditions (dark IR)":
                kwargs = {"bin_minutes": 30}

            # Generate plot
            success = plot_generator.generate_plot(
                plot_type, data_sets[data_key], roi_colors, plot_config, **kwargs
            )

            if success:
                # Save plot
                safe_name = plot_type.replace(" ", "_")
                filename = f"{safe_name}_{timestamp}.png"
                file_path = os.path.join(output_directory, filename)

                if save_plot(
                    plot_generator.figure, file_path, plot_config.get("dpi", 100)
                ):
                    saved_files.append(file_path)
                    print(f"Saved: {filename}")
                else:
                    print(f"Failed to save: {filename}")
            else:
                print(f"Failed to generate plot: {plot_type}")

        except Exception as e:
            print(f"Error generating {plot_type}: {str(e)}")

    return saved_files
