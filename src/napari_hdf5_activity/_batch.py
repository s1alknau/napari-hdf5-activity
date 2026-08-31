"""
_batch.py - Multi-dataset (batch) support for the extended analysis

Pools the ROIs of several saved analysis HDF5 files into one population so the
extended analysis (periodograms, cosinor, coherence, similarity, population
mean) runs on n = ROIs x datasets instead of a single recording.

ROI identity
------------
Every ROI-keyed dictionary in this plugin is ``Dict[int, ...]``.  To keep that
contract intact the batch loader re-keys the ROIs of the additional datasets
with a *composite* integer id::

    dataset 1  ->  1, 2, 3, ...            (unchanged)
    dataset 2  ->  2001, 2002, 2003, ...
    dataset 3  ->  3001, 3002, 3003, ...

Dataset 1 is deliberately left untouched, so a single-dataset run produces
byte-identical results and labels to before.  Only two things ever interpret
the integer: the label (``roi_label``) and the colour (``base_roi_id``).

Time alignment
--------------
Each additional dataset is aligned by one of two modes:

``own_start``
    ZT0 is that recording's own first sample.  Correct when every recording
    was started at the same point in the light cycle (the usual protocol),
    which is what makes pooled cosinor acrophases comparable.

``relative``
    The dataset is shifted by an explicit offset relative to dataset 1's ZT0.
    For recordings that started at a different light-cycle phase.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROI_STRIDE = 1000

ZT_MODE_OWN_START = "own_start"
ZT_MODE_RELATIVE = "relative"

# Core-analysis keys that map roi_id -> [(time_seconds, value), ...]
TIMESERIES_KEYS = (
    "merged_results",
    "merged_results_raw",
    "movement_data",
    "fraction_data",
    "quiescence_data",
    "sleep_data",
)

# Core-analysis keys that map roi_id -> anything (no time column)
ROI_KEYED_KEYS = (
    "roi_colors",
    "roi_statistics",
    "roi_summary",
    "thresholds",
)


# ---------------------------------------------------------------------------
# Composite ROI ids
# ---------------------------------------------------------------------------

def make_composite_id(dataset_idx: int, roi_id: int) -> int:
    """Build the pooled ROI key for *roi_id* of dataset *dataset_idx* (1-based).

    Dataset 1 keeps its plain ROI ids so single-dataset behaviour is unchanged.
    """
    if dataset_idx <= 1:
        return int(roi_id)
    return int(dataset_idx) * ROI_STRIDE + int(roi_id)


def split_composite_id(composite_id: int) -> Tuple[int, int]:
    """Return ``(dataset_idx, roi_id)`` for a pooled ROI key."""
    cid = int(composite_id)
    if cid < ROI_STRIDE:
        return 1, cid
    return cid // ROI_STRIDE, cid % ROI_STRIDE


def dataset_index(composite_id: int) -> int:
    """Return the 1-based dataset index a pooled ROI key belongs to."""
    return split_composite_id(composite_id)[0]


def base_roi_id(composite_id: int) -> int:
    """Return the within-dataset ROI number of a pooled ROI key.

    This is what colour assignment must key on so that ROI 1 of every dataset
    shares one colour.
    """
    return split_composite_id(composite_id)[1]


def is_composite(composite_id: int) -> bool:
    """True when the key carries a dataset index (i.e. dataset 2 or higher)."""
    return int(composite_id) >= ROI_STRIDE


def roi_label(composite_id: int, prefix: str = "ROI") -> str:
    """Human-readable ROI name.

    Dataset 1 keeps the existing ``"ROI 1"`` style; additional datasets are
    suffixed with their dataset number, e.g. ``"ROI1_2"`` for ROI 1 of the
    second dataset.
    """
    ds_idx, roi = split_composite_id(composite_id)
    if ds_idx <= 1:
        return f"{prefix} {roi}"
    return f"{prefix}{roi}_{ds_idx}"


def roi_short_label(composite_id: int) -> str:
    """Compact form without the ``ROI`` prefix: ``"1"`` or ``"1_2"``."""
    ds_idx, roi = split_composite_id(composite_id)
    return str(roi) if ds_idx <= 1 else f"{roi}_{ds_idx}"


def dataset_linestyle(composite_id: int) -> str:
    """Matplotlib linestyle per dataset, so same-colour traces stay distinct."""
    styles = ("-", "--", ":", "-.")
    return styles[(dataset_index(composite_id) - 1) % len(styles)]


def sort_pooled_ids(ids: Sequence[int]) -> List[int]:
    """Sort pooled ROI keys ROI-major, then by dataset.

    ROI1, ROI1_2, ROI1_3, ROI2, ROI2_2, ... keeps the same colour adjacent in
    legends and summary tables.
    """
    return sorted(ids, key=lambda cid: (base_roi_id(cid), dataset_index(cid)))


# ---------------------------------------------------------------------------
# Batch specification / result containers
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """One entry of a batch: a saved results file plus its time alignment."""

    file_path: str
    zt_mode: str = ZT_MODE_OWN_START
    zt_offset_hours: float = 0.0
    label: str = ""

    def offset_seconds(self) -> float:
        """Time shift applied after each recording is zeroed on its own start."""
        if self.zt_mode == ZT_MODE_RELATIVE:
            return float(self.zt_offset_hours) * 3600.0
        return 0.0


@dataclass
class BatchResult:
    """Pooled core-analysis data plus provenance for every pooled ROI."""

    core: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # {dataset_idx: {"times": [...], "temperature": [...]}} on the pooled time
    # base — each ROI is tested against its own dataset's temperature record.
    environment: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def n_datasets(self) -> int:
        return len(self.datasets)

    @property
    def n_rois(self) -> int:
        return len(self.provenance)

    def summary_lines(self) -> List[str]:
        """Short human-readable description of the batch composition."""
        lines = [f"Batch: {self.n_datasets} dataset(s), {self.n_rois} ROIs pooled"]
        for info in self.datasets:
            mode = (
                "ZT0 = own start"
                if info["zt_mode"] == ZT_MODE_OWN_START
                else f"ZT{info['zt_offset_hours']:+.2f} h rel. to dataset 1"
            )
            lines.append(
                f"  [{info['dataset_idx']}] {info['name']} — "
                f"{info['n_rois']} ROIs, {mode}, "
                f"{info['duration_hours']:.1f} h"
            )
        return lines


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

def _series_start(series: Sequence[Tuple[float, Any]]) -> Optional[float]:
    return float(series[0][0]) if series else None


def find_dataset_t0(core: Dict[str, Any]) -> float:
    """Earliest timestamp across all ROI timeseries of one loaded dataset."""
    starts = []
    for key in TIMESERIES_KEYS:
        data = core.get(key) or {}
        for series in data.values():
            t0 = _series_start(series)
            if t0 is not None:
                starts.append(t0)
    return min(starts) if starts else 0.0


def shift_series(
    series: Sequence[Tuple[float, Any]], shift_seconds: float
) -> List[Tuple[float, Any]]:
    """Return *series* with ``shift_seconds`` added to every timestamp."""
    return [(float(t) + shift_seconds, v) for t, v in series]


def dataset_duration_hours(core: Dict[str, Any]) -> float:
    """Span of the longest ROI timeseries in a loaded dataset, in hours."""
    spans = []
    for key in TIMESERIES_KEYS:
        data = core.get(key) or {}
        for series in data.values():
            if len(series) >= 2:
                spans.append(float(series[-1][0]) - float(series[0][0]))
    return (max(spans) / 3600.0) if spans else 0.0


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------

_COMPAT_PARAMS = (
    ("frame_interval", "frame interval"),
    ("bin_size_seconds", "bin size"),
)


def check_compatibility(loaded: List[Dict[str, Any]]) -> List[str]:
    """Compare acquisition parameters across loaded datasets.

    Returns a list of human-readable warnings; empty when everything matches.
    Pooling is never blocked here — the caller decides what to do.
    """
    warnings: List[str] = []
    if len(loaded) < 2:
        return warnings

    for key, nice_name in _COMPAT_PARAMS:
        values = []
        for entry in loaded:
            core_params = (entry.get("analysis_parameters") or {}).get("core") or {}
            values.append(core_params.get(key))
        known = [v for v in values if v is not None]
        if len(set(float(v) for v in known)) > 1:
            detail = ", ".join(
                f"[{i + 1}] {v if v is not None else '?'}" for i, v in enumerate(values)
            )
            warnings.append(
                f"Differing {nice_name} across datasets ({detail}) — "
                f"periodogram frequencies will not line up exactly."
            )

    roi_counts = [
        len((entry.get("core_analysis") or {}).get("fraction_data") or {})
        for entry in loaded
    ]
    if len(set(roi_counts)) > 1:
        warnings.append(
            f"Datasets contain different ROI counts ({roi_counts}) — "
            f"pooled statistics will be unbalanced across datasets."
        )

    return warnings


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def _rekey_timeseries(
    data: Dict[int, Any], dataset_idx: int, shift_seconds: float
) -> Dict[int, Any]:
    return {
        make_composite_id(dataset_idx, roi_id): shift_series(series, shift_seconds)
        for roi_id, series in data.items()
    }


def _rekey_plain(data: Dict[int, Any], dataset_idx: int) -> Dict[int, Any]:
    return {
        make_composite_id(dataset_idx, roi_id): value
        for roi_id, value in data.items()
    }


def _rekey_bouts(
    data: Dict[int, Any], dataset_idx: int, shift_seconds: float
) -> Dict[int, Any]:
    out = {}
    for roi_id, bouts in data.items():
        shifted = []
        for bout in bouts:
            bout = dict(bout)
            for key in ("start_time", "end_time"):
                if key in bout:
                    bout[key] = float(bout[key]) + shift_seconds
            shifted.append(bout)
        out[make_composite_id(dataset_idx, roi_id)] = shifted
    return out


def load_batch_results(
    specs: Sequence[DatasetSpec], log=None
) -> BatchResult:
    """Load and pool several saved analysis HDF5 files.

    Args:
        specs: One :class:`DatasetSpec` per file.  The first entry is the main
            dataset; its ROI ids and time base are left untouched and every
            other dataset is aligned relative to it.
        log: Optional ``callable(str)`` for progress messages.

    Returns:
        A :class:`BatchResult` whose ``core`` dict has exactly the shape of a
        single file's ``core_analysis`` group, so it can be assigned straight
        onto the widget.
    """
    import os

    from ._results_io import load_comprehensive_results

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    if not specs:
        return BatchResult()

    loaded: List[Dict[str, Any]] = []
    for spec in specs:
        _log(f"  Loading {os.path.basename(spec.file_path)} ...")
        entry = load_comprehensive_results(spec.file_path)
        if not entry or not entry.get("core_analysis"):
            raise ValueError(
                f"No core analysis results found in {os.path.basename(spec.file_path)}"
            )
        loaded.append(entry)

    result = BatchResult()
    result.warnings = check_compatibility(loaded)

    # Parameters and metadata come from the main dataset — the extended
    # analysis is run with one set of settings for the whole batch.
    result.parameters = loaded[0].get("analysis_parameters", {}) or {}
    result.metadata = loaded[0].get("metadata", {}) or {}

    pooled: Dict[str, Dict[int, Any]] = {}

    for idx, (spec, entry) in enumerate(zip(specs, loaded), start=1):
        core = entry.get("core_analysis") or {}
        t0 = find_dataset_t0(core)
        # Dataset 1 defines the time base and is never shifted.
        shift = 0.0 if idx == 1 else spec.offset_seconds() - t0

        for key in TIMESERIES_KEYS:
            data = core.get(key)
            if data:
                pooled.setdefault(key, {}).update(
                    _rekey_timeseries(data, idx, shift)
                )

        for key in ROI_KEYED_KEYS:
            data = core.get(key)
            if data:
                pooled.setdefault(key, {}).update(_rekey_plain(data, idx))

        bouts = core.get("quiescence_bouts")
        if bouts:
            pooled.setdefault("quiescence_bouts", {}).update(
                _rekey_bouts(bouts, idx, shift)
            )

        # Temperature stays per-dataset (it is a property of the recording, not
        # of an ROI) but must ride the same time shift as the activity data.
        env = core.get("environment_data")
        if env and env.get("temperature"):
            shifted_env = dict(env)
            shifted_env["times"] = [
                float(t) + shift for t in (env.get("times") or [])
            ]
            result.environment[idx] = shifted_env

        roi_ids = sorted((core.get("fraction_data") or core.get("merged_results") or {}).keys())
        name = spec.label or os.path.splitext(os.path.basename(spec.file_path))[0]
        result.datasets.append(
            {
                "dataset_idx": idx,
                "name": name,
                "file_path": spec.file_path,
                "zt_mode": spec.zt_mode if idx > 1 else ZT_MODE_OWN_START,
                "zt_offset_hours": 0.0 if idx == 1 else spec.zt_offset_hours,
                "shift_seconds": shift,
                "n_rois": len(roi_ids),
                "duration_hours": dataset_duration_hours(core),
                "roi_ids": roi_ids,
            }
        )

        for roi_id in roi_ids:
            cid = make_composite_id(idx, roi_id)
            result.provenance[cid] = {
                "dataset_idx": idx,
                "base_roi_id": roi_id,
                "source_file": spec.file_path,
                "dataset_name": name,
                "label": roi_label(cid),
                "shift_seconds": shift,
            }

        _log(
            f"    ✓ dataset {idx}: {len(roi_ids)} ROIs, "
            f"{dataset_duration_hours(core):.1f} h, shift {shift / 3600.0:+.2f} h"
        )

    # Per-dataset payloads that are not ROI-keyed: keep the main dataset's, so
    # the napari overlay and lighting bands still refer to something real.
    # (Temperature is the exception — it is kept per dataset in .environment.)
    main_core = loaded[0].get("core_analysis") or {}
    for key in ("led_data", "masks", "original_circles"):
        if main_core.get(key) is not None:
            pooled[key] = main_core[key]

    result.core = pooled
    return result


# ---------------------------------------------------------------------------
# Pairwise methods on pooled data
# ---------------------------------------------------------------------------

def spans_multiple_datasets(roi_ids: Sequence[int]) -> bool:
    """True when the ROI keys come from more than one dataset."""
    return len({dataset_index(cid) for cid in roi_ids}) > 1


def build_common_grid(
    movement_data: Dict[int, Sequence[Tuple[float, float]]],
    bin_size_seconds: float,
) -> Tuple[Any, Dict[int, Any]]:
    """Resample every ROI onto one shared absolute time grid.

    Pairwise methods (correlation, coherence) align their two inputs by sample
    index.  For a single recording that is the same thing as aligning by time,
    but pooled datasets may sit at different ZT offsets, so the signals have to
    be put on a common grid first.  Samples outside an ROI's own recording are
    ``NaN`` and are dropped per pair by :func:`overlapping_pair`.

    Returns:
        ``(grid_seconds, {roi_id: values_with_nan})``
    """
    import numpy as np

    starts, ends = [], []
    for series in movement_data.values():
        if series is not None and len(series) >= 2:
            starts.append(float(series[0][0]))
            ends.append(float(series[-1][0]))
    if not starts:
        return np.array([]), {}

    step = float(bin_size_seconds) if bin_size_seconds else 60.0
    grid = np.arange(min(starts), max(ends) + step, step)

    signals: Dict[int, Any] = {}
    for roi_id, series in movement_data.items():
        if series is None or len(series) < 2:
            continue
        times = np.array([t for t, _ in series], dtype=float)
        values = np.array([v for _, v in series], dtype=float)
        order = np.argsort(times)
        signals[roi_id] = np.interp(
            grid, times[order], values[order], left=np.nan, right=np.nan
        )
    return grid, signals


def overlapping_pair(signal1, signal2, min_samples: int = 10):
    """Restrict two grid-aligned signals to the samples both actually cover.

    Returns ``(a, b, n_overlap)``; ``a``/``b`` are ``None`` when the overlap is
    shorter than *min_samples*.
    """
    import numpy as np

    mask = np.isfinite(signal1) & np.isfinite(signal2)
    n_overlap = int(np.sum(mask))
    if n_overlap < min_samples:
        return None, None, n_overlap
    return signal1[mask], signal2[mask], n_overlap


def provenance_rows(result: BatchResult) -> List[Dict[str, Any]]:
    """Flat per-ROI provenance table for CSV export."""
    rows = []
    for cid in sort_pooled_ids(result.provenance.keys()):
        info = result.provenance[cid]
        rows.append(
            {
                "roi_label": info["label"],
                "roi_id": info["base_roi_id"],
                "dataset_index": info["dataset_idx"],
                "dataset_name": info["dataset_name"],
                "source_file": info["source_file"],
                "zt_shift_hours": info["shift_seconds"] / 3600.0,
            }
        )
    return rows
