"""Tests for multi-dataset (batch) pooling in the extended analysis."""

import numpy as np
import pytest

from napari_hdf5_activity import _batch as B
from napari_hdf5_activity._reader import get_roi_colors

BIN = 600.0                      # 10 min bins
N = int(96 * 3600 / BIN)         # 96 h recording


def synth(period_h=24.0, phase=0.0, t0=0.0, n=N, noise=0.25, seed=0):
    """A noisy cosine activity trace as [(time_seconds, value), ...]."""
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n) * BIN
    v = 0.5 + 0.3 * np.cos(2 * np.pi * (t - t0) / (period_h * 3600) + phase)
    v = v + rng.normal(0, noise, n)
    return [(float(a), float(b)) for a, b in zip(t, v)]


def pooled_data(n_roi=4, t0_second=0.0, n_second=N):
    data = {}
    for r in range(1, n_roi + 1):
        data[B.make_composite_id(1, r)] = synth(phase=0.1 * r, seed=r)
        data[B.make_composite_id(2, r)] = synth(
            phase=0.1 * r, t0=t0_second, n=n_second, seed=100 + r
        )
    return data


# ---------------------------------------------------------------- composite ids

@pytest.mark.parametrize(
    "dataset_idx, roi_id, expected",
    [(1, 1, 1), (1, 12, 12), (2, 1, 2001), (3, 12, 3012)],
)
def test_make_composite_id(dataset_idx, roi_id, expected):
    assert B.make_composite_id(dataset_idx, roi_id) == expected


@pytest.mark.parametrize("dataset_idx, roi_id", [(1, 1), (1, 999), (2, 1), (7, 42)])
def test_composite_id_roundtrip(dataset_idx, roi_id):
    cid = B.make_composite_id(dataset_idx, roi_id)
    assert B.split_composite_id(cid) == (dataset_idx, roi_id)
    assert B.dataset_index(cid) == dataset_idx
    assert B.base_roi_id(cid) == roi_id


@pytest.mark.parametrize(
    "cid, label",
    [(1, "ROI 1"), (12, "ROI 12"), (2001, "ROI1_2"), (3012, "ROI12_3")],
)
def test_roi_label(cid, label):
    assert B.roi_label(cid) == label


def test_dataset_one_labels_are_unchanged():
    """Single-dataset output must be byte-identical to the pre-batch behaviour."""
    assert [B.roi_label(r) for r in range(1, 5)] == [
        "ROI 1", "ROI 2", "ROI 3", "ROI 4",
    ]


def test_sort_pooled_ids_is_roi_major():
    assert B.sort_pooled_ids([2002, 1, 2001, 2]) == [1, 2001, 2, 2002]


# -------------------------------------------------------------------- colours

def test_colour_follows_roi_number_not_dataset():
    cols = get_roi_colors([1, 2, 3, 2001, 2002, 2003, 3001])
    assert cols[1] == cols[2001] == cols[3001]
    assert cols[2] == cols[2002]
    assert cols[3] == cols[2003]
    assert cols[1] != cols[2]


def test_dataset_linestyle_differs_per_dataset():
    assert B.dataset_linestyle(1) != B.dataset_linestyle(2001)
    assert B.dataset_linestyle(1) == B.dataset_linestyle(2)


# ------------------------------------------------------------- time alignment

def make_entry(t0, n_roi, n_pts=100, dt=60.0, bin_size=60):
    series = {
        r: [(t0 + i * dt, 0.1 * r + i * 0.001) for i in range(n_pts)]
        for r in range(1, n_roi + 1)
    }
    return {
        "core_analysis": {
            "merged_results": {r: list(v) for r, v in series.items()},
            "fraction_data": series,
            "quiescence_bouts": {
                r: [{"start_time": t0 + 100.0, "end_time": t0 + 200.0,
                     "duration": 100.0, "mean_movement": 0.1}]
                for r in range(1, n_roi + 1)
            },
            "led_data": {"times": [t0]},
        },
        "analysis_parameters": {
            "core": {"frame_interval": 5.0, "bin_size_seconds": bin_size}
        },
        "metadata": {},
    }


@pytest.fixture
def stub_loader(monkeypatch):
    """Serve canned results dicts instead of reading real HDF5 files."""
    files = {}

    def install(mapping):
        files.clear()
        files.update(mapping)
        monkeypatch.setattr(
            "napari_hdf5_activity._results_io.load_comprehensive_results",
            lambda p: files[p],
        )

    return install


def test_own_start_mode_zeroes_every_dataset(stub_loader):
    stub_loader({"a": make_entry(0.0, 3), "b": make_entry(98000.0, 3)})
    res = B.load_batch_results([B.DatasetSpec("a"), B.DatasetSpec("b")])

    assert res.n_datasets == 2
    assert res.n_rois == 6
    assert sorted(res.core["fraction_data"]) == [1, 2, 3, 2001, 2002, 2003]
    assert res.core["fraction_data"][1][0][0] == 0.0
    assert res.core["fraction_data"][2001][0][0] == 0.0
    assert res.core["quiescence_bouts"][2001][0]["start_time"] == 100.0


def test_relative_mode_shifts_to_requested_zt(stub_loader):
    stub_loader({"a": make_entry(0.0, 3), "b": make_entry(98000.0, 3)})
    res = B.load_batch_results([
        B.DatasetSpec("a"),
        B.DatasetSpec("b", zt_mode=B.ZT_MODE_RELATIVE, zt_offset_hours=8.0),
    ])

    assert res.core["fraction_data"][1][0][0] == 0.0
    assert res.core["fraction_data"][2001][0][0] == 8 * 3600.0
    assert res.core["quiescence_bouts"][2001][0]["start_time"] == 8 * 3600.0 + 100.0


def test_main_dataset_is_never_shifted(stub_loader):
    """Row 1 defines the time base even if it is marked relative."""
    stub_loader({"a": make_entry(5000.0, 2), "b": make_entry(0.0, 2)})
    res = B.load_batch_results([
        B.DatasetSpec("a", zt_mode=B.ZT_MODE_RELATIVE, zt_offset_hours=99.0),
        B.DatasetSpec("b"),
    ])
    assert res.core["fraction_data"][1][0][0] == 5000.0
    assert res.datasets[0]["zt_offset_hours"] == 0.0


def test_single_dataset_keeps_plain_roi_ids(stub_loader):
    stub_loader({"a": make_entry(0.0, 3)})
    res = B.load_batch_results([B.DatasetSpec("a")])
    assert sorted(res.core["fraction_data"]) == [1, 2, 3]


def test_incompatible_bin_size_warns(stub_loader):
    stub_loader({"a": make_entry(0.0, 3), "c": make_entry(0.0, 2, bin_size=120)})
    res = B.load_batch_results([B.DatasetSpec("a"), B.DatasetSpec("c")])
    assert any("bin size" in w for w in res.warnings)
    assert any("ROI counts" in w for w in res.warnings)


def test_provenance_rows_map_back_to_source(stub_loader):
    stub_loader({"a": make_entry(0.0, 2), "b": make_entry(0.0, 2)})
    res = B.load_batch_results([B.DatasetSpec("a"), B.DatasetSpec("b")])
    rows = B.provenance_rows(res)
    assert rows[0]["roi_label"] == "ROI 1"
    assert rows[1]["roi_label"] == "ROI1_2"
    assert rows[1]["source_file"] == "b"
    assert rows[1]["dataset_index"] == 2


# ----------------------------------------------------------- pooled analysis

def test_fft_runs_on_pooled_rois():
    from napari_hdf5_activity._circadian_fft import (
        analyze_roi_fft_patterns, generate_fft_summary)

    data = pooled_data()
    results = analyze_roi_fft_patterns(
        data, sampling_interval=BIN, min_period_hours=18, max_period_hours=30,
        bin_size_seconds=int(BIN), n_permutations=20,
    )
    assert len(results) == 8
    periods = [v["dominant_period"] for v in results.values() if "error" not in v]
    assert all(20 < p < 28 for p in periods)
    assert "ROI1_2" in generate_fft_summary(results)


def test_population_cosinor_uses_full_pooled_n():
    from napari_hdf5_activity._cosinor_analysis import population_cosinor

    data = pooled_data()
    series = [np.array([v for _, v in data[k]]) for k in sorted(data)]
    result = population_cosinor(series, period_hours=24.0, sampling_interval=BIN)
    assert result["n_individuals"] == 8
    assert result["n_significant"] == 8


def test_similarity_pools_and_flags_pooled_mode():
    from napari_hdf5_activity._circadian_similarity import (
        calculate_roi_correlation_matrix, generate_similarity_summary)

    result = calculate_roi_correlation_matrix(
        pooled_data(), sampling_interval=BIN, bin_size_seconds=int(BIN),
        max_lag_hours=12, significance_level=0.05,
    )
    assert result["pooled_datasets"] is True
    assert result["n_rois"] == 8
    assert result["n_pairs_skipped_no_overlap"] == 0
    assert np.isfinite(result["correlation_matrix"]).all()
    assert "Pooled datasets" in generate_similarity_summary(result, threshold=0.3)


def test_similarity_skips_pairs_without_enough_overlap():
    """Datasets far apart in ZT must not yield a correlation from a few points."""
    from napari_hdf5_activity._circadian_similarity import (
        calculate_roi_correlation_matrix)

    result = calculate_roi_correlation_matrix(
        pooled_data(t0_second=90 * 3600.0), sampling_interval=BIN,
        bin_size_seconds=int(BIN), max_lag_hours=12, significance_level=0.05,
    )
    assert result["n_pairs_skipped_no_overlap"] > 0
    # within-dataset pairs are untouched by the guard
    assert np.isfinite(result["correlation_matrix"][0, 1])


def test_coherence_skips_pairs_shorter_than_two_periods():
    from napari_hdf5_activity._circadian_coherence import calculate_coherence_matrix

    full = calculate_coherence_matrix(
        pooled_data(), sampling_interval=BIN, bin_size_seconds=int(BIN),
        target_period_hours=24.0,
    )
    assert full["pooled_datasets"] is True
    assert full["n_pairs_skipped_no_overlap"] == 0

    sparse = calculate_coherence_matrix(
        pooled_data(t0_second=90 * 3600.0), sampling_interval=BIN,
        bin_size_seconds=int(BIN), target_period_hours=24.0,
    )
    assert sparse["n_pairs_skipped_no_overlap"] > 0


def test_single_dataset_analysis_path_is_untouched():
    from napari_hdf5_activity._circadian_similarity import (
        calculate_roi_correlation_matrix, generate_similarity_summary)

    single = {r: synth(phase=0.1 * r, seed=r) for r in range(1, 5)}
    result = calculate_roi_correlation_matrix(
        single, sampling_interval=BIN, bin_size_seconds=int(BIN), max_lag_hours=12,
    )
    assert result["pooled_datasets"] is False
    summary = generate_similarity_summary(result, threshold=0.3)
    assert "Pooled datasets" not in summary
    assert "ROI 1" in summary


def test_cross_correlation_survives_lag_window_covering_half_the_signal():
    """Regression: the lag array and the correlation window must match length."""
    from napari_hdf5_activity._circadian_similarity import calculate_cross_correlation

    a = np.sin(np.arange(36) * 0.3)
    b = np.cos(np.arange(36) * 0.3)
    result = calculate_cross_correlation(
        a, b, max_lag_hours=12, sampling_interval=BIN
    )
    assert "error" not in result
    assert len(result["correlation"]) == len(result["lags"])


# --------------------------------------------------------- population plotting

def test_population_mean_marks_varying_n():
    from matplotlib.figure import Figure
    from napari_hdf5_activity._plot import PlotGenerator

    # dataset 2 starts 24 h in and is shorter, so n varies across the grid
    data = pooled_data(t0_second=24 * 3600.0, n_second=300)
    fig = Figure()
    assert PlotGenerator(fig)._plot_population_mean(
        data, get_roi_colors(sorted(data)),
        {"start_time": 0, "end_time": 1e9}, y_label="Fraction",
    )
    assert len(fig.get_axes()) == 2, "expected a twin axis carrying n(t)"
    labels = [ln.get_label() for ln in fig.get_axes()[0].get_lines()]
    assert any("n=4–8" in lb for lb in labels), labels


def test_population_mean_omits_n_axis_when_n_is_constant():
    from matplotlib.figure import Figure
    from napari_hdf5_activity._plot import PlotGenerator

    data = pooled_data()
    fig = Figure()
    PlotGenerator(fig)._plot_population_mean(
        data, get_roi_colors(sorted(data)),
        {"start_time": 0, "end_time": 1e9}, y_label="Fraction",
    )
    assert len(fig.get_axes()) == 1


def test_population_mean_x_axis_is_hours():
    from matplotlib.figure import Figure
    from napari_hdf5_activity._plot import PlotGenerator

    fig = Figure()
    data = {1: synth(seed=1), 2: synth(seed=2)}
    PlotGenerator(fig)._plot_population_mean(
        data, get_roi_colors([1, 2]),
        {"start_time": 0, "end_time": 1e9}, y_label="Fraction",
    )
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Time (h)"
    assert 90 < ax.get_xlim()[1] < 110, ax.get_xlim()
