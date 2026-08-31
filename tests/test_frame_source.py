"""Tests for FrameSource — the sequential frame reader behind the frame viewer."""

import h5py
import numpy as np
import pytest

from napari_hdf5_activity._frame_source import (
    LAYOUT_INDIVIDUAL,
    LAYOUT_STACKED,
    FrameSource,
    FrameSourceError,
)

zarr = pytest.importorskip("zarr", reason="zarr is an optional extra")


def _create_array(group, name, data):
    """Create an array in an h5py or zarr group, whichever generation is installed.

    zarr-python 3 renamed ``create_dataset`` to ``create_array``; h5py keeps
    ``create_dataset``. This keeps the fixtures working under both.
    """
    if hasattr(group, "create_array"):
        arr = group.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[:] = data
        return arr
    return group.create_dataset(name, data=data)


N, H, W = 12, 16, 20


def _stack(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (N, H, W), dtype=np.uint16)


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def h5_stacked(tmp_path):
    path = tmp_path / "stacked.h5"
    data = _stack()
    with h5py.File(path, "w") as f:
        _create_array(f, "frames", data)
    return str(path), data


@pytest.fixture
def zarr_stacked(tmp_path):
    path = tmp_path / "stacked.zarr"
    data = _stack(1)
    root = zarr.open(str(path), mode="w")
    _create_array(root, "frames", data)
    return str(path), data


@pytest.fixture
def h5_individual(tmp_path):
    path = tmp_path / "individual.h5"
    data = _stack(2)
    with h5py.File(path, "w") as f:
        g = f.create_group("images")
        for i in range(N):
            g.create_dataset(f"frame_{i:06d}", data=data[i])
    return str(path), data


@pytest.fixture
def h5_nested(tmp_path):
    """A group holding a stacked 'frames' array — the probe's case (b)."""
    path = tmp_path / "nested.h5"
    data = _stack(3)
    with h5py.File(path, "w") as f:
        f.create_group("images").create_dataset("frames", data=data)
    return str(path), data


ALL = ["h5_stacked", "zarr_stacked", "h5_individual", "h5_nested"]


# -------------------------------------------------------------------- basics

@pytest.mark.parametrize("fixture", ALL)
def test_reads_every_frame_correctly(fixture, request):
    path, data = request.getfixturevalue(fixture)
    with FrameSource(path) as src:
        assert src.n_frames == N
        for i in range(N):
            np.testing.assert_array_equal(src.read_frame(i), data[i])


@pytest.mark.parametrize("fixture", ALL)
def test_reports_layout_and_shape(fixture, request):
    path, _ = request.getfixturevalue(fixture)
    with FrameSource(path) as src:
        assert src.layout in (LAYOUT_STACKED, LAYOUT_INDIVIDUAL)
        assert tuple(src.frame_shape) == (H, W)
        assert src.dataset_name
        assert src.describe()


def test_individual_layout_is_recognised(h5_individual):
    path, _ = h5_individual
    with FrameSource(path) as src:
        assert src.layout == LAYOUT_INDIVIDUAL


def test_random_access_is_not_sequential_dependent(h5_stacked):
    """Scrubbing jumps around; order must not matter."""
    path, data = h5_stacked
    with FrameSource(path) as src:
        for i in (N - 1, 0, N // 2, 3, N - 2):
            np.testing.assert_array_equal(src.read_frame(i), data[i])


# ------------------------------------------------------------------ lifetime

def test_close_is_idempotent_and_releases(h5_stacked):
    path, _ = h5_stacked
    src = FrameSource(path).open()
    assert src.is_open
    src.close()
    assert not src.is_open
    src.close()  # must not raise


def test_reading_after_close_is_refused(h5_stacked):
    path, _ = h5_stacked
    src = FrameSource(path).open()
    src.close()
    with pytest.raises(FrameSourceError):
        src.read_frame(0)


def test_context_manager_closes_on_exception(h5_stacked):
    path, _ = h5_stacked
    src = FrameSource(path)
    with pytest.raises(ZeroDivisionError):
        with src:
            1 / 0
    assert not src.is_open


def test_file_is_actually_released(tmp_path):
    """The old code only set the handle to None — the file stayed open."""
    path = tmp_path / "lock.h5"
    with h5py.File(path, "w") as f:
        _create_array(f, "frames", _stack(4))

    src = FrameSource(str(path)).open()
    src.read_frame(0)
    src.close()
    # If the handle were still open, h5py would refuse write access here.
    with h5py.File(path, "a") as f:
        f.attrs["written_after_close"] = True


# -------------------------------------------------------------------- errors

def test_out_of_range_index(h5_stacked):
    path, _ = h5_stacked
    with FrameSource(path) as src:
        for bad in (-1, N, N + 5):
            with pytest.raises(IndexError):
                src.read_frame(bad)


def test_file_without_images_is_rejected(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        _create_array(f, "temperature", np.arange(10.0))
    with pytest.raises(FrameSourceError):
        FrameSource(str(path)).open()


def test_failed_open_does_not_leak_a_handle(tmp_path):
    """A rejected file must still be writable afterwards."""
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        _create_array(f, "temperature", np.arange(10.0))
    with pytest.raises(FrameSourceError):
        FrameSource(str(path)).open()
    with h5py.File(path, "a") as f:
        f.attrs["still_writable"] = True


# ------------------------------------------------- integration with the widget

@pytest.fixture
def widget(qapp):
    """The real analysis widget, built offscreen with a stand-in viewer.

    Depends on pytest-qt's ``qapp`` so the QApplication lifetime is managed by
    the plugin — building it by hand inside a test crashes the interpreter.
    """
    from unittest.mock import MagicMock

    from napari_hdf5_activity._widget import HDF5AnalysisWidget

    w = HDF5AnalysisWidget(MagicMock())
    w._log_message = lambda *_: None
    w._viewer_preload_frames = lambda: None   # no cache dialog in tests
    w._viewer_show_frame = lambda _i: None    # no Qt painting
    return w


@pytest.mark.parametrize("fixture", ["h5_stacked", "zarr_stacked", "h5_individual"])
def test_frame_viewer_loads_and_reads(fixture, request, widget):
    """Every layout must survive the widget's own load path."""
    path, data = request.getfixturevalue(fixture)
    widget.file_path = path
    widget._viewer_load_hdf5()

    assert widget.viewer_n_frames == N
    assert widget.viewer_dataset_name
    for i in (0, N // 2, N - 1):
        np.testing.assert_array_equal(widget._viewer_read_raw_frame(i), data[i])


def test_frame_viewer_releases_the_file(h5_stacked, widget, tmp_path):
    """Regression: the handle used to be dropped, never closed."""
    path, _ = h5_stacked
    widget.file_path = path
    widget._viewer_load_hdf5()
    source = widget.viewer_file_handle
    widget._viewer_read_raw_frame(0)

    widget._viewer_close_source()
    assert widget.viewer_file_handle is None
    assert not source.is_open
    with h5py.File(path, "a") as f:
        f.attrs["written_after_close"] = True


def test_switching_to_avi_closes_the_previous_file(h5_stacked, widget):
    """Leaving HDF5 for an AVI batch must not strand the open handle."""
    path, _ = h5_stacked
    widget.file_path = path
    widget._viewer_load_hdf5()
    source = widget.viewer_file_handle
    assert source.is_open

    widget._viewer_close_source()
    assert not source.is_open
