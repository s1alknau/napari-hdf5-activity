"""Format coverage: HDF5, Zarr and the AVI boundary.

Written after a Zarr store turned out to be unloadable through the Input tab:
``load_file`` called the HDF5-only ``detect_hdf5_structure_type``, which probes
a Zarr *directory* with h5py, gets a permission error, returns
``type="error"`` — and the loader aborted. Nothing caught it because the two
files named ``test_zarr_*.py`` in this directory contain no test functions at
all; pytest collects nothing from them.
"""

import os

import h5py
import numpy as np
import pytest

from napari_hdf5_activity._io_abstraction import (
    detect_file_format,
    is_supported_file,
    open_file_reader,
)
from napari_hdf5_activity._reader import (
    detect_file_structure_type,
    detect_hdf5_structure_type,
    get_first_frame_enhanced,
)
from napari_hdf5_activity._metadata import (
    UnsupportedFormatError,
    extract_hdf5_metadata,
    extract_hdf5_metadata_timeseries,
    get_nematostella_timeseries_summary,
)
from napari_hdf5_activity._temperature import extract_environment_from_file

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


N_FRAMES, H, W = 40, 64, 80


def _frames(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (N_FRAMES, H, W), dtype=np.uint16)


@pytest.fixture
def hdf5_recording(tmp_path):
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as f:
        f.attrs["frame_interval"] = 5.0
        _create_array(f, "frames", _frames())
        ts = f.create_group("timeseries")
        _create_array(ts, "temperature_celsius", np.full(N_FRAMES, 20.0, "f4"))
    return str(path)


@pytest.fixture
def zarr_recording(tmp_path):
    path = tmp_path / "rec.zarr"
    root = zarr.open(str(path), mode="w")
    root.attrs["frame_interval"] = 5.0
    _create_array(root, "frames", _frames())
    ts = root.create_group("timeseries")
    _create_array(ts, "temperature_celsius", np.full(N_FRAMES, 20.0, "f4"))
    return str(path)


# --------------------------------------------------------------- access layer

def test_format_detection(hdf5_recording, zarr_recording):
    assert detect_file_format(hdf5_recording) == "hdf5"
    assert detect_file_format(zarr_recording) == "zarr"
    assert is_supported_file(hdf5_recording)
    assert is_supported_file(zarr_recording)


@pytest.mark.parametrize("fixture", ["hdf5_recording", "zarr_recording"])
def test_reader_roundtrip(fixture, request):
    path = request.getfixturevalue(fixture)
    with open_file_reader(path) as reader:
        assert "frames" in reader.keys("/")
        assert reader.shape("frames") == (N_FRAMES, H, W)
        assert reader.read_frame("frames", 0).shape == (H, W)


# ------------------------------------------------------- structure detection

@pytest.mark.parametrize("fixture", ["hdf5_recording", "zarr_recording"])
def test_structure_detection_works_for_both(fixture, request):
    info = detect_file_structure_type(request.getfixturevalue(fixture))
    assert info["type"] == "stacked_frames"
    assert info["frame_count"] == N_FRAMES
    assert tuple(info["frame_shape"]) == (H, W)
    assert info["data_location"] == "frames"
    assert info["dataset_name"] == "frames"


def test_hdf5_only_detector_cannot_read_zarr(zarr_recording):
    """Documents exactly why load_file must not use this variant.

    Kept as a test rather than deleted: the HDF5-only detector is still the
    internal HDF5 branch of detect_file_structure_type, and someone reaching
    for it by name needs to see this.
    """
    assert detect_hdf5_structure_type(zarr_recording)["type"] == "error"


def test_agnostic_detector_is_a_superset_for_hdf5(hdf5_recording):
    """Switching call sites must not change anything for existing HDF5 data."""
    hdf5_only = detect_hdf5_structure_type(hdf5_recording)
    agnostic = detect_file_structure_type(hdf5_recording)
    for key, value in hdf5_only.items():
        assert str(agnostic.get(key)) == str(value), key
    # the agnostic variant adds one field, and only that one
    assert set(agnostic) - set(hdf5_only) == {"file_format"}


@pytest.mark.parametrize("fixture", ["hdf5_recording", "zarr_recording"])
def test_first_frame(fixture, request):
    display, processing, info = get_first_frame_enhanced(
        request.getfixturevalue(fixture)
    )
    assert display.shape == (H, W)
    assert processing.shape == (H, W)
    assert info["frame_count"] == N_FRAMES


# ------------------------------------------------------------- temperature

@pytest.mark.parametrize("fixture", ["hdf5_recording", "zarr_recording"])
def test_temperature_reads_from_both(fixture, request):
    env = extract_environment_from_file(request.getfixturevalue(fixture))
    assert env is not None
    assert len(env["temperature"]) == N_FRAMES
    assert env["temperature_source"] == "temperature_celsius"


# --------------------------------------------------------------- AVI boundary

def test_avi_is_outside_the_abstraction(tmp_path):
    """AVI is handled by its own branch, deliberately not by open_file_reader.

    Video has no group/dataset model, so forcing it into the FileReader
    interface would be contrived. load_file branches to _load_single_avi before
    structure detection ever runs — this test pins that boundary so nobody
    "fixes" it by half-adding AVI to the abstraction.
    """
    avi = tmp_path / "rec.avi"
    avi.write_bytes(b"not really a video")
    assert is_supported_file(str(avi)) is False
    with pytest.raises(ValueError):
        detect_file_format(str(avi))


def test_no_temperature_from_video(tmp_path):
    avi = tmp_path / "rec.avi"
    avi.write_bytes(b"not really a video")
    assert extract_environment_from_file(str(avi)) is None


# ------------------------------------------------------- metadata is HDF5-only

@pytest.mark.parametrize(
    "func",
    [
        extract_hdf5_metadata,
        extract_hdf5_metadata_timeseries,
        get_nematostella_timeseries_summary,
    ],
)
def test_metadata_rejects_zarr_loudly(func, zarr_recording):
    """The failure must be visible, not an empty-but-plausible result.

    These functions read HDF5 storage internals (libver, chunks, compression,
    fletcher32) and are HDF5-specific by design. Before the guard they returned
    zero datasets and zero groups with the real error buried in
    extraction_info["error"] — which nothing outside _metadata.py reads.
    """
    with pytest.raises(UnsupportedFormatError, match="not an HDF5 file"):
        func(zarr_recording)


def test_metadata_still_works_for_hdf5(hdf5_recording):
    metadata = extract_hdf5_metadata(hdf5_recording)
    assert metadata["datasets"], "expected the frames and temperature datasets"
    assert metadata["groups"], "expected the timeseries group"
    assert "frame_interval" in metadata["attributes"]["file_level"]
    assert "error" not in metadata["extraction_info"]


# ----------------------------------------------------- zarr format vs library

from napari_hdf5_activity._io_abstraction import (  # noqa: E402
    ZarrVersionError,
    detect_zarr_format_version,
    zarr_library_major,
)


def test_detects_the_store_format_version(zarr_recording, hdf5_recording):
    """v2 stores carry .zgroup, v3 stores carry zarr.json."""
    version = detect_zarr_format_version(zarr_recording)
    assert version in (2, 3)
    expected = 3 if os.path.exists(os.path.join(zarr_recording, "zarr.json")) else 2
    assert version == expected
    # not a Zarr store at all
    assert detect_zarr_format_version(hdf5_recording) is None


def test_library_major_matches_the_installed_zarr():
    assert zarr_library_major() == int(zarr.__version__.split(".")[0])


@pytest.mark.skipif(
    zarr_library_major() is not None and zarr_library_major() >= 3,
    reason="zarr-python 3 reads both formats, so there is nothing to refuse",
)
def test_v3_store_with_zarr2_fails_with_an_actionable_message(tmp_path):
    """A bare 'nothing found at path' tells the user nothing.

    zarr-python 2 reads format v2 only. All recordings from the current rig are
    v3, so this is the error users actually hit.
    """
    store = tmp_path / "v3.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')

    assert detect_zarr_format_version(str(store)) == 3
    with pytest.raises(ZarrVersionError, match="zarr>=3"):
        open_file_reader(str(store)).open()
