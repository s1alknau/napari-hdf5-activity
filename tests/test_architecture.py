"""Enforces the layering: only the access layer opens files.

Dependencies run one way — UI calls analysis calls domain readers calls the
access layer — and only the bottom layer touches h5py, zarr or cv2 directly.
Without a test the rule erodes silently: that is exactly how a Zarr store came
to be unloadable through the Input tab, and how a 130 GB file stayed open
because a handle was dropped instead of closed.

Anything above the access layer that still opens a file directly is listed in
ALLOWED below, with the reason. Adding a new one requires editing that list,
which is the point — it forces the question to be asked out loud.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "napari_hdf5_activity"

# Layer 1 — the only modules allowed to open files at will
ACCESS_LAYER = {"_io_abstraction.py", "_avi_reader.py", "_frame_source.py"}

# Layer 2 — one file type each, format-specific access is legitimate here
DOMAIN_READERS = {"_reader.py", "_results_io.py", "_metadata.py", "_temperature.py"}

# Layer 3 — computes on data handed to it, must never touch a file
ANALYSIS_LAYER = {
    "_calc.py",
    "_calc_adaptive.py",
    "_calc_calibration.py",
    "_calc_integration.py",
    "_circadian_coherence.py",
    "_circadian_fft.py",
    "_circadian_similarity.py",
    "_cosinor_analysis.py",
    "_fisher_analysis.py",
    "_batch.py",
    "_plot.py",
}

# Layer 4 — presentation. Reviewed exceptions only.
ALLOWED = {
    ("_widget.py", "h5py.File"): (
        2,
        "legacy marker check (guarded by an .h5/.hdf5 test) and the BASIC HDF5 "
        "STRUCTURE debug dump (guarded by _is_zarr)",
    ),
    ("_widget.py", "visititems"): (
        1,
        "part of the BASIC HDF5 STRUCTURE dump above; HDF5-only by nature",
    ),
    ("_widget_telemetry.py", "h5py.File"): (
        2,
        "emergency path when _io_abstraction fails to import, plus the "
        "HDF5-only recursive dataset tree; both guarded",
    ),
    ("_widget_frame_viewer.py", "cv2.VideoCapture"): (
        2,
        "AVI playback; video is deliberately outside the FileReader abstraction",
    ),
}

DIRECT_ACCESS = ("h5py.File", "cv2.VideoCapture", "visititems")


def _modules(names):
    return sorted(p for p in SRC.glob("*.py") if p.name in names)


def _count(path, needle):
    return path.read_text(encoding="utf-8").count(needle)


def _ui_modules():
    return sorted(
        p for p in SRC.glob("_widget*.py") if "Backup" not in str(p)
    )


# ------------------------------------------------------------- analysis layer

@pytest.mark.parametrize("module", sorted(ANALYSIS_LAYER))
def test_analysis_layer_never_opens_a_file(module):
    path = SRC / module
    if not path.exists():
        pytest.skip(f"{module} not present")
    for needle in DIRECT_ACCESS:
        assert _count(path, needle) == 0, (
            f"{module} opens files directly ({needle}). Analysis code should "
            f"receive data, not fetch it."
        )


@pytest.mark.parametrize("module", sorted(ANALYSIS_LAYER))
def test_analysis_layer_does_not_import_storage_libraries(module):
    path = SRC / module
    if not path.exists():
        pytest.skip(f"{module} not present")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "h5py" not in imported, f"{module} imports h5py"
    assert "zarr" not in imported, f"{module} imports zarr"


# --------------------------------------------------------------------- UI layer

@pytest.mark.parametrize("needle", DIRECT_ACCESS)
def test_ui_layer_direct_access_matches_the_allowlist(needle):
    """Every direct file access in a widget must be a reviewed exception."""
    for path in _ui_modules():
        found = _count(path, needle)
        expected, _reason = ALLOWED.get((path.name, needle), (0, ""))
        assert found == expected, (
            f"{path.name} has {found} × '{needle}', expected {expected}.\n"
            f"If this is intentional, add it to ALLOWED in {__file__} "
            f"together with the reason it cannot go through _io_abstraction."
        )


def test_frame_viewer_has_no_direct_hdf5_access():
    """The refactor that introduced FrameSource must not be undone."""
    path = SRC / "_widget_frame_viewer.py"
    assert _count(path, "h5py.File") == 0
    assert "FrameSource" in path.read_text(encoding="utf-8")


def test_allowlist_has_no_stale_entries():
    """A shrinking allowlist is progress; a stale one is misleading."""
    for (module, needle), (expected, reason) in ALLOWED.items():
        path = SRC / module
        assert path.exists(), f"ALLOWED names a module that is gone: {module}"
        actual = _count(path, needle)
        assert actual == expected, (
            f"ALLOWED says {module} has {expected} × '{needle}' "
            f"({reason}), but it has {actual}. Update the list."
        )


# ----------------------------------------------------------- access layer sanity

def test_access_layer_exists():
    for name in ACCESS_LAYER:
        assert (SRC / name).exists(), f"missing access-layer module: {name}"


def test_frame_source_goes_through_the_abstraction_only():
    """FrameSource must not reach past its own layer to h5py."""
    text = (SRC / "_frame_source.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("h5py") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("h5py")
