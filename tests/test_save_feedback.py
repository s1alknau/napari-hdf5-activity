"""A save must end in a visible confirmation - or a visible error."""

import pytest

pytest.importorskip("qtpy")
from napari_hdf5_activity import _widget_export as E  # noqa: E402


class _Box:
    """Records QMessageBox calls instead of opening windows."""

    calls = []

    @classmethod
    def information(cls, parent, title, text):
        cls.calls.append(("info", title, text))

    @classmethod
    def critical(cls, parent, title, text):
        cls.calls.append(("error", title, text))


class _Widget(E.ExportMixin):
    def __init__(self):
        self.log = []

    def _log_message(self, msg):
        self.log.append(msg)


@pytest.fixture
def widget(monkeypatch):
    _Box.calls = []
    monkeypatch.setattr(E, "QMessageBox", _Box)
    return _Widget()


def test_existing_file_is_confirmed_with_name_and_folder(widget, tmp_path):
    out = tmp_path / "results.h5"
    out.write_bytes(b"x")
    assert widget._notify_saved(str(out), "Results saved") is True
    kind, title, text = _Box.calls[-1]
    assert kind == "info"
    assert "results.h5" in text and str(tmp_path) in text


def test_missing_file_is_reported_as_failure(widget, tmp_path):
    """'Success' without a file on disk must not look like success."""
    out = tmp_path / "never_written.h5"
    assert widget._notify_saved(str(out)) is False
    kind, _, text = _Box.calls[-1]
    assert kind == "error"
    assert "never_written.h5" in text


def test_failure_is_shown_and_logged(widget):
    widget._notify_save_failed("disk full", title="HDF5 save failed")
    assert _Box.calls[-1] == ("error", "HDF5 save failed", "disk full")
    assert any("disk full" in m for m in widget.log)
