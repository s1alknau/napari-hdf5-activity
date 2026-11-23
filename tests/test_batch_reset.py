"""Leaving batch mode must drop the pooled data, not just empty the table.

The reset methods only touch the batch table, two text widgets and the log,
so they are tested on a light stand-in instead of a full napari widget.
"""

import types

import pytest

pytest.importorskip("qtpy")
from napari_hdf5_activity._widget_circadian import CircadianMixin  # noqa: E402


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def rowCount(self):
        return self._rows

    def setRowCount(self, n):
        self._rows = n


class _Text:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t

    def setPlainText(self, t):
        self.text = t


class _Widget(CircadianMixin):
    def __init__(self, rows=2, pooled=True):
        self.batch_table = _Table(rows)
        self.batch_status_label = _Text()
        self.fisher_results_text = _Text()
        self.log = []
        self._batch_result = types.SimpleNamespace(n_datasets=2) if pooled else None
        # pooled analysis state with composite ROI ids
        self.merged_results = {1001: [(0, 1.0)], 2001: [(0, 2.0)]}
        self.fraction_data = {1001: [(0, 0.5)]}
        self.roi_baseline_means = {1001: 0.1}
        self.fisher_analysis_results = {1001: {"x": 1}}
        self._pop_spectrum_data = object()

    def _log_message(self, msg):
        self.log.append(msg)


def test_reset_button_drops_pooled_data():
    w = _Widget()
    w._batch_clear()
    assert w.batch_table.rowCount() == 0
    assert w._batch_result is None
    assert not w._batch_is_active()
    assert w.merged_results == {}
    assert w.fraction_data == {}
    assert w.roi_baseline_means == {}
    assert w.fisher_analysis_results == {}
    assert w._pop_spectrum_data is None


def test_reset_keeps_single_dataset_data():
    """Without a pooled load, clearing the table must not wipe a normal analysis."""
    w = _Widget(rows=1, pooled=False)
    w._batch_clear()
    assert w.batch_table.rowCount() == 0
    assert w.merged_results  # untouched


def test_new_data_leaves_batch_mode():
    w = _Widget()
    w._leave_batch_mode("new file loaded")
    assert not w._batch_is_active()
    assert w.batch_table.rowCount() == 0
    assert w.merged_results == {}
    assert any("new file loaded" in m for m in w.log)


def test_leave_batch_mode_is_silent_when_nothing_is_batched():
    w = _Widget(rows=0, pooled=False)
    w._leave_batch_mode("new analysis started")
    assert w.log == []
    assert w.merged_results  # untouched
