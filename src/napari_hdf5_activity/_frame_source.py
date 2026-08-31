"""
_frame_source.py - Sequential frame access to one recording

The frame viewer scrubs through tens of thousands of frames, so the file has to
stay open between reads — re-opening per frame is not an option at 51 840
frames. Keeping a raw handle in the widget, however, produced three problems:

* the same attribute held either an ``h5py.File`` or a :class:`FileReader`,
  forcing a type check at every read site;
* every layout (stacked / individual frames, HDF5 / Zarr) added another branch
  to the loader, eight direct ``h5py.File`` calls in total;
* the handle was only ever set to ``None``, never closed — on a 130 GB
  recording the file stayed open until the garbage collector happened to run.

:class:`FrameSource` keeps the long-lived handle, which is the right call for
sequential access, but wraps it: one type, one read method, deterministic
close. Everything goes through ``open_file_reader``, so HDF5 and Zarr need no
separate code paths.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dataset names probed when structure detection cannot identify the layout
FALLBACK_DATASET_NAMES = ("frames", "images", "data")

LAYOUT_STACKED = "stacked"        # one (N, H, W) array
LAYOUT_INDIVIDUAL = "individual"  # one array per frame inside a group


class FrameSourceError(RuntimeError):
    """No readable image data was found in the recording."""


class FrameSource:
    """One open recording, addressed by frame index.

    Use as a context manager, or call :meth:`open` and :meth:`close` explicitly
    when the lifetime has to follow a widget rather than a block::

        with FrameSource(path) as src:
            frame = src.read_frame(0)

    Attributes are only meaningful once opened.
    """

    def __init__(self, path: str):
        self.path = path
        self._reader = None
        self._layout: Optional[str] = None
        self._dataset: Optional[str] = None
        self._frame_names: Optional[List[str]] = None
        self._n_frames: int = 0
        self._frame_shape: Tuple[int, ...] = ()
        self._file_format: Optional[str] = None

    # -- lifetime ------------------------------------------------------------

    def open(self) -> "FrameSource":
        from ._io_abstraction import open_file_reader

        self._reader = open_file_reader(self.path).open()
        try:
            self._resolve_layout()
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        """Release the file. Safe to call repeatedly."""
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    def __enter__(self) -> "FrameSource":
        return self.open() if self._reader is None else self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._reader is not None

    # -- what was found ------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._frame_shape

    @property
    def dataset_name(self) -> Optional[str]:
        return self._dataset

    @property
    def layout(self) -> Optional[str]:
        """``"stacked"`` or ``"individual"``."""
        return self._layout

    @property
    def file_format(self) -> Optional[str]:
        """``"hdf5"`` or ``"zarr"``."""
        return self._file_format

    def describe(self) -> str:
        """One line for the log."""
        return (
            f"{self._file_format or '?'} / {self._layout or '?'} · "
            f"{self._n_frames} frames · {self._dataset} · shape {self._frame_shape}"
        )

    # -- reading -------------------------------------------------------------

    def read_frame(self, index: int) -> np.ndarray:
        """Read one frame. Index is 0-based over the whole recording."""
        if self._reader is None:
            raise FrameSourceError("FrameSource is not open")
        if not 0 <= index < self._n_frames:
            raise IndexError(
                f"frame {index} out of range (0..{self._n_frames - 1})"
            )

        if self._layout == LAYOUT_INDIVIDUAL:
            path = f"{self._dataset}/{self._frame_names[index]}"
            return np.asarray(self._reader.read_all(path))
        return np.asarray(self._reader.read_frame(self._dataset, index))

    def read_attrs(self, path: str = "/") -> Dict[str, Any]:
        if self._reader is None:
            raise FrameSourceError("FrameSource is not open")
        return self._reader.get_attrs(path)

    # -- layout resolution ---------------------------------------------------

    def _resolve_layout(self) -> None:
        """Work out where the frames live, preferring structure detection."""
        if self._apply_structure_detection():
            return
        if self._probe_for_frames():
            return
        raise FrameSourceError(
            f"No readable image data in {self.path}. "
            f"Top-level keys: {self._reader.keys('/')}"
        )

    def _apply_structure_detection(self) -> bool:
        try:
            from ._reader import detect_file_structure_type

            info = detect_file_structure_type(self.path)
        except Exception:
            return False

        stype = info.get("type")
        if stype not in ("stacked_frames", "individual_frames"):
            return False

        self._file_format = info.get("file_format")
        self._n_frames = int(info.get("frame_count", 0) or 0)
        self._frame_shape = tuple(info.get("frame_shape") or ())
        dataset = info.get("data_location") or info.get("dataset_name")

        if stype == "stacked_frames":
            self._layout = LAYOUT_STACKED
            self._dataset = dataset or "frames"
            return self._n_frames > 0

        # individual frames: one array per frame inside a group
        template = info.get("key_template")
        keys = info.get("frame_keys")
        if template:
            self._frame_names = [template.format(i) for i in range(self._n_frames)]
        elif keys:
            self._frame_names = list(keys)
        else:
            return False
        self._layout = LAYOUT_INDIVIDUAL
        self._dataset = dataset or "images"
        return self._n_frames > 0

    def _probe_for_frames(self) -> bool:
        """Last resort when detection cannot classify the file.

        Mirrors the probe the frame viewer used to run with raw h5py, but goes
        through the reader so it works for Zarr as well.
        """
        reader = self._reader
        try:
            top = reader.keys("/")
        except Exception:
            return False

        for name in FALLBACK_DATASET_NAMES:
            if name not in top:
                continue

            # a) the name is itself a stacked array
            if reader.is_array(name):
                shape = reader.shape(name)
                if len(shape) >= 3:
                    self._layout = LAYOUT_STACKED
                    self._dataset = name
                    self._n_frames = int(shape[0])
                    self._frame_shape = tuple(shape[1:])
                    return True
                continue

            # b) a group holding a stacked "frames" array
            try:
                children = reader.keys(name)
            except Exception:
                continue
            if "frames" in children and reader.is_array(f"{name}/frames"):
                shape = reader.shape(f"{name}/frames")
                self._layout = LAYOUT_STACKED
                self._dataset = f"{name}/frames"
                self._n_frames = int(shape[0])
                self._frame_shape = tuple(shape[1:])
                return True

            # c) a group of individually stored frames
            frame_names = sorted(k for k in children if k.startswith("frame_"))
            if frame_names:
                self._layout = LAYOUT_INDIVIDUAL
                self._dataset = name
                self._frame_names = frame_names
                self._n_frames = len(frame_names)
                try:
                    self._frame_shape = tuple(
                        reader.shape(f"{name}/{frame_names[0]}")
                    )
                except Exception:
                    self._frame_shape = ()
                return True

        return False
