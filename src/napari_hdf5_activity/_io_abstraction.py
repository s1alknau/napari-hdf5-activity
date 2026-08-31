"""_io_abstraction.py — Unified file reader for HDF5 and Zarr formats.

Provides a common interface so the rest of the codebase does not need to
know which on-disk format is in use.

Usage (context manager — preferred for short-lived reads):

    with open_file_reader(path) as r:
        keys = r.keys("/")
        data = r.read_slice("frames", 0, 10)

Usage (persistent handle — frame viewer):

    r = open_file_reader(path)
    r.open()
    frame = r.read_frame("frames", 42)
    r.close()
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_file_format(path: str) -> str:
    """Return ``'hdf5'`` or ``'zarr'`` for *path*, or raise ``ValueError``.

    Zarr stores are recognised by:
    - any path whose name ends in ``.zarr``
    - a directory that contains a ``.zgroup``, ``.zarray``, or ``zarr.json`` marker
    - a zip-archive whose name ends in ``.zarr``

    HDF5 files are recognised by:
    - extensions ``.h5``, ``.hdf5``, ``.hdf``
    - the ``\\x89HDF`` magic bytes at the start of the file
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".h5", ".hdf5", ".hdf"):
        return "hdf5"

    if ext == ".zarr":
        return "zarr"

    if os.path.isdir(path):
        for marker in (".zgroup", ".zarray", ".zmetadata", "zarr.json"):
            if os.path.exists(os.path.join(path, marker)):
                return "zarr"

    if os.path.isfile(path):
        try:
            with open(path, "rb") as fh:
                magic = fh.read(8)
            if magic[:4] == b"\x89HDF":
                return "hdf5"
        except OSError:
            pass

    raise ValueError(f"Cannot determine file format for: {path}")


def is_supported_file(path: str) -> bool:
    """Return ``True`` if *path* is a recognised HDF5 or Zarr file."""
    try:
        detect_file_format(path)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ZarrVersionError(RuntimeError):
    """The installed zarr-python cannot read this store's format version."""


class FileReader(ABC):
    """Minimal common interface over HDF5 (h5py) and Zarr stores.

    Must be opened before use — either via the context-manager protocol or
    by calling :meth:`open` / :meth:`close` explicitly.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._handle = None

    # -- lifecycle -----------------------------------------------------------

    @abstractmethod
    def open(self) -> "FileReader":
        """Open the underlying store and return *self*."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the underlying store."""
        ...

    def __enter__(self) -> "FileReader":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    # -- structure queries ---------------------------------------------------

    @abstractmethod
    def keys(self, group: str = "/") -> List[str]:
        """Return immediate children of *group*."""
        ...

    @abstractmethod
    def is_array(self, path: str) -> bool:
        """``True`` if *path* refers to an array/dataset (not a group)."""
        ...

    @abstractmethod
    def shape(self, path: str) -> Tuple[int, ...]:
        """Return the shape of the array at *path*."""
        ...

    @abstractmethod
    def dtype(self, path: str) -> np.dtype:
        """Return the dtype of the array at *path*."""
        ...

    # -- data access ---------------------------------------------------------

    @abstractmethod
    def read_slice(self, path: str, start: int, end: int) -> np.ndarray:
        """Read ``[start:end]`` along the first axis of *path*."""
        ...

    @abstractmethod
    def read_frame(self, path: str, index: int) -> np.ndarray:
        """Read a single element at *index* along the first axis of *path*."""
        ...

    @abstractmethod
    def read_all(self, path: str) -> np.ndarray:
        """Read the entire array at *path*."""
        ...

    # -- metadata ------------------------------------------------------------

    @abstractmethod
    def get_attrs(self, path: str = "/") -> Dict[str, Any]:
        """Return all attributes of *path* as a plain Python ``dict``."""
        ...

    # -- convenience ---------------------------------------------------------

    def path_exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists inside the open store."""
        try:
            # Attempt to navigate; KeyError means it doesn't exist
            if path in ("/", ""):
                return True
            self.shape(path)  # will raise KeyError for groups too on some backends
            return True
        except (KeyError, Exception):
            # Try keys traversal as fallback
            parts = [p for p in path.strip("/").split("/") if p]
            if not parts:
                return True
            try:
                node_keys = self.keys("/")
                if parts[0] not in node_keys:
                    return False
                if len(parts) == 1:
                    return True
                # Walk deeper
                current = parts[0]
                for part in parts[1:]:
                    node_keys = self.keys(current)
                    if part not in node_keys:
                        return False
                    current = f"{current}/{part}"
                return True
            except Exception:
                return False


# ---------------------------------------------------------------------------
# HDF5 implementation
# ---------------------------------------------------------------------------

class HDF5FileReader(FileReader):
    """:class:`FileReader` backed by h5py."""

    def open(self) -> "HDF5FileReader":
        import h5py  # lazy import so zarr-only installs work
        self._h5py = h5py
        self._handle = h5py.File(self.path, "r")
        return self

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None

    # -- structure queries ---------------------------------------------------

    def keys(self, group: str = "/") -> List[str]:
        node = self._handle if group in ("/", "") else self._handle[group]
        return list(node.keys())

    def is_array(self, path: str) -> bool:
        import h5py
        return isinstance(self._handle[path], h5py.Dataset)

    def shape(self, path: str) -> Tuple[int, ...]:
        return tuple(self._handle[path].shape)

    def dtype(self, path: str) -> np.dtype:
        return self._handle[path].dtype

    # -- data access ---------------------------------------------------------

    def read_slice(self, path: str, start: int, end: int) -> np.ndarray:
        return self._handle[path][start:end]

    def read_frame(self, path: str, index: int) -> np.ndarray:
        return self._handle[path][index]

    def read_all(self, path: str) -> np.ndarray:
        return self._handle[path][:]

    # -- metadata ------------------------------------------------------------

    def get_attrs(self, path: str = "/") -> Dict[str, Any]:
        node = self._handle if path in ("/", "") else self._handle[path]
        result: Dict[str, Any] = {}
        for k, v in node.attrs.items():
            # h5py may return bytes; decode them for a uniform interface
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            result[k] = v
        return result


# ---------------------------------------------------------------------------
# Zarr implementation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Zarr format versions
# ---------------------------------------------------------------------------
# Two things are versioned independently and are easy to confuse:
#   * the *store format* — v2 uses .zgroup/.zarray, v3 uses zarr.json
#   * the *library*      — zarr-python 2.x reads v2 only, 3.x reads both
# A v3 store opened with zarr-python 2.x fails with a bare
# "PathNotFoundError: nothing found at path ''", which says nothing useful.

def detect_zarr_format_version(path: str) -> Optional[int]:
    """Return 2 or 3 for a Zarr store on disk, or ``None`` if undetermined."""
    if not os.path.isdir(path):
        return None  # zip store or similar — let zarr decide
    if os.path.exists(os.path.join(path, "zarr.json")):
        return 3
    if os.path.exists(os.path.join(path, ".zgroup")) or os.path.exists(
        os.path.join(path, ".zarray")
    ):
        return 2
    return None


def zarr_library_major() -> Optional[int]:
    """Major version of the installed zarr-python, or ``None`` if absent."""
    try:
        import zarr
    except ImportError:
        return None
    try:
        return int(str(zarr.__version__).split(".")[0])
    except (ValueError, AttributeError):
        return None


class ZarrFileReader(FileReader):
    """
    :class:`FileReader` backed by zarr.

    Requires ``zarr`` to be installed (``pip install zarr``).
    Supports directory stores, zip stores, and any other store that
    ``zarr.open()`` accepts.
    """

    def open(self) -> "ZarrFileReader":
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "zarr is required to open Zarr stores. "
                "Install it with:  pip install zarr"
            ) from exc
        self._zarr = zarr

        # Fail with something actionable when the library cannot read the store
        store_version = detect_zarr_format_version(self.path)
        lib_major = zarr_library_major()
        if store_version == 3 and lib_major is not None and lib_major < 3:
            raise ZarrVersionError(
                f"'{os.path.basename(self.path)}' is a Zarr v3 store "
                f"(it has a zarr.json), but zarr-python {zarr.__version__} is "
                f"installed and reads v2 only. Install zarr 3 to read both "
                f"formats:  pip install \"zarr>=3\"  (needs Python 3.11+)."
            )

        # A .zarr path that is a regular file is a zip store.
        # ZipStore location differs between zarr v2 (zarr.ZipStore) and
        # v3 (zarr.storage.ZipStore).  Try both, then fall back to a direct
        # open() call in case a future version handles zip automatically.
        if os.path.isfile(self.path):
            ZipStore = (
                getattr(zarr, "ZipStore", None)
                or getattr(getattr(zarr, "storage", None), "ZipStore", None)
            )
            if ZipStore is not None:
                try:
                    self._handle = zarr.open(ZipStore(self.path, mode="r"), mode="r")
                except TypeError:
                    # zarr v3 ZipStore may not accept a mode kwarg
                    self._handle = zarr.open(ZipStore(self.path), mode="r")
            else:
                # Last resort: let zarr figure it out
                self._handle = zarr.open(self.path, mode="r")
        else:
            self._handle = zarr.open(self.path, mode="r")
        return self

    def close(self) -> None:
        # Zarr stores do not require explicit closing, but we clear the handle
        # to stay consistent with the FileReader contract.
        self._handle = None

    def _node(self, path: str):
        """Return the zarr Array or Group at *path*."""
        if path in ("/", ""):
            return self._handle
        return self._handle[path]

    # -- structure queries ---------------------------------------------------

    def keys(self, group: str = "/") -> List[str]:
        return list(self._node(group).keys())

    def is_array(self, path: str) -> bool:
        import zarr
        return isinstance(self._node(path), zarr.Array)

    def shape(self, path: str) -> Tuple[int, ...]:
        return tuple(self._node(path).shape)

    def dtype(self, path: str) -> np.dtype:
        return np.dtype(self._node(path).dtype)

    # -- data access ---------------------------------------------------------

    def read_slice(self, path: str, start: int, end: int) -> np.ndarray:
        return np.asarray(self._node(path)[start:end])

    def read_frame(self, path: str, index: int) -> np.ndarray:
        return np.asarray(self._node(path)[index])

    def read_all(self, path: str) -> np.ndarray:
        return np.asarray(self._node(path)[:])

    # -- metadata ------------------------------------------------------------

    def get_attrs(self, path: str = "/") -> Dict[str, Any]:
        # Zarr attrs are already plain Python objects (JSON-derived)
        return dict(self._node(path).attrs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def open_file_reader(path: str) -> FileReader:
    """Return the appropriate :class:`FileReader` for *path*.

    The store is **not** opened automatically.  Call :meth:`~FileReader.open`
    or use the instance as a context manager::

        with open_file_reader(path) as r:
            data = r.read_slice("frames", 0, 100)
    """
    fmt = detect_file_format(path)
    if fmt == "hdf5":
        return HDF5FileReader(path)
    if fmt == "zarr":
        return ZarrFileReader(path)
    raise ValueError(f"Unsupported format '{fmt}' for: {path}")
