"""Helpers for starting worker processes from inside napari on Windows.

Windows has no ``fork``: every worker is a fresh interpreter that, by default,
re-executes the parent's ``__main__`` script.  Inside napari that script is
``napari-script.py``, so each worker imported the whole of napari (Qt, vispy,
numba ...), several hundred MB per worker, and wrote napari's theme icons to
the shared cache.  Workers starting at the same time collided on those files
(``PermissionError``) and died, which broke the pool.

:func:`worker_spawn_environment` wraps the creation and use of a pool so that

* the parent's ``__main__`` is not re-executed in the workers (the worker
  functions live in this package, not in ``__main__``),
* workers do not import CuPy (they compute on NumPy masks; initialising CUDA
  costs ~1 GB of commit charge per process),
* BLAS / OpenMP run single-threaded per worker (parallelism comes from the
  processes; a many-threaded OpenBLAS reserves hundreds of MB per process).

With all three, a worker's commit charge drops from ~1 GB to under 100 MB.
"""

import contextlib
import os
import sys

# Read by _reader.py at import time to skip the CuPy import in workers.
WORKER_ENV_FLAG = "NAPARI_HDF5_ACTIVITY_WORKER"

_WORKER_ENV = {
    WORKER_ENV_FLAG: "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def in_worker_process() -> bool:
    """True inside a worker started under :func:`worker_spawn_environment`."""
    return os.environ.get(WORKER_ENV_FLAG) == "1"


@contextlib.contextmanager
def worker_spawn_environment():
    """Context for creating and using a process pool (see module docstring).

    Workers are spawned lazily (on ``submit``), so the whole lifetime of the
    pool must be inside this context::

        with worker_spawn_environment(), ProcessPoolExecutor(n) as ex:
            ...
    """
    main = sys.modules.get("__main__")
    hidden_file = None
    # spawn re-runs __main__ via its __file__ when __main__ has no __spec__
    # (scripts such as napari-script.py).  Hiding __file__ while workers are
    # created makes them skip that step.
    if main is not None and getattr(main, "__spec__", None) is None:
        hidden_file = main.__dict__.pop("__file__", None)

    saved_env = {key: os.environ.get(key) for key in _WORKER_ENV}
    os.environ.update(_WORKER_ENV)
    try:
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if hidden_file is not None:
            main.__file__ = hidden_file
