"""worker_spawn_environment must hide __main__ and set worker env only temporarily."""

import os
import sys
import types

from napari_hdf5_activity._mp_utils import (
    WORKER_ENV_FLAG,
    in_worker_process,
    worker_spawn_environment,
)


def test_hides_script_main_file_and_restores_it(monkeypatch):
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = "napari-script.py"
    fake_main.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    with worker_spawn_environment():
        assert not hasattr(fake_main, "__file__")
    assert fake_main.__file__ == "napari-script.py"


def test_leaves_module_main_untouched(monkeypatch):
    """`python -m pkg` mains have a __spec__; spawn imports them by name."""
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = "pkg/__main__.py"
    fake_main.__spec__ = types.SimpleNamespace(name="pkg.__main__")
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    with worker_spawn_environment():
        assert fake_main.__file__ == "pkg/__main__.py"


def test_worker_env_is_set_inside_and_restored_after(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.delenv(WORKER_ENV_FLAG, raising=False)

    with worker_spawn_environment():
        assert in_worker_process()
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert not in_worker_process()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_restores_even_when_the_pool_raises(monkeypatch):
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = "napari-script.py"
    fake_main.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    monkeypatch.delenv(WORKER_ENV_FLAG, raising=False)

    try:
        with worker_spawn_environment():
            raise RuntimeError("pool broke")
    except RuntimeError:
        pass
    assert fake_main.__file__ == "napari-script.py"
    assert not in_worker_process()
