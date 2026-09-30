"""The suite-wide guard that keeps the suite out of the developer's real data dir.

``conftest.py`` redirects ``OPENAI4S_DATA_DIR`` per test, but anything resolved
before that redirect -- a module-scoped fixture, a cache it filled -- keeps the
real ``~/.openai4s``. The guard refuses a Store or a data-dir creation there
outright and records it, so the test fails even when the code under test
swallows the refusal. These tests
point the guard at a stand-in under ``tmp_path``; a broken guard must never be
able to prove itself against the real directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openai4s.store import Store, get_store


@pytest.fixture
def stand_in_real_dir(real_data_dir_guard, tmp_path, monkeypatch) -> Path:
    root = (tmp_path / "home" / ".openai4s").resolve()
    monkeypatch.setattr(real_data_dir_guard, "roots", (root,))
    return root


def test_the_guard_covers_this_machines_default_data_dir(real_data_dir_guard) -> None:
    assert (Path.home() / ".openai4s").resolve() in real_data_dir_guard.roots


def test_a_store_there_is_refused_before_it_touches_disk(
    real_data_dir_guard, stand_in_real_dir: Path, monkeypatch
) -> None:
    with pytest.raises(RuntimeError, match="real data dir"):
        Store(stand_in_real_dir / "openai4s.db")
    with pytest.raises(RuntimeError, match="real data dir"):
        get_store(stand_in_real_dir / "nested" / "openai4s.db")
    # Relative paths resolve against the cwd, as sqlite would open them.
    stand_in_real_dir.parent.mkdir(parents=True)
    monkeypatch.chdir(stand_in_real_dir.parent)
    with pytest.raises(RuntimeError, match="real data dir"):
        Store(Path(".openai4s") / "openai4s.db")

    # Refused before the schema preflight, the mkdir, or the connect.
    assert not stand_in_real_dir.exists()
    touched = real_data_dir_guard.drain()
    assert len(touched) == 3
    assert all(str(stand_in_real_dir) in entry for entry in touched)


def test_creating_the_data_dir_there_is_refused(
    real_data_dir_guard, stand_in_real_dir: Path, monkeypatch
) -> None:
    """``get_config()`` creates and hardens its data dir; that is a write too.

    Five Skill fixtures called it at module scope for ``skills_dir`` alone,
    before the per-test redirect, and so ran ``mkdir``/``chmod`` on the real
    ``~/.openai4s`` and its subdirectories.
    """

    from openai4s.config import Config, get_config

    with pytest.raises(RuntimeError, match="real data dir"):
        Config(data_dir=stand_in_real_dir).ensure_dirs()
    monkeypatch.setenv("OPENAI4S_DATA_DIR", str(stand_in_real_dir))
    with pytest.raises(RuntimeError, match="real data dir"):
        get_config()
    # What those fixtures needed, without touching the directory at all.
    assert get_config(initialize_dirs=False).data_dir == stand_in_real_dir

    assert not stand_in_real_dir.exists()
    assert len(real_data_dir_guard.drain()) == 2


def test_a_swallowed_refusal_is_still_recorded(
    real_data_dir_guard, stand_in_real_dir: Path
) -> None:
    try:
        get_store(stand_in_real_dir / "openai4s.db")
    except Exception:  # what the eval's run_system does with every error
        pass
    touched = real_data_dir_guard.drain()
    assert len(touched) == 1
    # The record names the caller, since no traceback reached the test.
    assert __file__ in touched[0]


def test_stores_elsewhere_open_normally(
    real_data_dir_guard, stand_in_real_dir: Path, tmp_path: Path
) -> None:
    Store(tmp_path / "elsewhere" / "openai4s.db").close()
    # A name that merely shares the protected root's prefix is not under it.
    sibling = stand_in_real_dir.with_name(".openai4s-other")
    Store(sibling / "openai4s.db").close()
    assert real_data_dir_guard.drain() == []
