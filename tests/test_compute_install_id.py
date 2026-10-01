"""The BYOC owner tag lives in the data dir, and an existing one survives the move.

`ComputeManager` tags every sandbox it creates with a per-install id, and the
resident helper refuses to poll, cancel or reuse a sandbox whose tag differs.
Losing the id therefore strands running, billing work. The id used to live at
`~/.openai4s/install-id` whatever the data dir was. A custom
`OPENAI4S_DATA_DIR` kept its id outside the data dir, a container whose data
dir is the volume minted a fresh id on every restart, and the offline suite and
the capture scripts read, or on a clean machine created, the developer's real
file.

Every test here points HOME at an empty tmp dir and deletes the conftest pin on
`OPENAI4S_INSTALL_ID`, so the resolution actually runs.
"""

import errno
import os
import re
from pathlib import Path

import pytest

import openai4s.compute.manager as manager_module
from openai4s.compute.manager import ComputeManager
from openai4s.config import Config


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENAI4S_INSTALL_ID", raising=False)
    return home


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("OPENAI4S_DATA_DIR", str(data))
    return data


def _legacy(home: Path) -> Path:
    return home / ".openai4s" / "install-id"


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_a_manager_on_a_custom_data_dir_writes_nothing_under_home(home, data_dir):
    manager = ComputeManager(Config())

    assert _tree(home) == []
    stored = (data_dir / "install-id").read_text("utf-8")
    assert stored == manager._install_id
    assert re.fullmatch(r"[0-9a-f]{32}", stored)
    # The point of persisting it: a restart reads back the same owner.
    assert ComputeManager(Config())._install_id == stored
    assert _tree(home) == []


def test_an_existing_legacy_id_is_kept_and_copied_into_the_data_dir(home, data_dir):
    legacy = _legacy(home)
    legacy.parent.mkdir()
    legacy.write_text("legacy-owner-tag\n", encoding="utf-8")
    before = legacy.stat()

    assert ComputeManager(Config())._install_id == "legacy-owner-tag"
    assert (data_dir / "install-id").read_text("utf-8") == "legacy-owner-tag"
    # Copied, not moved or rewritten: an older build, or another data dir on
    # this machine, may still read it.
    after = legacy.stat()
    assert legacy.read_text("utf-8") == "legacy-owner-tag\n"
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)

    # Once copied, the data dir is the record; the legacy file no longer steers.
    legacy.write_text("someone-else", encoding="utf-8")
    assert ComputeManager(Config())._install_id == "legacy-owner-tag"


def test_the_default_data_dir_still_reads_the_same_file(home, monkeypatch):
    # With no OPENAI4S_DATA_DIR the data dir *is* ~/.openai4s, so an install
    # that never customised it sees no change at all.
    monkeypatch.delenv("OPENAI4S_DATA_DIR", raising=False)
    legacy = _legacy(home)
    legacy.parent.mkdir()
    legacy.write_text("default-install", encoding="utf-8")

    assert Config().data_dir == home / ".openai4s"
    assert ComputeManager(Config())._install_id == "default-install"
    assert legacy.read_text("utf-8") == "default-install"


@pytest.mark.parametrize("planted", [b"", b"  \n", b"\xff\xfe not utf-8"])
def test_an_unusable_stored_id_is_replaced_rather_than_used(home, data_dir, planted):
    # The old resolver returned an empty file's "" as the owner tag, and a
    # non-UTF-8 one raised out of the constructor.
    data_dir.mkdir()
    (data_dir / "install-id").write_bytes(planted)

    install_id = ComputeManager(Config())._install_id

    assert re.fullmatch(r"[0-9a-f]{32}", install_id)
    assert (data_dir / "install-id").read_text("utf-8") == install_id
    assert _tree(home) == []


def test_an_unusable_stored_id_is_repaired_from_the_legacy_one(home, data_dir):
    legacy = _legacy(home)
    legacy.parent.mkdir()
    legacy.write_text("legacy-owner-tag", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "install-id").write_text("", encoding="utf-8")

    assert ComputeManager(Config())._install_id == "legacy-owner-tag"
    assert (data_dir / "install-id").read_text("utf-8") == "legacy-owner-tag"


def test_the_environment_override_wins_and_writes_nothing(home, data_dir, monkeypatch):
    monkeypatch.setenv("OPENAI4S_INSTALL_ID", "pinned-by-operator")

    assert ComputeManager(Config())._install_id == "pinned-by-operator"
    assert not (data_dir / "install-id").exists()
    assert _tree(home) == []


def test_a_manager_that_loses_the_publish_race_adopts_the_winner(
    home, data_dir, monkeypatch
):
    # Two sessions resolving on a fresh data dir. The old check-then-write let
    # both mint, and the one whose write lost tagged its sandboxes with an id
    # nothing would read back. Simulate the other manager publishing between
    # our read and our link.
    real_link = os.link

    def other_manager_wins(src, dst, *args, **kwargs):
        Path(dst).write_text("winner-owner-tag", encoding="utf-8")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(manager_module.os, "link", other_manager_wins)

    assert ComputeManager(Config())._install_id == "winner-owner-tag"
    assert (data_dir / "install-id").read_text("utf-8") == "winner-owner-tag"
    assert [p.name for p in data_dir.glob(".install-id-*")] == []


def test_a_filesystem_without_hard_links_still_persists_the_id(
    home, data_dir, monkeypatch
):
    def no_hard_links(*_args, **_kwargs):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(manager_module.os, "link", no_hard_links)

    install_id = ComputeManager(Config())._install_id

    assert (data_dir / "install-id").read_text("utf-8") == install_id
    assert [p.name for p in data_dir.glob(".install-id-*")] == []
    assert ComputeManager(Config())._install_id == install_id
