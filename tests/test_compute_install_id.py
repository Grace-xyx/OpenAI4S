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
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("hard_links", [True, False])
def test_concurrent_publishers_keep_the_same_id_during_repair_or_fallback(
    home, data_dir, monkeypatch, hard_links
):
    data_dir.mkdir()
    path = data_dir / "install-id"
    if hard_links:
        path.write_bytes(b"")
    else:

        def no_hard_links(*_args, **_kwargs):
            raise OSError(errno.EPERM, "hard links not supported")

        monkeypatch.setattr(manager_module.os, "link", no_hard_links)

    # Both sessions have already resolved an absent/corrupt id and minted a
    # candidate. Pause one just before publishing, then let the other try.
    # An unlocked replace lets the latter return an id the first overwrites.
    replacing = threading.Event()
    release = threading.Event()
    real_replace = os.replace

    def paused_replace(src, dst):
        if Path(src).read_text("utf-8") == "first-owner":
            replacing.set()
            assert release.wait(10)
        return real_replace(src, dst)

    monkeypatch.setattr(manager_module.os, "replace", paused_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(manager_module._publish_install_id, path, "first-owner")
        try:
            assert replacing.wait(10)
            second = pool.submit(
                manager_module._publish_install_id, path, "second-owner"
            )
            try:
                second.result(timeout=0.5)
            except TimeoutError:
                pass  # A serialized publisher waits for the first to finish.
        finally:
            release.set()
        owners = [first.result(timeout=10), second.result(timeout=10)]

    assert owners == [path.read_text("utf-8")] * 2
    assert ComputeManager(Config())._install_id == owners[0]
    assert list(data_dir.glob(".install-id-*")) == []


def test_an_unreadable_id_is_not_overwritten(home, data_dir, monkeypatch):
    data_dir.mkdir()
    path = data_dir / "install-id"
    path.write_text("existing-owner", encoding="utf-8")
    real_read = Path.read_text

    def denied_read(self, *args, **kwargs):
        if self == path:
            raise PermissionError(errno.EACCES, "read denied")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_read)

    # Retain the historical in-memory fallback on I/O failure, but never
    # treat lack of read permission as evidence that an owner tag is corrupt.
    assert ComputeManager(Config())._install_id
    assert path.read_bytes() == b"existing-owner"


@pytest.mark.parametrize("case", ["fresh", "corrupt", "no-hard-links"])
def test_separate_processes_publish_one_owner(home, data_dir, case):
    data_dir.mkdir()
    path = data_dir / "install-id"
    if case == "corrupt":
        path.write_bytes(b"\xff")
    code = """
import errno
import os
import sys
import time
from pathlib import Path
from openai4s.compute.manager import _publish_install_id

if sys.argv[3] == "no-hard-links":
    def no_hard_links(*args, **kwargs):
        raise OSError(errno.EPERM, "hard links not supported")
    os.link = no_hard_links
real_replace = os.replace
def slow_replace(src, dst):
    time.sleep(0.1)
    return real_replace(src, dst)
os.replace = slow_replace
print("ready", flush=True)
sys.stdin.readline()
print(_publish_install_id(Path(sys.argv[1]), sys.argv[2]), flush=True)
"""
    processes = []
    try:
        for i in range(4):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", code, str(path), f"owner-{i}", case],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for process in processes:
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        owners = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stderr
            owners.append(stdout.strip())
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=15)

    assert owners == [path.read_text("utf-8")] * 4
    assert ComputeManager(Config())._install_id == owners[0]
    assert list(data_dir.glob(".install-id-*")) == []


def test_a_crashed_publisher_releases_the_lock(home, data_dir):
    data_dir.mkdir()
    lock = data_dir / ".install-id.lock"
    code = """
import sys
from pathlib import Path
from openai4s.compute.manager import _install_id_lock
with _install_id_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
    finally:
        process.kill()
        process.communicate(timeout=15)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from openai4s.compute.manager import ComputeManager; "
            "print(ComputeManager._resolve_install_id(Path(sys.argv[1])))",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (data_dir / "install-id").read_text("utf-8")


def test_windows_lock_covers_byte_zero_and_closes_on_error(tmp_path, monkeypatch):
    calls = []

    def locking(fd, mode, length):
        calls.append((fd, mode, length, os.lseek(fd, 0, os.SEEK_CUR)))

    # The real POSIX branch is exercised by the process races above. Exercise
    # the Windows adapter without pretending this is a Windows runtime test.
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(LK_LOCK=1, locking=locking)
    )
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="publisher failed"):
        with manager_module._install_id_lock(tmp_path / "owner.lock"):
            raise RuntimeError("publisher failed")
    fd, mode, length, offset = calls[0]
    assert (mode, length, offset) == (1, 1, 0)
    with pytest.raises(OSError):
        os.fstat(fd)
