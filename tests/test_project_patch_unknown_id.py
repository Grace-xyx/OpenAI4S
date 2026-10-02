"""An edit of a project that does not exist has to say so.

`PUT`/`PATCH /projects/{id}` ran `UPDATE projects ... WHERE project_id=?` and
answered `_project_json(store.get_project(id) or {})`. An UPDATE that names no
row is not an error, and `_project_json({})` is `{}`, so renaming a deleted or
mistyped project answered 200 `{}`. A body naming no field on a real project
answers the full project, so the `{}` was not "nothing changed" either: a
client could not tell "renamed" from "no such project", and the workbench's
project modal closed as if the save had landed.

The 404 reuses the team project guard's sentence. Team mode already refused an
unknown id to a non-participant before the handler ran, and INV-13 is that a
missing project and somebody else's read identically, so the last test holds
both modes, both cases, and the admin who passes the guard to one body.

`GET /projects/{id}` keeps its documented `{}` for an unknown id, the same
compatibility shape `GET /frames/{id}` keeps. A read that returns no fields
claims nothing; an edit that returns no fields claimed success.

Ordinary requests use the real handler on a real socket, unmarked, so the
response schema capture observes the 404 and both verbs on a real project.
Marked race tests inject deletion before and after the real database update.
"""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from openai4s.config import Config, LLMConfig
from openai4s.server import gateway as gateway_mod
from openai4s.server import local_auth
from tests._ports import bound_gateway_server
from tests.test_team_auth_routes import (  # noqa: F401  (fixture reuse)
    _fast_pbkdf2,
    _login,
    _TeamDaemon,
)

API = "/api/v1"
MISSING = "project-that-was-never-created"


class _Hub:
    def emitter(self, root_frame_id):
        return lambda event: None

    def broadcast(self, root_frame_id, event):
        return None

    def has_subscriber(self, root_frame_id):
        return False

    def drop_frame(self, root_frame_id):
        return None


def _call(port, method, path, body=None, *, token=None, cookie=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        headers = {}
        if token is not None:
            headers[local_auth.TOKEN_HEADER] = token
        if cookie is not None:
            headers["Cookie"] = cookie
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn.request(method, API + path, body=payload, headers=headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        conn.close()


def test_an_edit_of_an_unknown_project_is_a_404_and_changes_nothing(tmp_path):
    httpd, port = bound_gateway_server()
    cfg = Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
        host="127.0.0.1",
        port=port,
    )
    runner = gateway_mod.SessionRunner(cfg, _Hub(), start_idle_sweeper=False)
    httpd.RequestHandlerClass = gateway_mod.make_handler(cfg, runner.hub, runner)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    token = local_auth.load_or_mint(cfg.data_dir)

    def call(method, path, body=None):
        return _call(port, method, path, body, token=token)

    try:
        status, project = call("POST", "/projects", {"name": "Renames"})
        assert status == 200, project
        pid = project["project_id"]

        # The positive controls: without them a route that refused everything
        # would pass every assertion about the unknown id below. Both verbs,
        # because both share the branch and both are documented.
        status, renamed = call("PATCH", f"/projects/{pid}", {"name": "Assay 7"})
        assert status == 200, renamed
        assert (renamed["project_id"], renamed["name"]) == (pid, "Assay 7")
        status, described = call("PUT", f"/projects/{pid}", {"description": "dose"})
        assert status == 200, described
        assert (described["name"], described["description"]) == ("Assay 7", "dose")

        # Nothing to change on a real project is still the full project, which
        # is why the old `{}` could only ever have meant "no such project".
        status, unchanged = call("PATCH", f"/projects/{pid}", {})
        assert status == 200, unchanged
        assert (unchanged["project_id"], unchanged["name"]) == (pid, "Assay 7")

        before = sorted(p["project_id"] for p in runner.store.list_projects())
        for method in ("PATCH", "PUT"):
            status, refused = call(method, f"/projects/{MISSING}", {"name": "Ghost"})
            assert status == 404, (method, refused)
            assert refused["error"] == "project not found"
            assert refused["code"] == "not_found"
            assert refused["status"] == 404
            assert refused.get("request_id")
        # Refused before the write: no project appears, none is renamed.
        assert runner.store.get_project(MISSING) is None
        assert sorted(p["project_id"] for p in runner.store.list_projects()) == before
        assert runner.store.get_project(pid)["name"] == "Assay 7"

        # The read keeps its documented compatibility shape, deliberately.
        status, read = call("GET", f"/projects/{MISSING}")
        assert (status, read) == (200, {})
    finally:
        httpd.shutdown()
        httpd.server_close()
        runner.close()


def _refusal(body: dict) -> dict:
    """The part of an error body that may not vary with why it was refused."""
    return {key: value for key, value in body.items() if key != "request_id"}


@pytest.mark.parametrize("method", ["PATCH", "PUT"])
def test_missing_and_not_yours_read_the_same_in_both_modes(tmp_path, method):
    """INV-13: which projects exist is the information being protected.

    Team mode answers a non-participant from the project guard before the
    handler runs; an admin passes the guard, and single-user mode has none, so
    both reach the handler. Four refusals, one body -- otherwise the wording
    alone says which path produced it.
    """
    solo = _TeamDaemon(tmp_path / "solo", team_mode=False)
    try:
        status, alone = _call(
            solo.port,
            method,
            f"/projects/{MISSING}",
            {"name": "Ghost"},
            token=solo.token,
        )
    finally:
        solo.close()
    assert status == 404, alone

    team = _TeamDaemon(tmp_path / "team")
    try:
        team.seed_user("alice", "fake-pw-a")
        team.seed_user("bob", "fake-pw-b")
        team.seed_user("root", "fake-pw-r", role="admin")
        alice = _login(team, "alice", "fake-pw-a")
        bob = _login(team, "bob", "fake-pw-b")
        root = _login(team, "root", "fake-pw-r")
        status, created = _call(
            team.port, "POST", "/projects", {"name": "p"}, cookie=alice
        )
        assert status == 200, created
        pid = created["project_id"]

        # The member's rename lands, so bob's 404 below is about bob.
        status, renamed = _call(
            team.port, method, f"/projects/{pid}", {"name": "Mine"}, cookie=alice
        )
        assert (status, renamed["name"]) == (200, "Mine"), renamed
        # And the admin's does too, so root's 404 below is about the id.
        status, renamed = _call(
            team.port, method, f"/projects/{pid}", {"name": "Ours"}, cookie=root
        )
        assert (status, renamed["name"]) == (200, "Ours"), renamed

        status, not_yours = _call(
            team.port, method, f"/projects/{pid}", {"name": "Bob's"}, cookie=bob
        )
        assert status == 404, not_yours
        status, missing = _call(
            team.port, method, f"/projects/{MISSING}", {"name": "Ghost"}, cookie=bob
        )
        assert status == 404, missing
        status, admin_missing = _call(
            team.port, method, f"/projects/{MISSING}", {"name": "Ghost"}, cookie=root
        )
        assert status == 404, admin_missing
        assert team.store.get_project(pid)["name"] == "Ours"
        assert team.store.get_project(MISSING) is None
    finally:
        team.close()

    assert (
        _refusal(alone)
        == _refusal(not_yours)
        == _refusal(missing)
        == _refusal(admin_missing)
    )


@pytest.mark.stubbed_backend
@pytest.mark.parametrize("method", ["PATCH", "PUT"])
@pytest.mark.parametrize("delete_before_write", [True, False])
def test_delete_during_edit_returns_404(
    tmp_path, monkeypatch, method, delete_before_write
):
    """Deletion after the existence check must not turn into a successful edit."""
    daemon = _TeamDaemon(tmp_path, team_mode=False)
    try:
        project = daemon.store.create_project(name="Renames")
        pid = project["project_id"]
        original_update = daemon.store.update_project

        def update_with_delete(project_id, **fields):
            assert project_id == pid
            assert fields == {"name": "Too late"}
            if delete_before_write:
                daemon.store.delete_project(project_id)
            original_update(project_id, **fields)
            if not delete_before_write:
                daemon.store.delete_project(project_id)

        monkeypatch.setattr(daemon.store, "update_project", update_with_delete)
        status, refused = _call(
            daemon.port,
            method,
            f"/projects/{pid}",
            {"name": "Too late"},
            token=daemon.token,
        )
        assert status == 404, refused
        assert refused["error"] == "project not found"
        assert refused["code"] == "not_found"
        assert refused["status"] == 404
        assert refused.get("request_id")
        assert daemon.store.get_project(pid) is None
    finally:
        daemon.close()
