"""Id-addressed enable/rename mutations must fail closed on a missing row.

Each route below answered 200 for a resource that does not exist —
indistinguishable from a write that landed, and the same defect class
#202/#206 closed for session/project edits:

- ``PUT|PATCH /folders/{id}`` renamed through a bare UPDATE that matches no
  row. The sibling DELETE stays idempotent by design.
- ``PUT|PATCH /agents/{name}/enabled`` wrote a capability row for ANY name —
  an orphan state row — while the ``GET /agents/{name}`` sibling answered
  404 for the same name.
- ``PUT|PATCH /connectors/{id}/enabled`` flipped a bare UPDATE that matches
  no row, while the edit and probe siblings answer 404 for the same id.

Every test here fails if the refusal is removed: the route goes back to a
200 that either writes nothing (rename) or writes an orphan row (enabled).
"""

from __future__ import annotations

import pytest

from openai4s.config import Config, LLMConfig
from openai4s.server import gateway as gateway_mod
from openai4s.store import get_store


class _Hub:
    def __init__(self):
        self.events = []

    def emitter(self, root_frame_id):
        def emit(event):
            event.setdefault("root_frame_id", root_frame_id)
            self.events.append(event)

        return emit

    def broadcast(self, root_frame_id, event):
        event.setdefault("root_frame_id", root_frame_id)
        self.events.append(event)


def _cfg(tmp_path):
    return Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
        max_turns=3,
    )


def _handler(cfg, runner):
    handler_cls = gateway_mod.make_handler(cfg, _Hub(), runner)
    handler = object.__new__(handler_cls)
    handler._query = lambda: {}
    seen: list[tuple[dict, int]] = []
    handler._json = lambda obj, code=200: seen.append((obj, code))
    return handler, seen


def _auth_headers(cfg) -> dict:
    """The credential `_route`'s token gate requires (`_api` skips that gate)."""
    from openai4s.server import local_auth

    return {local_auth.TOKEN_HEADER: local_auth.load_or_mint(cfg.data_dir)}


@pytest.mark.parametrize("method", ["PUT", "PATCH"])
def test_renaming_a_missing_folder_is_refused(tmp_path, method):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner)
        handler._body = lambda: {"name": "renamed"}
        # Driven through `_route` (not `_api` directly) because the refusal is
        # a raised GatewayError: only `_route` serializes it into the response
        # a client actually receives.
        handler.headers = _auth_headers(cfg)
        handler.path = "/api/v1/folders/fold_does_not_exist"
        handler._route(method)
        assert seen[-1] == ({"error": "folder not found"}, 404)
    finally:
        runner.close()


def test_renaming_an_existing_folder_still_lands(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    store = get_store(cfg.db_path)
    folder = store.create_folder(project_id="default", name="before")
    try:
        handler, seen = _handler(cfg, runner)
        handler._body = lambda: {"name": "after"}
        handler._api("PATCH", f"/folders/{folder['folder_id']}")
        assert seen[-1] == ({"ok": True}, 200)
        renamed = store.list_folders("default")
        assert [f["name"] for f in renamed] == ["after"]
    finally:
        runner.close()


@pytest.mark.parametrize("method", ["PUT", "PATCH"])
def test_enabling_a_missing_agent_is_refused(tmp_path, method):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    store = get_store(cfg.db_path)
    try:
        handler, seen = _handler(cfg, runner)
        handler._body = lambda: {"enabled": False}
        handler._api(method, "/agents/no-such-agent/enabled")
        assert seen[-1] == ({"error": "unknown agent"}, 404)
        # The refusal must happen before the write: asked to disable, yet no
        # orphan capability row exists, so the default still reads enabled.
        assert store.capability_state().is_enabled("specialist", "no-such-agent")
    finally:
        runner.close()


def test_enabling_a_known_agent_still_lands(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    store = get_store(cfg.db_path)
    name = gateway_mod._BUILTIN_AGENTS[0]["name"]
    try:
        handler, seen = _handler(cfg, runner)
        handler._body = lambda: {"enabled": False}
        handler._api("PATCH", f"/agents/{name}/enabled")
        body, code = seen[-1]
        assert code == 200
        assert body["ok"] is True and body["enabled"] is False
        assert not store.capability_state().is_enabled("specialist", name)
    finally:
        runner.close()


@pytest.mark.parametrize("method", ["PUT", "PATCH"])
def test_enabling_a_missing_connector_is_refused(tmp_path, method):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner)
        handler._body = lambda: {"enabled": True}
        handler._api(method, "/connectors/no-such-connector/enabled")
        assert seen[-1] == ({"error": "connector not found"}, 404)
    finally:
        runner.close()


def test_enabling_an_existing_connector_still_lands(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    store = get_store(cfg.db_path)
    store.upsert_connector(
        connector_id="conn-x",
        name="X",
        command=["x"],
        enabled=True,
    )
    try:
        handler, seen = _handler(cfg, runner)
        handler._body = lambda: {"enabled": False}
        handler._api("PATCH", "/connectors/conn-x/enabled")
        assert seen[-1] == ({"ok": True}, 200)
        assert store.get_connector("conn-x")["enabled"] is False
    finally:
        runner.close()
