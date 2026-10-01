"""Where a member's own LLM key may be sent (team mode, M4-1).

`tests/test_user_llm_keys.py` is about *whose* key a session runs on. This file
is about *where* it goes. The row names a provider and nothing else, and it was
swapped in by that name alone -- after `ModelProfileService.credential` had
refused to send a profile or environment key anywhere it was not entered for,
and had checked a keyless local server before any inherited key. So a member's
OpenAI key went to whatever `base_url` the session was pinned to: plain http to
a LAN Ollama, or an admin's third-party proxy.

The rule (`ModelProfileService.user_key_applies`, asked by `_llm_cfg` only): a
member's key goes to the endpoint the provider registry names for that provider,
over https, and never to a local host. Anywhere else the configuration's own
credential goes, exactly as for a member with no key of their own.

Every withheld case has a control beside it -- the same setup at the provider's
own endpoint, where the key *is* used -- because "a member's key is never used"
would satisfy every negative assertion here as well.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openai4s import llm
from openai4s.config import LLMConfig
from openai4s.server.errors import GatewayError
from openai4s.server.model_profiles import CREDENTIAL_LOCAL, ModelProfileService
from tests.test_team_auth_routes import (  # noqa: F401  (fixture reuse)
    _fast_pbkdf2,
    _login,
    _speak,
    _TeamDaemon,
)

MEMBER_KEY = "sk-member-own-openai"
GROUP_KEY = "sk-group-shared"
LAN = "http://192.168.1.20:11434/v1"
OPENAI = "https://api.openai.com/v1"


@pytest.fixture()
def daemon(tmp_path: Path):
    node = _TeamDaemon(tmp_path)
    node.seed_user("root", "fake-pw-r", role="admin")
    node.seed_user("alice", "fake-pw-a")
    node.seed_user("bob", "fake-pw-b")
    try:
        yield node
    finally:
        node.close()


@pytest.fixture()
def custom_provider():
    """Register providers for one test and take them away again."""
    registered: list[str] = []

    def register(name: str, base_url: str) -> None:
        llm.register_provider(
            name, wire="openai", base_url=base_url, model="lab-model", replace=True
        )
        registered.append(name)

    yield register
    for name in registered:
        llm.unregister_provider(name)


class _State:
    """Just enough SessionState for `_llm_cfg` and the Reviewer's config."""

    def __init__(self, root_frame_id: str):
        self.root_frame_id = root_frame_id
        self.model = None
        self.frozen_model_binding = None


def _use(daemon, **fields) -> None:
    """Make this the daemon's own configuration, the one an unpinned turn uses."""
    daemon.runner.cfg = replace(
        daemon.runner.cfg, llm=LLMConfig(**{"api_key": GROUP_KEY, **fields})
    )


def _own_key(daemon, username: str, provider: str, key: str = MEMBER_KEY) -> str:
    """Set a member's own key through the real self-service route."""
    cookie = _login(daemon, username, f"fake-pw-{username[0]}")
    payload = json.dumps({"provider": provider, "api_key": key}).encode("utf-8")
    head = (
        "\r\n".join(
            [
                "PUT /api/v1/auth/me/llm-key HTTP/1.1",
                f"Host: 127.0.0.1:{daemon.port}",
                "Content-Type: application/json",
                f"Content-Length: {len(payload)}",
                f"Cookie: {cookie}",
                "Connection: close",
            ]
        )
        + "\r\n\r\n"
    ).encode("ascii")
    status, raw = _speak(daemon.port, head + payload)
    assert status == 200, raw.split(b"\r\n\r\n", 1)[-1][:400]
    return cookie


def _session(daemon, username: str) -> str:
    user = daemon.store.team.get_user_by_username(username)
    frame_id = daemon.store.new_frame(kind="turn", project_id="p")
    daemon.store.team.set_session_owner(frame_id, user["id"])
    return frame_id


def _service(daemon) -> ModelProfileService:
    return ModelProfileService(
        daemon.store, daemon.runner.cfg, providers=lambda: llm.PROVIDERS
    )


def _profile(daemon, *, base_url: str, model: str, api_key: str = "") -> dict:
    name = f"{model}@{base_url or 'default'}"
    _service(daemon).create(
        {
            "name": name,
            "provider": "chatgpt",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        }
    )
    return next(p for p in daemon.store.list_model_profiles() if p["name"] == name)


def _pin(daemon, frame_id: str, profile: dict) -> None:
    """Pin the session the way a send does: to the active profile."""
    daemon.store.set_setting("active_model_profile", profile["id"])
    binding = daemon.runner.bind_model_revision(frame_id)
    assert binding["model_profile_id"] == profile["id"], binding


def _cfg_for(daemon, frame_id: str):
    return daemon.runner._llm_cfg(_State(frame_id))


# -- the provider's own endpoint: the control every case below is set against --


def test_the_providers_own_endpoint_gets_the_members_key(daemon):
    _use(daemon, provider="chatgpt")
    _own_key(daemon, "alice", "chatgpt")
    cfg = _cfg_for(daemon, _session(daemon, "alice"))
    assert cfg.base_url == OPENAI
    assert cfg.api_key == MEMBER_KEY


@pytest.mark.parametrize("spelling", [OPENAI + "/", "https://API.OpenAI.com/v1"])
def test_another_spelling_of_that_endpoint_is_still_that_endpoint(daemon, spelling):
    """Compared normalised, so the rule does not withhold a key from the very
    endpoint it was entered for over a trailing slash or the host's case."""
    _use(daemon, provider="chatgpt", base_url=spelling)
    _own_key(daemon, "alice", "chatgpt")
    assert _cfg_for(daemon, _session(daemon, "alice")).api_key == MEMBER_KEY


def test_a_pin_to_the_providers_own_endpoint_still_uses_the_members_key(daemon):
    """The pinned path's control: pinning is not what withholds the key."""
    _own_key(daemon, "alice", "chatgpt")
    session = _session(daemon, "alice")
    _pin(daemon, session, _profile(daemon, base_url="", model="gpt-5", api_key="pk"))
    cfg = _cfg_for(daemon, session)
    assert cfg.base_url == OPENAI
    assert cfg.api_key == MEMBER_KEY


# -- a local endpoint: never ---------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        LAN,
        "http://127.0.0.1:11434/v1",
        "http://localhost:1234/v1",
        "http://lab-gpu.local:8000/v1",
        "https://10.0.0.7:8443/v1",
    ],
)
def test_a_keyless_local_pin_never_receives_a_members_key(daemon, endpoint):
    """The case that was one click away: credential source `local`, and the
    member's cloud key went to it -- over plain http, for most of these."""
    _own_key(daemon, "alice", "chatgpt")
    session = _session(daemon, "alice")
    profile = _profile(daemon, base_url=endpoint, model="llama3.1")
    assert _service(daemon).credential(profile).source == CREDENTIAL_LOCAL
    _pin(daemon, session, profile)

    cfg = _cfg_for(daemon, session)
    assert cfg.base_url == endpoint
    assert cfg.api_key == "", "a keyless local server was handed a cloud key"


def test_a_local_pin_with_a_key_of_its_own_gets_that_key_not_the_members(daemon):
    """The rule is about the endpoint, not the credential source. A LAN server
    that wants a key has one saved on its profile (source `profile`); the
    member's OpenAI key is still not for it."""
    _own_key(daemon, "alice", "chatgpt")
    session = _session(daemon, "alice")
    _pin(daemon, session, _profile(daemon, base_url=LAN, model="m", api_key="lan-k"))
    assert _cfg_for(daemon, session).api_key == "lan-k"


# -- any other endpoint: the configuration's own credential --------------------


def test_an_admins_https_proxy_gets_the_proxys_key_not_the_members(daemon):
    _own_key(daemon, "alice", "chatgpt")
    session = _session(daemon, "alice")
    proxy = "https://openai-proxy.example/v1"
    _pin(daemon, session, _profile(daemon, base_url=proxy, model="m", api_key="px"))
    cfg = _cfg_for(daemon, session)
    assert cfg.base_url == proxy
    assert cfg.api_key == "px"


def test_an_operator_gateway_named_for_the_provider_is_not_its_endpoint(
    daemon, monkeypatch
):
    """`OPENAI4S_CHATGPT_BASE_URL` is the operator's choice of host, made for
    every user. The member's key was entered for OpenAI."""
    monkeypatch.setenv("OPENAI4S_CHATGPT_BASE_URL", "https://llm-gateway.example/v1")
    _use(daemon, provider="chatgpt")
    _own_key(daemon, "alice", "chatgpt")
    cfg = _cfg_for(daemon, _session(daemon, "alice"))
    assert cfg.base_url == "https://llm-gateway.example/v1"
    assert cfg.api_key == GROUP_KEY


def test_a_pin_carries_the_members_key_to_its_own_endpoint_and_no_further(daemon):
    """Two sessions of one member, one key: it follows the endpoint, not her."""
    _own_key(daemon, "alice", "chatgpt")
    hosted, local = _session(daemon, "alice"), _session(daemon, "alice")
    _pin(daemon, hosted, _profile(daemon, base_url=OPENAI, model="gpt-5", api_key="pk"))
    _pin(daemon, local, _profile(daemon, base_url=LAN, model="llama3.1"))
    assert _cfg_for(daemon, hosted).api_key == MEMBER_KEY
    assert _cfg_for(daemon, local).api_key == ""


# -- providers the registry defines --------------------------------------------


def test_a_custom_provider_at_its_registered_https_endpoint_gets_the_key(
    daemon, custom_provider
):
    """The control for the two below: a registered provider's own endpoint is
    what its name means, so a key entered under that name is for it."""
    custom_provider("lab-openai", "https://llm.lab.example/v1")
    _use(daemon, provider="lab-openai")
    _own_key(daemon, "alice", "lab-openai")
    assert _cfg_for(daemon, _session(daemon, "alice")).api_key == MEMBER_KEY


@pytest.mark.parametrize(
    "name, endpoint",
    [
        # https, its own endpoint -- withheld only because the host is private
        ("lab-private", "https://10.20.30.40:8443/v1"),
        # not local by the rule, its own endpoint -- withheld only for plain http
        ("lab-cleartext", "http://llm.lab.example:8000/v1"),
    ],
)
def test_a_custom_provider_at_a_local_or_cleartext_endpoint_does_not(
    daemon, custom_provider, name, endpoint
):
    custom_provider(name, endpoint)
    _use(daemon, provider=name)
    _own_key(daemon, "alice", name)
    cfg = _cfg_for(daemon, _session(daemon, "alice"))
    assert cfg.base_url == endpoint, "the request does go to the provider's own"
    assert cfg.api_key == GROUP_KEY


def test_an_unregistered_provider_has_no_endpoint_a_key_was_entered_for(daemon):
    """Fail closed. `chat()` refuses such a provider too, but the rule does not
    lean on that: it withholds a key whenever it cannot name the endpoint."""
    assert "deepseek" not in llm.PROVIDERS
    _use(daemon, provider="deepseek", base_url="https://api.deepseek.example/v1")
    _own_key(daemon, "alice", "deepseek")
    assert _cfg_for(daemon, _session(daemon, "alice")).api_key == GROUP_KEY


# -- refusal still means "we cannot honour what you asked for" -----------------


def test_an_unreadable_key_refuses_only_where_it_would_have_been_sent(daemon):
    """Scope is judged before the row or the secret is read. Where the key
    would never go, a broken slot is not a reason to refuse the turn; where it
    would, the existing refusal stands."""
    _use(daemon, provider="chatgpt")
    user_id = daemon.store.team.get_user_by_username("alice")["id"]
    daemon.store.user_keys.set_ref(user_id, "chatgpt", "secret:v2:gone/llm-user/x")
    hosted, local = _session(daemon, "alice"), _session(daemon, "alice")
    _pin(daemon, local, _profile(daemon, base_url=LAN, model="llama3.1"))

    assert _cfg_for(daemon, local).api_key == ""
    with pytest.raises(GatewayError) as caught:
        _cfg_for(daemon, hosted)
    assert caught.value.error_code == "user_key_unreadable"


# -- the request itself --------------------------------------------------------


def test_the_request_to_a_lan_server_carries_no_members_key(daemon, monkeypatch):
    """The config object is a proxy for what goes on the wire; this is the wire.
    `_post_json` is the facade's transport seam, so nothing leaves the process."""
    sent: list[tuple[str, dict]] = []

    def capture(url, payload, headers, timeout, **kw):
        sent.append((url, dict(headers)))
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(llm, "_post_json", capture)
    _own_key(daemon, "alice", "chatgpt")
    hosted, local = _session(daemon, "alice"), _session(daemon, "alice")
    _pin(daemon, hosted, _profile(daemon, base_url="", model="gpt-5", api_key="pk"))
    _pin(daemon, local, _profile(daemon, base_url=LAN, model="llama3.1"))

    for session in (hosted, local):
        llm.chat([{"role": "user", "content": "hi"}], _cfg_for(daemon, session))

    (hosted_url, hosted_headers), (lan_url, lan_headers) = sent
    assert hosted_url.startswith(OPENAI)
    assert hosted_headers.get("Authorization") == f"Bearer {MEMBER_KEY}"
    assert lan_url.startswith(LAN)
    assert not any(MEMBER_KEY in str(value) for value in lan_headers.values())
    assert "Authorization" not in lan_headers


# -- the Reviewer derives its own configuration from `_llm_cfg`'s --------------


def test_the_reviewer_does_not_carry_a_members_key_to_another_endpoint(daemon):
    """A session's Reviewer model is a member's own setting
    (`PUT /frames/{id}/review-settings`). Moving to a same-provider profile at
    another endpoint used to keep the key `_llm_cfg` chose for the agent's --
    so the rule above held for the turn and not for its review."""
    _use(daemon, provider="chatgpt")
    _own_key(daemon, "alice", "chatgpt")
    session = _session(daemon, "alice")
    _profile(daemon, base_url=LAN, model="llama3.1")
    _profile(daemon, base_url=OPENAI, model="gpt-5-mini")
    assert _cfg_for(daemon, session).api_key == MEMBER_KEY

    daemon.store.set_setting(f"review:model:{session}", "llama3.1")
    moved = daemon.runner._review_llm_cfg(_State(session))
    assert moved.base_url == LAN, "the Reviewer did move to the LAN profile"
    assert moved.api_key == ""

    # Control: a Reviewer on the same endpoint is still the member's to pay for.
    daemon.store.set_setting(f"review:model:{session}", "gpt-5-mini")
    stayed = daemon.runner._review_llm_cfg(_State(session))
    assert stayed.base_url == OPENAI and stayed.model == "gpt-5-mini"
    assert stayed.api_key == MEMBER_KEY


def test_nor_does_it_carry_the_groups_key_there(daemon):
    """The same move transplanted whatever key the agent ran on, the group's
    included. The profile's endpoint gets the profile's credential."""
    _use(daemon, provider="chatgpt")
    session = _session(daemon, "bob")
    _profile(daemon, base_url=LAN, model="llama3.1")
    assert _cfg_for(daemon, session).api_key == GROUP_KEY

    daemon.store.set_setting(f"review:model:{session}", "llama3.1")
    assert daemon.runner._review_llm_cfg(_State(session)).api_key == ""
