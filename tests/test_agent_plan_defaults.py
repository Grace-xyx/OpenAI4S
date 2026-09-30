"""An active Ark Agent Plan becomes DataPro's and Doubao Search's default.

Setting up an Agent Plan model used to authorize the two managed products only
while that model stayed selected: the key was reused at read time and never
saved for them, so after a switch to another provider both went dark until the
same key was pasted into their cards.  These tests pin the one-time adoption,
where it happens (every model activation path and daemon start), what it
refuses to touch (Coding Plan, platform and foreign keys; a later user choice),
and that removing the Volcengine configuration removes the adopted copy.
"""

from __future__ import annotations

import json

import pytest

from openai4s import datapro
from openai4s.config import Config, LLMConfig
from openai4s.server import gateway as gateway_mod
from openai4s.server.volcengine_connector import ProvisioningMaterial
from openai4s.store import get_store

PLAN_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
CODING_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
PLATFORM_URL = "https://ark.cn-beijing.volces.com/api/v3"
PLAN_KEY = "agent-plan-model-key-canary"
OTHER_KEY = "coding-plan-model-key-canary"


class _Hub:
    def emitter(self, root_frame_id):
        return lambda event: None

    def broadcast(self, root_frame_id, event):
        return None


class _Manager:
    def __init__(self):
        self.disconnects = []

    def disconnect(self, connector_id, cache_scope=None):
        self.disconnects.append((connector_id, cache_scope))


def _cfg(tmp_path):
    return Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
        max_turns=3,
    )


def _live(store, provider, base_url, key):
    store.set_setting("llm_provider", provider)
    store.set_setting("llm_base_url", base_url)
    store.set_secret_setting("llm_api_key", key, scope="llm")


# --- datapro: which endpoint is an Agent Plan, and what adoption writes ------


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("", True),  # the ark provider default is the Agent Plan gateway
        (PLAN_URL, True),
        (PLAN_URL + "/", True),
        ("https://ark.cn-shanghai.volces.com/api/plan/v3", True),
        (CODING_URL, False),
        (PLATFORM_URL, False),
        ("https://gateway.example.com/api/plan/v3", False),
        ("https://ark.cn-beijing.volces.com/api/planning/v3", False),
    ],
)
def test_agent_plan_endpoint_is_the_plan_path_on_a_volcengine_host(base_url, expected):
    assert datapro.is_agent_plan_endpoint(base_url) is expected


def test_adoption_saves_an_active_agent_plan_key_once(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    _live(store, "ark", "", PLAN_KEY)

    assert datapro.adopt_active_agent_plan_key(store) is True
    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY
    # Already adopted: nothing is written, and callers apply no defaults again.
    assert datapro.adopt_active_agent_plan_key(store) is False

    # The point of adopting: the products keep their key after a model switch.
    _live(store, "claude", "", "sk-ant-other-provider-key")
    assert datapro.resolve_agent_plan_key(store) == PLAN_KEY
    assert datapro.credential_state(store) == {
        "key_configured": True,
        "ark_key_reused": False,
    }


@pytest.mark.parametrize(
    ("provider", "base_url", "key"),
    [
        ("ark", CODING_URL, OTHER_KEY),
        ("ark", PLATFORM_URL, OTHER_KEY),
        ("ark", "https://gateway.example.com/api/plan/v3", OTHER_KEY),
        ("claude", PLAN_URL, OTHER_KEY),
        ("ark", PLAN_URL, "short"),
        ("ark", PLAN_URL, ""),
    ],
)
def test_adoption_refuses_anything_but_a_valid_agent_plan_key(
    tmp_path, provider, base_url, key
):
    store = get_store(_cfg(tmp_path).db_path)
    _live(store, provider, base_url, key)

    assert datapro.adopt_active_agent_plan_key(store) is False
    assert datapro.explicit_agent_plan_key(store) == ""


def test_a_coding_plan_model_no_longer_shadows_the_agent_plan_key(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    _live(store, "ark", CODING_URL, OTHER_KEY)
    # With nothing better saved, the Volcengine key is still tried, as before.
    assert datapro.resolve_agent_plan_key(store) == OTHER_KEY

    datapro.save_agent_plan_key(store, PLAN_KEY)

    # Saving on a card must not replace the key the Coding Plan model runs on...
    assert store.get_secret_setting("llm_api_key") == OTHER_KEY
    # ...and the dedicated Agent Plan Key is what the products now send.
    assert datapro.resolve_agent_plan_key(store) == PLAN_KEY
    assert datapro.credential_state(store) == {
        "key_configured": True,
        "ark_key_reused": False,
    }


def test_saving_the_key_still_mirrors_into_an_active_agent_plan_model(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    _live(store, "ark", PLAN_URL, "agent-plan-previous-key")

    datapro.save_agent_plan_key(store, PLAN_KEY)

    assert store.get_secret_setting("llm_api_key") == PLAN_KEY
    assert datapro.credential_state(store) == {
        "key_configured": True,
        "ark_key_reused": True,
    }


def test_a_key_the_user_saved_is_never_replaced_by_adoption(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    saved = "card-saved-agent-plan-key"
    datapro.save_agent_plan_key(store, saved)
    _live(store, "ark", PLAN_URL, PLAN_KEY)

    assert datapro.adopt_active_agent_plan_key(store) is False
    assert datapro.explicit_agent_plan_key(store) == saved
    # ...and a removal of the model's key cannot delete the user's own.
    assert datapro.forget_adopted_agent_plan_key(store, saved) is False
    assert datapro.explicit_agent_plan_key(store) == saved


def test_a_key_from_before_this_release_counts_as_the_users_own(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    legacy = "pre-origin-agent-plan-key"
    store.set_secret_setting(datapro.AGENT_PLAN_KEY_SETTING, legacy, scope="agent_plan")
    _live(store, "ark", "", PLAN_KEY)

    assert datapro.adopt_active_agent_plan_key(store) is False
    assert datapro.explicit_agent_plan_key(store) == legacy


def test_an_adopted_key_follows_the_plan_it_came_from(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    _live(store, "ark", PLAN_URL, PLAN_KEY)
    assert datapro.adopt_active_agent_plan_key(store) is True

    rotated = "agent-plan-rotated-model-key"
    store.set_secret_setting("llm_api_key", rotated, scope="llm")

    assert datapro.adopt_active_agent_plan_key(store) is True
    assert datapro.explicit_agent_plan_key(store) == rotated


def test_forget_clears_only_the_exact_adopted_key(tmp_path):
    store = get_store(_cfg(tmp_path).db_path)
    _live(store, "ark", PLAN_URL, PLAN_KEY)
    assert datapro.adopt_active_agent_plan_key(store) is True

    assert datapro.forget_adopted_agent_plan_key(store, OTHER_KEY) is False
    assert datapro.forget_adopted_agent_plan_key(store, "") is False
    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY

    assert datapro.forget_adopted_agent_plan_key(store, PLAN_KEY) is True
    assert datapro.explicit_agent_plan_key(store) == ""


# --- gateway: every activation path applies the defaults ---------------------


class _Route:
    """Drive `_api` directly, the way the other gateway route tests do."""

    def __init__(self, cfg, runner):
        self.replies = []
        self.body = {}
        handler = object.__new__(gateway_mod.make_handler(cfg, _Hub(), runner))
        handler._query = lambda: {}
        handler._body = lambda: self.body
        handler._json = lambda obj, code=200: self.replies.append((code, obj))
        self.handler = handler

    def __call__(self, method, path, body=None):
        self.body = body or {}
        self.handler._api(method, path)
        return self.replies[-1]


class _AgentPlanConnector:
    """The Ark CLI facade, answering one ready personal Agent Plan."""

    def connection(self, *, force=False):
        return {
            "installed": True,
            "state": "connected",
            "plans": [],
            "access": {"state": "ready", "plan_key": "agent-plan"},
            "login": {"state": "idle"},
            "cached": not force,
        }

    def refresh(self):
        return self.connection(force=True)

    def provisioning_material(
        self, plan_key=None, key_choice=None, endpoint_choice=None
    ):
        return ProvisioningMaterial(
            api_key=PLAN_KEY,
            plan_key="agent-plan",
            plan_name="Agent Plan",
            profile_name="agent-plan_cn-beijing",
            model="doubao-seed-2-0-pro-260215",
            region="cn-beijing",
            account_name="Alice",
        )


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    from openai4s import mcp_client

    monkeypatch.setattr(
        gateway_mod, "VolcengineConnectorService", lambda: _AgentPlanConnector()
    )
    manager = _Manager()
    monkeypatch.setattr(mcp_client, "manager", lambda: manager)
    cfg = _cfg(tmp_path)
    gateway_mod._seed_datapro_connector(cfg)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        yield cfg, get_store(cfg.db_path), _Route(cfg, runner)
    finally:
        runner.close()


def _datapro_state(call):
    status, payload = call("GET", "/datapro/config")
    assert status == 200
    return payload


def _switch_to_another_model(call):
    """Select a non-Ark model the way the UI does: activate its profile."""

    status, profile = call(
        "POST",
        "/model-profiles",
        {
            "name": "Other vendor",
            "provider": "claude",
            "model": "claude-sonnet-5-5",
            "api_key": "sk-ant-other-provider-key",
        },
    )
    assert status in (200, 201)
    assert call("POST", f"/model-profiles/{profile['id']}/activate")[0] == 200


def _switch_off_datapro(call):
    assert (
        call("PUT", "/connectors/volcengine-datapro/enabled", {"enabled": False})[0]
        == 200
    )
    assert (
        call("PUT", "/skills/catalog/volcengine-datapro/enabled", {"enabled": False})[0]
        == 200
    )
    state = _datapro_state(call)
    assert state["connector_enabled"] is False
    assert state["skill_enabled"] is False


@pytest.mark.stubbed_backend
def test_volcengine_agent_plan_setup_turns_on_both_products_for_good(gateway):
    _cfg_, store, call = gateway
    _switch_off_datapro(call)

    status, configured = call(
        "POST", "/volcengine/configure", {"plan_key": "agent-plan"}
    )

    assert status == 201
    assert PLAN_KEY not in json.dumps(configured)
    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY
    state = _datapro_state(call)
    assert state["connector_enabled"] is True
    assert state["skill_enabled"] is True
    assert state["key_configured"] is True
    status, doubao = call("GET", "/doubao-search/config")
    assert status == 200 and doubao["key_configured"] is True

    # Another model later: no second setup for either product.
    _switch_to_another_model(call)
    assert store.get_setting("llm_provider") == "claude"
    assert datapro.resolve_agent_plan_key(store) == PLAN_KEY
    assert call("GET", "/doubao-search/config")[1]["key_configured"] is True
    assert _datapro_state(call)["key_configured"] is True


@pytest.mark.stubbed_backend
def test_reactivating_an_adopted_plan_keeps_a_later_user_choice(gateway):
    _cfg_, store, call = gateway
    assert call("POST", "/volcengine/configure", {"plan_key": "agent-plan"})[0] == 201
    profile_id = store.get_setting("volcengine_model_profile_id")
    _switch_off_datapro(call)

    _switch_to_another_model(call)
    assert call("POST", f"/model-profiles/{profile_id}/activate")[0] == 200

    state = _datapro_state(call)
    assert state["connector_enabled"] is False
    assert state["skill_enabled"] is False
    assert state["key_configured"] is True


@pytest.mark.stubbed_backend
def test_disconnecting_volcengine_removes_the_adopted_copy(gateway):
    _cfg_, store, call = gateway
    assert call("POST", "/volcengine/configure", {"plan_key": "agent-plan"})[0] == 201
    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY

    status, _ = call("POST", "/volcengine/disconnect", {"confirm": True})

    assert status == 200
    assert datapro.explicit_agent_plan_key(store) == ""
    assert datapro.resolve_agent_plan_key(store) == ""


@pytest.mark.stubbed_backend
def test_disconnecting_volcengine_keeps_a_separately_saved_key(gateway):
    _cfg_, store, call = gateway
    assert call("POST", "/volcengine/configure", {"plan_key": "agent-plan"})[0] == 201
    separate = "separately-saved-agent-plan-key"
    _switch_to_another_model(call)
    assert call("POST", "/doubao-search/config", {"agent_plan_key": separate})[0] == 200

    assert call("POST", "/volcengine/disconnect", {"confirm": True})[0] == 200

    assert datapro.explicit_agent_plan_key(store) == separate


@pytest.mark.stubbed_backend
@pytest.mark.parametrize(
    ("base_url", "adopted"),
    [(PLAN_URL, True), ("", True), (CODING_URL, False), (PLATFORM_URL, False)],
)
def test_activating_a_manual_ark_profile_adopts_only_an_agent_plan(
    gateway, base_url, adopted
):
    _cfg_, store, call = gateway
    _switch_off_datapro(call)
    status, profile = call(
        "POST",
        "/model-profiles",
        {
            "name": "Ark",
            "provider": "ark",
            "base_url": base_url,
            "model": "doubao-seed-2.0-pro",
            "api_key": PLAN_KEY,
        },
    )
    assert status in (200, 201)

    assert call("POST", f"/model-profiles/{profile['id']}/activate")[0] == 200

    assert (datapro.explicit_agent_plan_key(store) == PLAN_KEY) is adopted
    state = _datapro_state(call)
    assert state["connector_enabled"] is adopted
    assert state["skill_enabled"] is adopted


@pytest.mark.stubbed_backend
def test_the_header_model_picker_adopts_an_agent_plan_profile(gateway):
    _cfg_, store, call = gateway
    status, profile = call(
        "POST",
        "/model-profiles",
        {
            "name": "Ark",
            "provider": "ark",
            "base_url": PLAN_URL,
            "model": "doubao-seed-2.0-pro",
            "api_key": PLAN_KEY,
        },
    )
    assert status in (200, 201)

    assert call("PUT", "/models/default", {"model_id": profile["id"]})[0] == 200

    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY


@pytest.mark.stubbed_backend
def test_first_run_onboarding_with_an_agent_plan_adopts_it(gateway):
    _cfg_, store, call = gateway
    # Onboarding validates the provider it replaces; the fixture's deepseek is
    # not one of the first-run choices.
    store.set_setting("llm_provider", "chatgpt")
    _switch_off_datapro(call)

    status, _ = call(
        "POST",
        "/onboarding/complete",
        {"provider": "ark", "model": "doubao-seed-2.0-pro", "api_key": PLAN_KEY},
    )

    assert status == 200
    assert store.get_setting("llm_base_url") == PLAN_URL
    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY
    assert _datapro_state(call)["connector_enabled"] is True


@pytest.mark.stubbed_backend
def test_a_daemon_start_saves_the_key_but_changes_no_switch(tmp_path, monkeypatch):
    from openai4s import mcp_client

    monkeypatch.setattr(mcp_client, "manager", lambda: _Manager())
    cfg = _cfg(tmp_path)
    gateway_mod._seed_datapro_connector(cfg)
    store = get_store(cfg.db_path)
    _live(store, "ark", PLAN_URL, PLAN_KEY)
    store.set_connector_enabled(datapro.CONNECTOR_ID, False)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        call = _Route(cfg, runner)

        assert datapro.explicit_agent_plan_key(store) == PLAN_KEY
        assert _datapro_state(call)["connector_enabled"] is False
    finally:
        runner.close()


@pytest.mark.stubbed_backend
def test_a_broker_that_refuses_the_write_does_not_fail_activation(gateway, monkeypatch):
    _cfg_, store, call = gateway
    real = store.set_secret_setting

    def _refuse_agent_plan(key, value, *, scope):
        if key == datapro.AGENT_PLAN_KEY_SETTING:
            raise RuntimeError("read-only secret backend")
        return real(key, value, scope=scope)

    monkeypatch.setattr(store, "set_secret_setting", _refuse_agent_plan)

    status, _ = call("POST", "/volcengine/configure", {"plan_key": "agent-plan"})

    assert status == 201
    assert store.get_setting("llm_provider") == "ark"
    # Read-time reuse still authorizes both products while the plan is active.
    assert datapro.resolve_agent_plan_key(store) == PLAN_KEY


def _agent_plan_profile(call, key=PLAN_KEY, name="Ark"):
    status, profile = call(
        "POST",
        "/model-profiles",
        {
            "name": name,
            "provider": "ark",
            "base_url": PLAN_URL,
            "model": "doubao-seed-2.0-pro",
            "api_key": key,
        },
    )
    assert status in (200, 201)
    return profile["id"]


@pytest.mark.stubbed_backend
def test_deleting_the_source_profile_removes_the_adopted_copy(gateway):
    _cfg_, store, call = gateway
    profile_id = _agent_plan_profile(call)
    assert call("POST", f"/model-profiles/{profile_id}/activate")[0] == 200
    _switch_to_another_model(call)
    assert datapro.resolve_agent_plan_key(store) == PLAN_KEY

    assert call("DELETE", f"/model-profiles/{profile_id}")[0] == 200

    # A profile the user removed must not leave its key behind as a copy.
    assert datapro.explicit_agent_plan_key(store) == ""
    assert datapro.resolve_agent_plan_key(store) == ""


@pytest.mark.stubbed_backend
def test_the_adopted_copy_lives_while_another_profile_holds_the_key(gateway):
    _cfg_, store, call = gateway
    first = _agent_plan_profile(call, name="Ark A")
    _agent_plan_profile(call, name="Ark B")
    assert call("POST", f"/model-profiles/{first}/activate")[0] == 200
    _switch_to_another_model(call)

    assert call("DELETE", f"/model-profiles/{first}")[0] == 200

    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY


@pytest.mark.stubbed_backend
def test_rekeying_the_source_profile_removes_the_old_copy(gateway):
    _cfg_, store, call = gateway
    profile_id = _agent_plan_profile(call)
    assert call("POST", f"/model-profiles/{profile_id}/activate")[0] == 200
    _switch_to_another_model(call)

    rotated = "agent-plan-rotated-model-key"
    status, _ = call("PATCH", f"/model-profiles/{profile_id}", {"api_key": rotated})
    assert status == 200
    assert datapro.explicit_agent_plan_key(store) == ""

    # The next activation adopts the new key.
    assert call("POST", f"/model-profiles/{profile_id}/activate")[0] == 200
    assert datapro.explicit_agent_plan_key(store) == rotated


@pytest.mark.stubbed_backend
def test_a_card_saved_key_survives_its_twin_profiles_deletion(gateway):
    _cfg_, store, call = gateway
    profile_id = _agent_plan_profile(call)
    assert call("POST", f"/model-profiles/{profile_id}/activate")[0] == 200
    # The user also pastes the same key on the Doubao card: now it is theirs.
    assert call("POST", "/doubao-search/config", {"agent_plan_key": PLAN_KEY})[0] == 200
    _switch_to_another_model(call)

    assert call("DELETE", f"/model-profiles/{profile_id}")[0] == 200

    assert datapro.explicit_agent_plan_key(store) == PLAN_KEY
