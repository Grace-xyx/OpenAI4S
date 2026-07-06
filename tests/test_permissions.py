"""Tests for the opencode-style tool-call permission gate: rule resolution
(store) + the blocking broker round-trip."""
import threading
import time

from openai4s.config import Config, LLMConfig
from openai4s.permissions import PermissionBroker, broker, suggest_patterns
from openai4s.store import get_store


def _store(tmp_path):
    cfg = Config(data_dir=tmp_path, llm=LLMConfig(provider="deepseek", api_key="k"))
    st = get_store(cfg.db_path)
    st.seed_default_permission_rules()
    return st


# --- rule resolution ------------------------------------------------------
def test_seed_defaults_and_fallback(tmp_path):
    st = _store(tmp_path)
    assert st.resolve_permission(tool="read_file", pattern_input="data.csv") == "allow"
    assert st.resolve_permission(tool="glob", pattern_input="**/*.py") == "allow"
    # gentle default: safe in-workspace / SSRF-guarded research tools allow
    assert st.resolve_permission(tool="write_file", pattern_input="out.txt") == "allow"
    assert st.resolve_permission(tool="edit_file", pattern_input="out.txt") == "allow"
    assert st.resolve_permission(tool="web_search", pattern_input="x") == "allow"
    assert st.resolve_permission(tool="env_setup", pattern_input="numpy") == "allow"
    # genuinely risky ones still ask
    assert st.resolve_permission(tool="bash", pattern_input="ls -la") == "ask"
    assert st.resolve_permission(tool="mcp_call", pattern_input="x") == "ask"
    # a tool with no rule at all falls back to ask (security-first)
    assert st.resolve_permission(tool="totally_unknown", pattern_input="x") == "ask"


def test_env_read_denied_even_over_conversation_allow(tmp_path):
    st = _store(tmp_path)
    # broad conversation allow for reads
    st.set_permission_rule(
        scope="conversation",
        scope_id="f3",
        tool="read_file",
        pattern="*",
        decision="allow",
    )
    # the more-specific global *.env deny still wins
    assert (
        st.resolve_permission(
            root_frame_id="f3", tool="read_file", pattern_input="cfg/.env"
        )
        == "deny"
    )
    # a normal read under the conversation allow is fine
    assert (
        st.resolve_permission(
            root_frame_id="f3", tool="read_file", pattern_input="cfg/data.csv"
        )
        == "allow"
    )


def test_conversation_allow_overrides_global_ask(tmp_path):
    st = _store(tmp_path)
    st.set_permission_rule(
        scope="conversation", scope_id="f1", tool="bash", pattern="*", decision="allow"
    )
    assert (
        st.resolve_permission(root_frame_id="f1", tool="bash", pattern_input="ls")
        == "allow"
    )
    # a different conversation is unaffected
    assert (
        st.resolve_permission(root_frame_id="other", tool="bash", pattern_input="ls")
        == "ask"
    )


def test_pattern_specificity(tmp_path):
    st = _store(tmp_path)
    st.set_permission_rule(
        scope="conversation",
        scope_id="f2",
        tool="bash",
        pattern="git *",
        decision="allow",
    )
    assert (
        st.resolve_permission(
            root_frame_id="f2", tool="bash", pattern_input="git push origin main"
        )
        == "allow"
    )
    # non-matching command still hits the global bash ask
    assert (
        st.resolve_permission(root_frame_id="f2", tool="bash", pattern_input="rm -rf /")
        == "ask"
    )


def test_project_scope(tmp_path):
    st = _store(tmp_path)
    # a project rule applies only within that project (use a non-default decision
    # so the isolation is observable against the gentle web_search=allow default)
    st.set_permission_rule(
        scope="project",
        scope_id="proj-x",
        tool="web_search",
        pattern="*",
        decision="deny",
    )
    assert (
        st.resolve_permission(
            project_id="proj-x", tool="web_search", pattern_input="caffeine"
        )
        == "deny"
    )
    # a different project falls back to the gentle default (web_search allow)
    assert (
        st.resolve_permission(
            project_id="proj-y", tool="web_search", pattern_input="caffeine"
        )
        == "allow"
    )


def test_upsert_and_delete_rule(tmp_path):
    st = _store(tmp_path)
    rid = st.set_permission_rule(
        scope="global", scope_id="", tool="bash", pattern="rm *", decision="deny"
    )
    assert st.resolve_permission(tool="bash", pattern_input="rm x") == "deny"
    # upsert same key flips the decision, does not duplicate
    rid2 = st.set_permission_rule(
        scope="global", scope_id="", tool="bash", pattern="rm *", decision="ask"
    )
    assert rid2 == rid
    assert st.resolve_permission(tool="bash", pattern_input="rm x") == "ask"
    st.delete_permission_rule(rid)
    # back to the seeded bash * -> ask
    assert st.resolve_permission(tool="bash", pattern_input="rm x") == "ask"


# --- broker round-trip ----------------------------------------------------
def test_broker_headless_allows_ask_but_enforces_deny(tmp_path):
    st = _store(tmp_path)
    b = PermissionBroker()
    # no UI channel registered -> ask degrades to allow (non-interactive runs)
    assert b.gate(store=st, frame_id=None, method="bash", target="ls")["allow"] is True
    # deny rules still bite even without a channel
    res = b.gate(store=st, frame_id=None, method="read_file", target="a/.env")
    assert res["allow"] is False


def test_broker_blocks_until_allowed_and_persists(tmp_path):
    st = _store(tmp_path)
    b = PermissionBroker()
    events = []
    b.register_channel("root1", lambda ev: events.append(ev))
    out = {}

    def run():
        out["res"] = b.gate(
            store=st, frame_id="root1", method="bash", target="pytest -q"
        )

    t = threading.Thread(target=run)
    t.start()
    # wait for the await_permission emit
    for _ in range(200):
        if any(e.get("type") == "await_permission" for e in events):
            break
        time.sleep(0.01)
    ask = next(e for e in events if e.get("type") == "await_permission")
    assert ask["tool"] == "bash" and ask["scopes"][0] == "once"
    assert b.resolve(
        ask["decision_id"], allow=True, scope="conversation", pattern="pytest *"
    )
    t.join(timeout=5)
    assert out["res"]["allow"] is True
    # a resolved event was emitted to clear the card
    assert any(e.get("type") == "permission_resolved" for e in events)
    # the conversation rule was persisted, so a matching call no longer asks
    assert (
        st.resolve_permission(
            root_frame_id="root1", tool="bash", pattern_input="pytest -q"
        )
        == "allow"
    )


def test_broker_deny_returns_soft_fail(tmp_path):
    st = _store(tmp_path)
    b = PermissionBroker()
    events = []
    b.register_channel("root2", lambda ev: events.append(ev))
    out = {}

    def run():
        # use a still-gated tool (bash) so the ask→deny round-trip actually prompts
        out["res"] = b.gate(
            store=st, frame_id="root2", method="bash", target="rm -rf /tmp/x"
        )

    t = threading.Thread(target=run)
    t.start()
    for _ in range(200):
        if any(e.get("type") == "await_permission" for e in events):
            break
        time.sleep(0.01)
    did = next(e for e in events if e.get("type") == "await_permission")["decision_id"]
    assert b.resolve(did, allow=False, scope="once", message="not now")
    t.join(timeout=5)
    assert out["res"]["allow"] is False
    assert "not now" in (out["res"].get("message") or "")


def test_broker_cancel_denies_pending(tmp_path):
    st = _store(tmp_path)
    b = PermissionBroker()
    events = []
    b.register_channel("root3", lambda ev: events.append(ev))
    out = {}

    def run():
        out["res"] = b.gate(
            store=st, frame_id="root3", method="bash", target="sleep 999"
        )

    t = threading.Thread(target=run)
    t.start()
    for _ in range(200):
        if any(e.get("type") == "await_permission" for e in events):
            break
        time.sleep(0.01)
    b.cancel_root("root3")
    t.join(timeout=5)
    assert out["res"]["allow"] is False


def test_suggest_patterns_generalizes():
    ps = suggest_patterns("bash", "git push origin main")
    assert ps[0] == "git push origin main"
    assert "git push *" in ps and "git *" in ps and ps[-1] == "*"
    ps2 = suggest_patterns("write_file", "results/out.csv")
    assert "results/*" in ps2 and "*.csv" in ps2


# --- end-to-end through the real HostDispatcher.__call__ ------------------
def _dispatcher(tmp_path):
    from openai4s.host_dispatch import build_dispatcher

    cfg = Config(data_dir=tmp_path, llm=LLMConfig(provider="deepseek", api_key="k"))
    st = get_store(cfg.db_path)
    st.seed_default_permission_rules()
    frame = st.new_frame(kind="turn")  # frame_id == its own root_frame_id
    disp = build_dispatcher(cfg, frame_id=frame)
    return disp, frame, st


def _wait_ask(events):
    for _ in range(300):
        for e in events:
            if e.get("type") == "await_permission":
                return e
        time.sleep(0.01)
    raise AssertionError("no await_permission emitted")


def test_dispatcher_gate_denies_bash_soft_fail(tmp_path):
    disp, frame, _ = _dispatcher(tmp_path)
    events = []
    broker().register_channel(frame, lambda ev: events.append(ev))
    try:
        out = {}
        t = threading.Thread(
            target=lambda: out.__setitem__(
                "r", disp("bash", [{"command": "echo should-not-run"}])
            )
        )
        t.start()
        ask = _wait_ask(events)
        broker().resolve(ask["decision_id"], allow=False, scope="once")
        t.join(timeout=8)
        # denied call returns the single-key soft-fail dict the worker raises
        assert set(out["r"].keys()) == {"error"}
        assert "Permission denied" in out["r"]["error"]
    finally:
        broker().unregister_channel(frame)


def test_dispatcher_gate_allows_and_runs_bash(tmp_path):
    disp, frame, _ = _dispatcher(tmp_path)
    events = []
    broker().register_channel(frame, lambda ev: events.append(ev))
    try:
        out = {}
        t = threading.Thread(
            target=lambda: out.__setitem__(
                "r", disp("bash", [{"command": "echo gate-ok"}])
            )
        )
        t.start()
        ask = _wait_ask(events)
        broker().resolve(ask["decision_id"], allow=True, scope="once")
        t.join(timeout=8)
        # allow → the real _m_bash ran and captured stdout
        assert "gate-ok" in (out["r"].get("stdout") or "")
    finally:
        broker().unregister_channel(frame)


def test_dispatcher_readonly_tool_not_gated_by_default(tmp_path):
    # glob is seeded 'allow', so a read-only tool must NOT emit a prompt.
    disp, frame, _ = _dispatcher(tmp_path)
    events = []
    broker().register_channel(frame, lambda ev: events.append(ev))
    try:
        # runs inline (no thread) — if it blocked on a prompt this would hang
        disp("glob", [{"pattern": "*.py"}])
        assert not any(e.get("type") == "await_permission" for e in events)
    finally:
        broker().unregister_channel(frame)


# --- review-fix regression tests -----------------------------------------
def test_deny_is_absolute_over_broader_scope_allow(tmp_path):
    # a conversation 'deny bash *' must beat a broader-scope specific 'allow git *'
    st = _store(tmp_path)
    st.set_permission_rule(
        scope="global", scope_id="", tool="bash", pattern="git *", decision="allow"
    )
    st.set_permission_rule(
        scope="conversation", scope_id="fD", tool="bash", pattern="*", decision="deny"
    )
    assert (
        st.resolve_permission(root_frame_id="fD", tool="bash", pattern_input="git push")
        == "deny"
    )
    # without the conversation deny, the specific global allow applies
    assert (
        st.resolve_permission(
            root_frame_id="other", tool="bash", pattern_input="git push"
        )
        == "allow"
    )


def test_exact_literal_pattern_with_metachars_matches_itself(tmp_path):
    from openai4s.store import _perm_match

    assert _perm_match("grep [a-z] file.txt", "grep [a-z] file.txt")  # exact literal
    st = _store(tmp_path)
    st.set_permission_rule(
        scope="conversation",
        scope_id="fM",
        tool="bash",
        pattern="ls a[1].txt",
        decision="allow",
    )
    assert (
        st.resolve_permission(
            root_frame_id="fM", tool="bash", pattern_input="ls a[1].txt"
        )
        == "allow"
    )


def test_reset_restores_modified_default_decision(tmp_path):
    st = _store(tmp_path)
    st.set_permission_rule(
        scope="global", scope_id="", tool="bash", pattern="*", decision="allow"
    )  # user loosens the default
    assert st.resolve_permission(tool="bash", pattern_input="rm x") == "allow"
    st.seed_default_permission_rules(force=True)  # reset
    assert st.resolve_permission(tool="bash", pattern_input="rm x") == "ask"


def test_exec_background_gate_target_is_the_code():
    from openai4s.host_dispatch import _gate_target

    assert _gate_target("exec_background", [{"code": "print(1)"}]) == "print(1)"


def test_is_secret_path_case_insensitive():
    from openai4s.host_dispatch import _is_secret_path

    assert _is_secret_path(".env") and _is_secret_path("cfg/.ENV")
    assert _is_secret_path("deploy/prod.env") and _is_secret_path("id_rsa")
    assert not _is_secret_path("notes.txt") and not _is_secret_path("main.py")


def test_secret_file_read_hard_denied_without_prompt(tmp_path):
    # read_file .env is blocked by the hard guard BEFORE the rule engine / prompt
    disp, frame, _ = _dispatcher(tmp_path)
    events = []
    broker().register_channel(frame, lambda ev: events.append(ev))
    try:
        r = disp("read_file", [{"path": "config/.ENV"}])  # case-insensitive
        assert set(r.keys()) == {"error"} and "secret" in r["error"].lower()
        assert not any(e.get("type") == "await_permission" for e in events)
    finally:
        broker().unregister_channel(frame)


def test_grep_and_glob_skip_secret_files(tmp_path):
    disp, frame, _ = _dispatcher(tmp_path)
    ws = disp._workspace()
    (ws / ".env").write_text("API_KEY=NEEDLE123\n", encoding="utf-8")
    (ws / "notes.txt").write_text("nothing here\n", encoding="utf-8")
    grep = disp("grep", [{"pattern": "NEEDLE123"}])
    assert not any(".env" in (m.get("file") or "") for m in grep.get("matches", []))
    glob = disp("glob", [{"pattern": "*"}])
    assert not any(m.endswith(".env") for m in glob.get("matches", []))
