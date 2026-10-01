"""Real kernel worker ``host.judge`` RPC and dispatcher gating decisions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Mapping

import pytest

from openai4s.config import Config, ExperimentalJudgmentFlags, LLMConfig
from openai4s.host_dispatch import GATEABLE_TOOLS, HostDispatcher
from openai4s.judgment.port import BackendError
from openai4s.judgment.registry import PROBE_TEMPLATE_ID
from openai4s.judgment.types import Answer, BackendReply, Question
from openai4s.kernel import Kernel
from openai4s.sdk.judgment import judge as sdk_judge


class FakeBackend:
    def evaluate(
        self,
        *,
        state: object,
        questions: Mapping[str, Question],
        model: str,
        timeout: float,
    ) -> BackendReply:
        answers = {qid: Answer(kind="noul", value=0.88) for qid in questions}
        return BackendReply(
            answers=answers,
            usage={"input_tokens": 4, "output_tokens": 0},
            model=model,
        )


class BoomBackend:
    def evaluate(self, **_kwargs: Any) -> BackendReply:
        raise BackendError("timeout", "boom")


class ExplodingBackend:
    def evaluate(self, **_kwargs: Any) -> BackendReply:
        raise RuntimeError("judgment backend exploded")


@pytest.fixture(autouse=True)
def _clear_judgment_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI4S_EXPERIMENTAL_JUDGMENT", raising=False)
    monkeypatch.delenv("OPENAI4S_JUDGMENT_SKILL_SUGGEST", raising=False)


def _dispatcher(tmp_path: Any, backend: Any) -> HostDispatcher:
    cfg = Config(
        data_dir=tmp_path / ".data",
        llm=LLMConfig(provider="deepseek", api_key="test-only"),
        experimental_judgment=ExperimentalJudgmentFlags(master=True),
    )
    dispatcher = HostDispatcher(cfg, workspace=tmp_path)
    dispatcher._judgment_service.backend_factory = lambda: backend
    return dispatcher


def test_judge_is_not_gateable_or_screened() -> None:
    assert "judge" not in GATEABLE_TOOLS
    assert "judge" not in HostDispatcher._SCREENED_METHODS


def test_sdk_rejects_direct_questions_list() -> None:
    with pytest.raises(ValueError, match="questions list"):
        sdk_judge(
            lambda _m, _a: None, "system.probe", {"text": "hi"}, questions={"q": {}}
        )


def test_kernel_host_judge_system_probe(tmp_path: Any) -> None:
    dispatcher = _dispatcher(tmp_path, FakeBackend())
    with Kernel(dispatcher=dispatcher, cwd=str(tmp_path)) as kernel:
        result = kernel.execute(
            "out = host.judge('system.probe', {'text': 'hi'})\n"
            "print(out['status'])\n"
            "print(out['template_id'])\n"
            "print(out['purpose'])"
        )
    assert result["error"] is None, result
    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    assert lines[0] == "ok"
    assert lines[1] == PROBE_TEMPLATE_ID
    assert lines[2] == "probe"


def test_kernel_unknown_template_is_runtime_error(tmp_path: Any) -> None:
    dispatcher = _dispatcher(tmp_path, FakeBackend())
    with Kernel(dispatcher=dispatcher, cwd=str(tmp_path)) as kernel:
        result = kernel.execute(
            "try:\n"
            "    host.judge('no.such.template', {'text': 'x'})\n"
            "    print('not-raised')\n"
            "except RuntimeError as exc:\n"
            "    print('caught')\n"
            "    print('unknown template' in str(exc))"
        )
    assert result["error"] is None, result
    stdout = result["stdout"]
    assert "caught" in stdout
    assert "True" in stdout
    assert "not-raised" not in stdout


def test_kernel_unavailable_is_normal_return(tmp_path: Any) -> None:
    dispatcher = _dispatcher(tmp_path, BoomBackend())
    with Kernel(dispatcher=dispatcher, cwd=str(tmp_path)) as kernel:
        result = kernel.execute(
            "out = host.judge('system.probe', {'text': 'hi'})\n"
            "print(out['status'])\n"
            "print(out['error_code'])"
        )
    assert result["error"] is None, result
    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    assert lines[0] == "unavailable"
    assert lines[1] == "timeout"


def test_kernel_disabled_is_normal_return(tmp_path: Any) -> None:
    cfg = Config(
        data_dir=tmp_path / ".data",
        llm=LLMConfig(provider="deepseek", api_key="test-only"),
        experimental_judgment=ExperimentalJudgmentFlags(master=False),
    )
    dispatcher = HostDispatcher(cfg, workspace=tmp_path)
    backend = FakeBackend()
    dispatcher._judgment_service.backend_factory = lambda: backend
    with Kernel(dispatcher=dispatcher, cwd=str(tmp_path)) as kernel:
        result = kernel.execute(
            "out = host.judge('system.probe', {'text': 'hi'})\n" "print(out['status'])"
        )
    assert result["error"] is None, result
    assert result["stdout"].strip() == "disabled"


def _sentinels() -> tuple[str, str, dict[str, Any]]:
    near = f"SENTINEL-02-{uuid.uuid4()}"
    far = f"SENTINEL-02-{uuid.uuid4()}"
    state = {"nest": {"secret": near, "pad": "p" * 600, "tail": far}}
    dumped = json.dumps(
        [{"template": "system.probe", "state": state}], ensure_ascii=False
    )
    assert dumped.find(near) < 500
    assert dumped.find(far) > 500
    return near, far, state


def _judge_rows(db_path: Any) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT args_preview, ok, result_digest FROM host_call_log "
            "WHERE method = 'judge' ORDER BY created_at"
        ).fetchall()


def _assert_previews_hide(rows: list[tuple], *secrets: str) -> None:
    assert rows, "judge wrote no host_call_log row"
    for preview, _ok, digest in rows:
        assert digest and len(digest) == 64
        for secret in secrets:
            assert secret not in (preview or "")


def test_kernel_judge_audit_omits_state_on_every_path(tmp_path: Any) -> None:
    """Success, unknown-template soft failure, and a raising backend.

    The service turns a backend exception into ``unavailable``. Each path
    still writes a host_call_log row, and the preview omits the state.
    ``audit_raw_state`` stays off, which is the default.
    """

    near, far, state = _sentinels()
    param = f"SENTINEL-02-{uuid.uuid4()}"
    dispatcher = _dispatcher(tmp_path / "ok", FakeBackend())
    assert dispatcher.store.get_setting("experimental.judgment.audit_raw_state") is None
    cell_state = f"state = {state!r}\n"
    with Kernel(dispatcher=dispatcher, cwd=str(tmp_path)) as kernel:
        success = kernel.execute(
            cell_state
            + "out = host.judge('system.probe', state)\n"
            + "print(out['status'])\n"
            + "print(out['template_id'])\n"
            + f"out2 = host.judge('system.probe', state, specs={param!r})\n"
            + "print(out2['status'])\n"
            + "try:\n"
            + "    host.judge('no.such.template', state)\n"
            + "    print('not-raised')\n"
            + "except RuntimeError as exc:\n"
            + "    print('caught')\n"
            + "    print('unknown template' in str(exc))\n"
        )
    assert success["error"] is None, success
    stdout = success["stdout"]
    assert "not-raised" not in stdout
    assert "caught" in stdout
    assert "True" in stdout
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines[0] == "ok"
    assert lines[1] == PROBE_TEMPLATE_ID
    assert lines[2] == "ok"
    rows = _judge_rows(dispatcher.store.db_path)
    assert len(rows) == 3
    _assert_previews_hide(rows[:2], near, far, param)
    # The unknown-template soft failure repeats the caller's id in its error
    # text, so that row records no digest of it.
    assert rows[2][2] is None
    for secret in (near, far, param):
        assert secret not in (rows[2][0] or "")
    assert [row[1] for row in rows] == [1, 1, 0]
    assert rows[0][0] == json.dumps(
        [{"template": "system.probe", "state": "<redacted judge state>"}],
        ensure_ascii=False,
    )
    assert rows[1][0] == json.dumps(
        [
            {
                "template": "system.probe",
                "state": "<redacted judge state>",
                "params": "<redacted judge params>",
            }
        ],
        ensure_ascii=False,
    )
    assert rows[2][0] == json.dumps(
        [{"template": "<unknown template>", "state": "<redacted judge state>"}],
        ensure_ascii=False,
    )
    assert "no.such.template" not in rows[2][0]

    boom = _dispatcher(tmp_path / "boom", ExplodingBackend())
    with Kernel(dispatcher=boom, cwd=str(tmp_path)) as kernel:
        failed = kernel.execute(
            f"state = {state!r}\n"
            "out = host.judge('system.probe', state)\n"
            "print(out['status'])\n"
            "print(out['error_code'])\n"
            "print(out['template_id'])\n"
        )
    assert failed["error"] is None, failed
    failed_lines = [line for line in failed["stdout"].splitlines() if line.strip()]
    assert failed_lines == ["unavailable", "unavailable", PROBE_TEMPLATE_ID]
    boom_rows = _judge_rows(boom.store.db_path)
    _assert_previews_hide(boom_rows, near, far)
    assert boom_rows[0][1] == 1
    assert boom_rows[0][0] == rows[0][0]


def test_dispatcher_exception_path_redacts_judge_args(tmp_path: Any) -> None:
    """A handler that raises still writes the projected preview."""

    near, far, state = _sentinels()
    dispatcher = _dispatcher(tmp_path, FakeBackend())

    def explode(_spec: dict) -> Any:
        raise RuntimeError("judge handler exploded")

    dispatcher._m_judge = explode
    with pytest.raises(RuntimeError, match="exploded"):
        dispatcher("judge", [{"template": "system.probe", "state": state}])
    rows = _judge_rows(dispatcher.store.db_path)
    _assert_previews_hide(rows, near, far)
    assert rows[0][1] == 0
    assert rows[0][0] == json.dumps(
        [{"template": "system.probe", "state": "<redacted judge state>"}],
        ensure_ascii=False,
    )


def test_audit_raw_state_does_not_restore_judge_args(
    tmp_path: Any, monkeypatch
) -> None:
    """The named event may carry state. The generic RPC preview does not."""

    events: list[tuple[str, dict]] = []

    def capture(event: str, **fields: Any) -> dict:
        events.append((event, fields))
        return {}

    monkeypatch.setattr("openai4s.observability.log_event", capture)
    near, far, state = _sentinels()
    dispatcher = _dispatcher(tmp_path, FakeBackend())
    dispatcher.store.set_setting("experimental.judgment.audit_raw_state", "true")
    with Kernel(dispatcher=dispatcher, cwd=str(tmp_path)) as kernel:
        result = kernel.execute(
            f"state = {state!r}\n"
            "out = host.judge('literature.screen', state)\n"
            "print(out['status'])\n"
        )
    assert result["error"] is None, result
    assert result["stdout"].strip() == "disabled"
    rows = _judge_rows(dispatcher.store.db_path)
    _assert_previews_hide(rows, near, far)
    assert rows[0][0] == json.dumps(
        [{"template": "literature.screen", "state": "<redacted judge state>"}],
        ensure_ascii=False,
    )
    judgment = [fields for name, fields in events if name == "judgment"]
    assert judgment
    assert judgment[0]["state"]["nest"]["secret"] == near
