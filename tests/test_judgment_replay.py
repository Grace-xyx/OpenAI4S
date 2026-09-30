"""host.judge is taped by the existing recorder and replays with zero network."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Mapping

import pytest

from openai4s.config import Config, ExperimentalJudgmentFlags, LLMConfig
from openai4s.host_dispatch import HostDispatcher
from openai4s.judgment.registry import PROBE_TEMPLATE_ID
from openai4s.judgment.types import Answer, BackendReply, Question
from openai4s.replay import _TAPE_EXCLUDE, TapeRecorder, _OpenAI4SReplay


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        *,
        state: object,
        questions: Mapping[str, Question],
        model: str,
        timeout: float,
    ) -> BackendReply:
        self.calls += 1
        return BackendReply(
            answers={qid: Answer(kind="noul", value=0.77) for qid in questions},
            usage={"input_tokens": 3, "output_tokens": 0},
            model=model,
        )


def test_judge_is_not_tape_excluded() -> None:
    assert "judge" not in _TAPE_EXCLUDE


def test_recorder_tapes_judge_and_replay_uses_tape(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    urlopen_calls: list[Any] = []

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        urlopen_calls.append((_args, _kwargs))
        raise AssertionError("urlopen must not run during judgment replay")

    monkeypatch.setattr("urllib.request.urlopen", _blocked)

    cfg = Config(
        data_dir=tmp_path / ".data",
        llm=LLMConfig(provider="deepseek", api_key="test-only"),
        experimental_judgment=ExperimentalJudgmentFlags(master=True),
    )
    dispatcher = HostDispatcher(cfg, workspace=tmp_path)
    backend = FakeBackend()
    dispatcher._judgment_service.backend_factory = lambda: backend
    recorder = TapeRecorder(tmp_path / "openai4s_tape.json")
    dispatcher.recorder = recorder

    sentinel = f"SENTINEL-02-{uuid.uuid4()}"
    spec = {"template": PROBE_TEMPLATE_ID, "state": {"text": "hi", "secret": sentinel}}
    live = dispatcher("judge", [spec])
    assert live["status"] == "ok"
    assert live["template_id"] == PROBE_TEMPLATE_ID
    assert backend.calls == 1
    assert recorder.records
    assert recorder.records[0]["method"] == "judge"
    assert recorder.records[0]["result"]["status"] == "ok"
    assert recorder.records[0]["args"][0]["state"]["secret"] == sentinel

    tape_path = recorder.flush()
    replay = _OpenAI4SReplay(recorder.records)
    replayed = replay.judge(spec)
    assert replayed["status"] == "ok"
    assert replayed["answers"]["alive"]["value"] == 0.77
    assert backend.calls == 1
    assert urlopen_calls == []
    assert tape_path.is_file()
    assert sentinel in tape_path.read_text(encoding="utf-8")
    with sqlite3.connect(dispatcher.store.db_path) as conn:
        preview = conn.execute(
            "SELECT args_preview FROM host_call_log WHERE method = 'judge'"
        ).fetchone()[0]
    assert sentinel not in preview
    assert preview == json.dumps(
        [{"template": PROBE_TEMPLATE_ID, "state": "<redacted judge state>"}],
        ensure_ascii=False,
    )
