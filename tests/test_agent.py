"""Agent loop + delegation + compaction tests, with the LLM mocked offline."""
import pytest

import openai4s.agent.compaction as comp_mod
import openai4s.agent.delegation as deleg_mod
import openai4s.agent.loop as loop_mod
from openai4s.agent import Agent
from openai4s.agent.delegation import DelegationError, DelegationRunner
from openai4s.config import get_config


class ScriptedLLM:
    """Returns queued replies in order; each call pops one."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, messages, cfg, **kw):
        self.calls.append(messages)
        content = (
            self._replies.pop(0)
            if self._replies
            else ("```python\nhost.submit_output({}, ['Finished the task'])\n```")
        )
        return {
            "content": content,
            "reasoning": None,
            "usage": {},
            "finish_reason": "stop",
            "raw": {},
        }


def test_code_as_action_cycle(monkeypatch):
    scripted = ScriptedLLM(
        [
            "Let me compute it.\n```python\nprint(6 * 7)\n```",
            "```python\nhost.submit_output({'answer': 42}, ['Computed the answer'])\n```",
        ]
    )
    monkeypatch.setattr(loop_mod, "chat", scripted)

    agent = Agent(use_skills=False, allow_delegate=False)
    result = agent.run("compute 6*7 and submit")
    # Completion is signalled through host.submit_output, not a text convention.
    assert result["stop_reason"] == "submitted"
    assert result["submitted_output"]["output"] == {"answer": 42}
    # 2 assistant turns happened
    assert len(scripted.calls) == 2


def test_no_code_block_nudge(monkeypatch):
    scripted = ScriptedLLM(
        [
            "I think the answer is 42.",  # no code -> nudge
            "```python\nhost.submit_output({'a': 1}, ['Answered the question'])\n```",
        ]
    )
    monkeypatch.setattr(loop_mod, "chat", scripted)
    result = Agent(use_skills=False, allow_delegate=False).run("hi")
    assert result["stop_reason"] == "submitted"


def test_max_turns_stop(monkeypatch):
    # never calls submit_output -> should stop at max_turns
    scripted = ScriptedLLM(["```python\nx = 1\n```"] * 10)
    monkeypatch.setattr(loop_mod, "chat", scripted)
    agent = Agent(use_skills=False, allow_delegate=False, max_turns=3)
    result = agent.run("loop forever")
    assert result["stop_reason"] == "max_turns"


# ---- compaction ----------------------------------------------------------


def test_estimate_tokens_monotonic():
    small = [{"role": "user", "content": "x"}]
    big = [{"role": "user", "content": "x" * 4000}]
    assert comp_mod.estimate_tokens(big) > comp_mod.estimate_tokens(small)


def test_should_compact_uses_window(monkeypatch):
    cfg = get_config()
    # ~1000 tokens of content
    msgs = [{"role": "user", "content": "x" * 4000}] * 10
    # Tiny window -> should compact; huge window -> should not.
    monkeypatch.setattr(cfg, "context_window_tokens", 100)
    monkeypatch.setattr(cfg, "compaction_trigger_ratio", 0.75)
    assert comp_mod.should_compact(msgs, cfg) is True
    monkeypatch.setattr(cfg, "context_window_tokens", 10_000_000)
    assert comp_mod.should_compact(msgs, cfg) is False


def test_compact_shrinks_and_preserves_head(monkeypatch):
    monkeypatch.setattr(comp_mod, "chat", ScriptedLLM(["SUMMARY TEXT"]))
    msgs = (
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        + [{"role": "assistant", "content": f"a{i}"} for i in range(6)]
        + [{"role": "user", "content": f"o{i}"} for i in range(6)]
    )
    out = comp_mod.compact(msgs, get_config(), keep_recent=4)
    assert len(out) < len(msgs)
    assert out[0]["content"] == "sys"  # system preserved
    assert out[1]["content"] == "task"  # original task preserved
    assert "SUMMARY TEXT" in out[2]["content"]  # summary injected
    assert out[-1]["content"] == "o5"  # most recent kept verbatim


# ---- delegation ----------------------------------------------------------


def test_delegate_fanout_cap():
    runner = DelegationRunner(get_config())
    with pytest.raises(DelegationError):
        runner({"request": ["t"] * (deleg_mod.FANOUT_CAP + 1)})


def test_delegate_single_and_list(monkeypatch):
    # Stub the leaf Agent.run so no real LLM/kernel is used.
    def fake_run(self, task):
        return {
            "stop_reason": "final",
            "submitted_output": {
                "output": {"echo": task},
                "completion_bullets": ["ok"],
            },
            "final_message": "FINAL",
        }

    monkeypatch.setattr(loop_mod.Agent, "run", fake_run)

    runner = DelegationRunner(get_config())
    one = runner({"request": "do X"})
    assert isinstance(one, dict)
    assert one["output"] == {"echo": "do X"}

    many = runner({"request": ["A", "B", "C"]})
    assert isinstance(many, list) and len(many) == 3
    assert {m["output"]["echo"] for m in many} == {"A", "B", "C"}


def test_delegate_session_cap(monkeypatch):
    def fake_run(self, task):
        return {"stop_reason": "final", "submitted_output": None, "final_message": None}

    monkeypatch.setattr(loop_mod.Agent, "run", fake_run)

    runner = DelegationRunner(get_config())
    runner._spawned = deleg_mod.SESSION_CAP  # pretend we're at the cap
    with pytest.raises(DelegationError):
        runner({"request": "one more"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
