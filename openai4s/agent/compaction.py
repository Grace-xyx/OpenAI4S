"""Context compaction.

When the running message list grows past a threshold, summarize the OLDER
turns into a single "continuation summary" message so the context stays bounded
while preserving task state.

Strategy:
  - Always keep the system message and the original user task (messages[0:2]).
  - Always keep the most recent `keep_recent` messages verbatim.
  - Summarize everything in between via host.llm-class call into one
    system-role note, and archive the raw compacted slice to disk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from openai4s.config import Config
from openai4s.llm import chat
from openai4s.prompts import SUMMARY_FORK

# Compaction is a "summary fork": a separate API call whose output the system
# reads directly, with a loud "this is not your turn" guard against injection
# from the transcript being summarized.
_SUMMARY_SYSTEM = SUMMARY_FORK


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for the whole message list.

    openai4s compacts as the context window fills, not by message count. We have
    no tokenizer in-process (pure stdlib), so approximate: ~4 chars per token,
    plus a small per-message overhead for role/formatting framing.
    """
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content) // 4 + 8
    return total


def should_compact(messages: list[dict], cfg: Config) -> bool:
    """True once the estimated prompt size crosses trigger_ratio * window."""
    budget = int(cfg.context_window_tokens * cfg.compaction_trigger_ratio)
    return estimate_tokens(messages) > budget


def compact(
    messages: list[dict],
    cfg: Config,
    *,
    keep_recent: int = 4,
    archive_dir: Path | None = None,
) -> list[dict]:
    """Return a new, shorter message list. No-op if nothing to compact."""
    # messages[0]=system, messages[1]=original user task. Keep both.
    head = messages[:2]
    tail = messages[-keep_recent:] if keep_recent > 0 else []
    middle = (
        messages[2 : len(messages) - keep_recent] if keep_recent > 0 else messages[2:]
    )
    if not middle:
        return messages

    convo = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in middle)
    summary_res = chat(
        [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": convo},
        ],
        cfg.llm,
        max_tokens=1024,
        temperature=0.2,
    )
    summary_text = summary_res.get("content", "") or "(compaction produced no summary)"

    if archive_dir is not None:
        _archive(archive_dir, middle, summary_text)

    note = {
        "role": "system",
        "content": (
            "[compacted history — earlier turns were summarized to save "
            "context; the kernel namespace still holds all prior "
            "variables]\n\n" + summary_text
        ),
    }
    return head + [note] + tail


def _archive(archive_dir: Path, middle: list[dict], summary: str) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    (archive_dir / f"compaction-{stamp}.json").write_text(
        json.dumps(
            {"summary": summary, "compacted_messages": middle},
            ensure_ascii=False,
            indent=2,
        ),
        "utf-8",
    )
