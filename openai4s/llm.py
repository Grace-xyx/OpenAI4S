"""Multi-provider LLM client (pure stdlib urllib).

This is the one place the whole system talks to a base model. It speaks THREE
wire formats behind a single normalized `chat` entrypoint:

  wire="openai"     OpenAI-compatible /chat/completions
                    -> ark (Volcengine plan gateway), chatgpt
  wire="anthropic"  Messages API /v1/messages (x-api-key + version header)
                    -> claude
  wire="gemini"     generateContent (x-goog-api-key)
                    -> gemini

Multimodal (image) input is supported for providers whose `vision` flag is set
(ark, chatgpt, claude, gemini). A message's `content` may be either a plain
string or a list of normalized parts:

  {"type": "text", "text": "..."}
  {"type": "image", "url": "https://..."}
  {"type": "image", "data": "<base64>", "mime": "image/png"}

Each wire format translates these parts into its own schema.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


# --- provider registry ----------------------------------------------------
# Each provider maps to a wire format, a default base_url + model, and whether
# it accepts image parts. base_url/model here are DEFAULTS; config may override.
PROVIDERS: dict[str, dict[str, Any]] = {
    # Volcengine Ark "plan" gateway (火山方舟) — one OpenAI-compatible endpoint that
    # fronts many model families (doubao / glm / kimi / deepseek / minimax). Pick
    # the concrete model per config/profile; the key + endpoint are shared.
    "ark": {
        "wire": "openai",
        "base_url": "https://ark.cn-beijing.volces.com/api/plan/v3",
        "model": "doubao-seed-2.0-pro",
        "vision": True,
    },
    # Official first-party endpoints for the frontier labs.
    "chatgpt": {
        "wire": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5",
        "vision": True,
    },
    "claude": {
        "wire": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-5",
        "vision": True,
    },
    "gemini": {
        "wire": "gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-2.5-flash",
        "vision": True,
    },
}

# Model ids served by the Ark plan/v3 gateway (all share the `ark` provider's
# endpoint + key). Surfaced as ready-to-pick model profiles in Customize → Models.
ARK_PLAN_MODELS: tuple[tuple[str, str], ...] = (
    ("doubao-seed-2.0-pro", "Doubao Seed 2.0 Pro"),
    ("doubao-seed-2.0-code", "Doubao Seed 2.0 Code"),
    ("doubao-seed-2.0-lite", "Doubao Seed 2.0 Lite"),
    ("doubao-seed-2.0-mini", "Doubao Seed 2.0 Mini"),
    ("glm-5.2", "GLM 5.2"),
    ("kimi-k2.7-code", "Kimi K2.7 Code"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("minimax-m3", "MiniMax M3"),
    ("minimax-m2.7", "MiniMax M2.7"),
    ("kimi-k2.6", "Kimi K2.6"),
)

_ANTHROPIC_VERSION = "2023-06-01"


def provider_spec(name: str) -> dict[str, Any]:
    spec = PROVIDERS.get(name.lower())
    if spec is None:
        raise LLMError(
            f"unknown provider {name!r}; known: {', '.join(sorted(PROVIDERS))}"
        )
    return spec


def supports_vision(provider: str) -> bool:
    return bool(provider_spec(provider).get("vision"))


# --- low-level HTTP -------------------------------------------------------
def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise LLMError(f"LLM HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"LLM connection error: {e.reason}") from e


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _post_sse(url: str, payload: dict, headers: dict, timeout: float, on_event) -> None:
    """POST and iterate a Server-Sent-Events stream, calling on_event(dict) per
    `data:` JSON line. Used by the OpenAI Responses wire."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise LLMError(f"LLM HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"LLM connection error: {e.reason}") from e
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                evt = json.loads(chunk)
            except ValueError:
                continue
            on_event(evt)
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass


# --- content-part translation --------------------------------------------
def _is_parts(content: Any) -> bool:
    return isinstance(content, list)


def _to_openai_content(content: Any) -> Any:
    if not _is_parts(content):
        return content
    out: list[dict] = []
    for p in content:
        if p.get("type") == "text":
            out.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image":
            if p.get("url"):
                url = p["url"]
            else:
                url = f"data:{p.get('mime', 'image/png')};base64,{p.get('data', '')}"
            out.append({"type": "image_url", "image_url": {"url": url}})
    return out


def _to_anthropic_content(content: Any) -> Any:
    if not _is_parts(content):
        return content
    out: list[dict] = []
    for p in content:
        if p.get("type") == "text":
            out.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image":
            if p.get("url"):
                src = {"type": "url", "url": p["url"]}
            else:
                src = {
                    "type": "base64",
                    "media_type": p.get("mime", "image/png"),
                    "data": p.get("data", ""),
                }
            out.append({"type": "image", "source": src})
    return out


def _to_gemini_parts(content: Any) -> list[dict]:
    if not _is_parts(content):
        return [{"text": str(content)}]
    out: list[dict] = []
    for p in content:
        if p.get("type") == "text":
            out.append({"text": p.get("text", "")})
        elif p.get("type") == "image":
            if p.get("url"):
                out.append({"file_data": {"file_uri": p["url"]}})
            else:
                out.append(
                    {
                        "inline_data": {
                            "mime_type": p.get("mime", "image/png"),
                            "data": p.get("data", ""),
                        }
                    }
                )
    return out


def _guard_vision(provider: str, messages: list[dict]) -> None:
    """Raise a clear error if image parts are sent to a text-only provider."""
    if supports_vision(provider):
        return
    for m in messages:
        if _is_parts(m.get("content")) and any(
            p.get("type") == "image" for p in m["content"]
        ):
            raise LLMError(
                f"provider {provider!r} has no vision support; image parts are "
                f"only accepted by: "
                f"{', '.join(p for p in PROVIDERS if PROVIDERS[p]['vision'])}"
            )


# --- per-wire callers -----------------------------------------------------
def _chat_openai(
    messages, cfg, base, model, max_tokens, temperature, stop, on_delta=None
) -> dict:
    url = f"{base.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {**m, "content": _to_openai_content(m.get("content"))} for m in messages
        ],
        "max_tokens": max_tokens or cfg.max_tokens,
        "temperature": cfg.temperature if temperature is None else temperature,
    }
    if stop:
        payload["stop"] = stop
    # Some OpenAI-compatible proxies (e.g. apiany.org, behind Cloudflare) reject
    # urllib's default UA and expose reasoning models — allow env overrides.
    effort = os.environ.get("OPENAI4S_LLM_REASONING_EFFORT")
    if effort:
        payload["reasoning_effort"] = effort
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
        "User-Agent": os.environ.get("OPENAI4S_LLM_USER_AGENT", _BROWSER_UA),
    }
    # Real token streaming: when a delta callback is supplied AND streaming isn't
    # explicitly disabled, POST with `stream:true` and forward each token to
    # on_delta as it arrives, so prose renders live instead of one blob per turn.
    # Falls back to the blocking path if the stream can't even start (some proxies
    # 4xx on `stream`), so a provider that refuses SSE still works.
    want_stream = on_delta is not None and os.environ.get(
        "OPENAI4S_LLM_STREAM", "1"
    ) not in ("0", "false", "no", "off")
    if want_stream:
        try:
            return _chat_openai_stream(url, dict(payload), headers, cfg, on_delta)
        except _StreamStartError:
            pass  # SSE refused before any bytes — retry blocking below
    body = _post_json(url, payload, headers, cfg.timeout_s)
    try:
        choice = body["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected OpenAI-wire response: {body}") from e
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content"),
        "usage": body.get("usage", {}),
        "finish_reason": choice.get("finish_reason"),
        "raw": body,
    }


class _StreamStartError(Exception):
    """The streaming request failed before yielding any data — safe to fall back
    to a blocking call (nothing was emitted to the client yet)."""


def _chat_openai_stream(url, payload, headers, cfg, on_delta) -> dict:
    payload["stream"] = True
    # Ask for a usage row on the terminal chunk (ignored by proxies that don't
    # grok it; harmless when unsupported).
    payload["stream_options"] = {"include_usage": True}
    headers = {**headers, "Accept": "text/event-stream"}
    parts: list[str] = []
    reasoning: list[str] = []
    state: dict[str, Any] = {"usage": {}, "finish": None, "started": False}

    def _on_event(evt: dict) -> None:
        state["started"] = True
        if evt.get("usage"):
            state["usage"] = evt["usage"]
        choices = evt.get("choices") or []
        if not choices:
            return
        ch = choices[0]
        delta = ch.get("delta") or {}
        piece = delta.get("content")
        if piece:
            parts.append(piece)
            try:
                on_delta(piece)
            except Exception:  # noqa: BLE001 — a UI callback must never kill the stream
                pass
        rc = delta.get("reasoning_content") or delta.get("reasoning")
        if rc:
            reasoning.append(rc)
        if ch.get("finish_reason"):
            state["finish"] = ch["finish_reason"]

    timeout = max(cfg.timeout_s, 60.0)
    try:
        _post_sse(url, payload, headers, timeout, _on_event)
    except LLMError:
        # Connection/HTTP error. If we already streamed tokens, surfacing a hard
        # error would duplicate/contradict what the user saw — but nothing was
        # committed downstream yet, so re-raise as a start error only when we
        # never emitted, else propagate.
        if not state["started"]:
            raise _StreamStartError()
        raise
    return {
        "content": "".join(parts),
        "reasoning": "".join(reasoning) or None,
        "usage": state["usage"],
        "finish_reason": state["finish"] or "stop",
        "raw": None,
    }


def _chat_anthropic(
    messages, cfg, base, model, max_tokens, temperature, stop, on_delta=None
) -> dict:
    url = f"{base.rstrip('/')}/v1/messages"
    # Anthropic takes a top-level `system` string, not a system message.
    system_txt = ""
    conv: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            system_txt += c if isinstance(c, str) else ""
            continue
        conv.append(
            {"role": m["role"], "content": _to_anthropic_content(m.get("content"))}
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": conv,
        "max_tokens": max_tokens or cfg.max_tokens,
        "temperature": cfg.temperature if temperature is None else temperature,
    }
    if system_txt:
        payload["system"] = system_txt
    if stop:
        payload["stop_sequences"] = stop
    headers = {
        "Content-Type": "application/json",
        "x-api-key": cfg.api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    body = _post_json(url, payload, headers, cfg.timeout_s)
    try:
        blocks = body["content"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    except (KeyError, TypeError) as e:
        raise LLMError(f"Unexpected Anthropic-wire response: {body}") from e
    return {
        "content": text,
        "reasoning": None,
        "usage": body.get("usage", {}),
        "finish_reason": body.get("stop_reason"),
        "raw": body,
    }


def _chat_gemini(
    messages, cfg, base, model, max_tokens, temperature, stop, on_delta=None
) -> dict:
    url = f"{base.rstrip('/')}/v1beta/models/{model}:generateContent"
    system_txt = ""
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            c = m.get("content")
            system_txt += c if isinstance(c, str) else ""
            continue
        g_role = "model" if role == "assistant" else "user"
        contents.append({"role": g_role, "parts": _to_gemini_parts(m.get("content"))})
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens or cfg.max_tokens,
            "temperature": cfg.temperature if temperature is None else temperature,
        },
    }
    if system_txt:
        payload["systemInstruction"] = {"parts": [{"text": system_txt}]}
    if stop:
        payload["generationConfig"]["stopSequences"] = stop
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": cfg.api_key,
    }
    body = _post_json(url, payload, headers, cfg.timeout_s)
    try:
        cand = body["candidates"][0]
        parts = cand["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected Gemini-wire response: {body}") from e
    return {
        "content": text,
        "reasoning": None,
        "usage": body.get("usageMetadata", {}),
        "finish_reason": cand.get("finishReason"),
        "raw": body,
    }


def _flatten_text(content: Any) -> str:
    """Collapse a string-or-parts message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict)
            and p.get("type") in ("text", "input_text", "output_text")
        )
    return str(content)


def _chat_responses(
    messages, cfg, base, model, max_tokens, temperature, stop, on_delta=None
) -> dict:
    """OpenAI Responses API (Codex `wire_api = responses`). Streams SSE:
    text arrives as `response.output_text.delta` events; usage on
    `response.completed`. System messages become `instructions`."""
    url = f"{base.rstrip('/')}/responses"
    instructions: list[str] = []
    input_items: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = _flatten_text(m.get("content"))
        if role == "system":
            if text:
                instructions.append(text)
        else:
            ptype = "output_text" if role == "assistant" else "input_text"
            input_items.append(
                {"role": role, "content": [{"type": ptype, "text": text}]}
            )
    if not input_items:
        input_items.append(
            {"role": "user", "content": [{"type": "input_text", "text": ""}]}
        )
    effort = os.environ.get("OPENAI4S_LLM_REASONING_EFFORT") or "high"
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "reasoning": {"effort": effort},
        "store": False,
        "stream": True,
    }
    # NB: this proxy rejects `max_output_tokens` — leave it off.
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
        "User-Agent": os.environ.get("OPENAI4S_LLM_USER_AGENT", _BROWSER_UA),
        "Accept": "text/event-stream",
    }
    text_parts: list[str] = []
    state: dict[str, Any] = {"usage": {}, "error": None}

    def _on_event(evt: dict) -> None:
        t = evt.get("type")
        if t == "response.output_text.delta":
            d = evt.get("delta") or ""
            if d:
                text_parts.append(d)
                if on_delta:
                    try:
                        on_delta(d)
                    except (
                        Exception
                    ):  # noqa: BLE001 - a UI callback must not kill the stream
                        pass
        elif t == "response.completed":
            u = (evt.get("response") or {}).get("usage") or {}
            state["usage"] = {
                "prompt_tokens": u.get("input_tokens"),
                "completion_tokens": u.get("output_tokens"),
                "total_tokens": u.get("total_tokens"),
                "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
            }
        elif t == "response.output_item.done":
            item = evt.get("item") or {}
            if item.get("type") == "message" and not text_parts:
                for part in item.get("content") or []:
                    if part.get("text"):
                        text_parts.append(part["text"])
        elif t in ("response.failed", "response.error", "error"):
            resp = evt.get("response") or evt
            err = (resp.get("error") or {}) if isinstance(resp, dict) else {}
            # a flat `error` event carries `message` at the top level, while
            # response.failed nests it under response.error
            state["error"] = err.get("message") or evt.get("message") or str(evt)[:400]

    # Idle (no-bytes) timeout for the stream. Respect the configured timeout so a
    # stalled/hung model finalises the turn promptly instead of "running forever";
    # keep a 60s floor so a heavy-reasoning model that pauses between events isn't
    # cut off (raise OPENAI4S_LLM_TIMEOUT for such models).
    timeout = max(cfg.timeout_s, 60.0)
    _post_sse(url, payload, headers, timeout, _on_event)
    if state["error"]:
        raise LLMError(f"responses API error: {state['error']}")
    return {
        "content": "".join(text_parts),
        "reasoning": None,
        "usage": state["usage"],
        "finish_reason": "stop",
        "raw": None,
    }


_WIRE_DISPATCH = {
    "openai": _chat_openai,
    "anthropic": _chat_anthropic,
    "gemini": _chat_gemini,
    "responses": _chat_responses,
}


def chat(
    messages: list[dict[str, Any]],
    cfg: LLMConfig,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    on_delta=None,
) -> dict[str, Any]:
    """One blocking chat-completion call against the configured provider.

    `on_delta(text)` (if given) is called with streamed text chunks when the
    provider wire supports it (the `responses` and `openai` wires stream via
    SSE; set OPENAI4S_LLM_STREAM=0 to force the blocking path). The
    `anthropic` and `gemini` wires ignore it and return the full text at once.

    Returns a normalized dict:
        {"content": str, "reasoning": str|None, "usage": {...},
         "finish_reason": ..., "raw": {...}}
    """
    if not cfg.api_key:
        raise LLMError(
            f"no API key configured for provider {cfg.provider!r}: set the "
            f"OPENAI4S_{cfg.provider.upper()}_API_KEY (or generic OPENAI4S_LLM_API_KEY) "
            f"environment variable, or add it to a .env file at the repo root. "
            f"See .env.example."
        )
    spec = provider_spec(cfg.provider)
    _guard_vision(cfg.provider, messages)
    base = cfg.base_url or spec["base_url"]
    model = cfg.model or spec["model"]
    caller = _WIRE_DISPATCH[spec["wire"]]
    return caller(
        messages, cfg, base, model, max_tokens, temperature, stop, on_delta=on_delta
    )
