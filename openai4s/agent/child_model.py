"""Where a delegated child's model requests go, and the credential they carry.

A delegation spec may override the child's model: `model` as a bare model id,
or as a mapping of `provider` / `model` / `base_url` / `max_tokens` /
`temperature` / `timeout_s`, plus a top-level `provider`. The child's
`LLMConfig` used to be a copy of the parent's with those fields `setattr`-ed
on, so the parent's resolved `api_key` went wherever the override pointed. A
Python Cell is agent-written code, and `host.delegate({"request": ...,
"model": {"base_url": "https://anywhere/v1"}})` reached that copy intact: one
prompt-injected Cell could send the parent's key -- on the Web, the active
profile's key or a team member's own -- to a host it chose. The LLM transport
applies no egress allowlist, so nothing downstream stopped it. A free-form
endpoint was also a way for a Cell the sandbox keeps off the network to have
the daemon POST to any host, a private one included.

The rule is the one `ModelProfileService.credential` already applies to
profiles and pinned revisions: a key goes only to the destination it was
resolved for, where a destination is `(provider, effective endpoint)`.

* A child whose destination is its parent's keeps the parent's credential --
  whatever the parent was dispatched under, decided for exactly that endpoint.
  Changing the model id or a generation knob does not move a child.
* A child that moves never takes the parent's key. It may move only to a
  destination this install has configured, and takes that destination's own
  credential. On the Web the gateway answers through the model profiles (a
  saved profile's endpoint, or a provider's own configured endpoint) and the
  session owner's own key rule; elsewhere :func:`environment_credential`
  answers from the process environment alone.
* An endpoint nobody configured is refused, not dispatched keyless: the
  disclosure is not only the key but the request itself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from openai4s.config import LLMConfig, provider_env_api_key
from openai4s.endpoint_identity import normalize_endpoint
from openai4s.llm.resolve import keyless_endpoint

#: The keys a spec's `model` mapping may set. Anything else -- an `api_key`
#: included -- is ignored rather than honoured.
MODEL_OVERRIDE_KEYS = (
    "provider",
    "model",
    "base_url",
    "max_tokens",
    "temperature",
    "timeout_s",
)

_TEXT_KEYS = frozenset(("provider", "model", "base_url"))


class ChildModelError(ValueError):
    """A model override this install will not dispatch a child under."""


@dataclass(frozen=True)
class ChildCredential:
    """What a moved child is dispatched under.

    `api_key` is empty for a keyless local endpoint and for every unusable
    answer. Never serialised.
    """

    api_key: str = field(repr=False)
    source: str
    usable: bool = True


#: Answers "what may a request to exactly this destination carry?" for a child
#: whose destination is not its parent's. Receives the child's keyless
#: `LLMConfig`. `None` means the install has configured no such destination; an
#: unusable answer means it has, with no credential to send there.
CredentialResolver = Callable[[LLMConfig], "ChildCredential | None"]


def destination(config: Any) -> tuple[str, str]:
    """`(provider, endpoint)`: where a request under `config` is sent.

    The endpoint is the one `chat()` dispatches to, `cfg.base_url or
    spec["base_url"]`, compared credential-free and normalised so a trailing
    slash or the host's case is still the same endpoint.
    """
    provider = str(getattr(config, "provider", "") or "").strip().lower()
    base = str(getattr(config, "base_url", "") or "")
    if not base:
        base = _registry_base_url(provider)
    return provider, normalize_endpoint(base)


def _registry_base_url(provider: str) -> str:
    try:
        from openai4s.llm.registry import provider_spec

        return str(provider_spec(provider).get("base_url") or "")
    except Exception:  # noqa: BLE001 - an unregistered provider has none
        return ""


def _registered(provider: str) -> bool:
    try:
        from openai4s.llm.registry import provider_spec

        provider_spec(provider)
    except Exception:  # noqa: BLE001
        return False
    return True


def environment_credential(config: LLMConfig) -> ChildCredential | None:
    """The credential this process's environment holds for `config`'s destination.

    The answer when no model profiles exist (the CLI): the only destination the
    environment configures for a provider is that provider's own endpoint, as
    `LLMConfig` resolves it (`OPENAI4S_<P>_BASE_URL`, `OPENAI4S_LLM_BASE_URL`,
    then the registry). Anything else is `None`. At that endpoint: keyless when
    it is local, before any key; otherwise the key the environment holds for
    exactly that provider (:func:`provider_env_api_key` -- never the generic
    `OPENAI4S_LLM_API_KEY`, which belongs to the provider the process was
    started for).
    """
    provider, endpoint = destination(config)
    if not provider or not endpoint or not _registered(provider):
        return None
    try:
        own = destination(
            LLMConfig(provider=provider, base_url="", model="unused", api_key="unused")
        )
    except Exception:  # noqa: BLE001 - an unresolvable default configures nothing
        return None
    if endpoint != own[1]:
        return None
    if keyless_endpoint(provider, str(config.model or ""), endpoint):
        return ChildCredential("", "local")
    key = provider_env_api_key(provider)
    if key:
        return ChildCredential(key, "environment")
    return ChildCredential("", "missing", usable=False)


def model_overrides(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The `LLMConfig` fields a delegation spec asks to change."""
    overrides: dict[str, Any] = {}
    model = spec.get("model")
    if isinstance(model, Mapping):
        for key in MODEL_OVERRIDE_KEYS:
            value = model.get(key)
            if value is not None:
                overrides[key] = str(value) if key in _TEXT_KEYS else value
    elif model:
        overrides["model"] = str(model)
    if spec.get("provider"):
        overrides["provider"] = str(spec["provider"])
    return overrides


def _validate_base_url_override(base_url: str) -> None:
    """Reject raw URL parts that endpoint identity intentionally omits.

    The transport sends the raw URL. A query or fragment changes the request
    it makes even when ``normalize_endpoint`` says the destination is unchanged;
    a redirect from that request can forward its Authorization header. Userinfo
    and control characters must not be accepted as alternate spellings either.
    """
    invalid = (
        base_url != base_url.strip()
        or any(char in base_url for char in "?#")
        or any(ord(char) < 32 or ord(char) == 127 for char in base_url)
    )
    try:
        parts = urlsplit(base_url)
        _ = parts.port  # A malformed port raises ValueError only when read.
    except ValueError:
        invalid = True
    else:
        invalid = invalid or (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.port == 0
            or parts.username is not None
            or parts.password is not None
        )
    if invalid:
        raise ChildModelError(
            "delegate: child base_url must be an absolute http(s) URL without "
            "userinfo, query, fragment, or control characters"
        )


def child_llm_config(
    parent: LLMConfig,
    spec: Mapping[str, Any],
    resolve: CredentialResolver | None = None,
) -> LLMConfig:
    """The child's `LLMConfig`, its credential decided for where it is sent.

    Raises :class:`ChildModelError` for an override that moves the child to a
    destination with no configured credential, and for one `LLMConfig`
    refuses.
    """
    fields = model_overrides(spec)
    if fields.get("base_url"):
        _validate_base_url_override(fields["base_url"])
    if "provider" in fields and (
        fields["provider"].strip().lower() != str(parent.provider or "").strip().lower()
    ):
        # Same rule as `SessionRunner._llm_cfg`: a provider switch must not
        # inherit the old provider's concrete endpoint or model. Left empty,
        # `LLMConfig.__post_init__` resolves the new provider's own.
        fields.setdefault("base_url", "")
        fields.setdefault("model", "")
    try:
        child = dataclasses.replace(parent, **fields)
    except (TypeError, ValueError) as error:
        raise ChildModelError(f"invalid child model override: {error}") from error
    # `replace` re-runs `__post_init__`, which fills an empty key from the
    # environment -- including the generic key that belongs to another
    # provider. So the key is always assigned afterwards, never left to it:
    # a keyless local parent's child stays keyless too.
    if destination(child) == destination(parent):
        child.api_key = parent.api_key
        return child
    child.api_key = ""
    provider, endpoint = destination(child)
    where = f"{provider or '(no provider)'!r} at {endpoint or '(no endpoint)'}"
    try:
        credential = (resolve or environment_credential)(child)
    except ChildModelError:
        raise
    except Exception as error:  # noqa: BLE001 - surfaced as a refusal
        raise ChildModelError(
            f"delegate: could not resolve a credential for {where}: {error}"
        ) from error
    if credential is None:
        raise ChildModelError(
            f"delegate: the model override sends this child to {where}, which is "
            "not a configured model endpoint. A child may change its model id "
            "freely, but may move only to its parent's endpoint, a saved model "
            "profile's, or a provider's own configured endpoint; the parent's "
            "credential is never sent anywhere else"
        )
    if not credential.usable:
        raise ChildModelError(
            f"delegate: no credential is configured for {where}; the parent's "
            "credential is not sent to an endpoint it was not resolved for"
        )
    child.api_key = credential.api_key
    return child


__all__ = [
    "ChildCredential",
    "ChildModelError",
    "CredentialResolver",
    "MODEL_OVERRIDE_KEYS",
    "child_llm_config",
    "destination",
    "environment_credential",
    "model_overrides",
]
