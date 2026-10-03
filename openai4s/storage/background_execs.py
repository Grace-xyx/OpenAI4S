"""Bounded receipts for Web background cells.

A receipt is written before the worker exists. It stores a SHA-256 of the
code and the character count, never the source. Output is a head snapshot,
replaced wholesale on each flush. ``effective_status`` is the only rule that
turns a non-terminal row into ``outcome_unknown``: the row's daemon instance
is not the reader, or this process no longer holds the job. ``get`` and
``list`` both use it, and neither rewrites the row.

The constructor is passive. DDL is applied by the numbered Store migration.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Mapping

from openai4s.storage.migrations import apply_ddl_script

MAX_PERSISTED_OUTPUT_BYTES = 256 * 1024
# Per owner (``owner_user_id``), not per table; see ``_clear_quota_locked``.
MAX_TOTAL_OUTPUT_BYTES = 128 * 1024 * 1024
TERMINAL_TTL_MS = 7 * 24 * 60 * 60 * 1000
MAX_ERROR_CHARS = 2000

LAUNCHING = "launching"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
INTERRUPTED = "interrupted"
LAUNCH_FAILED = "launch_failed"
OUTCOME_UNKNOWN = "outcome_unknown"

NON_TERMINAL_STATUSES = frozenset({LAUNCHING, RUNNING})
TERMINAL_STATUSES = frozenset(
    {DONE, FAILED, INTERRUPTED, LAUNCH_FAILED, OUTCOME_UNKNOWN}
)
FINISH_STATUSES = frozenset({DONE, FAILED, INTERRUPTED, LAUNCH_FAILED})
STATUSES = NON_TERMINAL_STATUSES | TERMINAL_STATUSES

OUTCOME_UNKNOWN_ERROR = (
    "daemon restarted before a terminal state was recorded; " "the job was not re-run"
)

_COLUMNS = (
    "exec_id",
    "root_frame_id",
    "frame_id",
    "owner_user_id",
    "daemon_instance",
    "origin",
    "code_sha256",
    "code_chars",
    "env_generation",
    "status",
    "error",
    "interrupted",
    "output",
    "output_bytes",
    "output_truncated",
    "created_at",
    "updated_at",
    "started_at",
    "ended_at",
)

RECEIPT_SCHEMA = """
CREATE TABLE IF NOT EXISTS background_exec_receipts (
    exec_id TEXT PRIMARY KEY,
    root_frame_id TEXT NOT NULL,
    frame_id TEXT,
    owner_user_id TEXT,
    daemon_instance TEXT NOT NULL,
    origin TEXT,
    code_sha256 TEXT NOT NULL,
    code_chars INTEGER NOT NULL,
    env_generation TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'launching','running','done','failed','interrupted',
        'launch_failed','outcome_unknown'
    )),
    error TEXT,
    interrupted INTEGER NOT NULL DEFAULT 0,
    output TEXT NOT NULL DEFAULT '',
    output_bytes INTEGER NOT NULL DEFAULT 0,
    output_truncated INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    started_at INTEGER,
    ended_at INTEGER
);
CREATE INDEX IF NOT EXISTS ix_background_exec_root
    ON background_exec_receipts(root_frame_id, created_at);
CREATE INDEX IF NOT EXISTS ix_background_exec_status
    ON background_exec_receipts(status, ended_at);
"""


def create_background_exec_receipts_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, called from the numbered Store migration."""
    apply_ddl_script(conn, RECEIPT_SCHEMA)


def truncation_marker(limit: int) -> str:
    return f"\n...(truncated at {int(limit)} bytes)"


def bound_output(text: str, *, limit: int) -> tuple[str, int, bool]:
    """Keep a UTF-8 head within ``limit`` bytes, with an explicit cut marker."""

    limit = int(limit)
    if limit < 1:
        raise ValueError("persisted output limit must be positive")
    raw = str(text or "").encode("utf-8")
    if len(raw) <= limit:
        return str(text or ""), len(raw), False
    marker = truncation_marker(limit).encode("utf-8")
    budget = limit - len(marker)
    if budget <= 0:
        stored = marker[:limit].decode("utf-8", errors="ignore")
        return stored, len(stored.encode("utf-8")), True
    head = raw[:budget]
    while head and (head[-1] & 0xC0) == 0x80:
        head = head[:-1]
    if head and head[-1] >= 0xC0:
        head = head[:-1]
    stored = head.decode("utf-8") + marker.decode("utf-8")
    encoded = stored.encode("utf-8")
    if len(encoded) > limit:
        stored = encoded[:limit].decode("utf-8", errors="ignore")
        encoded = stored.encode("utf-8")
    return stored, len(encoded), True


def effective_status(row: Mapping[str, Any], current_instance: str) -> str:
    """Non-terminal rows with no live owner read as ``outcome_unknown``.

    This is the only derivation. It does not write. A row qualifies when its
    ``daemon_instance`` is not ``current_instance``, or when this process has
    no live job for its ``exec_id``. Terminal rows pass through.
    """

    status = str(row.get("status") or "")
    if status in TERMINAL_STATUSES:
        return status
    if str(row.get("daemon_instance") or "") != str(current_instance):
        return OUTCOME_UNKNOWN
    # Lazy: this repository must not import the kernel package at load.
    from openai4s.kernel.background import background_job_is_live

    if not background_job_is_live(str(row.get("exec_id") or "")):
        return OUTCOME_UNKNOWN
    return status


def project_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
    """Host-facing receipt. ``status`` is already the effective status."""

    status = str(row.get("status") or "")
    started = row.get("started_at")
    ended = row.get("ended_at")
    created = row.get("created_at")
    error = row.get("error")
    return {
        "exec_id": str(row.get("exec_id") or ""),
        "status": status,
        "done": status not in NON_TERMINAL_STATUSES,
        "stdout": str(row.get("output") or ""),
        "interrupted": bool(row.get("interrupted")),
        "error": None if error is None else str(error),
        "started_at": int(started) if isinstance(started, int) else None,
        "ended_at": int(ended) if isinstance(ended, int) else None,
        "persistent": True,
        "source": "receipt",
        "output_truncated": bool(row.get("output_truncated")),
        "code_sha256": str(row.get("code_sha256") or ""),
        "created_at": int(created) if isinstance(created, int) else 0,
    }


def _scope_predicate(root_frame_id: str) -> tuple[str, tuple[str]]:
    """Rows visible to one session root. Other roots match nothing."""

    return "root_frame_id=?", (str(root_frame_id),)


def _cap_error(error: str | None) -> str | None:
    if error is None:
        return None
    text = str(error)
    if len(text) <= MAX_ERROR_CHARS:
        return text
    return text[:MAX_ERROR_CHARS]


class BackgroundExecReceiptRepository:
    """Receipts over the Store connection. No process control."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: Any,
        *,
        clock_ms: Callable[[], int],
        max_output_bytes: int = MAX_PERSISTED_OUTPUT_BYTES,
        max_total_output_bytes: int = MAX_TOTAL_OUTPUT_BYTES,
        terminal_ttl_ms: int = TERMINAL_TTL_MS,
    ) -> None:
        self._conn = conn
        self._lock = lock
        self._clock_ms = clock_ms
        self._max_output_bytes = int(max_output_bytes)
        self._max_total_output_bytes = int(max_total_output_bytes)
        self._terminal_ttl_ms = int(terminal_ttl_ms)

    def begin(
        self,
        *,
        exec_id: str,
        root_frame_id: str,
        frame_id: str | None,
        owner_user_id: str | None,
        daemon_instance: str,
        origin: str | None,
        code_sha256: str,
        code_chars: int,
        env_generation: str | None = None,
    ) -> None:
        exec_id = _required("exec_id", exec_id)
        root_frame_id = _required("root_frame_id", root_frame_id)
        daemon_instance = _required("daemon_instance", daemon_instance)
        digest = str(code_sha256 or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("code_sha256 must be 64 hex characters")
        chars = int(code_chars)
        if chars < 0:
            raise ValueError("code_chars must be >= 0")
        generation = _optional_text(env_generation)
        now = int(self._clock_ms())
        with self._lock:
            self._conn.execute(
                "INSERT INTO background_exec_receipts("
                "exec_id, root_frame_id, frame_id, owner_user_id, daemon_instance,"
                "origin, code_sha256, code_chars, env_generation, status, error,"
                "interrupted, output, output_bytes, output_truncated,"
                "created_at, updated_at, started_at, ended_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,NULL,0,'',0,0,?,?,NULL,NULL)",
                (
                    exec_id,
                    root_frame_id,
                    _optional_text(frame_id),
                    _optional_text(owner_user_id),
                    daemon_instance,
                    _optional_text(origin),
                    digest,
                    chars,
                    generation,
                    LAUNCHING,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def mark_running(self, exec_id: str, started_at: int) -> None:
        now = int(self._clock_ms())
        with self._lock:
            self._conn.execute(
                "UPDATE background_exec_receipts SET status=?, started_at=?, "
                "updated_at=? WHERE exec_id=? AND status=?",
                (RUNNING, int(started_at), now, str(exec_id), LAUNCHING),
            )
            self._conn.commit()

    def note_env_generation(self, exec_id: str, generation: str | None) -> None:
        text = _optional_text(generation)
        if text is None:
            return
        now = int(self._clock_ms())
        with self._lock:
            self._conn.execute(
                "UPDATE background_exec_receipts SET env_generation=?, updated_at=? "
                "WHERE exec_id=? AND env_generation IS NULL",
                (text, now, str(exec_id)),
            )
            self._conn.commit()

    def save_output(self, exec_id: str, text: str, truncated: bool) -> None:
        stored, nbytes, cut = bound_output(text, limit=self._max_output_bytes)
        flag = 1 if truncated or cut else 0
        now = int(self._clock_ms())
        with self._lock:
            self._conn.execute(
                "UPDATE background_exec_receipts SET output=?, output_bytes=?, "
                "output_truncated=?, updated_at=? "
                "WHERE exec_id=? AND status IN ('launching','running')",
                (stored, nbytes, flag, now, str(exec_id)),
            )
            self._conn.commit()

    def finish(
        self,
        exec_id: str,
        *,
        status: str,
        error: str | None,
        interrupted: bool,
        ended_at: int,
        output: str,
        truncated: bool,
    ) -> None:
        if status not in FINISH_STATUSES:
            raise ValueError(f"unsupported terminal status {status!r}")
        stored, nbytes, cut = bound_output(output, limit=self._max_output_bytes)
        now = int(self._clock_ms())
        with self._lock:
            self._conn.execute(
                "UPDATE background_exec_receipts SET status=?, error=?, "
                "interrupted=?, ended_at=?, output=?, output_bytes=?, "
                "output_truncated=?, updated_at=? "
                "WHERE exec_id=? AND status IN ('launching','running')",
                (
                    status,
                    _cap_error(error),
                    1 if interrupted else 0,
                    int(ended_at),
                    stored,
                    nbytes,
                    1 if truncated or cut else 0,
                    now,
                    str(exec_id),
                ),
            )
            self._conn.commit()

    def get(
        self,
        exec_id: str,
        *,
        root_frame_id: str,
        current_instance: str,
    ) -> dict[str, Any] | None:
        clause, params = _scope_predicate(root_frame_id)
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM background_exec_receipts "
                f"WHERE exec_id=? AND {clause}",
                (str(exec_id), *params),
            ).fetchone()
        return self._public(row, current_instance=current_instance)

    def list(
        self,
        root_frame_id: str,
        *,
        limit: int = 50,
        current_instance: str,
    ) -> list[dict[str, Any]]:
        clause, params = _scope_predicate(root_frame_id)
        bounded = max(1, min(int(limit), 50))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM background_exec_receipts "
                f"WHERE {clause} ORDER BY created_at DESC LIMIT ?",
                (*params, bounded),
            ).fetchall()
        return [
            item
            for item in (
                self._public(row, current_instance=current_instance) for row in rows
            )
            if item is not None
        ]

    def prune(self, now: int, *, current_instance: str) -> dict[str, int]:
        """Mark foreign non-terminal rows, drop expired terminals, trim output.

        Non-terminal rows are never deleted. Output is cleared from each
        owner's oldest terminal row first until that owner's stored total is
        within the quota. Victims are chosen in one ordered read per owner
        over quota and cleared in fixed-size batches, so a total is not
        recomputed once per row.
        """

        moment = int(now)
        current = str(current_instance or "")
        cutoff = moment - self._terminal_ttl_ms
        terminals = tuple(sorted(TERMINAL_STATUSES))
        terminal_marks = ",".join("?" for _ in terminals)
        with self._lock:
            marked = self._conn.execute(
                "UPDATE background_exec_receipts SET status=?, error=?, "
                "ended_at=COALESCE(ended_at, ?), updated_at=? "
                "WHERE daemon_instance!=? AND status IN ('launching','running')",
                (OUTCOME_UNKNOWN, OUTCOME_UNKNOWN_ERROR, moment, moment, current),
            )
            deleted = self._conn.execute(
                "DELETE FROM background_exec_receipts "
                f"WHERE status IN ({terminal_marks}) "
                "AND ended_at IS NOT NULL AND ended_at < ?",
                (*terminals, cutoff),
            )
            cleared = self._clear_quota_locked(moment)
            self._conn.commit()
        return {
            "marked_unknown": int(marked.rowcount or 0),
            "deleted": int(deleted.rowcount or 0),
            "cleared": cleared,
        }

    def _clear_quota_locked(self, moment: int) -> int:
        """Clear each owner's oldest terminal output until it is in quota.

        The quota applies per ``owner_user_id``. Rows with no owner form one
        group, which in single-user mode is every row. With one shared table
        cap, any team member who could run background cells could clear the
        stored output of every other member. Now one owner's jobs only ever
        clear that owner's own rows.

        Caller holds the repository lock. Within an owner the order matches
        the previous per-row loop: ``ended_at``, then ``created_at``, then
        ``exec_id``. Updates run in fixed-size batches.
        """

        over = self._conn.execute(
            "SELECT owner_user_id, COALESCE(SUM(output_bytes), 0) AS total "
            "FROM background_exec_receipts GROUP BY owner_user_id "
            "HAVING COALESCE(SUM(output_bytes), 0) > ?",
            (self._max_total_output_bytes,),
        ).fetchall()
        if not over:
            return 0
        terminals = tuple(sorted(TERMINAL_STATUSES))
        marks = ",".join("?" for _ in terminals)
        chosen: list[str] = []
        for group in over:
            excess = int(group["total"] or 0) - self._max_total_output_bytes
            victims = self._conn.execute(
                "SELECT exec_id, output_bytes FROM background_exec_receipts "
                f"WHERE owner_user_id IS ? AND status IN ({marks}) "
                "AND output_bytes>0 "
                "ORDER BY ended_at ASC, created_at ASC, exec_id ASC",
                (group["owner_user_id"], *terminals),
            ).fetchall()
            freed = 0
            for victim in victims:
                if freed >= excess:
                    break
                chosen.append(str(victim["exec_id"]))
                freed += int(victim["output_bytes"] or 0)
        cleared = 0
        batch = 400
        for start in range(0, len(chosen), batch):
            chunk = chosen[start : start + batch]
            placeholders = ",".join("?" for _ in chunk)
            result = self._conn.execute(
                "UPDATE background_exec_receipts SET output='', output_bytes=0, "
                "output_truncated=1, updated_at=? "
                f"WHERE exec_id IN ({placeholders})",
                (moment, *chunk),
            )
            cleared += int(result.rowcount or 0)
        return cleared

    def _public(self, row: Any, *, current_instance: str) -> dict[str, Any] | None:
        if row is None:
            return None
        item = {key: row[key] for key in _COLUMNS}
        status = effective_status(item, current_instance)
        if (
            status == OUTCOME_UNKNOWN
            and str(item.get("status") or "") in NON_TERMINAL_STATUSES
            and str(item.get("daemon_instance") or "") == str(current_instance)
        ):
            # The owner can write its terminal state and leave the live set
            # between the SELECT that produced this row and the live check.
            # It leaves only after that write commits, so one re-read finds
            # the terminal row when there is one.
            with self._lock:
                fresh = self._conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM background_exec_receipts "
                    "WHERE exec_id=?",
                    (str(item["exec_id"]),),
                ).fetchone()
            if fresh is not None:
                item = {key: fresh[key] for key in _COLUMNS}
                status = effective_status(item, current_instance)
        item["status"] = status
        item["interrupted"] = int(item["interrupted"] or 0)
        item["output_bytes"] = int(item["output_bytes"] or 0)
        item["output_truncated"] = int(item["output_truncated"] or 0)
        item["code_chars"] = int(item["code_chars"] or 0)
        return item


class BoundBackgroundReceipts:
    """Repository calls already bound to one session and one daemon instance."""

    def __init__(
        self,
        repository: BackgroundExecReceiptRepository,
        *,
        root_frame_id: str,
        frame_id: str | None,
        daemon_instance: str,
        owner_user_id: str | None,
        env_generation: str | None = None,
    ) -> None:
        self._repository = repository
        self._root_frame_id = str(root_frame_id or "")
        self._frame_id = _optional_text(frame_id)
        self._daemon_instance = str(daemon_instance or "")
        self._owner_user_id = _optional_text(owner_user_id)
        self._env_generation = _optional_text(env_generation)

    def begin(self, *, exec_id: str, code: str, origin: str) -> None:
        import hashlib

        source = str(code)
        self._repository.begin(
            exec_id=exec_id,
            root_frame_id=self._root_frame_id,
            frame_id=self._frame_id,
            owner_user_id=self._owner_user_id,
            daemon_instance=self._daemon_instance,
            origin=origin,
            code_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            code_chars=len(source),
            env_generation=self._env_generation,
        )

    def mark_running(self, exec_id: str, started_at: int) -> None:
        self._repository.mark_running(exec_id, started_at)

    def note_env_generation(self, exec_id: str, generation: str | None) -> None:
        self._repository.note_env_generation(exec_id, generation)

    def save_output(self, exec_id: str, text: str, truncated: bool) -> None:
        self._repository.save_output(exec_id, text, truncated)

    def finish(
        self,
        exec_id: str,
        *,
        status: str,
        error: str | None,
        interrupted: bool,
        ended_at: int,
        output: str,
        truncated: bool,
    ) -> None:
        self._repository.finish(
            exec_id,
            status=status,
            error=error,
            interrupted=interrupted,
            ended_at=ended_at,
            output=output,
            truncated=truncated,
        )

    @property
    def daemon_instance(self) -> str:
        """The process id these receipts are written and read under."""

        return self._daemon_instance

    def get(self, exec_id: str) -> dict[str, Any] | None:
        return self._repository.get(
            exec_id,
            root_frame_id=self._root_frame_id,
            current_instance=self._daemon_instance,
        )

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._repository.list(
            self._root_frame_id,
            limit=limit,
            current_instance=self._daemon_instance,
        )

    def prune(self, now: int) -> dict[str, int]:
        return self._repository.prune(now, current_instance=self._daemon_instance)


def _required(name: str, value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "MAX_PERSISTED_OUTPUT_BYTES",
    "MAX_TOTAL_OUTPUT_BYTES",
    "OUTCOME_UNKNOWN",
    "OUTCOME_UNKNOWN_ERROR",
    "TERMINAL_TTL_MS",
    "BackgroundExecReceiptRepository",
    "BoundBackgroundReceipts",
    "bound_output",
    "create_background_exec_receipts_schema",
    "effective_status",
    "project_receipt",
]
