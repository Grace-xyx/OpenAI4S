"""Control tools for independent background Python-cell jobs.

Execution remains owned by the existing per-dispatcher BackgroundExecutor;
these classes only expose submit, list, peek, and interrupt orchestration.
They do not provide a shell or a replacement scientific runtime.
"""

from __future__ import annotations

from typing import Any

from openai4s.tools.base import Tool
from openai4s.tools.contexts import ControlToolContext
from openai4s.tools.taxonomy import resource_key


class SubmitBackgroundExecTool(Tool):
    """Submit one long-running Python cell to its independent worker."""

    name = "exec_background"
    host_method = "exec_background"
    description = (
        "Submit a long-running Python cell to an independent background job. "
        "A Web session writes a persistent receipt (code SHA-256, not the "
        "source) before the worker starts and reports persistent true. A CLI or "
        "sub-agent job reports persistent false and stays in memory. After a "
        "daemon restart an unfinished Web job reads as outcome_unknown and is "
        "not replayed."
    )
    parameters = {
        "properties": {
            "code": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200000,
            }
        },
        "required": ["code"],
    }
    read_only = False
    side_effect_class = "runtime_mutation"
    permission_target_key = "code"
    resource_key_prefix = "background"
    resource_target_default = "jobs"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        spec = {"code": arguments["code"]}
        # The native schema does not allow origin spoofing. Preserve the
        # existing Host SDK's trusted keyword for backwards compatibility.
        if "origin" in arguments:
            spec["origin"] = arguments["origin"]
        return runtime.invoke(self.host_method, spec)


class ListBackgroundExecsTool(Tool):
    """List non-secret status projections for background jobs."""

    name = "exec_list"
    host_method = "exec_list"
    description = (
        "List background execution jobs and their current states. A Web "
        "session merges in-memory jobs with this session's persistent "
        "receipts. outcome_unknown means a restart left the job unfinished "
        "and it was not re-run."
    )
    parameters = {"properties": {}, "required": []}
    requires_approval = False
    resource_key_prefix = "background"
    resource_target_default = "jobs"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> list[dict]:
        del arguments
        return runtime.invoke(self.host_method)


class PeekBackgroundExecTool(Tool):
    """Read accumulated output and status for one job."""

    name = "exec_peek"
    host_method = "exec_peek"
    description = (
        "Read accumulated stdout and status for one background job. "
        "persistent true is the session receipt, including outcome_unknown "
        "after a restart. persistent false is an in-process job (CLI or "
        "sub-agent)."
    )
    parameters = {
        "properties": {"exec_id": {"type": "string", "minLength": 1, "maxLength": 256}},
        "required": ["exec_id"],
    }
    requires_approval = False
    resource_key_prefix = "background_exec"
    resource_target_key = "exec_id"

    def resource_keys(self, arguments: Any) -> tuple[str, ...]:
        exec_id = (
            arguments
            if isinstance(arguments, str)
            else (arguments or {}).get("exec_id")
        )
        return (resource_key("background_exec", exec_id),)

    def execute(
        self,
        runtime: ControlToolContext,
        arguments: dict | str,
    ) -> dict:
        exec_id = arguments if isinstance(arguments, str) else arguments["exec_id"]
        return runtime.invoke(self.host_method, exec_id)


class InterruptBackgroundExecTool(Tool):
    """Stop one exact job through BackgroundExecutor's idempotent interrupt."""

    name = "exec_interrupt"
    host_method = "exec_interrupt"
    description = (
        "Interrupt one running background job by exact execution ID. A "
        "receipt with no live process returns interrupt_undelivered and does "
        "not claim the stop was delivered."
    )
    parameters = {
        "properties": {"exec_id": {"type": "string", "minLength": 1, "maxLength": 256}},
        "required": ["exec_id"],
    }
    read_only = False
    # Stopping an exact job mutates runtime state, but remains approval-free like
    # stop_child: it cannot create work or widen authority.
    requires_approval = False
    side_effect_class = "runtime_mutation"
    resource_key_prefix = "background_exec"
    resource_target_key = "exec_id"

    def resource_keys(self, arguments: Any) -> tuple[str, ...]:
        exec_id = (
            arguments
            if isinstance(arguments, str)
            else (arguments or {}).get("exec_id")
        )
        return (resource_key("background_exec", exec_id),)

    def execute(
        self,
        runtime: ControlToolContext,
        arguments: dict | str,
    ) -> dict:
        exec_id = arguments if isinstance(arguments, str) else arguments["exec_id"]
        return runtime.invoke(self.host_method, exec_id)


__all__ = [
    "InterruptBackgroundExecTool",
    "ListBackgroundExecsTool",
    "PeekBackgroundExecTool",
    "SubmitBackgroundExecTool",
]
