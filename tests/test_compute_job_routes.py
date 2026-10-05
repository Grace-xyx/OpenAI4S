"""The compute-jobs routes must give a domain failure an HTTP status.

Every refusal from `JobManager` arrives as a soft ``{"error": ...}`` dict,
and the routes serialized all of them as 200 — so an unknown job read as a
successful lookup, a refused cancel as a cancel that landed, and a refused
submit as an accepted one. `api()` in the web client only throws on
non-2xx, so the workbench showed all three as success: a job that did not
exist rendered "output empty", and a refused submit disappeared silently.

The mapping mirrors the rule `_skill_result_status` already applies to the
skill surface: the gateway is where a domain failure becomes an HTTP one.

- `GET /compute/jobs/{unknown}` and `POST .../cancel` answer 404.
- Submit refusals answer by their stable `code`: client input (`empty
  command`, `job_bad_deadline`, `job_cwd_escape`) → 400, `job_capacity` →
  429, `job_workspace_unavailable` → 500, `job_manager_closed` → 503.

Every refusal test here fails if the mapping is removed: the route goes
back to a 200 carrying the same body.
"""

from __future__ import annotations

from openai4s.config import Config, LLMConfig
from openai4s.server import gateway as gateway_mod


class _Hub:
    def __init__(self):
        self.events = []

    def emitter(self, root_frame_id):
        def emit(event):
            event.setdefault("root_frame_id", root_frame_id)
            self.events.append(event)

        return emit

    def broadcast(self, root_frame_id, event):
        event.setdefault("root_frame_id", root_frame_id)
        self.events.append(event)


def _cfg(tmp_path):
    return Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
        max_turns=3,
    )


def _handler(cfg, runner, body=None):
    handler_cls = gateway_mod.make_handler(cfg, _Hub(), runner)
    handler = object.__new__(handler_cls)
    handler._query = lambda: {}
    handler._body = lambda: dict(body or {})
    seen: list[tuple[dict, int]] = []
    handler._json = lambda obj, code=200: seen.append((obj, code))
    return handler, seen


def test_reading_a_missing_job_is_a_404(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner)
        handler._api("GET", "/compute/jobs/job-that-never-ran")
        assert seen[-1] == ({"error": "job not found"}, 404)
    finally:
        runner.close()


def test_cancelling_a_missing_job_is_a_404(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner)
        handler._api("POST", "/compute/jobs/job-that-never-ran/cancel")
        assert seen[-1] == ({"error": "job not found"}, 404)
    finally:
        runner.close()


def test_submitting_an_empty_command_is_a_400(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner, {"command": "   "})
        handler._api("POST", "/compute/jobs")
        assert seen[-1] == (
            {"error": "empty command", "code": "job_empty_command"},
            400,
        )
    finally:
        runner.close()


def test_submitting_a_bad_deadline_is_a_400(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner, {"command": "x", "deadline_s": "soon"})
        handler._api("POST", "/compute/jobs")
        body, code = seen[-1]
        assert code == 400
        assert body["code"] == "job_bad_deadline"
    finally:
        runner.close()


def test_submitting_an_escaping_cwd_is_a_400(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(cfg, runner, {"command": "x", "cwd": "../.."})
        handler._api("POST", "/compute/jobs")
        assert seen[-1] == (
            {"error": "cwd escapes the jobs root", "code": "job_cwd_escape"},
            400,
        )
    finally:
        runner.close()


def test_a_real_job_reads_and_cancels_with_200(tmp_path):
    cfg = _cfg(tmp_path)
    runner = gateway_mod.SessionRunner(cfg, _Hub())
    try:
        handler, seen = _handler(
            cfg,
            runner,
            {"command": "import time; time.sleep(30)", "kind": "python"},
        )
        handler._api("POST", "/compute/jobs")
        submitted, code = seen[-1]
        assert code == 200, submitted
        job_id = submitted["id"]
        try:
            handler._api("GET", f"/compute/jobs/{job_id}")
            row, code = seen[-1]
            assert code == 200
            assert row["id"] == job_id and row["status"] in ("queued", "running")

            handler._api("POST", f"/compute/jobs/{job_id}/cancel")
            stopped, code = seen[-1]
            assert code == 200
            assert stopped.get("ok") is True

            # Read the terminal row too: the frozen [ok] shape must cover both
            # the running and the finished state (exit_code/finished_at are
            # null in one and set in the other).
            handler._api("GET", f"/compute/jobs/{job_id}")
            row, code = seen[-1]
            assert code == 200
            assert row["status"] in ("cancelled", "failed", "completed")
        finally:
            # Never leave a real process behind, whatever the asserts did.
            # Cancel on a terminal job is idempotent.
            handler._api("POST", f"/compute/jobs/{job_id}/cancel")
    finally:
        runner.close()
