"""Host-side remote-compute transport.

The worker's ``host.compute`` SDK routes every call to
``host_call("compute_<op>", [kw])``; the dispatcher forwards those here. This
module owns the real work the SDK only describes:

  * provider discovery — scan ``skills/remote-compute-<id>/provider.json`` for
    an ``id`` and a ``provider.py`` that exports ``PROVIDER``.
  * byoc transport      — spawn the confined ``openai4s_compute_provider``
    helper (oneshot mode) per op, staging inputs/outputs through a temp dir and
    handing the credential on the helper's stdin so the process environment is
    never a secret carrier.
  * ssh transport       — run a job script / one-off command over an SSH alias.
  * a background poller  — harvest terminal jobs into ``hpc/<job_id>/`` and mark
    them done so ``.result()`` can read them non-blocking.

Two provider families share one manager:
  "byoc:<id>"  bring-your-own-compute sandbox (e.g. "byoc:nvidia").
  "ssh:<alias>" a job over an existing SSH connection.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Repo-root openai4s_compute_provider (the confined helper package).
_HELPER_MAIN = str(
    Path(__file__).resolve().parent.parent.parent
    / "openai4s_compute_provider"
    / "__main__.py"
)

# Job-wrapper templates (ported alongside this module).
_TMPL_DIR = Path(__file__).resolve().parent / "templates"


class ComputeError(RuntimeError):
    """Surface as {'error', 'error_kind', ...} on the wire; the SDK turns a
    non-status error into a RuntimeError carrying .error_kind."""

    def __init__(
        self, msg: str, kind: str = "transient", concurrency: dict | None = None
    ):
        super().__init__(msg)
        self.error_kind = kind
        self.concurrency = concurrency


def _discover_providers(skills_dir: Path) -> dict[str, dict]:
    """Map provider id -> {id, dir, provider_py, meta}. A provider is a
    ``remote-compute-<id>`` skill dir with a ``provider.json`` (declaring its
    ``id``) and a ``provider.py`` exporting ``PROVIDER``."""
    out: dict[str, dict] = {}
    if not skills_dir.exists():
        return out
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("remote-compute-"):
            continue
        # ssh is a built-in family (no confined helper), not a byoc provider.
        if child.name == "remote-compute-ssh":
            continue
        pj = child / "provider.json"
        pp = child / "provider.py"
        if not (pj.exists() and pp.exists()):
            continue
        try:
            meta = json.loads(pj.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        pid = meta.get("id")
        if not pid:
            continue
        out[str(pid)] = {
            "id": str(pid),
            "dir": child,
            "provider_py": str(pp),
            "meta": meta,
        }
    return out


class ComputeManager:
    """One per session/kernel. Owns provider discovery, job bookkeeping, and a
    lazy background poller. Thread-safe for the handful of ops the dispatcher
    drives."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._providers = _discover_providers(Path(cfg.skills_dir))
        self._install_id = self._resolve_install_id()
        self._jobs: dict[str, dict] = {}
        # byoc sandbox reuse: provider-id -> sandbox_id (warm container).
        self._sandboxes: dict[str, str] = {}
        self._lock = threading.RLock()
        self._limit: int | None = None
        self._hpc_root = Path(cfg.data_dir) / "hpc"
        self._hpc_root.mkdir(parents=True, exist_ok=True)

    # --- discovery / capability ------------------------------------------
    def has_any_provider(self) -> bool:
        return bool(self._providers) or self._has_ssh_skill()

    def _has_ssh_skill(self) -> bool:
        """The ssh:* family is enabled by the remote-compute-ssh skill being
        installed (it ships the worked example + gate), not merely by the user
        happening to have an ~/.ssh/config."""
        return (Path(self.cfg.skills_dir) / "remote-compute-ssh").is_dir()

    def provider_caps(self) -> dict:
        return {
            f"byoc:{pid}": p["meta"].get("max_concurrent")
            for pid, p in self._providers.items()
        }

    @staticmethod
    def _resolve_install_id() -> str:
        """A stable per-install id used as the byoc sandbox owner tag. Persist
        it under the data dir so reconcile can find sandboxes across runs."""
        env = os.environ.get("OPENAI4S_INSTALL_ID")
        if env:
            return env
        path = Path.home() / ".openai4s" / "install-id"
        try:
            if path.exists():
                return path.read_text("utf-8").strip()
            path.parent.mkdir(parents=True, exist_ok=True)
            iid = uuid.uuid4().hex
            path.write_text(iid, encoding="utf-8")
            return iid
        except OSError:
            return uuid.uuid4().hex

    # --- provider family routing -----------------------------------------
    def _split(self, provider: str) -> tuple[str, str]:
        fam, _, rest = provider.partition(":")
        if fam not in ("byoc", "ssh") or not rest:
            raise ComputeError(
                f"unknown provider target {provider!r}; expected "
                f"'byoc:<id>' or 'ssh:<alias>'",
                "invalid_request",
            )
        return fam, rest

    def _byoc(self, pid: str) -> dict:
        p = self._providers.get(pid)
        if p is None:
            raise ComputeError(
                f"byoc provider {pid!r} is not configured (no "
                f"skills/remote-compute-{pid}/provider.json found)",
                "not_found",
            )
        return p

    # --- concurrency ------------------------------------------------------
    def _live_count(self) -> int:
        return sum(
            1
            for j in self._jobs.values()
            if j.get("status") in ("submitted", "running", "queued")
        )

    def set_concurrency(self, kw: dict) -> dict:
        with self._lock:
            self._limit = int(kw["max_concurrent"])
        return {"live": self._live_count(), "limit": self._limit}

    def status(self, kw: dict) -> dict:
        return {
            "live": self._live_count(),
            "limit": self._limit,
            "daemon_live": True,
            "provider_caps": self.provider_caps(),
        }

    # --- byoc helper transport -------------------------------------------
    def _run_helper(
        self,
        prov: dict,
        op: str,
        req: dict,
        creds: dict,
        stage: Path,
        expect_confined: bool = False,
    ) -> dict:
        """Spawn the confined helper in oneshot mode for one op. The credential
        rides on the helper's stdin (never its environment); req/reply cross
        via the stage dir."""
        (stage / "req.json").write_text(
            json.dumps({**req, "stage": str(stage), "install_id": self._install_id}),
            encoding="utf-8",
        )
        argv = [
            sys.executable,
            "-I",
            _HELPER_MAIN,
            "oneshot",
            prov["provider_py"],
            op,
            str(stage),
            "1" if expect_confined else "0",
        ]
        # Scrub inherited secrets from the child env; the helper's own prologue
        # also drops the provider's secret_env_prefixes.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("NGC_", "NVIDIA_", "HF_"))
        }
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, env=env)
        proc.stdin.write((json.dumps({"op": "auth", **creds}) + "\n").encode("utf-8"))
        proc.stdin.close()
        proc.wait()
        reply_path = stage / "reply.json"
        if not reply_path.exists():
            raise ComputeError(
                f"provider helper for op {op!r} exited (rc={proc.returncode}) "
                f"without a reply",
                "transient",
            )
        reply = json.loads(reply_path.read_text("utf-8"))
        if not reply.get("ok"):
            raise ComputeError(
                reply.get("msg") or "provider op failed",
                reply.get("kind") or "transient",
            )
        return reply

    def _provider_creds(self, prov: dict) -> dict:
        """Collect the provider's declared secret env vars into the auth
        payload the helper reads from stdin. The provider.json's
        ``helperEnv``/``secret_env`` lists which env keys to forward."""
        keys = prov["meta"].get("secret_env") or []
        return {k: os.environ[k] for k in keys if k in os.environ}

    # --- submit -----------------------------------------------------------
    def submit(self, kw: dict) -> dict:
        provider = kw["provider"]
        fam, rest = self._split(provider)
        with self._lock:
            if self._limit is not None and self._live_count() >= self._limit:
                raise ComputeError(
                    "session concurrency limit reached",
                    "session_concurrency_full",
                    {"live": self._live_count(), "limit": self._limit},
                )
        if fam == "ssh":
            return self._submit_ssh(rest, kw)
        return self._submit_byoc(rest, kw)

    def _stage_inputs(
        self, stage: Path, inputs: list | None, command: str, timeout_s: int
    ) -> Path:
        """Build the in.tar.gz the helper untars into /work: the wrapper, the
        run.sh (command), and every staged input flat in the root."""
        work = stage / "work"
        work.mkdir()
        wrapper = (_TMPL_DIR / "wrapper.sh.tmpl").read_text("utf-8")
        run = (
            (_TMPL_DIR / "run.sh.tmpl")
            .read_text("utf-8")
            .replace("{{COMMAND}}", command)
        )
        (work / "_openai4s_wrapper.sh").write_text(wrapper, encoding="utf-8")
        (work / "run.sh").write_text(run, encoding="utf-8")
        for inp in inputs or []:
            src = inp.get("src") or inp.get("remote_path")
            if not src:
                continue
            dst = inp.get("dst_filename") or Path(src).name
            src_path = Path(src) if os.path.isabs(src) else Path.cwd() / src
            if src_path.exists():
                shutil.copy2(src_path, work / dst)
        tgz = stage / "in.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(work, arcname=".")
        return tgz

    def _submit_byoc(self, pid: str, kw: dict) -> dict:
        prov = self._byoc(pid)
        creds = self._provider_creds(prov)
        job_id = "job-" + uuid.uuid4().hex[:12]
        timeout_s = int(kw.get("timeout_seconds") or 14400)
        with tempfile.TemporaryDirectory(prefix="openai4s-byoc-stage-") as td:
            stage = Path(td)
            # 1. create (or reuse) the sandbox.
            sid = self._sandboxes.get(pid) or kw.get("reuse_job_id")
            if not sid or not self._sandboxes.get(pid):
                spec = (kw.get("provider_params") or {}).get(pid, {})
                tags = {"openai4s-session": self._install_id, "openai4s-job": job_id}
                rep = self._run_helper(
                    prov,
                    "create",
                    {"spec": spec, "tags": tags, "app_name": "openai4s"},
                    creds,
                    stage,
                    expect_confined=False,
                )
                sid = rep["sandbox_id"]
                self._sandboxes[pid] = sid
            # 2. stage inputs then submit.
            self._stage_inputs(stage, kw.get("inputs"), kw["command"], timeout_s)
            self._run_helper(
                prov, "submit", {"sandbox_id": sid, "timeout": timeout_s}, creds, stage
            )
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "provider": f"byoc:{pid}",
                "sandbox_id": sid,
                "status": "running",
                "outputs": kw.get("outputs"),
                "creds": bool(creds),
            }
        return {
            "job_id": job_id,
            "status": "running",
            "concurrency": {"live": self._live_count(), "limit": self._limit},
            "egress": prov["meta"].get("egress"),
        }

    # --- ssh --------------------------------------------------------------
    def _submit_ssh(self, alias: str, kw: dict) -> dict:
        job_id = "job-" + uuid.uuid4().hex[:12]
        workdir = f"~/.openai4s-jobs/{job_id}"
        script = kw["command"]
        remote = (
            f"mkdir -p {workdir} && cd {workdir} && "
            f"cat > run.sh && nohup bash run.sh "
            f"> stdout.log 2> stderr.log & echo $!"
        )
        try:
            proc = subprocess.run(
                ["ssh", alias, remote],
                input=script.encode("utf-8"),
                capture_output=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            raise ComputeError(f"ssh submit failed: {e}", "transient")
        if proc.returncode != 0:
            raise ComputeError(
                proc.stderr.decode("utf-8", "replace") or "ssh submit failed",
                "transient",
            )
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "provider": f"ssh:{alias}",
                "alias": alias,
                "workdir": workdir,
                "status": "running",
                "pid": proc.stdout.decode().strip(),
                "outputs": kw.get("outputs"),
            }
        return {
            "job_id": job_id,
            "status": "running",
            "remote_workdir": workdir,
            "concurrency": {"live": self._live_count(), "limit": self._limit},
        }

    # --- result / harvest -------------------------------------------------
    def result(self, kw: dict) -> dict:
        job_id = kw["job_id"]
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ComputeError(f"no such job {job_id!r}", "not_found")
        fam, rest = self._split(job["provider"])
        if fam == "ssh":
            return self._result_ssh(job)
        return self._result_byoc(job)

    def _result_byoc(self, job: dict) -> dict:
        prov = self._byoc(job["provider"].split(":", 1)[1])
        creds = self._provider_creds(prov)
        with tempfile.TemporaryDirectory(prefix="openai4s-byoc-stage-") as td:
            stage = Path(td)
            rep = self._run_helper(
                prov,
                "wait",
                {"sandbox_id": job["sandbox_id"], "poll_seconds": 5},
                creds,
                stage,
            )
            if not rep.get("ready"):
                return {
                    "status": "running",
                    "job_id": job["job_id"],
                    "hint": "job still running — use wait_for_notification",
                }
            out_files = self._harvest(job["job_id"], stage)
        job["status"] = "failed" if rep.get("job_exit_code") else "done"
        return {
            "status": job["status"],
            "exit_code": rep.get("job_exit_code"),
            "output_files": out_files,
            "featured_files": out_files,
            "stdout_tail": rep.get("stdout_tail", ""),
            "stderr_tail": rep.get("stderr_tail", ""),
            "job_wall_s": rep.get("job_wall_s"),
            "left_on_remote": False,
        }

    def _harvest(self, job_id: str, stage: Path) -> list[str]:
        dest = self._hpc_root / job_id
        dest.mkdir(parents=True, exist_ok=True)
        tgz = stage / "out.tar.gz"
        files: list[str] = []
        if tgz.exists():
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(dest)
            for p in sorted(dest.rglob("*")):
                if p.is_file():
                    files.append(str(p))
        return files

    def _result_ssh(self, job: dict) -> dict:
        alias, workdir = job["alias"], job["workdir"]
        # Non-blocking: is the pid still alive?
        check = subprocess.run(
            [
                "ssh",
                alias,
                f"kill -0 {job['pid']} 2>/dev/null && echo RUNNING || "
                f"cat {workdir}/.rc 2>/dev/null || echo 0",
            ],
            capture_output=True,
            timeout=30,
        )
        out = check.stdout.decode().strip()
        if out == "RUNNING":
            return {"status": "running", "job_id": job["job_id"]}
        dest = self._hpc_root / job["job_id"]
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "scp",
                "-O",
                "-q",
                f"{alias}:{workdir}/stdout.log",
                f"{alias}:{workdir}/stderr.log",
                str(dest),
            ],
            capture_output=True,
            timeout=120,
        )
        job["status"] = "done"
        return {
            "status": "done",
            "exit_code": 0,
            "output_files": [str(p) for p in dest.iterdir()],
            "featured_files": [],
            "remote_workdir": workdir,
            "left_on_remote": True,
        }

    # --- cancel / close / ssh command / scp -------------------------------
    def cancel(self, kw: dict) -> dict:
        with self._lock:
            job = self._jobs.get(kw["job_id"])
        if job is None:
            raise ComputeError(f"no such job {kw['job_id']!r}", "not_found")
        fam, rest = self._split(job["provider"])
        if fam == "ssh":
            subprocess.run(
                ["ssh", job["alias"], f"kill -TERM {job['pid']}"],
                capture_output=True,
                timeout=30,
            )
        else:
            prov = self._byoc(rest)
            with tempfile.TemporaryDirectory(prefix="openai4s-byoc-stage-") as td:
                self._run_helper(
                    prov,
                    "terminate",
                    {"sandbox_id": job["sandbox_id"]},
                    self._provider_creds(prov),
                    Path(td),
                )
        job["status"] = "cancelled"
        return {"status": "cancelled"}

    def close(self, kw: dict) -> dict:
        provider = kw["provider"]
        fam, rest = self._split(provider)
        if fam == "byoc":
            sid = self._sandboxes.pop(rest, None)
            if sid:
                prov = self._byoc(rest)
                with tempfile.TemporaryDirectory(prefix="openai4s-byoc-stage-") as td:
                    try:
                        self._run_helper(
                            prov,
                            "terminate",
                            {"sandbox_id": sid},
                            self._provider_creds(prov),
                            Path(td),
                        )
                    except ComputeError:
                        pass
        for jid in kw.get("job_ids") or []:
            j = self._jobs.get(jid)
            if j and j.get("status") in ("submitted", "running", "queued"):
                j["status"] = "closed"
        return {"status": "closed"}

    def ssh(self, kw: dict) -> dict:
        """One synchronous command (call_command). byoc runs it inside the
        warm sandbox; ssh runs it over the alias."""
        provider = kw["provider"]
        fam, rest = self._split(provider)
        cmd = kw["command"]
        timeout_s = int(kw.get("timeout_seconds") or 60)
        if fam == "ssh":
            shell = ["ssh"]
            if kw.get("login_shell"):
                shell += ["-t"]
            shell += [rest, cmd]
            proc = subprocess.run(shell, capture_output=True, timeout=timeout_s)
            return {
                "stdout": proc.stdout.decode("utf-8", "replace")[:65536],
                "stderr": proc.stderr.decode("utf-8", "replace")[:65536],
                "exit_code": proc.returncode,
            }
        raise ComputeError(
            "call_command on a byoc provider requires a live sandbox; "
            "submit a job instead",
            "invalid_request",
        )

    def scp(self, kw: dict) -> dict:
        if self._split(kw["provider"])[0] != "ssh":
            raise ComputeError("download/upload is ssh-only", "invalid_request")
        alias = kw["provider"].split(":", 1)[1]
        if kw["direction"] == "down":
            local = kw.get("local") or Path(kw["remote"]).name
            subprocess.run(
                ["scp", "-O", "-q", f"{alias}:{kw['remote']}", local],
                capture_output=True,
                timeout=300,
            )
            return {"local": str(local)}
        subprocess.run(
            ["scp", "-O", "-q", kw["local"], f"{alias}:{kw['remote']}"],
            capture_output=True,
            timeout=300,
        )
        return {"remote": kw["remote"]}
