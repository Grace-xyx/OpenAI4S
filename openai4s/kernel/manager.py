"""Host-side kernel manager.

Spawns worker.py as a long-lived subprocess and drives the JSON-per-line
protocol. When the worker emits a `host_call` frame mid-execution, this manager
routes it to the host RPC dispatcher and writes back a `host_response` frame —
this is the inner synchronous RPC loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

_WORKER = Path(__file__).resolve().parent / "worker.py"

# A host-call dispatcher: (method:str, args:list) -> data. Raises to signal error.
Dispatcher = Callable[[str, list], Any]


class Kernel:
    def __init__(
        self,
        dispatcher: Dispatcher | None = None,
        cwd: str | None = None,
        mode: str = "repl",
        python: str | None = None,
        env_root: str | None = None,
        env_name: str | None = None,
    ):
        self.dispatcher = dispatcher
        self.mode = mode
        self.cwd = cwd
        # Which interpreter runs worker.py, and (for a conda env) its prefix — so
        # cells run in a *selected* prebuilt environment rather than always the
        # daemon's own Python. Defaults to sys.executable (the base kernel).
        self.python = python or sys.executable
        self.env_root = env_root
        self.env_name = env_name
        self.generation = 0  # bumped on every (re)spawn
        self._proc = self._spawn()

    def _spawn(self) -> "subprocess.Popen":
        return subprocess.Popen(
            [self.python, "-u", str(_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
            env=self._child_env(),
        )

    def _child_env(self) -> dict:
        import os

        env = dict(os.environ)
        # Ensure the worker can import openai4s.* even when it runs under a
        # different (conda) interpreter than the daemon.
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = repo_root + (
            (":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
        )
        # splice gate: tell the worker which SDK surface to assemble.
        env["OPENAI4S_KERNEL_MODE"] = self.mode
        # Activate the selected conda env for the worker's subprocesses: prepend
        # its bin/ to PATH (so pip/mafft/iqtree/Rscript resolve inside the env)
        # and advertise the prefix, matching a `conda activate`.
        if self.env_root:
            bindir = str(Path(self.env_root) / "bin")
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
            env["CONDA_PREFIX"] = str(self.env_root)
            if self.env_name:
                env["CONDA_DEFAULT_ENV"] = self.env_name
        return env

    def _send(self, obj: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _readline(self) -> dict | None:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            return {}
        return json.loads(line)

    def execute(
        self,
        code: str,
        origin: str = "agent",
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Run one cell; block until the response frame, servicing host_calls.

        `on_chunk` (if given) is invoked with each live stdout chunk — used by
        the background executor to expose a running cell's output to exec_peek.
        """
        cell_id = str(uuid.uuid4())
        self._send({"type": "execute", "id": cell_id, "code": code, "origin": origin})

        stdout_chunks: list[str] = []
        while True:
            frame = self._readline()
            if frame is None:
                # Worker died; surface stderr for debugging.
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"kernel worker exited unexpectedly: {err}")
            ftype = frame.get("type")
            if ftype == "response":
                if stdout_chunks and not frame.get("stdout"):
                    frame["stdout"] = "".join(stdout_chunks)
                return frame
            if ftype == "host_call":
                self._service_host_call(frame)
            elif ftype == "stdout_chunk":
                text = frame.get("text", "")
                stdout_chunks.append(text)
                if on_chunk is not None and text:
                    on_chunk(text)
            elif ftype == "log":
                # diagnostic from worker; ignore or log
                pass

    @property
    def pid(self) -> int:
        return self._proc.pid

    def interrupt(self) -> None:
        """Deliver ONE SIGINT to the worker ( exec_interrupt).

        The worker's one-shot handler raises KeyboardInterrupt inside user code
        and self-disarms, so the interrupt stops the cell but keeps the kernel
        (and its namespace) alive.
        """
        import os
        import signal

        try:
            os.kill(self._proc.pid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

    def _service_host_call(self, frame: dict) -> None:
        call_id = frame.get("id")
        method = frame.get("method", "")
        args = frame.get("args", [])
        if self.dispatcher is None:
            self._send(
                {
                    "type": "host_response",
                    "id": call_id,
                    "error": "no host dispatcher configured",
                }
            )
            return
        try:
            data = self.dispatcher(method, args)
            # soft-fail contract: a single-key {"error": msg} return is a
            # soft failure the worker must raise, not a normal result.
            if isinstance(data, dict) and set(data.keys()) == {"error"}:
                self._send(
                    {"type": "host_response", "id": call_id, "error": data["error"]}
                )
            else:
                self._send({"type": "host_response", "id": call_id, "data": data})
        except Exception as e:  # noqa: BLE001
            self._send({"type": "host_response", "id": call_id, "error": str(e)})

    def restart(self) -> None:
        """Tear down the worker and spawn a clean one — a brand-new namespace.

        Used after a mid-task ``pip install`` so freshly installed packages are
        picked up by a fresh process, and to clear a wedged/polluted kernel. The
        caller is responsible for re-running any bootstrap (skill sidecars, etc.)
        against the new process — the ``Kernel`` object itself is reused so all
        references held by the session stay valid.
        """
        old = self._proc
        try:
            old.stdin and old.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
            old.stdin and old.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            old.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                old.kill()
                old.wait(timeout=2)  # reap so we don't leak a zombie per restart
            except Exception:  # noqa: BLE001
                pass
        for stream in (old.stdin, old.stdout, old.stderr):
            try:
                stream and stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._proc = self._spawn()
        self.generation += 1

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def shutdown(self) -> None:
        try:
            self._send({"type": "shutdown"})
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()

    def __enter__(self) -> "Kernel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
