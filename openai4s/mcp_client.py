"""Minimal MCP (Model Context Protocol) stdio client — pure stdlib.

Speaks newline-delimited JSON-RPC 2.0 to a spawned MCP server process, enough to
power the Connectors feature: handshake (initialize + initialized), tools/list,
tools/call. A process-wide MCPManager caches one live connection per connector id
so repeated tool calls reuse the same server.

Not a full MCP implementation (no resources/prompts/sampling), but interoperable
with any standard stdio MCP server for the tools surface the agent uses.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
_DEFAULT_TIMEOUT = 30.0


class MCPError(RuntimeError):
    pass


class MCPConnection:
    def __init__(
        self, command: list[str], env: dict | None = None, cwd: str | None = None
    ):
        self.command = command
        self._id = 0
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )
        self._init()

    # -- wire ----------------------------------------------------------------
    def _send(self, obj: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _read_reply(self, want_id: int) -> dict:
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise MCPError("MCP server closed the connection")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # skip notifications / unrelated ids
            if msg.get("id") != want_id:
                continue
            if "error" in msg and msg["error"] is not None:
                raise MCPError(str(msg["error"].get("message") or msg["error"]))
            return msg.get("result") or {}

    def _request(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            self._id += 1
            mid = self._id
            self._send(
                {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
            )
            return self._read_reply(mid)

    def _notify(self, method: str, params: dict | None = None) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- lifecycle -----------------------------------------------------------
    def _init(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "openai4s", "version": "1.0.0"},
            },
        )
        try:
            self._notify("notifications/initialized")
        except Exception:  # noqa: BLE001
            pass

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- tools ---------------------------------------------------------------
    def list_tools(self) -> list[dict]:
        res = self._request("tools/list")
        return res.get("tools", []) if isinstance(res, dict) else []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        res = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        # normalize content blocks -> plain text for the agent
        text_parts = []
        for block in res.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return {
            "is_error": bool(res.get("isError")),
            "text": "\n".join(text_parts),
            "raw": res,
        }


class MCPManager:
    """One live connection per connector id (lazily connected, cached)."""

    def __init__(self) -> None:
        self._conns: dict[str, MCPConnection] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _argv(config: dict) -> list[str]:
        cmd = config.get("command")
        args = config.get("args") or []
        if isinstance(cmd, list):
            argv = list(cmd) + list(args)
        elif isinstance(cmd, str) and cmd.strip():
            argv = cmd.split() + list(args)
        else:
            raise MCPError("connector has no command")
        return argv

    def _connect(self, config: dict) -> MCPConnection:
        import os

        env = dict(os.environ)
        env.update(config.get("env") or {})
        return MCPConnection(self._argv(config), env=env, cwd=config.get("cwd"))

    def get(self, connector_id: str, config: dict) -> MCPConnection:
        with self._lock:
            conn = self._conns.get(connector_id)
            if conn is not None and conn.alive():
                return conn
            if conn is not None:
                conn.close()
            conn = self._connect(config)
            self._conns[connector_id] = conn
            return conn

    def probe(self, config: dict) -> dict:
        """Connect fresh, list tools, close. Returns {ok, tools|error}."""
        try:
            conn = self._connect(config)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        try:
            tools = conn.list_tools()
            return {"ok": True, "tools": tools}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        finally:
            conn.close()

    def list_tools(self, connector_id: str, config: dict) -> list[dict]:
        return self.get(connector_id, config).list_tools()

    def call_tool(
        self, connector_id: str, config: dict, tool: str, arguments: dict | None = None
    ) -> dict:
        return self.get(connector_id, config).call_tool(tool, arguments)

    def disconnect(self, connector_id: str) -> None:
        with self._lock:
            conn = self._conns.pop(connector_id, None)
        if conn is not None:
            conn.close()

    def shutdown(self) -> None:
        with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
        for c in conns:
            c.close()


# a process-wide manager (the daemon is single-process)
_MANAGER: MCPManager | None = None


def manager() -> MCPManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = MCPManager()
    return _MANAGER


def example_server_config() -> dict:
    """Config for the bundled example server (always available)."""
    return {"command": [sys.executable, "-m", "openai4s.mcp_servers.example_server"]}
