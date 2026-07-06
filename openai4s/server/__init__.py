"""HTTP gateway: full web UI + REST + WebSocket (pure stdlib).

`serve`/`build_server` now front the rich openai4s-local web UI backed by the
openai4s Code-as-Action engine (see gateway.py). The original minimal single-
page daemon is preserved in daemon.py as `serve_minimal`/`build_minimal_server`.
"""
from openai4s.server.gateway import build_app_server as build_server
from openai4s.server.gateway import serve_app as serve

__all__ = ["build_server", "serve"]
