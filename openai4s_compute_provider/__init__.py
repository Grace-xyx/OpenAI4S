"""SDK for BYOC (bring-your-own-compute) providers — the shared hardening and
lifecycle every confined provider process runs.

A provider is a ``provider.py`` that exports ``PROVIDER = <ByocProvider impl>``;
the ``__main__`` entrypoint loads it and runs either the per-op oneshot helper
(``run_oneshot``) or the long-lived repl kernel (``run_repl``). Both share one
prologue that scrubs the environment of secrets BEFORE any provider import and
BEFORE the credential is read, so a leak is impossible by construction.

This package is intentionally split by concern; import from the top level:

    from openai4s_compute_provider import WORK, ByocError, ExecResult

Layout:
  _constants.py  wire limits, exit codes, sandbox paths, error kinds
  _protocol.py   the ByocProvider / ExecResult contract + ByocError
  _channel.py    fd-3 control channel, auth handshake, stdout scrubber
  _resident.py   ByocResident — the prologue + oneshot/repl op loop

Stdlib-only. Provider shims (the only files that import a third-party SDK)
live in ``skills/remote-compute-<id>/provider.py``.
"""
from __future__ import annotations

from ._channel import ScrubWriter, read_auth, write_event, write_ready
from ._constants import (
    BASE_ERROR_KINDS,
    COMPRESSED_CAP_DEFAULT,
    EXIT_PROTOCOL,
    IDLE_TIMEOUT_S,
    STAGE_PREFIX,
    TAIL_BYTES,
    WORK,
)
from ._protocol import ByocError, ByocProvider, ExecResult
from ._resident import ByocResident

__all__ = [
    "ByocError",
    "ByocProvider",
    "ByocResident",
    "ExecResult",
    "ScrubWriter",
    "read_auth",
    "write_event",
    "write_ready",
    "BASE_ERROR_KINDS",
    "COMPRESSED_CAP_DEFAULT",
    "EXIT_PROTOCOL",
    "IDLE_TIMEOUT_S",
    "STAGE_PREFIX",
    "TAIL_BYTES",
    "WORK",
]
