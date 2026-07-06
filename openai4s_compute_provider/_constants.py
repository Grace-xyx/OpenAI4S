"""Shared constants for the BYOC provider runtime.

Kept in one place so the resident, the control channel, and any provider shim
all agree on the same wire limits, exit codes, and sandbox paths.
"""
from __future__ import annotations

import re

# ── stream / harvest limits ──────────────────────────────────────────────────
TAIL_BYTES = 8 * 1024
TAIL_RING_BYTES = 256 * 1024
CHUNK = 64 * 1024
IDLE_TIMEOUT_S = 15 * 60
# Compressed-bytes default if the host omits output_cap_bytes (it normally
# doesn't). Intentionally 2x the host's 5 GiB decompressed cap.
COMPRESSED_CAP_DEFAULT = 10 * 2**30

# ── sandbox paths ────────────────────────────────────────────────────────────
# Paired with the host's BYOC stage-prefix (its transport's mkStage()).
STAGE_PREFIX = "/tmp/openai4s-byoc-stage-"
# Remote workdir under the provider sandbox; the wrapper + harvest paths are
# all relative to this. A local-subprocess simulator may rewrite it for tests.
WORK = "/work"

# ── exit codes ───────────────────────────────────────────────────────────────
EXIT_PROTOCOL = 70  # auth-handshake violation or signal — sysexits.h EX_SOFTWARE
EXIT_UNCONFINED = 71  # oneshot self-enforced confinement check failed

# ── fd-3 control channel (repl mode only) ────────────────────────────────────
# The per-op oneshot helper reads its credential from stdin (read_auth(fd=0))
# and writes nothing back; it has no fd-3. The repl kernel uses fd-3 because
# its stdin/stdout are the cell-execution channel, so the bidirectional
# ready/auth/event side-band needs a separate fd. fd 3 is a host-owned
# socketpair end carrying newline-framed JSON, 256 KiB line-capped on both
# sides.
FD_CTRL = 3
LINE_CAP = 256 * 1024

# ── classification ───────────────────────────────────────────────────────────
# .job_env keys that look like forwarded credentials (the only values worth
# scrubbing from stdout/stderr tails — agent-supplied job_env values aren't).
CRED_KEY_RE = re.compile(
    r"(?i)(?:^|_)(?:TOKEN|SECRET|KEY|PASS(?:WORD|WD)?|PWD|PW|PAT|CREDENTIAL|AUTH|BEARER|COOKIE)(?:_|$)"
)
BASE_ERROR_KINDS = frozenset(
    {
        "not_found",
        "unauthorized",
        "rate_limited",
        "quota_exhausted",
        "invalid_request",
        "transient",
        "image_build_failed",
        "network_denied",
        "network_bridge_down",
        "ownership_mismatch",
        "provider_degraded",
        "result_rejected",
    }
)
