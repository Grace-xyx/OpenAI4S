# Security

> ⚠️ Read this before exposing the daemon beyond `localhost`.

The daemon runs agent-authored code with **no OS-level sandbox** (no Seatbelt / bubblewrap) — `kernel/execute`, `compute/jobs`, and `host.bash` are equivalent to a shell on the host. This is fine for a single-user local tool bound to `127.0.0.1` (the default). On top of that, [`openai4s.security`](../openai4s/security) adds software layers reverse-engineered from Claude Science — all **opt-out via env**, all **fail-open** when no base model is set:

| layer | env (default) | what it does |
|---|---|---|
| **Pre-exec classifier** | `OPENAI4S_SAFETY` (`heuristic`) | screens every *agent-authored* cell before it runs (`heuristic` / `llm` / `off`); your own Notebook cells are never screened |
| **`dlopen` audit hook** | `OPENAI4S_SAFETY_AUDIT_HOOK` (on) | `sys.addaudithook` refuses `ctypes.dlopen` of a `.so` from an agent-writable path |
| **Biosecurity screener** | `OPENAI4S_BIOSECURITY` (on) | trajectory screener (ALLOW / ESCALATE / BLOCK) on biosecurity-relevant content |
| **Injection detector** | `OPENAI4S_INJECTION_SCAN` (on) | annotates tool-returned content (web / PDF / MCP) so the model treats it as **data, not instructions** |
| **Egress allowlist** | `OPENAI4S_EGRESS` (`off`) | fences `web_fetch` / `web_search` / `bash` to science APIs & package indexes; blocked domains recover via `host.request_network_access(domain=…)`, which **you** approve |

Additional enforcement: an opencode-style **permission broker** gates risk-bearing tools, a **secret-file guard** blocks `.env` / `*.key` / `id_rsa` from all file tools, and every file/shell op is **workspace-jailed**.

## Remote access

The daemon binds `127.0.0.1` by default. Reach the UI over an SSH tunnel — **never** expose `0.0.0.0` on an untrusted network:

```bash
ssh -L 8760:127.0.0.1:8760 user@your-host
```

If you must bind a non-loopback address (`OPENAI4S_HOST=0.0.0.0`) or set `OPENAI4S_REQUIRE_TOKEN=1`, the server prints a one-time access token at startup and rejects any request without `?token=…` (`401`).
