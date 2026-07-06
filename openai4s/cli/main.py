"""openai4s CLI: serve / status / stop / url / run / setup.

  openai4s serve    start the daemon (foreground; use & or nohup to background)
  openai4s status   is the daemon up? (reads pidfile + /health)
  openai4s stop     stop the running daemon
  openai4s url      print the local web UI url
  openai4s run "<task>"   run one Code-as-Action task (in-process, no daemon)
  openai4s setup    create the four default conda envs from envs/*.yml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from openai4s.config import get_config


def _write_state(cfg) -> None:
    cfg.pidfile.write_text(str(os.getpid()), "utf-8")
    cfg.statefile.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": cfg.host,
                "port": cfg.port,
                "started_at": int(time.time()),
            }
        ),
        "utf-8",
    )


def _clear_state(cfg) -> None:
    for p in (cfg.pidfile, cfg.statefile):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _read_pid(cfg) -> int | None:
    try:
        return int(cfg.pidfile.read_text("utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _url(cfg) -> str:
    return f"http://{cfg.host}:{cfg.port}/"


def cmd_serve(args) -> int:
    from openai4s.server import serve

    cfg = get_config()
    existing = _read_pid(cfg)
    if existing and _pid_alive(existing):
        print(f"daemon already running (pid {existing}) at {_url(cfg)}")
        return 1
    _write_state(cfg)
    print(f"openai4s listening at {_url(cfg)} (model={cfg.llm.model})")
    print("web UI ready. Ctrl-C to stop.")

    def _graceful(signum, frame):
        _clear_state(cfg)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _graceful)
    if not os.environ.get("OPENAI4S_NO_OPEN") and not getattr(args, "no_open", False):

        def _open():
            time.sleep(1.0)
            try:
                import webbrowser

                webbrowser.open(_url(cfg))
            except Exception:
                pass

        import threading

        threading.Thread(target=_open, daemon=True).start()
    try:
        serve(cfg, block=True)
    finally:
        _clear_state(cfg)
    return 0


def cmd_status(args) -> int:
    cfg = get_config()
    pid = _read_pid(cfg)
    if not pid or not _pid_alive(pid):
        print("daemon: not running")
        return 1
    # confirm via /health
    try:
        with urllib.request.urlopen(_url(cfg) + "health", timeout=3) as r:
            health = json.loads(r.read().decode("utf-8"))
        print(f"daemon: running (pid {pid}) at {_url(cfg)}")
        print(f"  model    : {health.get('model')}")
        print(f"  data_dir : {health.get('data_dir')}")
        return 0
    except urllib.error.URLError:
        print(f"daemon: pid {pid} alive but /health unreachable")
        return 2


def cmd_stop(args) -> int:
    cfg = get_config()
    pid = _read_pid(cfg)
    if not pid or not _pid_alive(pid):
        print("daemon: not running")
        _clear_state(cfg)
        return 1
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    _clear_state(cfg)
    print(f"daemon stopped (pid {pid})")
    return 0


def cmd_url(args) -> int:
    print(_url(get_config()))
    return 0


def cmd_run(args) -> int:
    from openai4s.agent import Agent

    cfg = get_config()
    result = Agent(cfg=cfg, verbose=args.verbose).run(args.task)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n=== stop_reason:", result["stop_reason"], "===")
        if result.get("submitted_output"):
            print(
                "submitted_output:",
                json.dumps(result["submitted_output"], ensure_ascii=False, indent=2),
            )
        if result.get("final_message"):
            print("final:", result["final_message"])
    return 0


# --------------------------------------------------------------------------- #
#  setup — create the four default conda environments from envs/*.yml
# --------------------------------------------------------------------------- #
# The four default envs, in the order we create them (python first: it's the
# default kernel env). Names must match the `name:` in each envs/<name>.yml.
_DEFAULT_ENVS = ["python", "phylo", "r", "struct"]

# Conda-family tools we know how to drive, fastest first.
_CONDA_TOOLS = ["micromamba", "mamba", "conda"]


def _envs_dir() -> Path:
    """The repo's ``envs/`` directory (sibling of the ``openai4s`` package)."""
    return Path(__file__).resolve().parents[2] / "envs"


def _find_conda_tool() -> str | None:
    """First available of micromamba / mamba / conda on PATH, or None."""
    for tool in _CONDA_TOOLS:
        if shutil.which(tool):
            return tool
    return None


def _existing_env_names() -> set[str]:
    """Names of conda envs that already exist.

    Prefers the daemon's own discovery (:mod:`openai4s.kernel.environments`,
    which honours ``OPENAI4S_ENV_ROOTS`` and the reference-daemon envs dir);
    falls back to ``conda env list`` parsing if that import isn't available."""
    try:
        from openai4s.kernel.environments import discover_environments

        return {e.name for e in discover_environments(force=True)}
    except Exception:  # noqa: BLE001 — fall back to CLI probing
        pass
    tool = _find_conda_tool()
    if not tool:
        return set()
    try:
        out = subprocess.run(
            [tool, "env", "list"], capture_output=True, text=True, timeout=30
        )
    except Exception:  # noqa: BLE001
        return set()
    names: set[str] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # rows look like:  "python   *  /path/to/envs/python"
        first = line.split()[0]
        if first and first != "*":
            names.add(first)
        # also take the leaf dir of the prefix path, if any
        path = line.split()[-1]
        if os.sep in path:
            names.add(Path(path).name)
    return names


def _create_cmd(tool: str, name: str, yml: Path) -> list[str]:
    """The env-creation argv for ``tool`` from spec file ``yml``.

    micromamba/mamba/conda all accept ``env create -f <file>``; conda derives
    the env name from the file's ``name:`` field."""
    return [tool, "env", "create", "-f", str(yml)]


def cmd_setup(args) -> int:
    tool = _find_conda_tool()
    if not tool:
        print("error: no conda/mamba/micromamba found on PATH.", file=sys.stderr)
        print(
            "       install one (e.g. micromamba) and re-run `openai4s setup`.",
            file=sys.stderr,
        )
        return 1

    envs_dir = _envs_dir()
    if not envs_dir.is_dir():
        print(f"error: envs directory not found: {envs_dir}", file=sys.stderr)
        return 1

    if args.only:
        if args.only not in _DEFAULT_ENVS:
            print(
                f"error: unknown env '{args.only}' "
                f"(choices: {', '.join(_DEFAULT_ENVS)})",
                file=sys.stderr,
            )
            return 1
        wanted = [args.only]
    else:
        wanted = list(_DEFAULT_ENVS)

    existing = set() if args.dry_run else _existing_env_names()

    print(
        f"using '{tool}' to create envs from {envs_dir}"
        + (" (dry-run)" if args.dry_run else "")
    )
    created = 0
    failed = 0
    for name in wanted:
        yml = envs_dir / f"{name}.yml"
        if not yml.is_file():
            print(f"  [{name}] skip: spec file missing ({yml})")
            failed += 1
            continue
        if name in existing:
            print(f"  [{name}] already exists — skipping")
            continue
        cmd = _create_cmd(tool, name, yml)
        if args.dry_run:
            print(f"  [{name}] would run: {' '.join(cmd)}")
            continue
        print(f"  [{name}] creating… ({' '.join(cmd)})")
        try:
            rc = subprocess.run(cmd).returncode
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] error: {exc}", file=sys.stderr)
            failed += 1
            continue
        if rc == 0:
            print(f"  [{name}] created")
            created += 1
        else:
            print(f"  [{name}] FAILED (exit {rc})", file=sys.stderr)
            failed += 1

    if args.dry_run:
        return 0
    print(f"done: {created} created, {failed} failed")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openai4s", description="openai4s CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="start the daemon (foreground)")
    ps.add_argument("--no-open", action="store_true", help="don't open a browser")
    ps.set_defaults(fn=cmd_serve)
    sub.add_parser("status", help="check daemon status").set_defaults(fn=cmd_status)
    sub.add_parser("stop", help="stop the daemon").set_defaults(fn=cmd_stop)
    sub.add_parser("url", help="print the web UI url").set_defaults(fn=cmd_url)

    pr = sub.add_parser("run", help="run one Code-as-Action task in-process")
    pr.add_argument("task", help="the task description")
    pr.add_argument("--json", action="store_true", help="emit full JSON result")
    pr.add_argument("-v", "--verbose", action="store_true", help="stream turns")
    pr.set_defaults(fn=cmd_run)

    pu = sub.add_parser(
        "setup", help="create the four default conda envs from envs/*.yml"
    )
    pu.add_argument(
        "--only",
        metavar="NAME",
        choices=_DEFAULT_ENVS,
        help="create just one env (%(choices)s)",
    )
    pu.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run, without executing",
    )
    pu.set_defaults(fn=cmd_setup)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
