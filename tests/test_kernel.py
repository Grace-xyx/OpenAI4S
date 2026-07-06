"""Kernel tests: persistent namespace, print capture, error attribution,
usage accounting, and host_call RPC round-trip (dispatcher stubbed)."""
import pytest

from openai4s.kernel import Kernel


def _echo_dispatcher(method, args):
    if method == "ping":
        return "pong"
    if method == "add":
        return sum(args[0]["nums"])
    raise ValueError(f"unknown method {method}")


def test_print_capture():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute("print('hello')")
        assert r["stdout"] == "hello\n"
        assert r["error"] is None


def test_persistent_namespace():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        k.execute("x = 41")
        r = k.execute("print(x + 1)")
        assert r["stdout"].strip() == "42"


def test_expr_echo():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute("21 * 2")
        assert r["stdout"].strip() == "42"


def test_error_lineno():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute("a = 1\nb = 2\nraise ValueError('boom')")
        assert r["error"] is not None
        assert "ValueError" in r["error"]
        assert r["trace"]["error_lineno"] == 3


def test_usage_accounting():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute("sum(range(1000))")
        u = r["usage"]
        assert set(u) == {"wall_s", "cpu_s", "peak_rss_kb"}
        assert u["wall_s"] >= 0 and u["peak_rss_kb"] > 0


def test_host_call_roundtrip():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute("reply = host._call('ping', [])\n" "print(reply)")
        assert r["stdout"].strip() == "pong"


def test_host_call_with_args():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute("print(host._call('add', [{'nums': [1, 2, 3, 4]}]))")
        assert r["stdout"].strip() == "10"


def test_host_call_error_propagates():
    with Kernel(dispatcher=_echo_dispatcher) as k:
        r = k.execute(
            "try:\n"
            "    host._call('nope', [])\n"
            "except RuntimeError as e:\n"
            "    print('caught:', 'unknown method' in str(e))"
        )
        assert r["stdout"].strip() == "caught: True"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
