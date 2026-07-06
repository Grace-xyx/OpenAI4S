"""Offline unit tests for the NVIDIA byoc compute provider.

These exercise the provider shim (skills/remote-compute-nvidia/provider.py) and
the host-side discovery path WITHOUT Docker, a GPU, or any network: every
`docker` invocation is intercepted by a fake subprocess layer, so the whole
suite runs on a laptop CI runner. What we assert:

  * the provider is discoverable (provider.json → id "nvidia")
  * both forms (hosted / self_hosted) build the right `docker run` argv
  * exec injects the endpoint + key env the job's run.sh reads
  * docker stderr maps onto the right structured ByocError kind
  * credentials never leak into a stdout/stderr tail (token scrub)
  * terminate is idempotent on an already-gone container
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest

from openai4s.compute.manager import _discover_providers

_REPO = Path(__file__).resolve().parent.parent
_PROVIDER_DIR = _REPO / "skills" / "remote-compute-nvidia"


def _load_provider_module():
    """Import the provider.py the same way the confined loader does — by file
    location, so the on-disk skill is what's under test."""
    # the base package it imports must be importable
    import openai4s_compute_provider  # noqa: F401

    spec = importlib.util.spec_from_file_location(
        "nvidia_provider_under_test", _PROVIDER_DIR / "provider.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeCompleted:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def provider(monkeypatch):
    """A NvidiaProvider whose `docker version` probe passes, with a recorder
    capturing every `docker` argv the provider shells out to."""
    mod = _load_provider_module()
    calls: list[list[str]] = []
    scripted: dict[str, _FakeCompleted] = {}

    def fake_run(argv, **kw):
        calls.append(list(argv))
        # `docker version` is the availability probe — always succeed.
        if argv[:2] == ["docker", "version"]:
            return _FakeCompleted(0, b"Docker version 27.0\n")
        key = argv[1] if len(argv) > 1 else ""
        return scripted.get(key, _FakeCompleted(0, b"cid-deadbeef\n"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    p = mod.NvidiaProvider(repl=False)
    p.import_and_patch()  # verifies docker present (mocked)
    return p, mod, calls, scripted


# --- discovery -----------------------------------------------------------


def test_nvidia_provider_is_discoverable():
    provs = _discover_providers(_REPO / "skills")
    assert "nvidia" in provs
    meta = provs["nvidia"]["meta"]
    assert meta["id"] == "nvidia"
    assert "NGC_API_KEY" in meta["secret_env"]
    assert "NVIDIA_API_KEY" in meta["secret_env"]


# --- create: two forms ---------------------------------------------------


def test_create_self_hosted_builds_gpu_run(provider):
    p, _mod, calls, _scripted = provider
    p.apply_auth({"NGC_API_KEY": "nvapi-secretkey12345"})
    cid = p.create_sandbox(
        {"mode": "self_hosted", "image": "nvcr.io/nim/meta/esmfold2:1.0.0"},
        install_id="inst-1",
    )
    assert cid == "cid-deadbeef"
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    assert "--gpus" in run and "all" in run
    assert "nvcr.io/nim/meta/esmfold2:1.0.0" in run
    # ownership label stamped
    assert any(a == "openai4s-install-id=inst-1" for a in run)
    # NGC login happened before run
    assert any(c[:2] == ["docker", "login"] for c in calls)


def test_create_hosted_builds_keepalive_no_gpu(provider):
    p, mod, calls, _scripted = provider
    p.apply_auth({"NVIDIA_API_KEY": "nvapi-hostedkey6789"})
    cid = p.create_sandbox({"mode": "hosted"}, install_id="inst-2")
    assert cid == "cid-deadbeef"
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    assert "--gpus" not in run  # hosted needs no local GPU
    assert mod.HOSTED_KEEPALIVE_IMAGE in run
    assert "infinity" in run


def test_create_rejects_bad_mode(provider):
    p, mod, _calls, _scripted = provider
    with pytest.raises(mod.ByocError) as ei:
        p.create_sandbox({"mode": "quantum"}, install_id="inst-3")
    assert ei.value.kind == "invalid_request"


def test_self_hosted_requires_image(provider):
    p, mod, _calls, _scripted = provider
    with pytest.raises(mod.ByocError) as ei:
        p.create_sandbox({"mode": "self_hosted"}, install_id="inst-4")
    assert ei.value.kind == "invalid_request"


# --- exec: env injection -------------------------------------------------


def test_exec_injects_hosted_endpoint_and_key(provider, monkeypatch):
    p, mod, _calls, _scripted = provider
    p.apply_auth({"NVIDIA_API_KEY": "nvapi-execkeyABCDEF"})
    p._mode = "hosted"

    captured = {}

    class _FakeProc:
        stdin = None
        stdout = None
        stderr = None

        def wait(self):
            return 0

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    p.exec("cid-x", ["bash", "run.sh"])
    argv = captured["argv"]
    joined = " ".join(argv)
    assert f"OPENAI4S_NIM_URL={mod.HOSTED_URL}" in joined
    assert f"OPENAI4S_NIM_HEALTH={mod.HEALTH_PATH}" in joined
    assert "NVIDIA_API_KEY=nvapi-execkeyABCDEF" in joined


def test_exec_self_hosted_uses_localhost(provider, monkeypatch):
    p, mod, _calls, _scripted = provider
    p._mode = "self_hosted"

    class _FakeProc:
        stdin = None
        stdout = None
        stderr = None

        def wait(self):
            return 0

    captured = {}
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda argv, **kw: (captured.__setitem__("argv", argv), _FakeProc())[1],
    )
    p.exec("cid-y", ["bash", "run.sh"])
    joined = " ".join(captured["argv"])
    assert f"OPENAI4S_NIM_URL={mod.SELF_HOSTED_URL}" in joined
    # self-hosted must NOT inject the hosted API key
    assert "NVIDIA_API_KEY" not in joined


# --- error mapping -------------------------------------------------------


@pytest.mark.parametrize(
    "stderr,kind",
    [
        ("Error: No such container: cid-x", "not_found"),
        ("unauthorized: authentication required", "unauthorized"),
        (
            "docker: could not select device driver with capabilities: [[gpu]]",
            "provider_degraded",
        ),
        ("toomanyrequests: too many requests to nvcr.io", "rate_limited"),
        ("some unexpected docker failure", "transient"),
    ],
)
def test_error_output_mapping(provider, stderr, kind):
    p, _mod, _calls, _scripted = provider
    err = p._map_err_output(stderr)
    assert err.kind == kind


# --- token scrub ---------------------------------------------------------


def test_token_scrub_redacts_keys(provider):
    p, _mod, _calls, _scripted = provider
    text = "logged in with nvapi-abc123DEF456ghi and " "nvcf-XYZ789tokenvalue here"
    scrubbed = p.token_scrub_regex.sub("[redacted]", text)
    assert "nvapi-abc123DEF456ghi" not in scrubbed
    assert "nvcf-XYZ789tokenvalue" not in scrubbed
    assert scrubbed.count("[redacted]") == 2


# --- terminate idempotency ----------------------------------------------


def test_terminate_idempotent_on_missing(provider):
    p, _mod, calls, scripted = provider
    # scripted `docker rm` returns a not-found stderr
    scripted["rm"] = _FakeCompleted(1, b"", b"Error: No such container: cid-gone")
    # must NOT raise — terminate swallows not_found
    p.terminate("cid-gone")
    assert any(c[:2] == ["docker", "rm"] for c in calls)


def test_read_owner_returns_none_when_missing(provider):
    p, _mod, _calls, scripted = provider
    scripted["inspect"] = _FakeCompleted(1, b"", b"Error: No such object: cid-gone")
    assert p.read_owner("cid-gone") is None
