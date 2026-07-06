"""Pytest fixtures + path setup for the openai4s test suite."""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(autouse=True)
def isolated_openai4s_home(tmp_path, monkeypatch):
    """Keep tests off the developer's real ~/.openai4s database."""
    import openai4s.config as config_mod
    import openai4s.store as store_mod

    def reset_singletons():
        for st in list(store_mod._STORES.values()):
            try:
                st.close()
            except Exception:
                pass
        store_mod._STORES.clear()
        config_mod._CONFIG = None

    reset_singletons()
    monkeypatch.setenv("OPENAI4S_DATA_DIR", str(tmp_path / "openai4s-data"))
    monkeypatch.setenv("OPENAI4S_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("OPENAI4S_DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI4S_LLM_API_KEY", "test-key")
    yield
    reset_singletons()
