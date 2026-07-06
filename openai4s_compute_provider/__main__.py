"""Generic entrypoint — one loader for every compute provider.

    python -I .../openai4s_compute_provider/__main__.py oneshot <provider.py> <op> <stage> <expectConfined>
    python -I .../openai4s_compute_provider/__main__.py repl    <provider.py>

Invoked as a script (not -m) because -I strips PYTHONPATH/cwd from sys.path,
so this file inserts the package's parent only — provider.py is loaded via
spec_from_file_location, so the skill dir is never on sys.path and unverified
sibling .py files there cannot shadow stdlib or third-party imports."""
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from openai4s_compute_provider import ByocResident  # noqa: E402

mode, provider_py, *rest = sys.argv[1:]

spec = importlib.util.spec_from_file_location("_byoc_provider", provider_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

r = ByocResident(mod.PROVIDER(repl=(mode == "repl")))
if mode == "repl":
    r.run_repl()
else:
    # argv[0] is a placeholder (script-name slot, ignored); run_oneshot
    # reads argv[1:4] = op, stage, expect_confined.
    r.run_oneshot([provider_py, *rest])
