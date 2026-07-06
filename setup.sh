#!/usr/bin/env bash
# OpenAI4S · environment setup (uv)
# Creates the .venv and installs the project + tooling with uv. Run this once,
# then launch the app with ./start.sh.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found. Install it first:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pip install uv" >&2
  exit 1
fi

# 1) first-run config (secrets optional — set your model in the UI later)
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "· created .env from .env.example"
fi

# 2) create .venv and install: core + dev tools (pytest, pre-commit) + science extra
uv sync --extra science
echo "· environment ready → .venv/"

# 3) enable the git pre-commit hook (black · isort · ruff) for contributors
uv run pre-commit install >/dev/null 2>&1 && echo "· pre-commit hook installed" || true

echo "· setup complete — launch with ./start.sh"
