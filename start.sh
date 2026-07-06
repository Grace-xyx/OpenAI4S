#!/usr/bin/env bash
# OpenAI4S · start the daemon + web UI from an already-configured environment.
# Run ./setup.sh first. Opens http://127.0.0.1:8760/ (set your model in the UI).
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/openai4s ]; then
  RUN=(.venv/bin/openai4s)
elif [ -x .venv/bin/python ]; then
  RUN=(.venv/bin/python -m openai4s)
else
  echo "no .venv found — run ./setup.sh first." >&2
  exit 1
fi

echo "· starting OpenAI4S → http://127.0.0.1:8760/  (set your model in Customize → Models)"
exec "${RUN[@]}" serve "$@"
