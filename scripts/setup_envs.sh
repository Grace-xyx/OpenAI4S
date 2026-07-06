#!/bin/sh
# setup_envs.sh — create openai4s's four default conda environments.
#
# Thin wrapper around `openai4s setup`. Forwards all args, so:
#   ./scripts/setup_envs.sh              # create all four (python/phylo/r/struct)
#   ./scripts/setup_envs.sh --only python
#   ./scripts/setup_envs.sh --dry-run
#
# Make it executable once with:  chmod +x scripts/setup_envs.sh
set -eu
exec python -m openai4s setup "$@"
