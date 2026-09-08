#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
exec uv run --frozen cisco-webex-byova-ws-adapter "$@"
