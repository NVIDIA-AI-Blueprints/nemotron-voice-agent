#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint of the Frontend Delegation Hermes image: the backend gateway on 127.0.0.1:8790 (it
# spawns one Hermes worker per session, inside this container), then the voice server of one adapter
# on 7860 once the gateway is healthy. If either process exits, the other is stopped and the
# container exits with that status. No `set -e`: a non-zero `wait -n` must still reach stop().
#
#   entrypoint.sh [ADAPTER]      ADAPTER overrides FDH_ADAPTER (pipecat | tau | any adapter folder)
#
# The adapter's module is found by convention: `pipecat` runs examples.frontend_delegation_hermes.server
# (the default server.py); any other adapter <a> runs examples.frontend_delegation_hermes.<a>_server.
# Defaults (profile, gateway config) come only from the Compose service, never from this script.
set -uo pipefail
cd /app
mkdir -p logs
export PYTHONPATH=/app/src FDH_HERMES_PYTHON=/app/.venv/bin/python
CFG=src/examples/frontend_delegation_hermes/config
PKG=examples.frontend_delegation_hermes

usage() {
  echo "usage: entrypoint.sh [ADAPTER]  (needs FDH_ADAPTER or ADAPTER, FDH_PROFILE, FDH_GATEWAY_CONFIG)" >&2
  exit 2
}

adapter="${1:-${FDH_ADAPTER:-}}"
[ -n "$adapter" ] || usage
[ -n "${FDH_PROFILE:-}" ] || { echo "FDH_PROFILE is not set" >&2; usage; }
[ -n "${FDH_GATEWAY_CONFIG:-}" ] || { echo "FDH_GATEWAY_CONFIG is not set" >&2; usage; }
if [ "$adapter" = pipecat ]; then module="$PKG.server"; else module="$PKG.${adapter}_server"; fi
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)" 2>/dev/null \
  || { echo "unknown adapter '$adapter' (no module $module)" >&2; exit 2; }

gw="" voice=""

stop() {
  [ -n "$voice" ] && kill -TERM "$voice" 2>/dev/null
  [ -n "$gw" ] && kill -TERM "$gw" 2>/dev/null
  [ -n "$voice" ] && wait "$voice"
  [ -n "$gw" ] && wait "$gw"
}
trap 'stop; exit 143' TERM INT

FDH_GATEWAY_HOST=127.0.0.1 FDH_GATEWAY_PORT=8790 \
  python -m "$PKG.core.sidecar.gateway_server" \
    --config "$CFG/$FDH_GATEWAY_CONFIG" --max-sessions "${FDH_GATEWAY_MAX_SESSIONS:-8}" &
gw=$!
ready=0
for _ in $(seq 60); do
  curl -fs -m 2 http://127.0.0.1:8790/health >/dev/null && { ready=1; break; }
  kill -0 "$gw" 2>/dev/null || break
  sleep 0.5
done
[ "$ready" = 1 ] || { echo "gateway not healthy" >&2; stop; exit 1; }

echo "starting adapter '$adapter' ($module) with profile $FDH_PROFILE and gateway config $FDH_GATEWAY_CONFIG"
python -m "$module" --config "$CFG/profiles/$FDH_PROFILE" --port 7860 &
voice=$!

rc=0
wait -n "$gw" "$voice" || rc=$?
stop
exit "$rc"
