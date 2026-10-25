#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Hermes runtime of the Frontend Delegation Agent example: the FDA gateway (it spawns one Hermes
# worker per session, inside this container) on port 8790. It is the gateway half of the former
# entrypoint.sh, with the same paths and arguments. The voice side runs in the shared app container
# and connects to ws://<this container>:8790/v1/backend (FDA_BACKEND_URL).
#
# Settings come from docker/fda/gateway.env and the Compose environment (the eval overlay sets
# FDA_GATEWAY_CONFIG=gateway.eval.yaml). FDA_GATEWAY_HOST defaults to loopback; the Compose service sets
# 0.0.0.0, because the voice side is in another container.
set -euo pipefail
cd /app
mkdir -p logs
export PYTHONPATH=/app/src FDA_HERMES_PYTHON=/app/.venv/bin/python
: "${FDA_GATEWAY_CONFIG:?FDA_GATEWAY_CONFIG is not set (docker/fda/gateway.env)}"
: "${NVIDIA_API_KEY:?NVIDIA_API_KEY is not set (.env): Hermes needs it for the backend LLM}"   # presence only
echo "FDA gateway: config $FDA_GATEWAY_CONFIG, source $(cat /app/SOURCE_SHA256 2>/dev/null || echo unknown)"
exec env FDA_GATEWAY_HOST="${FDA_GATEWAY_HOST:-127.0.0.1}" FDA_GATEWAY_PORT=8790 \
  /app/.venv/bin/python -m examples.frontend_delegation_agent.core.sidecar.gateway_server \
    --config "src/examples/frontend_delegation_agent/config/$FDA_GATEWAY_CONFIG" \
    --max-sessions "${FDA_GATEWAY_MAX_SESSIONS:-8}"
