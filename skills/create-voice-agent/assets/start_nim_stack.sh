#!/usr/bin/env bash
# PROC: LLM → TTS → ASR on shared GPU. REF hardware-probe Step 2d.
# Use only when ALL slots need start. If some slots already match confirmed models,
# probe per deployment.md §Local selective slot reuse and up -d individual services instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE=(docker compose -f docker-compose.nim.yml)
WAIT_TIMEOUT="${NIM_START_TIMEOUT:-3600}"

[[ -f .env ]] || { echo "Missing .env"; exit 1; }
set -a
# shellcheck disable=SC1091
source .env
set +a

mkdir -p .nim-cache/asr
chmod 777 .nim-cache/asr 2>/dev/null || true

wait_healthy() {
  local service="$1"
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < deadline )); do
    local cid status
    cid="$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null || true)"
    if [[ -n "$cid" ]]; then
      status="$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null || echo unknown)"
      [[ "$status" == "healthy" ]] && return 0
      [[ "$status" == "unhealthy" ]] && { docker logs --tail 40 "$cid" 2>&1; return 1; }
    fi
    sleep 15
  done
  return 1
}

"${COMPOSE[@]}" up -d nvidia-llm
wait_healthy nvidia-llm
"${COMPOSE[@]}" --profile full up -d tts-service
wait_healthy tts-service
"${COMPOSE[@]}" --profile full up -d asr-service
wait_healthy asr-service
"${COMPOSE[@]}" --profile full ps
