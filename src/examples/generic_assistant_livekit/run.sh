#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch and clean up the LiveKit generic assistant. A thin wrapper around `docker compose`: it prints every command
# it runs, and the README lists the equivalent plain commands. Self-contained, so it keeps working if this folder moves.
#
#   run.sh up <cloud|server|single-gpu> [--no-wait]  Start one profile (stops another running profile first), show
#                                            its status, then wait until every healthcheck passes (--no-wait skips)
#   run.sh down [profile]                    Stop a profile, or all profiles when none is given
#   run.sh clean [profile] [--image] [-y]    Stop and delete volumes (credentials, NIM caches); --image also removes the image
#   run.sh ps [profile]                      Show container status
#   run.sh logs <profile> [service]          Follow logs. Services: agent, livekit, ui, init, llm, asr, tts, speech
#   run.sh tunnel <user@host>                Run on your laptop: forward the UI and LiveKit ports from the host
#   run.sh wait <profile> [seconds]          Wait until every container with a healthcheck is healthy (default 3600),
#                                            printing progress every 15 seconds
#
# Environment: WAIT_INTERVAL sets the seconds between `wait` progress lines (default 15). ENV_FILE selects the .env
# file (default: the repository root .env three levels above this folder, else a .env in this folder). LOCAL_UI_PORT
# sets the laptop port of `tunnel` (default 3000). DRY_RUN=1 prints the docker and ssh commands without running them or
# the docker-based checks.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$DIR/docker-compose.yaml"
IMAGE="generic-assistant-livekit:latest"
PROFILES="cloud server single-gpu"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }
info() { echo "$*" >&2; }

usage() { sed -n '5,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit "${1:-2}"; }

profile_name() {
  case "$1" in
    cloud) echo "generic-assistant-livekit" ;;
    server) echo "generic-assistant-livekit/server" ;;
    single-gpu) echo "generic-assistant-livekit/single-gpu" ;;
    *) die "unknown profile '$1'. Use one of: $PROFILES" ;;
  esac
}

# Short service names for `logs`, resolved per profile.
service_name() {
  local profile="$1" svc="$2"
  case "$svc" in
    agent)
      case "$profile" in
        cloud) echo livekit-agent ;;
        server) echo livekit-agent-server ;;
        single-gpu) echo livekit-agent-single-gpu ;;
      esac ;;
    livekit) echo livekit-server ;;
    ui) echo livekit-ui ;;
    init) echo livekit-init ;;
    llm)
      case "$profile" in
        server) echo livekit-nvidia-llm ;;
        single-gpu) echo livekit-nvidia-llm-vllm ;;
        *) die "the cloud profile has no local LLM" ;;
      esac ;;
    asr)
      case "$profile" in
        server) echo livekit-nemotron-asr ;;
        single-gpu) echo livekit-nemo-speech ;;
        *) die "the cloud profile has no local ASR" ;;
      esac ;;
    tts)
      case "$profile" in
        server) echo livekit-magpie-tts ;;
        single-gpu) echo livekit-nemo-speech ;;
        *) die "the cloud profile has no local TTS" ;;
      esac ;;
    speech)
      [ "$profile" = single-gpu ] || die "'speech' exists only in the single-gpu profile"
      echo livekit-nemo-speech ;;
    *) echo "$svc" ;;
  esac
}

# Host ports each profile publishes (LiveKit and the UI are shared by all profiles).
profile_ports() {
  local ui
  ui="$(ui_port)"
  case "$1" in
    cloud) echo "7880 7881 $ui" ;;
    server) echo "7880 7881 $ui 9000 9001 50151 50152 18000" ;;
    single-gpu) echo "7880 7881 $ui 50051 18000" ;;
  esac
}

find_env_file() {
  if [ -n "${ENV_FILE:-}" ]; then echo "$ENV_FILE"; return; fi
  local candidate
  for candidate in "$DIR/../../../.env" "$DIR/.env"; do
    if [ -f "$candidate" ]; then echo "$candidate"; return; fi
  done
  echo "$DIR/../../../.env"
}

ENV_PATH="$(find_env_file)"

env_value() { # env_value KEY: value from the environment, else from the .env file
  local key="$1" value
  value="${!key:-}"
  if [ -z "$value" ] && [ -f "$ENV_PATH" ]; then
    value="$(grep -E "^${key}=" "$ENV_PATH" | tail -n 1 | cut -d= -f2- || true)"
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
  fi
  echo "$value"
}

dc() { # dc <compose args...>: run docker compose against this example, printing the command
  local cmd=(docker compose --env-file "$ENV_PATH" -f "$COMPOSE_FILE" "$@")
  info "+ ${cmd[*]}"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  "${cmd[@]}"
}

dc_quiet() { # same, but without printing; empty output in a dry run
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  docker compose --env-file "$ENV_PATH" -f "$COMPOSE_FILE" "$@"
}

require_docker() {
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
  local version major minor
  version="$(docker compose version --short 2>/dev/null)" || die "the 'docker compose' plugin is required"
  version="${version#v}"
  major="${version%%.*}"
  minor="${version#*.}"
  minor="${minor%%.*}"
  # docker-compose.yaml uses the !override tag, which needs Docker Compose 2.24 or newer
  if [ "$major" -lt 2 ] 2>/dev/null || { [ "$major" = 2 ] && [ "$minor" -lt 24 ] 2>/dev/null; }; then
    die "Docker Compose $version is too old. This example needs 2.24 or newer"
  fi
}

require_env_file() {
  [ -f "$ENV_PATH" ] || die "no .env file at $ENV_PATH. Copy the template: cp $DIR/.env.example <repo root>/.env (or set ENV_FILE)"
}

agent_service() { # the agent service of a profile. It belongs to that profile only, unlike the shared LiveKit services
  case "$1" in
    cloud) echo livekit-agent ;;
    server) echo livekit-agent-server ;;
    single-gpu) echo livekit-agent-single-gpu ;;
  esac
}

running_profiles() { # profiles whose agent has a container. The LiveKit server and UI are shared by all profiles
  local name p
  for name in $PROFILES; do
    p="$(profile_name "$name")"
    if [ -n "$(dc_quiet --profile "$p" ps -q "$(agent_service "$name")" 2>/dev/null || true)" ]; then echo "$name"; fi
  done
}

ui_port() { # the host UI port: the shell, then .env, then the default
  local port
  port="$(env_value LIVEKIT_UI_PORT)"
  echo "${port:-3000}"
}

check_keys() {
  local profile="$1" key
  case "$profile" in
    cloud | server) key=NVIDIA_API_KEY ;;
    single-gpu) key=HF_TOKEN ;;
  esac
  [ -n "$(env_value "$key")" ] || die "$key is empty. Set it in $ENV_PATH for the $profile profile"
}

check_single_gpu() {
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  local root="$DIR/../../.."
  if [ -d "$root/models" ] || [ -d "$root/scripts" ]; then
    if [ ! -d "$root/models/nemo-speech" ]; then
      die "speech weights not found in $root/models/nemo-speech. Run: bash $root/scripts/download-nemo-speech-models.sh"
    fi
  fi
  if [ -r /proc/meminfo ]; then
    local total avail
    total="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if [ -n "$total" ] && [ -n "$avail" ] && [ $((avail * 100)) -lt $((total * 45)) ]; then
      warn "only $((avail / 1048576)) GiB of $((total / 1048576)) GiB memory is available. The LLM needs about 45% of the unified memory free on a DGX Spark. Stop other model servers first (free -h)."
    fi
  fi
}

check_ports() {
  if [ "$DRY_RUN" = 1 ] || ! command -v ss >/dev/null 2>&1; then return 0; fi
  local port busy=""
  for port in $(profile_ports "$1"); do
    if [ -n "$(ss -ltnH "sport = :$port" 2>/dev/null)" ]; then busy="$busy $port"; fi
  done
  [ -z "$busy" ] || die "host port(s)$busy already in use. Find the owner with: docker ps --format '{{.Names}} {{.Ports}}' | grep -E ':($(echo "$busy" | xargs | tr ' ' '|'))->'"
}

access_hint() { # printed last, so it is the output you see when the start finishes
  local host
  host="$(hostname -f 2>/dev/null || hostname)"
  info ""
  info "Open http://localhost:$(ui_port)/ in a browser on this machine."
  info "Browser on another computer? The UI is not reachable from there until you start a tunnel ON THAT COMPUTER:"
  info "  src/examples/generic_assistant_livekit/run.sh tunnel <user>@${host}"
  info "  (or: ssh -N -L 3000:127.0.0.1:3000 -L 7880:127.0.0.1:7880 -L 7881:127.0.0.1:7881 <user>@${host})"
  info "Keep it running, and open http://localhost:3000/ there. Replace <user>@${host} with the SSH name you use."
}

cmd_up() {
  [ $# -ge 1 ] || usage
  local name="$1" wait=1 p other arg
  shift
  for arg in "$@"; do
    case "$arg" in
      --no-wait) wait=0 ;;
      --wait) ;; # the default; accepted for clarity
      *) die "unknown option '$arg' for up. Options: --no-wait" ;;
    esac
  done
  p="$(profile_name "$name")"
  require_docker
  require_env_file
  check_keys "$name"
  for other in $(running_profiles); do
    if [ "$other" != "$name" ]; then
      info "Another profile ($other) is running. Stopping it first."
      dc --profile "$(profile_name "$other")" down
    fi
  done
  if [ -z "$(dc_quiet --profile "$p" ps -q 2>/dev/null || true)" ]; then
    [ "$name" != single-gpu ] || check_single_gpu
    check_ports "$name"
  fi
  dc --profile "$p" up -d
  info ""
  dc --profile "$p" ps
  info ""
  info "Started the $name profile."
  if [ "$wait" = 1 ]; then
    if [ "$name" != cloud ]; then
      info "The first start downloads and loads models and can take a long time. Ctrl+C stops only this wait: the"
      info "containers keep running, and you can resume with: $0 wait $name"
    fi
    cmd_wait "$name"
  else
    info "Not waiting. Block until every healthcheck passes with: $0 wait $name"
    access_hint
  fi
}

cmd_down() {
  require_docker
  if [ $# -ge 1 ]; then
    dc --profile "$(profile_name "$1")" down
  else
    local name
    for name in $PROFILES; do dc --profile "$(profile_name "$name")" down; done
  fi
}

cmd_clean() {
  local name="" image=0 yes=0 arg
  for arg in "$@"; do
    case "$arg" in
      --image) image=1 ;;
      -y | --yes) yes=1 ;;
      --keep-image) die "--keep-image is not needed: clean keeps the image unless you pass --image" ;;
      -*) die "unknown option '$arg' for clean. Options: --image, -y" ;;
      *) name="$arg" ;;
    esac
  done
  require_docker
  if [ "$yes" != 1 ] && [ "$DRY_RUN" != 1 ]; then
    info "This deletes this example's volumes: the generated LiveKit credentials and the NIM model caches."
    if [ "$image" = 1 ]; then info "It also removes the example image, which the next start rebuilds."; fi
    info "Downloaded Hugging Face and NeMo-Speech weights and pulled sidecar images are not touched."
    read -r -p "Continue? [y/N] " reply
    case "$reply" in y | Y | yes) ;; *) die "cancelled" ;; esac
  fi
  if [ -n "$name" ]; then
    dc --profile "$(profile_name "$name")" down -v
  else
    local n
    for n in $PROFILES; do dc --profile "$(profile_name "$n")" down -v; done
  fi
  if [ "$image" = 1 ]; then
    info "+ docker image rm $IMAGE"
    if [ "$DRY_RUN" != 1 ]; then docker image rm "$IMAGE" || true; fi
  fi
}

port_busy() { # port_busy <port>: succeed when something listens on the local port
  if command -v lsof >/dev/null 2>&1; then
    [ -n "$(lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null)" ]
  elif command -v ss >/dev/null 2>&1; then
    [ -n "$(ss -ltnH "sport = :$1" 2>/dev/null)" ]
  else
    return 1
  fi
}

cmd_tunnel() { # run on your laptop: forward the UI, LiveKit signaling, and LiveKit TCP media to the host
  [ $# -ge 1 ] || die "usage: run.sh tunnel <user@host>   (run it on the computer that has the browser)"
  local host="$1" ui="${LOCAL_UI_PORT:-3000}" port
  command -v ssh >/dev/null 2>&1 || die "ssh is not installed"
  if [ "$DRY_RUN" != 1 ]; then
    for port in "$ui" 7880 7881; do
      port_busy "$port" && die "local port $port is already in use. Find the owner with: lsof -nP -iTCP:$port -sTCP:LISTEN (set LOCAL_UI_PORT to move the UI port)"
    done
  fi
  local cmd=(ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30
    -L "$ui:127.0.0.1:$(ui_port)" -L 7880:127.0.0.1:7880 -L 7881:127.0.0.1:7881 "$host")
  info "+ ${cmd[*]}"
  info "Tunnel to $host. Keep this window open, and open http://localhost:$ui/ in your browser. Ctrl+C stops it."
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  exec "${cmd[@]}"
}

cmd_ps() {
  require_docker
  if [ $# -ge 1 ]; then
    dc --profile "$(profile_name "$1")" ps
  else
    dc --profile "$(profile_name cloud)" --profile "$(profile_name server)" --profile "$(profile_name single-gpu)" ps
  fi
}

cmd_logs() {
  [ $# -ge 1 ] || usage
  require_docker
  local name="$1" svc=""
  if [ $# -ge 2 ]; then svc="$(service_name "$name" "$2")"; fi
  if [ -n "$svc" ]; then
    dc --profile "$(profile_name "$name")" logs -f --tail 50 "$svc"
  else
    dc --profile "$(profile_name "$name")" logs -f --tail 50
  fi
}

cmd_wait() {
  [ $# -ge 1 ] || usage
  require_docker
  local name="$1" timeout="${2:-3600}" p start now ids id status state pending detail cname last label unready hinted=0
  p="$(profile_name "$name")"
  start="$(date +%s)"
  info "Waiting up to ${timeout}s for the $name profile to become healthy (progress every ${WAIT_INTERVAL:-15}s)..."
  if [ "$DRY_RUN" = 1 ]; then
    access_hint
    return 0
  fi
  while true; do
    ids="$(dc_quiet --profile "$p" ps -q)"
    pending=""
    unready=0
    for id in $ids; do
      status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$id")"
      state="$(docker inspect -f '{{.State.Status}}' "$id")"
      cname="$(docker inspect -f '{{.Name}}' "$id" | sed 's#^/##')"
      if [ "$state" = restarting ] || { [ "$state" = exited ] && [ -n "$status" ]; }; then
        detail="$cname is $state. Check: $0 logs $name"
        pending="$pending
  - $detail"
      elif [ -n "$status" ] && [ "$status" != healthy ]; then
        label="starting"
        if [ "$status" = unhealthy ]; then label="not ready yet"; unready=1; fi
        last="$(docker logs --tail 1 "$id" 2>&1 | tr -d '\r' | cut -c1-110)"
        pending="$pending
  - $cname: $label${last:+ | last log: $last}"
      fi
    done
    if [ -z "$pending" ] && ! dc_quiet --profile "$p" logs --tail 300 "$(agent_service "$name")" 2>/dev/null | grep -q "registered worker"; then
      pending="
  - $(agent_service "$name"): starting, not registered with LiveKit yet"
    fi
    now="$(date +%s)"
    if [ -z "$pending" ]; then
      info "[$(( (now - start) / 60 ))m$(( (now - start) % 60 ))s] All healthy. Ready to connect."
      access_hint
      return 0
    fi
    [ $((now - start)) -lt "$timeout" ] || die "timed out after ${timeout}s. Still starting:$pending"
    info "[$(( (now - start) / 60 ))m$(( (now - start) % 60 ))s] still starting:$pending"
    if [ "$unready" = 1 ] && [ "$hinted" = 0 ]; then
      info "  'not ready yet' is not an error: a first start can take longer than the container's health check allows."
      info "  Worry only if a container is restarting or its last log line stops changing."
      hinted=1
    fi
    sleep "${WAIT_INTERVAL:-15}"
  done
}

main() {
  [ $# -ge 1 ] || usage
  local command="$1"
  shift
  case "$command" in
    up) cmd_up "$@" ;;
    down) cmd_down "$@" ;;
    clean) cmd_clean "$@" ;;
    ps) cmd_ps "$@" ;;
    logs) cmd_logs "$@" ;;
    wait) cmd_wait "$@" ;;
    tunnel) cmd_tunnel "$@" ;;
    -h | --help | help) usage 0 ;;
    *) usage ;;
  esac
}

main "$@"
