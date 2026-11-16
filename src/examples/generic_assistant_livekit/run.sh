#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch and clean up the LiveKit generic assistant. A thin wrapper around `docker compose`: it prints every command
# it runs, and the README lists the equivalent plain commands. Self-contained, so it keeps working if this folder moves.
#
#   run.sh up <cloud|server|single-gpu> [--sip] [--no-wait]
#                                            Start one profile (stops another running profile first), show its status,
#                                            then wait until every healthcheck passes (--no-wait skips). --sip adds
#                                            Redis and the LiveKit SIP service for phone calls. Changing --sip on a
#                                            running profile restarts it; volumes and credentials are kept
#   run.sh check <profile>                   Verify a profile started with --sip: containers, listening ports, Redis
#                                            exposure, the SIP log. Prints PASS or FAIL per check
#   run.sh sip-setup <profile> --number +1555... [--scenario bank] [--allowed-address CIDR]
#                                            Create the inbound SIP trunk and dispatch rule for one number
#   run.sh sip-address                       Run on the Linux host: print its LAN address and network, the router and the
#                                            public address, and whether the public address can receive calls
#   run.sh sip-network --lan-ip IP --lan CIDR --signaling CIDR [--media CIDR]
#                                            Print the router and firewall steps (changes nothing)
#   run.sh down [profile]                    Stop a profile, or all profiles when none is given
#   run.sh clean [profile] [--image] [-y]    Stop and delete volumes (credentials, NIM caches); --image also removes the image
#   run.sh ps [profile]                      Show container status
#   run.sh logs <profile> [service]          Follow logs. Services: agent, livekit, ui, init, llm, asr, tts, speech,
#                                            redis, sip
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
OVERLAYS="sip"  # optional add-ons, each with its own Compose profile
DRY_RUN="${DRY_RUN:-0}"

die() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }
info() { echo "$*" >&2; }

usage() { sed -n '5,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit "${1:-2}"; }

profile_name() {
  case "$1" in
    cloud) echo "generic-assistant-livekit" ;;
    server) echo "generic-assistant-livekit/server" ;;
    single-gpu) echo "generic-assistant-livekit/single-gpu" ;;
    *) die "unknown profile '$1'. Use one of: $PROFILES" ;;
  esac
}

overlay_profile() {
  case "$1" in
    sip) echo "generic-assistant-livekit/sip" ;;
    *) die "unknown overlay '$1'" ;;
  esac
}

overlay_service() {
  case "$1" in
    sip) echo livekit-sip ;;
  esac
}

pflags() { # pflags <profile> [overlay...]: the --profile flags for a profile and the given overlays
  local out o
  out="--profile $(profile_name "$1")"
  shift
  for o in "$@"; do out="$out --profile $(overlay_profile "$o")"; done
  echo "$out"
}

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
detect_overlays() { # detect_overlays <profile>: the overlays that already have a container (running or stopped)
  local o
  for o in $OVERLAYS; do
    if [ -n "$(dc_quiet $(pflags "$1" "$o") ps -a -q "$(overlay_service "$o")" 2>/dev/null || true)" ]; then echo "$o"; fi
  done
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
    redis | sip) echo "livekit-$svc" ;;
    speech)
      [ "$profile" = single-gpu ] || die "'speech' exists only in the single-gpu profile"
      echo livekit-nemo-speech ;;
    *) echo "$svc" ;;
  esac
}

# Host ports the overlays publish (SIP uses host networking).
overlay_ports() {
  local o
  for o in "$@"; do
    case "$o" in
      sip) echo "5060 6379" ;;
    esac
  done
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
  local version major minor patch rest
  version="$(docker compose version --short 2>/dev/null)" || die "the 'docker compose' plugin is required"
  version="${version#v}"
  major="${version%%.*}"
  rest="${version#*.}"
  minor="${rest%%.*}"
  patch="${rest#*.}"
  patch="${patch%%[!0-9]*}"
  # docker-compose.yaml uses the !override tag, which needs Docker Compose 2.24.4 or newer
  if [ "$major" -lt 2 ] 2>/dev/null \
    || { [ "$major" = 2 ] && [ "$minor" -lt 24 ] 2>/dev/null; } \
    || { [ "$major" = 2 ] && [ "$minor" = 24 ] && [ "${patch:-0}" -lt 4 ] 2>/dev/null; }; then
    die "Docker Compose $version is too old. This example needs 2.24.4 or newer"
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

check_ports() { # check_ports <profile> [overlay...]
  if [ "$DRY_RUN" = 1 ] || ! command -v ss >/dev/null 2>&1; then return 0; fi
  local port busy="" name="$1"
  shift
  for port in $(profile_ports "$name") $(overlay_ports "$@"); do
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

check_overlays() { # check_overlays <overlay...>: fail early on settings the overlays cannot work without
  local o
  for o in "$@"; do
    case "$o" in
      sip)
        [ "$(uname -s)" = Linux ] || die "--sip needs host networking, which Docker supports only on Linux"
        ;;
    esac
  done
}

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
cmd_up() {
  [ $# -ge 1 ] || usage
  local name="$1" wait=1 p other arg want="" have o
  shift
  for arg in "$@"; do
    case "$arg" in
      --no-wait) wait=0 ;;
      --wait) ;; # the default; accepted for clarity
      --sip) want="$want sip" ;;
      *) die "unknown option '$arg' for up. Options: --sip, --no-wait" ;;
    esac
  done
  # Normalize to the fixed order, so the requested overlays compare equal to what is running.
  local ordered=""
  for o in $OVERLAYS; do case " $want " in *" $o "*) ordered="$ordered $o" ;; esac; done
  want="${ordered# }"
  p="$(profile_name "$name")"
  require_docker
  require_env_file
  check_keys "$name"
  check_overlays $want
  for other in $(running_profiles); do
    if [ "$other" != "$name" ]; then
      info "Another profile ($other) is running. Stopping it first."
      dc $(pflags "$other" $OVERLAYS) down
    fi
  done
  have="$(detect_overlays "$name" | xargs)"
  # The overlays change the LiveKit server config, which it reads at start, so a different set means a restart.
  if [ -n "$(dc_quiet $(pflags "$name" $OVERLAYS) ps -a -q 2>/dev/null || true)" ] && [ "$have" != "$want" ]; then
    info "The overlays changed (running: ${have:-none}, requested: ${want:-none}). Restarting the $name profile; volumes are kept."
    dc $(pflags "$name" $OVERLAYS) down
  fi
  if [ -z "$(dc_quiet --profile "$p" ps -q 2>/dev/null || true)" ]; then
    [ "$name" != single-gpu ] || check_single_gpu
    check_ports "$name" $want
  fi
  # LIVEKIT_SIP reaches the livekit-init service through Compose interpolation.
  case " $want " in *" sip "*) export LIVEKIT_SIP=1 ;; *) export LIVEKIT_SIP= ;; esac
  dc $(pflags "$name" $want) up -d
  info ""
  dc $(pflags "$name" $want) ps
  info ""
  info "Started the $name profile${want:+ with: $want}."
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
  if [ -n "$want" ]; then overlay_hint $want; fi
}

overlay_hint() {
  local o
  for o in "$@"; do
    case "$o" in
      sip)
        info "Phone calls: SIP listens on 5060 and RTP on 10000-10100/udp of this host. Nothing reaches them until you"
        info "forward them and restrict the firewall to your provider's ranges: $0 sip-network --help, then see the guide."
        ;;
    esac
  done
}

# After down or clean, no container of this example may remain, including the optional SIP and Redis ones.
warn_leftovers() {
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  local left
  left="$(dc_quiet --profile "$(profile_name cloud)" --profile "$(profile_name server)" \
    --profile "$(profile_name single-gpu)" --profile "$(overlay_profile sip)" \
    ps -a --format '{{.Name}}' 2>/dev/null || true)"
  if [ -n "$left" ]; then
    warn "containers of this example are still present:"
    printf '  %s\n' $left >&2
  else
    info "No containers of this example remain."
  fi
}

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
cmd_down() {
  require_docker
  if [ $# -ge 1 ]; then
    dc $(pflags "$1" $OVERLAYS) down
  else
    local name
    for name in $PROFILES; do dc $(pflags "$name" $OVERLAYS) down; done
  fi
  warn_leftovers
}

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
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
    dc $(pflags "$name" $OVERLAYS) down -v
  else
    local n
    for n in $PROFILES; do dc $(pflags "$n" $OVERLAYS) down -v; done
  fi
  if [ "$image" = 1 ]; then
    info "+ docker image rm $IMAGE"
    if [ "$DRY_RUN" != 1 ]; then docker image rm "$IMAGE" || true; fi
  fi
  warn_leftovers
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

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
cmd_ps() {
  require_docker
  if [ $# -ge 1 ]; then
    dc $(pflags "$1" $OVERLAYS) ps
  else
    dc --profile "$(profile_name cloud)" --profile "$(profile_name server)" --profile "$(profile_name single-gpu)" \
      --profile "$(overlay_profile sip)" ps
  fi
}

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
cmd_logs() {
  [ $# -ge 1 ] || usage
  require_docker
  local name="$1" svc=""
  if [ $# -ge 2 ]; then svc="$(service_name "$name" "$2")"; fi
  if [ -n "$svc" ]; then
    dc $(pflags "$name" $OVERLAYS) logs -f --tail 50 "$svc"
  else
    dc $(pflags "$name" $OVERLAYS) logs -f --tail 50
  fi
}

# shellcheck disable=SC2046,SC2086 # pflags prints words, and the profile names contain no spaces
cmd_wait() {
  [ $# -ge 1 ] || usage
  require_docker
  local name="$1" timeout="${2:-3600}" p start now id svc status state code pending cname last label unready hinted=0 pf
  p="$(profile_name "$name")"
  # Wait for the overlays that exist, so a deployment without --sip is not held up by the SIP containers.
  pf="$(pflags "$name" $(detect_overlays "$name"))"
  start="$(date +%s)"
  info "Waiting up to ${timeout}s for the $name profile to become healthy (progress every ${WAIT_INTERVAL:-15}s)..."
  if [ "$DRY_RUN" = 1 ]; then
    access_hint
    return 0
  fi
  while true; do
    pending=""
    unready=0
    # Check every service of the profile, stopped ones included (`ps -a`). Only livekit-init may be stopped, and only
    # after it exits successfully. A container without a health check must still be running, so a crashed sidecar or
    # agent cannot pass as ready.
    for svc in $(dc_quiet $pf config --services); do
      id="$(dc_quiet $pf ps -a -q "$svc" | head -n 1)"
      if [ -z "$id" ]; then
        pending="$pending
  - $svc: not created"
        continue
      fi
      state="$(docker inspect -f '{{.State.Status}}' "$id")"
      cname="$(docker inspect -f '{{.Name}}' "$id" | sed 's#^/##')"
      if [ "$svc" = livekit-init ]; then
        code="$(docker inspect -f '{{.State.ExitCode}}' "$id")"
        if [ "$state" != exited ] || [ "$code" != 0 ]; then
          pending="$pending
  - $cname: has not finished setting up LiveKit ($state, exit code $code). Check: $0 logs $name init"
        fi
        continue
      fi
      status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$id")"
      if [ "$state" != running ]; then
        pending="$pending
  - $cname is $state. Check: $0 logs $name"
      elif [ -n "$status" ] && [ "$status" != healthy ]; then
        label="starting"
        if [ "$status" = unhealthy ]; then label="not ready yet"; unready=1; fi
        last="$(docker logs --tail 1 "$id" 2>&1 | tr -d '\r' | cut -c1-110)"
        pending="$pending
  - $cname: $label${last:+ | last log: $last}"
      fi
    done
    if [ -z "$pending" ] && ! dc_quiet $pf logs --tail 300 "$(agent_service "$name")" 2>/dev/null | grep -q "registered worker"; then
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

CHECK_FAILED=0
pass() { info "  PASS  $*"; }
fail() { info "  FAIL  $*"; CHECK_FAILED=1; }

service_state() { # service_state <flags> <service>: "state" or "state/health", or "missing"
  local id
  id="$(dc_quiet $1 ps -a -q "$2" 2>/dev/null | head -n 1)"
  if [ -z "$id" ]; then echo missing; return; fi
  local state health
  state="$(docker inspect -f '{{.State.Status}}' "$id")"
  health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$id")"
  echo "$state${health:+/$health}"
}

check_service() { # check_service <overlay-flags> <service> [expected-state-prefix]
  local state
  state="$(service_state "$1" "$2")"
  case "$state" in
    "${3:-running}"*) pass "$2 is $state" ;;
    *) fail "$2 is $state. Check: $0 logs <profile> ${2#livekit-}" ;;
  esac
}

check_listening() { # check_listening <port> <label>: something listens on the host port (TCP or UDP)
  if ! command -v ss >/dev/null 2>&1; then info "  skip  $2: ss is not installed"; return; fi
  if [ -n "$(ss -lntuH "sport = :$1" 2>/dev/null)" ]; then pass "$2 listens on port $1"; else fail "nothing listens on port $1 ($2)"; fi
}

check_image_arch() { # check_image_arch <service>: print the image architecture, so an image without arm64 is obvious
  local image arch
  image="$(dc_quiet $1 config --images 2>/dev/null | grep -E "$2" | head -n 1)"
  [ -n "$image" ] || return 0
  arch="$(docker image inspect -f '{{.Architecture}}' "$image" 2>/dev/null || true)"
  if [ -n "$arch" ]; then info "  info  $image is $arch (this host: $(uname -m))"; fi
}

check_sip() { # check_sip <flags>
  local flags="$1" id published logs
  check_service "$flags" livekit-redis "running/healthy"
  check_service "$flags" livekit-sip
  check_listening 6379 "Redis"
  check_listening 5060 "SIP"
  check_image_arch "$flags" "livekit/sip"
  id="$(dc_quiet $flags ps -a -q livekit-redis 2>/dev/null | head -n 1)"
  if [ -n "$id" ]; then
    published="$(docker port "$id" 6379 2>/dev/null || true)"
    case "$published" in
      "") fail "Redis publishes no port, so the SIP service cannot reach it" ;;
      *0.0.0.0* | *"[::]"*) fail "Redis is published on every address ($published). It must be 127.0.0.1 only" ;;
      *) pass "Redis is published on loopback only ($published)" ;;
    esac
  fi
  logs="$(dc_quiet $flags logs --tail 100 livekit-sip 2>&1 || true)"
  # Per-call lines (statistics, timestamps) are not service errors: their counters include words such as "packets_failed".
  logs="$(printf '%s\n' "$logs" | grep -v -e 'call statistics' -e 'signaling timestamps' -e 'SIP call ended' || true)"
  if printf '%s' "$logs" | grep -qiE 'panic|fatal|permission denied|connection refused|no such file|failed|cannot'; then
    fail "the SIP log shows an error. Last lines:"
    printf '%s\n' "$logs" | tail -n 3 | sed 's/^/          /' >&2
  else
    pass "no error in the recent SIP log"
  fi
}

# shellcheck disable=SC2046 # pflags prints words, and the profile names contain no spaces
cmd_check() {
  [ $# -eq 1 ] || die "usage: run.sh check <cloud|server|single-gpu>   (checks a profile started with --sip)"
  require_docker
  local name="$1" flags
  profile_name "$name" >/dev/null
  flags="$(pflags "$name" sip)"
  info "Checking the SIP overlay of the $name profile:"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  check_sip "$flags"
  if [ "$CHECK_FAILED" = 0 ]; then
    info "All checks passed."
    info "Not covered: a real call. Continue with: $0 sip-setup $name --number <e164>, then call it (see SIP.md)."
  else
    die "some checks failed"
  fi
}

cmd_sip_setup() { # run inside the agent container, which shares the LiveKit server's network and has the credentials
  [ $# -ge 1 ] || die "usage: run.sh sip-setup <profile> --number +15551234567 [--scenario bank] [--allowed-address CIDR]"
  require_docker
  local name="$1"
  shift
  [ -n "$(detect_overlays "$name" | grep -x sip || true)" ] || [ "$DRY_RUN" = 1 ] \
    || die "the $name profile is not running with --sip. Start it with: $0 up $name --sip"
  # shellcheck disable=SC2046
  dc $(pflags "$name" sip) exec "$(agent_service "$name")" python -m examples.generic_assistant_livekit.sip_setup trunk "$@"
}

cmd_sip_address() { # read-only: reports the host's addresses for the SIP setup; run it on the Linux host
  [ "$(uname -s)" = "Linux" ] || die "run sip-address on the Linux host that will receive the calls (it reads that host's network)"
  command -v ip >/dev/null 2>&1 || die "the 'ip' command is required"
  local route router dev lan_ip cidr network public verdict=0
  route="$(ip -4 route get 1.1.1.1 2>/dev/null | head -n 1)"
  lan_ip="$(sed -n 's/.* src \([0-9.]*\).*/\1/p' <<<"$route")"
  [ -n "$lan_ip" ] || die "no default network route found. Is the host online?"
  router="$(sed -n 's/.* via \([0-9.]*\).*/\1/p' <<<"$route")"
  dev="$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")"
  cidr="$(ip -4 -o addr show dev "$dev" 2>/dev/null | sed -n 's/.* inet \([0-9.]*\/[0-9]*\).*/\1/p' | grep "^$lan_ip/" | head -n 1 || true)"
  network="$cidr"
  if [ -n "$cidr" ] && command -v python3 >/dev/null 2>&1; then
    network="$(python3 -c 'import ipaddress, sys; print(ipaddress.ip_interface(sys.argv[1]).network)' "$cidr" 2>/dev/null || echo "$cidr")"
  fi
  public="$(curl -fsS --max-time 8 https://ifconfig.me 2>/dev/null || true)"
  printf 'LAN address:    %s  (interface %s)\n' "$lan_ip" "${dev:-unknown}"
  printf 'LAN network:    %s\n' "${network:-unknown}"
  printf 'Router address: %s\n' "${router:-unknown}"
  printf 'Public address: %s\n\n' "${public:-unknown}"
  case "$public" in
    "")
      echo "FAIL: could not read the public address (is outbound internet access allowed?)"
      verdict=1
      ;;
    10.* | 192.168.* | 172.1[6-9].* | 172.2[0-9].* | 172.3[01].* | 127.* | 169.254.*)
      echo "FAIL: $public is a private address, so a proxy or another router is in front. Phone calls cannot reach this host."
      verdict=1
      ;;
    100.6[4-9].* | 100.[7-9][0-9].* | 100.1[01][0-9].* | 100.12[0-7].*)
      echo "FAIL: $public is in the carrier-grade NAT range (100.64.0.0/10). Your ISP shares one address among many customers,"
      echo "      so inbound calls cannot reach this host. Use a cloud VM, or ask your ISP for a public address."
      verdict=1
      ;;
    *) echo "OK: $public looks like a public address." ;;
  esac
  cat <<EOF

Next, by hand:
  1. Open the router's page${router:+ (http://$router)} and compare its WAN or Internet IP address with the public address above.
     They must be the same. If they differ, the router is behind another NAT, and calls cannot reach this host.
  2. Use the LAN address and network above in step 2:
       $0 sip-network --lan-ip ${lan_ip} --lan ${network:-<LAN CIDR>} --signaling <CIDR> --media <CIDR>
  3. Use the public address in the provider's origination URI: sip:<public address>:5060;transport=tcp
EOF
  return "$verdict"
}

cmd_sip_network() { # prints guidance only; runs in a throwaway container of the example image, so no stack is needed
  require_docker
  require_env_file
  # shellcheck disable=SC2046
  dc $(pflags cloud) run --rm --no-deps livekit-init python -m examples.generic_assistant_livekit.sip_setup network "$@"
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
    check) cmd_check "$@" ;;
    sip-setup) cmd_sip_setup "$@" ;;
    sip-network) cmd_sip_network "$@" ;;
    sip-address) cmd_sip_address "$@" ;;
    -h | --help | help) usage 0 ;;
    *) usage ;;
  esac
}

main "$@"
