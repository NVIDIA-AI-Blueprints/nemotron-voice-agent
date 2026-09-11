#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
#
# simulate_concurrency.sh
#
# Drives a load test by running multiple `benchmark.py` workers in parallel
# (one OS process per simulated user) and then asks `benchmark.py` to fold
# their results into per-run and sweep-wide summaries.
#
# Why not put the parallelism in Python?
#   `benchmark.py` is intentionally single-client. Concurrency lives here so
#   you can swap the orchestrator (GNU parallel, k8s job, etc.) without
#   touching the client implementation.
#
# What "synchronized metric windows" means:
#   For a level with N clients and stagger D, the metric window opens at
#   `now + (N-1)*D`, when the last worker begins its connection attempt. Each
#   worker receives the same `--metrics-start-time` and `--session-end-time`.
#   This synchronizes wall-clock accounting; it is not a connection-readiness
#   barrier, so inspect per-client logs for late session initialization.
#
# Usage examples:
#     ./simulate_concurrency.sh --clients "1"
#     ./simulate_concurrency.sh --clients "1 2 4 8" --test-duration 60
#     ./simulate_concurrency.sh --host my-host --port 7860 --clients "4" --no-save-audio
#
# `-h` / `--help` prints the full flag list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="$SCRIPT_DIR/benchmark.py"

# Prefer the project's uv-managed venv when available.
if command -v uv >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/../../pyproject.toml" ]]; then
  PY=("uv" "run" "--no-sync" "python3")
else
  PY=("python3")
fi

HOST="localhost"
PORT="7860"
PROTOCOL="rtvi"
CLIENT_COUNTS="1"
CLIENT_START_DELAY="1"
TEST_DURATION="300"
COOLDOWN="10"
REVERSE_BARGE_IN_THRESHOLD="0.4"
TURN_RESPONSE_TIMEOUT="10"
DATASET_DIR="$SCRIPT_DIR/dataset"
OUTPUT_DIR="$SCRIPT_DIR"
SAVE_AUDIO="1"
RUN_FAILED="0"
REALTIME_PATH="/v1/realtime"
REALTIME_SCHEME="wss"
REALTIME_MODEL=""
REALTIME_VOICE=""
REALTIME_INSTRUCTIONS=""
REALTIME_INPUT_MODE="audio"
REALTIME_TEXT_INPUTS=()
REALTIME_OUTPUT_MODALITY="audio"
REALTIME_TURN_MODE="automatic"
REALTIME_VAD_SILENCE_MS="800"
REALTIME_API_KEY_ENV=""
REALTIME_CA_FILE=""
REALTIME_INSECURE="0"
REALTIME_CLIENT_TOOLS_CONFIG=""
REALTIME_TOOL_TIMEOUT="120"
REALTIME_MAX_TOOL_ROUNDS="4"
REALTIME_SESSION_TIMEOUT="60"

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --host HOST                          (default: localhost)
  --port PORT                          (default: 7860)
  --protocol PROTOCOL                  rtvi or openai-realtime (default: rtvi)
  --clients "N1 N2 ..."                Concurrency levels to run (default: "1")
  --client-start-delay SECONDS         Stagger between client connects (default: 1)
  --test-duration SECONDS              Turn-admission interval per run  (default: 300)
  --cooldown SECONDS                   Pause between sweep levels        (default: 10)
  --reverse-barge-in-threshold SECS    See benchmark.py --help            (default: 0.4)
  --turn-response-timeout SECS         First-output/activity timeout      (default: 10)
  --dataset-dir DIR                    16 kHz mono WAVs                  (default: ./dataset)
  --output-dir DIR                     Where results land                (default: ./)
  --no-save-audio                      Skip per-client output WAVs
  --realtime-scheme SCHEME             ws or wss (default: wss)
  --realtime-path PATH                 Realtime WebSocket path (default: /v1/realtime)
  --realtime-model MODEL               Optional immutable model query parameter
  --realtime-voice VOICE               Optional nested session audio output voice
  --realtime-instructions TEXT         Optional Realtime session instructions
  --realtime-input-mode MODE           audio or text (default: audio)
  --realtime-text-input TEXT           Text-mode prompt; repeat to cycle prompts
  --realtime-output-modality MODE      audio or text (default: audio)
  --realtime-turn-mode MODE            automatic or manual (default: automatic)
  --realtime-vad-silence-ms MS         Silence for automatic mode (default: 800)
  --realtime-api-key-env NAME          Optional environment variable containing Bearer token
  --realtime-ca-file FILE              Optional CA bundle for verified wss
  --realtime-insecure                  Disable wss certificate/hostname verification explicitly
  --realtime-client-tools-config FILE  Realtime function/MCP tools and exact-name function handlers
  --realtime-tool-timeout SECS         Correlated tool-output timeout     (default: 120)
  --realtime-max-tool-rounds N         Maximum tool-call rounds per turn  (default: 4)
  --realtime-session-timeout SECS      Realtime session-init timeout      (default: 60)
  -h, --help                           Show this message
USAGE
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)                         HOST="$2"; shift 2 ;;
    --port)                         PORT="$2"; shift 2 ;;
    --protocol)                     PROTOCOL="$2"; shift 2 ;;
    --clients)                      CLIENT_COUNTS="$2"; shift 2 ;;
    --client-start-delay)           CLIENT_START_DELAY="$2"; shift 2 ;;
    --test-duration)                TEST_DURATION="$2"; shift 2 ;;
    --cooldown)                     COOLDOWN="$2"; shift 2 ;;
    --reverse-barge-in-threshold)   REVERSE_BARGE_IN_THRESHOLD="$2"; shift 2 ;;
    --turn-response-timeout)        TURN_RESPONSE_TIMEOUT="$2"; shift 2 ;;
    --dataset-dir)                  DATASET_DIR="$2"; shift 2 ;;
    --output-dir)                   OUTPUT_DIR="$2"; shift 2 ;;
    --no-save-audio)                SAVE_AUDIO="0"; shift ;;
    --realtime-scheme)              REALTIME_SCHEME="$2"; shift 2 ;;
    --realtime-path)                REALTIME_PATH="$2"; shift 2 ;;
    --realtime-model)               REALTIME_MODEL="$2"; shift 2 ;;
    --realtime-voice)               REALTIME_VOICE="$2"; shift 2 ;;
    --realtime-instructions)        REALTIME_INSTRUCTIONS="$2"; shift 2 ;;
    --realtime-input-mode)          REALTIME_INPUT_MODE="$2"; shift 2 ;;
    --realtime-text-input)          REALTIME_TEXT_INPUTS+=("$2"); shift 2 ;;
    --realtime-output-modality)     REALTIME_OUTPUT_MODALITY="$2"; shift 2 ;;
    --realtime-turn-mode)           REALTIME_TURN_MODE="$2"; shift 2 ;;
    --realtime-vad-silence-ms)      REALTIME_VAD_SILENCE_MS="$2"; shift 2 ;;
    --realtime-api-key-env)         REALTIME_API_KEY_ENV="$2"; shift 2 ;;
    --realtime-ca-file)             REALTIME_CA_FILE="$2"; shift 2 ;;
    --realtime-insecure)            REALTIME_INSECURE="1"; shift ;;
    --realtime-client-tools-config) REALTIME_CLIENT_TOOLS_CONFIG="$2"; shift 2 ;;
    --realtime-tool-timeout)        REALTIME_TOOL_TIMEOUT="$2"; shift 2 ;;
    --realtime-max-tool-rounds)     REALTIME_MAX_TOOL_ROUNDS="$2"; shift 2 ;;
    --realtime-session-timeout)     REALTIME_SESSION_TIMEOUT="$2"; shift 2 ;;
    -h|--help)                      print_usage; exit 0 ;;
    *)                              echo "Unknown option: $1" >&2; print_usage >&2; exit 1 ;;
  esac
done

if [[ "$PROTOCOL" != "rtvi" && "$PROTOCOL" != "openai-realtime" ]]; then
  echo "Invalid --protocol: $PROTOCOL (expected rtvi or openai-realtime)" >&2
  exit 1
fi
if [[ "$PROTOCOL" == "openai-realtime" ]]; then
  if [[ "$REALTIME_INPUT_MODE" != "audio" && "$REALTIME_INPUT_MODE" != "text" ]]; then
    echo "Invalid --realtime-input-mode: $REALTIME_INPUT_MODE (expected audio or text)" >&2
    exit 1
  fi
  if [[ "$REALTIME_INPUT_MODE" == "text" && "${#REALTIME_TEXT_INPUTS[@]}" -eq 0 ]]; then
    echo "--realtime-input-mode text requires at least one --realtime-text-input" >&2
    exit 1
  fi
  if [[ "$REALTIME_SCHEME" != "ws" && "$REALTIME_SCHEME" != "wss" ]]; then
    echo "Invalid --realtime-scheme: $REALTIME_SCHEME (expected ws or wss)" >&2
    exit 1
  fi
  if [[ "$REALTIME_SCHEME" == "ws" && -n "$REALTIME_API_KEY_ENV" ]]; then
    echo "Bearer credentials cannot be sent over an unencrypted ws connection" >&2
    exit 1
  fi
  if [[ "$REALTIME_SCHEME" == "ws" && "$REALTIME_INSECURE" == "1" ]]; then
    echo "--realtime-insecure applies only to wss" >&2
    exit 1
  fi
  if [[ "$REALTIME_SCHEME" == "ws" && -n "$REALTIME_CA_FILE" ]]; then
    echo "--realtime-ca-file applies only to wss" >&2
    exit 1
  fi
  if [[ "$REALTIME_INSECURE" == "1" && -n "$REALTIME_CA_FILE" ]]; then
    echo "--realtime-insecure and --realtime-ca-file are mutually exclusive" >&2
    exit 1
  fi
  if [[ -n "$REALTIME_CA_FILE" && ! -f "$REALTIME_CA_FILE" ]]; then
    echo "Realtime CA file not found: $REALTIME_CA_FILE" >&2
    exit 1
  fi
  if [[ -n "$REALTIME_CLIENT_TOOLS_CONFIG" && ! -f "$REALTIME_CLIENT_TOOLS_CONFIG" ]]; then
    echo "Realtime client-tools config not found: $REALTIME_CLIENT_TOOLS_CONFIG" >&2
    exit 1
  fi
  if [[ ! "$REALTIME_MAX_TOOL_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --realtime-max-tool-rounds: $REALTIME_MAX_TOOL_ROUNDS" >&2
    exit 1
  fi
fi

if [[ "$PROTOCOL" != "openai-realtime" || "$REALTIME_INPUT_MODE" == "audio" ]] && [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output path is not a directory: $OUTPUT_DIR" >&2
  exit 1
fi

# Single python helper for arithmetic that bash can't easily do (floats).
py_calc() {
  "${PY[@]}" -c "$@"
  return $?
}

timestamp="$(date -u +%Y%m%d_%H%M%S_%N)_${BASHPID}"
read -ra CLIENT_COUNTS_ARR <<<"$CLIENT_COUNTS"
total_runs="${#CLIENT_COUNTS_ARR[@]}"
if [[ "$total_runs" -eq 0 ]]; then
  echo "--clients must contain at least one positive integer" >&2
  exit 1
fi
declare -A SEEN_CLIENT_COUNTS=()
for num_clients in "${CLIENT_COUNTS_ARR[@]}"; do
  if [[ ! "$num_clients" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid client count: $num_clients (expected a positive integer)" >&2
    exit 1
  fi
  if [[ -n "${SEEN_CLIENT_COUNTS[$num_clients]:-}" ]]; then
    echo "Duplicate client count: $num_clients" >&2
    exit 1
  fi
  SEEN_CLIENT_COUNTS[$num_clients]="1"
done

# Sweep (>1 level) → perf_suite_<ts>/  | single level → results_<ts>/
if [[ "$total_runs" -gt 1 ]]; then
  suite_dir="$OUTPUT_DIR/perf_suite_${timestamp}"
  if ! mkdir "$suite_dir"; then
    echo "Refusing to reuse existing suite directory: $suite_dir" >&2
    exit 1
  fi
else
  suite_dir=""
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                 VOICE AGENT PERF BENCHMARK                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Host:Port     : ${HOST}:${PORT}"
echo "Protocol      : ${PROTOCOL}"
if [[ "$PROTOCOL" == "openai-realtime" && "$REALTIME_INPUT_MODE" == "text" ]]; then
  echo "Text prompts  : ${#REALTIME_TEXT_INPUTS[@]}"
else
  echo "Dataset       : ${DATASET_DIR}"
fi
echo "Client counts : ${CLIENT_COUNTS}"
echo "Test duration : ${TEST_DURATION}s"
[[ -n "$suite_dir" ]] && echo "Suite dir     : ${suite_dir}"
echo ""

run_index=0
for num_clients in "${CLIENT_COUNTS_ARR[@]}"; do
  run_index=$((run_index + 1))

  if [[ -n "$suite_dir" ]]; then
    run_dir="$suite_dir/run_${num_clients}_clients"
  else
    run_dir="$OUTPUT_DIR/results_${timestamp}"
  fi
  if ! mkdir "$run_dir"; then
    echo "Refusing to reuse existing run directory: $run_dir" >&2
    exit 1
  fi

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  RUN ${run_index}/${total_runs}: ${num_clients} parallel client(s) → ${run_dir}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Give all clients the same wall-clock turn-admission interval.
  metrics_start="$(py_calc "import time; print(time.time() + max(0.0, ($num_clients - 1) * $CLIENT_START_DELAY))")"
  session_end="$(py_calc "print($metrics_start + $TEST_DURATION)")"

  pids=()
  for i in $(seq 1 "$num_clients"); do
    start_delay="$(py_calc "print(($i - 1) * $CLIENT_START_DELAY)")"
    stream_id="client_${i}_${timestamp}"
    client_dir="$run_dir/$stream_id"
    if ! mkdir "$client_dir"; then
      echo "Refusing to reuse existing client directory: $client_dir" >&2
      exit 1
    fi

    audio_args=()
    if [[ "$SAVE_AUDIO" == "1" ]]; then
      audio_args+=(--audio-output-path "$client_dir/audio_output_${stream_id}.wav")
    else
      audio_args+=(--no-save-audio)
    fi

    protocol_args=(--protocol "$PROTOCOL")
    if [[ "$PROTOCOL" == "openai-realtime" ]]; then
      protocol_args+=(
        --realtime-scheme "$REALTIME_SCHEME"
        --realtime-path "$REALTIME_PATH"
        --realtime-input-mode "$REALTIME_INPUT_MODE"
        --realtime-output-modality "$REALTIME_OUTPUT_MODALITY"
        --realtime-turn-mode "$REALTIME_TURN_MODE"
        --realtime-vad-silence-ms "$REALTIME_VAD_SILENCE_MS"
        --realtime-tool-timeout "$REALTIME_TOOL_TIMEOUT"
        --realtime-max-tool-rounds "$REALTIME_MAX_TOOL_ROUNDS"
        --realtime-session-timeout "$REALTIME_SESSION_TIMEOUT"
      )
      [[ -n "$REALTIME_MODEL" ]] && protocol_args+=(--realtime-model "$REALTIME_MODEL")
      [[ -n "$REALTIME_VOICE" ]] && protocol_args+=(--realtime-voice "$REALTIME_VOICE")
      [[ -n "$REALTIME_INSTRUCTIONS" ]] && protocol_args+=(--realtime-instructions "$REALTIME_INSTRUCTIONS")
      for text_input in "${REALTIME_TEXT_INPUTS[@]}"; do
        protocol_args+=(--realtime-text-input "$text_input")
      done
      [[ -n "$REALTIME_API_KEY_ENV" ]] && protocol_args+=(--realtime-api-key-env "$REALTIME_API_KEY_ENV")
      [[ -n "$REALTIME_CA_FILE" ]] && protocol_args+=(--realtime-ca-file "$REALTIME_CA_FILE")
      [[ "$REALTIME_INSECURE" == "1" ]] && protocol_args+=(--realtime-insecure)
      [[ -n "$REALTIME_CLIENT_TOOLS_CONFIG" ]] && \
        protocol_args+=(--realtime-client-tools-config "$REALTIME_CLIENT_TOOLS_CONFIG")
    fi

    "${PY[@]}" "$BENCHMARK_PY" \
      --host "$HOST" --port "$PORT" \
      "${protocol_args[@]}" \
      --dataset-dir "$DATASET_DIR" \
      --stream-id "$stream_id" \
      --start-delay "$start_delay" \
      --metrics-start-time "$metrics_start" \
      --session-end-time "$session_end" \
      --test-duration "$TEST_DURATION" \
      --reverse-barge-in-threshold "$REVERSE_BARGE_IN_THRESHOLD" \
      --turn-response-timeout "$TURN_RESPONSE_TIMEOUT" \
      --result-path "$client_dir/result_${stream_id}.json" \
      --logger-path "$client_dir/benchmark_${stream_id}.log" \
      "${audio_args[@]}" \
      >"$client_dir/process_stdout.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "  worker pid=$pid exited non-zero (continuing to produce reports)" >&2
      RUN_FAILED="1"
    fi
  done

  if ! "${PY[@]}" "$BENCHMARK_PY" --aggregate-run-dir "$run_dir" --num-clients "$num_clients"; then
    RUN_FAILED="1"
  fi

  if [[ "$run_index" -lt "$total_runs" ]]; then
    sleep "$COOLDOWN"
  fi
done

if [[ -n "$suite_dir" ]]; then
  echo ""
  if ! "${PY[@]}" "$BENCHMARK_PY" --aggregate-suite-dir "$suite_dir"; then
    RUN_FAILED="1"
  fi
else
  # Single-level run: emit the same results.{txt,tsv,json} table (one row)
  # so single runs are directly comparable with sweep outputs.
  echo ""
  if ! "${PY[@]}" "$BENCHMARK_PY" --aggregate-suite-dir "$run_dir"; then
    RUN_FAILED="1"
  fi
fi

exit "$RUN_FAILED"
