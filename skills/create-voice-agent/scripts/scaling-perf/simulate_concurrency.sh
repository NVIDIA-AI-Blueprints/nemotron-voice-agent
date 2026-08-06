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
#   `now + (N-1)*D` so every worker is connected before measurement starts.
#   Each worker is told the same `--metrics-start-time` and `--session-end-time`.
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
CLIENT_COUNTS="1"
CLIENT_START_DELAY="1"
TEST_DURATION="300"
COOLDOWN="10"
REVERSE_BARGE_IN_THRESHOLD="0.4"
DATASET_DIR="$SCRIPT_DIR/dataset"
OUTPUT_DIR="$SCRIPT_DIR"
SAVE_AUDIO="1"

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --host HOST                          (default: localhost)
  --port PORT                          (default: 7860)
  --clients "N1 N2 ..."                Concurrency levels to run (default: "1")
  --client-start-delay SECONDS         Stagger between client connects (default: 1)
  --test-duration SECONDS              Metric collection window per run  (default: 300)
  --cooldown SECONDS                   Pause between sweep levels        (default: 10)
  --reverse-barge-in-threshold SECS    See benchmark.py --help            (default: 0.4)
  --dataset-dir DIR                    16 kHz mono WAVs                  (default: ./dataset)
  --output-dir DIR                     Where results land                (default: ./)
  --no-save-audio                      Skip per-client output WAVs
  -h, --help                           Show this message
USAGE
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)                         HOST="$2"; shift 2 ;;
    --port)                         PORT="$2"; shift 2 ;;
    --clients)                      CLIENT_COUNTS="$2"; shift 2 ;;
    --client-start-delay)           CLIENT_START_DELAY="$2"; shift 2 ;;
    --test-duration)                TEST_DURATION="$2"; shift 2 ;;
    --cooldown)                     COOLDOWN="$2"; shift 2 ;;
    --reverse-barge-in-threshold)   REVERSE_BARGE_IN_THRESHOLD="$2"; shift 2 ;;
    --dataset-dir)                  DATASET_DIR="$2"; shift 2 ;;
    --output-dir)                   OUTPUT_DIR="$2"; shift 2 ;;
    --no-save-audio)                SAVE_AUDIO="0"; shift ;;
    -h|--help)                      print_usage; exit 0 ;;
    *)                              echo "Unknown option: $1" >&2; print_usage >&2; exit 1 ;;
  esac
done

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

# Single python helper for arithmetic that bash can't easily do (floats).
py_calc() {
  "${PY[@]}" -c "$@"
  return $?
}

timestamp="$(date +%Y%m%d_%H%M%S)"
read -ra CLIENT_COUNTS_ARR <<<"$CLIENT_COUNTS"
total_runs="${#CLIENT_COUNTS_ARR[@]}"

# Sweep (>1 level) → perf_suite_<ts>/  | single level → results_<ts>/
if [[ "$total_runs" -gt 1 ]]; then
  suite_dir="$OUTPUT_DIR/perf_suite_${timestamp}"
  mkdir -p "$suite_dir"
else
  suite_dir=""
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                 VOICE AGENT PERF BENCHMARK                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Host:Port     : ${HOST}:${PORT}"
echo "Dataset       : ${DATASET_DIR}"
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
  mkdir -p "$run_dir"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  RUN ${run_index}/${total_runs}: ${num_clients} parallel client(s) → ${run_dir}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Synchronize all clients in this level on a single metrics window.
  metrics_start="$(py_calc "import time; print(time.time() + max(0.0, ($num_clients - 1) * $CLIENT_START_DELAY))")"
  session_end="$(py_calc "print($metrics_start + $TEST_DURATION)")"

  pids=()
  for i in $(seq 1 "$num_clients"); do
    start_delay="$(py_calc "print(($i - 1) * $CLIENT_START_DELAY)")"
    stream_id="client_${i}_$(date +%s%N | cut -c1-13)"
    client_dir="$run_dir/$stream_id"
    mkdir -p "$client_dir"

    audio_args=()
    if [[ "$SAVE_AUDIO" == "1" ]]; then
      audio_args+=(--audio-output-path "$client_dir/audio_output_${stream_id}.wav")
    else
      audio_args+=(--no-save-audio)
    fi

    "${PY[@]}" "$BENCHMARK_PY" \
      --host "$HOST" --port "$PORT" \
      --dataset-dir "$DATASET_DIR" \
      --stream-id "$stream_id" \
      --start-delay "$start_delay" \
      --metrics-start-time "$metrics_start" \
      --session-end-time "$session_end" \
      --test-duration "$TEST_DURATION" \
      --reverse-barge-in-threshold "$REVERSE_BARGE_IN_THRESHOLD" \
      --result-path "$client_dir/result_${stream_id}.json" \
      --logger-path "$client_dir/benchmark_${stream_id}.log" \
      "${audio_args[@]}" \
      >"$client_dir/process_stdout.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || echo "  worker pid=$pid exited non-zero (continuing)" >&2
  done

  "${PY[@]}" "$BENCHMARK_PY" --aggregate-run-dir "$run_dir" --num-clients "$num_clients"

  if [[ "$run_index" -lt "$total_runs" ]]; then
    sleep "$COOLDOWN"
  fi
done

if [[ -n "$suite_dir" ]]; then
  echo ""
  "${PY[@]}" "$BENCHMARK_PY" --aggregate-suite-dir "$suite_dir"
else
  # Single-level run: emit the same results.{txt,tsv,json} table (one row)
  # so single runs are directly comparable with sweep outputs.
  echo ""
  "${PY[@]}" "$BENCHMARK_PY" --aggregate-suite-dir "$run_dir"
fi
