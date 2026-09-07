#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Stable CLI entrypoint for the scaling performance benchmark."""

# ruff: noqa: D103

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from benchmark_aggregation import (
    run_aggregate_run,
    run_aggregate_suite,
)
from benchmark_core import (
    BOT_INTRO_TIMEOUT,
    REALTIME_TURN_RESPONSE_TIMEOUT,
    TURN_RESPONSE_TIMEOUT,
    PerfClient,
    RunLogger,
    request_shutdown,
    round3,
)
from realtime_client import RealtimeClient
from realtime_transport import (
    DEFAULT_CONNECT_TIMEOUT,
    resolve_api_key,
    resolve_auth_scheme,
    resolve_ws_url,
)
from rtvi_client import RTVIClient

WS_CONNECT_TIMEOUT = 30


def resolve_protocol(*, protocol: str = "", ws_url: str = "", explicit_ws_url: str = "") -> str:
    """Resolve the CLI-selected wire protocol."""
    value = (protocol or "").strip().lower()
    if value:
        if value not in {"rtvi", "realtime"}:
            raise ValueError(f"unsupported protocol {protocol!r}; expected rtvi or realtime")
        if value == "rtvi" and (explicit_ws_url or "").strip():
            raise ValueError("--protocol rtvi conflicts with an explicit --ws-url")
        return value
    return "realtime" if (ws_url or "").strip() else "rtvi"


def _signal_handler(signum, frame):
    del frame
    request_shutdown(signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_path).resolve() if args.result_path else out / f"result_{args.stream_id}.json"
    logger_path = Path(args.logger_path).resolve() if args.logger_path else out / f"benchmark_{args.stream_id}.log"
    if args.audio_output_path:
        audio_output_path: Path | None = Path(args.audio_output_path).resolve()
    elif args.save_audio:
        audio_output_path = out / f"audio_output_{args.stream_id}.wav"
    else:
        audio_output_path = None
    return result_path, logger_path, audio_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Single-client voice-agent benchmark. Use simulate_concurrency.sh for parallel runs."
    )
    parser.add_argument("--host", default="localhost", help="WebSocket host (RTVI)")
    parser.add_argument("--port", type=int, default=7860, help="WebSocket port (RTVI)")
    parser.add_argument(
        "--protocol",
        default="",
        help="rtvi or realtime. Defaults to realtime when --ws-url is set, otherwise rtvi.",
    )
    parser.add_argument(
        "--ws-url",
        default="",
        help="OpenAI Realtime WebSocket URL (env OPENAI_REALTIME_WS_URL). Implies --protocol realtime.",
    )
    parser.add_argument(
        "--auth-scheme",
        default="",
        help="Authorization scheme for OPENAI_REALTIME_API_KEY (env OPENAI_REALTIME_AUTH_SCHEME; default: Bearer).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=None,
        help="WebSocket handshake timeout in seconds (default: 30 RTVI, 60 realtime).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for a Realtime or RTVI wss:// endpoint.",
    )
    intro_group = parser.add_mutually_exclusive_group()
    intro_group.add_argument(
        "--drain-bot-intro",
        dest="drain_bot_intro",
        action="store_true",
        default=None,
        help="Wait for and discard an initial bot utterance. Default on for both protocols.",
    )
    intro_group.add_argument(
        "--skip-bot-intro",
        dest="drain_bot_intro",
        action="store_false",
        help="Do not wait for an initial bot utterance.",
    )
    parser.add_argument(
        "--bot-intro-timeout",
        type=float,
        default=BOT_INTRO_TIMEOUT,
        help=f"Seconds to wait for initial bot audio before continuing (default: {BOT_INTRO_TIMEOUT}).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=script_dir / "dataset",
        help="Directory containing 16 kHz mono WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Default parent directory for result/log/audio outputs",
    )
    parser.add_argument("--stream-id", default="", help="Stream id; auto-generated when empty")
    parser.add_argument("--start-delay", type=float, default=0.0, help="Seconds to wait before connecting")
    parser.add_argument(
        "--metrics-start-time",
        type=float,
        help="Unix epoch seconds when metric collection should begin (defaults to now+start_delay)",
    )
    parser.add_argument(
        "--session-end-time",
        type=float,
        help="Unix epoch seconds when this client should stop (defaults to metrics_start+test_duration)",
    )
    parser.add_argument("--test-duration", type=float, default=300.0, help="Metric collection window in seconds")
    parser.add_argument(
        "--reverse-barge-in-threshold",
        type=float,
        default=0.4,
        help=(
            "Latency (seconds) below which a turn is classified as a reverse "
            "barge-in and excluded from average/p95 reporting. Turns are still "
            "recorded in individual_latencies for analysis."
        ),
    )
    parser.add_argument(
        "--turn-response-timeout",
        type=float,
        default=None,
        help=(
            "Per-turn timeout (seconds) waiting for the bot's first audio frame "
            "after the input audio file finishes sending. Defaults to "
            f"{TURN_RESPONSE_TIMEOUT} for RTVI and {REALTIME_TURN_RESPONSE_TIMEOUT} "
            "for realtime. On timeout the turn is recorded as a failed_turn."
        ),
    )
    parser.add_argument("--result-path", type=Path, help="Write client result JSON here")
    parser.add_argument("--logger-path", type=Path, help="Write client log here")
    parser.add_argument("--audio-output-path", type=Path, help="Write client output WAV here")
    parser.set_defaults(save_audio=True)
    parser.add_argument(
        "--no-save-audio",
        dest="save_audio",
        action="store_false",
        help="Disable writing the default per-client output WAV",
    )
    aggregate = parser.add_argument_group("aggregation modes (post-run)")
    aggregate.add_argument(
        "--aggregate-run-dir",
        type=Path,
        help="Fold a directory of client_*/result_*.json into benchmark_summary.json",
    )
    aggregate.add_argument(
        "--aggregate-suite-dir",
        type=Path,
        help="Fold run_*_clients/benchmark_summary.json files into results.{tsv,txt,json}",
    )
    aggregate.add_argument(
        "--num-clients",
        type=int,
        help="Configured client count for --aggregate-run-dir (defaults to file count)",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    if not args.dataset_dir.is_dir():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        return 2
    audio_files = sorted(path for path in args.dataset_dir.iterdir() if path.suffix.lower() == ".wav")
    if not audio_files:
        print(f"No .wav files found in {args.dataset_dir}", file=sys.stderr)
        return 2
    if not args.stream_id:
        args.stream_id = f"client_1_{str(time.time_ns())[:13]}"
    if args.metrics_start_time is not None and args.session_end_time is None:
        args.session_end_time = args.metrics_start_time + args.test_duration
    result_path, logger_path, audio_output_path = _resolve_paths(args)

    ws_url = resolve_ws_url(args.ws_url)
    try:
        protocol = resolve_protocol(protocol=args.protocol, ws_url=ws_url, explicit_ws_url=args.ws_url)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if protocol == "realtime" and not ws_url:
        print("error: --ws-url or OPENAI_REALTIME_WS_URL is required for realtime", file=sys.stderr)
        return 2
    connect_timeout = (
        float(args.connect_timeout)
        if args.connect_timeout is not None
        else (DEFAULT_CONNECT_TIMEOUT if protocol == "realtime" else WS_CONNECT_TIMEOUT)
    )
    turn_response_timeout = (
        float(args.turn_response_timeout)
        if args.turn_response_timeout is not None
        else (REALTIME_TURN_RESPONSE_TIMEOUT if protocol == "realtime" else TURN_RESPONSE_TIMEOUT)
    )
    logger = RunLogger(logger_path)
    if protocol == "realtime":
        protocol_client = RealtimeClient(
            stream_id=args.stream_id,
            ws_url=ws_url,
            logger=logger,
            api_key=resolve_api_key(),
            auth_scheme=resolve_auth_scheme(args.auth_scheme),
            connect_timeout=connect_timeout,
            verify_tls=not args.insecure,
            audio_files=audio_files,
        )
    else:
        protocol_client = RTVIClient(
            stream_id=args.stream_id,
            host=args.host,
            port=int(args.port),
            logger=logger,
            connect_timeout=connect_timeout,
            verify_tls=not args.insecure,
        )
    client = PerfClient(
        stream_id=args.stream_id,
        protocol_client=protocol_client,
        audio_files=audio_files,
        start_delay=float(args.start_delay),
        metrics_start_time=(float(args.metrics_start_time) if args.metrics_start_time is not None else None),
        session_end_time=(float(args.session_end_time) if args.session_end_time is not None else None),
        test_duration=float(args.test_duration),
        reverse_barge_in_threshold=float(args.reverse_barge_in_threshold),
        turn_response_timeout=turn_response_timeout,
        audio_output_path=audio_output_path,
        logger=logger,
        drain_bot_intro=args.drain_bot_intro,
        bot_intro_timeout=float(args.bot_intro_timeout),
    )
    result = await client.run()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(
        f"client={args.stream_id} turns={result.num_turns} "
        f"avg_latency={round3(result.average_latency)}s "
        f"glitch={result.glitch_detected} "
        f"result={result_path}"
    )
    return 1 if result.error else 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.aggregate_run_dir is not None:
        return run_aggregate_run(args.aggregate_run_dir, args.num_clients)
    if args.aggregate_suite_dir is not None:
        return run_aggregate_suite(args.aggregate_suite_dir)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
