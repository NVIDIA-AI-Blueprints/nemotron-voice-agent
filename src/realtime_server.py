# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Standalone OpenAI Realtime-compatible API server (no browser UI routes)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from realtime.protocol import MAX_REALTIME_EVENT_BYTES
from server import create_realtime_app
from utils import parse_env_bool, parse_env_int


def _parse_min_int(value: str, minimum: int = 1) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"must be >= {minimum}")
    return parsed


def _configure_logging(verbose: int) -> None:
    logger.remove()
    logger.configure(extra={"stream_id": "-"})
    logger.add(
        sys.stderr,
        level="TRACE" if verbose else "DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "{name}:{function}:{line} - "
            "[stream_id={extra[stream_id]}] "
            "<level>{message}</level>"
        ),
    )


def _ssl_config(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    if not parse_env_bool("PIPELINE_TLS", default=True):
        return "http", {}
    if args.tls_cert and args.tls_key:
        return "https", {"ssl_certfile": args.tls_cert, "ssl_keyfile": args.tls_key}

    from utils import ensure_self_signed_cert

    cert_dir = Path(__file__).resolve().parent.parent / ".certs"
    cert_file, key_file = ensure_self_signed_cert(cert_dir)
    return "https", {"ssl_certfile": cert_file, "ssl_keyfile": key_file}


def app_factory():
    """Build an API-only app instance for each uvicorn worker."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prompt-file", type=str, default="")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return create_realtime_app(host=args.host, prompt_file=args.prompt_file)


def main() -> None:
    """Start the standalone Realtime API with optional multi-worker scaling."""
    parser = argparse.ArgumentParser(description="Nemotron Voice Agent Realtime API Server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Optional prompt catalog YAML path for the selected pipeline",
    )
    parser.add_argument("--tls-cert", type=str, help="Path to TLS certificate file")
    parser.add_argument("--tls-key", type=str, help="Path to TLS key file")
    parser.add_argument(
        "--workers",
        type=lambda value: _parse_min_int(value, 1),
        default=None,
        help="Number of stateless Realtime API workers",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    _configure_logging(args.verbose)
    workers = args.workers if args.workers is not None else parse_env_int("UVICORN_WORKERS", 1, min_value=1)
    os.environ["UVICORN_WORKERS"] = str(workers)
    scheme, ssl_kwargs = _ssl_config(args)
    logger.info(f"Realtime API ready -> {scheme}://{args.host}:{args.port}/v1/realtime (workers={workers})")

    if workers > 1:
        uvicorn.run(
            "realtime_server:app_factory",
            host=args.host,
            port=args.port,
            workers=workers,
            factory=True,
            ws_max_size=MAX_REALTIME_EVENT_BYTES,
            **ssl_kwargs,
        )
    else:
        uvicorn.run(
            create_realtime_app(host=args.host, prompt_file=args.prompt_file),
            host=args.host,
            port=args.port,
            ws_max_size=MAX_REALTIME_EVENT_BYTES,
            **ssl_kwargs,
        )


if __name__ == "__main__":
    main()
