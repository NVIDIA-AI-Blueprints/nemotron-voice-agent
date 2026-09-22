# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Narrow executable contract for the managed VoiceClaw image."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voiceclaw import container


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voiceclaw-runtime", description="Run the VoiceClaw image runtime")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve VoiceClaw in the foreground")
    serve.add_argument(
        "--config",
        type=Path,
        help="VoiceClaw YAML configuration (defaults to the managed volume path)",
    )
    serve.add_argument("--ui", action="store_true", help="serve the optional VoiceClaw browser UI")

    commands.add_parser("healthcheck", help="probe the running facade liveness endpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one of the two image-owned runtime operations."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "healthcheck":
        container.main(["--healthcheck"])
        return

    forwarded: list[str] = []
    if arguments.config is not None:
        forwarded.extend(("--config", str(arguments.config)))
    if arguments.ui:
        forwarded.append("--ui")
    container.main(forwarded)


if __name__ == "__main__":
    main()
