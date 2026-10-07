# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Narrow executable contract for the VoiceClaw runtime image."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voiceclaw import container, installer_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voiceclaw-runtime", description="Run the VoiceClaw image runtime")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve VoiceClaw in the foreground")
    serve.add_argument(
        "--config",
        type=Path,
        help="VoiceClaw YAML configuration (defaults to VOICECLAW_CONFIG or the image configuration)",
    )
    serve.add_argument("--ui", action="store_true", help="serve the optional VoiceClaw browser UI")

    commands.add_parser("healthcheck", help="probe selected-agent readiness")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one of the image-owned runtime operations."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if Path("/etc/voiceclaw-nemoclaw-container-v1").is_file():
        if arguments.command == "healthcheck":
            installer_runtime.run_healthcheck()
        else:
            if arguments.config is not None or arguments.ui:
                parser.error("the installer profile does not accept configuration or UI overrides")
            installer_runtime.main()
        return
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
