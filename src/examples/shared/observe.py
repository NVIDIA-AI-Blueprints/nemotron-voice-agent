# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default observation hook of the Live parts: one debug log line per event."""

from loguru import logger


def log_observation(kind: str, **fields) -> None:
    """Default observation hook: one debug log line per event."""
    logger.debug(f"live.{kind} {fields}")
