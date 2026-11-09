# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default observation hook: one debug log line per event."""

from loguru import logger


def log_observation(kind: str, **fields) -> None:
    """Log one debug line per observed event."""
    logger.debug(f"live.{kind} {fields}")
