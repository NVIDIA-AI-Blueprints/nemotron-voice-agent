# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Webcam worker package."""

from examples.omni_assistant_subagents.subagents.webcam.agent import (
    SPEAKER_STATE_PREFIXES,
    WEBCAM_CONTEXT_PREFIX,
    WEBCAM_FIRST_SIGHT_PREFIX,
    WEBCAM_SUMMARY_TASK_NAME,
    WebcamAgent,
)

__all__ = [
    "SPEAKER_STATE_PREFIXES",
    "WEBCAM_CONTEXT_PREFIX",
    "WEBCAM_FIRST_SIGHT_PREFIX",
    "WEBCAM_SUMMARY_TASK_NAME",
    "WebcamAgent",
]
