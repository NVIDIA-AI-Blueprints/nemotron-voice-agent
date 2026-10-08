# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat-completions implementations (NVIDIA NIM, vLLM, hosted endpoints)."""

from .backend import ChatCompletionsBackend
from .frontend import ChatCompletionsFrontend

__all__ = ["ChatCompletionsBackend", "ChatCompletionsFrontend"]
