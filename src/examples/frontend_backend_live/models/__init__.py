# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend and backend model roles: base classes, registry and implementations.

Importing this package registers every shipped implementation. To add one, subclass ``Frontend`` or
``Backend``, decorate it with ``@FRONTENDS.register("name")`` / ``@BACKENDS.register("name")``, import the
module here, and name it in ``config.yaml``.
"""

from . import chat_completions, openai_responses  # noqa: F401 - importing registers the providers
from .base import Backend, Frontend
from .registry import BACKENDS, FRONTENDS

__all__ = ["BACKENDS", "FRONTENDS", "Backend", "Frontend"]
