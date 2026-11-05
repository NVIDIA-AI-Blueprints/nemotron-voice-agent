# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The steps the app's function outputs take before the backend reads them, by their names in core.yaml.

A step implements ``AppOutputStep`` (``examples.frontend_backend_live.delegation_core.backend.port``) and builds itself
from its ``backend.app_output_steps`` block; each IM session gets a fresh set. ``shape_hints``
(``examples.frontend_backend_live.delegation_core.backend.recovery``) adds the written form of an id worth trying to an
app output that found nothing. Adding a step is one class and one entry here.
"""

from collections.abc import Mapping

from examples.frontend_backend_live.delegation_core.backend.recovery import ShapeHints

# Each step by its name in backend.app_output_steps.
APP_OUTPUT_STEPS: Mapping[str, type] = {"shape_hints": ShapeHints}
