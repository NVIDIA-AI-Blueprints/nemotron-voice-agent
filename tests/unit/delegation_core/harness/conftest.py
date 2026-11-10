# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's tests need the packages of the harness dependency group, and are skipped without them."""

import pytest

from tests.unit.core_helpers import HARNESS_GROUP, NO_HARNESS_GROUP

if not HARNESS_GROUP:
    pytest.skip(NO_HARNESS_GROUP, allow_module_level=True)
