# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small JSON files that the agent worker publishes into the state volume for the UI server. Standard library only."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEPLOYMENT_FILE = "deployment.json"  # the deployment mode and the models in use


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write `payload` as JSON. The write is atomic, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".state-")
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Return the JSON in `path`, or None when the file is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
