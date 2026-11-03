# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional bounded pool timing sidecars without inspecting stream payloads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from pathlib import Path

SCHEMA = "nva.offline.pool-metrics.v1"
ENVIRONMENT = "NVA_OFFLINE_POOL_METRICS_DIR"
IDENTITY = re.compile(r"[0-9a-f]{64}\Z")
FIELDS = frozenset(
    {
        "worker_pid",
        "worker_id",
        "server_metrics_session_id",
        "admission_started_monotonic_ns",
        "admission_acquired_monotonic_ns",
        "identity_check_started_monotonic_ns",
        "identity_check_finished_monotonic_ns",
        "backend_connect_started_monotonic_ns",
        "backend_connect_finished_monotonic_ns",
        "relay_finished_monotonic_ns",
        "backend_close_started_monotonic_ns",
        "backend_close_finished_monotonic_ns",
        "handler_return_verified_monotonic_ns",
        "profile_finished_monotonic_ns",
        "handler_started_monotonic_ns",
        "native_app_exit_monotonic_ns",
        "lease_released_monotonic_ns",
        "diagnostics_persistence_started_monotonic_ns",
        "diagnostics_persistence_completed_monotonic_ns",
        "status_polls",
        "upstream_send_attempts",
        "upstream_send_failures",
        "upstream_send_wall_ns_sum",
        "upstream_send_wall_ns_max",
        "downstream_send_attempts",
        "downstream_send_failures",
        "downstream_send_wall_ns_sum",
        "downstream_send_wall_ns_max",
        "reusable",
        "outcome",
    }
)
OUTCOMES = frozenset({"returned", "cancelled", "error", "IDLE", "QUARANTINED", "DRAINING", "LEASED"})


def lease_digest(lease_id):
    """Expose only an opaque digest of the private admission identifier."""
    return hashlib.sha256(lease_id.encode("ascii")).hexdigest()


def valid_document(value):
    """Validate the entire scalar schema before persistence or publication."""
    if not isinstance(value, dict) or set(value) - FIELDS - {"schema", "kind", "lease_sha256", "pid"}:
        return False
    if (
        value.get("schema") != SCHEMA
        or not isinstance(value.get("kind"), str)
        or value["kind"] not in {"worker", "proxy"}
        or not isinstance(value.get("lease_sha256"), str)
        or IDENTITY.fullmatch(value["lease_sha256"]) is None
        or type(value.get("pid")) is not int
        or value["pid"] <= 1
    ):
        return False
    for name, item in value.items():
        if name not in FIELDS:
            continue
        if item is None:
            continue
        if name == "outcome":
            if not isinstance(item, str) or item not in OUTCOMES:
                return False
        elif name == "worker_id":
            if not isinstance(item, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", item) is None:
                return False
        elif name == "server_metrics_session_id":
            if not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{32}", item) is None:
                return False
        elif type(item) not in (bool, int, float) or (type(item) is float and not math.isfinite(item)) or item < 0:
            return False
    return True


class PoolTelemetry:
    """Drop optional observations when a bounded I/O budget is exhausted."""

    def __init__(self, root: Path, *, max_pending=32):
        """Bind a trusted directory and a finite optional write budget."""
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("Pool observation capacity must be positive")
        self.root = Path(root)
        self.max_pending = max_pending
        self.tasks = set()
        self.submitted = self.written = self.rejected = self.dropped = 0
        self.closed = False

    @classmethod
    def from_environment(cls):
        """Accept only the trusted operator environment, never session input."""
        value = os.environ.get(ENVIRONMENT, "").strip()
        return cls(Path(value)) if value else None

    def submit(self, kind, digest, **fields):
        """Schedule scalar persistence without blocking a native handler."""
        value = {"schema": SCHEMA, "kind": kind, "lease_sha256": digest, "pid": os.getpid(), **fields}
        if not valid_document(value):
            self.rejected += 1
            return
        if self.closed or len(self.tasks) >= self.max_pending:
            self.dropped += 1
            return
        self.submitted += 1
        task = asyncio.create_task(self._persist(value))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _persist(self, value):
        try:
            await asyncio.to_thread(self._write, value)
        except Exception:
            self.rejected += 1
        else:
            self.written += 1

    def _write(self, value):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("Pool observation directory must be operator-owned")
        self.root.chmod(0o700)
        path = self.root / f"{value['lease_sha256']}.{value['kind']}.json"
        temporary = path.with_suffix(".part")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output, allow_nan=False)
                output.write("\n")
            # Readers see only complete immutable files; link refuses an
            # existing destination instead of replacing another observation.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def status(self):
        """Expose observation coverage counters without payloads or credentials."""
        return {key: getattr(self, key) for key in ("submitted", "written", "rejected", "dropped")}

    async def aclose(self):
        """Bound observer shutdown; it never certifies native provider cleanup."""
        self.closed = True
        tasks = tuple(self.tasks)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=2)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
