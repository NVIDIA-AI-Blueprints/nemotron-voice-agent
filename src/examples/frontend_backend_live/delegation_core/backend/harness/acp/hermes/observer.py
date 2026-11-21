# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes observer for exact model attempts and native MCP tool origin."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from .evidence import DIRECTORY, INPUT, INVOCATION, PRIVATE_ORIGIN, ModelOrigin, OriginStore, _read, _write

PLUGIN_NAME = "voice-agent-steering-evidence"
MAX_ATTEMPTS = 512


def install_observer(home: Path) -> None:
    """Install or update the observer in an explicitly selected dedicated Hermes profile."""
    facts = home.lstat()
    if not stat.S_ISDIR(facts.st_mode) or facts.st_uid != os.getuid() or facts.st_mode & 0o077:
        raise ValueError("Hermes evidence needs a private profile directory")
    target = home / "plugins" / PLUGIN_NAME
    if observer_installed(home):
        return
    source = Path(__file__)
    files = {"__init__.py": source.read_bytes(), "evidence.py": source.with_name("evidence.py").read_bytes()}
    manifest = f"name: {PLUGIN_NAME}\nversion: 1.0.0\nkind: standalone\n".encode()
    try:
        target.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        facts = target.lstat()
        if not stat.S_ISDIR(facts.st_mode) or facts.st_uid != os.getuid() or facts.st_mode & 0o077:
            raise ValueError("Hermes observer needs an owned private plugin directory") from None
        if _read(target / "plugin.yaml") != manifest:
            raise ValueError("The existing Hermes plugin is not the voice agent observer") from None
        with TemporaryDirectory(dir=target) as staging:
            for name, data in files.items():
                _read(target / name)
                _write(Path(staging) / name, data)
            for name in files:
                (Path(staging) / name).replace(target / name)
    else:
        for name, data in files.items():
            _write(target / name, data)
        _write(target / "plugin.yaml", manifest)


def observer_installed(home: Path) -> bool:
    """Check the selected profile for the exact repository-owned observer."""
    source = Path(__file__)
    target = home / "plugins" / PLUGIN_NAME
    try:
        return all(
            (target / name).read_bytes() == path.read_bytes()
            for name, path in (("__init__.py", source), ("evidence.py", source.with_name("evidence.py")))
        )
    except OSError:
        return False


def _current_store() -> OriginStore | None:
    from agent.runtime_cwd import resolve_agent_cwd

    try:
        return OriginStore(resolve_agent_cwd() / DIRECTORY)
    except (OSError, ValueError):
        return None


def _input_markers(store: OriginStore, invocation: str, messages: list[dict]) -> tuple[str, ...] | None:
    inputs: dict[str, None] = {}
    active = False
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        lines = content.splitlines()
        if lines and lines[0].startswith(INVOCATION):
            active = lines[0] == INVOCATION + invocation
            inputs.clear()
        elif active:
            for line in lines:
                if line.startswith(INPUT) and store.issued(invocation, identity := line[len(INPUT) :]):
                    inputs[identity] = None
    return tuple(inputs) if active else None


def _assigned_tool(name: str, server: str) -> str | None:
    legacy = f"mcp__{server}__"
    if name.startswith(legacy):
        return name[len(legacy) :] or None
    component = re.sub(r"[^A-Za-z0-9_]", "_", server)
    found = re.fullmatch(rf"mcp__editor_{re.escape(component)}_[0-9a-f]{{8}}__([A-Za-z0-9_]+)", name)
    return found[1] if found else None


@dataclass
class _Attempt:
    store: OriginStore
    origin: ModelOrigin
    tools: dict[str, str] = field(default_factory=dict)


class _Observer:
    """Join the native attempt and tool hooks without retaining caller text."""

    def __init__(self):
        """Keep a bounded record of recent request attempts."""
        self.prepared: OrderedDict[tuple[str, str, str], OriginStore] = OrderedDict()
        self.attempts: OrderedDict[tuple[str, str, str], _Attempt] = OrderedDict()
        self.lock = Lock()

    def prepare(self, request, session_id, turn_id, api_request_id, api_mode, **fields):
        """Invalidate reused identities and check that every advertised tool belongs to the bridge."""
        key = session_id, turn_id, api_request_id
        with self.lock:
            self.prepared.pop(key, None)
            self.attempts.pop(key, None)
        store = _current_store()
        if store is None:
            return
        if (
            api_mode != "chat_completions"
            or not all(key)
            or any(
                not isinstance(row, dict)
                or row.get("type") != "function"
                or not isinstance(function := row.get("function"), dict)
                or not isinstance(name := function.get("name"), str)
                or _assigned_tool(name, store.server) is None
                for row in request.get("tools") or ()
            )
        ):
            store.withdraw_start()
            return
        with self.lock:
            self.prepared[key] = store
            if len(self.prepared) > MAX_ATTEMPTS:
                self.prepared.popitem(last=False)

    def before(self, request_messages, session_id, turn_id, api_request_id, started_at, **fields):
        """Read only the adapter-issued markers present in the effective request."""
        key = session_id, turn_id, api_request_id
        with self.lock:
            store = self.prepared.pop(key, None)
        if store is None:
            return
        try:
            invocation = store.invocation()
        except (OSError, ValueError):
            return
        inputs = _input_markers(store, invocation, request_messages)
        if inputs is None:
            return
        origin = ModelOrigin(invocation, session_id, turn_id, api_request_id, started_at, inputs)
        store.record_start(origin, sys.executable)
        with self.lock:
            self.attempts[key] = _Attempt(store, origin)
            if len(self.attempts) > MAX_ATTEMPTS:
                self.attempts.popitem(last=False)

    def after(self, session_id, turn_id, api_request_id, started_at, assistant_message, **fields):
        """Bind the successful model response and original tool names to its exact attempt."""
        key = session_id, turn_id, api_request_id
        with self.lock:
            attempt = self.attempts.get(key)
            if attempt is None or attempt.origin.started_at != started_at:
                return
            names: dict[str, str] = {}
            duplicates: set[str] = set()
            for call in assistant_message.tool_calls or ():
                if call.id in names:
                    duplicates.add(call.id)
                elif call.id:
                    names[call.id] = call.function.name
            attempt.tools = {identity: name for identity, name in names.items() if identity not in duplicates}
        # The foreground can retire while the provider call completes.
        if attempt.store.invocation() != attempt.origin.invocation:
            return
        content = assistant_message.content or ""
        attempt.store.record_model(attempt.origin, content.strip())

    def tool(self, tool_name, args, next_call, session_id, turn_id, api_request_id, tool_call_id, **fields) -> Any:
        """Attach origin to this session's exact MCP call before the native transport sends it."""
        store = _current_store()
        target = _assigned_tool(tool_name, store.server) if store is not None else None
        if target is None:
            return next_call(args)
        with self.lock:
            attempt = self.attempts.get((session_id, turn_id, api_request_id))
            original = attempt.tools.get(tool_call_id) if attempt is not None else None
        try:
            if (
                attempt is None
                or original != tool_name
                or attempt.store.path != store.path
                or attempt.origin.invocation != store.invocation()
                or PRIVATE_ORIGIN in args
            ):
                return json.dumps({"error": "The tool call has no authorized model origin."})
            reference = store.record_tool(attempt.origin, tool_call_id, target, args)
        except (OSError, ValueError, TypeError):
            return json.dumps({"error": "The tool call evidence is unavailable."})
        return next_call({**args, PRIVATE_ORIGIN: reference})


def register(ctx: Any) -> None:
    """Attach only to the documented Hermes observer and execution boundaries."""
    observer = _Observer()
    ctx.register_middleware("llm_request", observer.prepare)
    ctx.register_hook("pre_api_request", observer.before)
    ctx.register_hook("post_api_request", observer.after)
    ctx.register_middleware("tool_execution", observer.tool)
