# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's settings: the ``backend.harness`` block and the default of every key.

A profile names only what differs from these defaults. :meth:`HarnessSettings.parse` checks the block when the core is
built and turns each name (``protocol``, ``steer``, ``permission``, and the protocol's own, such as ``transport`` and
``agent``) into its class, so the protocol's packages load then, never at import. The block says only how to reach the
agent, by ``command`` or ``url``, and which ``agent`` class reads it; the agent's model, key, tools and home are its
operator's, and nothing here writes its configuration.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from delegation_core.tables import pick

from .protocol import PERMISSIONS, PROTOCOLS, STEERING


@dataclass(frozen=True)
class Bridge:
    """Where the MCP bridge listens, and the URL the runtime is told (``None``: the address it bound)."""

    host: str
    port: int  # 0: any free port
    advertise: str | None

    @classmethod
    def parse(cls, raw: Any) -> Bridge:
        """Validate ``backend.harness.bridge``: ``bind`` as ``host:port``, ``advertise`` as a URL or null.

        Raises:
            ValueError: A key it does not have, or a bind that is not ``host:port`` with a port number.
        """
        if not isinstance(raw, Mapping) or set(raw) - {"bind", "advertise"}:
            raise ValueError("backend.harness.bridge takes only bind and advertise")
        host, _, port = str(raw.get("bind", "127.0.0.1:8791")).rpartition(":")
        if not host or not port.isdigit():
            raise ValueError("backend.harness.bridge.bind must be host:port")
        advertise = raw.get("advertise")
        if advertise is not None and (
            not isinstance(advertise, str) or not advertise.startswith(("http://", "https://"))
        ):
            raise ValueError("backend.harness.bridge.advertise must be an http(s) URL or null")
        return cls(host, int(port), advertise.rstrip("/") if advertise else None)

    def url(self, bound_port: int) -> str:
        """The bridge's base URL as the runtime reaches it."""
        return self.advertise or f"http://{self.host}:{bound_port}"


@dataclass(frozen=True)
class HarnessPrompts:
    """The words the harness adds to what its agent reads (model-facing text lives in YAML).

    Each template's fields are filled by ``harness/render.py``; a profile may reword any of them.
    """

    # The conductor's input for one decide(): the new requests, the work records, the conversation it has not read.
    conductor_input: str = (
        "New requests to place:\n{arrivals}\n\n"
        "Work records:\n{works}\n\n"
        "Conversation since your last turn:\n{conversation}"
    )
    arrival: str = "[{delegation}] {text}"
    work: str = "[{work}] {state}, carried by {carrier}; requests: {requests}\n{tasks}"
    task: str = "  task {task}: {state} for {seconds} s{brief}; steps: {steps}; newest note: {note}"
    brief: str = ', brief "{brief}"'
    none: str = "none"
    # A worker's first prompt and its parts: the task's view of the conversation, as a transcript.
    worker_input: str = "The conversation and your task:\n\n{transcript}"
    user: str = "{text}"
    assistant: str = "Assistant: {text}"
    call: str = "Assistant called {name} with {arguments}"
    output: str = "{name} returned: {output}"
    # The plain results the bridge gives the agent for calls it does not run or cannot report.
    not_run: str = "Not run: the caller said something new."
    stopped: str = "Stopped: the caller no longer wants this."
    unclear: str = "Outcome unknown: check whether it happened before retrying."
    # The conductor's control call that comes when no request waits to be placed.
    no_placement: str = "Not applied: no request waits to be placed now."

    # Each template and the fields it is given.
    FIELDS = {
        "conductor_input": ("arrivals", "works", "conversation"),
        "arrival": ("delegation", "text"),
        "work": ("work", "state", "carrier", "requests", "tasks"),
        "task": ("task", "state", "seconds", "brief", "steps", "note"),
        "brief": ("brief",),
        "worker_input": ("transcript",),
        "user": ("text",),
        "assistant": ("text",),
        "call": ("name", "arguments"),
        "output": ("name", "output"),
    }

    @classmethod
    def parse(cls, raw: Any) -> HarnessPrompts:
        """Validate ``backend.harness.prompts``: known keys only, each non-empty text, each template's fields known.

        Raises:
            ValueError: An unknown key, a value that is not text, or a template naming a field it is not given.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("backend.harness.prompts must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"backend.harness.prompts has unknown keys: {', '.join(sorted(map(str, unknown)))}")
        for key, value in raw.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"backend.harness.prompts.{key} must be non-empty text")
            try:
                value.format(**dict.fromkeys(cls.FIELDS.get(key, ()), ""))
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(f"backend.harness.prompts.{key} names a field it is not given: {exc}") from None
        return cls(**raw)


# Every key of backend.harness and its default (core.yaml names the agent and how to reach it: command or url).
DEFAULTS: Mapping[str, Any] = {
    "protocol": "acp",
    "agent": "acp",
    "transport": "stdio",
    "command": [],
    "url": None,
    "max_running": 3,
    "turns_per_process": None,
    "steer": "next_prompt",
    "runtime_label": "runtime",
    "permission": "reject",
    "start_timeout_seconds": 60,
    "ready_timeout_seconds": 15,
    "turn_timeout_seconds": 300,
    "conductor_timeout_seconds": 20,
    "stop_timeout_seconds": 10,
    "stop_grace_seconds": 5,
    "tool_timeout_seconds": 60,
    "bridge": {"bind": "127.0.0.1:8791", "advertise": "http://127.0.0.1:8791"},
    "work_root": None,
    "env": {},
    "pass_env": ["PATH", "LANG", "LC_ALL", "TZ"],
    "agents": {},
    "prompts": {},
}
SECONDS = (
    "start_timeout_seconds",
    "ready_timeout_seconds",
    "turn_timeout_seconds",
    "conductor_timeout_seconds",
    "stop_timeout_seconds",
    "stop_grace_seconds",
    "tool_timeout_seconds",
)
# Where the sessions' work folders go when work_root is null: outside any git checkout, and outside /tmp, where a stray
# .git folder of another program could make /tmp a project root for an agent's search for project files.
DEFAULT_WORK_ROOT = "~/.cache/voiceclaw/harness"


def _strings(where: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"backend.harness.{where} must be a list of non-empty strings")
    return tuple(value)


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"backend.harness.{where} must be a whole number of at least 1")
    return value


def _url(value: Any) -> str | None:
    """A URL of the block, or ``None``: one :func:`urlsplit` reads, so the description can show it (:func:`_shown`)."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        with contextlib.suppress(ValueError):
            urlsplit(value)
            return value
    raise ValueError("backend.harness.url must be a URL or null")


def _shown(url: str | None) -> str | None:
    """``url`` without what may carry a credential: its user and password, its query and its fragment."""
    if url is None:
        return None
    scheme, netloc, path, _, _ = urlsplit(url)
    return urlunsplit((scheme, netloc.rpartition("@")[2], path, "", ""))


def _path(value: Any, where: str) -> Path:
    """An absolute path of the block, ``~`` expanded."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"backend.harness.{where} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"backend.harness.{where} must be an absolute path (or start with ~)")
    return path


@dataclass(frozen=True)
class HarnessSettings:
    """The harness's settings, every name already turned into what it names (made by :meth:`parse`)."""

    names: Mapping[str, str]  # protocol, agent, transport, steer, permission: the names, for describe()
    protocol: Any  # the HarnessProtocol (harness/protocol.py), with the agent behind it
    steering: Any  # a SteerStrategy
    permission: Any  # a PermissionAnswer
    command: tuple[str, ...]
    url: str | None
    max_running: int
    turns_per_process: int | None  # None: no limit (the agent knows none, and none is set)
    runtime_label: str  # the server label of the calls the agent runs itself (its own tools, not the bridge's)
    start_timeout_seconds: float
    ready_timeout_seconds: float
    turn_timeout_seconds: float
    conductor_timeout_seconds: float
    stop_timeout_seconds: float
    stop_grace_seconds: float
    tool_timeout_seconds: float
    bridge: Bridge
    work_root: Path
    env: Mapping[str, str]
    pass_env: tuple[str, ...]
    prompts: HarnessPrompts

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> HarnessSettings:
        """Validate the ``backend.harness`` block over the defaults, and turn its names into their classes.

        Raises:
            ValueError: An unknown key, a value of the wrong kind, a name no table has, an agent or transport the
                protocol refuses, a steering the agent cannot take, or ``max_running`` over ``turns_per_process``.
        """
        if not isinstance(block, Mapping):
            raise ValueError("backend.harness must be a mapping")
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"backend.harness has unknown keys: {', '.join(sorted(map(str, unknown)))}")
        raw = {**DEFAULTS, **block}
        names = {key: raw[key] for key in ("protocol", "agent", "transport", "steer", "permission")}
        command = _strings("command", raw["command"]) if raw["command"] else ()
        url = _url(raw["url"])
        protocol = pick(PROTOCOLS, names["protocol"], "harness protocol")().from_settings({**raw, "command": command})
        agent = protocol.agent
        steering = pick(STEERING, names["steer"], "harness steering")().from_settings(raw)
        permission = pick(PERMISSIONS, names["permission"], "permission answer")()()
        values: dict[str, Any] = {"max_running": _count(raw["max_running"], "max_running")}
        set_turns = raw["turns_per_process"]
        turns = agent.turns_per_process if set_turns is None else _count(set_turns, "turns_per_process")
        for key in SECONDS:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"backend.harness.{key} must be a positive number of seconds")
            values[key] = float(value)
        if not isinstance(raw["runtime_label"], str) or not raw["runtime_label"].strip():
            raise ValueError("backend.harness.runtime_label must be non-empty text")
        if steering.native and not agent.native_steer:
            raise ValueError(
                f"backend.harness.steer {names['steer']!r} needs an agent that takes a steer in the middle of a turn; "
                f"agent {names['agent']!r} takes none"
            )
        if turns is not None and values["max_running"] > turns:
            raise ValueError(
                f"backend.harness.max_running must not be over turns_per_process: a process of agent "
                f"{names['agent']!r} runs at most {turns} turns at once, and a turn waiting for the app holds its place"
            )
        env = raw["env"]
        if not isinstance(env, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError("backend.harness.env must map variable names to text")
        return cls(
            names=names,
            protocol=protocol,
            steering=steering,
            permission=permission,
            command=command,
            url=url,
            turns_per_process=turns,
            runtime_label=raw["runtime_label"],
            bridge=Bridge.parse(raw["bridge"]),
            work_root=_path(raw["work_root"] or DEFAULT_WORK_ROOT, "work_root"),
            env=dict(env),
            pass_env=_strings("pass_env", raw["pass_env"]) if raw["pass_env"] else (),
            prompts=HarnessPrompts.parse(raw["prompts"]),
            **values,
        )

    @property
    def agent(self) -> Any:
        """The agent behind the protocol (``harness/protocol.py``: ``Agent``)."""
        return self.protocol.agent

    def describe(self) -> dict:
        """The settings as plain JSON, by name: no environment values (they may name a key's place).

        The URL is shown with no user, password, query or fragment: they may carry the served agent's credential.
        """
        return {
            **self.names,
            "command": list(self.command),
            "url": _shown(self.url),
            "max_running": self.max_running,
            "turns_per_process": self.turns_per_process,
        }
