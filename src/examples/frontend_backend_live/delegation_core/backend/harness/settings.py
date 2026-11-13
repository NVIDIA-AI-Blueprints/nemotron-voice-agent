# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's settings: the ``backend.harness`` block and the default of every key."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, PrivateAttr, StrictStr, model_validator

from examples.frontend_backend_live.delegation_core.config import FORMAT_ERRORS, Seconds, Settings, Text, WholeNumber
from examples.frontend_backend_live.delegation_core.tables import pick

from .protocol import PERMISSIONS, PROTOCOLS, Agent, HarnessProtocol, PermissionAnswer, SteerStrategy
from .steering import agent_steering

# The default lies outside any git checkout, and outside /tmp, where another program's stray .git could make /tmp the
# project root that an agent searches for its project files.
DEFAULT_WORK_ROOT = "~/.cache/delegation-core/harness"


def _address(bind: str) -> tuple[str, str]:
    """Split ``bind`` into its host, without an IPv6 address's brackets (``[::1]:8791``), and its port."""
    host, _, port = bind.rpartition(":")
    return host.removeprefix("[").removesuffix("]"), port


def _host_and_port(bind: str) -> str:
    host, port = _address(bind)
    if not host or not port.isdigit():
        raise ValueError("must be host:port")
    return bind


def _http_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError("must be an http(s) URL or null")
    return _readable_url(url).rstrip("/")


class Bridge(Settings):
    """The ``backend.harness.bridge`` block: where the MCP bridge binds, and the URL the agent is told."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.harness.bridge")

    bind: Annotated[StrictStr, AfterValidator(_host_and_port)] = "127.0.0.1:8791"
    advertise: Annotated[StrictStr, AfterValidator(_http_url)] | None = None

    @property
    def host(self) -> str:
        """The address it binds; an IPv6 one without its brackets."""
        return _address(self.bind)[0]

    @property
    def port(self) -> int:
        """The port it binds (0: any free port)."""
        return int(_address(self.bind)[1])

    def url(self, bound_port: int) -> str:
        """Return the base URL the agent reaches the bridge at: ``advertise``, else the bound address."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return self.advertise or f"http://{host}:{bound_port}"


class HarnessPrompts(Settings):
    """The ``backend.harness.prompts`` templates: the words the harness adds to what its agent reads."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.harness.prompts")

    conductor_input: Text
    arrival: Text
    work: Text
    task: Text
    brief: Text
    question: Text
    after_call: Text
    answer: Text
    none: Text
    worker_input: Text
    user: Text
    assistant: Text
    call: Text
    output: Text
    unnamed_call: Text

    @model_validator(mode="after")
    def names_only_its_fields(self) -> HarnessPrompts:
        """Refuse a template that names a field it is not given (:data:`FIELDS`)."""
        for key, text in self:
            try:
                text.format(**dict.fromkeys(FIELDS.get(key, ()), ""))
            except FORMAT_ERRORS as exc:
                raise ValueError(f"backend.harness.prompts.{key} names a field it is not given: {exc}") from None
        return self


FIELDS = {
    "conductor_input": ("arrivals", "works", "conversation"),
    "arrival": ("delegation", "text"),
    "work": ("work", "state", "after_call", "carrier", "requests", "tasks"),
    "task": ("task", "state", "seconds", "brief", "question", "steps", "note", "answer"),
    "brief": ("brief",),
    "question": ("question",),
    "answer": ("answer",),
    "worker_input": ("transcript",),
    "user": ("text",),
    "assistant": ("text",),
    "call": ("name", "arguments"),
    "output": ("name", "output"),
}


def _readable_url(url: str) -> str:
    """Refuse a URL that :func:`urlsplit` cannot read, since the description shows it (:func:`_shown`)."""
    with contextlib.suppress(ValueError):
        urlsplit(url)
        return url
    raise ValueError("must be a URL or null")


def _shown(url: str | None) -> str | None:
    """Return ``url`` without what may carry a credential: its user, password, query and fragment."""
    if url is None:
        return None
    scheme, netloc, path, _, _ = urlsplit(url)
    return urlunsplit((scheme, netloc.rpartition("@")[2], path, "", ""))


def _absolute(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError("must be an absolute path (or start with ~)")
    return path


# A list of non-empty strings; an empty one, or null, is none: pass_env: null passes no variable.
Strings = Annotated[tuple[Text, ...], BeforeValidator(lambda value: value or ())]


class HarnessSettings(Settings):
    """The ``backend.harness`` block, every name turned into what it names when it is validated."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.harness")

    protocol_name: Text = Field("acp", alias="protocol")
    agent_name: Text = Field("acp", alias="agent")
    transport: Text = "stdio"
    command: Strings = ()
    url: Annotated[Text, AfterValidator(_readable_url)] | None = None
    max_running: WholeNumber = 3
    # The turns one runtime process runs at once; null: the agent class's own limit (Hermes runs 4).
    turns_per_process: WholeNumber | None = None
    # The server label of the calls the agent runs with its own tools.
    runtime_label: Text = "runtime"
    permission_name: Text = Field("reject", alias="permission")
    start_timeout_seconds: Seconds = 60.0
    ready_timeout_seconds: Seconds = 15.0
    # How long one turn may run, not counting a worker's waits for the app; it bounds a conductor's turn too.
    turn_timeout_seconds: Seconds = 300.0
    # How long a stop waits for the turn to end, and how long a cancel sent in the background may take.
    stop_timeout_seconds: Seconds = 10.0
    stop_grace_seconds: Seconds = 5.0
    # One call of a tool source or a work tool through the bridge; past it, its outcome is unclear.
    tool_timeout_seconds: Seconds = 60.0
    bridge: Bridge = Field(default_factory=lambda: Bridge(advertise="http://127.0.0.1:8791"))
    # Where the work folders of the runtime processes go; null: DEFAULT_WORK_ROOT.
    work_root: Annotated[Path, BeforeValidator(lambda value: value or DEFAULT_WORK_ROOT), AfterValidator(_absolute)] = (
        Field(DEFAULT_WORK_ROOT, validate_default=True)
    )
    env: dict[str, StrictStr] = {}
    pass_env: Strings = ("PATH", "LANG", "LC_ALL", "TZ")
    agents: dict[str, Any] = {}
    prompts: HarnessPrompts

    _protocol: Any = PrivateAttr(None)
    _permission: Any = PrivateAttr(None)
    _steering: Any = PrivateAttr(None)

    @model_validator(mode="after")
    def resolved(self) -> HarnessSettings:
        """Turn each name into its class, and take the agent class's turn limit when the block sets none.

        Raises:
            ValueError: A name no table has, a block the protocol refuses, or ``max_running`` over a process's turns.
        """
        protocol = pick(PROTOCOLS, self.protocol_name, "harness protocol")().from_settings(
            self.model_dump(by_alias=True)
        )
        permission = pick(PERMISSIONS, self.permission_name, "permission answer")()()
        turns = protocol.agent.turns_per_process if self.turns_per_process is None else self.turns_per_process
        if turns is not None and self.max_running > turns:
            raise ValueError(
                f"backend.harness.max_running must not be over turns_per_process: a process of agent "
                f"{self.agent_name!r} runs at most {turns} turns at once, and a turn waiting for the app holds its "
                "place"
            )
        settings = self.model_copy(update={"turns_per_process": turns})
        settings._protocol, settings._permission = protocol, permission
        settings._steering = agent_steering(protocol.agent)
        return settings

    @property
    def protocol(self) -> HarnessProtocol:
        """The agent's protocol, with the agent behind it."""
        return self._protocol

    @property
    def agent(self) -> Agent:
        """The agent behind the protocol."""
        return self._protocol.agent

    @property
    def permission(self) -> PermissionAnswer:
        """The answer to the agent's permission requests."""
        return self._permission

    @property
    def steering(self) -> SteerStrategy:
        """How a newer request reaches a running worker, as the agent class takes it."""
        return self._steering

    def describe(self) -> dict:
        """Return the settings as plain JSON, without environment values or what in a URL may carry a credential."""
        return {
            "protocol": self.protocol_name,
            "agent": self.agent_name,
            "transport": self.transport,
            "permission": self.permission_name,
            "command": list(self.command),
            "url": _shown(self.url),
            "max_running": self.max_running,
            "turns_per_process": self.turns_per_process,
            "bridge": _shown(self.bridge.advertise),
        }
