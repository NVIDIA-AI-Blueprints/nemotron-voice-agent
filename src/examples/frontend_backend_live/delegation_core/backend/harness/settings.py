# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's settings: the ``backend.harness`` block and the default of every key.

The settings name only what differs from these defaults. :class:`HarnessSettings` checks the block when the core is
built and turns each name (``protocol``, ``permission``, and the protocol's own, such as ``transport`` and ``agent``)
into its class, so the protocol's packages load then, never at import. The agent class decides how a newer request
reaches a running worker (``steering.agent_steering``). The block says only how to reach the agent, by ``command`` or
``url``, and which ``agent`` class reads it; the agent's model, key, tools and home are its operator's, and nothing here
writes its configuration.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, PrivateAttr, StrictStr, model_validator

from examples.frontend_backend_live.delegation_core.config import FORMAT_ERRORS, Seconds, Settings, Text, WholeNumber
from examples.frontend_backend_live.delegation_core.tables import pick

from .protocol import PERMISSIONS, PROTOCOLS
from .steering import agent_steering

# Where the sessions' work folders go when work_root is null: outside any git checkout, and outside /tmp, where a stray
# .git folder of another program could make /tmp a project root for an agent's search for project files.
DEFAULT_WORK_ROOT = "~/.cache/delegation-core/harness"


def _address(bind: str) -> tuple[str, str]:
    """``bind``'s host, an IPv6 one without the brackets it is written in (``[::1]:8791``), and its port."""
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
    return url.rstrip("/")


class Bridge(Settings):
    """Where the MCP bridge listens (``bind``, ``host:port``; port 0: any free port), and the URL the runtime is told.

    ``advertise`` is that URL; ``None``: the address it bound.
    """

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
        """The bridge's base URL as the runtime reaches it; an IPv6 address goes in brackets."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return self.advertise or f"http://{host}:{bound_port}"


class HarnessPrompts(Settings):
    """The words the harness adds to what its agent reads, from ``backend.harness.prompts`` in the core's texts.

    Each template's fields are filled by ``harness/render.py``; the calls the bridge answers itself read the core's
    texts (a worker's door through ``TaskContext.texts``, the conductor's through its ``ControlTools``). Every key is
    required, and a template may name only the fields it is given (:data:`FIELDS`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.harness.prompts")

    # The conductor's input for one decide(): the new requests, the work records, the conversation it has not read.
    conductor_input: Text
    arrival: Text
    work: Text
    task: Text
    brief: Text
    none: Text  # an empty part
    # A worker's first prompt and its parts: the task's view of the conversation, as a transcript.
    worker_input: Text
    user: Text
    assistant: Text
    call: Text
    output: Text
    unnamed_call: Text  # the name an output's call reads as when the call is not in the transcript

    @model_validator(mode="after")
    def names_only_its_fields(self) -> HarnessPrompts:
        """Each template names only the fields it is given."""
        for key, text in self:
            try:
                text.format(**dict.fromkeys(FIELDS.get(key, ()), ""))
            except FORMAT_ERRORS as exc:
                raise ValueError(f"backend.harness.prompts.{key} names a field it is not given: {exc}") from None
        return self


# Each template of HarnessPrompts and the fields it is given.
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


def _readable_url(url: str) -> str:
    """A URL :func:`urlsplit` reads, so the description can show it (:func:`_shown`)."""
    with contextlib.suppress(ValueError):
        urlsplit(url)
        return url
    raise ValueError("must be a URL or null")


def _shown(url: str | None) -> str | None:
    """``url`` without what may carry a credential: its user and password, its query and its fragment."""
    if url is None:
        return None
    scheme, netloc, path, _, _ = urlsplit(url)
    return urlunsplit((scheme, netloc.rpartition("@")[2], path, "", ""))


def _absolute(path: Path) -> Path:
    """``path`` with ``~`` expanded, which must then be absolute."""
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError("must be an absolute path (or start with ~)")
    return path


# A list of non-empty strings; an empty one, or null, is none.
Strings = Annotated[tuple[Text, ...], BeforeValidator(lambda value: value or ())]


class HarnessSettings(Settings):
    """The harness's settings, every name turned into what it names when they are validated.

    ``protocol``, ``agent`` and ``permission`` name classes: the fields ``protocol_name``, ``agent_name`` and
    ``permission_name`` hold the names, and :attr:`protocol`, :attr:`agent` and :attr:`permission` are what they name.
    ``turns_per_process`` is the agent class's own limit when the block sets none.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.harness")

    protocol_name: Text = Field("acp", alias="protocol")  # a name in harness/protocol.py PROTOCOLS
    agent_name: Text = Field("acp", alias="agent")  # acp (a plain ACP agent) | hermes: the agent class
    transport: Text = "stdio"  # stdio: start command | websocket: connect to url, which the operator serves
    command: Strings = ()  # stdio: the agent to start, once per runtime process
    url: Annotated[Text, AfterValidator(_readable_url)] | None = None  # websocket only: where the agent is served
    max_running: WholeNumber = 3  # tasks running at once; 1: no conductor process, the steer rule places requests
    # How many turns one runtime process runs at once; None: the agent class's own limit (Hermes runs 4).
    turns_per_process: WholeNumber | None = None
    runtime_label: Text = "runtime"  # the server label of the calls the agent runs itself (its own tools)
    permission_name: Text = Field("reject", alias="permission")  # the answer to the agent's permission requests
    start_timeout_seconds: Seconds = 60.0
    ready_timeout_seconds: Seconds = 15.0
    turn_timeout_seconds: Seconds = 300.0
    conductor_timeout_seconds: Seconds = 20.0
    stop_timeout_seconds: Seconds = 10.0  # how long a worker's stop, or the conductor's cancel, waits for its turn
    stop_grace_seconds: Seconds = 5.0
    tool_timeout_seconds: Seconds = 60.0
    bridge: Bridge = Field(default_factory=lambda: Bridge(advertise="http://127.0.0.1:8791"))
    # Where the sessions' work folders go; null: DEFAULT_WORK_ROOT. A served agent (websocket) must see it at the same
    # path.
    work_root: Annotated[Path, BeforeValidator(lambda value: value or DEFAULT_WORK_ROOT), AfterValidator(_absolute)] = (
        Field(DEFAULT_WORK_ROOT, validate_default=True)
    )
    env: dict[str, StrictStr] = {}  # stdio: more variables for a started agent
    pass_env: Strings = ("PATH", "LANG", "LC_ALL", "TZ")  # stdio: all a started agent gets of this server's environment
    agents: dict[str, Any] = {}  # the values only one agent class reads, by its name; the class checks its block
    prompts: HarnessPrompts  # the core's texts at backend.harness.prompts

    _protocol: Any = PrivateAttr(None)
    _permission: Any = PrivateAttr(None)
    _steering: Any = PrivateAttr(None)

    @model_validator(mode="after")
    def resolved(self) -> HarnessSettings:
        """Turn each name into its class; the agent's own turn limit holds when the block sets none.

        Raises:
            ValueError: A name no table has, an agent or transport the protocol refuses, or ``max_running`` over the
                turns a process runs.
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
    def protocol(self) -> Any:
        """The HarnessProtocol (``harness/protocol.py``), with the agent behind it."""
        return self._protocol

    @property
    def agent(self) -> Any:
        """The agent behind the protocol (``harness/protocol.py``: ``Agent``)."""
        return self._protocol.agent

    @property
    def permission(self) -> Any:
        """The PermissionAnswer."""
        return self._permission

    @property
    def steering(self) -> Any:
        """The SteerStrategy the agent class takes (``steering.agent_steering``)."""
        return self._steering

    def describe(self) -> dict:
        """The settings as plain JSON, by name: no environment values (they may name a key's place).

        The URL is shown with no user, password, query or fragment: they may carry the served agent's credential.
        """
        return {
            "protocol": self.protocol_name,
            "agent": self.agent_name,
            "transport": self.transport,
            "permission": self.permission_name,
            "command": list(self.command),
            "url": _shown(self.url),
            "max_running": self.max_running,
            "turns_per_process": self.turns_per_process,
        }
