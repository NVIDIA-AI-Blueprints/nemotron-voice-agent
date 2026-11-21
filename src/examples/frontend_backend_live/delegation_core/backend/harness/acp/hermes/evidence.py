# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private evidence shared by the Hermes observer and the live example."""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import stat
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeGuard

PRIVATE_ORIGIN = "__voice_agent_native_origin"
DIRECTORY = ".voice-agent-evidence"
INVOCATION = "Voice-Agent invocation "
INPUT = "Voice-Agent input "
MAX_RECEIPT_BYTES = 32_768


def origin_reference(value: object) -> TypeGuard[str]:
    """Recognize a versioned opaque reference without treating it as proof of origin."""
    return isinstance(value, str) and re.fullmatch(r"nva1_[A-Za-z0-9_-]{32}", value) is not None


def nonce(value: object) -> TypeGuard[str]:
    """Recognize an opaque identifier issued by the voice agent."""
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{32}", value) is not None


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: Path) -> bytes:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
        facts = os.fstat(source.fileno())
        if not stat.S_ISREG(facts.st_mode) or facts.st_uid != os.getuid() or facts.st_mode & 0o077:
            raise ValueError("Hermes evidence file is not private")
        data = source.read(MAX_RECEIPT_BYTES + 1)
        if len(data) > MAX_RECEIPT_BYTES:
            raise ValueError("Hermes evidence file exceeds its size bound")
        return data


def _write(path: Path, data: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as target:
        target.write(data)


@dataclass(frozen=True)
class ModelOrigin:
    """A model attempt inside one native turn and one voice-agent invocation."""

    invocation: str
    session: str
    turn: str
    request: str
    started_at: float
    inputs: tuple[str, ...]

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> ModelOrigin:
        """Validate and decode a signed receipt body."""
        if (
            not isinstance(value, dict)
            or not nonce(value.get("invocation"))
            or not all(isinstance(value.get(key), str) and value[key] for key in ("session", "turn", "request"))
            or type(value.get("started_at")) not in (float, int)
            or not isinstance(inputs := value.get("inputs"), list)
            or not all(nonce(item) for item in inputs)
        ):
            raise ValueError("Invalid Hermes model origin")
        return cls(**{**value, "inputs": tuple(inputs)})


@dataclass(frozen=True)
class ToolOrigin:
    """One exact tool dispatch, without the argument values in its receipt."""

    model: ModelOrigin
    call: str


class OriginStore:
    """Sign and verify evidence inside an adapter-owned session folder."""

    def __init__(self, path: Path):
        """Open the same private evidence store on either side of ACP."""
        facts = path.lstat()
        if not stat.S_ISDIR(facts.st_mode) or facts.st_uid != os.getuid() or facts.st_mode & 0o077:
            raise ValueError("Hermes evidence directory is not private")
        self.path = path
        self.key = _read(path / "key")
        self.server = _read(path / "server").decode()
        if len(self.key) != 32 or not self.server:
            raise ValueError("Invalid Hermes evidence scope")

    @classmethod
    def create(cls, workspace: Path, server: str) -> OriginStore:
        """Create a new private store for one session's assigned MCP server."""
        path = workspace / DIRECTORY
        path.mkdir(mode=0o700)
        _write(path / "key", secrets.token_bytes(32))
        _write(path / "server", server.encode())
        return cls(path)

    def begin(self, invocation: str) -> None:
        """Authorize a unique host marker for a prompt in this workspace."""
        if not nonce(invocation):
            raise ValueError("Invalid Hermes invocation marker")
        self._replace(self.path / "invocation", invocation.encode())

    def invocation(self) -> str:
        """Read the active marker registered by the adapter for this workspace."""
        current = _read(self.path / "invocation").decode()
        if not nonce(current):
            raise ValueError("Invalid Hermes invocation marker")
        return current

    def issue_input(self, invocation: str, identity: str) -> None:
        """Register a private marker before supplying it in a correction."""
        if not nonce(identity) or invocation != self.invocation():
            raise ValueError("Hermes input does not belong to this invocation")
        _write(self.path / f"issued-{identity}", invocation.encode())

    def issued(self, invocation: str, identity: str) -> bool:
        """Verify that the adapter issued a marker for this exact invocation."""
        if not nonce(identity):
            return False
        try:
            return _read(self.path / f"issued-{identity}").decode() == invocation
        except (OSError, ValueError):
            return False

    def _replace(self, target: Path, data: bytes) -> None:
        pending = self.path / secrets.token_hex(16)
        try:
            _write(pending, data)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)

    def _sign(self, value: dict[str, Any]) -> bytes:
        return _json([value, hmac.digest(self.key, _json(value), "sha256").hex()])

    def _verify(self, path: Path) -> dict[str, Any]:
        data = json.loads(_read(path))
        if not isinstance(data, list) or len(data) != 2 or not isinstance(data[0], dict):
            raise ValueError("Invalid Hermes evidence body")
        value, signature = data
        standard = hmac.digest(self.key, _json(value), "sha256").hex()
        if not isinstance(signature, str) or not hmac.compare_digest(signature, standard):
            raise ValueError("Invalid Hermes evidence signature")
        return value

    def record_start(self, origin: ModelOrigin, interpreter: str) -> None:
        """Attest that the installed observer saw a qualified request from the active prompt."""
        value = {"kind": "start", "model": asdict(origin), "interpreter": interpreter}
        self._replace(self.path / f"started-{origin.invocation}", self._sign(value))

    def started(self, invocation: str) -> tuple[ModelOrigin, Path] | None:
        """Read the qualified native attempt and its local helper interpreter."""
        if not nonce(invocation):
            return None
        try:
            value = self._verify(self.path / f"started-{invocation}")
            model = ModelOrigin.from_wire(value["model"])
            interpreter = Path(value["interpreter"])
            if (
                value["kind"] == "start"
                and model.invocation == invocation
                and interpreter.is_absolute()
                and interpreter.is_file()
            ):
                return model, interpreter
        except (OSError, ValueError, TypeError, KeyError):
            return None
        return None

    def withdraw_start(self) -> None:
        """Revoke readiness when a later request exposes tools outside this bridge."""
        try:
            invocation = self.invocation()
            (self.path / f"started-{invocation}").unlink(missing_ok=True)
        except (OSError, ValueError):
            return

    def record_model(self, origin: ModelOrigin, content: str = "") -> None:
        """Publish evidence only after a provider response for the matching attempt."""
        value = {
            "kind": "model",
            "model": asdict(origin),
            "text_sha256": sha256(content.encode()).hexdigest(),
            "text_length": len(content),
        }
        proof = self._sign(value)
        for identity in origin.inputs:
            if nonce(identity):
                self._replace(self.path / f"input-{identity}", proof)

    def model_receipt(self, invocation: str, identity: str) -> tuple[ModelOrigin, str, int | None] | None:
        """Read an authenticated response carrying that exact input and invocation."""
        if not nonce(identity):
            return None
        try:
            value = self._verify(self.path / f"input-{identity}")
            model = ModelOrigin.from_wire(value["model"])
            digest = value["text_sha256"]
            length = value.get("text_length")
            if (
                value["kind"] == "model"
                and model.invocation == invocation
                and identity in model.inputs
                and isinstance(digest, str)
                and re.fullmatch(r"[a-f0-9]{64}", digest)
                and (length is None or (type(length) is int and length >= 0))
            ):
                return model, digest, length
        except (OSError, ValueError, TypeError, KeyError):
            return None
        return None

    def record_tool(self, origin: ModelOrigin, call: str, tool: str, arguments: dict[str, Any]) -> str:
        """Write a scoped receipt and return the only value passed through Hermes tool metadata."""
        reference = "nva1_" + secrets.token_urlsafe(24)
        claim = {
            "kind": "tool",
            "model": asdict(origin),
            "call": call,
            "tool": tool,
            "arguments_sha256": sha256(_json(arguments)).hexdigest(),
        }
        _write(self.path / reference, self._sign(claim))
        return reference

    def tool_receipt(self, reference: object, tool: str, arguments: dict[str, Any]) -> ToolOrigin:
        """Verify the reference and exact business arguments before routing a tool."""
        if not origin_reference(reference):
            raise FileNotFoundError("Hermes tool receipt is missing")
        value = self._verify(self.path / reference)
        if (
            value.get("kind") != "tool"
            or value.get("tool") != tool
            or value.get("arguments_sha256") != sha256(_json(arguments)).hexdigest()
            or not isinstance(call := value.get("call"), str)
            or not call
        ):
            raise ValueError("Hermes tool receipt does not match the call")
        try:
            return ToolOrigin(ModelOrigin.from_wire(value["model"]), call)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid Hermes tool origin") from error
