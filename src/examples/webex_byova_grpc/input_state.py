# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Track secure keypad input for a Webex BYOVA pipeline session."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import Frame, InputTransportMessageFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

VERIFICATION_CONFIRMATION = "Thank you, I have verified your details. How can I help you today?"


@dataclass(frozen=True)
class _AuthenticationField:
    """Describe one keypad field of the authentication sequence."""

    name: str
    context_label: str
    lead_in: str
    request: str


_FIELDS: tuple[_AuthenticationField, ...] = (
    _AuthenticationField(
        name="phone_number",
        context_label="phone number",
        lead_in="Hello, I'm Nova from Northstar Services. To authenticate,",
        request="enter your 10 digit phone number on the keypad",
    ),
    _AuthenticationField(
        name="date_of_birth",
        context_label="date of birth in DDMMYYYY format",
        lead_in="Phone number received,",
        request="enter your 8 digit date of birth on the keypad, in day month year format",
    ),
)
_FIELDS_BY_NAME = {field.name: field for field in _FIELDS}


def keypad_prompt(field_name: str) -> str:
    """Return the sentence that first asks the caller for one field."""
    field = _FIELDS_BY_NAME[field_name]
    return f"{field.lead_in} please {field.request}."


def keypad_reprompt(field_name: str, reason: str = "") -> str:
    """Return the sentence that asks the caller to enter one field again."""
    field = _FIELDS_BY_NAME[field_name]
    preface = f"I could not accept that entry because {reason}. " if reason else ""
    return f"{preface}Please {field.request}."


class AuthenticationPhase(Enum):
    """Represent how far a session has progressed through the keypad fields."""

    NEED_TOOL = "need_tool"
    COLLECTING = "collecting"
    COMPLETE = "complete"


class WebexInputState:
    """Sequence the keypad fields a session collects before offering support."""

    def __init__(
        self,
        context: LLMContext,
        *,
        all_tools: ToolsSchema | None = None,
        keypad_tools: ToolsSchema | None = None,
    ) -> None:
        """Create session-local input state backed by an LLM context."""
        self._context = context
        self._all_tools = all_tools
        self._keypad_tools = keypad_tools
        self._index = 0
        self._phase = AuthenticationPhase.NEED_TOOL

    @property
    def phase(self) -> AuthenticationPhase:
        """Return the current authentication phase."""
        return self._phase

    @property
    def expected_field(self) -> str | None:
        """Return the field this phase still requires from the caller."""
        if self._phase is AuthenticationPhase.COMPLETE:
            return None
        return _FIELDS[self._index].name

    @property
    def pending_field(self) -> str | None:
        """Return the field Cisco is currently collecting."""
        return self.expected_field if self._phase is AuthenticationPhase.COLLECTING else None

    def begin_collection(self, field_name: str) -> bool:
        """Arm the collection that the current phase requires."""
        if field_name != self.expected_field or self._phase is not AuthenticationPhase.NEED_TOOL:
            return False
        self._phase = AuthenticationPhase.COLLECTING
        logger.info("Webex secure keypad collection started for field={}", field_name)
        return True

    def cancel_collection(self) -> None:
        """Stop expecting keypad input once a terminal action is authorized."""
        self._phase = AuthenticationPhase.COMPLETE

    def rollback_collection(self) -> None:
        """Require the keypad tool again when arming the adapter fails."""
        if self._phase is AuthenticationPhase.COLLECTING:
            self._phase = AuthenticationPhase.NEED_TOOL

    def force_expected_tool(self) -> bool:
        """Expose only the keypad tool while a field is still required."""
        if self.expected_field is None:
            return False
        if self._keypad_tools is not None:
            self._context.set_tools(self._keypad_tools)
        self._context.set_tool_choice("auto")
        return True

    def restore_automatic_tool_choice(self) -> None:
        """Allow normal LLM tool selection again."""
        if self._all_tools is not None:
            self._context.set_tools(self._all_tools)
        self._context.set_tool_choice("auto")

    def complete_collection(self, field_name: str, value: str) -> bool:
        """Add one collected value to context and advance the sequence."""
        if field_name != self.pending_field:
            return False
        field = _FIELDS_BY_NAME[field_name]
        self._context.add_message(
            {
                "role": "user",
                "content": (
                    f"Secure keypad input for {field.context_label}: {value}. "
                    "Treat this value as sensitive and never read it aloud."
                ),
            }
        )
        logger.info("Webex secure keypad collection completed for field={}", field_name)

        self._index += 1
        if self._index < len(_FIELDS):
            self._phase = AuthenticationPhase.NEED_TOOL
            return True

        self._phase = AuthenticationPhase.COMPLETE
        # The demo checks entry format only, so verification always succeeds
        # once every field is collected.
        self._context.add_message(
            {
                "role": "user",
                "content": (
                    "The requested details are collected and verification succeeded. You have "
                    f'already told the caller: "{VERIFICATION_CONFIRMATION}" Never repeat that '
                    "confirmation or any keypad instruction, and never ask for these details again."
                ),
            }
        )
        self.restore_automatic_tool_choice()
        return True

    def collection_failed(self, field_name: str, reason: str) -> bool:
        """Report that Cisco rejected the entry for the active collection."""
        if field_name != self.pending_field:
            return False
        logger.info(
            "Webex secure keypad collection rejected for field={} reason={}",
            field_name,
            reason,
        )
        return True


class WebexInputMessageProcessor(FrameProcessor):
    """Consume adapter control messages on the transport's downstream path."""

    def __init__(
        self,
        state: WebexInputState,
        queue_llm_run: Callable[[], Awaitable[None]],
        speak: Callable[[str], Awaitable[None]],
    ) -> None:
        """Bind session input state to the pipeline's LLM and speech triggers."""
        super().__init__(name="webex-input-message-processor")
        self._state = state
        self._queue_llm_run = queue_llm_run
        self._speak = speak

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Handle typed DTMF messages and pass all other frames downstream."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InputTransportMessageFrame):
            message = frame.message if isinstance(frame.message, dict) else {}
            if await self._handle_adapter_message(str(message.get("type", "")), message.get("data")):
                return
        await self.push_frame(frame, direction)

    async def _handle_adapter_message(self, message_type: str, data: object) -> bool:
        """Apply one adapter keypad message and report whether it was consumed."""
        payload = data if isinstance(data, dict) else {}
        field_name = str(payload.get("field", ""))
        if message_type == "webex-dtmf":
            if not self._state.complete_collection(field_name, str(payload.get("value", ""))):
                logger.warning("Ignored unexpected Webex DTMF completion")
            elif self._state.force_expected_tool():
                # Another field is required, so let the LLM request it.
                await self._queue_llm_run()
            else:
                await self._speak(VERIFICATION_CONFIRMATION)
            return True
        if message_type == "webex-dtmf-error":
            reason = str(payload.get("reason", ""))
            if self._state.collection_failed(field_name, reason):
                # The adapter keeps the same field armed, so the caller can
                # simply enter the value again.
                await self._speak(keypad_reprompt(field_name, reason))
            return True
        return False
