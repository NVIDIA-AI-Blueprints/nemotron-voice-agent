# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""LLM tool handlers for Webex call control and secure keypad input."""

from __future__ import annotations

from collections.abc import Mapping

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties

from examples.webex_byova.input_state import WebexInputState, keypad_prompt, keypad_reprompt

_MAX_REASON_LENGTH = 240
_ROUTES = {"customer_service", "billing", "technical_support"}


def _input_state(params: FunctionCallParams) -> WebexInputState | None:
    resources = params.app_resources
    if not isinstance(resources, Mapping):
        return None
    state = resources.get("webex_input_state")
    return state if isinstance(state, WebexInputState) else None


async def _emit_control(params: FunctionCallParams, action: str, **data: object) -> None:
    payload = {"type": "webex-call-control", "action": action, **data}
    await params.pipeline_worker.queue_frame(RTVIServerMessageFrame(data=payload))


async def handle_transfer_to_human(params: FunctionCallParams) -> None:
    """Request a transfer based on explicit conversational intent."""
    arguments = params.arguments or {}
    reason = str(arguments.get("reason", "caller_requested_human")).strip()[:_MAX_REASON_LENGTH]
    route = str(arguments.get("route", "customer_service")).strip()
    if route not in _ROUTES:
        route = "customer_service"
    input_state = _input_state(params)
    if input_state:
        input_state.cancel_collection()
    await _emit_control(params, "transfer_to_human", reason=reason, metadata={"route": route})
    logger.info("LLM requested Webex transfer route={}", route)
    await params.result_callback(
        {
            "status": "transfer_requested",
            "instruction": "Briefly tell the caller you are connecting them to a human now.",
        }
    )


async def handle_end_call(params: FunctionCallParams) -> None:
    """Request call termination based on explicit goodbye intent."""
    arguments = params.arguments or {}
    reason = str(arguments.get("reason", "caller_requested_end")).strip()[:_MAX_REASON_LENGTH]
    input_state = _input_state(params)
    if input_state:
        input_state.cancel_collection()
    await _emit_control(params, "end_call", reason=reason)
    logger.info("LLM requested Webex call termination")
    await params.result_callback(
        {
            "status": "call_end_requested",
            "instruction": "Say one brief goodbye without asking another question.",
        }
    )


async def handle_request_keypad_input(params: FunctionCallParams) -> None:
    """Request the next authentication value from the telephone keypad."""
    input_state = _input_state(params)
    if input_state is None:
        await params.result_callback({"error": "Webex keypad state is unavailable"})
        return
    field = input_state.expected_field
    if field is None:
        input_state.restore_automatic_tool_choice()
        await params.result_callback({"error": "authentication is already complete"})
        return

    # The caller can speak instead of using the keypad, so a repeat call while
    # the same field is armed only re-speaks the instruction.
    reprompt = input_state.pending_field == field
    if not reprompt:
        if not input_state.begin_collection(field):
            input_state.restore_automatic_tool_choice()
            await params.result_callback({"error": "keypad collection could not be started"})
            return
        try:
            await _emit_control(params, "request_keypad_input", field=field)
        except Exception:
            input_state.rollback_collection()
            raise
    input_state.restore_automatic_tool_choice()
    logger.info("LLM requested Webex secure keypad field={} reprompt={}", field, reprompt)
    await params.result_callback(
        {"status": "keypad_requested", "field": field},
        properties=FunctionCallResultProperties(run_llm=False),
    )
    instruction = keypad_reprompt(field) if reprompt else keypad_prompt(field)
    await params.pipeline_worker.queue_frame(TTSSpeakFrame(instruction))


TOOL_HANDLERS = {
    "transfer_to_human": handle_transfer_to_human,
    "end_call": handle_end_call,
    "request_keypad_input": handle_request_keypad_input,
}
