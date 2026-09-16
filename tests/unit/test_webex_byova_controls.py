# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for Webex BYOVA LLM tools and secure DTMF state."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from pipecat.frames.frames import InputTransportMessageFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from examples.webex_byova.input_state import (
    VERIFICATION_CONFIRMATION,
    AuthenticationPhase,
    WebexInputMessageProcessor,
    WebexInputState,
)
from examples.webex_byova.tool_handlers import (
    handle_end_call,
    handle_request_keypad_input,
    handle_transfer_to_human,
)
from examples.webex_byova.tools import build_tools_schema

REPO_ROOT = Path(__file__).parents[2]


def test_input_state_adds_completed_dtmf_to_context() -> None:
    """Add completed secure keypad input to the LLM context."""
    context = LLMContext([{"role": "system", "content": "customer care"}])
    input_state = WebexInputState(context)
    assert input_state.begin_collection("phone_number")
    assert input_state.complete_collection("phone_number", "4155550123")
    assert "4155550123" in context.messages[-1]["content"]


def test_any_well_formed_entries_verify_and_open_the_conversation() -> None:
    """Accept any format-valid demo entries instead of matching fixed values."""
    context = LLMContext([])
    input_state = WebexInputState(context)

    input_state.begin_collection("phone_number")
    input_state.complete_collection("phone_number", "9876543210")
    input_state.begin_collection("date_of_birth")

    assert input_state.complete_collection("date_of_birth", "31121985")
    assert input_state.phase is AuthenticationPhase.COMPLETE
    assert "verification succeeded" in context.messages[-1]["content"]
    assert "never ask for these details again" in context.messages[-1]["content"]


def _tool_params(arguments: dict, input_state: WebexInputState) -> SimpleNamespace:
    return SimpleNamespace(
        arguments=arguments,
        app_resources={"webex_input_state": input_state},
        pipeline_worker=SimpleNamespace(queue_frame=AsyncMock()),
        result_callback=AsyncMock(),
    )


def test_keypad_tool_arms_gate_and_emits_typed_control() -> None:
    """Use an RTVI control event instead of transcript matching."""
    input_state = WebexInputState(LLMContext([]))
    input_state.begin_collection("phone_number")
    input_state.complete_collection("phone_number", "4155550123")
    params = _tool_params({}, input_state)

    asyncio.run(handle_request_keypad_input(params))

    assert input_state.pending_field == "date_of_birth"
    control_frame = params.pipeline_worker.queue_frame.await_args_list[0].args[0]
    assert control_frame.data == {
        "type": "webex-call-control",
        "action": "request_keypad_input",
        "field": "date_of_birth",
    }
    speech_frame = params.pipeline_worker.queue_frame.await_args_list[1].args[0]
    assert "day month year" in speech_frame.text
    assert "hash" not in speech_frame.text
    params.result_callback.assert_awaited_once()
    assert params.result_callback.await_args.kwargs["properties"].run_llm is False


@pytest.mark.parametrize(
    ("handler", "arguments", "expected_action"),
    [
        (
            handle_transfer_to_human,
            {"reason": "caller explicitly requested help", "route": "billing"},
            "transfer_to_human",
        ),
        (handle_end_call, {"reason": "caller said goodbye"}, "end_call"),
    ],
)
def test_terminal_tools_emit_typed_controls(handler, arguments, expected_action) -> None:
    """Emit explicit terminal actions and release any keypad gate."""
    input_state = WebexInputState(LLMContext([]))
    input_state.begin_collection("phone_number")
    params = _tool_params(arguments, input_state)

    asyncio.run(handler(params))

    assert input_state.pending_field is None
    frame = params.pipeline_worker.queue_frame.await_args.args[0]
    assert frame.data["action"] == expected_action


def test_webex_catalog_exposes_only_call_control_tools() -> None:
    """Keep the customer-care tool surface focused and deterministic."""
    schema, names = build_tools_schema(
        REPO_ROOT / "src/examples/webex_byova/pipeline.py",
        ["transfer_to_human", "end_call", "request_keypad_input"],
    )
    assert schema is not None
    assert names == ["transfer_to_human", "end_call", "request_keypad_input"]

    with (REPO_ROOT / "src/examples/webex_byova/prompts.yaml").open(encoding="utf-8") as stream:
        prompt = yaml.safe_load(stream)["customer_care"]
    assert prompt["tools_available"] == names


def test_customer_care_session_starts_with_phone_dtmf() -> None:
    """Force the phone tool rather than arming DTMF outside the LLM."""
    source = (REPO_ROOT / "src/examples/webex_byova/pipeline.py").read_text(encoding="utf-8")

    assert 'input_state.begin_collection("phone_number")' not in source
    assert "input_state.force_expected_tool()" in source
    assert "Initial Webex authentication tool call queued" in source


def test_authentication_state_forces_each_keypad_tool_in_order() -> None:
    """Require phone and then DOB before returning to automatic tool choice."""
    all_tools, _ = build_tools_schema(
        REPO_ROOT / "src/examples/webex_byova/pipeline.py",
        ["transfer_to_human", "end_call", "request_keypad_input"],
    )
    keypad_tools, _ = build_tools_schema(
        REPO_ROOT / "src/examples/webex_byova/pipeline.py",
        ["request_keypad_input"],
    )
    context = LLMContext([], tools=all_tools, tool_choice="auto")
    state = WebexInputState(context, all_tools=all_tools, keypad_tools=keypad_tools)

    assert state.force_expected_tool()
    assert context.tools is keypad_tools
    assert state.begin_collection("phone_number")
    state.restore_automatic_tool_choice()
    assert state.complete_collection("phone_number", "4155550123")
    assert state.expected_field == "date_of_birth"
    assert state.phase is AuthenticationPhase.NEED_TOOL

    assert state.force_expected_tool()
    assert context.tools is keypad_tools
    assert state.begin_collection("date_of_birth")
    state.restore_automatic_tool_choice()
    assert state.complete_collection("date_of_birth", "01011990")
    assert state.expected_field is None
    assert state.phase is AuthenticationPhase.COMPLETE
    assert context.tools is all_tools
    assert context.tool_choice == "auto"


def test_transport_dtmf_message_updates_context_and_runs_llm() -> None:
    """Consume adapter DTMF where transport input actually emits messages."""
    context = LLMContext([])
    state = WebexInputState(context)
    assert state.begin_collection("phone_number")
    queue_llm_run = AsyncMock()
    processor = WebexInputMessageProcessor(state, queue_llm_run, AsyncMock())
    frame = InputTransportMessageFrame(
        message={
            "type": "webex-dtmf",
            "data": {"field": "phone_number", "value": "4155550123"},
        }
    )

    asyncio.run(processor.process_frame(frame, FrameDirection.DOWNSTREAM))

    queue_llm_run.assert_awaited_once()
    assert "4155550123" in context.messages[-1]["content"]
    assert state.expected_field == "date_of_birth"
    assert state.phase is AuthenticationPhase.NEED_TOOL
    assert context.tool_choice == "auto"


def test_final_field_speaks_verification_without_extra_inference() -> None:
    """Confirm verification deterministically once both fields are collected."""
    context = LLMContext([])
    state = WebexInputState(context)
    state.begin_collection("phone_number")
    state.complete_collection("phone_number", "9876543210")
    assert state.begin_collection("date_of_birth")
    queue_llm_run = AsyncMock()
    speak = AsyncMock()
    processor = WebexInputMessageProcessor(state, queue_llm_run, speak)
    frame = InputTransportMessageFrame(
        message={
            "type": "webex-dtmf",
            "data": {"field": "date_of_birth", "value": "31121985"},
        }
    )

    asyncio.run(processor.process_frame(frame, FrameDirection.DOWNSTREAM))

    assert state.phase is AuthenticationPhase.COMPLETE
    speak.assert_awaited_once_with(VERIFICATION_CONFIRMATION)
    queue_llm_run.assert_not_awaited()


def test_rejected_entry_reprompts_the_same_field_without_new_inference() -> None:
    """Speak a retry prompt while the adapter keeps the same field armed."""
    state = WebexInputState(LLMContext([]))
    assert state.begin_collection("phone_number")
    queue_llm_run = AsyncMock()
    speak = AsyncMock()
    processor = WebexInputMessageProcessor(state, queue_llm_run, speak)
    frame = InputTransportMessageFrame(
        message={
            "type": "webex-dtmf-error",
            "data": {"field": "phone_number", "reason": "expected exactly 10 digits"},
        }
    )

    asyncio.run(processor.process_frame(frame, FrameDirection.DOWNSTREAM))

    queue_llm_run.assert_not_awaited()
    assert state.pending_field == "phone_number"
    assert "10 digit" in speak.await_args.args[0]


def test_keypad_tool_reprompts_when_collection_is_already_armed() -> None:
    """Re-speak the instruction instead of failing when the caller speaks."""
    input_state = WebexInputState(LLMContext([]))
    params = _tool_params({}, input_state)
    asyncio.run(handle_request_keypad_input(params))
    params.pipeline_worker.queue_frame.reset_mock()
    params.result_callback.reset_mock()

    asyncio.run(handle_request_keypad_input(params))

    assert input_state.pending_field == "phone_number"
    assert params.result_callback.await_args.args[0]["status"] == "keypad_requested"
    reprompt = params.pipeline_worker.queue_frame.await_args.args[0]
    assert "10 digit" in reprompt.text
    # A repeat ask for the phone number must not claim it already arrived.
    assert "Phone number received" not in reprompt.text
