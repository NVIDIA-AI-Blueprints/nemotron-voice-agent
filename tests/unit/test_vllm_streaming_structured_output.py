# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

import enum
import importlib
import sys
import types
from dataclasses import dataclass

import pytest


class RequestStatus(enum.Enum):
    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()


@dataclass
class StreamingUpdate:
    sampling_params: object

    @classmethod
    def from_request(cls, request):
        if not request.resumable:
            return None
        return cls(sampling_params=request.sampling_params)


class Scheduler:
    def _update_request_as_session(self, session, update):
        session.sampling_params = update.sampling_params
        session.status = RequestStatus.WAITING


@dataclass
class Request:
    sampling_params: object
    structured_output_request: object = None
    resumable: bool = True
    status: RequestStatus = RequestStatus.WAITING_FOR_STREAMING_REQ


PATCH_MODULE = "examples.shared.vllm_streaming_input.structured_output"


@pytest.fixture
def patched_vllm(monkeypatch):
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": types.ModuleType("vllm.v1.core.sched.scheduler"),
        "vllm.v1.request": types.ModuleType("vllm.v1.request"),
    }
    modules["vllm.v1.core.sched.scheduler"].Scheduler = type("Scheduler", (Scheduler,), {})
    modules["vllm.v1.request"].RequestStatus = RequestStatus
    modules["vllm.v1.request"].StreamingUpdate = type("StreamingUpdate", (StreamingUpdate,), {})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop(PATCH_MODULE, None)
    importlib.import_module(PATCH_MODULE).keep_structured_output()
    yield modules["vllm.v1.core.sched.scheduler"].Scheduler(), modules["vllm.v1.request"].StreamingUpdate
    sys.modules.pop(PATCH_MODULE, None)


def test_resumed_answer_chunk_decodes_with_its_grammar(patched_vllm):
    scheduler, streaming_update = patched_vllm
    session = Request("prefill")
    grammar = object()

    scheduler._update_request_as_session(session, streaming_update.from_request(Request("answer", grammar)))

    assert session.sampling_params == "answer"
    assert session.structured_output_request is grammar
    assert session.status is RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


def test_resumed_prefill_chunk_drops_the_previous_grammar(patched_vllm):
    scheduler, streaming_update = patched_vllm
    session = Request("answer", object())

    scheduler._update_request_as_session(session, streaming_update.from_request(Request("prefill")))

    assert session.structured_output_request is None
    assert session.status is RequestStatus.WAITING


def test_request_that_is_not_resumable_has_no_update(patched_vllm):
    _, streaming_update = patched_vllm
    request = Request("answer", object())
    request.resumable = False

    assert streaming_update.from_request(request) is None
