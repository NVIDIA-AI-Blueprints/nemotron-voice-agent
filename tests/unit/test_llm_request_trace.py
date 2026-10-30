# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify optional request snapshots preserve native SDK and frame behavior."""

import asyncio
import hashlib
import json
import unittest
from types import SimpleNamespace

import httpx
from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, omit
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings

from examples.shared.llm_request_trace import LLMRequestTraceMixin, current_llm_request_trace


class Trace:
    """Retain the actual detached objects passed to the optional recorder."""

    def __init__(self):
        """Allocate observations without using a provider or filesystem."""
        self.events = []
        self.errors = []

    def record(self, event, **fields):
        """Keep references so later mutation would expose a snapshot defect."""
        self.events.append({"event": event, **fields})

    def _note_failure(self, error):
        self.errors.append(type(error).__name__)

    def of_type(self, event):
        """Return a boundary's observations."""
        return [item for item in self.events if item["event"] == event]


class NativeStub:
    """Make delegation, argument identity and cancellation independently visible."""

    def __init__(self, *, client=None, parameters=None, outcome=None):
        """Keep native inputs unchanged for identity checks."""
        self._client = client
        self.parameters = parameters
        self.outcome = outcome
        self.calls = []

    def build_chat_completion_params(self, params_from_context):
        """Return exactly the injected native object."""
        self.calls.append(("build", params_from_context))
        return self.parameters

    async def process_frame(self, frame, direction):
        """Prepare one request and expose the scope seen by native callbacks."""
        self.calls.append(("process", frame, direction))
        self.build_chat_completion_params({"messages": frame.context.get_messages()})
        snapshot = current_llm_request_trace()
        if self.outcome:
            raise self.outcome
        return snapshot

    async def cleanup(self):
        """Expose exact cleanup delegation."""
        self.calls.append(("cleanup",))
        return "native_cleanup_result"


class TracedStub(LLMRequestTraceMixin, NativeStub):
    """Exercise mixin delegation without running any native processor loop."""


class NativeTestService(LLMRequestTraceMixin, NvidiaLLMService):
    """Use real native conversion and OpenAI serialization with an in-memory transport."""

    supports_developer_role = False

    def __init__(self, client, **kwargs):
        """Supply a real OpenAI client without touching a global factory."""
        self.test_client = client
        super().__init__(api_key="test-key", base_url="https://unit.invalid/v1", **kwargs)
        # Native _process_context normally performs this before requesting a
        # stream; these direct request tests bypass that processor lifecycle.
        self._reset_response_state()

    def create_client(self, **kwargs):
        """Inject the isolated client before the mixin attaches its HTTP hook."""
        return self.test_client


def sse_response(content="answer"):
    """Return one choices-bearing native completion and the normal EOF marker."""
    chunk = {
        "id": "unit-completion",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "unit-model",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
    }
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
    )


class LLMRequestTraceTests(unittest.IsolatedAsyncioTestCase):
    """Observe exact native requests and isolate optional recorder failures."""

    async def test_native_parameter_identity_and_immutable_omitted_snapshots(self):
        """Copy supported values without mutating or stringifying native sentinels."""
        trace = Trace()
        params = {
            "messages": [{"role": "user", "content": "original"}],
            "model": "unit",
            "temperature": NOT_GIVEN,
            "top_p": omit,
            "extra_body": {"nested": [1, {"thinking": False}]},
            "extra_headers": {"Authorization": "secret-token"},
        }
        service = TracedStub(parameters=params, evaluation_trace=trace)
        invocation = {"messages": params["messages"]}
        result = service.build_chat_completion_params(invocation)
        self.assertIs(result, params)
        self.assertIs(service.calls[0][1], invocation)
        self.assertEqual(len(service.calls), 1)
        params["messages"][0]["content"] = "changed"
        params["extra_body"]["nested"][1]["thinking"] = True
        event = trace.of_type("llm_request_prepared")[0]
        self.assertEqual(event["parameters"]["messages"][0]["content"], "original")
        self.assertFalse(event["parameters"]["extra_body"]["nested"][1]["thinking"])
        self.assertEqual(event["parameters"]["temperature"], {"$openai_omitted": "NotGiven"})
        self.assertEqual(event["parameters"]["top_p"], {"$openai_omitted": "Omit"})
        self.assertEqual(event["excluded_call_option_names"], ["extra_headers"])
        self.assertNotIn("secret-token", json.dumps(trace.events))
        self.assertEqual(current_llm_request_trace(), {})

    async def test_disabled_trace_preserves_all_native_operations_and_no_hook(self):
        """The default path installs no HTTP hook or request observation."""
        client = SimpleNamespace(_client=SimpleNamespace(event_hooks={"request": []}))
        params = {"messages": []}
        service = TracedStub(parameters=params, client=client)
        frame = LLMContextFrame(LLMContext())
        result = await service.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(result, {})
        self.assertEqual(client._client.event_hooks["request"], [])
        self.assertEqual([c[0] for c in service.calls], ["process", "build"])
        self.assertEqual(await service.cleanup(), "native_cleanup_result")

    async def test_process_scope_survives_native_callbacks_and_resets_after_error_or_cancel(self):
        """Observation identity cannot leak from a failed or canceled native request."""
        for outcome in [None, RuntimeError("native"), asyncio.CancelledError()]:
            with self.subTest(outcome=type(outcome).__name__):
                trace = Trace()
                service = TracedStub(parameters={"messages": []}, outcome=outcome, evaluation_trace=trace)
                frame = LLMContextFrame(LLMContext())
                frame.metadata["nva_offline_response_generation"] = "generation-1"
                if outcome:
                    with self.assertRaises(type(outcome)) as caught:
                        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
                    self.assertIs(caught.exception, outcome)
                else:
                    identity = await service.process_frame(frame, FrameDirection.DOWNSTREAM)
                    self.assertEqual(identity["context_frame_id"], frame.id)
                    self.assertEqual(identity["generation"], "generation-1")
                    self.assertEqual(identity["request_id"], trace.of_type("llm_request_prepared")[0]["request_id"])
                    identity["generation"] = "caller-mutation"
                self.assertEqual(current_llm_request_trace(), {})
                self.assertEqual([c[0] for c in service.calls], ["process", "build"])

    async def test_unsupported_snapshot_and_recorder_failure_cannot_change_native_result(self):
        """Tracing failures remain separate from native provider control flow."""
        trace = Trace()
        params = {"messages": iter(["do not consume"])}
        service = TracedStub(parameters=params, evaluation_trace=trace)
        self.assertIs(service.build_chat_completion_params({}), params)
        self.assertEqual(next(params["messages"]), "do not consume")
        self.assertIn("TypeError", trace.errors)

        def fail(*args, **kwargs):
            raise ValueError("trace-failure")

        trace.record = fail
        params["messages"] = []
        self.assertIs(service.build_chat_completion_params({}), params)
        self.assertIn("ValueError", trace.errors)
        service.outcome = RuntimeError("provider-failure")
        with self.assertRaises(RuntimeError) as caught:
            await service.process_frame(LLMContextFrame(LLMContext()), FrameDirection.DOWNSTREAM)
        self.assertIs(caught.exception, service.outcome)

    async def test_overlapping_frame_scopes_keep_context_and_generation_identity(self):
        """Independent processing tasks cannot overwrite each other's request identity."""
        entered = 0
        ready = asyncio.Event()

        class ConcurrentNative(NativeStub):
            def build_chat_completion_params(self, params_from_context):
                return params_from_context

            async def process_frame(self, frame, direction):
                nonlocal entered
                self.build_chat_completion_params({"messages": frame.context.get_messages()})
                first = current_llm_request_trace()
                entered += 1
                if entered == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 1)
                return first, current_llm_request_trace()

        class ConcurrentService(LLMRequestTraceMixin, ConcurrentNative):
            pass

        trace = Trace()
        service = ConcurrentService(evaluation_trace=trace)
        frames = [LLMContextFrame(LLMContext(messages=[{"role": "user", "content": x}])) for x in ["first", "second"]]
        for i, frame in enumerate(frames):
            frame.metadata["nva_offline_response_generation"] = f"generation-{i}"
        results = await asyncio.gather(*(service.process_frame(f, FrameDirection.DOWNSTREAM) for f in frames))
        for i, (before, after) in enumerate(results):
            self.assertEqual(before, after)
            self.assertEqual(after["context_frame_id"], frames[i].id)
            self.assertEqual(after["generation"], f"generation-{i}")
        self.assertNotEqual(results[0][0]["request_id"], results[1][0]["request_id"])
        self.assertEqual(current_llm_request_trace(), {})

    async def client(self, handler, *, max_retries=0):
        """Create an isolated real SDK client whose transport cannot reach a network."""
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="secret-header-value",
            base_url="https://unit.invalid/v1",
            default_query={"private-query": "hidden"},
            http_client=http,
            max_retries=max_retries,
        )
        self.addAsyncCleanup(client.close)
        return client

    async def drain(self, service, context):
        """Consume the unmodified native stream until its ordinary EOF."""
        stream = await service.get_chat_completions(context)
        return [chunk async for chunk in stream]

    async def test_real_native_conversion_sdk_merge_and_serialized_body_are_distinct(self):
        """Capture the final SDK body even when extra_body overrides prepared messages."""
        wire = []

        def handler(request):
            wire.append(request.content)
            return sse_response()

        trace = Trace()
        client = await self.client(handler)
        service = NativeTestService(
            client,
            evaluation_trace=trace,
            settings=NvidiaLLMSettings(
                model="unit-model",
                max_tokens=5000,
                extra={"extra_body": {"messages": [{"role": "user", "content": "wire override"}], "thinking": False}},
            ),
        )
        context = LLMContext(messages=[{"role": "developer", "content": "convert this role"}])
        chunks = await self.drain(service, context)
        self.assertEqual(len(chunks), 1)
        prepared = trace.of_type("llm_request_prepared")[0]
        sent = trace.of_type("llm_http_request")[0]
        self.assertEqual(prepared["parameters"]["messages"][0]["role"], "user")
        self.assertEqual(prepared["parameters"]["messages"][0]["content"], "convert this role")
        self.assertEqual(sent["request_body"]["messages"][0]["content"], "wire override")
        self.assertEqual(sent["request_body"]["max_tokens"], 5000)
        self.assertFalse(sent["request_body"]["thinking"])
        self.assertNotIn("extra_body", sent["request_body"])
        self.assertEqual(sent["serialized_body_sha256"], hashlib.sha256(wire[0]).hexdigest())
        self.assertEqual(sent["request_body"], json.loads(wire[0]))
        self.assertEqual(sent["request_id"], prepared["request_id"])
        self.assertEqual(sent["http_attempt"], 1)
        context.get_messages()[0]["content"] = "later mutation"
        self.assertEqual(prepared["parameters"]["messages"][0]["content"], "convert this role")
        self.assertNotIn("secret-header-value", json.dumps(trace.events))
        self.assertNotIn("private-query", json.dumps(trace.events))
        self.assertEqual(current_llm_request_trace(), {})

    async def test_sdk_retries_have_one_prepared_request_and_actual_attempts(self):
        """Retain native SDK retry behavior and each actual serialized dispatch."""
        attempts = []

        def handler(request):
            attempts.append(request.content)
            if len(attempts) == 1:
                return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "busy"}})
            return sse_response()

        trace = Trace()
        service = NativeTestService(await self.client(handler, max_retries=1), evaluation_trace=trace)
        await self.drain(service, LLMContext(messages=[{"role": "user", "content": "question"}]))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(len(trace.of_type("llm_request_prepared")), 1)
        observed = trace.of_type("llm_http_request")
        self.assertEqual([e["http_attempt"] for e in observed], [1, 2])
        self.assertEqual([e["sdk_retry_count"] for e in observed], [0, 1])
        self.assertEqual(observed[0]["request_id"], observed[1]["request_id"])

    async def test_enabling_capture_keeps_real_native_serialized_request_byte_identical(self):
        """A traced and default native call send the same prepared request body."""
        wire = []

        def handler(request):
            wire.append(request.content)
            return sse_response()

        settings = dict(
            model="unit-model",
            max_tokens=5000,
            extra={"extra_body": {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}},
        )
        baseline = NativeTestService(await self.client(handler), settings=NvidiaLLMSettings(**settings))
        trace = Trace()
        observed = NativeTestService(
            await self.client(handler), evaluation_trace=trace, settings=NvidiaLLMSettings(**settings)
        )
        for service in [baseline, observed]:
            await self.drain(service, LLMContext(messages=[{"role": "user", "content": "same question"}]))
        self.assertEqual(wire[0], wire[1])
        self.assertEqual(trace.of_type("llm_http_request")[0]["request_body"], json.loads(wire[0]))
        self.assertEqual(baseline._client._client.event_hooks["request"], [])

    async def test_native_timeout_retry_child_task_keeps_request_identity(self):
        """The native wait_for retry inherits the same request identity."""
        count = 0
        block_first = False

        async def handler(request):
            nonlocal count
            count += 1
            if block_first and count == 1:
                await asyncio.sleep(1)
            return sse_response()

        trace = Trace()
        service = NativeTestService(
            await self.client(handler), evaluation_trace=trace, retry_on_timeout=False, retry_timeout_secs=0.01
        )
        # Warm the SDK's deferred imports with a mock response. Otherwise the
        # deadline could expire before HTTP dispatch, correctly yielding only
        # one observed HTTP attempt despite two native invocation attempts.
        await self.drain(service, LLMContext(messages=[{"role": "user", "content": "warm mock"}]))
        service._retry_on_timeout = True
        trace.events.clear()
        count = 0
        block_first = True
        await self.drain(service, LLMContext(messages=[{"role": "user", "content": "question"}]))
        observed = trace.of_type("llm_http_request")
        self.assertEqual(count, 2)
        self.assertEqual([e["http_attempt"] for e in observed], [1, 2])
        self.assertEqual([e["sdk_retry_count"] for e in observed], [0, 0])
        self.assertEqual(len({e["request_id"] for e in observed}), 1)
        self.assertEqual(current_llm_request_trace(), {})

    async def test_provider_status_error_and_trace_hook_failure_both_preserve_native_error(self):
        """A broken recorder cannot replace the SDK's actual provider exception."""
        trace = Trace()
        service = NativeTestService(
            await self.client(lambda request: httpx.Response(503, json={"error": {"message": "unavailable"}})),
            evaluation_trace=trace,
        )

        def fail(*args, **kwargs):
            raise ValueError("broken recorder")

        trace.record = fail
        with self.assertRaises(APIStatusError) as caught:
            await service.get_chat_completions(LLMContext(messages=[{"role": "user", "content": "question"}]))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertTrue(trace.errors)
        self.assertEqual(current_llm_request_trace(), {})

    async def test_same_client_overlapping_requests_do_not_cross_attribute_traces(self):
        """Two services sharing a client retain distinct concurrent request identities."""
        entered = 0
        ready = asyncio.Event()

        async def handler(request):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1)
            return sse_response()

        client = await self.client(handler)
        left, right = Trace(), Trace()
        a = NativeTestService(client, evaluation_trace=left)
        b = NativeTestService(client, evaluation_trace=right)
        await asyncio.gather(
            self.drain(a, LLMContext(messages=[{"role": "user", "content": "left"}])),
            self.drain(b, LLMContext(messages=[{"role": "user", "content": "right"}])),
        )
        self.assertEqual(len(left.of_type("llm_http_request")), 1)
        self.assertEqual(len(right.of_type("llm_http_request")), 1)
        first, second = left.of_type("llm_http_request")[0], right.of_type("llm_http_request")[0]
        self.assertEqual(first["request_body"]["messages"][0]["content"], "left")
        self.assertEqual(second["request_body"]["messages"][0]["content"], "right")
        self.assertNotEqual(first["request_id"], second["request_id"])
        hook_b = b._llm_trace_http_hook
        # The native service is not set up here, so test cleanup delegation
        # separately from native task-manager requirements.
        from unittest.mock import AsyncMock, patch

        with patch.object(NvidiaLLMService, "cleanup", AsyncMock(return_value="done")):
            self.assertEqual(await a.cleanup(), "done")
        self.assertEqual(client._client.event_hooks["request"], [hook_b])

    async def test_shared_client_untraced_requests_are_never_retained_by_traced_service(self):
        """Sharing an SDK client cannot make unrelated request bodies observable."""
        wire = []

        def handler(request):
            wire.append(json.loads(request.content))
            return sse_response()

        client = await self.client(handler)
        trace = Trace()
        traced = NativeTestService(client, evaluation_trace=trace)
        untraced = NativeTestService(client)
        stream = await client.chat.completions.create(
            model="unit-model", messages=[{"role": "user", "content": "unowned direct request"}], stream=True
        )
        self.assertEqual(len([chunk async for chunk in stream]), 1)
        await self.drain(untraced, LLMContext(messages=[{"role": "user", "content": "unowned native request"}]))
        self.assertEqual(trace.of_type("llm_http_request"), [])
        self.assertEqual(trace.of_type("llm_request_prepared"), [])
        self.assertEqual(
            [item["messages"][0]["content"] for item in wire],
            [
                "unowned direct request",
                "unowned native request",
            ],
        )
        self.assertNotIn("unowned", json.dumps(trace.events))
        # The existing hook must still capture its own native invocation.
        await self.drain(traced, LLMContext(messages=[{"role": "user", "content": "owned native request"}]))
        self.assertEqual(len(trace.of_type("llm_http_request")), 1)
        self.assertEqual(trace.of_type("llm_http_request")[0]["request_body"], wire[2])
        self.assertEqual(current_llm_request_trace(), {})

    async def test_actual_addressed_service_captures_after_join_and_reply_format(self):
        """The production MRO snapshots parameters after addressed-reply changes."""
        from examples.generic.pipeline import AddressedEvaluationNvidiaLLMService
        from examples.generic.silent_reply import ADDRESSED_REPLY_FORMAT

        class AddressedTestService(AddressedEvaluationNvidiaLLMService):
            def __init__(self, client, **kwargs):
                self.test_client = client
                super().__init__(api_key="test-key", base_url="https://unit.invalid/v1", **kwargs)
                self._reset_response_state()

            def create_client(self, **kwargs):
                return self.test_client

        wire = []

        def handler(request):
            wire.append(json.loads(request.content))
            return sse_response('{"addressed_to_assistant":true,"reply":"ready"}')

        trace = Trace()
        service = AddressedTestService(
            await self.client(handler), evaluation_trace=trace, settings=NvidiaLLMSettings(model="unit-model")
        )
        context = LLMContext(
            messages=[
                {"role": "system", "content": "reply as addressed"},
                {"role": "user", "content": "Speaker 1: first sentence."},
                {"role": "user", "content": "Speaker 1: second sentence."},
            ]
        )
        chunks = await self.drain(service, context)
        self.assertEqual(chunks[0].choices[0].delta.content, "ready")
        prepared = trace.of_type("llm_request_prepared")[0]
        captured = trace.of_type("llm_http_request")[0]
        expected = [
            {"role": "system", "content": "reply as addressed"},
            {"role": "user", "content": "Speaker 1: first sentence. second sentence."},
        ]
        self.assertEqual(prepared["parameters"]["messages"], expected)
        self.assertEqual(prepared["parameters"]["response_format"], ADDRESSED_REPLY_FORMAT)
        self.assertEqual(captured["request_body"]["messages"], expected)
        self.assertEqual(captured["request_body"]["response_format"], ADDRESSED_REPLY_FORMAT)
        self.assertEqual(captured["request_body"], wire[0])
        self.assertEqual(captured["request_id"], prepared["request_id"])
        self.assertEqual(len(context.get_messages()), 3)

    async def test_non_chat_and_unread_stream_requests_are_not_reinterpreted(self):
        """Unsupported bodies are reported without consuming an original stream."""
        trace = Trace()
        service = NativeTestService(await self.client(lambda request: sse_response()), evaluation_trace=trace)
        service._observe_llm_http_request(httpx.Request("POST", "https://unit.invalid/v1/embeddings", json={"x": 1}))
        self.assertEqual(trace.of_type("llm_http_request"), [])

        class NativeObserverStub(NativeStub):
            async def get_chat_completions(self, request):
                self._observe_llm_http_request(request)
                return request

        class ScopedObserver(LLMRequestTraceMixin, NativeObserverStub):
            pass

        owned = ScopedObserver(evaluation_trace=trace)
        request = httpx.Request("POST", "https://unit.invalid/v1/chat/completions", content=iter([b"{}"]))
        self.assertIs(await owned.get_chat_completions(request), request)
        self.assertIn("RequestNotRead", trace.errors)
        self.assertEqual(trace.of_type("llm_http_request"), [])


if __name__ == "__main__":
    unittest.main()
