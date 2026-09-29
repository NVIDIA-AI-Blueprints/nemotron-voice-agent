# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import concurrent.futures
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from examples.frontend_backend_agent.airline.tools import TOOLS_SCHEMA as DELEGATE_TOOLS_SCHEMA
from examples.generic.tools import build_tools_schema
from examples.shared.tool_runtime import (
    schema_validating_tool_handler,
    terminal_tool_handler,
    tool_parameter_schema,
)
from realtime.tool_schema import compile_tool_arguments_validator, tool_argument_validation_failure

_GENERIC_MODULE = Path(__file__).resolve().parents[2] / "src" / "examples" / "generic" / "tools.py"
_OUTER_VALIDATION_WALL_CLOCK_SECS = 1.0


def _params(
    arguments: Any,
    results: list[tuple[Any, Any]],
    *,
    name: str,
    call_id: str,
) -> FunctionCallParams:
    async def result_callback(result: Any, *, properties: Any = None) -> None:
        results.append((result, properties))

    return FunctionCallParams(
        function_name=name,
        tool_call_id=call_id,
        arguments=arguments,
        llm=MagicMock(),
        pipeline_worker=MagicMock(),
        context=MagicMock(),
        result_callback=result_callback,
    )


class ToolSchemaCompilationTests(unittest.TestCase):
    def test_declared_generic_and_delegate_parameter_schemas_are_selected(self) -> None:
        generic_schema, names = build_tools_schema(_GENERIC_MODULE, ["calculate_bmi"])

        self.assertEqual(names, ["calculate_bmi"])
        self.assertEqual(
            tool_parameter_schema(generic_schema, "calculate_bmi")["required"],
            ["weight_kg", "height_m"],
        )
        self.assertEqual(
            tool_parameter_schema(DELEGATE_TOOLS_SCHEMA, "call_backend")["required"],
            ["query"],
        )

    def test_schema_compilation_rejects_invalid_dialect_and_external_reference(self) -> None:
        async def handler(params: FunctionCallParams) -> None:  # pragma: no cover
            await params.result_callback({"ok": True})

        invalid_schemas = (
            {"type": 7},
            {"$schema": "https://example.com/unknown-dialect", "type": "object"},
            {"type": "object", "properties": {"value": {"$ref": "https://example.com/value.json"}}},
            {"type": "object", "$defs": {}, "properties": {"value": {"$ref": "#/$defs/missing"}}},
        )
        for parameters in invalid_schemas:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                schema_validating_tool_handler(handler, parameters=parameters)

    def test_schema_compilation_enforces_bounded_regex_and_structure_contract(self) -> None:
        invalid_schemas = (
            {
                "$defs": {
                    "nested": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "string",
                    }
                }
            },
            {"type": "string", "format": "regex"},
            {
                "type": "object",
                "patternProperties": {"^safe$": {"type": "string"}},
                "unevaluatedProperties": False,
            },
            {"type": "object", "properties": {f"field_{index}": {} for index in range(200)}},
        )
        for parameters in invalid_schemas:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                compile_tool_arguments_validator(parameters)

    def test_budget_is_isolated_across_concurrent_validations(self) -> None:
        validator = compile_tool_arguments_validator(
            {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "^(a+)+$"}},
                "required": ["value"],
            }
        )
        pathological = {"value": "a" * 50_000 + "!"}
        valid = {"value": "a"}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            slow_future = executor.submit(tool_argument_validation_failure, validator, pathological)
            valid_future = executor.submit(tool_argument_validation_failure, validator, valid)
            slow_failure = slow_future.result(timeout=_OUTER_VALIDATION_WALL_CLOCK_SECS)
            valid_failure = valid_future.result(timeout=_OUTER_VALIDATION_WALL_CLOCK_SECS)

        self.assertEqual(slow_failure.code, "tool_schema_error")
        self.assertIsNone(valid_failure)


class TerminalToolHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_handler_exception_returns_one_sanitized_terminal_failure(self) -> None:
        async def handler(_params: FunctionCallParams) -> None:
            raise RuntimeError("private provider detail")

        results: list[tuple[Any, Any]] = []
        wrapped = terminal_tool_handler(handler, parameters={"type": "object"}, timeout_secs=1.0)

        await wrapped(_params({}, results, name="failing_tool", call_id="call_failure"))

        self.assertEqual(len(results), 1)
        result, properties = results[0]
        self.assertEqual(result["error"]["code"], "tool_execution_error")
        self.assertNotIn("private provider detail", result["error"]["message"])
        self.assertIsNone(properties)

    async def test_handler_timeout_cancels_work_and_returns_one_terminal_failure(self) -> None:
        handler_cancelled = asyncio.Event()

        async def handler(_params: FunctionCallParams) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                handler_cancelled.set()

        results: list[tuple[Any, Any]] = []
        wrapped = terminal_tool_handler(handler, parameters={"type": "object"}, timeout_secs=0.01)

        await wrapped(_params({}, results, name="slow_tool", call_id="call_timeout"))

        self.assertTrue(handler_cancelled.is_set())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["error"]["code"], "tool_timeout")
        self.assertIsNone(results[0][1])

    async def test_handler_without_callback_returns_missing_result_failure(self) -> None:
        async def handler(_params: FunctionCallParams) -> None:
            return None

        results: list[tuple[Any, Any]] = []
        wrapped = terminal_tool_handler(handler, parameters={"type": "object"}, timeout_secs=1.0)

        await wrapped(_params({}, results, name="silent_tool", call_id="call_missing"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["error"]["code"], "tool_result_missing")

    async def test_empty_success_is_normalized_to_a_truthy_terminal_result(self) -> None:
        async def handler(params: FunctionCallParams) -> None:
            await params.result_callback(None)

        results: list[tuple[Any, Any]] = []
        wrapped = terminal_tool_handler(handler, parameters={"type": "object"}, timeout_secs=1.0)

        await wrapped(_params({}, results, name="empty_tool", call_id="call_empty"))

        self.assertEqual(results, [({"ok": True, "result": None}, None)])

    async def test_callbacks_after_the_first_terminal_result_are_ignored(self) -> None:
        progress = FunctionCallResultProperties(is_final=False)

        async def handler(params: FunctionCallParams) -> None:
            await params.result_callback({"progress": 1}, properties=progress)
            await params.result_callback({"value": "first"})
            await params.result_callback({"value": "late"})

        results: list[tuple[Any, Any]] = []
        wrapped = terminal_tool_handler(handler, parameters={"type": "object"}, timeout_secs=1.0)

        await wrapped(_params({}, results, name="chatty_tool", call_id="call_duplicate"))

        self.assertEqual(
            results,
            [
                ({"progress": 1}, progress),
                ({"value": "first"}, None),
            ],
        )

    async def test_cancelling_wrapper_delivers_terminal_cancellation_then_reraises(self) -> None:
        handler_started = asyncio.Event()
        handler_cancelled = asyncio.Event()

        async def handler(_params: FunctionCallParams) -> None:
            handler_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                handler_cancelled.set()

        results: list[tuple[Any, Any]] = []
        wrapped = terminal_tool_handler(handler, parameters={"type": "object"}, timeout_secs=10.0)
        task = asyncio.create_task(wrapped(_params({}, results, name="cancel_tool", call_id="call_cancel")))
        await handler_started.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(handler_cancelled.is_set())
        self.assertEqual(len(results), 1)
        result, properties = results[0]
        self.assertEqual(result["error"]["code"], "tool_cancelled")
        self.assertIsNotNone(properties)
        self.assertFalse(properties.run_llm)
        self.assertTrue(properties.is_final)


class ToolArgumentValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_pathological_regex_and_reference_dag_are_bounded_without_handler(self) -> None:
        calls: list[FunctionCallParams] = []

        async def handler(params: FunctionCallParams) -> None:
            calls.append(params)

        definitions: dict[str, Any] = {"node_0": {"type": "object"}}
        for index in range(1, 23):
            previous = f"#/$defs/node_{index - 1}"
            definitions[f"node_{index}"] = {
                "allOf": [{"$ref": previous}, {"$ref": previous}],
            }
        cases = (
            (
                {
                    "type": "object",
                    "properties": {"value": {"type": "string", "pattern": "^(a+)+$"}},
                    "required": ["value"],
                },
                {"value": "a" * 50_000 + "!"},
            ),
            (
                {
                    "type": "object",
                    "patternProperties": {"^(a+)+$": {"type": "string"}},
                    "additionalProperties": False,
                },
                {"a" * 50_000 + "!": "value"},
            ),
            (
                {"$defs": definitions, "$ref": "#/$defs/node_22"},
                {},
            ),
        )

        for index, (parameters, arguments) in enumerate(cases):
            with self.subTest(index=index):
                wrapped = schema_validating_tool_handler(handler, parameters=parameters)
                results: list[tuple[Any, Any]] = []
                started = time.monotonic()
                await wrapped(_params(arguments, results, name="bounded", call_id=f"call_bounded_{index}"))
                elapsed = time.monotonic() - started

                self.assertLess(elapsed, _OUTER_VALIDATION_WALL_CLOCK_SECS)
                self.assertEqual(results[0][0]["error"]["code"], "tool_schema_error")

        self.assertEqual(calls, [])

    async def test_oversized_arguments_are_terminal_without_handler(self) -> None:
        calls: list[FunctionCallParams] = []
        results: list[tuple[Any, Any]] = []

        async def handler(params: FunctionCallParams) -> None:
            calls.append(params)

        wrapped = schema_validating_tool_handler(handler, parameters={"type": "object"})
        await wrapped(
            _params(
                {f"field_{index}": index for index in range(300)},
                results,
                name="bounded",
                call_id="call_oversized",
            )
        )

        self.assertEqual(calls, [])
        self.assertEqual(results[0][0]["error"]["code"], "invalid_tool_arguments")

    async def test_declared_server_and_delegate_tools_reject_invalid_arguments_before_handler(self) -> None:
        generic_schema, _ = build_tools_schema(_GENERIC_MODULE, ["calculate_bmi"])
        calls: list[FunctionCallParams] = []

        async def handler(params: FunctionCallParams) -> None:
            calls.append(params)
            await params.result_callback({"ok": True})

        cases = (
            (
                "calculate_bmi",
                generic_schema,
                [({"weight_kg": "PRIVATE-VALUE", "height_m": 1.8}, "PRIVATE-VALUE", "constraint: type")],
            ),
            (
                "call_backend",
                DELEGATE_TOOLS_SCHEMA,
                [
                    ({}, None, None),
                    ({"query": "find a flight", "private": "DO-NOT-ECHO"}, "DO-NOT-ECHO", None),
                ],
            ),
        )
        for name, schema, invalid_cases in cases:
            wrapped = terminal_tool_handler(
                handler,
                parameters=tool_parameter_schema(schema, name),
                timeout_secs=1.0,
            )
            for index, (arguments, private_value, message_fragment) in enumerate(invalid_cases):
                with self.subTest(tool=name, index=index):
                    results: list[tuple[Any, Any]] = []
                    await wrapped(_params(arguments, results, name=name, call_id=f"call_{name}_{index}"))
                    self.assertEqual(len(results), 1)
                    result, properties = results[0]
                    self.assertIsNone(properties)
                    self.assertEqual(result["error"]["code"], "invalid_tool_arguments")
                    if private_value is not None:
                        self.assertNotIn(private_value, result["error"]["message"])
                    if message_fragment is not None:
                        self.assertIn(message_fragment, result["error"]["message"])

        self.assertEqual(calls, [])

    async def test_local_references_formats_and_finite_json_are_enforced_without_coercion(self) -> None:
        calls: list[FunctionCallParams] = []

        async def handler(params: FunctionCallParams) -> None:
            calls.append(params)

        wrapped = schema_validating_tool_handler(
            handler,
            parameters={
                "$defs": {"request_id": {"type": "string", "format": "uuid"}},
                "type": "object",
                "properties": {"request_id": {"$ref": "#/$defs/request_id"}, "score": {"type": "number"}},
                "required": ["request_id", "score"],
                "additionalProperties": False,
            },
        )

        valid = {"request_id": "123e4567-e89b-12d3-a456-426614174000", "score": 1.0}
        results: list[tuple[Any, Any]] = []
        params = _params(valid, results, name="rank", call_id="call_valid")
        await wrapped(params)
        self.assertEqual(calls, [params])
        self.assertEqual(results, [])
        self.assertIs(params.arguments, valid)

        for index, arguments in enumerate(
            (
                {"request_id": "not-a-uuid", "score": 1.0},
                {"request_id": "123e4567-e89b-12d3-a456-426614174000", "score": float("nan")},
                ["not", "an", "object"],
            )
        ):
            invalid_results: list[tuple[Any, Any]] = []
            await wrapped(_params(arguments, invalid_results, name="rank", call_id=f"call_invalid_{index}"))
            self.assertEqual(invalid_results[0][0]["error"]["code"], "invalid_tool_arguments")

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
