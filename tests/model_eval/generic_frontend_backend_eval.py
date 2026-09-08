#!/usr/bin/env python3
# ruff: noqa: D101, D103
"""Repeated live-NIM qualification for the Generic Frontend/Backend agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from examples.frontend_backend_agent.generic.tools import CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL  # noqa: E402

PROMPTS_PATH = REPO_ROOT / "src/examples/frontend_backend_agent/prompts.yaml"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Case:
    name: str
    user: str
    expected: tuple[str, ...]
    history: tuple[dict[str, Any], ...] = ()


TALKER_CASES = (
    Case("current_weather", "What is the weather in Pune right now?", ("call_backend",)),
    Case("forecast", "Will it rain in Pune tomorrow?", ("call_backend",)),
    Case("stock", "What is NVIDIA trading at right now?", ("call_backend",)),
    Case("latest_web", "What is the latest verified NVIDIA news?", ("call_backend",)),
    Case(
        "latest_web_spoken",
        "Search the web for the latest news about artificial intelligence.",
        ("call_backend",),
    ),
    Case("explicit_lookup", "Check the latest NVIDIA AI announcement using current sources.", ("call_backend",)),
    Case("stale_challenge", "That answer is old. Check the current one.", ("call_backend",)),
    Case("bmi", "I weigh 70 kilograms and am 1.75 metres tall. What is my BMI?", ("call_backend",)),
    Case("bmi_spoken", "What's my BMI if I'm 70 kilos and 1.75 meters?", ("call_backend",)),
    Case("random", "Give me one random integer from 20 through 40.", ("call_backend",)),
    Case("random_spoken", "Give me a random number between one and one hundred.", ("call_backend",)),
    Case(
        "repeat_weather_after_progress_history",
        "Repeat that weather.",
        ("call_backend",),
        history=(
            {"role": "user", "content": "How about Nairobi?"},
            {"role": "assistant", "content": "Let me check Nairobi's current weather."},
            {
                "role": "developer",
                "content": json.dumps(
                    {
                        "type": "async_tool",
                        "status": "finished",
                        "result": {"response_text": "In Nairobi, it is 17.4 degrees C."},
                    }
                ),
            },
            {"role": "assistant", "content": "In Nairobi, it is 17.4 degrees C with cloudy skies."},
        ),
    ),
    Case("stable_direct", "Briefly explain photosynthesis.", ("direct",)),
    Case("cancel", "Never mind, stop that request.", ("cancel_backend",)),
)

THINKER_CASES = (
    Case("current_weather", "Get the current weather in Pune.", ("get_weather",)),
    Case("forecast", "Will it rain in Pune tomorrow?", ("web_search",)),
    Case("stock", "Get the current NVIDIA stock price.", ("get_stock_price",)),
    Case("latest_web", "Find the latest verified NVIDIA news.", ("web_search",)),
    Case("bmi", "Calculate BMI for 70 kilograms and 1.75 metres.", ("calculate_bmi",)),
    Case("random", "Generate one random integer from 20 through 40 inclusive.", ("generate_random_number",)),
    Case("multi", "Get the current Tokyo weather and current NVIDIA stock price.", ("get_weather", "get_stock_price")),
    Case("missing_bmi", "I weigh 70 kilograms. What is my BMI?", ("response_hint:height_m",)),
)


def prompts() -> dict[str, str]:
    data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    return {key: value["content"] for key, value in data.items()}


def prompt_few_shots(key: str) -> tuple[dict[str, Any], ...]:
    data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    entry = data.get(key, {})
    raw_messages = entry.get("few_shots", ()) if isinstance(entry, dict) else ()
    return tuple(dict(message) for message in raw_messages)


def tool_names(message: dict[str, Any]) -> tuple[str, ...]:
    calls = message.get("tool_calls") or []
    return tuple(call.get("function", {}).get("name", "") for call in calls)


def validate_talker(message: dict[str, Any], expected: tuple[str, ...]) -> tuple[bool, str]:
    names = tool_names(message)
    content = str(message.get("content") or "").strip()
    actual = names or (("direct",) if content else ())
    if actual != expected:
        return False, f"expected={expected!r} actual={actual!r} content={content[:100]!r}"
    if names and content:
        return False, "tool call included pre-tool assistant content"
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            return False, f"invalid arguments JSON: {exc}"
        if fn.get("name") == "cancel_backend" and not _semantically_empty(args):
            return False, "cancel_backend arguments must be empty"
        if fn.get("name") == "call_backend":
            allowed = {"query", "filler_text"}
            if set(args) - allowed:
                return False, f"unexpected call_backend fields: {sorted(set(args) - allowed)}"
            if not isinstance(args.get("query"), str) or not args["query"].strip():
                return False, "call_backend query is empty"
            filler = str(args.get("filler_text") or "").strip()
            if not filler:
                return False, "call_backend filler_text is empty"
            words = filler.split()
            if not 3 <= len(words) <= 12 or len(filler) > 96:
                return False, f"call_backend filler_text is not short: {filler!r}"
    return True, "ok"


def _semantically_empty(arguments: dict[str, Any]) -> bool:
    """Accept the local NIM's known JSON-string wrapper only when it contains an empty object."""
    if not arguments:
        return True
    if len(arguments) != 1 or next(iter(arguments)) not in {"args", "arguments"}:
        return False
    wrapped = next(iter(arguments.values()))
    if not isinstance(wrapped, str):
        return False
    try:
        return json.loads(wrapped) == {}
    except json.JSONDecodeError:
        return False


def parse_thinker(content: str) -> dict[str, Any]:
    text = THINK_RE.sub("", content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    return json.loads(text)


def planned_tools(plan: dict[str, Any]) -> tuple[str, ...]:
    if "tool_calls" in plan:
        return tuple(item.get("tool", "") for item in plan["tool_calls"])
    if plan.get("tool") == "response_hint":
        needed = ",".join(plan.get("params_needed") or [])
        return (f"response_hint:{needed}",)
    return (str(plan.get("tool") or ""),)


async def post(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(url, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


async def run_talker(
    args: argparse.Namespace,
    system_prompt: str,
    few_shots: tuple[dict[str, Any], ...] = (),
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(args.concurrency)
    url = f"{args.lightning_url.rstrip('/')}/v1/chat/completions"

    async def one(case: Case, iteration: int) -> tuple[str, bool, str]:
        async with semaphore, httpx.AsyncClient(timeout=args.timeout) as client:
            body = {
                "model": args.lightning_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *few_shots,
                    *case.history,
                    {"role": "user", "content": case.user},
                ],
                "tools": [CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL],
                "tool_choice": "auto",
                "max_tokens": 512,
                "temperature": 0.0,
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": args.talker_reasoning},
            }
            try:
                message = await post(client, url, body)
                ok, detail = validate_talker(message, case.expected)
            except Exception as exc:  # test harness must record transport failures
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            return f"talker/{case.name}/{iteration + 1}", ok, detail

    return await asyncio.gather(*(one(case, i) for case in TALKER_CASES for i in range(args.repeats)))


async def run_thinker(args: argparse.Namespace, system_prompt: str) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(args.concurrency)
    url = f"{args.super_url.rstrip('/')}/v1/chat/completions"

    async def one(case: Case, iteration: int) -> tuple[str, bool, str]:
        async with semaphore, httpx.AsyncClient(timeout=args.timeout) as client:
            payload = {
                "untrusted_user_request": case.user,
                "enabled_tools": [
                    "get_weather",
                    "get_stock_price",
                    "web_search",
                    "calculate_bmi",
                    "generate_random_number",
                ],
                "session_state": {},
                "runtime_context": {
                    "local_datetime": "2026-08-19T14:30:00+05:30",
                    "date": "2026-08-19",
                    "timezone": "Asia/Kolkata",
                },
            }
            body = {
                "model": args.super_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                "max_tokens": 2048,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 1024},
            }
            try:
                message = await post(client, url, body)
                plan = parse_thinker(str(message.get("content") or ""))
                actual = planned_tools(plan)
                ok = actual == case.expected
                detail = "ok" if ok else f"expected={case.expected!r} actual={actual!r} plan={plan!r}"
            except Exception as exc:  # test harness must record transport or parse failures
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            return f"thinker/{case.name}/{iteration + 1}", ok, detail

    return await asyncio.gather(*(one(case, i) for case in THINKER_CASES for i in range(args.repeats)))


def report(results: list[tuple[str, bool, str]]) -> bool:
    counts: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    for name, ok, detail in results:
        group = "/".join(name.split("/")[:2])
        counts[f"{group}:total"] += 1
        counts[f"{group}:pass"] += int(ok)
        if not ok:
            failures.append((name, detail))
    for group in sorted({key.rsplit(":", 1)[0] for key in counts}):
        print(f"{group}: {counts[f'{group}:pass']}/{counts[f'{group}:total']}")
    if failures:
        print("\nFailures:")
        for name, detail in failures[:50]:
            print(f"- {name}: {detail}")
        if len(failures) > 50:
            print(f"- ... {len(failures) - 50} additional failures")
    return not failures


async def main_async(args: argparse.Namespace) -> int:
    prompt_map = prompts()
    results: list[tuple[str, bool, str]] = []
    if args.component in {"talker", "all"}:
        results.extend(await run_talker(args, prompt_map["generic_talker"], prompt_few_shots("generic_talker")))
    if args.component in {"thinker", "all"}:
        results.extend(await run_thinker(args, prompt_map["generic_thinker"]))
    return 0 if report(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("talker", "thinker", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--lightning-url", default="http://127.0.0.1:18000")
    parser.add_argument("--super-url", default="http://127.0.0.1:18001")
    parser.add_argument("--lightning-model", default="nvidia/nemotron-3.5-lightning")
    parser.add_argument("--super-model", default="nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--talker-reasoning", action="store_true")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
