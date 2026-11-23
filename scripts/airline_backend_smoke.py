# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Demo Air backend prompt and tools against a real model, one scripted caller at a time.

The script plays the backend of the Frontend/Backend Live airline desk: it gives the model the ``Backend.airline``
and ``Channel.airline`` prompts and the ``airline`` tools, sends the caller's turns, runs the tool calls on a fresh
``AirlineTools``, and prints what the model called and said. It checks only what a program can check (tool statuses
and spoken-text rules). Read the transcript to judge the conversation. The callers are ``book`` (search, book, seat,
meal), ``manage`` (cancel), ``scope`` (a city Demo Air does not serve, where it flies) and ``disruption`` (a cancelled
flight, rebooked at no charge).

Usage, from the repository root, with ``NVIDIA_API_KEY`` set in the environment:

    uv run python scripts/airline_backend_smoke.py
    uv run python scripts/airline_backend_smoke.py --script manage scope
    uv run python scripts/airline_backend_smoke.py --script book --turns 1 4 --model <model>

The script never prints the key. It makes network calls to the endpoint that you name with ``--base-url``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.tools import (
    AirlineTools,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import load_prompts

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
MAX_ROUNDS = 6
# The prompt asks for about 35 words (40 for a read-back); more than this is too long to listen to.
MAX_WORDS = 45

# What the voice cannot say well: markdown, an ISO date, a 24-hour clock time.
SPOKEN_RULES = {
    "markdown": re.compile(r"[*#`]|^\s*[-•]\s", re.MULTILINE),
    "iso date": re.compile(r"\b20\d\d-\d\d-\d\d\b"),
    "24-hour time": re.compile(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b(?!\s*[AP]M)", re.IGNORECASE),
}
# The demo note, which the caller should hear once per call ("the sample account" and "a demo account" are not it).
DEMO_NOTE = re.compile(r"\b(?:demo|sample|made-up|made up|fictional)\b(?!\s+account)|\bcharged\b", re.IGNORECASE)
# An airline that is not Demo Air: the desk has one airline, and the model must name no other.
REAL_AIRLINE = re.compile(
    r"\b(?:american|delta|united|southwest|jetblue|alaska airlines|spirit|frontier|lufthansa|emirates|"
    r"british airways|air canada)\b",
    re.IGNORECASE,
)
# A caller's time-of-day wish, which must reach flights_search as a filter.
TIME_OF_DAY = re.compile(r"\b(?:morning|afternoon|evening|red-eye|overnight)\b", re.IGNORECASE)
# The words the demo note may add to the answer that says it.
DEMO_NOTE_WORDS = 14
# A change claimed as done, which a read-back (a needs_confirmation result) must never say.
DONE_CLAIM = re.compile(
    r"\b(?:I've|I have|you're|you are|is now|are now|all set|done)\b.{0,40}?"
    r"\b(?:booked|changed|cancell?ed|upgraded|set|confirmed|done)\b|\ball set\b|\byou're booked\b",
    re.IGNORECASE,
)


def caller_scripts(today: datetime) -> dict[str, list[str]]:
    """Return each scripted caller's turns by name; the booking caller's date is a few days ahead of ``today``."""
    day = (today + timedelta(days=5)).strftime("%B %d").replace(" 0", " ")
    return {
        "book": [
            f"Hi, I need a flight from Boston to Denver on {day}.",
            "Anything leaving in the afternoon?",
            "Book the first one please.",
            "My user ID is jane doe zero zero one.",
            "Yes, go ahead.",
            "Can I get a window seat and a vegetarian meal on that?",
            "Yes, do it.",
            "How often is that flight on time?",
        ],
        # The sign-in result carries no data_notice, so the cancel's read-back is the first result that does.
        "manage": [
            "My user ID is jane doe zero zero one.",
            "Please cancel booking K seven Q two Z P.",
            "Yes, cancel it.",
            "What trips do I have now?",
        ],
        # A place that Demo Air does not serve gets its closest city, a far one the list of cities; a flight
        # that the caller then picks is found as usual.
        "scope": [
            f"Hi, I want a flight from Boise to Denver on {day}.",
            "Where do you fly?",
            f"Okay, then Seattle to Miami on {day}.",
            "What about flying to London?",
        ],
        # Marcus Lee's flight today was cancelled: the desk says so, finds another flight, and rebooks it free.
        "disruption": [
            "My user ID is marcus lee zero zero two.",
            "Is my flight today on time?",
            "Yes, please find me another one today.",
            "Move me to the first one.",
            "Yes, go ahead.",
        ],
    }


def chat_tools(tools: list[dict]) -> list[dict]:
    """Return the Responses function tools in the chat completions shape."""
    return [
        {"type": "function", "function": {k: tool[k] for k in ("name", "description", "parameters")}} for tool in tools
    ]


def spoken_problems(text: str, demo_said: bool = False, read_back: bool = False, notice: bool = False) -> list[str]:
    """Return the names of the rules that ``text`` breaks.

    ``demo_said`` is whether the demo note was said earlier in the call; ``read_back`` is whether this turn's
    change waits for a yes (a ``needs_confirmation`` result and no ``confirmed`` one); ``notice`` is whether a
    result of this turn carried ``data_notice``.
    """
    found = [name for name, rule in SPOKEN_RULES.items() if rule.search(text)]
    if REAL_AIRLINE.search(text):
        found.append("names an airline other than Demo Air")
    has_note = bool(DEMO_NOTE.search(text))
    limit = MAX_WORDS + (DEMO_NOTE_WORDS if has_note and not demo_said else 0)
    if len(text.split()) > limit:
        found.append(f"over {limit} words")
    if demo_said and has_note:
        found.append("demo note repeated")
    if notice and not demo_said and not has_note:
        found.append("data_notice not said")
    if read_back and DONE_CLAIM.search(text):
        found.append("read-back claims the change is done")
    if read_back and not text.rstrip().endswith("?"):
        found.append("read-back does not end with its question")
    return found


async def run(client: Any, model: str, turns: list[str]) -> int:
    """Play ``turns`` against ``model``; return the number of problems found."""
    prompts = load_prompts()
    system = f"{prompts.text('Backend', 'airline')}\n\n{prompts.text('Channel', 'airline')}"
    desk = AirlineTools()
    tools = chat_tools(desk.tools())
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    problems, call_seq, demo_said = 0, 0, False
    for number, turn in enumerate(turns, 1):
        messages.append({"role": "user", "content": turn})
        print(f"\n[{number}] CALLER: {turn}")
        called: list[str] = []
        statuses: list[str] = []
        notice = False
        for _ in range(MAX_ROUNDS):
            reply = (
                (await client.chat.completions.create(model=model, messages=messages, tools=tools, temperature=0.0))
                .choices[0]
                .message
            )
            calls = reply.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": c.id,
                                    "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                                }
                                for c in calls
                            ]
                        }
                        if calls
                        else {}
                    ),
                }
            )
            if not calls:
                text = (reply.content or "").strip()
                print(f"    BACKEND SAYS: {text} ({len(text.split())} words)")
                read_back = "needs_confirmation" in statuses and "confirmed" not in statuses
                found = spoken_problems(text, demo_said, read_back, notice)
                if TIME_OF_DAY.search(turn) and "flights_search" not in called:
                    found.append("a time-of-day wish answered without flights_search")
                demo_said = demo_said or bool(DEMO_NOTE.search(text))
                if found:
                    problems += len(found)
                    print(f"    PROBLEM: {', '.join(found)}")
                break
            for call in calls:
                call_seq += 1
                output = await desk.execute(f"smoke_{call_seq}", call.function.name, call.function.arguments)
                result = json.loads(output)
                status = result.get("status")
                notice = notice or "data_notice" in result
                called.append(call.function.name)
                statuses.append(str(status))
                print(f"    TOOL {call.function.name}({call.function.arguments}) -> {status}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        else:
            problems += 1
            print(f"    PROBLEM: no spoken answer after {MAX_ROUNDS} rounds")
    print(f"\nDone: {problems} problem(s) found by the checks above. Read the transcript for the rest.")
    return problems


def main() -> int:
    """Parse the options, build the client, and run the script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.getenv("AIRLINE_SMOKE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("AIRLINE_SMOKE_MODEL", DEFAULT_MODEL))
    scripts = caller_scripts(datetime.now(UTC))
    parser.add_argument(
        "--script", nargs="*", choices=list(scripts), default=list(scripts), help="play these callers (default: all)"
    )
    parser.add_argument("--turns", type=int, nargs="*", default=[], help="play only these turn numbers (1-based)")
    args = parser.parse_args()
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        print("Set NVIDIA_API_KEY in the environment first.", file=sys.stderr)
        return 2
    from openai import AsyncOpenAI

    async def play() -> int:
        client = AsyncOpenAI(base_url=args.base_url, api_key=key)
        problems = 0
        for name in args.script:
            turns = scripts[name]
            if args.turns:
                turns = [turn for number, turn in enumerate(turns, 1) if number in args.turns]
            print(f"\n=== {name}")
            problems += await run(client, args.model, turns)
        return problems

    return 1 if asyncio.run(play()) else 0


if __name__ == "__main__":
    sys.exit(main())
