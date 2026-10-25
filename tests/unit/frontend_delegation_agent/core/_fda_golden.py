# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Golden prompt renders of the baseline (every feature off), captured at agent commit 3a7e04a.

Regenerated once with only the example values changed to invented ones (prompts.yaml
normalization_note; tau3-voice-domain-specific-changes-genericization.md item 2).

``render_all()`` renders every prompt the agents see with the shipped catalogs and the tau2 airline
``session.update``: the frontend system prompt and state block, the backend SOUL, system prompt and
context lines, the status prompts and the local tool-result messages. ``test_fda_golden_prompts``
asserts byte equality with ``fixtures/golden_prompts.json`` (JSON, so the whitespace pre-commit hooks
cannot alter the renders: the tau2 policy has trailing spaces).

Regenerate only for an intended baseline change (tau3-failure-fixes-plan.md section 1, rule 2)::

    PYTHONPATH=src uv run python tests/unit/frontend_delegation_agent/core/_fda_golden.py --write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN_FILE = HERE / "fixtures" / "golden_prompts.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _fda_voice_fakes import PACKAGE, DelegationHarness, tau2_session_update  # noqa: E402

from examples.frontend_delegation_agent.core.frontend.decider import DecisionContext, LLMDecider  # noqa: E402
from examples.frontend_delegation_agent.core.frontend.delegate_tool import delegate_tool  # noqa: E402
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates  # noqa: E402

_STATES = {
    "no_session": DecisionContext(system_prompt="", messages=[], user_text="hi", state="NO_SESSION"),
    "working": DecisionContext(
        system_prompt="",
        messages=[],
        user_text="any update?",
        state="WORKING",
        current_request="I want to change my flight",
        elapsed_s=12,
        recent_activity=["get_reservation_details: done (ok)"],
    ),
    "idle_question": DecisionContext(
        system_prompt="", messages=[], user_text="yes", state="IDLE", backend_asked_question=True
    ),
}


async def _frontend_system() -> tuple[str, str]:
    harness = DelegationHarness()
    await harness.start(tau2_session_update("airline"))
    assert harness.manager is not None
    prompt = harness.manager._system_prompt  # noqa: SLF001 - the exact prompt the frontend model gets
    runtime = harness.runtime
    await harness.close()
    return prompt, repr(delegate_tool(runtime.prompts))


def render_all() -> dict[str, str]:
    """Every baseline render, by golden file name."""
    out: dict[str, str] = {}
    system, tool = asyncio.run(_frontend_system())
    out["frontend_system_airline.txt"] = system
    out["frontend_delegate_tool.txt"] = tool
    harness = DelegationHarness()
    prompts = harness.runtime.prompts
    decider = LLMDecider(object(), prompts, timeout_ms=1000, max_repair_attempts=1, on_contract_error="delegate")  # type: ignore[arg-type]
    for name, context in _STATES.items():
        out[f"frontend_state_{name}.txt"] = decider.messages(context)[0]["content"]
    out["frontend_contract_repair.txt"] = prompts.render("contract_repair")
    out["frontend_status_verbalize.txt"] = prompts.render("status_verbalize", summary_text="task: change flight")
    out["frontend_status_template.txt"] = prompts.render(
        "status_template", current_tool="get_reservation_details", current_tool_phrase="looking it up", done_count=2
    )
    backend = BackendTemplates(PACKAGE / "config" / "prompts" / "customer_service" / "backend" / "hermes.yaml")
    airline = tau2_session_update("airline")["session"]["instructions"]
    out["backend_soul.txt"] = backend.render("backend_soul")
    out["backend_system_airline.txt"] = backend.render("backend_system", instructions=airline)
    out["backend_system_empty.txt"] = backend.render("backend_system", instructions="")
    out["backend_context_lines.txt"] = "\n".join(
        [
            backend.render("context_user", during_run=True, text="and my other booking"),
            backend.render("context_user", during_run=False, text="hello"),
            backend.render("context_frontend", partial=False, text="Sure, one moment."),
            backend.render("context_status", partial=True, text="Still checking"),
            backend.render("context_controller", partial=False, text="Sorry."),
            backend.render("context_delivery_note", outcome="partial", heard_text="Your flight"),
            backend.render("context_delivery_note", outcome="not_heard", heard_text=""),
            backend.render("context_requeued_request", text="cancel it"),
            backend.render(
                "context_side_effect", calls=[{"name": "cancel_reservation", "arguments": "{}", "result": "ok"}]
            ),
            backend.render("backend_delegation_message", block='[earlier] user: "hi"', text="cancel it"),
            backend.render("backend_steer", block="", text="make it two bags"),
            backend.render("backend_error_spoken"),
            backend.render("backend_unavailable"),
        ]
    )
    out["local_tool_messages.txt"] = _local_messages()
    return out


def _local_messages() -> str:
    from examples.frontend_delegation_agent.core.text.messages import ToolCall  # noqa: PLC0415

    harness = DelegationHarness()
    normalizer = harness.runtime.parts(None).argument_normalizer  # type: ignore[arg-type]
    assert normalizer is not None
    invalid = normalizer.screen(
        [ToolCall(id="c1", name="get_user_details", arguments_json='{"user_id": "aalto_ah"}')], frozenset()
    )
    call = ToolCall(id="c2", name="get_user_details", arguments_json='{"user_id": "jordan_lee_8264"}')
    key = normalizer.screen([call], frozenset()).keys["c2"]
    failed = normalizer.screen([call], frozenset({key}))
    return "\n---\n".join(answer.message for answer in (*invalid.local, *failed.local))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite the golden files")
    args = parser.parse_args()
    renders = render_all()
    if not args.write:
        for name, text in renders.items():
            print(f"== {name} ({len(text)} chars)")
        return
    GOLDEN_FILE.write_text(json.dumps(renders, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(renders)} renders to {GOLDEN_FILE}")


def load_golden() -> dict[str, str]:
    """The stored renders by name."""
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
