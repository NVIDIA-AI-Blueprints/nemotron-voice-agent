# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Golden renders of the deployed eval profile at agent commit 124acf7 (config hash a30f6df59a8b).

fix-plan-f1-f2-f4-f8.md section 1: its control, ``realtime_eval_124acf7.yaml`` with ``gateway.eval_124acf7.yaml``
(the eval pair with every switch of that plan off), must render and decide exactly as the deployment the
2026-10-07/08 runs measured. The goldens were captured from agent 124acf7's ``realtime_eval.yaml`` and
``gateway.eval.yaml`` before any change of that plan.

``render_all()`` renders, for the airline, retail and telecom ``session.update`` fixtures:

* the frontend system prompt of the control profile and the backend system prompt of its gateway;
* the recovery-note texts the eval voice profile appends to a failed person lookup;
* the guarded frontend decision for a table of invented turns (``apply_guards`` with the profile's guard settings).

Regenerate only for an intended change of the deployed behaviour::

    PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden_eval.py --write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN_FILE = HERE / "fixtures" / "golden_eval_124acf7.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _fdh_variants import variant  # noqa: E402
from _fdh_voice_fakes import PACKAGE, DelegationHarness, profile_config, tau2_session_update  # noqa: E402

from examples.frontend_delegation_hermes.core.frontend.decider import DecisionContext, apply_guards  # noqa: E402
from examples.frontend_delegation_hermes.core.frontend.delegate_tool import Decision  # noqa: E402
from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc  # noqa: E402
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates  # noqa: E402

DOMAINS = ("airline", "retail", "telecom")
PROFILE = variant("profiles/realtime_eval_124acf7.yaml")
GATEWAY = variant("gateway.eval_124acf7.yaml")
VOICE_PROMPTS = PACKAGE / "config" / "prompts" / "customer_service" / "voice.yaml"
NOTE_KEYS = ("lookup_not_found_hint_format", "lookup_not_found_hint_words", "tool_argument_invalid_readback")

#: (state, backend asked a question, user text) x the model's decision: silent, speaking, delegating.
TURNS = (
    ("IDLE", False, "Okay."),
    ("IDLE", False, "Hello? Are you still there?"),
    ("IDLE", False, "Sorry, could you repeat that?"),
    ("IDLE", False, "What do you need from me?"),
    ("IDLE", False, "Give me a moment."),
    ("IDLE", True, "Okay."),
    ("IDLE", True, "Yes."),
    ("NO_SESSION", False, "Hello?"),
    ("NO_SESSION", False, "I want to change my order"),
    ("WORKING", False, "Mm-hmm."),
    ("WORKING", False, "Any update?"),
    ("WORKING", False, "Are you still there?"),
    ("WORKING", False, "And my other booking too"),
)
DECISIONS = {
    "silent": Decision(delegate=False, filler_text=""),
    "speaks": Decision(delegate=False, filler_text="You're welcome!"),
    "task": Decision(delegate=True, filler_text="One moment.", request="task"),
}


async def _frontend_system(domain: str) -> str:
    harness = DelegationHarness(profile_config(PROFILE))
    await harness.start(tau2_session_update(domain))
    assert harness.manager is not None
    prompt = harness.manager._system_prompt  # noqa: SLF001 - the exact prompt the frontend model gets
    await harness.close()
    return prompt


def _decisions() -> str:
    config = profile_config(PROFILE)
    lines = []
    for state, asked, text in TURNS:
        context = DecisionContext(
            system_prompt="", messages=[], user_text=text, state=state, backend_asked_question=asked
        )
        for name, decision in DECISIONS.items():
            out = apply_guards(
                decision,
                context,
                backend_question_needs_delegate=config.delegation.backend_question_needs_delegate,
                backchannel_words=config.delegation.backchannel_words,
            )
            lines.append(
                f"{state}|asked={asked}|{text}|{name} -> delegate={out.delegate} filler={out.filler_text!r} "
                f"request={out.request} repair={out.repair!r}"
            )
    return "\n".join(lines)


def _notes() -> str:
    import yaml  # noqa: PLC0415

    catalog = yaml.safe_load(VOICE_PROMPTS.read_text(encoding="utf-8"))
    return "\n---\n".join(f"{key}:\n{catalog[key]['content']}" for key in NOTE_KEYS)


def render_all() -> dict[str, str]:
    """Every deployed-eval render, by name."""
    out: dict[str, str] = {}
    features = gc.load_gateway_config(GATEWAY).prompt_features
    backend = BackendTemplates(
        PACKAGE / "config" / "prompts" / "customer_service" / "backend" / "hermes.yaml", prompt_features=features
    )
    for domain in DOMAINS:
        out[f"frontend_system_{domain}.txt"] = asyncio.run(_frontend_system(domain))
        instructions = tau2_session_update(domain)["session"]["instructions"]
        out[f"backend_system_{domain}.txt"] = backend.render("backend_system", instructions=instructions)
    out["backend_system_empty.txt"] = backend.render("backend_system", instructions="")
    out["voice_notes.txt"] = _notes()
    out["guarded_decisions.txt"] = _decisions()
    return out


def load_golden() -> dict[str, str]:
    """The stored renders by name."""
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite the golden file")
    args = parser.parse_args()
    renders = render_all()
    if not args.write:
        for name, text in renders.items():
            print(f"== {name} ({len(text)} chars)")
        return
    GOLDEN_FILE.write_text(json.dumps(renders, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(renders)} renders to {GOLDEN_FILE}")


if __name__ == "__main__":
    main()
