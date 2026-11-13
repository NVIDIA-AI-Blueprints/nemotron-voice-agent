# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
import json
import re

import pytest
import yaml

from examples.generic_assistant_livekit import scenarios as sc


def tools_by_name(scenario_id):
    return {fn.__name__: fn for fn in sc.SCENARIOS[scenario_id].make_tools()}


def test_registry_and_fallback():
    assert set(sc.SCENARIOS) == {"generic", "insurance", "clinic", "bank"}
    assert sc.get_scenario("nope").id == sc.DEFAULT_SCENARIO
    assert sc.get_scenario(None).id == sc.DEFAULT_SCENARIO
    assert sc.get_scenario(["bank"]).id == sc.DEFAULT_SCENARIO
    assert sc.get_scenario("bank").id == "bank"
    assert all(c["examples"] and c["description"] and c["name"] for c in sc.scenario_catalog())


def test_every_tool_has_docstring_and_args_section():
    for s in sc.SCENARIOS.values():
        for fn in s.make_tools():
            assert fn.__doc__, fn.__name__
            if fn.__code__.co_argcount:
                assert "Args:" in fn.__doc__, fn.__name__


def test_prompts_only_mention_tools_that_exist():
    for scenario in sc.SCENARIOS.values():
        names = {fn.__name__ for fn in scenario.make_tools()}
        mentioned = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", scenario.instructions))
        assert mentioned, scenario.id
        assert mentioned <= names, (scenario.id, mentioned - names)


def test_voice_style_is_appended_only_where_requested():
    prompts = yaml.safe_load(sc.PROMPTS_FILE.read_text())
    style = prompts["voice_style"]["content"].strip()
    for scenario_id, (key, _) in sc.USE_CASES.items():
        has_style = style in sc.SCENARIOS[scenario_id].instructions
        assert has_style == bool(prompts[key].get("voice_style")), scenario_id
    assert not has_style_for("generic")
    assert has_style_for("bank")


def has_style_for(scenario_id):
    style = yaml.safe_load(sc.PROMPTS_FILE.read_text())["voice_style"]["content"].strip()
    return style in sc.SCENARIOS[scenario_id].instructions


def test_the_bank_description_shows_the_demo_pin():
    assert sc.DEMO_PIN in sc.SCENARIOS["bank"].description


def test_instructions_are_single_prompts_without_stray_whitespace():
    for scenario in sc.SCENARIOS.values():
        assert scenario.instructions == scenario.instructions.strip()
        assert scenario.greeting and scenario.greeting == scenario.greeting.strip()


def test_insurance_flow():
    t = tools_by_name("insurance")
    assert t["lookup_policy"]("aut-1001")["holder"] == "Jordan Lee"
    with pytest.raises(ValueError):
        t["lookup_policy"]("AUT-9999")
    claim = t["file_claim"]("AUT-1001", "2026-11-10", "Rear-ended at a light", 1800)
    assert t["check_claim_status"](claim["claim_id"])["status"] == "received"
    assert t["file_claim"]("AUT-2002", "2026-11-10", "x", 100)["note"]  # liability-only warning


def test_clinic_flow_and_double_booking():
    t = tools_by_name("clinic")
    slot = t["list_available_slots"]("dermatology")[0]
    booking = t["book_appointment"](slot["slot_id"], "Pat Kim", "rash")
    with pytest.raises(ValueError, match="taken"):
        t["book_appointment"](slot["slot_id"], "Other", "x")
    assert t["cancel_appointment"](booking["confirmation_id"])["cancelled"] == booking["confirmation_id"]
    t["book_appointment"](slot["slot_id"], "Other", "x")  # free again


def test_bank_requires_pin_and_locks():
    t = tools_by_name("bank")
    with pytest.raises(ValueError, match="not verified"):
        t["get_balance"]("checking")
    for _ in range(3):
        with pytest.raises(ValueError, match="incorrect"):
            t["verify_identity"]("0000")
    with pytest.raises(ValueError, match="Too many"):
        t["verify_identity"](sc.DEMO_PIN)


def test_bank_transfer():
    t = tools_by_name("bank")
    t["verify_identity"](sc.DEMO_PIN)
    out = t["transfer_funds"]("checking", "savings", 200)
    assert out["savings_usd"] == 10450.0
    with pytest.raises(ValueError, match="Insufficient"):
        t["transfer_funds"]("checking", "savings", 10**6)


def test_state_is_per_session():
    first, second = tools_by_name("bank"), tools_by_name("bank")
    first["verify_identity"](sc.DEMO_PIN)
    with pytest.raises(ValueError, match="not verified"):
        second["get_balance"]("checking")


def test_weather_unit_defaults_to_fahrenheit_and_can_switch(monkeypatch):
    seen = []

    async def fake(location, fahrenheit=False):
        seen.append(fahrenheit)
        return {}

    monkeypatch.setattr(sc.shared, "current_weather", fake)
    asyncio.run(sc.get_weather("Paris"))
    asyncio.run(sc.get_weather("Tokyo", "celsius"))
    asyncio.run(sc.get_weather("Tokyo", unit="fahrenheit"))
    assert seen == [True, False, True]


def test_livekit_builds_schemas_and_traces_calls():
    pytest.importorskip("livekit.agents")
    from livekit.agents import llm

    from examples.generic_assistant_livekit.tool_wrapper import traced_tool

    events = []
    for s in sc.SCENARIOS.values():
        for fn in s.make_tools():
            tool = traced_tool(fn, lambda kind, **d: events.append((kind, d)))
            schema = llm.utils.build_legacy_openai_schema(tool)
            assert schema["function"]["name"] == fn.__name__
            assert schema["function"]["description"]
    tool = traced_tool(tools_by_name("insurance")["lookup_policy"], lambda kind, **d: events.append((kind, d)))
    events.clear()
    out = asyncio.run(tool(policy_number="AUT-1001"))
    assert json.loads(out)["vehicle"] == "2022 Honda Civic"
    assert [k for k, _ in events] == ["tool_call", "tool_result"] and events[1][1]["ok"]
    events.clear()
    assert asyncio.run(tool(policy_number="bad")).startswith("Error:")
    assert events[1][1]["ok"] is False
