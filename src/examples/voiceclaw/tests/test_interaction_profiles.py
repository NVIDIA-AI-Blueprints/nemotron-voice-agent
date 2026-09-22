# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from pathlib import Path

import pytest
import yaml

from voiceclaw.domain.capabilities import CapabilityToolRegistry, ToolProjectionState
from voiceclaw.domain.models import BackendCapabilities, BackendOperation, Durability
from voiceclaw.interaction_profiles import (
    INTERACTION_PROFILE_SCHEMA,
    MAX_DELEGATED_GOAL_BYTES,
    MAX_DELEGATED_GOAL_CHARACTERS,
    ArgumentBinding,
    InteractionOperation,
    InteractionProfileError,
    SessionScope,
    WorkCardinality,
    load_interaction_profile_catalog,
    parse_interaction_profile_catalog,
)

CATALOG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "interaction_profiles.v2.yaml"


def test_packaged_example_profiles_are_versioned_deterministic_and_complete() -> None:
    first = load_interaction_profile_catalog()
    second = load_interaction_profile_catalog()

    assert first.schema_version == INTERACTION_PROFILE_SCHEMA
    assert first.digest == second.digest
    assert first.digest.startswith("sha256:")
    assert len(first.digest.removeprefix("sha256:")) == 64
    assert set(first.profiles) == {"stateless", "single_stateful", "conductor", "specialized"}
    assert first.resolve("stateless").session_scope is SessionScope.NONE
    assert first.resolve("stateless").work_cardinality is WorkCardinality.MANY
    assert first.resolve("single_stateful").work_cardinality is WorkCardinality.SINGLE
    assert first.resolve("conductor").session_scope is SessionScope.ATTACHMENT
    assert first.resolve("specialized").work_cardinality is WorkCardinality.PER_TARGET
    assert all("conversation.respond" in profile.tools for profile in first.profiles.values())


def test_profiles_render_only_closed_canonical_argument_schemas() -> None:
    catalog = load_interaction_profile_catalog()

    direct = catalog.resolve("stateless").tool("conversation.respond")
    status = catalog.resolve("conductor").tool("work.status")
    targeted = catalog.resolve("specialized").tool("work.delegate")

    assert direct.operations == frozenset()
    assert direct.render_input_schema() == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert status.argument_binding is ArgumentBinding.WORK_ID_OPTIONAL
    assert status.arguments == ("work_id",)
    assert status.required_arguments == ()
    assert targeted.argument_binding is ArgumentBinding.TARGET_GOAL_REQUIRED
    assert targeted.required_arguments == ("agent", "goal")
    assert targeted.render_input_schema(enum_values={"agent": ("finance", "procurement")}) == {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Stable configured target identifier.",
                "enum": ["finance", "procurement"],
            },
            "goal": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_DELEGATED_GOAL_CHARACTERS,
                "description": (
                    "Complete standalone objective for this target reconstructed from relevant conversation, "
                    "constraints, and artifact references without invented facts."
                ),
            },
        },
        "required": ["agent", "goal"],
        "additionalProperties": False,
    }
    assert MAX_DELEGATED_GOAL_CHARACTERS * 4 == MAX_DELEGATED_GOAL_BYTES


def test_single_stateful_separates_work_commands_from_agent_query_answers() -> None:
    profile = load_interaction_profile_catalog().resolve("single_stateful")

    assert profile.tool("work.delegate").operations == frozenset(
        {InteractionOperation.SUBMIT, InteractionOperation.STEER}
    )
    assert profile.tool("work.delegate").argument_binding is ArgumentBinding.GOAL_REQUIRED
    assert profile.tool("work.answer_agent").operations == frozenset(
        {InteractionOperation.ANSWER_QUERY, InteractionOperation.RESPOND_PERMISSION}
    )
    assert profile.tool("work.answer_agent").argument_binding is ArgumentBinding.QUERY_ID_REQUIRED


def test_profile_operations_only_intersect_authoritative_backend_evidence() -> None:
    delegate = load_interaction_profile_catalog().resolve("conductor").tool("work.delegate")

    assert delegate.operations == frozenset({InteractionOperation.SUBMIT, InteractionOperation.STEER})
    assert delegate.authorized_operations({"work.submit", "work.cancel"}) == frozenset({InteractionOperation.SUBMIT})


def test_per_backend_override_changes_only_prose_and_leaves_catalog_immutable() -> None:
    catalog = load_interaction_profile_catalog()
    original = catalog.resolve("specialized")
    overrides = {
        "conversation.respond": {"description": "Use this route for an immediate local response."},
        "work.delegate": {
            "description": "Route substantial Work to one configured target.",
            "property_descriptions": {"agent": "Configured target key."},
        },
    }

    overridden = catalog.resolve(
        "specialized",
        prose_overrides=overrides,
    )
    overrides["work.delegate"]["property_descriptions"]["agent"] = "Mutated caller copy."

    assert overridden.tool("conversation.respond").description == "Use this route for an immediate local response."
    assert overridden.tool("work.delegate").description == "Route substantial Work to one configured target."
    assert overridden.tool("work.delegate").property_descriptions["agent"] == "Configured target key."
    assert (
        overridden.tool("work.delegate").property_descriptions["goal"]
        == (original.tool("work.delegate").property_descriptions["goal"])
    )
    assert overridden.tool("work.delegate").operations == original.tool("work.delegate").operations
    assert overridden.tool("work.delegate").argument_binding is original.tool("work.delegate").argument_binding
    assert overridden.tool("work.delegate").render_input_schema()["required"] == ["agent", "goal"]
    assert overridden.resolved_digest != original.resolved_digest
    assert catalog.digest == load_interaction_profile_catalog().digest
    assert catalog.resolve("specialized") is original
    assert original.tool("conversation.respond").description != overridden.tool("conversation.respond").description
    with pytest.raises(TypeError):
        overridden.tools["work.delegate"] = original.tool("work.delegate")  # type: ignore[index]
    with pytest.raises(TypeError):
        overridden.tool("work.delegate").property_descriptions["agent"] = "Mutation"  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"work.erase": {"description": "Erase Work."}}, "unknown tools"),
        ({"work.delegate": {"operations": ["work.cancel"]}}, "unknown keys"),
        (
            {"work.delegate": {"property_descriptions": {"backend_hint": "Unsafe routing hint."}}},
            "unknown arguments",
        ),
        ({"conversation.respond": {"property_descriptions": {}}}, "must not be empty"),
    ],
)
def test_prose_override_rejects_semantics_and_arbitrary_names(overrides: object, message: str) -> None:
    profile = load_interaction_profile_catalog().resolve("specialized")

    with pytest.raises(InteractionProfileError, match=message):
        profile.with_prose_overrides(overrides)  # type: ignore[arg-type]


def test_operator_catalog_may_change_prose_but_not_normalized_semantics(tmp_path: Path) -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    raw["profiles"]["specialized"]["tools"]["work.delegate"]["description"] = "Backend-specific safe copy."
    raw["profiles"]["specialized"]["tools"]["work.delegate"]["property_descriptions"]["agent"] = (
        "Backend-specific target label."
    )
    override = tmp_path / "interaction-profiles.yaml"
    override.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    profile = load_interaction_profile_catalog(override).resolve("specialized")

    assert profile.tool("work.delegate").description == "Backend-specific safe copy."
    assert profile.tool("work.delegate").property_descriptions["agent"] == "Backend-specific target label."
    assert profile.tool("work.delegate").argument_binding is ArgumentBinding.TARGET_GOAL_REQUIRED


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["profiles"]["specialized"].__setitem__("session_scope", "attachment"),
            "incompatible session_scope=attachment and work_cardinality=per_target",
        ),
        (
            lambda raw: raw["profiles"]["single_stateful"].__setitem__("work_cardinality", "many"),
            "work.cancel must use operations",
        ),
        (
            lambda raw: raw["profiles"]["stateless"]["tools"]["work.delegate"].__setitem__(
                "operations", ["work.cancel"]
            ),
            "work.delegate must use operations",
        ),
        (
            lambda raw: raw["profiles"]["conductor"]["tools"]["work.status"].__setitem__(
                "argument_binding", "work_id_required"
            ),
            "work.status must use operations",
        ),
        (
            lambda raw: raw["profiles"]["stateless"]["tools"].__setitem__(
                "work.erase", raw["profiles"]["stateless"]["tools"]["work.cancel"]
            ),
            "unsupported canonical tools: work.erase",
        ),
        (
            lambda raw: raw["profiles"]["specialized"]["tools"]["work.delegate"]["property_descriptions"].__setitem__(
                "backend_hint", "Unsafe routing hint."
            ),
            "unknown backend_hint",
        ),
    ],
)
def test_catalog_rejects_invalid_combinations_and_noncanonical_shapes(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    mutate(raw)
    override = tmp_path / "invalid-interaction-profiles.yaml"
    override.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InteractionProfileError, match=message):
        load_interaction_profile_catalog(override)


def test_custom_profile_name_subset_copy_and_catalog_order_load_and_project() -> None:
    catalog = parse_interaction_profile_catalog(
        {
            "schema_version": INTERACTION_PROFILE_SCHEMA,
            "profiles": {
                "research_helper": {
                    "session_scope": "attachment",
                    "work_cardinality": "many",
                    "tools": {
                        "work.status": {
                            "operations": ["work.status"],
                            "argument_binding": "work_id_optional",
                            "description": "Check the research target's authoritative Work state.",
                            "property_descriptions": {
                                "work_id": "Backend-issued research Work identifier.",
                            },
                        },
                        "conversation.respond": {
                            "operations": [],
                            "argument_binding": "none",
                            "description": "Use a short direct conversational response.",
                            "property_descriptions": {},
                        },
                        "work.delegate": {
                            "operations": ["work.submit", "work.steer"],
                            "argument_binding": "goal_required",
                            "description": "Send substantial research to the configured target.",
                            "property_descriptions": {"goal": "Standalone research objective."},
                        },
                    },
                }
            },
        }
    )
    profile = catalog.resolve("research_helper")
    registry = CapabilityToolRegistry(profile=profile)
    capabilities = BackendCapabilities(
        backend_kind="research",
        target_label="Research target",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.STATUS}),
        durability=Durability.SESSION,
        sessionful=True,
    )

    tools = registry.project(capabilities, state=ToolProjectionState())

    assert profile.name == "research_helper"
    assert tuple(profile.tools) == ("work.status", "conversation.respond", "work.delegate")
    assert [tool.name for tool in tools] == ["work.delegate", "work.status"]
    assert tools[1].description == "Check the research target's authoritative Work state."
    assert tools[1].input_schema["properties"]["work_id"]["description"] == ("Backend-issued research Work identifier.")


def test_model_visible_tool_order_is_canonical_when_yaml_order_changes() -> None:
    raw = {
        "schema_version": INTERACTION_PROFILE_SCHEMA,
        "profiles": {
            "ordered_profile": {
                "session_scope": "attachment",
                "work_cardinality": "many",
                "tools": {
                    "work.status": {
                        "operations": ["work.status"],
                        "argument_binding": "work_id_optional",
                        "description": "Read Work state.",
                        "property_descriptions": {"work_id": "Backend-issued Work identifier."},
                    },
                    "conversation.respond": {
                        "operations": [],
                        "argument_binding": "none",
                        "description": "Respond directly.",
                        "property_descriptions": {},
                    },
                    "work.delegate": {
                        "operations": ["work.submit", "work.steer"],
                        "argument_binding": "goal_required",
                        "description": "Delegate Work.",
                        "property_descriptions": {"goal": "Standalone objective."},
                    },
                },
            }
        },
    }
    reordered = yaml.safe_load(yaml.safe_dump(raw, sort_keys=True))
    first = parse_interaction_profile_catalog(raw)
    second = parse_interaction_profile_catalog(reordered)
    capabilities = BackendCapabilities(
        backend_kind="ordered",
        target_label="Ordered target",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.STATUS}),
        durability=Durability.SESSION,
        sessionful=True,
    )

    first_tools = CapabilityToolRegistry(profile=first.resolve("ordered_profile")).project(
        capabilities,
        state=ToolProjectionState(),
    )
    second_tools = CapabilityToolRegistry(profile=second.resolve("ordered_profile")).project(
        capabilities,
        state=ToolProjectionState(),
    )

    assert first.digest == second.digest
    assert first.resolve("ordered_profile").resolved_digest == second.resolve("ordered_profile").resolved_digest
    assert [tool.name for tool in first_tools] == ["work.delegate", "work.status"]
    assert [tool.name for tool in second_tools] == ["work.delegate", "work.status"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda profile: profile["tools"].pop("conversation.respond"),
            "must define conversation.respond",
        ),
        (
            lambda profile: profile["tools"]["work.delegate"].__setitem__("operations", ["work.submit", "work.cancel"]),
            "work.delegate must use operations",
        ),
        (
            lambda profile: (
                profile["tools"]["work.delegate"].__setitem__("argument_binding", "work_id_required"),
                profile["tools"]["work.delegate"].__setitem__(
                    "property_descriptions", {"work_id": "Unsafe model-selected Work identifier."}
                ),
            ),
            "argument_binding=goal_required",
        ),
    ],
)
def test_custom_profiles_fail_closed_on_invalid_normalized_semantics(mutate, message: str) -> None:
    profile = {
        "session_scope": "none",
        "work_cardinality": "many",
        "tools": {
            "conversation.respond": {
                "operations": [],
                "argument_binding": "none",
                "description": "Respond directly.",
                "property_descriptions": {},
            },
            "work.delegate": {
                "operations": ["work.submit"],
                "argument_binding": "goal_required",
                "description": "Delegate Work.",
                "property_descriptions": {"goal": "Standalone objective."},
            },
        },
    }
    mutate(profile)

    with pytest.raises(InteractionProfileError, match=message):
        parse_interaction_profile_catalog(
            {
                "schema_version": INTERACTION_PROFILE_SCHEMA,
                "profiles": {"custom_profile": profile},
            }
        )


def test_catalog_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    text = CATALOG.read_text(encoding="utf-8")
    override = tmp_path / "duplicate-interaction-profiles.yaml"
    duplicated = text.replace("schema_version:", "schema_version: duplicate\nschema_version:", 1)
    override.write_text(duplicated, encoding="utf-8")

    with pytest.raises(InteractionProfileError, match="duplicate"):
        load_interaction_profile_catalog(override)


def test_schema_narrowing_rejects_unknown_or_malformed_values() -> None:
    tool = load_interaction_profile_catalog().resolve("specialized").tool("work.delegate")

    with pytest.raises(InteractionProfileError, match="unknown arguments"):
        tool.render_input_schema(enum_values={"backend_hint": ("x",)})
    with pytest.raises(InteractionProfileError, match="cannot be narrowed"):
        tool.render_input_schema(enum_values={"goal": ("x",)})
    with pytest.raises(InteractionProfileError, match="unique bounded strings"):
        tool.render_input_schema(enum_values={"agent": ("finance", "finance")})
