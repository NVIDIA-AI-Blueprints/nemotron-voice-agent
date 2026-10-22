# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from voiceclaw.model_contracts import (
    MODEL_CONTRACT_SCHEMA,
    ModelContractError,
    load_model_contract_catalog,
)

CATALOG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "model_contracts.v1.yaml"


def _without_optional_delivery_templates(tmp_path: Path, *names: str) -> Path:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    templates = raw["profiles"]["default"]["frontend"]["instruction_templates"]
    for name in names:
        templates.pop(name)
    override = tmp_path / "legacy-model-contracts.v1.yaml"
    override.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return override


def test_packaged_catalog_is_versioned_deterministic_and_complete() -> None:
    first = load_model_contract_catalog()
    second = load_model_contract_catalog()

    assert first.schema_version == MODEL_CONTRACT_SCHEMA
    assert first.profile == "default"
    assert first.digest == second.digest
    assert first.digest.startswith("sha256:")
    assert len(first.digest.removeprefix("sha256:")) == 64
    normalized_instructions = " ".join(first.static_instructions.split())
    assert "conversational voice interface" in normalized_instructions
    assert "complete and standalone" in normalized_instructions
    assert "Formulate each delegated goal from the relevant conversation, not merely the latest turn" in (
        normalized_instructions
    )
    assert "Preserve the user's intent and confirmed constraints" in normalized_instructions
    assert "Do not invent facts, requirements, or missing antecedents" in normalized_instructions
    assert "ask one concise clarification" in normalized_instructions
    assert "advertised tool schema and its trusted description" in normalized_instructions
    assert "Do not assume target-specific behavior from this static policy" in normalized_instructions
    assert "supplies routing, session, Work, and correlation identifiers" in normalized_instructions
    acknowledgement = " ".join(first.instruction_templates["task_acknowledgement"].split())
    assert "single conversational assistant" in acknowledgement
    assert "Commit to the user-visible action and intended deliverable as your own" in acknowledgement
    assert "Describe neither a handoff nor the implementation path" in acknowledgement
    assert "State no answer, conclusion, recommendation, finding, progress, timing, or completion" in acknowledgement
    delivery = " ".join(first.instruction_templates["result_delivery"].split())
    assert "authoritative semantic result material" in delivery
    assert "one natural spoken sentence of at most 35 words" in delivery
    assert "Use only facts in payload_text" in delivery
    assert "supplies no result facts" in delivery
    assert "hidden component boundaries" in delivery
    status_delivery = " ".join(first.instruction_templates["status_delivery"].split())
    assert "current local status of target_request" in status_delivery
    assert "Do not guess progress" in status_delivery
    control_delivery = " ".join(first.instruction_templates["control_delivery"].split())
    assert "confirmed local control outcome" in control_delivery
    assert "Do not claim backend cancellation" in control_delivery
    assert "Do not repeat, summarize, explain, answer, or elaborate the goal" not in normalized_instructions
    assert "supplies the exact finalized turn" not in normalized_instructions
    assert "do not narrate rich display content" in normalized_instructions
    assert set(first.instruction_templates) == {
        "task_acknowledgement",
        "result_delivery",
        "status_delivery",
        "control_delivery",
        "server_policy",
        "untrusted_session",
        "untrusted_response",
        "response_context",
        "application_response_turn",
        "dynamic_projection",
        "bootstrap_projection",
    }
    assert first.failure_copy_title == "Request failed"
    assert first.failure_copy("session_busy").title == "Request failed"
    assert first.failure_copy("session_busy").display == "Another request is already active."
    assert first.failure_copy("session_busy").speech == "Another request is already active."


def test_v1_catalog_accepts_absent_local_control_delivery_extensions(tmp_path: Path) -> None:
    catalog = load_model_contract_catalog(
        _without_optional_delivery_templates(tmp_path, "status_delivery", "control_delivery")
    )

    assert catalog.schema_version == MODEL_CONTRACT_SCHEMA
    assert "status_delivery" not in catalog.instruction_templates
    assert "control_delivery" not in catalog.instruction_templates
    with pytest.raises(ModelContractError, match="unknown instruction template: status_delivery"):
        catalog.render_instruction(
            "status_delivery",
            response_context_json="{}",
            status_context_json="{}",
        )


def test_optional_v1_delivery_extension_still_validates_its_placeholders(tmp_path: Path) -> None:
    override = tmp_path / "contracts.yaml"
    override.write_text(
        CATALOG.read_text(encoding="utf-8").replace("${status_context_json}", "${content}", 1),
        encoding="utf-8",
    )

    with pytest.raises(ModelContractError, match="status_delivery placeholders must be"):
        load_model_contract_catalog(override)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("remove_required", "missing task_acknowledgement"),
        ("add_unknown", "unknown future_delivery"),
    ],
)
def test_optional_v1_extensions_do_not_relax_template_key_validation(
    tmp_path: Path,
    change: str,
    expected: str,
) -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    templates = raw["profiles"]["default"]["frontend"]["instruction_templates"]
    if change == "remove_required":
        templates.pop("task_acknowledgement")
    else:
        templates["future_delivery"] = "Unsupported extension."
    override = tmp_path / "contracts.yaml"
    override.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ModelContractError, match=expected):
        load_model_contract_catalog(override)


def test_failure_copy_uses_default_for_unknown_stable_code() -> None:
    catalog = load_model_contract_catalog()

    assert catalog.failure_copy("future_backend_failure") is catalog.default_failure_copy
    assert catalog.failure_copy("future_backend_failure").display == "I couldn't complete that request."


def test_failure_copy_rejects_invalid_lookup_codes() -> None:
    catalog = load_model_contract_catalog()

    for code in ("", "Session-Busy", "contains.dot", "a" * 129):
        with pytest.raises(ModelContractError, match="failure code must match"):
            catalog.failure_copy(code)


def test_failure_copy_catalog_is_deeply_immutable() -> None:
    catalog = load_model_contract_catalog()

    with pytest.raises(TypeError):
        catalog.failure_copy_overrides["new_code"] = catalog.default_failure_copy  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        catalog.default_failure_copy.display = "changed"  # type: ignore[misc]


def test_failure_copy_override_changes_digest_and_both_channels(tmp_path: Path) -> None:
    original = load_model_contract_catalog()
    override = tmp_path / "contracts.yaml"
    override.write_text(
        CATALOG.read_text(encoding="utf-8").replace(
            "display: Another request is already active.\n            speech: Another request is already active.",
            "display: A different request is active.\n            speech: Please wait for the active request.",
            1,
        ),
        encoding="utf-8",
    )

    selected = load_model_contract_catalog(override)

    assert selected.digest != original.digest
    assert selected.failure_copy("session_busy").display == "A different request is active."
    assert selected.failure_copy("session_busy").speech == "Please wait for the active request."


def test_template_values_are_bounded_and_not_recursively_interpreted() -> None:
    catalog = load_model_contract_catalog()
    prompt = catalog.render_result_envelope(
        user_goal_json='"Keep ${display_budget_bytes} literal"',
        result_schema="voiceclaw.result.v1",
        speech_budget_bytes=4096,
        display_budget_bytes=8192,
    )

    assert 'user_goal_json="Keep ${display_budget_bytes} literal"' in prompt
    assert "at most 8192 UTF-8 bytes" in prompt
    with pytest.raises(ModelContractError, match="size limit"):
        catalog.render_instruction("response_context", context_json="x" * (512 * 1024))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda text: text.replace("schema_version:", "schema_version: duplicate\nschema_version:", 1), "duplicate"),
        (lambda text: text + "\nunknown_top_level: true\n", "invalid keys"),
        (
            lambda text: text.replace("${display_budget_bytes}", "${unsupported_placeholder}", 1),
            "placeholders must be",
        ),
        (
            lambda text: text.replace(
                "      failure_copy:\n",
                "      response_templates: {}\n\n      failure_copy:\n",
                1,
            ),
            "unknown response_templates",
        ),
        (lambda text: text.replace("        title: Request failed\n", "", 1), "missing title"),
        (
            lambda text: text.replace(
                "          speech: I couldn't complete that request.\n",
                "          speech: I couldn't complete that request.\n          extra: rejected\n",
                1,
            ),
            "invalid keys",
        ),
        (lambda text: text.replace("          session_busy:\n", "          Session-Busy:\n", 1), "must match"),
        (
            lambda text: text.replace(
                "          speech: I couldn't complete that request.\n",
                '          speech: "contains\\tcontrol"\n',
                1,
            ),
            "control characters",
        ),
        (
            lambda text: text.replace(
                "          speech: I couldn't complete that request.\n",
                "          speech: ${untrusted_placeholder}\n",
                1,
            ),
            "placeholders must be",
        ),
        (
            lambda text: text.replace(
                "          speech: I couldn't complete that request.\n",
                f"          speech: {'x' * 2_001}\n",
                1,
            ),
            "size limit",
        ),
    ],
)
def test_override_catalog_rejects_invalid_content(
    tmp_path: Path,
    mutate,
    expected: str,
) -> None:
    override = tmp_path / "contracts.yaml"
    override.write_text(mutate(CATALOG.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(ModelContractError, match=expected):
        load_model_contract_catalog(override)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="unknown model-contract profile"):
        load_model_contract_catalog(profile="missing")
