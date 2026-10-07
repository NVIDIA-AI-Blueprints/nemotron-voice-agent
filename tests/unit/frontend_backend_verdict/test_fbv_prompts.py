# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: shipped prompts carry no target-domain assumptions (regression guard).

This is a guard, not proof: a fixed noun list cannot show that every domain
assumption is absent. Human prompt review is the actual gate (plan section 11).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CATALOG = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "examples"
    / "frontend_backend_verdict"
    / "text"
    / "config"
    / "prompts.yaml"
)

EXAMPLE_DOMAINS = {
    "library": ("books?", "library"),
    "fitness": ("gym", "fitness class"),
    "delivery": ("parcels?",),
    "dining": ("reservation", "dinner"),
    "utility": ("meter", "electricity"),
    "it_support": ("laptop", "hardware"),
}


def _mentions(text: str, nouns: tuple[str, ...]) -> bool:
    """Whole-word match, so 'parameter' never counts as 'meter'."""
    return any(re.search(rf"\b{noun}\b", text, re.IGNORECASE) for noun in nouns)


BENCHMARK_DOMAINS = ("airline", "flight", "retail", "telecom", "banking")


def _catalog() -> dict[str, str]:
    data = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    return {key: item["content"] for key, item in data.items()}


def test_catalog_has_both_prompts() -> None:
    assert {"frontend", "backend"} <= set(_catalog())


def test_frontend_examples_span_at_least_five_domains() -> None:
    _, examples = _split_frontend()
    present = {name for name, nouns in EXAMPLE_DOMAINS.items() if _mentions(examples, nouns)}
    assert len(present) >= 5, f"only {sorted(present)} present"


def test_no_example_domain_appears_twice_as_a_subject() -> None:
    _, examples = _split_frontend()
    blocks = [block for block in examples.split('User: "') if block.strip()]
    subjects = []
    for block in blocks:
        matched = [name for name, nouns in EXAMPLE_DOMAINS.items() if _mentions(block, nouns)]
        subjects.extend(matched)
    assert len(subjects) == len(set(subjects)), f"repeated example domain in {subjects}"


def test_prompt_body_carries_no_domain_nouns() -> None:
    body, _ = _split_frontend()
    offenders = [name for name, nouns in EXAMPLE_DOMAINS.items() if _mentions(body, nouns)]
    assert offenders == [], f"domain nouns leaked into the prompt body: {offenders}"


def test_no_benchmark_domain_anywhere() -> None:
    catalog = _catalog()
    for key, content in catalog.items():
        lowered = content.lower()
        offenders = [name for name in BENCHMARK_DOMAINS if re.search(rf"\b{name}", lowered)]
        assert offenders == [], f"benchmark domain {offenders} appears in the {key} prompt"


def test_placeholders_are_present() -> None:
    catalog = _catalog()
    for placeholder in ("{persona}", "{capabilities}", "{unsupported_reply}", "{agent_name}"):
        assert placeholder in catalog["frontend"]
    assert "{domain_policy}" in catalog["backend"]


def test_backend_prompt_returns_final_text_verbatim() -> None:
    backend = _catalog()["backend"].lower()
    assert "verbatim" in backend
    assert "tool" in backend


def _split_frontend() -> tuple[str, str]:
    content = _catalog()["frontend"]
    marker = "\nExamples:\n"
    index = content.index(marker)
    return content[:index], content[index:]
