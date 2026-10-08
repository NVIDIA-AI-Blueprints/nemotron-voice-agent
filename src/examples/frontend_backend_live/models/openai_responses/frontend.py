# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI talker: the Responses API with a strict JSON-schema decision."""

from __future__ import annotations

from openai import AsyncOpenAI

from ...config_manager.schema import ModelEndpoint, ReliabilityConfig
from ..base import Frontend
from ..decision import TalkerDecision, decision_from_output_text, decision_schema, unusable_decision
from ..registry import FRONTENDS


def parse_talker_response(response) -> TalkerDecision:
    """Convert a Responses result to a decision.

    Never interprets partial JSON or a refusal as a business routing decision. A completed refusal is
    spoken directly; empty, malformed or incomplete decisions delegate. Diagnostics keep shape and
    status, not raw model text.
    """
    metadata = {
        "response_id": response.id,
        "response_status": response.status,
        "incomplete_reason": getattr(response.incomplete_details, "reason", None),
        "output_types": [item.type for item in response.output],
        "output_text_chars": len(response.output_text),
    }
    if response.status == "completed":
        refusals = [
            content.refusal
            for item in response.output
            if item.type == "message"
            for content in item.content
            if content.type == "refusal"
        ]
        if refusals:
            return TalkerDecision("speak", " ".join(refusals), {**metadata, "recovery": "refusal"})
        return decision_from_output_text(response.output_text, metadata)
    if response.status == "incomplete":
        return unusable_decision(metadata, "incomplete_output")
    raise RuntimeError(f"Talker response did not complete: {response.status}")


@FRONTENDS.register("openai-responses")
class OpenAIResponsesFrontend(Frontend):
    """Talker on the OpenAI Responses API."""

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig, client: AsyncOpenAI | None = None):
        """Create the client unless one is injected."""
        super().__init__(endpoint, reliability)
        self.client = client or AsyncOpenAI(
            api_key=endpoint.api_key or "not-needed",
            base_url=endpoint.base_url,
            timeout=reliability.api_timeout_seconds,
            max_retries=0,
        )

    async def complete_once(self, instructions, history, actions, max_output_tokens):
        """Make one Responses call with reasoning off and a strict decision schema."""
        response = await self.client.responses.create(
            model=self.endpoint.model,
            instructions=instructions,
            input=history,
            reasoning={"effort": "none"},
            max_output_tokens=max_output_tokens,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "voice_decision",
                    "strict": True,
                    "schema": decision_schema(actions),
                }
            },
        )
        return parse_talker_response(response)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()
