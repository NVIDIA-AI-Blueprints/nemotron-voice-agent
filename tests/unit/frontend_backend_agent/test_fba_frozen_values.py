# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, E501

"""The registry and catalog values that choose an eval session's LLMs and speech are frozen.

Plan ``misc/frontend-backend-verdict/fba-removal-plan.md``, section 2.2.1b. A Realtime session's
``registry-default`` selectors resolve to the example's registry ``defaults``, and the route fills
the model and endpoints from these catalog entries (``src/server.py`` ``_sanitize_realtime_config``),
which override the ``eval`` profile. The files may gain entries (after these); the values recorded
here at ``84affa7`` never change.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
FBA = ROOT / "src" / "examples" / "frontend_backend_agent"
MODEL = "nvidia/nemotron-realtime-frontend-backend"
EXAMPLE = "frontend-backend-agent"

REALTIME_PROFILE = {
    "default": True,
    "label": "Nemotron Realtime Frontend/Backend Agent",
    "pipeline_mode": "frontend-backend-agent",
    "selectors": {
        "asr_id": "registry-default",
        "llm_id": "registry-default",
        "prompt_key": "frontend",
        "thinker_llm_id": "registry-default",
        "tts_id": "registry-default",
    },
}

EXAMPLE_FIELDS = {
    "agent_prompt_keys": [
        "backend",
        "cascade_voice_addendum",
        "backend_history_request",
        "backend_history_context_full",
        "backend_history_context_backend_turns",
        "backend_history_context_transcript",
        "backend_history_guidance_full",
        "backend_history_guidance_backend_turns",
        "backend_history_guidance_transcript",
        "identifier_note_voice",
        "tool_argument_invalid",
        "tool_argument_invalid_readback",
        "tool_argument_invalid_spell_all",
        "tool_call_already_failed",
        "lookup_not_found_hint",
        "lookup_not_found_hint_words",
        "frontend_task_in_progress",
    ],
    "bot": "examples.frontend_backend_agent.pipeline:bot",
    "defaults": {
        "asr": ["nemotron-asr-streaming-english"],
        "llm": ["nemotron-3.5-lightning"],
        "prompt": ["frontend"],
        "thinker-llm": ["nemotron-3-ultra"],
        "tts": ["magpie-multilingual-tts"],
    },
    "slots": ["llm", "thinker-llm", "asr", "tts"],
}

#: Per platform and slot: the first entry's id and its fields.
CATALOGS = {
    "cloud": {
        "asr": {
            "entry": {
                "function_id": "4155ae85-73e1-4936-b47f-87b9de165651",
                "model": "nemotron-asr-streaming",
                "name": "Nemotron ASR Streaming English",
                "server": "grpc.nvcf.nvidia.com:443",
            },
            "first": "nemotron-asr-streaming-english",
        },
        "llm": {
            "entry": {
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}',
                "max_tokens": 1024,
                "model_id": "nvidia/nvidia/nemotron-3.5-lightning",
                "name": "Nemotron 3.5 Lightning frontend (Inference Hub, reasoning off)",
                "system_prompt": "",
                "temperature": 0.0,
            },
            "first": "nemotron-3.5-lightning",
        },
        "thinker-llm": {
            "entry": {
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":1024}}',
                "max_tokens": 4096,
                "model_id": "nvidia/nvidia/nemotron-3-ultra",
                "name": "Nemotron 3 Ultra backend (Inference Hub, reasoning on)",
                "temperature": 0.0,
            },
            "first": "nemotron-3-ultra",
        },
        "tts": {
            "entry": {
                "function_id": "500bfea0-ba3d-4158-8276-1d04daedfdcd",
                "model": "magpie-tts-multilingual",
                "name": "Magpie TTS Multilingual",
                "server": "grpc.nvcf.nvidia.com:443",
                "synthesis_mode": "stitched",
                "voice_id": "Magpie-Multilingual.EN-US.Aria",
            },
            "first": "magpie-multilingual-tts",
        },
    },
    "server": {
        "asr": {
            "entry": {
                "function_id": "",
                "model": "cache-aware-parakeet-rnnt-multi-asr-streaming-sortformer",
                "name": "Nemotron ASR Streaming English",
                "server": "nemotron-asr-streaming-english:50052",
            },
            "first": "nemotron-asr-streaming-english",
        },
        "llm": {
            "entry": {
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}',
                "max_tokens": 1024,
                "model_id": "nvidia/nvidia/nemotron-3.5-lightning",
                "name": "Nemotron 3.5 Lightning frontend (Inference Hub, reasoning off)",
                "system_prompt": "",
                "temperature": 0.0,
            },
            "first": "nemotron-3.5-lightning",
        },
        "thinker-llm": {
            "entry": {
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":1024}}',
                "max_tokens": 4096,
                "model_id": "nvidia/nvidia/nemotron-3-ultra",
                "name": "Nemotron 3 Ultra backend (Inference Hub, reasoning on)",
                "temperature": 0.0,
            },
            "first": "nemotron-3-ultra",
        },
        "tts": {
            "entry": {
                "function_id": "",
                "model": "magpie-tts-multilingual",
                "name": "Magpie TTS",
                "server": "magpie-multilingual-tts-service:50051",
                "synthesis_mode": "stitched",
                "voice_id": "Magpie-Multilingual.EN-US.Aria",
            },
            "first": "magpie-multilingual-tts",
        },
    },
    "singlegpu": {
        "asr": {
            "entry": {
                "function_id": "",
                "model": "nemotron-speech-streaming-en-0.6b",
                "name": "Nemotron Speech Streaming",
                "server": "nemo-speech:50051",
            },
            "first": "nemotron-asr-streaming-english",
        },
        "llm": {
            "entry": {
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}',
                "max_tokens": 1024,
                "model_id": "nvidia/nvidia/nemotron-3.5-lightning",
                "name": "Nemotron 3.5 Lightning frontend (Inference Hub, reasoning off)",
                "system_prompt": "",
                "temperature": 0.0,
            },
            "first": "nemotron-3.5-lightning",
        },
        "thinker-llm": {
            "entry": {
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":1024}}',
                "max_tokens": 4096,
                "model_id": "nvidia/nvidia/nemotron-3-ultra",
                "name": "Nemotron 3 Ultra backend (Inference Hub, reasoning on)",
                "temperature": 0.0,
            },
            "first": "nemotron-3-ultra",
        },
        "tts": {
            "entry": {
                "function_id": "",
                "model": "magpie-tts-multilingual",
                "name": "Magpie TTS (NeMo-Speech.cpp)",
                "server": "nemo-speech:50051",
                "synthesis_mode": "stitched",
                "voice_id": "John",
            },
            "first": "magpie-multilingual-tts",
        },
    },
}


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _catalogs() -> dict[str, dict[str, dict[str, Any]]]:
    local = _load(FBA / "services.local.yaml")
    return {"cloud": _load(FBA / "services.cloud.yaml"), "server": local["server"], "singlegpu": local["singlegpu"]}


class FrozenValuesTests(unittest.TestCase):
    def test_realtime_model_profile(self) -> None:
        registry = _load(ROOT / "examples_registry.yaml")
        self.assertEqual(registry["realtime_models"][MODEL], REALTIME_PROFILE)

    def test_example_slots_defaults_prompts_and_bot(self) -> None:
        example = _load(ROOT / "examples_registry.yaml")["examples"][EXAMPLE]
        self.assertEqual({key: example[key] for key in EXAMPLE_FIELDS}, EXAMPLE_FIELDS)

    def test_catalog_entries_are_unchanged_and_first(self) -> None:
        catalogs = _catalogs()
        for platform, slots in CATALOGS.items():
            for slot, pinned in slots.items():
                with self.subTest(platform=platform, slot=slot):
                    entries = catalogs[platform][slot]
                    self.assertEqual(next(iter(entries)), pinned["first"])
                    self.assertEqual(entries[pinned["first"]], pinned["entry"])


if __name__ == "__main__":
    unittest.main()
