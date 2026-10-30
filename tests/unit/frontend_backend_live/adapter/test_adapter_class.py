# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The frontend adapter class contract and readiness."""

import asyncio
import json
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_live.adapter import readiness as readiness_module
from examples.frontend_backend_live.adapter.adapter import (
    LIVE_AUDIO_IN_RATE,
    CascadeAdapter,
    CascadeAdapterSettings,
)
from examples.frontend_backend_live.adapter.contract import FrontendAdapterClass, ServiceStarting
from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from examples.frontend_backend_live.adapter.readiness import EndpointReadiness
from examples.frontend_backend_live.adapter.registry import FRONTEND_ADAPTERS
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, ConfigError, load_live_config
from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
from examples.frontend_backend_live.media.services import PerSentence, Stitched
from examples.frontend_backend_live.media.turns import FinalizeAtVerdictUserTurnStopStrategy, Shared, Silence, SmartTurn
from examples.frontend_backend_live.models import FRONTENDS
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts.catalog import CATALOG_PATH, CHANNEL_PROMPT_KEY, catalog_prompt
from examples.frontend_backend_live.tables import pick
from live.ports import EXCHANGE_RATE, FrontendAdapter
from live.protocol import SessionConfig
from tests.unit.cascade_helpers import LIVE_CONFIG as CASCADE_TEST_CONFIG
from tests.unit.frontend_backend_live.adapter.adapter_helpers import RecordingPort
from utils import NVCF_GRPC_SERVER, NVCF_LLM_BASE_URL, load_yaml_file

# The shipped frontend.cascade block, as the root hands it to the class.
SHIPPED_BLOCK = dict(load_live_config(CONFIG_PATH).blocks["cascade"])
# The example's own catalog reader, with one talker model: its prompts are the shipped prompts.yaml.
SHIPPED_CATALOG = ExampleCatalog({"llm": [{"model_id": "org/front", "base_url": "http://front/v1"}]})
# The end-to-end tests' configuration, with the talker's prompt named as config.yaml names it.
TALKER_CONFIG = replace(CASCADE_TEST_CONFIG, prompt="voice")
TALKER_SETTINGS = CascadeAdapterSettings(live=TALKER_CONFIG, turn_end=SmartTurn(TALKER_CONFIG.speech))
LOCAL = {"asr": "nemotron-asr:50052", "tts": "magpie-tts:50051", "base_url": "http://nvidia-llm:8000/v1"}
CLOUD = {"asr": NVCF_GRPC_SERVER, "tts": NVCF_GRPC_SERVER, "base_url": NVCF_LLM_BASE_URL}


def members(protocol) -> set[str]:
    return {name for name in vars(protocol) if not name.startswith("_")} | set(protocol.__annotations__)


class FakeCatalog:
    """The example's catalog as the root reads it once: models by slot, services by slot, prompts by key."""

    def __init__(self, recipe=LOCAL):
        self.recipe = recipe
        self.asked: list[tuple] = []

    def model(self, slot, overrides=None):
        self.asked.append(("model", slot, dict(overrides or {})))
        return ModelEndpoint(model=f"model-of-{slot}", base_url=self.recipe["base_url"], api_key="nvapi-secret")

    def service(self, slot):
        self.asked.append(("service", slot))
        return {"server": self.recipe[slot], "voice_id": "Magpie-Multilingual.EN-US.Aria"}

    def prompt(self, key):
        return f"The {key} prompt."


def session(**changes) -> SessionConfig:
    return SessionConfig(model="live-1", **changes)


# ------------------------------------------------------------------------------------- the table and contract
async def build_as_the_root_does(table, name, block, catalog, config):
    """The root's four steps, with no branch on the adapter's name."""
    cls = pick(table, name, "frontend adapter")
    settings = cls.Settings.parse(block)
    await cls.readiness(settings, catalog).ensure_ready()
    return cls.from_config(config, "websocket", None, settings, catalog)


class ContractTests(unittest.IsolatedAsyncioTestCase):
    def test_the_table_names_the_cascade_and_every_entry_keeps_the_class_contract(self):
        self.assertEqual(FRONTEND_ADAPTERS, {"cascade": CascadeAdapter})
        for name, cls in FRONTEND_ADAPTERS.items():
            with self.subTest(name):
                self.assertLessEqual(members(FrontendAdapterClass), set(dir(cls)))
                self.assertLessEqual(members(FrontendAdapter), set(dir(cls)))
        self.assertIs(CascadeAdapter.Settings, CascadeAdapterSettings)
        self.assertTrue(CascadeAdapter.hosts_rtvi)

    async def test_the_cascade_builds_through_the_same_four_steps(self):
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt") as stt,
            patch("examples.frontend_backend_live.adapter.adapter.build_tts") as tts,
            patch.object(readiness_module, "is_endpoint_reachable", return_value=True),
        ):
            adapter = await build_as_the_root_does(
                FRONTEND_ADAPTERS, "cascade", SHIPPED_BLOCK, FakeCatalog(), session()
            )
        self.assertIsInstance(adapter, CascadeAdapter)
        self.assertEqual((adapter.stt, adapter.tts), (stt.return_value, tts.return_value))


# ---------------------------------------------------------------------------------------------- settings
class SettingsTests(unittest.TestCase):
    def test_the_adapter_reads_its_own_speech_key(self):
        self.assertEqual(
            CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": {"tts_mode": "stitched"}}).tts_mode, "stitched"
        )
        self.assertEqual(CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": None}).tts_mode, "per_sentence")

    def test_the_settings_build_the_turn_end_the_speech_block_names(self):
        for name, cls in {"smart_turn": SmartTurn, "silence": Silence, "shared": Shared}.items():
            with self.subTest(name):
                settings = CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": {"turn_end": name}})
                self.assertIs(type(settings.turn_end), cls)

    def test_a_bad_speech_block_is_refused_with_a_message_that_names_it(self):
        cases = {
            "frontend.cascade.speech has unknown keys: tts_rate": {"tts_rate": 22050},
            "Unknown speech.tts_mode 'streaming'; known: per_sentence, stitched": {"tts_mode": "streaming"},
            "Unknown speech.turn_end 'vad'; known: shared, silence, smart_turn": {"turn_end": "vad"},
            "speech.vad_stop_seconds must be below speech.turn_silence_seconds: Smart Turn judges a pause before the "
            "silence that ends the turn anyway": {"vad_stop_seconds": 1.2},  # as long as the default silence
            "speech.vad_start_seconds must not be above speech.vad_stop_seconds, the VAD stop it is paired with": {
                "vad_start_seconds": 0.3
            },
            "speech.vad_start_seconds must not be above speech.turn_silence_seconds, the VAD stop it is paired with": {
                "turn_end": "silence",
                "vad_start_seconds": 1.5,
            },
            "frontend.cascade.speech must be a mapping": ["per_sentence"],
        }
        for message, speech in cases.items():
            with self.subTest(message), self.assertRaises(ValueError) as refused:
                CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": speech})
            self.assertEqual(str(refused.exception), message)

    def test_every_other_key_is_checked_by_the_frontends_settings(self):
        with self.assertRaisesRegex(ConfigError, "frontend.cascade has unknown keys: spech"):
            CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "spech": {}})
        with self.assertRaisesRegex(ConfigError, "must be a mapping"):
            CascadeAdapterSettings.parse([])

    def test_the_cascades_own_rules_are_checked_when_the_settings_are_parsed(self):
        with self.assertRaisesRegex(ValueError, "Unknown result wording 'poetic'"):
            CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "delegation": {"result_wording": "poetic"}})

    def test_a_provider_the_table_does_not_name_is_refused_when_the_settings_are_read(self):
        # The parked Realtime frontend is in no table, so the read refuses it as it refuses a typo.
        with self.assertRaises(ValueError) as refused:
            CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "provider": "realtime"})
        known = ", ".join(sorted(FRONTENDS))
        self.assertEqual(str(refused.exception), f"Unknown frontend.cascade.provider 'realtime'; known: {known}")
        for provider in FRONTENDS:
            with self.subTest(provider):
                self.assertEqual(
                    CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "provider": provider}).live.provider, provider
                )


# -------------------------------------------------------------------------------------------- from_config
class FromConfigTests(unittest.TestCase):
    def build(self, config=None, tts_mode="per_sentence", catalog=None):
        settings = replace(TALKER_SETTINGS, tts_mode=tts_mode)
        catalog = catalog or FakeCatalog()
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt") as stt,
            patch("examples.frontend_backend_live.adapter.adapter.build_tts") as tts,
        ):
            adapter = CascadeAdapter.from_config(config or session(), "websocket", None, settings, catalog)
        return adapter, catalog, stt, tts

    def test_the_adapter_builds_its_talker_and_its_own_speech_services_and_connects_nothing(self):
        config = session(instructions="Be brief.", audio={"output": {"voice": {"id": "Custom.Voice"}}})
        adapter, catalog, stt, tts = self.build(config)
        voice = {"tts_voice_id": "Custom.Voice"}
        self.assertEqual(adapter.frontend_endpoint.model, "model-of-llm")
        self.assertIn(("model", "llm", voice), catalog.asked)
        self.assertEqual(adapter.instructions, "Be brief.")
        stt.assert_called_once_with(voice, catalog.service("asr"), TALKER_SETTINGS.turn_end)
        (stop,) = adapter.user_params.user_turn_strategies.stop  # the same turn end's turn detection
        self.assertIsInstance(stop, FinalizeAtVerdictUserTurnStopStrategy)
        self.assertEqual(adapter.user_params.user_mute_strategies, [])  # a live session has no welcome to wait for
        tts.assert_called_once_with({**voice, "tts_synthesis_mode": "per_sentence"}, catalog.service("tts"))
        self.assertIsInstance(adapter.speech, PerSentence)
        self.assertEqual((adapter.transport, adapter.live_config), ("websocket", TALKER_CONFIG))
        self.assertIsNone(adapter.cascade)  # nothing is built or connected before start
        self.assertIsNone(adapter.task)

    def test_the_turn_end_builds_the_speech_to_text_and_the_user_aggregators_settings_together(self):
        settings = CascadeAdapterSettings.parse(
            {**SHIPPED_BLOCK, "speech": {"turn_end": "silence", "turn_silence_seconds": 0.8}}
        )
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt") as stt,
            patch("examples.frontend_backend_live.adapter.adapter.build_tts"),
        ):
            adapter = CascadeAdapter.from_config(session(), "websocket", None, settings, FakeCatalog())
        self.assertIs(stt.call_args.args[2], settings.turn_end)
        self.assertIsInstance(settings.turn_end, Silence)
        self.assertEqual(adapter.user_params.vad_analyzer.params.stop_secs, 0.8)

    def test_the_tts_mode_sets_the_synthesis_mode_and_the_text_hold(self):
        adapter, _, _, tts = self.build(tts_mode="stitched")
        self.assertEqual(tts.call_args.args[0], {"tts_synthesis_mode": "stitched"})
        self.assertIsInstance(adapter.speech, Stitched)

    def test_a_session_without_instructions_or_with_blank_ones_gets_the_catalogs_prompt(self):
        for blank in (None, "", " ", "\n\t \n"):
            with self.subTest(instructions=blank):
                adapter, catalog, _, _ = self.build(session(instructions=blank))
                self.assertEqual(adapter.instructions, catalog.prompt("voice"))
        adapter, _, _, _ = self.build(session(instructions="  Be brief.\n"))
        self.assertEqual(adapter.instructions, "  Be brief.\n")  # the client's own text, word for word

    def test_the_shipped_default_is_the_domain_free_voice_entry_as_written(self):
        settings = CascadeAdapterSettings.parse(SHIPPED_BLOCK)
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt"),
            patch("examples.frontend_backend_live.adapter.adapter.build_tts"),
        ):
            adapter = CascadeAdapter.from_config(session(), "websocket", None, settings, SHIPPED_CATALOG)
        entry = load_yaml_file(CATALOG_PATH)[settings.live.prompt]
        self.assertTrue(entry["internal"])  # the repository UI's prompt list leaves it out
        self.assertEqual(adapter.instructions, entry["content"].strip())  # as written: no channel context follows
        for cafe in ("Bluebird", "Remy", catalog_prompt(CHANNEL_PROMPT_KEY)):
            self.assertNotIn(cafe, adapter.instructions)

    def test_describe_is_the_adapters_half_of_info_with_no_key(self):
        info = CascadeAdapter.describe(TALKER_SETTINGS, FakeCatalog())
        self.assertEqual(set(info), {"frontend", "prompt_version", "speech"})  # the console's prefills are the root's
        self.assertEqual(
            info["frontend"],
            {
                "provider": "chat-completions",
                "slot": "llm",
                "model": "model-of-llm",
                "base_url": LOCAL["base_url"],
                "prompt": "voice",
            },
        )
        self.assertEqual(
            info["speech"],
            {
                "tts_mode": "per_sentence",
                "turn_end": "smart_turn",
                "vad_start_seconds": 0.12,
                "vad_stop_seconds": 0.2,
                "turn_silence_seconds": 1.2,
            },
        )
        self.assertNotIn("nvapi-secret", json.dumps(info))

    def test_describe_gives_the_seconds_each_turn_end_runs_with(self):
        cases = {
            "silence": {
                "vad_start_seconds": 0.16,
                "vad_stop_seconds": 0.8,  # its pause is its VAD's stop
                "turn_silence_seconds": 0.8,
            },
            "shared": {},  # the shared pipeline's environment switches set its seconds
        }
        for name, seconds in cases.items():
            with self.subTest(name):
                speech = {"turn_end": name, "vad_start_seconds": 0.16, "turn_silence_seconds": 0.8}
                settings = CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": speech})
                info = CascadeAdapter.describe(settings, FakeCatalog())
                self.assertEqual(info["speech"], {"tts_mode": "per_sentence", "turn_end": name, **seconds})


# ---------------------------------------------------------------------------------------------- readiness
class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    def probed(self, answers):
        """Patch the reachability probe: ``answers`` by address; record each address and the thread it ran on."""
        calls = []

        def probe(address):
            calls.append((address, threading.current_thread() is threading.main_thread()))
            return answers.get(address, True)

        return patch.object(readiness_module, "is_endpoint_reachable", probe), calls

    async def test_a_cloud_recipe_probes_nothing(self):
        readiness = CascadeAdapter.readiness(TALKER_SETTINGS, FakeCatalog(CLOUD))
        probe, calls = self.probed({})
        with probe:
            await readiness.ensure_ready()
        self.assertEqual(calls, [])

    async def test_a_local_recipe_probes_each_service_off_the_event_loop(self):
        readiness = CascadeAdapter.readiness(TALKER_SETTINGS, FakeCatalog(LOCAL))
        probe, calls = self.probed({})
        with probe:
            await readiness.ensure_ready()
        self.assertEqual(calls, [(LOCAL["asr"], False), (LOCAL["tts"], False), (LOCAL["base_url"], False)])

    async def test_a_service_still_starting_is_named(self):
        readiness = CascadeAdapter.readiness(TALKER_SETTINGS, FakeCatalog(LOCAL))
        probe, _ = self.probed({LOCAL["tts"]: False})
        with probe, self.assertRaises(ServiceStarting) as starting:
            await readiness.ensure_ready()
        self.assertEqual((starting.exception.name, str(starting.exception)), ("tts", "tts is still starting"))

    async def test_a_speech_entry_with_no_server_is_the_cloud_and_is_not_probed(self):
        self.assertEqual(EndpointReadiness({"asr": NVCF_GRPC_SERVER, "llm": NVCF_LLM_BASE_URL + "/"}).services, {})


# -------------------------------------------------------------------------------------- a text-only host
class Passthrough(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class TextHost(PipelineHost):
    """A second PipelineHost adapter with no speech services: the host still gives it a gate and a tracker."""

    async def start(self, session, delegation):
        self.session = session
        processors = self.audio_pipeline(
            [Passthrough()], transport="websocket", connection=None, audio_in_rate=LIVE_AUDIO_IN_RATE
        )
        await self.run_pipeline(processors, stt=None, tts=None)


class TextOnlyHostTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_text_only_host_gets_the_input_gate_and_the_playout_tracker(self):
        host, port = TextHost(), RecordingPort()
        await host.start(port, None)
        self.addAsyncCleanup(host.stop_pipeline)
        await host.set_muted(True)  # a host with no gate of its own must not raise
        await host.feed_audio(bytes(4800), EXCHANGE_RATE)
        async with asyncio.timeout(2):
            while port.input_ms < 100:
                await asyncio.sleep(0.01)
        self.assertTrue(host.gate.muted)
        self.assertIsNotNone(host.playout)


# ---------------------------------------------------------------------------------------- the history half
class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, reports_events):
        adapter = CascadeAdapter(
            session(instructions="Be brief."),
            "websocket",
            None,
            live_config=TALKER_CONFIG,
            frontend_endpoint=ModelEndpoint(model="talker-model", base_url="http://talker"),
            instructions="Be brief.",
            stt=None,
            tts=None,
        )
        adapter.delegation = SimpleNamespace(reports_events=reports_events)
        frontend = SimpleNamespace(name="chat-completions", reply_format="Reply in JSON.")
        prompts = SimpleNamespace(
            name="default",
            frame_before="Before.",
            frame_after="After.",
            routing="Route.",
            actions=("speak", "delegate"),
            call_hints=SimpleNamespace(talker=""),
            runtime_label="Runtime:",
            decision_cue="Decide.",
        )
        adapter.agent = SimpleNamespace(talker=Talker(frontend, prompts), frontend=frontend)
        adapter.cascade = SimpleNamespace(
            conversation=SimpleNamespace(
                labeled=lambda: [{"role": "system", "content": "Be brief."}], kinds=lambda: []
            ),
            turns=SimpleNamespace(work_line=lambda: "Work." if reports_events else ""),
            heard_log=SimpleNamespace(lines=lambda: [("caller", "C: A latte."), ("context", "X: Loyalty: gold.")]),
            worknote=SimpleNamespace(open=lambda: ["item_d1"]),
        )
        return adapter

    async def test_the_adapters_half_in_responses_mode(self):
        history = await self.adapter(reports_events=True).history()
        self.assertEqual(history["frontend"]["model"], "talker-model")
        self.assertEqual(
            history["frontend"]["prompt"],
            [
                {"source": "prompt set default: the voice's role", "text": "Before."},
                {"source": "session.instructions", "text": "Be brief."},
                {"source": "prompt set default: how the instructions apply", "text": "After."},
                {"source": "prompt set default: routing rules", "text": "Route."},
                {"source": "reply format (chat-completions frontend)", "text": "Reply in JSON."},
                {"source": "runtime data, after the conversation", "text": "Runtime:\nWork.\n\nDecide."},
            ],
        )
        pending = [{"kind": "caller", "text": "C: A latte."}, {"kind": "context", "text": "X: Loyalty: gold."}]
        self.assertEqual(history["backend"], {"busy": True, "pending": pending})

    async def test_client_mode_has_no_server_side_backend(self):
        history = await self.adapter(reports_events=False).history()
        self.assertIsNone(history["backend"])
        self.assertEqual(history["frontend"]["messages"], [])  # the talker's voice history: no turn yet
        runtime = {"source": "runtime data, after the conversation", "text": "Runtime:\nDecide."}  # no work line
        self.assertEqual(history["frontend"]["prompt"][-1], runtime)

    async def test_blank_instructions_show_as_the_catalog_entry_that_replaced_them(self):
        adapter = self.adapter(reports_events=True)
        adapter.config = session(instructions=" \n")
        prompt = (await adapter.history())["frontend"]["prompt"]
        self.assertEqual(prompt[1], {"source": f"prompts.yaml: {TALKER_CONFIG.prompt}", "text": "Be brief."})

    async def test_nothing_before_the_start(self):
        adapter = self.adapter(reports_events=True)
        adapter.cascade = None
        self.assertIsNone(await adapter.history())


if __name__ == "__main__":
    unittest.main()
