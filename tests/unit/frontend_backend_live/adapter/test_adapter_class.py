# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The frontend adapter class contract and readiness."""

import asyncio
import json
import threading
import unittest
from functools import partial
from unittest.mock import patch

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

from examples.frontend_backend_live.adapter import readiness as readiness_module
from examples.frontend_backend_live.adapter.adapter import (
    LIVE_AUDIO_IN_RATE,
    CascadeAdapter,
    CascadeAdapterSettings,
    seed_messages,
)
from examples.frontend_backend_live.adapter.bridge import LiveOutputTranscript
from examples.frontend_backend_live.adapter.contract import FrontendAdapterClass, ServiceStarting
from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from examples.frontend_backend_live.adapter.readiness import EndpointReadiness
from examples.frontend_backend_live.adapter.registry import FRONTEND_ADAPTERS
from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.tables import pick
from examples.frontend_backend_live.media.turns import SmartTurn, VADWhileBotSilentUserTurnStartStrategy
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.talker import LLM_SERVICES, RESPONSE_FORMATS, TalkerChat, TalkerTools
from examples.frontend_backend_live.voice import Voice
from examples.shared import audio_recorder
from gateways.live.ports import FrontendAdapter
from gateways.live.protocol import AudioFormat, SessionConfig
from tests.unit.cascade_helpers import LIVE_CONFIG as CASCADE_TEST_CONFIG
from tests.unit.core_helpers import EXAMPLE_SETTINGS, FakeChat
from tests.unit.frontend_backend_live.adapter.adapter_helpers import FakeSocket, RecordingPort, tone
from tests.unit.frontend_helpers import ENDPOINT, Port
from utils import NVCF_GRPC_SERVER, NVCF_LLM_BASE_URL

# The shipped frontend.cascade block, as the root hands it to the class.
SHIPPED_BLOCK = dict(EXAMPLE_SETTINGS["frontend"]["cascade"])
# The helpers the root hands the class with its block: the delegation core's.
HELPERS = {"pick": pick, "message_text": message_text}
# The example's own catalog reader, with one talker model.
SHIPPED_CATALOG = ExampleCatalog({"llm": [{"model_id": "org/front", "base_url": "http://front/v1"}]})
# The end-to-end tests' configuration, with the talker's prompt named as the shipped settings name it.
TALKER_CONFIG = CASCADE_TEST_CONFIG.model_copy(update={"prompt": "generic"})
TALKER_SETTINGS = CascadeAdapterSettings(
    live=TALKER_CONFIG,
    talkers=partial(pick, LLM_SERVICES, kind="talker_style"),
    turn_end=SmartTurn(TALKER_CONFIG.speech),
    response_formats=partial(pick, RESPONSE_FORMATS, kind="response_format"),
    message_text=message_text,
)
LOCAL = {"asr": "nemotron-asr:50052", "tts": "magpie-tts:50051", "base_url": "http://nvidia-llm:8000/v1"}
# The talker's services entry.
TALKER_ENTRY = {"model_id": "model-of-llm", "system_prompt": "You are a helpful assistant."}
CLOUD = {"asr": NVCF_GRPC_SERVER, "tts": NVCF_GRPC_SERVER, "base_url": NVCF_LLM_BASE_URL}


def members(protocol) -> set[str]:
    return {name for name in vars(protocol) if not name.startswith("_")} | set(protocol.__annotations__)


class FakeCatalog:
    """The example's catalog as the root reads it once: models by slot and services by slot.

    Each slot's model comes from the entry whose services key is ``key-of-<slot>``.
    """

    def __init__(self, recipe=LOCAL):
        self.recipe = recipe
        self.asked: list[tuple] = []

    def model(self, slot, overrides=None):
        self.asked.append(("model", slot, dict(overrides or {})))
        model, base_url = f"model-of-{slot}", self.recipe["base_url"]
        return ModelEndpoint(model=model, base_url=base_url, api_key="nvapi-secret", service_key=f"key-of-{slot}")

    def service(self, slot):
        self.asked.append(("service", slot))
        speech = {"server": self.recipe.get(slot), "voice_id": "Magpie-Multilingual.EN-US.Aria"}
        return {"llm": TALKER_ENTRY}.get(slot, speech)


def session(**changes) -> SessionConfig:
    return SessionConfig(model="live-1", **changes)


# ------------------------------------------------------------------------------------- the table and contract
async def build_as_the_root_does(table, name, block, catalog, config):
    """The root's four steps, with no branch on the adapter's name."""
    cls = pick(table, name, "frontend adapter")
    settings = cls.Settings.parse(block, **HELPERS)
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
    def test_the_settings_build_the_callers_turns_from_the_speech_block(self):
        settings = CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": {"interrupt_min_words": 3}}, **HELPERS)
        self.assertIsInstance(settings.turn_end, SmartTurn)
        sound, words = settings.turn_end.user_params(welcome_enabled=False).user_turn_strategies.start
        self.assertIsInstance(sound, VADWhileBotSilentUserTurnStartStrategy)
        self.assertIsInstance(words, MinWordsUserTurnStartStrategy)
        self.assertEqual(settings.turn_end.describe()["interrupt_min_words"], 3)

    def test_a_bad_speech_block_is_refused_with_a_message_that_names_it(self):
        cases = {
            "speech.vad_stop_seconds must be below speech.turn_silence_seconds: Smart Turn judges a pause before the "
            "silence that ends the turn anyway": {"vad_stop_seconds": 1.2},  # as long as the default silence
            "speech.vad_start_seconds must not be above speech.vad_stop_seconds, the VAD stop it is paired with": {
                "vad_start_seconds": 0.3
            },
        }
        for message, speech in cases.items():
            with self.subTest(message), self.assertRaises(ValueError) as refused:
                CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": speech}, **HELPERS)
            self.assertEqual(str(refused.exception), message)

    def test_every_other_key_is_checked_by_the_frontends_settings(self):
        for block, message in (
            ({**SHIPPED_BLOCK, "spech": {}}, "frontend.cascade\nspech\n.*Extra inputs are not permitted"),
            ({**SHIPPED_BLOCK, "speech": {"tts_rate": 22050}}, "speech.tts_rate\n.*Extra inputs are not permitted"),
            ({**SHIPPED_BLOCK, "speech": ["per_sentence"]}, "speech\n.*valid dictionary"),
            ({**SHIPPED_BLOCK, "speech": {"interrupt_min_words": 0}}, "speech.interrupt_min_words\n.*greater than 0"),
            ([], "frontend.cascade\n.*valid dictionary"),
        ):
            with self.subTest(block=block), self.assertRaisesRegex(ValueError, message):
                CascadeAdapterSettings.parse(block, **HELPERS)

    def test_the_block_names_the_talkers_style_by_model_and_no_talker_service(self):
        self.assertNotIn("provider", SHIPPED_BLOCK)
        with self.assertRaisesRegex(ValueError, "frontend.cascade\nprovider\n.*Extra inputs are not permitted"):
            CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "provider": "chat-completions"}, **HELPERS)
        known = ", ".join(sorted(LLM_SERVICES))
        unknown = {**SHIPPED_BLOCK, "talker_style": {"nemotron-lightning": "chat-completions"}}
        with self.assertRaisesRegex(ValueError, f"^Unknown talker_style 'chat-completions'; known: {known}$"):
            CascadeAdapterSettings.parse(unknown, **HELPERS)  # refused when the root reads the block


# -------------------------------------------------------------------------------------------- from_config
class FromConfigTests(unittest.TestCase):
    def build(self, config=None, catalog=None, settings=TALKER_SETTINGS):
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
        stt.assert_called_once_with(voice, catalog.service("asr"))
        sound, words = adapter.user_params.user_turn_strategies.start  # the same turns' start and turn detection
        self.assertIsInstance(sound, VADWhileBotSilentUserTurnStartStrategy)
        self.assertIsInstance(words, MinWordsUserTurnStartStrategy)
        (stop,) = adapter.user_params.user_turn_strategies.stop
        self.assertIsInstance(stop, TurnAnalyzerUserTurnStopStrategy)
        self.assertEqual(adapter.user_params.user_mute_strategies, [])  # a live session has no welcome to wait for
        tts.assert_called_once_with(voice, catalog.service("tts"))  # in its catalog entry's synthesis mode
        self.assertEqual((adapter.transport, adapter.live_config), ("websocket", TALKER_CONFIG))
        self.assertIs(adapter.talker, TalkerChat)  # the block names no style for the model: the JSON decision
        self.assertIsNone(adapter.frontend)  # nothing is built or connected before start
        self.assertIsNone(adapter.task)

    def test_the_talkers_style_is_the_one_the_block_names_for_its_models_services_key(self):
        for styles, style, service in (
            ({"key-of-llm": "json"}, "json", TalkerChat),
            ({"key-of-llm": "tools"}, "tools", TalkerTools),
            ({"another-model": "tools"}, "json", TalkerChat),  # a key the block does not list: the JSON decision
        ):
            with self.subTest(styles):
                settings = CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "talker_style": styles}, **HELPERS)
                adapter, _, _, _ = self.build(settings=settings)
                self.assertIs(adapter.talker, service)
                info = CascadeAdapter.describe(settings, FakeCatalog())
                self.assertEqual(info["frontend"]["talker_style"], style)

    def test_the_speech_block_sets_the_user_aggregators_vad_and_not_the_speech_to_text(self):
        settings = CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": {"vad_stop_seconds": 0.3}}, **HELPERS)
        catalog = FakeCatalog()
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt") as stt,
            patch("examples.frontend_backend_live.adapter.adapter.build_tts"),
        ):
            adapter = CascadeAdapter.from_config(session(), "websocket", None, settings, catalog)
        stt.assert_called_once_with({}, catalog.service("asr"))
        self.assertEqual(adapter.user_params.vad_analyzer.params.stop_secs, 0.3)

    def test_a_session_without_instructions_or_with_blank_ones_gets_the_configured_talker_entry(self):
        for blank in (None, "", " ", "\n\t \n"):
            with self.subTest(instructions=blank):
                adapter, _, _, _ = self.build(session(instructions=blank))
                self.assertEqual(adapter.instructions, adapter.prompts.text("Talker", "generic"))
        adapter, _, _, _ = self.build(session(instructions="  Be brief.\n"))
        self.assertEqual(adapter.instructions, "  Be brief.\n")  # the client's own text, word for word

    def test_the_shipped_default_is_the_domain_free_generic_talker_as_written(self):
        settings = CascadeAdapterSettings.parse(SHIPPED_BLOCK, **HELPERS)
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt"),
            patch("examples.frontend_backend_live.adapter.adapter.build_tts"),
        ):
            adapter = CascadeAdapter.from_config(session(), "websocket", None, settings, SHIPPED_CATALOG)
        prompts = adapter.prompts
        self.assertIn(settings.live.prompt, prompts.agent_prompts["Talker"])  # read-only in the UI, never selected
        self.assertEqual(adapter.instructions, prompts.text("Talker", "generic"))  # as written: no channel context
        for cafe in ("Bluebird", "Remy", prompts.text("Channel", "cafe")):
            self.assertNotIn(cafe, adapter.instructions)

    def test_describe_is_the_adapters_half_of_info_with_no_key(self):
        info = CascadeAdapter.describe(TALKER_SETTINGS, FakeCatalog())
        self.assertEqual(set(info), {"frontend", "speech"})
        self.assertEqual(
            info["frontend"],
            {
                "talker_style": "json",
                "slot": "llm",
                "model": "model-of-llm",
                "base_url": LOCAL["base_url"],
                "prompt": "generic",
            },
        )
        self.assertEqual(
            info["speech"],
            {
                "tts_mode": "per_sentence",
                "vad_start_seconds": 0.12,
                "vad_stop_seconds": 0.2,
                "turn_silence_seconds": 1.2,
                "interrupt_min_words": 2,
            },
        )
        self.assertNotIn("nvapi-secret", json.dumps(info))

    def test_describe_gives_the_text_to_speechs_synthesis_mode(self):
        class StitchedCatalog(FakeCatalog):
            def service(self, slot):
                return {**super().service(slot), "synthesis_mode": "stitched"}

        info = CascadeAdapter.describe(TALKER_SETTINGS, StitchedCatalog())
        self.assertEqual(info["speech"]["tts_mode"], "stitched")

    def test_describe_gives_the_settings_the_callers_turns_run_with(self):
        speech = {"vad_start_seconds": 0.16, "turn_silence_seconds": 0.8, "interrupt_min_words": 3}
        settings = CascadeAdapterSettings.parse({**SHIPPED_BLOCK, "speech": speech}, **HELPERS)
        info = CascadeAdapter.describe(settings, FakeCatalog())
        self.assertEqual(info["speech"], {"tts_mode": "per_sentence", "vad_stop_seconds": 0.2, **speech})


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
    """A second PipelineHost adapter with no speech services: the host still gives it a gate and a transcript."""

    async def start(self, session, delegation):
        self.session = session
        edges = self.edges(transport="websocket", connection=None, audio_in_rate=LIVE_AUDIO_IN_RATE, stt=None)
        self.processors = edges.around([Passthrough()], [])
        await self.run_pipeline(PipelineWorker(Pipeline(self.processors), **edges.worker), self.processors)


class TextOnlyHostTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_text_only_host_gets_the_input_gate_and_the_output_transcript(self):
        # The shared audio recorder's switches at their defaults, whatever the environment sets: nothing records.
        self.enterContext(patch.object(audio_recorder, "ENABLE_ASR_DUMP", False))
        self.enterContext(patch.object(audio_recorder, "ENABLE_TTS_DUMP", False))
        host, port = TextHost(), RecordingPort(socket=FakeSocket(AudioFormat(rate=16000)))
        await host.start(port, None)
        self.addAsyncCleanup(host.stop_pipeline)
        await host.set_muted(True)  # a host with no gate of its own must not raise
        port.socket.send("an append", audio=bytes(3200))  # 100 ms of 16 kHz PCM16 from the primary WebSocket
        async with asyncio.timeout(2):
            while port.input_ms < 100:
                await asyncio.sleep(0.01)
        self.assertTrue(host.gate.muted)
        self.assertIsInstance(host.processors[-1], LiveOutputTranscript)

    async def test_the_shared_audio_recorder_follows_the_transcript_only_when_an_audio_dump_is_on(self):
        with (
            patch.object(audio_recorder, "ENABLE_ASR_DUMP", False),
            patch.object(audio_recorder, "ENABLE_TTS_DUMP", False),
        ):  # the switches' defaults: nothing records
            host = TextHost()
            await host.start(RecordingPort(), None)
            self.addAsyncCleanup(host.stop_pipeline)
            self.assertIsNone(host.recorder)
            self.assertIsInstance(host.processors[-1], LiveOutputTranscript)
        with (
            patch.object(audio_recorder, "ENABLE_ASR_DUMP", True),
            patch.object(audio_recorder, "_validate_dump_dir"),
            patch.object(audio_recorder, "_write_wav"),
        ):
            host, port = TextHost(), RecordingPort(socket=FakeSocket(AudioFormat(rate=16000)))
            await host.start(port, None)
            self.addAsyncCleanup(host.stop_pipeline)
            self.assertIsInstance(host.recorder, AudioBufferProcessor)
            self.assertIs(host.processors[-1], host.recorder)  # after the output transcript
            self.assertIsInstance(host.processors[-2], LiveOutputTranscript)
            # It records from the pipeline's start: the caller's audio, past every processor, is in its buffer.
            for index in range(3):
                port.socket.send(f"audio {index}", audio=tone(100, rate=16000))
            async with asyncio.timeout(2):
                while not host.recorder.has_audio():
                    await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------------------- the history half
class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, reports_events=True):
        """A session's adapter whose voice runs on the shipped prompt file, as its start builds it."""
        adapter = CascadeAdapter(
            session(instructions="Be brief."),
            "websocket",
            None,
            live_config=TALKER_CONFIG,
            frontend_endpoint=ModelEndpoint(model="talker-model", base_url="http://talker"),
            instructions="Be brief.",
            prompts=load_prompts(),
            stt=None,
            tts=None,
            item_text=message_text,
        )
        prompts = adapter.prompts
        adapter.delegation = Port(reports_events=reports_events)
        adapter.voice = Voice(
            config=TALKER_CONFIG,
            prompts=prompts.cascade,
            lines=prompts.lines,
            json_rules=prompts.cascade.json_rules,
            empty_turn_note=prompts.cascade.empty_turn_note,
            delegation=adapter.delegation,
            item_text=message_text,
        )
        context = LLMContext(adapter.voice.seed(seed_messages(adapter.config, adapter.instructions)))
        level = RESPONSE_FORMATS["json_schema"]
        talker = TalkerChat.from_config(ENDPOINT, TALKER_CONFIG.reliability, level, FakeChat())
        adapter.voice.bind(talker, context)
        self.addAsyncCleanup(adapter.voice.close)
        return adapter

    def parts(self, adapter, runtime):
        prompts = adapter.prompts.cascade
        return [
            {"source": "agent_prompts.Cascade.frame_before", "text": prompts.frame_before},
            {"source": "session.instructions", "text": "Be brief."},
            {"source": "agent_prompts.Cascade.frame_after", "text": prompts.frame_after},
            {"source": "agent_prompts.Cascade.routing", "text": prompts.routing},
            {"source": "agent_prompts.Cascade.json_rules", "text": prompts.json_rules},
            {"source": "runtime data, after the conversation", "text": runtime},
        ]

    async def test_the_adapters_half_in_responses_mode(self):
        adapter = self.adapter(reports_events=True)
        voice = adapter.voice
        voice.handoffs.delegate("A latte.", "", decided_at=0)
        await asyncio.gather(*voice.handoffs.tasks)
        voice.heard_log.caller("Make it oat milk.")
        history = await adapter.history()
        self.assertEqual(history["frontend"]["model"], "talker-model")
        work = voice.worknote.line()
        self.assertTrue(work)  # the open delegation's line
        prompts = adapter.prompts.cascade
        runtime = f"{prompts.runtime_label}\n{work}\n\n{prompts.decision_cue}"
        self.assertEqual(history["frontend"]["prompt"], self.parts(adapter, runtime))
        self.assertTrue(history["backend"]["busy"])
        [pending] = history["backend"]["pending"]
        self.assertEqual(pending["kind"], "caller")
        self.assertTrue(pending["text"].endswith("Make it oat milk."))

    async def test_client_mode_has_no_server_side_backend(self):
        adapter = self.adapter(reports_events=False)
        history = await adapter.history()
        self.assertIsNone(history["backend"])
        self.assertEqual(history["frontend"]["messages"], [])  # the talker's context: no turn yet
        prompts = adapter.prompts.cascade
        runtime = f"{prompts.runtime_label}\n{prompts.decision_cue}"  # no work line
        self.assertEqual(history["frontend"]["prompt"], self.parts(adapter, runtime))

    async def test_blank_instructions_show_as_the_talker_entry_that_replaced_them(self):
        adapter = self.adapter(reports_events=True)
        adapter.config = session(instructions=" \n")
        prompt = (await adapter.history())["frontend"]["prompt"]
        self.assertEqual(prompt[1], {"source": f"agent_prompts.Talker.{TALKER_CONFIG.prompt}", "text": "Be brief."})

    async def test_nothing_before_the_start(self):
        adapter = self.adapter(reports_events=True)
        adapter.voice = None
        self.assertIsNone(await adapter.history())


if __name__ == "__main__":
    unittest.main()
