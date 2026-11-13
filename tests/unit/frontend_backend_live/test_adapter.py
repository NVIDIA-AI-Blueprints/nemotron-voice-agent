# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The cascade frontend adapter: its settings, each session's build, its readiness, and its half of history."""

import asyncio
import io
import json
import threading
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext

import config_store
import utils
from examples.frontend_backend_live.adapter import readiness as readiness_module
from examples.frontend_backend_live.adapter.adapter import CascadeAdapter, CascadeAdapterSettings, seed_messages
from examples.frontend_backend_live.adapter.contract import ServiceStarting
from examples.frontend_backend_live.adapter.readiness import EndpointReadiness
from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.tables import pick
from examples.frontend_backend_live.media.turns import SmartTurn
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.talker import LLM_SERVICES, RESPONSE_FORMATS, TalkerChat, TalkerTools
from examples.frontend_backend_live.voice import Voice
from examples.shared.prewarm import peek_cached_tts_config
from gateways.live.protocol import SessionConfig
from tests.unit.cascade_helpers import LIVE_CONFIG, scripted_talker, transcript_turns
from tests.unit.core_helpers import EXAMPLE_SETTINGS, FakeChat
from tests.unit.frontend_helpers import ENDPOINT, Port
from utils import NVCF_GRPC_SERVER, NVCF_LLM_BASE_URL, SELF_HOSTED_SOURCE

SHIPPED_BLOCK = dict(EXAMPLE_SETTINGS["frontend"]["cascade"])
HELPERS = {"pick": pick, "message_text": message_text}
TALKER_CONFIG = LIVE_CONFIG.model_copy(update={"prompt": "generic"})
TALKER_SETTINGS = CascadeAdapterSettings(
    live=TALKER_CONFIG,
    talkers=partial(pick, LLM_SERVICES, kind="talker_style"),
    turn_end=SmartTurn(TALKER_CONFIG.speech),
    response_formats=partial(pick, RESPONSE_FORMATS, kind="response_format"),
    message_text=message_text,
)
LOCAL = {"asr": "nemotron-asr:50052", "tts": "magpie-tts:50051", "base_url": "http://nvidia-llm:8000/v1"}
CLOUD = {"asr": NVCF_GRPC_SERVER, "tts": NVCF_GRPC_SERVER, "base_url": NVCF_LLM_BASE_URL}
DEFAULT_VOICE, CLAIRE = "Magpie-Multilingual.EN-US.Aria", "Magpie-Multilingual.EN-US.Claire"
# The NIMs as a host-run server reaches them, at their published ports, which accept connections while a model loads.
NIMS = {"asr": "localhost:50152", "tts": "localhost:50151", "llm": "http://localhost:18000/v1"}
READY_URLS = [
    "http://localhost:9001/v1/health/ready",
    "http://localhost:9000/v1/health/ready",
    "http://localhost:18000/v1/health/ready",
]


class FakeCatalog:
    """The example's catalog as the root reads it: each slot's model has the services key ``key-of-<slot>``."""

    def __init__(self, recipe=LOCAL, **tts):
        self.recipe, self.tts, self.asked = recipe, tts, []

    def model(self, slot, overrides=None):
        self.asked.append(("model", slot, dict(overrides or {})))
        model, base_url = f"model-of-{slot}", self.recipe["base_url"]
        return ModelEndpoint(model=model, base_url=base_url, api_key="nvapi-secret", service_key=f"key-of-{slot}")

    def service(self, slot):
        if slot == "llm":
            return {"model_id": "model-of-llm", "system_prompt": "You are a helpful assistant."}
        speech = {"server": self.recipe.get(slot), "voice_id": DEFAULT_VOICE}
        return {**speech, **self.tts} if slot == "tts" else speech


class ListingService:
    """A text-to-speech service as the shared voice list reads it: it lists ``voices`` under ``prefix``, or fails."""

    def __init__(self, prefix="Magpie-Multilingual", voices="EN-US.Aria:0,EN-US.Claire:1"):
        self.parameters = {"language_code": "en-US", "voice_name": prefix, "subvoices": voices}

    def _initialize_client(self):
        pass

    async def _create_synthesis_config(self):
        if not self.parameters["subvoices"]:
            raise RuntimeError("synthesis config unavailable")
        return SimpleNamespace(model_config=[SimpleNamespace(parameters=self.parameters)])


def session(**changes) -> SessionConfig:
    return SessionConfig(model="live-1", **changes)


def voiced(voice) -> SessionConfig:
    return session(audio={"output": {"voice": voice}})


def parse(**block) -> CascadeAdapterSettings:
    return CascadeAdapterSettings.parse({**SHIPPED_BLOCK, **block}, **HELPERS)


class SettingsTests(unittest.TestCase):
    def test_a_bad_block_is_refused_with_what_is_wrong(self):
        styles = ", ".join(sorted(LLM_SERVICES))
        for block, error, message in (
            ({"oops": 1}, ValueError, "for frontend.cascade\noops\n"),
            ({"speech": {"tts_rate": 1}}, ValueError, "for frontend.cascade\nspeech.tts_rate\n"),
            (
                {"speech": {"vad_stop_seconds": 1.2}},
                ValueError,
                "^speech.vad_stop_seconds must be below speech.turn_silence_seconds: Smart Turn judges a pause "
                "before the silence that ends the turn anyway$",
            ),
            (
                {"speech": {"vad_start_seconds": 0.3}},
                ValueError,
                "^speech.vad_start_seconds must not be above speech.vad_stop_seconds, the VAD stop it is paired with$",
            ),
            (
                {"talker_style": {"nemotron-lightning": "chat-completions"}},
                ValueError,
                f"^Unknown talker_style 'chat-completions'; known: {styles}$",
            ),
            ({"prompt": "generc"}, KeyError, r"agent_prompts\.Talker\.generc was not found in .*; known: generic"),
        ):
            with self.subTest(block=block), self.assertRaisesRegex(error, message):
                parse(**block)


class FromConfigTests(unittest.TestCase):
    def build(self, config=None, catalog=None, settings=TALKER_SETTINGS, service=None):
        """Build with the text-to-speech faked and its voice list uncached; ``self.listing`` fetches the list."""
        catalog = catalog or FakeCatalog()
        with (
            patch("examples.frontend_backend_live.adapter.adapter.build_stt") as stt,
            patch("examples.frontend_backend_live.adapter.adapter.build_tts") as tts,
            patch("gateways.realtime.voice.load_service_entry", return_value={"voice_id": DEFAULT_VOICE}),
            patch.dict(config_store._store, clear=True),
            patch("examples.shared.prewarm._create_tts_service", return_value=service or ListingService()) as listing,
        ):
            self.listing = listing
            adapter = CascadeAdapter.from_config(config or session(), "websocket", None, settings, catalog)
            tts_entry = {"server": "", "voice_id": "", "model": "", **catalog.service("tts")}
            self.listed = peek_cached_tts_config(tts_entry["server"], tts_entry["voice_id"], "", tts_entry["model"])
        return adapter, catalog, stt, tts

    def test_from_config_builds_the_talker_and_speech_unconnected(self):
        config = session(instructions="Be brief.", audio={"output": {"voice": {"id": CLAIRE}}})
        adapter, catalog, stt, tts = self.build(config)
        voice = {"tts_voice_id": CLAIRE}
        self.assertEqual(adapter.frontend_endpoint.model, "model-of-llm")
        self.assertIn(("model", "llm", voice), catalog.asked)
        self.assertEqual(adapter.instructions, "Be brief.")
        stt.assert_called_once_with(voice, catalog.service("asr"))
        tts.assert_called_once_with(voice, catalog.service("tts"))
        self.assertEqual(adapter.user_params.user_mute_strategies, [])  # a live session has no welcome to wait for
        self.assertEqual((adapter.transport, adapter.live_config), ("websocket", TALKER_CONFIG))
        self.assertIs(adapter.talker, TalkerChat)
        self.assertEqual((adapter.frontend, adapter.task), (None, None))

    def test_a_custom_voice_is_checked_against_its_tts_voices(self):
        logged = []
        self.addCleanup(logger.remove, logger.add(logged.append, level="WARNING", format="{message}"))
        unlisted = ListingService(voices="")
        for asked, service, catalog, spoken, warning in (
            (CLAIRE, ListingService(), FakeCatalog(), CLAIRE, None),
            ("Custom.Typo", ListingService(), FakeCatalog(), DEFAULT_VOICE, "Unknown TTS voice 'Custom.Typo'"),
            (CLAIRE, unlisted, FakeCatalog(), DEFAULT_VOICE, f"cannot verify {CLAIRE!r}"),
            (DEFAULT_VOICE, unlisted, FakeCatalog(), DEFAULT_VOICE, None),
            (CLAIRE, unlisted, FakeCatalog(voice_id=""), CLAIRE, f"cannot verify {CLAIRE!r}; using {CLAIRE!r}"),
        ):
            with self.subTest(asked=asked, voices=bool(service.parameters["subvoices"]), own=catalog.tts):
                logged.clear()
                _, catalog, _, tts = self.build(voiced({"id": asked}), catalog=catalog, service=service)
                tts.assert_called_once_with({"tts_voice_id": spoken}, catalog.service("tts"))
                own = catalog.service("tts")["voice_id"]
                self.listing.assert_called_once_with("magpie-tts:50051", own, "", "")
                warned = [line for line in logged if "TTS voice" in line]
                self.assertEqual(len(warned), 1 if warning else 0)
                self.assertIn(warning or "", "".join(warned))
        _, catalog, _, tts = self.build(voiced("marin"))
        tts.assert_called_once_with({}, catalog.service("tts"))  # a built-in name speaks with the TTS's own voice
        self.listing.assert_not_called()

    def test_a_custom_voice_uses_its_own_tts_names_not_the_first(self):
        # Chatterbox serves the session, as when Magpie is scaled to 0, while the services catalog's first is Magpie's.
        male, female = "Chatterbox-Multilingual.en-US.Male", "Chatterbox-Multilingual.en-US.Female"
        chatterbox = {"server": "chatterbox-tts:50051", "voice_id": male, "model": "chatterbox-tts-multilingual"}
        catalog = FakeCatalog(**chatterbox)
        service = ListingService("Chatterbox-Multilingual", "en-US.Male:0,en-US.Female:1")
        _, _, _, tts = self.build(voiced({"id": female}), catalog=catalog, service=service)
        tts.assert_called_once_with({"tts_voice_id": female}, chatterbox)
        self.listing.assert_called_once_with(chatterbox["server"], male, "", chatterbox["model"])
        self.assertEqual({voice["id"] for voice in self.listed["voices"]}, {male, female})
        _, _, _, tts = self.build(voiced({"id": female}), catalog=catalog, service=ListingService(voices=""))
        tts.assert_called_once_with({"tts_voice_id": male}, chatterbox)
        self.listing.assert_called_once_with(chatterbox["server"], male, "", chatterbox["model"])

    def test_the_talker_style_is_the_blocks_for_the_models_key(self):
        for styles, style, service in (
            ({"key-of-llm": "json"}, "json", TalkerChat),
            ({"key-of-llm": "tools"}, "tools", TalkerTools),
            ({"another-model": "tools"}, "json", TalkerChat),
        ):
            with self.subTest(styles):
                settings = parse(talker_style=styles)
                self.assertIs(self.build(settings=settings)[0].talker, service)
                self.assertEqual(CascadeAdapter.describe(settings, FakeCatalog())["frontend"]["talker_style"], style)

    def test_the_speech_block_sets_the_vad_not_the_speech_to_text(self):
        adapter, catalog, stt, _ = self.build(settings=parse(speech={"vad_stop_seconds": 0.3}))
        stt.assert_called_once_with({}, catalog.service("asr"))
        self.assertEqual(adapter.user_params.vad_analyzer.params.stop_secs, 0.3)

    def test_blank_instructions_get_the_configured_talker_prompt(self):
        for blank in (None, " \n"):
            with self.subTest(instructions=blank):
                adapter, _, _, _ = self.build(session(instructions=blank))
                self.assertEqual(adapter.instructions, adapter.prompts.text("Talker", "generic"))
        adapter, _, _, _ = self.build(session(instructions="  Be brief.\n"))
        self.assertEqual(adapter.instructions, "  Be brief.\n")

    def test_the_shipped_default_is_the_generic_talker_as_written(self):
        catalog = ExampleCatalog({"llm": [{"model_id": "org/front", "base_url": "http://front/v1"}]})
        settings = parse()
        adapter, _, _, _ = self.build(catalog=catalog, settings=settings)
        prompts = adapter.prompts
        self.assertIn(settings.live.prompt, prompts.agent_prompts["Talker"])
        self.assertEqual(adapter.instructions, prompts.text("Talker", "generic"))
        for cafe in ("Bluebird", "Remy", prompts.text("Channel", "cafe")):
            self.assertNotIn(cafe, adapter.instructions)

    def test_describe_is_the_adapters_half_of_info_with_no_key(self):
        info = CascadeAdapter.describe(TALKER_SETTINGS, FakeCatalog())
        frontend = {"talker_style": "json", "slot": "llm", "model": "model-of-llm", "base_url": LOCAL["base_url"]}
        self.assertEqual(info["frontend"], {**frontend, "prompt": "generic"})
        speech = {"vad_start_seconds": 0.12, "vad_stop_seconds": 0.2, "turn_silence_seconds": 1.2}
        self.assertEqual(
            info["speech"], {"tts_mode": "per_sentence", **speech, "interrupt_min_words": 2, "spelling_hold": True}
        )
        self.assertEqual(set(info), {"frontend", "speech"})
        self.assertNotIn("nvapi-secret", json.dumps(info))
        stitched = CascadeAdapter.describe(TALKER_SETTINGS, FakeCatalog(synthesis_mode="stitched"))
        self.assertEqual(stitched["speech"]["tts_mode"], "stitched")


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The self-hosted catalog names the health_path of the talker of NIMS only; LOCAL's talker has none.
        talker = {"base_url": NIMS["llm"], "health_path": "/v1/health/ready"}
        catalog = [(SELF_HOSTED_SOURCE, {"llm": {"nemotron-lightning": talker}})]
        self.enterContext(patch.object(utils, "service_catalog_sources", return_value=catalog))

    def probed(self, answers):
        """Patch the TCP probe: ``answers`` by address; record each address and the thread it ran on."""
        calls = []

        def probe(address, timeout):
            calls.append((address, threading.current_thread() is threading.main_thread()))
            return "ready" if answers.get(address, True) else "starting"

        return patch.object(readiness_module, "_probe_tcp", probe), calls

    async def test_a_nim_without_ready_json_is_still_starting(self):
        probe, _ = self.probed({})
        not_ready = patch("urllib.request.urlopen", lambda url, timeout: io.BytesIO(b'{"status": "not ready"}'))
        with probe, not_ready, self.assertRaises(ServiceStarting) as starting:
            await EndpointReadiness(NIMS).ensure_ready()
        self.assertEqual(starting.exception.name, "asr")

    async def test_only_a_local_recipe_is_probed_off_the_event_loop(self):
        local = [(LOCAL["asr"], False), (LOCAL["tts"], False), (LOCAL["base_url"], False)]
        for name, recipe, probed in (("cloud", CLOUD, []), ("local", LOCAL, local)):
            with self.subTest(name):
                readiness = CascadeAdapter.readiness(TALKER_SETTINGS, FakeCatalog(recipe))
                probe, calls = self.probed({})
                with probe:
                    await readiness.ensure_ready()
                self.assertEqual(calls, probed)

    async def test_a_service_still_starting_is_named(self):
        readiness = CascadeAdapter.readiness(TALKER_SETTINGS, FakeCatalog(LOCAL))
        probe, _ = self.probed({LOCAL["tts"]: False})
        with probe, self.assertRaises(ServiceStarting) as starting:
            await readiness.ensure_ready()
        self.assertEqual((starting.exception.name, str(starting.exception)), ("tts", "tts is still starting"))

    async def test_a_probe_past_its_bound_means_still_starting(self):
        released = threading.Event()
        self.addCleanup(released.set)
        hung = patch.object(readiness_module, "is_ready", lambda name, address: released.wait(3) or True)
        bound = patch.object(readiness_module, "_CONNECT_HEALTH_TIMEOUT_SECS", 0)
        with hung, bound, self.assertRaises(ServiceStarting) as starting:
            await EndpointReadiness(NIMS).ensure_ready()
        self.assertEqual(starting.exception.name, "asr")

    async def test_each_start_probes_a_port_anew(self):
        readiness = EndpointReadiness({"asr": "asr-sidecar:50099"})
        with patch("socket.create_connection", side_effect=[ConnectionRefusedError(), io.BytesIO()]):
            with self.assertRaises(ServiceStarting):
                await readiness.ensure_ready()
            await readiness.ensure_ready()

    async def test_ready_nims_are_asked_at_their_readiness_urls(self):
        probe, calls = self.probed({})
        asked = []

        def answer(url, timeout):
            asked.append(url)
            return io.BytesIO(b'{"status": "ready"}')

        with probe, patch("urllib.request.urlopen", answer):
            await EndpointReadiness(NIMS).ensure_ready()
        self.assertEqual((asked, calls), (READY_URLS, []))


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, reports_events=True):
        """Build a session's adapter with its voice on the shipped prompt file, as its start does."""
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
            user_params=transcript_turns(),
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
        talker = scripted_talker(TalkerChat, FakeChat()).from_config(
            ENDPOINT, TALKER_CONFIG.reliability, RESPONSE_FORMATS["json_schema"]
        )
        adapter.voice.bind(talker, context)
        self.addAsyncCleanup(adapter.voice.close)
        return adapter

    def parts(self, adapter, runtime):
        texts = adapter.prompts.cascade
        sources = [f"agent_prompts.Cascade.{name}" for name in ("frame_before", "frame_after", "routing", "json_rules")]
        named = [{"source": source, "text": getattr(texts, source.split(".")[-1])} for source in sources]
        instructions = {"source": "session.instructions", "text": "Be brief."}
        return [named[0], instructions, *named[1:], {"source": "runtime data, after the conversation", "text": runtime}]

    async def test_history_in_responses_mode_shows_the_backend_busy(self):
        adapter = self.adapter(reports_events=True)
        voice = adapter.voice
        voice.handoffs.delegate("A latte.", "", decided_at=0)
        await asyncio.gather(*voice.handoffs.tasks)
        voice.heard_log.caller("Make it oat milk.")
        history = await adapter.history()
        self.assertEqual(history["frontend"]["model"], "talker-model")
        work = voice.worknote.line()
        self.assertTrue(work)
        texts = adapter.prompts.cascade
        runtime = f"{texts.runtime_label}\n{work}\n\n{texts.decision_cue}"
        self.assertEqual(history["frontend"]["prompt"], self.parts(adapter, runtime))
        self.assertTrue(history["backend"]["busy"])
        [pending] = history["backend"]["pending"]
        self.assertEqual(pending["kind"], "caller")
        self.assertTrue(pending["text"].endswith("Make it oat milk."))

    async def test_client_mode_has_no_server_side_backend(self):
        adapter = self.adapter(reports_events=False)
        history = await adapter.history()
        self.assertIsNone(history["backend"])
        self.assertEqual(history["frontend"]["messages"], [])
        texts = adapter.prompts.cascade
        runtime = f"{texts.runtime_label}\n{texts.decision_cue}"
        self.assertEqual(history["frontend"]["prompt"], self.parts(adapter, runtime))

    async def test_blank_instructions_show_as_the_talker_entry(self):
        adapter = self.adapter(reports_events=True)
        adapter.config = session(instructions=" \n")
        prompt = (await adapter.history())["frontend"]["prompt"]
        self.assertEqual(prompt[1], {"source": f"agent_prompts.Talker.{TALKER_CONFIG.prompt}", "text": "Be brief."})

    async def test_the_context_meter_counts_the_talkers_context(self):
        adapter = self.adapter(reports_events=True)
        self.assertEqual(adapter.context_window_tokens(), TALKER_CONFIG.memory.context_tokens)
        adapter.voice.context = LLMContext([{"role": "user", "content": "hi"}])
        small = adapter.context_tokens()
        adapter.voice.context = LLMContext([{"role": "user", "content": "hi " * 500}])
        self.assertGreater(adapter.context_tokens(), small + 300)
        adapter.voice = None
        self.assertIsNone(await adapter.history())
        self.assertEqual((adapter.context_tokens(), adapter.context_window_tokens()), (None, None))
