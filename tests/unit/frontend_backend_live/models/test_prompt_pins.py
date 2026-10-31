# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The prompt set and the example's prompt catalog, pinned.

Each text a model reads is pinned by its sha256, so an unintended change fails here; update a digest on purpose when a
text changes.
"""

import asyncio
import hashlib
import json
import re
import unittest
from pathlib import Path

import yaml

from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_schema
from examples.frontend_backend_live.prompts import DEFAULT_PROMPT_SET, PROMPT_SETS, default, get_prompt_set
from examples.frontend_backend_live.prompts.catalog import LINES_KEY, catalog_line
from tests.unit.cascade_helpers import ScriptedFrontend

CATALOG = Path(__file__).resolve().parents[4] / "src/examples/frontend_backend_live/prompts.yaml"
DEFAULT = default.PROMPT_SET


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class PromptTextTests(unittest.TestCase):
    # Each text of the set, by its field: the talker's role, frame and routing rules; the writer's frame, commentary,
    # answer and progress rules; the work note's texts, the labels, the marks of the talker's memory, the words of each
    # delegation's input, the writer's notes, the schema hints and the normalizer's prompt.
    DIGESTS = {
        "frame_before": "04cc4b783a0d3e808443a0e2b2afceab6f676c8839670590131c4d6fe1b299ef",
        "frame_after": "a77da78e0549fa2f0d1af6e4e860994f715eb0f2ab0b55ec17bb3cd280be3e9e",
        "routing": "43fa57dd0502b5f427b043597c8a31e1308b70df2dec8db2f1dfef71e813fbce",
        "commentary": "146751021894e1c4fe0087a48c765134e5845b5f0ecc9beb8eedf3b0d220716a",
        "answer": "af7ebcc6e7569591135df28b5ce7be26d88a5cfa050aa4fbd9b6db0f5ba9e364",
        "progress": "502d51902e0db2546702c47fc78cb60bf4f1ab4fbf195d3c51058f29deb3b6cc",
        "writer_frame_before": "44ad2a21473dedf43f52cb17dc89eb3832a5673a77ecac5201ce133deb02c18d",
        "writer_frame_after": "e26efe4b9939af11b8c172fcdca27d889062044d4605be26a5f1c11663af913f",
        "work_in_progress": "1e25d42ff394424d13d2998eba4933fd06251d7dace71ddd0831aaaa35d787a3",
        "no_work_in_progress": "3f24d1a22ca78b922fc3532e3078e0c76961a8d5d60b9c609ce43b1a80a6dc27",
        "answer_ready_label": "b9f74dbe050460e447c8125ee599991fb6c40c18fa0b72886c7894b29c331d57",
        "typed_request_label": "d627e456ba2045be399c0b24fab0477a3dc4bb083a4b5a9efe344e39439e659a",
        "work_texts.item": "b51d1fd1c2d176204e95abfd041bc590561f27ff2ef019f687d808b8ace28659",
        "work_texts.running": "dfba91d1f5e7a6e1221ae8d3b7b3020024672394110ea6b3a1a1ad7887aeda78",
        "work_texts.ended": "570c5dab41df0d4aba74ea16192f816812d163d2bdc0a9b114ea038b026b5df7",
        "work_texts.ended_call": "6092824c2221f9d1f020852a657c87aa951365bc4214e40eafaa3e89d689a093",
        "work_texts.steps": "7e90fe19d0ad043702b971010fba62653fb096feb9839e898157f8b1ecb4f5f3",
        "work_texts.waiting": "4cc451c049461bae7d3988d38c906f4892542fdfe1b11acbd2b55b3e5ed9153d",
        "work_texts.update": "1ec37e5cc1854ad8ec0efdbd50fd05ff40096f1c3a6c7f501ccb60d4b71cb8de",
        # One label for every update the writer words: backend results, the application's message, or both.
        "update_label": "6b0a9d9c0cf427392b6c7c2edd7a918805c43c3a1f9730fea3b1866723af9048",
        "result_label": "3810840cb33d449b9773250caf2d56e76334854d884f79cfc0e3fb00e7f23552",
        "context_label": "8c45c98ce9ced7d0af7a9d5e66dadc36da1ffe22ffb8d2923e8c7cdeb2124ef3",
        "cut_mark": "76329e2e2e198f10e8f4e2c63eff11869ed44eec64e1eb9f0a10d10c14b98a6a",
        "transcript.header": "4c2e4a894cb656f4321329e995e726729a20c7d2b471b34788add4920332bb77",
        "transcript.omitted": "1956be9a88aba7da383e388cee6c2b9327ffb512db9fedd888f460741d8baac8",
        "transcript.caller": "9bc9b83c11ddcb417258094c42d71f0b548df6b26b39b91e64042143f19112e7",
        "transcript.agent": "11b39c93777e8f1f3983bdba7c72b22fe68cfea20c677e9de53e17cb7dbfb19f",
        "transcript.context": "5ddf5ee2937b330446d7f9e17da86157e3e65957ead3b37feaaa9d8fae85cbc0",
        "transcript.instruction": "0ee73fe16bcc5a556c3ac7e2f93dc4f51399e19b4bf11c6ba0647df350f8b6ae",
        "transcript.earlier": "d37eabb25d10e1438b464ccae5c05d765615b794278a3878f1858892e6fd413d",
        "transcript.unheard": "b2175e0a06996e03b87589c32fa1de8395d5f012b0bf785af0ede0455ae6fd70",
        "writer_notes.told": "ee6fad0e114c43a91e4e5f22cfe201320c4b7d73ccdfe30d2b6d33ca1651105b",
        "writer_notes.newer": "29a2291222b78f4f9a06efa65e90a834e0073da9df63ec50246062417a30638b",
        "call_hints.talker": "cf8cff9d45974d3fc5e4a9a9ab11c32c942a115209ff1a0914580214ef80c790",
        "call_hints.answer": "67aa75610fa6ccf2e908f271b04abe30ac4a3338e1c136c5e2de9566ae659fda",
        "call_hints.normalizer": "98cd8284427cee7816dcb4732c12bb6a1c2cadcb4fdcf5ce3eb53a72c4d5557b",
        "call_hints.progress": "69df152fbd4e8e5fa457d37eb6ebf9bbef3447bafb9a00b842b70fbb9a51872a",
        "normalizer_prompt.instructions": "3978976a4b16f66994070faa78c374eeed3cbaf76f494b29b4895ee433a6c051",
        "normalizer_prompt.newest_turn": "673e1768822895ba593fbfbb07d3e6b4f170a0e4ac8d31f1735d6c766bcd2e07",
        "caller_label": "1ec7b80cf30f2317fb81885fc40658a1e482ba2955eb96524444d85016c61194",
        # The routing decision's runtime data, after the conversation: its label, then the work line, the newest turn
        # and the cue.
        "runtime_label": "0bc1ff415b7cb0d73d1ffeec3d1676480e33fe1d61c2749cd7ebd961e0cfd41a",
        "decision_cue": "556ccc32a84e84ee0cbeb4cbf0191db2d4568db1324fdbbaab0d19703e68bdb7",
        "newest_turn": "10553a9d1f8dbc67a1a3046ffa6ebd130dec0a409185960cb45c803f42510acf",
    }

    @staticmethod
    def field(prompts, path: str) -> str:
        value = prompts
        for name in path.split("."):
            value = getattr(value, name)
        return value

    def test_the_set_sends_its_texts_byte_for_byte(self):
        for path, expected in self.DIGESTS.items():
            with self.subTest(field=path):
                self.assertEqual(digest(self.field(DEFAULT, path)), expected)


class PromptSetTableTests(unittest.TestCase):
    def test_the_table_names_the_one_set_and_it_is_the_default(self):
        self.assertEqual(PROMPT_SETS, {"default": DEFAULT})
        self.assertEqual((DEFAULT.name, DEFAULT_PROMPT_SET), ("default", "default"))
        self.assertIs(get_prompt_set(""), DEFAULT)

    def test_an_unknown_set_names_the_known_ones(self):
        with self.assertRaisesRegex(ValueError, r"Unknown prompt version 'v9'; known: default$"):
            get_prompt_set("v9")


class FrameTests(unittest.TestCase):
    CLIENT = "You are Sam.\nNever read out a full card number."
    MARKERS = f"<application_instructions>\n{CLIENT}\n</application_instructions>"

    def test_the_talker_reads_the_clients_instructions_word_for_word_inside_its_frame(self):
        # Its role before them, how they apply after them, then the routing rules; the work line comes after the
        # conversation, in the runtime data.
        frontend = ScriptedFrontend([TalkerDecision("delegate", "")])
        asyncio.run(Talker(frontend, DEFAULT).decide(self.CLIENT, [{"role": "user", "content": "Yes."}], "WORK"))
        call = frontend.calls[0]
        self.assertEqual(
            call["instructions"], default.FRAME_BEFORE + self.CLIENT + default.FRAME_AFTER + "\n" + DEFAULT.routing
        )
        runtime = "\n" + default.RUNTIME_LABEL + "\nWORK\n\n" + default.DECISION_CUE
        self.assertEqual(call["history"][-1], {"role": "user", "content": runtime})
        self.assertIn(self.MARKERS, DEFAULT.framed(self.CLIENT))

    def test_the_writer_reads_them_word_for_word_inside_its_own_frame(self):
        self.assertIn(self.MARKERS, DEFAULT.writer_framed(self.CLIENT))
        self.assertIn("limits on what the caller may be told", DEFAULT.writer_frame_after)
        self.assertIn("say each one in turn", DEFAULT.answer)

    def test_routing_has_no_listen_and_the_talkers_schema_states_the_sets_hint(self):
        self.assertNotIn("listen", DEFAULT.routing.lower())
        schema = decision_schema(Talker(ScriptedFrontend([]), DEFAULT).spec)
        self.assertEqual(schema["properties"]["speech"]["description"], DEFAULT.call_hints.talker)


class ExampleTests(unittest.TestCase):
    # Words of the cafe's and the airline's domains: the examples come from other businesses.
    DEMO_WORDS = re.compile(r"\b(cafe|coffee|latte|croissant|order|airline|flight|reservation|baggage|cabin)", re.I)

    @staticmethod
    def examples() -> list[tuple[str, dict]]:
        """Each example's caller line and its decision, in order."""
        lines = DEFAULT.routing.split("Examples from other businesses.", 1)[1].splitlines()
        return [(lines[index - 1], json.loads(line)) for index, line in enumerate(lines) if line.startswith("{")]

    def test_the_examples_come_from_other_businesses_and_show_every_action(self):
        examples = self.examples()
        for caller, decision in examples:
            with self.subTest(caller=caller):
                self.assertTrue(caller.startswith("Caller: "))
                self.assertIsNone(self.DEMO_WORDS.search(caller + decision["speech"]))
        self.assertEqual({decision["action"] for _, decision in examples}, set(DEFAULT.actions))

    def test_the_examples_are_decisions_the_floor_keeps(self):
        talker = Talker(ScriptedFrontend([]), DEFAULT)
        for caller, example in self.examples():
            with self.subTest(caller=caller):
                decision = TalkerDecision(example["action"], example["speech"])
                self.assertEqual(talker.floored(decision), decision)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))

    def test_the_example_prompt_catalog_has_talker_thinker_and_channel(self):
        self.assertTrue(self.catalog["talker"]["default"])
        for key in ("talker", "thinker", "channel"):
            self.assertTrue(self.catalog[key]["content"].strip(), key)
        self.assertNotIn("web_search", self.catalog["thinker"]["content"])

    def test_every_line_said_word_for_word_is_read_by_name(self):
        for name, text in self.catalog[LINES_KEY].items():
            with self.subTest(line=name):
                self.assertEqual(catalog_line(name), text)
        self.assertEqual(catalog_line("holding"), "One moment, I'm still working on your earlier request.")
        with self.assertRaisesRegex(KeyError, "no_such_line"):
            catalog_line("no_such_line")


if __name__ == "__main__":
    unittest.main()
