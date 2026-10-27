# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The prompt sets and the example's prompt catalog, pinned.

Each text a model reads is pinned by its sha256, so an unintended change fails here; update a digest on purpose when a
text changes.
"""

import asyncio
import hashlib
import unittest
from pathlib import Path

import yaml

from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_schema
from examples.frontend_backend_live.prompts import (
    DEFAULT_PROMPT_SET,
    PROMPT_SETS,
    get_prompt_set,
    promptset,
    v1,
    v2,
    v3,
)
from examples.frontend_backend_live.prompts.catalog import LINES_KEY, catalog_line
from tests.unit.cascade_helpers import ScriptedFrontend

CATALOG = Path(__file__).resolve().parents[4] / "src/examples/frontend_backend_live/prompts.yaml"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class PromptTextTests(unittest.TestCase):
    # Each set's own texts.
    DIGESTS = {
        (v1, "ROUTING_INSTRUCTIONS"): "40365a9f3d1220147e84a8b2c989095f7aad440b15832b14dd20c91f92a8b5ea",
        (v1, "COMMENTARY_INSTRUCTIONS"): "e50f058f61635840fc3f6a54b46044c0669c5ec78f99373968d632b382d9e055",
        (v1, "ANSWER_INSTRUCTIONS"): "cf6778d693ac771d41001d4bbe7eb3aedb88970195d0d36203294f0c339a73b2",
        (v2, "ROUTING_INSTRUCTIONS"): "6547b7fc7b545b164ff788f5865b62feb12e8fed27dc4cac186f14854d28025d",
        (v2, "COMMENTARY_INSTRUCTIONS"): "dd3ce14d515204c756dce38a0a14c9564057e4a69ddf2e6ecb792e1d28ce5c68",
        (v2, "ANSWER_INSTRUCTIONS"): "91a3f2b0efe7de3adf2fd64d394bc3c59dee09f129bcaa09a0370d8759383666",
        (v2, "FRAME_BEFORE"): "e36e628b82fc794707c240327adcd36c89e4dafc998767f8be39a226960a2f99",
        # The frame, ending with the presence bullet.
        (v2, "FRAME_AFTER"): "7172d743577947a46ece822ca58f73022d668ee2d76dcacee499fcd14cea9cb7",
        # v3, which is not the default set.
        (v3, "FRAME_BEFORE"): "398e9d10dd8ce476e2dd3033d5a03511fa0501d23611ccb122fbb93a92080c00",
        (v3, "FRAME_AFTER"): "c387b91b3bd07098af255585e504df3d74918afc49ded42f455eac2de6042a2e",
        (v3, "ROUTING_INSTRUCTIONS"): "73e943acb918d6a0f44a00baf9c58f783245734d14961e584026290ffa309320",
        (v3, "COMMENTARY_INSTRUCTIONS"): "bcf760ba13b3b0b18d8d7212d353a13026ec58db8acc5927a4b9f809b6a363a3",
        (v3, "ANSWER_INSTRUCTIONS"): "5fc965afb9c2613f0bc94d5c58b07fe0a3f718c2e2a1190ead46809ff78824fe",
        (v3, "WRITER_FRAME_BEFORE"): "b8057379f8e3512801380101a4f19f429a4d99c14b7ef4ff32df0b28ce7b3a7d",
        (v3, "WRITER_FRAME_AFTER"): "45b59c51c9ff1556484be1a001fe60ed13dec2b31f60130fb80bd217bcb02641",
    }
    # v3's own field texts, beside the ones it shares with v1 and v2.
    V3_FIELDS = {
        "update_label": "e3cbd8602b29997448268712580fb7ea3cd00beb329ea906cc1960a70a7ab04e",
        "commentary_label": "c70c29135fefe4e855b2fc60acfabbe2f5a7ee3cc8789c99d072edb0d2ad07bc",
        "caller_label": "1ec7b80cf30f2317fb81885fc40658a1e482ba2955eb96524444d85016c61194",
        "call_hints.talker": "96dae2847a434d2540d4931c75093be871f34993e0e1356e115062600a8e987a",
        "call_hints.answer": "3c59456f93f529c0b6b69de2e6c3530a23e1df5db85a55fd6641fe1cb18d3339",
    }
    # The cascade's texts in the sets: the work note's, the typed request's label, the writer's update label and notes,
    # the schema hints and the normalizer's prompt.
    MOVED = {
        "work_in_progress": "54cd1356597fb8fa38f4018e7e1982f7a50a5c7db90dc4969e5a15c6102280f7",
        "no_work_in_progress": "3f24d1a22ca78b922fc3532e3078e0c76961a8d5d60b9c609ce43b1a80a6dc27",
        "typed_request_label": "d627e456ba2045be399c0b24fab0477a3dc4bb083a4b5a9efe344e39439e659a",
        "update_label": "d18fd16595eb301de83a37798401a1569310f00bacc2ab0cfdf7b4e3f26e8951",
        "commentary_label": "d18fd16595eb301de83a37798401a1569310f00bacc2ab0cfdf7b4e3f26e8951",
        "writer_notes.told": "45c8846c6b2dc5796d59eb06c5a5d4193086fa99b79584eb532e42486fd55ff0",
        "writer_notes.must_speak": "01ebcbcb38943858b03ffa24c742e70027611580f65ab69c8a242412d9260c1e",
        "writer_notes.newer": "6037e68aa4f69d550e9321c065968f89d65f259cc561f94e27d2504ecca26237",
        "call_hints.talker": "bdebd0499fac71be986f3dd8415d5574796dc1dcf4d808532c8885f8a7bb84a5",
        "call_hints.commentary": "d2a763a8412673e1b9a8b66aa046728e37dfb54295c6616e56db7e6d532a9052",
        "call_hints.answer": "6673036369d28eb158684114ae37d75239e6f3bfa8c41997aa28bf0454d79eb1",
        # The normalizer's hint is the answer's text, stated per role.
        "call_hints.normalizer": "6673036369d28eb158684114ae37d75239e6f3bfa8c41997aa28bf0454d79eb1",
        "normalizer_prompt.instructions": "524156dd50e24028998ad8c943f97b19c1d92f40c1ede4b84d8174046888a4dc",
        "normalizer_prompt.newest_turn": "673e1768822895ba593fbfbb07d3e6b4f170a0e4ac8d31f1735d6c766bcd2e07",
    }

    @staticmethod
    def field(prompts, path: str) -> str:
        value = prompts
        for name in path.split("."):
            value = getattr(value, name)
        return value

    def test_each_sets_own_texts_are_unchanged(self):
        for (module, name), expected in self.DIGESTS.items():
            with self.subTest(prompt=f"{module.__name__.rsplit('.', 1)[-1]}.{name}"):
                self.assertEqual(digest(getattr(module, name)), expected)

    def test_v1_and_v2_send_the_moved_texts_byte_for_byte(self):
        for prompts in (v1.PROMPT_SET, v2.PROMPT_SET):
            for path, expected in self.MOVED.items():
                with self.subTest(prompt_set=prompts.name, field=path):
                    self.assertEqual(digest(self.field(prompts, path)), expected)

    def test_v3_keeps_the_shared_texts_and_pins_its_own(self):
        for path, expected in self.MOVED.items():
            with self.subTest(field=path):
                self.assertEqual(digest(self.field(v3.PROMPT_SET, path)), self.V3_FIELDS.get(path, expected))
        for path, expected in self.V3_FIELDS.items():
            with self.subTest(field=path):
                self.assertEqual(digest(self.field(v3.PROMPT_SET, path)), expected)
        self.assertEqual((v3.PROMPT_SET.frame_before, v3.PROMPT_SET.frame_after), (v3.FRAME_BEFORE, v3.FRAME_AFTER))
        self.assertEqual(
            (v3.PROMPT_SET.writer_frame_before, v3.PROMPT_SET.writer_frame_after),
            (v3.WRITER_FRAME_BEFORE, v3.WRITER_FRAME_AFTER),
        )

    def test_v1_and_v2_route_with_speak_or_delegate_and_v1_has_no_frame(self):
        self.assertEqual((v1.PROMPT_SET.actions, v2.PROMPT_SET.actions), (("speak", "delegate"), ("speak", "delegate")))
        self.assertEqual((v1.PROMPT_SET.frame_before, v1.PROMPT_SET.frame_after), ("", ""))
        self.assertEqual((v2.PROMPT_SET.frame_before, v2.PROMPT_SET.frame_after), (v2.FRAME_BEFORE, v2.FRAME_AFTER))
        for prompts in (v1.PROMPT_SET, v2.PROMPT_SET):
            with self.subTest(prompt_set=prompts.name):
                # Nothing that later sets add as data changes what v1 and v2 send.
                self.assertEqual(prompts.caller_label, "")
                self.assertEqual(prompts.writer_framed("Be Remy."), "Be Remy.")

    def test_the_answer_ready_label_is_the_work_notes_new_state(self):
        self.assertEqual(promptset.ANSWER_READY_LABEL, "answer ready, said next")
        self.assertEqual({prompts.answer_ready_label for prompts in PROMPT_SETS.values()}, {"answer ready, said next"})

    def test_the_writer_notes_compose_as_the_presenter_wrote_them(self):
        notes = v2.PROMPT_SET.writer_notes
        self.assertEqual(
            notes.told.format(told="Let me check."),
            '\n(you already told the caller: "Let me check." -- do not repeat that, report only the result)',
        )
        self.assertTrue(notes.newer.format(requests='"a"; "b"').startswith("\n(the caller has since asked something"))


class PromptSetTableTests(unittest.TestCase):
    def test_the_table_names_each_set_once_and_the_default_is_v2(self):
        self.assertEqual(sorted(PROMPT_SETS), ["v1", "v2", "v3"])
        for name, prompts in PROMPT_SETS.items():
            self.assertEqual(prompts.name, name)
        self.assertEqual(DEFAULT_PROMPT_SET, "v2")
        self.assertIs(get_prompt_set(""), PROMPT_SETS["v2"])

    def test_an_unknown_set_names_the_known_ones(self):
        with self.assertRaisesRegex(ValueError, r"Unknown prompt version 'v9'; known: v1, v2, v3"):
            get_prompt_set("v9")


class FrameTests(unittest.TestCase):
    # The presence bullet, word for word.
    F2_BULLET = """- Where they say which turns go to the backend, which turns you answer yourself, or when to stay
  silent, follow them. Staying silent while the backend works means delegating its turns with empty
  speech; a caller who asks whether you are still there, or asks you to hold on, still gets a short
  reply from you."""

    def test_v2s_frame_ends_with_the_measured_presence_bullet(self):
        self.assertTrue(v2.FRAME_AFTER.endswith(self.F2_BULLET))
        self.assertEqual(v2.FRAME_AFTER.count("Staying silent"), 1)  # the bullet is the only presence bullet

    def test_v2_frames_the_clients_instructions_word_for_word_and_v1_does_not(self):
        # The talker reads the client's text inside v2's frame: its role before it, how the text applies after it.
        client = "You are a helpful agent.\nYou can book tables."
        for version, before, after in (("v1", "", ""), ("v2", v2.FRAME_BEFORE, v2.FRAME_AFTER)):
            with self.subTest(prompt_version=version):
                frontend, prompts = ScriptedFrontend([TalkerDecision("delegate", "")]), get_prompt_set(version)
                asyncio.run(Talker(frontend, prompts).decide(client, [{"role": "user", "content": "Yes."}], "WORK"))
                sent = frontend.calls[0]["instructions"]
                self.assertEqual(sent, before + client + after + "\n" + prompts.routing + "\nWORK")
        self.assertIn(
            f"<application_instructions>\n{client}\n</application_instructions>", get_prompt_set("v2").framed(client)
        )

    def test_routing_has_no_listen_and_the_talkers_schema_only_allows_speak_or_delegate(self):
        self.assertNotIn("listen", v1.ROUTING_INSTRUCTIONS.lower())
        for version in ("v1", "v2"):
            with self.subTest(prompt_version=version):
                talker = Talker(ScriptedFrontend([]), get_prompt_set(version))
                schema = decision_schema(talker.spec)
                self.assertEqual(schema["properties"]["action"]["enum"], ["speak", "delegate"])
                self.assertEqual(
                    schema["properties"]["speech"]["description"], get_prompt_set(version).call_hints.talker
                )


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

    def test_the_reviewed_lines_name_no_system_and_say_through_once(self):
        lines = self.catalog[LINES_KEY]
        self.assertNotIn("system", lines["app_wait_line"])
        self.assertEqual(lines["unclear"].count("through"), 1)
        self.assertIn("can't tell whether", lines["unclear"])


if __name__ == "__main__":
    unittest.main()
