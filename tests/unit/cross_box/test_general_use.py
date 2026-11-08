# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""The general-use suite: each change of the iteration checked outside customer service, with fake models.

A café order through the UI bot's core, a generic Live client (one that runs its own tools, and one that sends plain
function tools to the core), and sessions in other languages. Every value is made up.
"""

import unittest

from examples.frontend_backend_live.config_manager.schema import ExampleConfig
from examples.frontend_backend_live.delegation_core.config import CoreSettings, WriteGateSettings
from examples.frontend_backend_live.heard_log import HeardLog
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.spoken import ends_spelled, spelled_runs
from examples.frontend_backend_live.talker import relay_line
from examples.shared.delegation_contract import ReplyUnheard, ends_with_question, sentences
from tests.unit.core_helpers import CORE_TEXTS, EXAMPLE_SETTINGS
from tests.unit.delegation_core.helpers import (
    TEXTS,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    answer,
    ask,
    call_item,
    function_calls,
    read_all,
)
from tests.unit.delegation_core.test_im_calls import Round, outputs
from tests.unit.frontend_backend_live.test_voice import StillVoice
from tests.unit.frontend_helpers import PROMPTS

ON = WriteGateSettings(enabled=True)


def tool(name: str, **hints) -> dict:
    made = {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
    return {**made, "annotations": hints} if hints else made


LOOKUP, CHANGE = tool("find_slot"), tool("book_slot")  # a client's plain tools: declared by nobody
LABELED = tool("find_slot", readOnlyHint=True)
DECLARED = tool("book_slot", readOnlyHint=False, destructiveHint=True)


def gate_holds(traces) -> list[dict]:
    return [fields for kind, fields in traces if kind == "im.write_gate" and fields["outcome"] != "ran"]


class CafeTests(IMCase):
    """The café flow through the UI bot's core."""

    async def test_a_cafe_order_through_the_ui_bot_core_is_never_held(self):
        # The café's tools are server tools of the core: never gated, even with the gate on and a cut confirmation.
        adapter = ScriptedAdapter([OwnCall("manage_cart"), answer("A latte is in your cart. Anything else?")])
        im = self.session(adapter, write_gate=ON)
        stream = im.admit(ask("A latte."), delegation_id="D1", reply_unheard=ReplyUnheard.QUESTION).stream
        await read_all(stream)
        self.assertEqual((adapter.runs[0].routes, gate_holds(self.traces)), (["backend"], []))

    async def test_anything_else_after_each_item_is_recorded_and_listed(self):
        adapter = ScriptedAdapter(*[[answer(f"{item} added. Anything else?")] for item in ("Latte", "Muffin")])
        im = self.session(adapter)
        for number, item in enumerate(("A latte.", "A muffin."), 1):
            await read_all(im.admit(ask(item), delegation_id=f"D{number}", reply_unheard=ReplyUnheard.NONE).stream)
        self.assertEqual(im.question_lines(), [{"question": "Anything else?", "asked": 2, "heard_in_full": 1}])

    def test_the_situation_bullet_is_todays_and_a_missed_lookup_gets_no_shape_hint(self):
        from tests.unit.frontend_backend_live.test_prompts import SITUATION_BULLET

        settings = CoreSettings.parse(ExampleConfig.model_validate(EXAMPLE_SETTINGS).ui_bot_core, CORE_TEXTS)
        self.assertIn(SITUATION_BULLET, settings.texts.caller_note)
        self.assertNotIn("shape_hints", settings.backend.app_output_steps)  # the ID-shape hint is off by default

    def test_a_b_l_t_gets_a_spelled_note_and_a_hold_and_drinks_get_neither(self):
        log = HeardLog(PROMPTS.cascade)
        log.caller("can I get a B L T")
        log.caller("two large lattes and a muffin")
        spelled = PROMPTS.cascade.transcript.spelled
        caller = PROMPTS.cascade.transcript.caller
        self.assertEqual(
            [text for _, text in log.lines()],
            [f"{caller}: can I get a B L T [{spelled}: aBLT]", f"{caller}: two large lattes and a muffin"],
        )
        self.assertTrue(ends_spelled("can I get a B L T"))
        self.assertFalse(ends_spelled("two large lattes and a muffin"))


class LiveClientTests(IMCase):
    """A generic Live client: plain tools to the core, or its own tools with commentary."""

    async def first(self, *scripts, tools, fact=ReplyUnheard.QUESTION):
        adapter = ScriptedAdapter(*scripts)
        im = self.session(adapter, write_gate=ON)
        stream = im.admit(ask("Book me in."), delegation_id="D1", settings={"tools": tools}, reply_unheard=fact).stream
        return adapter, im, await read_all(stream)

    async def test_a_live_client_with_plain_tools_gets_at_most_one_held_round_per_request(self):
        lookup = [Round(call_item("find_slot", {}, "c1")), Round(call_item("find_slot", {}, "c2"))]
        adapter, _, events = await self.first(lookup, tools=[LOOKUP, CHANGE])
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["refused", "app"], ["c2"]))
        self.traces.clear()
        _, _, events = await self.first(
            [Round(call_item("find_slot", {}, "c1"))], tools=[LOOKUP], fact=ReplyUnheard.NONE
        )
        self.assertEqual((function_calls(events), gate_holds(self.traces)), (["c1"], []))  # every answer heard

    async def test_a_live_client_that_labels_its_lookup_read_only_is_never_held_for_it(self):
        _, _, events = await self.first([Round(call_item("find_slot", {}, "c1"))], tools=[LABELED])
        self.assertEqual(function_calls(events), ["c1"])

    async def test_a_live_client_that_declares_its_change_gets_one_hold_then_an_answer(self):
        script = [Round(call_item("book_slot", {}, f"c{n}")) for n in (1, 2)]
        adapter, im, events = await self.first([*script, answer("Shall I book the 3 pm slot?")], tools=[DECLARED])
        self.assertEqual(function_calls(events), [])
        self.assertEqual([f["outcome"] for f in gate_holds(self.traces)], ["held", "held_again"])

    async def test_a_lookup_and_a_change_in_one_round_are_held_together_at_one_round(self):
        both = Round(call_item("find_slot", {}, "c1"), call_item("book_slot", {}, "c2"))
        adapter, _, _ = await self.first([both, answer()], tools=[LOOKUP, DECLARED])
        self.assertEqual(adapter.runs[0].routes, ["refused", "refused"])
        held = set(outputs(adapter.runs[0].histories[-1]).values())
        self.assertEqual(held, {TEXTS.unheard_undeclared_output, TEXTS.unheard_write_output})

    async def test_a_barge_in_command_on_a_declared_write_is_held_once_then_runs(self):
        adapter, im, _ = await self.first(
            [Round(call_item("book_slot", {}, "c1")), answer("Book 3 pm?")],
            [Round(call_item("book_slot", {}, "c1"))],
            tools=[DECLARED],
        )
        d2 = im.admit(ask("Yes."), delegation_id="D2", settings={"tools": [DECLARED]}, reply_unheard=ReplyUnheard.NONE)
        self.assertEqual(function_calls(await read_all(d2.stream)), ["c1"])
        self.assertEqual(len(gate_holds(self.traces)), 1)  # one extra confirmation

    async def test_a_live_client_commentary_is_never_dropped_and_gets_no_lead_in_in_front(self):
        still = StillVoice()
        still.voice.commentary("Your slot is free.", "del_9")
        for _ in range(4):  # background speech cuts it four times
            still.unheard(still.say("Your slot is free."))
        still.voice._lead_in("del_10", "One moment.")
        self.assertEqual(len(still.voice.owed), 1)
        self.assertEqual([f["reason"] for f in still.kinds("acknowledgement.skipped")], ["answer_due"])

    async def test_a_live_clients_commentary_with_a_question_is_worded_and_never_read_aloud(self):
        still = StillVoice()
        still.voice.commentary("The user asked 'Is it vegan?' Tell them yes.", "del_9")
        relay = still.voice._relay_call(still.voice._take(), []).plan
        self.assertEqual(relay.asked, "")  # not checked: the writer's line stands
        decision = TalkerDecision("speak", "Yes, it is vegan.")
        self.assertEqual(relay_line(decision, relay.content, [], asked=relay.asked), "Yes, it is vegan.")


class LanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_spanish_session_splits_sentences_and_questions_by_the_shared_rule_and_keeps_its_lead_ins(self):
        text = "Su mesa está lista. ¿Quiere que la reserve?"
        self.assertEqual(sentences(text), ["Su mesa está lista.", "¿Quiere que la reserve?"])
        self.assertTrue(ends_with_question(sentences(text)[-1]))
        still = StillVoice()
        still.voice._lead_in("del_1", "Un momento, lo reviso.")
        self.assertEqual([line.text for line in still.voice.waiting_lines], ["Un momento, lo reviso."])

    def test_a_spanish_spelling_is_noted_with_the_same_code(self):
        self.assertEqual(spelled_runs("me llamo J U A N"), ["JUAN"])

    def test_a_japanese_session_whose_asr_spaces_its_characters_gets_no_note_and_no_hold(self):
        for text in ("コ ー ヒ ー を く だ さ い", "予 約 し ま す"):
            self.assertEqual((spelled_runs(text), ends_spelled(text)), ([], False))

    def test_french_and_portuguese_ordinary_speech_is_rarely_held_and_never_after_a_word(self):
        turns = [
            "il y a un problème avec ma commande",
            "je voudrais un café et un croissant",
            "c'est pour emporter",
            "merci beaucoup",
            "oui c'est bon",
            "quero o café e o bolo de hoje",
            "obrigado pela ajuda",
            "pode ser às três",
            "e o pão também",
            "sim, está certo",
        ] * 2
        held = [turn for turn in turns if ends_spelled(turn)]
        self.assertLessEqual(len(held) * 20, len(turns))  # at most 1 in 20
        for turn in turns:
            if len(turn.split()[-1].strip(".,!?")) >= 2:
                self.assertFalse(ends_spelled(turn))


if __name__ == "__main__":
    unittest.main()
