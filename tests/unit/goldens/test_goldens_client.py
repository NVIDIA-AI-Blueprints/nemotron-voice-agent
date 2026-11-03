# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

from tests.unit.goldens.harness import WORK_RUNNING, GoldenCase, answer_round, decision, result_item, tool_call_round

# A typed image part. The server never decodes it: it only carries the part or drops it.
IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="
TEAS = '{"teas": ["Earl Grey", "Matcha", "Rooibos"]}'


def typed_item(text, event_id, *, image_url=None):
    """A ``response.item.create`` that queues a typed user message (text, and an image part when given)."""
    content = [{"type": "input_text", "text": text}]
    if image_url:
        content.append({"type": "input_image", "image_url": image_url})
    item = {"type": "message", "role": "user", "content": content}
    return {"type": "response.item.create", "event_id": event_id, "item": item}


def response_create(event_id):
    return {"type": "response.create", "event_id": event_id}


def context_append(kind, content, event_id):
    """A ``session.<kind>.append`` (instructions, thinking or commentary) that names no delegation."""
    return {"type": f"session.{kind}.append", "event_id": event_id, "delegation_id": None, "content": content}


class ClientCommandGoldens(GoldenCase):
    """Goldens for the client's own commands: typed input, settings updates and context appends."""

    async def serve(self, session=None, **script):
        """Start the call as ``GoldenCase.serve`` does; then hold each backend round until the rest is quiet.

        The scripted backend answers as soon as its round starts, so a lead-in spoken a moment earlier may not have
        reached the talker's context yet, and the writer's history and the talker's memory would hold it or not
        depending on task scheduling alone. A real backend takes far longer than a lead-in takes to reach the context.
        So, in this file only, each backend request is sent once nothing but the delegated work is busy: the turn or
        append that delegated has finished and every utterance is in the talker's context. The request itself is
        unchanged.
        """
        started = await super().serve(session, **script)
        create = self.backend.create

        async def create_once_the_rest_is_quiet(**request):
            await self.wait_until(
                lambda: all(reason == WORK_RUNNING for reason in self.pending()),
                "the rest of the call to be quiet before a backend round",
            )
            return await create(**request)

        # FakeChat is its own .chat.completions, so the backend model's calls reach this wrapper.
        self.backend.create = create_once_the_rest_is_quiet
        return started

    async def refused(self, command):
        """Send ``command`` and return the ``error`` that answers it; the golden pins its code."""
        after = len(self.ws.from_server)
        self.send(command)
        return await self.wait_for(
            "error", after=after, where=lambda e: e["error"]["client_event_id"] == command["event_id"]
        )

    async def update_backend(self, responses, event_id):
        """Send a ``session.update`` that changes ``delegation.responses``; return ``session.updated``."""
        after = len(self.ws.from_server)
        patch = {"delegation": {"type": "responses", "responses": responses}}
        self.send({"type": "session.update", "event_id": event_id, "session": patch})
        return await self.wait_for("session.updated", after=after)

    async def acknowledged(self, kind, event_id):
        """Feed caller audio until the append's acknowledgment is released, and return it.

        An acknowledgment waits until the input clock passes the end of the 200 ms span its append claims
        (``live/events/timeline.py``). The client sends no audio otherwise, so two 100 ms packets of silence are
        enough; ``wait_quiet`` does not wait for acknowledgments.
        """
        self.feed_silence(2)
        return await self.wait_for(f"session.{kind}.appended", where=lambda e: e["client_event_id"] == event_id)

    async def test_g11_typed_input(self):
        """G11: typed input, with no caller speech.

        A typed message with a text part and an image part, then response.create, starts a typed run, and the
        backend calls the client's get_menu tool. While the run waits for the output, the client types a second
        message and asks to continue too early, which is refused with the official function_call_outputs_required.
        The output and a response.create then continue the run: the continuation carries the output, then the
        second message. Each typed message enters the talker's memory, so the writer speaks the backend's
        answer knowing both.
        """
        await self.serve(
            frontend=[decision("speak", "Rooibos is our caffeine-free tea.")],  # the writer; no turn reaches the talker
            backend=[tool_call_round(), answer_round("Rooibos is the only caffeine-free tea.")],
        )
        self.send(typed_item("What teas do you have?", "typed_question", image_url=IMAGE_URL))
        self.send(response_create("typed_run"))
        call = await self.wait_for_call("get_menu")
        # The client types again once the server waits for it: the first round has ended.
        await self.wait_quiet()
        self.send(typed_item("Only caffeine-free ones, please.", "typed_correction"))
        await self.refused(response_create("too_early"))
        await self.wait_quiet()
        self.send({**result_item(call["call_id"], TEAS), "event_id": "output"})
        self.send(response_create("continue"))
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G11")

    async def test_g12_session_update_then_a_delegation(self):
        """G12: session.update, then a delegation.

        A first update changes the backend's model, instructions and output limit, and turns parallel tool calls
        off. The caller asks a question; the talker delegates with a lead-in, and the backend's first round, on the
        new settings, calls the client's get_menu tool. While the run waits for the output, a second update changes
        only the instructions. The client returns the output and continues: the continuation is a request of its own,
        which carries the session's settings of its time, and the Work takes them, so the
        continuation round goes out with the second update's instructions. The writer speaks the backend's answer.
        """
        await self.serve(
            frontend=[decision("delegate", "Let me check."), decision("speak", "We have Earl Grey and Matcha.")],
            backend=[tool_call_round(), answer_round("We have Earl Grey and Matcha.")],
        )
        await self.update_backend(
            {
                "model": "org/other-model",
                "instructions": "Answer from the menu only.",
                "max_output_tokens": 300,
                "parallel_tool_calls": False,
            },
            "first_update",
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        # The second update reaches the run while it waits for the client, between its two rounds.
        await self.wait_quiet()
        await self.update_backend({"instructions": "Answer in one short sentence."}, "second_update")
        self.send(result_item(call["call_id"], TEAS))
        self.send(response_create("continue"))
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G12")

    async def test_g14_an_instruction_append(self):
        """G14: instruction appends.

        The caller says hello and the talker answers. A first instruction append is decided at once, with no caller
        turn: the talker greets the caller by name. A second instruction append makes the talker delegate with a
        lead-in; the backend's request carries it as the current instruction, and the writer speaks the answer. Each
        acknowledgment is released once the caller's audio passes the end of its span.
        """
        await self.serve(
            frontend=[
                decision("speak", "Hello, how can I help?"),  # the talker, on the caller's hello
                decision("speak", "Welcome back, Sam."),  # the talker, on the first instruction
                decision("delegate", "Let me check today's special."),  # the talker, on the second instruction
                decision("speak", "Today's special is a Matcha latte."),  # the writer
            ],
            backend=[answer_round("The special today is a Matcha latte.")],
        )
        await self.say("Hi.")
        await self.wait_quiet()
        self.send(context_append("instructions", "The caller is Sam, a regular. Greet Sam by name.", "greet"))
        # Quiet only after the talker has decided the instruction and its reply has reached the text-to-speech.
        await self.wait_quiet()
        await self.acknowledged("instructions", "greet")
        self.send(context_append("instructions", "Tell the caller about today's special.", "special"))
        await self.wait_quiet()
        await self.acknowledged("instructions", "special")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G14")

    async def test_g15_a_thinking_append_reaches_the_next_delegation_once(self):
        """G15: a thinking append reaches the next delegation once.

        The client appends silent context before the caller speaks, and nothing is said. The caller asks a question;
        the talker sees the context and delegates with a lead-in, and the backend's request carries the context. The
        caller asks a second question, which the talker also delegates: its request carries the voice turns since
        the first delegation and no application context. The writer speaks each answer.
        """
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),  # the talker, on the first question
                decision("speak", "We have Earl Grey and Matcha."),  # the writer
                decision("delegate", "Let me look."),  # the talker, on the second question
                decision("speak", "We have a decaf espresso and a decaf latte."),  # the writer
            ],
            backend=[
                answer_round("We have Earl Grey and Matcha."),
                answer_round("Decaf espresso and decaf latte."),
            ],
        )
        self.send(context_append("thinking", "The caller is a regular and prefers decaf.", "context"))
        await self.acknowledged("thinking", "context")
        await self.wait_quiet()
        await self.say("What teas do you have?")
        await self.wait_quiet()
        await self.say("And coffee?")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G15")

    async def test_x6_voice_turns_then_a_typed_run_then_a_voice_delegation(self):
        """X6: voice turns, then a typed run, then a voice delegation.

        The caller and the talker exchange two spoken turns, and nothing is delegated. The client then types a
        question and runs it; the question enters the talker's memory, and the writer speaks the backend's
        answer. The typed run leaves the voice turns said before it to the next delegation. The caller then asks a
        question the talker delegates with a lead-in: its request carries those turns, the typed run's answer and the
        new question.
        """
        await self.serve(
            frontend=[
                decision("speak", "Hello, how can I help?"),  # the talker, on the caller's hello
                decision("speak", "Sure. Would you like tea or coffee?"),  # the talker
                decision("speak", "Yes, we are open on Sunday from nine to five."),  # the writer, typed run
                decision("delegate", "Let me check."),  # the talker, on the spoken question
                decision("speak", "We have Earl Grey and Matcha."),  # the writer
            ],
            backend=[
                answer_round("Yes. On Sunday the cafe is open from 9:00 to 17:00."),
                answer_round("We have Earl Grey and Matcha."),
            ],
        )
        await self.say("Hi.")
        await self.wait_quiet()
        await self.say("I'd like something warm to drink.")
        await self.wait_quiet()
        self.send(typed_item("Is the cafe open on Sunday?", "typed_question"))
        self.send(response_create("typed_run"))
        await self.wait_quiet()
        await self.say("Tea, please. What teas do you have?")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("X6")
