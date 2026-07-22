# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from importlib import util
from pathlib import Path

from pipecat.frames.frames import LLMMessagesAppendFrame, LLMTextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

_CI_BOT_PATH = Path(__file__).resolve().parents[1] / "pipecat_evals" / "ci" / "ci_bot.py"
_SPEC = util.spec_from_file_location("ci_eval_bot", _CI_BOT_PATH)
assert _SPEC and _SPEC.loader
ci_eval_bot = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_eval_bot)


class CIEvalBotTests(unittest.TestCase):
    def test_empty_user_message_emits_observable_fallback_response(self) -> None:
        async def run() -> None:
            responder = ci_eval_bot.CIEvalResponder()
            frames = []

            async def _push_frame(frame, _direction=FrameDirection.DOWNSTREAM) -> None:
                frames.append(frame)

            responder.push_frame = _push_frame
            await responder.process_frame(
                LLMMessagesAppendFrame(messages=[{"role": "user", "content": []}]),
                FrameDirection.DOWNSTREAM,
            )

            transcriptions = [frame.text for frame in frames if isinstance(frame, TranscriptionFrame)]
            responses = [frame.text for frame in frames if isinstance(frame, LLMTextFrame)]
            self.assertEqual(transcriptions, ["(empty message)"])
            self.assertEqual(responses, ["CI eval bot turn 1: (empty message)"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
