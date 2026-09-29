# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for Frontend/Backend Agent barge-in state."""

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, UserStartedSpeakingFrame

from examples.frontend_backend_agent.src.barge_in import BargeInState


def test_barge_in_state_marks_user_speech_during_bot_audio_once() -> None:
    """Active bot audio marks exactly one consumed interruption."""
    state = BargeInState()

    state.observe(BotStartedSpeakingFrame())
    state.observe(UserStartedSpeakingFrame())

    assert state.consume_interrupted_speech() is True
    assert state.consume_interrupted_speech() is False


def test_barge_in_state_does_not_mark_idle_user_speech() -> None:
    """Idle user speech must not be reported as a barge-in."""
    state = BargeInState()

    state.observe(BotStartedSpeakingFrame())
    state.observe(BotStoppedSpeakingFrame())
    state.observe(UserStartedSpeakingFrame())

    assert state.consume_interrupted_speech() is False


def test_new_user_turn_replaces_stale_interruption_marker() -> None:
    """A later idle turn clears an unconsumed earlier interruption marker."""
    state = BargeInState()

    state.observe(BotStartedSpeakingFrame())
    state.observe(UserStartedSpeakingFrame())
    state.observe(BotStoppedSpeakingFrame())
    state.observe(UserStartedSpeakingFrame())

    assert state.consume_interrupted_speech() is False
