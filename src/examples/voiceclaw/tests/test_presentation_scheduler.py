# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from dataclasses import replace

import pytest

from voiceclaw.application.presentation import PresentationScheduler
from voiceclaw.domain.models import (
    DisplayPayload,
    DisplayState,
    FrontendActivity,
    ModelActivity,
    PresentationPriority,
    PresentationRecord,
    ResultState,
    SpeechState,
    WorkResult,
)


def _result(
    suffix: str,
    sequence: int,
    *,
    session_id: str = "session-a",
    priority: PresentationPriority = PresentationPriority.CURRENT_RESULT,
) -> WorkResult:
    return WorkResult(
        presentation_id=f"presentation-{suffix}",
        session_id=session_id,
        attachment_id=f"attachment-{session_id}",
        work_id=f"work-{suffix}",
        result_id=f"result-{suffix}",
        sequence=sequence,
        display=DisplayPayload(
            kind="work.result",
            title=f"Result {suffix}",
            body=f"Detailed result {suffix}",
            data={"result": suffix},
        ),
        speech_text=f"Result {suffix} is ready.",
        priority=priority,
    )


def test_two_results_display_while_model_busy_then_speak_in_order() -> None:
    scheduler = PresentationScheduler()
    scheduler.set_activity(
        "session-a",
        FrontendActivity(connected=True, model=ModelActivity.GENERATING),
    )

    scheduler.offer(_result("a", 41))
    scheduler.offer(_result("b", 42))

    assert scheduler.claim_next_display("session-a").presentation_id == "presentation-a"
    assert scheduler.claim_next_display("session-a").presentation_id == "presentation-b"
    assert scheduler.claim_next_display("session-a") is None
    assert scheduler.claim_next_speech("session-a") is None
    assert scheduler.pending_speech_count("session-a") == 2

    scheduler.set_activity("session-a", FrontendActivity(connected=True))
    first = scheduler.claim_next_speech("session-a")
    assert first is not None
    assert first.presentation_id == "presentation-a"
    assert scheduler.claim_next_speech("session-a") is None

    scheduler.mark_speech_started("session-a", first.presentation_id)
    scheduler.mark_speech_completed("session-a", first.presentation_id)
    second = scheduler.claim_next_speech("session-a")
    assert second is not None
    assert second.presentation_id == "presentation-b"


def test_display_and_speech_receipts_are_independent() -> None:
    scheduler = PresentationScheduler()
    scheduler.set_activity(
        "session-a",
        FrontendActivity(connected=True, model=ModelActivity.GENERATING),
    )
    scheduler.offer(_result("a", 1))

    displayed = scheduler.mark_displayed("session-a", "presentation-a")

    assert displayed.display_state is DisplayState.DELIVERED
    assert displayed.speech_state is SpeechState.QUEUED

    scheduler.set_activity("session-a", FrontendActivity(connected=True))
    claimed = scheduler.claim_next_speech("session-a")
    assert claimed is not None
    scheduler.mark_speech_started("session-a", "presentation-a")
    completed = scheduler.mark_speech_completed("session-a", "presentation-a")
    assert completed.result_state is ResultState.ACKNOWLEDGED


def test_priority_precedes_sequence_for_unclaimed_speech() -> None:
    scheduler = PresentationScheduler()
    scheduler.set_activity("session-a", FrontendActivity(connected=True))
    scheduler.offer(_result("normal", 1, priority=PresentationPriority.CURRENT_RESULT))
    scheduler.offer(_result("question", 2, priority=PresentationPriority.BLOCKING_INPUT))

    claimed = scheduler.claim_next_speech("session-a")

    assert claimed is not None
    assert claimed.presentation_id == "presentation-question"


def test_sessions_are_isolated() -> None:
    scheduler = PresentationScheduler()
    scheduler.set_activity(
        "session-a",
        FrontendActivity(connected=True, model=ModelActivity.GENERATING),
    )
    scheduler.set_activity("session-b", FrontendActivity(connected=True))
    scheduler.offer(_result("a", 1, session_id="session-a"))
    scheduler.offer(_result("b", 1, session_id="session-b"))

    assert scheduler.claim_next_speech("session-a") is None
    claimed_b = scheduler.claim_next_speech("session-b")
    assert claimed_b is not None
    assert claimed_b.session_id == "session-b"


def test_duplicate_result_is_idempotent() -> None:
    scheduler = PresentationScheduler()
    result = _result("a", 1)

    first = scheduler.offer(result)
    duplicate = scheduler.offer(result)

    assert first.created is True
    assert duplicate.created is False
    assert len(scheduler.records("session-a")) == 1
    assert scheduler.claim_next_display("session-a") is not None
    assert scheduler.claim_next_display("session-a") is None


def test_duplicate_result_across_attachment_epochs_is_idempotent() -> None:
    scheduler = PresentationScheduler()
    result = _result("a", 7)

    first = scheduler.offer(result)
    replay = scheduler.offer(
        replace(
            result,
            attachment_id="attachment-reconnected",
            sequence=result.sequence,
        )
    )

    assert first.created is True
    assert replay.created is False
    assert replay.record == first.record
    assert replay.record.attachment_id == result.attachment_id
    assert replay.record.sequence == result.sequence


def test_conflicting_presentation_id_is_rejected() -> None:
    scheduler = PresentationScheduler()
    result = _result("a", 1)
    scheduler.offer(result)

    with pytest.raises(ValueError, match="presentation identity conflicts"):
        scheduler.offer(
            replace(
                result,
                work_id="work-conflicting",
                result_id="result-conflicting",
                display=DisplayPayload(kind="work.result", title="Conflicting result"),
            )
        )

    assert scheduler.records("session-a") == (PresentationRecord.from_result(result),)


def test_replayed_result_with_new_local_presentation_id_is_idempotent() -> None:
    scheduler = PresentationScheduler()
    result = _result("a", 1)
    scheduler.offer(result)

    replay = scheduler.offer(replace(result, presentation_id="presentation-replayed"))

    assert replay.created is False
    assert replay.record.presentation_id == result.presentation_id
    assert len(scheduler.records("session-a")) == 1


def test_interrupted_speech_is_not_automatically_requeued() -> None:
    scheduler = PresentationScheduler()
    scheduler.set_activity("session-a", FrontendActivity(connected=True))
    scheduler.offer(_result("a", 1))
    claimed = scheduler.claim_next_speech("session-a")
    assert claimed is not None

    scheduler.mark_speech_started("session-a", claimed.presentation_id)
    interrupted = scheduler.mark_speech_interrupted("session-a", claimed.presentation_id)

    assert interrupted.speech_state is SpeechState.INTERRUPTED
    assert scheduler.claim_next_speech("session-a") is None
