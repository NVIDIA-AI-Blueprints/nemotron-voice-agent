# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import threading

import pytest

from voiceclaw.domain.response_only import ResponseOnlyResultEnvelope
from voiceclaw.domain.volatile import (
    VolatileCancelDisposition,
    VolatileCapacityError,
    VolatileDeliveryOutcome,
    VolatileDisplayState,
    VolatilePlaybackState,
    VolatileRequestNotFoundError,
    VolatileSpeechClaimOwner,
    VolatileSpeechState,
    VolatileStaleClaimError,
    VolatileTransitionError,
    VolatileWorkLedger,
    VolatileWorkState,
)


def _complete(
    ledger: VolatileWorkLedger,
    *,
    session_id: str = "session-1",
    request_id: str = "request-1",
    speech: str | None = "The result is ready.",
):
    ledger.admit(session_id, request_id, summary="Compare two data structures")
    ledger.begin_dispatch(session_id, request_id)
    ledger.begin_invocation(session_id, request_id)
    return ledger.complete_with_result(
        session_id,
        request_id,
        result_id=f"result-{request_id}",
        presentation_id=f"presentation-{request_id}",
        envelope=ResponseOnlyResultEnvelope(display="# Result\n\nComplete.", speech=speech),
    )


def _finish_delivery(
    ledger: VolatileWorkLedger,
    *,
    session_id: str,
    request_id: str,
) -> None:
    ledger.mark_display_delivered(session_id, request_id)
    snapshot = ledger.get(session_id, request_id)
    assert snapshot.result is not None
    if snapshot.result.speech_state is VolatileSpeechState.NOT_REQUESTED:
        return
    claim = ledger.claim_speech(session_id, request_id, owner=VolatileSpeechClaimOwner.AUTOMATIC)
    assert claim is not None
    ledger.mark_speech_started(claim)
    ledger.mark_speech_delivered(claim, turn_id=f"turn-{request_id}")
    ledger.record_playback(
        claim,
        state=VolatilePlaybackState.HEARD,
        heard_through_ms=100,
        audio_end_ms=100,
    )


def test_admission_is_idempotent_but_conflicting_identity_is_rejected() -> None:
    ledger = VolatileWorkLedger()

    first = ledger.admit("session-1", "request-1", summary="  compare   the trees ")
    replay = ledger.admit("session-1", "request-1", summary="compare the trees")

    assert replay == first
    assert first.state is VolatileWorkState.LOCALLY_QUEUED
    assert first.effects_started is False
    with pytest.raises(VolatileTransitionError, match="different admission evidence"):
        ledger.admit("session-1", "request-1", summary="different request")


def test_request_identity_is_strictly_scoped_to_its_session() -> None:
    ledger = VolatileWorkLedger()
    first = ledger.admit("session-a", "same-id", summary="first session")
    second = ledger.admit("session-b", "same-id", summary="second session")

    assert first.local_request_id == second.local_request_id
    assert ledger.get("session-a", "same-id").summary == "first session"
    assert ledger.get("session-b", "same-id").summary == "second session"
    with pytest.raises(VolatileRequestNotFoundError):
        ledger.get("session-c", "same-id")


def test_cancel_before_effects_is_terminal_and_never_crosses_dispatch_fence() -> None:
    ledger = VolatileWorkLedger()
    ledger.admit("session-1", "request-1", summary="do work")

    disposition = ledger.cancel_before_effects("session-1", "request-1")
    snapshot = ledger.get("session-1", "request-1")

    assert disposition is VolatileCancelDisposition.CANCELLED
    assert snapshot.state is VolatileWorkState.CANCELLED
    assert snapshot.effects_started is False
    assert snapshot.failure_code is None
    assert ledger.cancel_before_effects("session-1", "request-1") is VolatileCancelDisposition.ALREADY_TERMINAL
    with pytest.raises(VolatileTransitionError):
        ledger.begin_dispatch("session-1", "request-1")


def test_dispatch_remains_cancellable_until_invocation_crosses_the_effect_fence() -> None:
    ledger = VolatileWorkLedger()
    ledger.admit("session-1", "request-1", summary="do work")

    dispatching = ledger.begin_dispatch("session-1", "request-1")

    assert dispatching.state is VolatileWorkState.DISPATCHING
    assert dispatching.effects_started is False
    assert ledger.cancel_before_effects("session-1", "request-1") is VolatileCancelDisposition.CANCELLED

    ledger.admit("session-1", "request-2", summary="do other work")
    ledger.begin_dispatch("session-1", "request-2")
    waiting = ledger.begin_invocation("session-1", "request-2")

    assert waiting.state is VolatileWorkState.WAITING_FOR_RESPONSE
    assert waiting.effects_started is True
    assert ledger.cancel_before_effects("session-1", "request-2") is VolatileCancelDisposition.TOO_LATE


def test_effect_fence_and_cancel_are_atomic_under_concurrency() -> None:
    ledger = VolatileWorkLedger()
    ledger.admit("session-1", "request-1", summary="do work")
    ledger.begin_dispatch("session-1", "request-1")
    barrier = threading.Barrier(3)
    outcomes: list[object] = []

    def invoke() -> None:
        barrier.wait()
        try:
            outcomes.append(ledger.begin_invocation("session-1", "request-1").state)
        except VolatileTransitionError:
            outcomes.append("invocation_rejected")

    def cancel() -> None:
        barrier.wait()
        outcomes.append(ledger.cancel_before_effects("session-1", "request-1"))

    invocation_thread = threading.Thread(target=invoke)
    cancel_thread = threading.Thread(target=cancel)
    invocation_thread.start()
    cancel_thread.start()
    barrier.wait()
    invocation_thread.join()
    cancel_thread.join()

    snapshot = ledger.get("session-1", "request-1")
    if snapshot.effects_started:
        assert snapshot.state is VolatileWorkState.WAITING_FOR_RESPONSE
        assert VolatileCancelDisposition.TOO_LATE in outcomes
    else:
        assert snapshot.state is VolatileWorkState.CANCELLED
        assert VolatileCancelDisposition.CANCELLED in outcomes
        assert "invocation_rejected" in outcomes


def test_failure_distinguishes_local_failure_from_ambiguous_post_effect_outcome() -> None:
    ledger = VolatileWorkLedger()
    ledger.admit("session-1", "local-failure", summary="local")
    failed = ledger.fail("session-1", "local-failure", failure_code="invalid_request")

    ledger.admit("session-1", "ambiguous", summary="remote")
    ledger.begin_dispatch("session-1", "ambiguous")
    ledger.begin_invocation("session-1", "ambiguous")
    unknown = ledger.fail(
        "session-1",
        "ambiguous",
        failure_code="connection_lost",
        outcome_unknown=True,
    )

    assert failed.state is VolatileWorkState.FAILED
    assert failed.effects_started is False
    assert unknown.state is VolatileWorkState.OUTCOME_UNKNOWN
    assert unknown.effects_started is True

    ledger.admit("session-1", "not-dispatched", summary="not dispatched")
    with pytest.raises(VolatileTransitionError, match="before backend effects"):
        ledger.fail(
            "session-1",
            "not-dispatched",
            failure_code="connection_lost",
            outcome_unknown=True,
        )


def test_success_commits_result_and_delivery_state_atomically() -> None:
    ledger = VolatileWorkLedger()

    snapshot = _complete(ledger)

    assert snapshot.state is VolatileWorkState.SUCCEEDED
    assert snapshot.result is not None
    assert snapshot.result.result_id == "result-request-1"
    assert snapshot.result.display_state is VolatileDisplayState.READY
    assert snapshot.result.speech_state is VolatileSpeechState.READY
    assert snapshot.result.envelope.display.startswith("# Result")

    replay = ledger.complete_with_result(
        "session-1",
        "request-1",
        result_id="result-request-1",
        presentation_id="presentation-request-1",
        envelope=ResponseOnlyResultEnvelope(display="# Result\n\nComplete.", speech="The result is ready."),
    )
    assert replay == snapshot


def test_result_without_speech_tracks_display_without_creating_a_claim() -> None:
    ledger = VolatileWorkLedger()
    snapshot = _complete(ledger, speech=None)

    assert snapshot.result is not None
    assert snapshot.result.speech_state is VolatileSpeechState.NOT_REQUESTED
    assert ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.AUTOMATIC) is None
    delivered = ledger.mark_display_delivered("session-1", "request-1")
    assert delivered.result is not None
    assert delivered.result.display_state is VolatileDisplayState.DELIVERED


def test_exactly_one_path_can_claim_result_speech() -> None:
    ledger = VolatileWorkLedger()
    _complete(ledger)
    barrier = threading.Barrier(3)
    claims = []

    def claim(owner: VolatileSpeechClaimOwner) -> None:
        barrier.wait()
        claims.append(ledger.claim_speech("session-1", "request-1", owner=owner))

    automatic = threading.Thread(target=claim, args=(VolatileSpeechClaimOwner.AUTOMATIC,))
    status = threading.Thread(target=claim, args=(VolatileSpeechClaimOwner.STATUS,))
    automatic.start()
    status.start()
    barrier.wait()
    automatic.join()
    status.join()

    granted = [claim for claim in claims if claim is not None]
    assert len(granted) == 1
    snapshot = ledger.get("session-1", "request-1")
    assert snapshot.result is not None
    assert snapshot.result.speech_state is VolatileSpeechState.CLAIMED
    assert snapshot.result.claim_owner is granted[0].owner


def test_unsent_claim_can_be_released_but_stale_claim_cannot_mutate_new_owner() -> None:
    ledger = VolatileWorkLedger()
    _complete(ledger)
    first = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.AUTOMATIC)
    assert first is not None

    released = ledger.release_speech_claim(first)
    second = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.STATUS)

    assert released.result is not None
    assert released.result.speech_state is VolatileSpeechState.READY
    assert second is not None
    assert second.revision > first.revision
    with pytest.raises(VolatileStaleClaimError):
        ledger.mark_speech_started(first)


def test_playback_transitions_to_heard_and_is_idempotent() -> None:
    ledger = VolatileWorkLedger()
    _complete(ledger)
    claim = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.AUTOMATIC)
    assert claim is not None

    speaking = ledger.mark_speech_started(claim)
    delivered = ledger.mark_speech_delivered(claim, turn_id="turn-1")
    heard = ledger.record_playback(
        claim,
        state=VolatilePlaybackState.HEARD,
        heard_through_ms=2400,
        audio_end_ms=2400,
    )
    replay = ledger.record_playback(
        claim,
        state=VolatilePlaybackState.HEARD,
        heard_through_ms=2400,
        audio_end_ms=2400,
    )

    assert speaking.result is not None
    assert speaking.result.speech_state is VolatileSpeechState.SPEAKING
    assert delivered.result is not None
    assert delivered.result.speech_state is VolatileSpeechState.AWAITING_PLAYBACK_RECEIPT
    assert heard.result is not None
    assert heard.result.speech_state is VolatileSpeechState.HEARD
    assert replay == heard


def test_interrupted_result_requires_an_explicit_recap_claim() -> None:
    ledger = VolatileWorkLedger()
    _complete(ledger)
    original = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.AUTOMATIC)
    assert original is not None
    ledger.mark_speech_delivered(original, turn_id="turn-original")
    interrupted = ledger.record_playback(
        original,
        state=VolatilePlaybackState.INTERRUPTED,
        heard_through_ms=600,
        audio_end_ms=2000,
    )

    assert interrupted.result is not None
    assert interrupted.result.speech_state is VolatileSpeechState.INTERRUPTED
    assert (
        ledger.claim_speech(
            "session-1",
            "request-1",
            owner=VolatileSpeechClaimOwner.AUTOMATIC,
        )
        is None
    )
    assert (
        ledger.claim_speech(
            "session-1",
            "request-1",
            owner=VolatileSpeechClaimOwner.STATUS,
        )
        is None
    )

    recap = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.RECAP)
    assert recap is not None
    assert recap.source_state is VolatileSpeechState.INTERRUPTED
    ledger.mark_speech_delivered(recap, turn_id="turn-recap")
    final = ledger.record_playback(
        recap,
        state=VolatilePlaybackState.HEARD,
        heard_through_ms=900,
        audio_end_ms=900,
    )
    assert final.result is not None
    assert final.result.speech_state is VolatileSpeechState.HEARD


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (VolatileDeliveryOutcome.SKIPPED, VolatileSpeechState.SKIPPED),
        (VolatileDeliveryOutcome.CANCELLED, VolatileSpeechState.CANCELLED),
        (VolatileDeliveryOutcome.FAILED, VolatileSpeechState.FAILED),
    ],
)
def test_no_playback_delivery_outcomes_are_terminal(outcome, expected) -> None:
    ledger = VolatileWorkLedger()
    _complete(ledger)
    claim = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.AUTOMATIC)
    assert claim is not None

    snapshot = ledger.record_delivery_outcome(claim, outcome=outcome)

    assert snapshot.result is not None
    assert snapshot.result.delivery_outcome is outcome
    assert snapshot.result.speech_state is expected
    assert snapshot.result.speech_state.terminal is True


def test_request_bound_evicts_only_oldest_quiescent_record() -> None:
    ledger = VolatileWorkLedger(max_requests_per_session=2)
    _complete(ledger, request_id="old")
    _finish_delivery(ledger, session_id="session-1", request_id="old")
    ledger.admit("session-1", "active", summary="keep active")

    ledger.admit("session-1", "new", summary="new request")

    with pytest.raises(VolatileRequestNotFoundError):
        ledger.get("session-1", "old")
    assert ledger.get("session-1", "active").state is VolatileWorkState.LOCALLY_QUEUED
    assert ledger.get("session-1", "new").state is VolatileWorkState.LOCALLY_QUEUED


def test_request_bound_never_evicts_active_or_undelivered_results() -> None:
    active_ledger = VolatileWorkLedger(max_requests_per_session=1)
    active_ledger.admit("session-1", "active", summary="active")
    with pytest.raises(VolatileCapacityError, match="active or undelivered"):
        active_ledger.admit("session-1", "new", summary="new")

    result_ledger = VolatileWorkLedger(max_requests_per_session=1)
    _complete(result_ledger)
    with pytest.raises(VolatileCapacityError, match="active or undelivered"):
        result_ledger.admit("session-1", "new", summary="new")


def test_session_bound_evicts_quiescent_session_but_preserves_active_session() -> None:
    ledger = VolatileWorkLedger(max_sessions=2, max_requests_per_session=2)
    ledger.admit("quiescent", "done", summary="done")
    ledger.fail("quiescent", "done", failure_code="local_failure")
    ledger.admit("active", "work", summary="active")

    ledger.admit("new", "work", summary="new")

    with pytest.raises(VolatileRequestNotFoundError):
        ledger.session("quiescent")
    assert ledger.session("active").requests[0].state is VolatileWorkState.LOCALLY_QUEUED
    assert ledger.session("new").requests[0].state is VolatileWorkState.LOCALLY_QUEUED


def test_session_bound_fails_closed_when_every_session_is_live() -> None:
    ledger = VolatileWorkLedger(max_sessions=1)
    ledger.admit("active", "work", summary="active")

    with pytest.raises(VolatileCapacityError, match="active or undelivered"):
        ledger.admit("other", "work", summary="other")


def test_session_discard_requires_quiescence_unless_caller_explicitly_fences_it() -> None:
    ledger = VolatileWorkLedger()
    ledger.admit("session-1", "request-1", summary="active")

    with pytest.raises(VolatileTransitionError, match="active or undelivered"):
        ledger.discard_session("session-1")
    assert ledger.discard_session("session-1", force=True) is True
    assert ledger.discard_session("session-1", force=True) is False


def test_invalid_playback_boundaries_are_rejected() -> None:
    ledger = VolatileWorkLedger()
    _complete(ledger)
    claim = ledger.claim_speech("session-1", "request-1", owner=VolatileSpeechClaimOwner.AUTOMATIC)
    assert claim is not None
    ledger.mark_speech_delivered(claim, turn_id="turn-1")

    with pytest.raises(ValueError, match="complete audio boundary"):
        ledger.record_playback(
            claim,
            state=VolatilePlaybackState.HEARD,
            heard_through_ms=99,
            audio_end_ms=100,
        )
    with pytest.raises(ValueError, match="stop before"):
        ledger.record_playback(
            claim,
            state=VolatilePlaybackState.INTERRUPTED,
            heard_through_ms=100,
            audio_end_ms=100,
        )
