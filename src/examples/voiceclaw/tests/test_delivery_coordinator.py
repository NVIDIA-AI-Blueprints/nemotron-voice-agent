# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
from pathlib import Path

import pytest

from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.application.delivery import DeliveryCoordinator
from voiceclaw.application.presentation import PresentationScheduler
from voiceclaw.domain.models import (
    DisplayPayload,
    FrontendActivity,
    InputActivity,
    ModelActivity,
    ResultAvailable,
    SessionBinding,
    SpeechRoute,
    SpeechState,
    WorkResult,
)
from voiceclaw.ports.presentation import DisplayOffer, SpeechRequest


class RecordingClient:
    def __init__(self) -> None:
        self.offers: list[DisplayOffer] = []

    async def offer_display(self, offer: DisplayOffer) -> None:
        self.offers.append(offer)


class RecordingSpeech:
    def __init__(self) -> None:
        self.requests: list[SpeechRequest] = []
        self.interruptions: list[tuple[str, str]] = []

    async def speak(self, request: SpeechRequest) -> None:
        self.requests.append(request)

    async def interrupt(self, session_id: str, presentation_id: str) -> None:
        self.interruptions.append((session_id, presentation_id))


class FailingClient:
    async def offer_display(self, offer: DisplayOffer) -> None:
        raise RuntimeError("display unavailable")


def _result(suffix: str, sequence: int) -> WorkResult:
    return WorkResult(
        presentation_id=f"presentation-{suffix}",
        session_id="session-a",
        attachment_id="attachment-a",
        work_id=f"work-{suffix}",
        result_id=f"result-{suffix}",
        sequence=sequence,
        display=DisplayPayload(kind="work.result", title=f"Result {suffix}", body=f"Full {suffix}"),
        speech_text=f"Result {suffix} is ready.",
    )


def _coordinator(path: Path) -> tuple[DeliveryCoordinator, SqliteStateStore, RecordingClient, RecordingSpeech]:
    store = SqliteStateStore(path)
    store.save_session(
        SessionBinding(
            session_id="session-a",
            conversation_id="conversation-a",
            backend_profile="default",
            attachment_id="attachment-a",
        )
    )
    client = RecordingClient()
    speech = RecordingSpeech()
    coordinator = DeliveryCoordinator(
        scheduler=PresentationScheduler(),
        state_store=store,
        client=client,
        speech=speech,
    )
    return coordinator, store, client, speech


def test_backend_result_rejects_an_unsupported_speech_route() -> None:
    """An adapter cannot revive a direct-to-TTS path with an unchecked string."""
    with pytest.raises(ValueError, match="direct_tts"):
        WorkResult(
            presentation_id="presentation-route",
            session_id="session-a",
            attachment_id="attachment-a",
            work_id="work-route",
            result_id="result-route",
            sequence=1,
            display=DisplayPayload(kind="work.result", title="Result"),
            speech_text="Presentation material.",
            speech_route="direct_tts",
        )

    with pytest.raises(ValueError, match="direct_tts"):
        ResultAvailable(
            result_id="result-route",
            presentation_id="presentation-route",
            display=DisplayPayload(kind="work.result", title="Result"),
            speech_text="Presentation material.",
            speech_route="direct_tts",
        )

    with pytest.raises(ValueError, match="direct_tts"):
        SpeechRequest(
            session_id="session-a",
            presentation_id="presentation-route",
            attachment_id="attachment-a",
            work_id="work-route",
            result_id="result-route",
            text="Presentation material.",
            route="direct_tts",
            priority=30,
        )


def test_supported_speech_route_is_normalized_to_the_enum() -> None:
    """String-backed adapter data normalizes to the sole model-mediated route."""
    result = WorkResult(
        presentation_id="presentation-route",
        session_id="session-a",
        attachment_id="attachment-a",
        work_id="work-route",
        result_id="result-route",
        sequence=1,
        display=DisplayPayload(kind="work.result", title="Result"),
        speech_text="Presentation material.",
        speech_route="frontend_model",
    )

    assert result.speech_route is SpeechRoute.FRONTEND_MODEL


def test_busy_frontend_gets_both_ui_results_before_sequential_speech(tmp_path) -> None:
    async def scenario() -> None:
        coordinator, store, client, speech = _coordinator(tmp_path / "state.db")
        try:
            await coordinator.update_activity(
                "session-a",
                FrontendActivity(connected=True, model=ModelActivity.GENERATING),
            )
            await coordinator.ingest_result(_result("a", 41))
            await coordinator.ingest_result(_result("b", 42))

            assert [offer.presentation_id for offer in client.offers] == ["presentation-a", "presentation-b"]
            assert speech.requests == []

            await coordinator.update_activity("session-a", FrontendActivity(connected=True))
            assert [request.presentation_id for request in speech.requests] == ["presentation-a"]

            coordinator.speech_started("session-a", "presentation-a")
            await coordinator.speech_completed("session-a", "presentation-a")
            assert [request.presentation_id for request in speech.requests] == [
                "presentation-a",
                "presentation-b",
            ]
            assert all(request.text.endswith("is ready.") for request in speech.requests)
            assert all("Full" not in request.text for request in speech.requests)
        finally:
            store.close()

    asyncio.run(scenario())


def test_interrupt_stops_speech_without_any_backend_port(tmp_path) -> None:
    async def scenario() -> None:
        coordinator, store, _, speech = _coordinator(tmp_path / "state.db")
        try:
            await coordinator.update_activity("session-a", FrontendActivity(connected=True))
            await coordinator.ingest_result(_result("a", 1))
            coordinator.speech_started("session-a", "presentation-a")

            interrupted = await coordinator.interrupt_speech("session-a", "presentation-a")

            assert interrupted.speech_state.value == "interrupted"
            assert speech.interruptions == [("session-a", "presentation-a")]
        finally:
            store.close()

    asyncio.run(scenario())


def test_display_failure_does_not_block_independent_speech(tmp_path) -> None:
    async def scenario() -> None:
        coordinator, store, _, speech = _coordinator(tmp_path / "state.db")
        coordinator = DeliveryCoordinator(
            scheduler=PresentationScheduler(),
            state_store=store,
            client=FailingClient(),
            speech=speech,
        )
        try:
            await coordinator.update_activity("session-a", FrontendActivity(connected=True))
            try:
                await coordinator.ingest_result(_result("a", 1))
            except ExceptionGroup as error:
                assert [str(item) for item in error.exceptions] == ["display unavailable"]
            else:
                raise AssertionError("display failure was not surfaced")

            assert [request.presentation_id for request in speech.requests] == ["presentation-a"]
        finally:
            store.close()

    asyncio.run(scenario())


def test_user_speech_barge_in_interrupts_active_presentation(tmp_path) -> None:
    async def scenario() -> None:
        coordinator, store, _, speech = _coordinator(tmp_path / "state.db")
        try:
            await coordinator.update_activity("session-a", FrontendActivity(connected=True))
            await coordinator.ingest_result(_result("a", 1))
            coordinator.speech_started("session-a", "presentation-a")

            await coordinator.update_activity(
                "session-a",
                FrontendActivity(connected=True, input=InputActivity.LISTENING),
            )

            assert speech.interruptions == [("session-a", "presentation-a")]
            record = store.get_presentation("presentation-a")
            assert record is not None
            assert record.speech_state.value == "interrupted"
        finally:
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("speech_started", [False, True])
def test_same_process_reconnect_reoffers_display_and_reclaims_speech_lease(tmp_path, speech_started) -> None:
    async def scenario() -> None:
        coordinator, store, client, speech = _coordinator(tmp_path / "state.db")
        try:
            await coordinator.update_activity("session-a", FrontendActivity(connected=True))
            await coordinator.ingest_result(_result("a", 1))
            if speech_started:
                coordinator.speech_started("session-a", "presentation-a")
                assert store.get_presentation("presentation-a").speech_state is SpeechState.SPEAKING
            else:
                assert store.get_presentation("presentation-a").speech_state is SpeechState.CLAIMED

            await coordinator.restore_pending("session-a")

            assert [offer.presentation_id for offer in client.offers] == [
                "presentation-a",
                "presentation-a",
            ]
            assert [request.presentation_id for request in speech.requests] == [
                "presentation-a",
                "presentation-a",
            ]
            restored = store.get_presentation("presentation-a")
            assert restored is not None
            assert restored.speech_state is SpeechState.CLAIMED
        finally:
            store.close()

    asyncio.run(scenario())
