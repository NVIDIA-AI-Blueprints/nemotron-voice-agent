# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio

import pytest

from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.application.delivery import DeliveryCoordinator
from voiceclaw.application.events import (
    BackendEventCoordinator,
    EventDisposition,
    MisroutedEventError,
    OrderedEventInbox,
    UnknownSessionError,
)
from voiceclaw.application.presentation import PresentationScheduler
from voiceclaw.domain.models import (
    AgentQueryChanged,
    AgentQueryKind,
    AgentQueryState,
    BackendEvent,
    BackendEventKind,
    DisplayPayload,
    FrontendActivity,
    ModelActivity,
    PresentationRecord,
    ResultAvailable,
    SessionBinding,
    WorkResult,
    WorkState,
    WorkStateChanged,
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

    async def speak(self, request: SpeechRequest) -> None:
        self.requests.append(request)

    async def interrupt(self, session_id: str, presentation_id: str) -> None:
        return None


def _event(sequence: int, *, session_id: str = "session-a", attachment_id: str = "attachment-a") -> BackendEvent:
    return BackendEvent(
        event_id=f"event-{sequence}",
        session_id=session_id,
        attachment_id=attachment_id,
        sequence=sequence,
        kind=BackendEventKind.WORK_STATE_CHANGED,
        work_id="work-a",
        payload=WorkStateChanged(state=WorkState.RUNNING),
    )


def test_duplicates_are_ignored_and_gaps_pause_projection(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(
            SessionBinding(
                session_id="session-a",
                conversation_id="conversation-a",
                backend_profile="default",
                attachment_id="attachment-a",
                last_applied_sequence=10,
            )
        )
        inbox = OrderedEventInbox(store)

        assert inbox.classify(_event(10)).disposition is EventDisposition.DUPLICATE
        gap = inbox.classify(_event(12))
        assert gap.disposition is EventDisposition.GAP
        assert gap.expected_sequence == 11
        assert store.get_session("session-a").last_applied_sequence == 10

        assert inbox.classify(_event(11)).disposition is EventDisposition.APPLY
        assert inbox.commit(_event(11)) == 11
        assert inbox.classify(_event(12)).disposition is EventDisposition.APPLY


def test_wrong_attachment_and_unknown_session_are_never_routed(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(
            SessionBinding(
                session_id="session-a",
                conversation_id="conversation-a",
                backend_profile="default",
                attachment_id="attachment-a",
            )
        )
        inbox = OrderedEventInbox(store)

        with pytest.raises(MisroutedEventError):
            inbox.classify(_event(1, attachment_id="attachment-b"))
        with pytest.raises(UnknownSessionError):
            inbox.classify(_event(1, session_id="session-missing"))


def test_canonical_events_drive_work_projection_and_result_delivery_once(tmp_path) -> None:
    async def scenario() -> None:
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-a",
                )
            )
            scheduler = PresentationScheduler()
            scheduler.set_activity(
                "session-a",
                FrontendActivity(connected=True, model=ModelActivity.GENERATING),
            )
            client = RecordingClient()
            speech = RecordingSpeech()
            delivery = DeliveryCoordinator(
                scheduler=scheduler,
                state_store=store,
                client=client,
                speech=speech,
            )
            coordinator = BackendEventCoordinator(
                inbox=OrderedEventInbox(store),
                state_store=store,
                delivery=delivery,
            )

            await coordinator.ingest(_event(1))
            result_event = BackendEvent(
                event_id="event-2",
                session_id="session-a",
                attachment_id="attachment-a",
                sequence=2,
                kind=BackendEventKind.RESULT_AVAILABLE,
                work_id="work-a",
                payload=ResultAvailable(
                    result_id="result-a",
                    presentation_id="presentation-a",
                    display=DisplayPayload(kind="work.result", title="Done", body="Full result"),
                    speech_text="The work is done.",
                ),
            )
            applied = await coordinator.ingest(result_event)
            duplicate = await coordinator.ingest(result_event)

            projection = store.get_work_projection("attachment-a", "work-a")
            assert projection is not None
            assert projection.state is WorkState.RUNNING
            assert applied.committed_sequence == 2
            assert duplicate.decision.disposition is EventDisposition.DUPLICATE
            assert [offer.presentation_id for offer in client.offers] == ["presentation-a"]
            assert speech.requests == []
            assert store.get_session("session-a").last_applied_sequence == 2

    asyncio.run(scenario())


def test_agent_query_events_create_read_only_session_projection(tmp_path) -> None:
    async def scenario() -> None:
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-a",
                )
            )
            coordinator = BackendEventCoordinator(
                inbox=OrderedEventInbox(store),
                state_store=store,
                delivery=DeliveryCoordinator(
                    scheduler=PresentationScheduler(),
                    state_store=store,
                    client=RecordingClient(),
                    speech=RecordingSpeech(),
                ),
            )

            requested = BackendEvent(
                event_id="event-1",
                session_id="session-a",
                attachment_id="attachment-a",
                sequence=1,
                kind=BackendEventKind.AGENT_QUERY_CHANGED,
                work_id="work-a",
                payload=AgentQueryChanged(
                    query_id="query-a",
                    kind=AgentQueryKind.PERMISSION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    prompt="Allow the external action?",
                ),
            )
            application = await coordinator.ingest(requested)

            assert application.agent_query_projection_changed is True
            assert [query.query_id for query in store.pending_agent_queries("session-a")] == ["query-a"]

            forwarded = BackendEvent(
                event_id="event-2",
                session_id="session-a",
                attachment_id="attachment-a",
                sequence=2,
                kind=BackendEventKind.AGENT_QUERY_CHANGED,
                work_id="work-a",
                payload=AgentQueryChanged(
                    query_id="query-a",
                    kind=AgentQueryKind.PERMISSION,
                    state=AgentQueryState.RESPONSE_FORWARDED,
                    blocking=True,
                    prompt="Allow the external action?",
                ),
            )
            forwarded_application = await coordinator.ingest(forwarded)

            projection = store.get_session_agent_query_projection("session-a", "query-a")
            assert forwarded_application.agent_query_projection_changed is True
            assert projection is not None
            assert projection.state is AgentQueryState.RESPONSE_FORWARDED
            assert projection.state.terminal is False
            assert store.pending_agent_queries("session-a") == ()
            assert store.get_session("session-a").last_applied_sequence == 2

    asyncio.run(scenario())


def test_saved_result_replay_after_reattach_deduplicates_before_cursor_commit(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "state.db"
        with SqliteStateStore(path) as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-a",
                    backend_session_id="backend-session-a",
                    last_applied_sequence=8,
                )
            )
            # Crash window: result evidence is durable, but the event cursor
            # was not advanced and no external presentation was attempted.
            DeliveryCoordinator(
                scheduler=PresentationScheduler(),
                state_store=store,
                client=RecordingClient(),
                speech=RecordingSpeech(),
            ).stage_result(
                WorkResult(
                    presentation_id="presentation-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    work_id="work-a",
                    result_id="result-a",
                    sequence=9,
                    display=DisplayPayload(kind="work.result", title="Done", body="Full result"),
                    speech_text="The work is done.",
                )
            )
            assert store.get_session("session-a").last_applied_sequence == 8

        with SqliteStateStore(path) as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-b",
                    backend_session_id="backend-session-a",
                    last_applied_sequence=8,
                )
            )
            client = RecordingClient()
            delivery = DeliveryCoordinator(
                scheduler=PresentationScheduler(),
                state_store=store,
                client=client,
                speech=RecordingSpeech(),
            )
            coordinator = BackendEventCoordinator(
                inbox=OrderedEventInbox(store),
                state_store=store,
                delivery=delivery,
            )
            replay = BackendEvent(
                event_id="event-replayed",
                session_id="session-a",
                attachment_id="attachment-b",
                sequence=9,
                kind=BackendEventKind.RESULT_AVAILABLE,
                work_id="work-a",
                payload=ResultAvailable(
                    result_id="result-a",
                    presentation_id="presentation-a",
                    display=DisplayPayload(kind="work.result", title="Done", body="Full result"),
                    speech_text="The work is done.",
                ),
            )

            application = await coordinator.ingest(replay)

            assert application.committed_sequence == 9
            assert application.presentation_id == "presentation-a"
            assert [offer.presentation_id for offer in client.offers] == ["presentation-a"]
            persisted = store.get_result_presentation("session-a", "work-a", "result-a", 9)
            assert persisted is not None
            assert persisted.attachment_id == "attachment-a"
            assert persisted.sequence == 9
            assert store.get_session("session-a").last_applied_sequence == 9
            assert store._connection.execute("SELECT COUNT(*) FROM presentations").fetchone()[0] == 1  # noqa: SLF001

    asyncio.run(scenario())


def test_conflicting_presentation_does_not_advance_ordered_event_cursor(tmp_path) -> None:
    async def scenario() -> None:
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-a",
                )
            )
            await BackendEventCoordinator(
                inbox=OrderedEventInbox(store),
                state_store=store,
                delivery=DeliveryCoordinator(
                    scheduler=PresentationScheduler(),
                    state_store=store,
                    client=RecordingClient(),
                    speech=RecordingSpeech(),
                ),
            ).ingest(_event(1))
            assert store.save_presentation(
                PresentationRecord.from_result(
                    WorkResult(
                        presentation_id="presentation-a",
                        session_id="session-a",
                        attachment_id="attachment-a",
                        work_id="work-conflicting",
                        result_id="result-conflicting",
                        sequence=99,
                        display=DisplayPayload(kind="work.result", title="Conflicting result"),
                    )
                )
            )
            coordinator = BackendEventCoordinator(
                inbox=OrderedEventInbox(store),
                state_store=store,
                delivery=DeliveryCoordinator(
                    scheduler=PresentationScheduler(),
                    state_store=store,
                    client=RecordingClient(),
                    speech=RecordingSpeech(),
                ),
            )
            event = BackendEvent(
                event_id="event-2",
                session_id="session-a",
                attachment_id="attachment-a",
                sequence=2,
                kind=BackendEventKind.RESULT_AVAILABLE,
                work_id="work-a",
                payload=ResultAvailable(
                    result_id="result-a",
                    presentation_id="presentation-a",
                    display=DisplayPayload(kind="work.result", title="Done"),
                ),
            )

            with pytest.raises(ValueError, match="presentation identity conflicts"):
                await coordinator.ingest(event)

            binding = store.get_session("session-a")
            assert binding is not None
            assert binding.last_applied_sequence == 1

    asyncio.run(scenario())
