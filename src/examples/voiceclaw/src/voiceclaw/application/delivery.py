# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Coordinate direct UI projection and safely queued speech delivery."""

from __future__ import annotations

import asyncio

from voiceclaw.application.presentation import PresentationScheduler
from voiceclaw.domain.models import FrontendActivity, InputActivity, PresentationRecord, SpeechRoute, WorkResult
from voiceclaw.ports.presentation import (
    ClientPresentationPort,
    DisplayOffer,
    FrontendSpeechPort,
    SpeechRequest,
)
from voiceclaw.ports.state import StateStore


class DeliveryCoordinator:
    """Keep display and speech independent while preserving shared identity."""

    def __init__(
        self,
        *,
        scheduler: PresentationScheduler,
        state_store: StateStore,
        client: ClientPresentationPort,
        speech: FrontendSpeechPort,
    ) -> None:
        """Bind domain scheduling to persistence and external delivery ports."""
        self._scheduler = scheduler
        self._state_store = state_store
        self._client = client
        self._speech = speech

    async def ingest_result(self, result: WorkResult) -> PresentationRecord:
        """Persist one result, publish its UI payload, and attempt safe speech."""
        record = self.stage_result(result)
        await self.deliver_pending(result.session_id)
        return record

    def stage_result(self, result: WorkResult) -> PresentationRecord:
        """Durably stage a result before advancing its backend event cursor."""
        candidate = PresentationRecord.from_result(result)
        stored_by_presentation = self._state_store.get_presentation(result.presentation_id)
        stored_by_result = self._state_store.get_result_presentation(*result.immutable_identity)
        if (
            stored_by_presentation is not None
            and stored_by_result is not None
            and stored_by_presentation.presentation_id != stored_by_result.presentation_id
        ):
            raise ValueError("presentation identity conflicts with persisted result evidence")
        stored = stored_by_presentation or stored_by_result
        if stored is not None:
            if not stored.has_same_immutable_evidence(candidate):
                raise ValueError("presentation identity conflicts with persisted result evidence")
            self._scheduler.restore(stored)
            return stored

        outcome = self._scheduler.offer(result)
        if not outcome.created:
            return outcome.record
        if not self._state_store.save_presentation(outcome.record):
            raise ValueError("presentation identity conflicts with persisted result evidence")
        return outcome.record

    async def deliver_pending(self, session_id: str) -> None:
        """Drive independently staged UI and speech deliveries for one session."""
        await self._drive_pending(session_id)

    async def restore_pending(self, session_id: str) -> None:
        """Restore pending channels and reclaim leases from a prior client."""
        for record in self._state_store.pending_presentations(session_id):
            recovered = self._scheduler.restore(record, reclaim_delivery_leases=True)
            if recovered != record:
                self._state_store.update_presentation(recovered)
        await self._drive_pending(session_id)

    async def update_activity(self, session_id: str, activity: FrontendActivity) -> None:
        """Update the speech gate and pump one item if the floor became free."""
        self._scheduler.set_activity(session_id, activity)
        active = self._scheduler.active_speech_id(session_id)
        if active is not None and activity.input is not InputActivity.IDLE:
            await self.interrupt_speech(session_id, active)
        await self._pump_speech(session_id)

    def display_completed(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Persist a client display receipt without affecting speech state."""
        record = self._scheduler.mark_displayed(session_id, presentation_id)
        self._state_store.update_presentation(record)
        return record

    def speech_started(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Persist playback start without asserting heard-through completion."""
        record = self._scheduler.mark_speech_started(session_id, presentation_id)
        self._state_store.update_presentation(record)
        return record

    async def speech_completed(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Persist heard-through completion and offer the next queued speech."""
        record = self._scheduler.mark_speech_completed(session_id, presentation_id)
        self._state_store.update_presentation(record)
        await self._pump_speech(session_id)
        return record

    async def interrupt_speech(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Stop local playback without sending a backend Work cancellation."""
        try:
            await self._speech.interrupt(session_id, presentation_id)
        finally:
            record = self._scheduler.mark_speech_interrupted(session_id, presentation_id)
            self._state_store.update_presentation(record)
        return record

    async def _drive_pending(self, session_id: str) -> None:
        displays: list[PresentationRecord] = []
        while (display := self._scheduler.claim_next_display(session_id)) is not None:
            displays.append(display)
        outcomes = await asyncio.gather(
            *(self._publish_display(display) for display in displays),
            self._pump_speech(session_id),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
        if failures:
            raise ExceptionGroup("one or more presentation channels failed", failures)

    async def _publish_display(self, record: PresentationRecord) -> None:
        try:
            await self._client.offer_display(
                DisplayOffer(
                    session_id=record.session_id,
                    presentation_id=record.presentation_id,
                    attachment_id=record.attachment_id,
                    work_id=record.work_id,
                    result_id=record.result_id,
                    sequence=record.sequence,
                    payload=record.display,
                )
            )
        except Exception:
            failed = self._scheduler.mark_display_failed(record.session_id, record.presentation_id)
            self._state_store.update_presentation(failed)
            raise

    async def _pump_speech(self, session_id: str) -> PresentationRecord | None:
        record = self._scheduler.claim_next_speech(session_id)
        if record is None:
            return None
        self._state_store.update_presentation(record)
        if record.speech_route is not SpeechRoute.FRONTEND_MODEL:
            failed = self._scheduler.mark_speech_failed(record.session_id, record.presentation_id)
            self._state_store.update_presentation(failed)
            raise ValueError("unsupported speech route")
        try:
            await self._speech.speak(
                SpeechRequest(
                    session_id=record.session_id,
                    presentation_id=record.presentation_id,
                    attachment_id=record.attachment_id,
                    work_id=record.work_id,
                    result_id=record.result_id,
                    text=record.speech_text or "",
                    route=record.speech_route,
                    priority=record.priority,
                )
            )
        except Exception:
            failed = self._scheduler.mark_speech_failed(record.session_id, record.presentation_id)
            self._state_store.update_presentation(failed)
            raise
        return record
