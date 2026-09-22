# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Independent client-display and frontend-speech ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from voiceclaw.domain.models import DisplayPayload, PresentationPriority, SpeechRoute


@dataclass(frozen=True, slots=True)
class DisplayOffer:
    """Rich UI projection routed directly to one client session."""

    session_id: str
    presentation_id: str
    attachment_id: str
    work_id: str
    result_id: str
    sequence: int
    payload: DisplayPayload


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    """Approved spoken-presentation material without the rich display payload."""

    session_id: str
    presentation_id: str
    attachment_id: str
    work_id: str
    result_id: str
    text: str
    route: SpeechRoute
    priority: PresentationPriority

    def __post_init__(self) -> None:
        """Fail closed unless the request uses the model-mediated speech route."""
        object.__setattr__(self, "route", SpeechRoute(self.route))


class ClientPresentationPort(Protocol):
    """Publish UI material to the socket bound to exactly one session."""

    async def offer_display(self, offer: DisplayOffer) -> None:
        """Publish display material without invoking the frontend model."""
        ...


class FrontendSpeechPort(Protocol):
    """Render only approved speech after the scheduler grants the floor."""

    async def speak(self, request: SpeechRequest) -> None:
        """Render queued presentation material using the frontend model."""
        ...

    async def interrupt(self, session_id: str, presentation_id: str) -> None:
        """Stop local speech without cancelling backend Work."""
        ...
