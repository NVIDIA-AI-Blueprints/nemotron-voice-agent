# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thinker booking-flow tool implementation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
from loguru import logger

from examples.frontend_backend_agent.airline.backend import BookingBackend
from examples.frontend_backend_agent.airline.branding import flight_label, user_facing_carrier, user_facing_record
from examples.frontend_backend_agent.airline.slot_parsing import (
    bool_slot,
    extract_selected_flight,
    flight_identity,
    has_booking_preferences,
    slot,
    summary_passenger_name,
)
from examples.frontend_backend_agent.airline.state import BookingDraft, ThinkerSessionState
from examples.frontend_backend_agent.src.protocol import response_hint, tool_result


class BookingTool:
    """Stateful booking-flow implementation for one Thinker session."""

    def __init__(self, *, state: ThinkerSessionState, backend: BookingBackend) -> None:
        """Create a booking tool bound to session state and the booking backend."""
        self._state = state
        self._backend = backend

    async def continue_booking(self, slots: dict[str, Any]) -> dict[str, Any]:
        """Advance the current booking flow from planned slots."""
        attempt = self._state.booking_attempt
        if attempt is not None:
            selected = slot(slots, "flight_selected")
            if selected and extract_selected_flight(selected, self._state.search_results) != attempt.flight:
                return booking_outcome_unknown()
            for key in ("passenger_name", "seat_pref", "meal_pref"):
                value = slot(slots, key)
                if value and value != getattr(attempt, key):
                    return booking_outcome_unknown()
            if bool_slot(slots, "confirmed") is True:
                return await self._finalize_booking()
            return booking_outcome_unknown()
        if not self._state.search_results:
            return response_hint(
                reason="missing_search_context",
                action="req_flight_search",
                params_needed=["origin_city", "dest_city", "date"],
                response_text="I need to search flights before booking. Where are you flying from, where to, and when?",
                context="booking",
            )

        selected_value = slot(slots, "flight_selected")
        selected = extract_selected_flight(selected_value, self._state.search_results)
        if selected_value is not None and selected is None:
            if self._state.booking_draft is not None:
                self._state.booking_draft = None
                self._state.waiting_for_preferences = False
                self._state.waiting_for_confirmation = False
            return response_hint(
                reason="params_missing",
                action="req_params",
                params_needed=["flight_selected"],
                response_text=(
                    "I could not match that flight selection. "
                    "Which flight from the search results would you like to book?"
                ),
                context="booking",
            )

        use_defaults = bool_slot(slots, "use_defaults") is True
        has_preferences = has_booking_preferences(slots)
        if selected is not None and self._state.booking_draft is not None:
            draft = self._state.booking_draft
            if flight_identity(selected) != flight_identity(draft.flight):
                draft.flight = selected
                self._merge_preferences(slots)
                if self._state.waiting_for_preferences and not use_defaults and not has_preferences:
                    self._state.waiting_for_confirmation = False
                    return self._preferences_hint()
                self._state.waiting_for_preferences = False
                self._state.waiting_for_confirmation = True
                return self._confirmation_hint()

        if self._state.waiting_for_confirmation:
            self._merge_preferences(slots)
            confirmed = bool_slot(slots, "confirmed")
            if confirmed is True:
                return await self._finalize_booking()
            if confirmed is False:
                self._state.reset_booking()
                return response_hint(
                    reason="booking_cancelled",
                    action="cancelled",
                    response_text="No problem, I have not booked it.",
                    context="booking",
                )
            return response_hint(
                reason="confirm_required",
                action="confirm_booking",
                response_text="Should I confirm this booking?",
                context="booking",
            )

        if self._state.booking_draft is None:
            if selected is None:
                return response_hint(
                    reason="params_missing",
                    action="req_params",
                    params_needed=["flight_selected"],
                    response_text="Which flight from the search results would you like to book?",
                    context="booking",
                )
            passenger_name = slot(slots, "passenger_name")
            self._state.booking_draft = BookingDraft(flight=selected, passenger_name=passenger_name)

        self._merge_preferences(slots)
        if use_defaults or has_preferences:
            self._state.waiting_for_preferences = False
            self._state.waiting_for_confirmation = True
            return self._confirmation_hint()

        self._state.waiting_for_preferences = True
        return self._preferences_hint()

    def _preferences_hint(self) -> dict[str, Any]:
        return response_hint(
            reason="params_optional",
            action="req_params",
            params_needed=["seat_pref", "meal_pref"],
            defaults_available=True,
            response_text="Do you have a seat or meal preference? Otherwise I will use the defaults.",
            context="booking",
        )

    def _confirmation_hint(self) -> dict[str, Any]:
        draft = self._state.booking_draft
        if draft is None:
            raise RuntimeError("booking draft missing while building confirmation")
        flight = draft.flight
        seat = draft.seat_pref or "default seat"
        meal = draft.meal_pref or "default meal"
        price = flight.get("price_usd")
        price_text = f", for ${price}" if price is not None else ""
        passenger_name = summary_passenger_name(draft.passenger_name)
        summary = {
            "route": f"{flight['origin_city']} to {flight['dest_city']}",
            "date": flight["date"],
            "flight_id": flight["flight_id"],
            "carrier": user_facing_carrier(flight),
            "seat_pref": draft.seat_pref,
            "meal_pref": draft.meal_pref,
            "price_usd": price,
        }
        if passenger_name:
            summary["passenger_name"] = passenger_name
        passenger_text = f" for {passenger_name}" if passenger_name else ""
        return response_hint(
            reason="confirm_required",
            action="confirm_booking",
            summary=summary,
            response_text=(
                f"Here is the booking summary{passenger_text}: {flight_label(flight)} from "
                f"{flight['origin_city']} to {flight['dest_city']} on {flight['date']}, "
                f"{seat}, {meal}{price_text}. Shall I confirm?"
            ),
            context="booking",
        )

    async def _finalize_booking(self) -> dict[str, Any]:
        """Submit the confirmed booking once, reusing the attempt's idempotency key on retries."""
        draft = self._state.booking_attempt or self._state.booking_draft
        if draft is None:
            return response_hint(
                reason="params_missing",
                action="req_params",
                params_needed=["flight_selected"],
                response_text="Which flight should I book?",
                context="booking",
            )
        if self._state.booking_attempt is None:
            self._state.booking_attempt = replace(draft, flight=dict(draft.flight))
        draft = self._state.booking_attempt
        try:
            record = await self._backend.create_booking(
                passenger_name=draft.passenger_name,
                flight=draft.flight,
                seat_pref=draft.seat_pref,
                meal_pref=draft.meal_pref,
                idempotency_key=draft.idempotency_key,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.warning(f"booking backend unreachable while creating PNR: {exc}")
            self._state.booking_attempt = None
            return response_hint(
                reason="tool_error",
                action="retry",
                error=str(exc),
                response_text="I could not reach the booking system, so nothing was booked. Please try again shortly.",
                context="booking",
            )
        except Exception as exc:
            logger.warning(f"booking backend failed while creating PNR: {exc}")
            return booking_unverified(str(exc))
        if record is None:
            self._state.booking_attempt = None
            return response_hint(
                reason="tool_error",
                action="req_params",
                params_needed=["flight_selected"],
                error="Booking backend could not create the PNR.",
                response_text="I could not complete that booking. Please choose another flight.",
                context="booking",
            )
        if not record.get("pnr"):
            logger.warning("booking backend returned a record without a PNR; keeping the attempt for retry")
            return booking_unverified("Booking backend returned no PNR.")
        self._state.booking_attempt = None
        self._state.reset_booking()
        return tool_result(
            tool="booking",
            status="success",
            data={"booking": user_facing_record(record), "pnr": record.get("pnr")},
            response_text=f"Your booking is confirmed. Your PNR is {record.get('pnr')}.",
            context="booking",
        )

    def _merge_preferences(self, slots: dict[str, Any]) -> None:
        """Copy seat, meal, and passenger slots into the editable draft."""
        draft = self._state.booking_draft
        if draft is None:
            return
        seat = slot(slots, "seat_pref")
        meal = slot(slots, "meal_pref")
        passenger = slot(slots, "passenger_name")
        if seat:
            draft.seat_pref = seat
        if meal:
            draft.meal_pref = meal
        if passenger:
            draft.passenger_name = passenger


def booking_outcome_unknown() -> dict[str, Any]:
    """Ask for reconfirmation when a submitted booking has no receipt."""
    return response_hint(
        reason="booking_outcome_unknown",
        action="confirm_booking",
        response_text=(
            "That booking might have completed. Stopping here does not cancel a booked ticket. "
            "Confirm the same booking again to check the result."
        ),
        context="booking",
    )


def booking_unverified(error: str) -> dict[str, Any]:
    """Ask for reconfirmation when a sent booking request has no usable receipt."""
    return response_hint(
        reason="tool_error",
        action="retry",
        error=error,
        response_text="I could not verify whether that booking completed. Confirm again to check the same booking.",
        context="booking",
    )


def booking_dropped_unverified() -> dict[str, Any]:
    """Tell the caller that stopping dropped a booking whose outcome is unknown."""
    return response_hint(
        reason="cancelled",
        action="cancelled",
        response_text=(
            "Okay, I stopped that. I could not confirm whether the last booking went through, "
            "and stopping does not cancel a ticket that was already booked."
        ),
        context="booking",
    )
