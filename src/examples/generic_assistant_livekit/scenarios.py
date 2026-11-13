# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use cases: each pairs a prompt from prompts.yaml with its own tools, so one stack can show different scenarios.

Everything here is fictional demo data held in memory for one session. Nothing calls a real system, except the weather
lookup in the generic scenario. Tools are plain functions: LiveKit builds the tool schema from the type hints and the
docstring `Args:` section. Standard library plus PyYAML only, so the UI server and the unit tests can use this module
without the agent environment.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml

from . import tools as shared

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")
DEFAULT_SCENARIO = "generic"

# ---------------------------------------------------------------- shared tools


def get_current_datetime(timezone: str = "") -> dict:
    """Get the current date and time.

    Args:
        timezone: Optional IANA timezone name such as "America/New_York". Leave empty for local time.
    """
    return shared.current_datetime(timezone)


async def get_weather(location: str, unit: Literal["fahrenheit", "celsius"] = "fahrenheit") -> dict:
    """Get the current weather for a city. Reports Fahrenheit by default; both units are supported.

    Args:
        location: City name, for example "Paris" or "Austin, Texas".
        unit: "fahrenheit" (the default) or "celsius". Use "celsius" whenever the user asks for Celsius or metric.
    """
    return await shared.current_weather(location, fahrenheit=(unit != "celsius"))


# ------------------------------------------------------------- insurance claim

POLICIES = {
    "AUT-1001": {
        "holder": "Jordan Lee",
        "vehicle": "2022 Honda Civic",
        "coverage": "comprehensive and collision",
        "deductible_usd": 500,
        "status": "active",
    },
    "AUT-2002": {
        "holder": "Sam Rivera",
        "vehicle": "2019 Ford F-150",
        "coverage": "liability only",
        "deductible_usd": 1000,
        "status": "active",
    },
}
SEEDED_CLAIMS = {
    "CLM-7001": {
        "policy": "AUT-1001",
        "status": "adjuster assigned",
        "next_step": "the adjuster will call within 2 business days",
    },
    "CLM-7002": {"policy": "AUT-2002", "status": "approved", "next_step": "payment is issued within 5 business days"},
}


def _insurance_tools() -> list[Callable]:
    claims = dict(SEEDED_CLAIMS)
    numbers = itertools.count(7100)

    def lookup_policy(policy_number: str) -> dict:
        """Look up an auto insurance policy: holder, vehicle, coverage, deductible and status.

        Args:
            policy_number: The policy number, such as AUT-1001.
        """
        policy = POLICIES.get(policy_number.upper())
        if not policy:
            raise ValueError(f"No policy found with number {policy_number}.")
        return {"policy_number": policy_number.upper(), **policy}

    def file_claim(policy_number: str, incident_date: str, description: str, estimated_damage_usd: float) -> dict:
        """File a new auto insurance claim for an incident. Confirm the details with the caller first.

        Args:
            policy_number: The policy number, such as AUT-1001.
            incident_date: Date of the incident as YYYY-MM-DD.
            description: One-sentence description of what happened.
            estimated_damage_usd: The caller's rough estimate of the damage in US dollars.
        """
        policy = POLICIES.get(policy_number.upper())
        if not policy:
            raise ValueError(f"No policy found with number {policy_number}.")
        claim_id = f"CLM-{next(numbers)}"
        covered = policy["coverage"] != "liability only"
        claims[claim_id] = {
            "policy": policy_number.upper(),
            "status": "received",
            "next_step": "an adjuster is assigned within 1 business day",
        }
        return {
            "claim_id": claim_id,
            "status": "received",
            "deductible_usd": policy["deductible_usd"],
            "note": None
            if covered
            else "This policy is liability only, so damage to the policyholder's own vehicle is not covered.",
            "next_step": claims[claim_id]["next_step"],
        }

    def check_claim_status(claim_id: str) -> dict:
        """Check the status of an existing claim.

        Args:
            claim_id: The claim number, such as CLM-7001.
        """
        claim = claims.get(claim_id.upper())
        if not claim:
            raise ValueError(f"No claim found with number {claim_id}.")
        return {"claim_id": claim_id.upper(), **claim}

    return [lookup_policy, file_claim, check_claim_status, get_current_datetime]


# --------------------------------------------------------------- clinic booking

SPECIALTIES = ("general practice", "dermatology", "pediatrics")
TIMES = ("09:00", "10:30", "14:00", "15:30")


def _clinic_tools() -> list[Callable]:
    def make_slots() -> dict[str, dict]:
        slots, day, added = {}, date.today(), 0
        while added < 5:
            day += timedelta(days=1)
            if day.weekday() >= 5:  # weekdays only
                continue
            added += 1
            for specialty in SPECIALTIES:
                for t in TIMES:
                    slot_id = f"{specialty[:3].upper()}-{day:%m%d}-{t.replace(':', '')}"
                    slots[slot_id] = {
                        "slot_id": slot_id,
                        "specialty": specialty,
                        "date": day.isoformat(),
                        "weekday": day.strftime("%A"),
                        "time": t,
                    }
        return slots

    slots = make_slots()
    bookings: dict[str, dict] = {}
    numbers = itertools.count(4001)

    def list_available_slots(
        specialty: Literal["general practice", "dermatology", "pediatrics"], date_filter: str = ""
    ) -> list[dict]:
        """List open appointment slots (at most 6, earliest first).

        Args:
            specialty: Which kind of doctor the patient needs.
            date_filter: Optional date as YYYY-MM-DD to show only that day.
        """
        booked = {booking["slot_id"] for booking in bookings.values()}
        found = [
            s
            for s in slots.values()
            if s["specialty"] == specialty
            and s["slot_id"] not in booked
            and (not date_filter or s["date"] == date_filter)
        ]
        if not found:
            raise ValueError("No open slots match. Try another day or specialty.")
        return sorted(found, key=lambda s: (s["date"], s["time"]))[:6]

    def book_appointment(slot_id: str, patient_name: str, reason: str) -> dict:
        """Book an appointment in a slot returned by list_available_slots. Confirm with the patient first.

        Args:
            slot_id: The slot_id of the chosen slot.
            patient_name: The patient's full name.
            reason: Short reason for the visit.
        """
        slot = slots.get(slot_id.upper())
        if not slot:
            raise ValueError(f"Unknown slot {slot_id}.")
        if slot["slot_id"] in {b["slot_id"] for b in bookings.values()}:
            raise ValueError("That slot was just taken. Please offer another.")
        confirmation = f"APT-{next(numbers)}"
        bookings[confirmation] = {**slot, "patient_name": patient_name, "reason": reason}
        return {"confirmation_id": confirmation, **bookings[confirmation]}

    def cancel_appointment(confirmation_id: str) -> dict:
        """Cancel a booked appointment.

        Args:
            confirmation_id: The confirmation number, such as APT-4001.
        """
        booking = bookings.pop(confirmation_id.upper(), None)
        if not booking:
            raise ValueError(f"No booking found with confirmation {confirmation_id}.")
        return {
            "cancelled": confirmation_id.upper(),
            "was": f"{booking['weekday']} {booking['date']} at {booking['time']}",
        }

    return [list_available_slots, book_appointment, cancel_appointment, get_current_datetime]


# ------------------------------------------------------------------- bank

DEMO_PIN = "1234"


def _bank_tools() -> list[Callable]:
    state = {"verified": False, "attempts": 0, "card_blocked": False}
    balances = {"checking": 2480.55, "savings": 10250.00}
    history = {
        "checking": [
            ("2026-11-09", "Grocery Market", -64.20),
            ("2026-11-07", "Payroll deposit", 2100.00),
            ("2026-11-05", "Coffee House", -4.75),
            ("2026-11-03", "Electric utility", -88.10),
            ("2026-11-01", "Rent", -1450.00),
        ],
        "savings": [("2026-11-01", "Monthly transfer in", 500.00), ("2026-10-31", "Interest", 8.12)],
    }

    def require_verified() -> None:
        if not state["verified"]:
            raise ValueError("Identity not verified yet. Ask the caller for their 4-digit PIN and use verify_identity.")

    def verify_identity(pin: str) -> dict:
        """Verify the caller with their 4-digit phone banking PIN. Required before any account information.

        Args:
            pin: The four digits the caller says, without separators.
        """
        if state["attempts"] >= 3:
            raise ValueError("Too many failed attempts. The caller must contact a branch.")
        if pin.replace(" ", "").replace("-", "") != DEMO_PIN:
            state["attempts"] += 1
            raise ValueError(f"That PIN is incorrect ({3 - state['attempts']} attempts left).")
        state["verified"] = True
        return {"verified": True, "customer": "Alex Morgan"}

    def get_balance(account_type: Literal["checking", "savings"]) -> dict:
        """Get the current balance of one of the caller's accounts.

        Args:
            account_type: checking or savings.
        """
        require_verified()
        return {"account": account_type, "balance_usd": round(balances[account_type], 2)}

    def recent_transactions(account_type: Literal["checking", "savings"], count: int = 3) -> list[dict]:
        """List the most recent transactions on an account.

        Args:
            account_type: checking or savings.
            count: How many to return (1 to 5).
        """
        require_verified()
        rows = history[account_type][: max(1, min(count, 5))]
        return [{"date": d, "description": desc, "amount_usd": amt} for d, desc, amt in rows]

    def transfer_funds(
        from_account: Literal["checking", "savings"], to_account: Literal["checking", "savings"], amount_usd: float
    ) -> dict:
        """Move money between the caller's own accounts. Read the amount back and get a yes before calling.

        Args:
            from_account: Source account.
            to_account: Destination account.
            amount_usd: Amount in US dollars, greater than zero.
        """
        require_verified()
        if from_account == to_account:
            raise ValueError("Source and destination are the same account.")
        if amount_usd <= 0:
            raise ValueError("The amount must be greater than zero.")
        if amount_usd > balances[from_account]:
            raise ValueError(f"Insufficient funds: {from_account} has ${balances[from_account]:.2f}.")
        balances[from_account] -= amount_usd
        balances[to_account] += amount_usd
        today = datetime.now().strftime("%Y-%m-%d")
        history[from_account].insert(0, (today, f"Transfer to {to_account}", -amount_usd))
        history[to_account].insert(0, (today, f"Transfer from {from_account}", amount_usd))
        return {
            "transferred_usd": amount_usd,
            "checking_usd": round(balances["checking"], 2),
            "savings_usd": round(balances["savings"], 2),
        }

    def report_lost_card() -> dict:
        """Block the caller's debit card because it is lost or stolen and order a replacement."""
        require_verified()
        state["card_blocked"] = True
        return {"card": "blocked", "replacement": "arrives in 3 to 5 business days", "reference": "LC-88213"}

    return [verify_identity, get_balance, recent_transactions, transfer_funds, report_lost_card]


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class Scenario:
    """One use case: its prompt text, UI text, and a factory for its tools."""

    id: str
    name: str
    description: str
    instructions: str
    greeting: str  # instruction for the opening line
    examples: tuple[str, ...]
    make_tools: Callable[[], list[Callable]] = field(repr=False)


# Scenario id -> (prompt key in prompts.yaml, tool factory). A factory builds fresh demo state for each session.
USE_CASES: dict[str, tuple[str, Callable[[], list[Callable]]]] = {
    "generic": ("generic_assistant", lambda: [get_current_datetime, get_weather]),
    "insurance": ("insurance_claims", _insurance_tools),
    "clinic": ("clinic_booking", _clinic_tools),
    "bank": ("bank_assistant", _bank_tools),
}


def load_scenarios(prompts_file: Path = PROMPTS_FILE) -> dict[str, Scenario]:
    """Build the scenarios from the prompt catalog. An entry with `voice_style: true` gets the shared voice style."""
    prompts = yaml.safe_load(prompts_file.read_text())
    style = prompts["voice_style"]["content"].strip()
    scenarios = {}
    for scenario_id, (key, make_tools) in USE_CASES.items():
        entry = prompts[key]
        instructions = entry["content"].strip()
        if entry.get("voice_style"):
            instructions = f"{instructions} {style}"
        scenarios[scenario_id] = Scenario(
            id=scenario_id,
            name=entry["name"],
            description=entry["description"].strip(),
            instructions=instructions,
            greeting=entry["greeting"].strip(),
            examples=tuple(entry["examples"]),
            make_tools=make_tools,
        )
    return scenarios


SCENARIOS = load_scenarios()


def get_scenario(scenario_id: object) -> Scenario:
    """Return the requested scenario, or the default when the id is missing or unknown."""
    return (
        SCENARIOS.get(scenario_id, SCENARIOS[DEFAULT_SCENARIO])
        if isinstance(scenario_id, str)
        else SCENARIOS[DEFAULT_SCENARIO]
    )


def scenario_catalog() -> list[dict]:
    """Return what the UI needs to render the picker."""
    return [
        {"id": s.id, "name": s.name, "description": s.description, "examples": list(s.examples)}
        for s in SCENARIOS.values()
    ]
