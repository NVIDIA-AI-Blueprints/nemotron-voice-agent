# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's conductor (``port.Conductor``): an agent session that places requests through the control tools."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    Decision,
    DecisionRejected,
    Place,
    PlacementExpired,
)

from . import usage
from .bridge import ConductorDoor
from .protocol import (
    HarnessSession,
    Mailbox,
    PromptDispatched,
    PromptEnded,
    Runtime,
    Wake,
    usage_total,
)
from .runtime import cancel_in_background, give_back

if TYPE_CHECKING:
    from .adapter import HarnessAdapter


class ConductorRoundFailed(RuntimeError):
    """The current native turn failed, after its session was safely retired."""


class _Turn:
    """One conductor turn: it reads the session's events until its prompt ends, and ignores the rest."""

    def __init__(
        self,
        prompt: int,
        total_before: int | None,
        started: float,
        delegation_ids: tuple[str, ...] = (),
        conversation_ref: str | None = None,
        *,
        round_number: int,
        session_id: str,
    ):
        self.prompt = prompt
        self.total_before = total_before
        self.started = started
        self.timer: asyncio.TimerHandle | None = None
        self.ended: PromptEnded | None = None
        self.failure: Exception | None = None
        self.delegation_ids = delegation_ids
        self.conversation_ref = conversation_ref
        self.round, self.session_id = round_number, session_id
        self.usage_reported = False
        self.control_completed = False
        self.control_started = False
        self.superseded = False
        self.decision = None
        self.receipt: str | None = None
        self.invocation_id: str | None = None
        self.rejection: DecisionRejected | None = None

    def prompt_dispatched(self, event: PromptDispatched) -> None:
        if event.prompt == self.prompt:
            self.invocation_id = event.invocation_id

    def prompt_ended(self, event: PromptEnded) -> None:
        if event.prompt == self.prompt:
            self.ended = event
            self.invocation_id = event.invocation_id

    def __getattr__(self, name: str):
        """Ignore every other event: the conductor's text is never used."""
        return _ignore


def _ignore(event: object) -> None:
    """Ignore an event."""


class HarnessConductor:
    """The conductor of one IM session, in a runtime process of its own so workers never take its turns."""

    model_placement = True

    def __init__(self, host: HarnessAdapter):
        """Place through a session of ``host``'s conductor process; nothing opens until warm or a decision."""
        self.host, self.settings = host, host.settings
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: ConductorDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self.blocks: tuple[str, ...] = ()
        self.fresh = True
        self.turn: _Turn | None = None
        self.total: int | None = None
        self._opening: asyncio.Task | None = None
        self._reading = asyncio.Lock()
        self.attached = True
        self._attachment_generation = 0
        self.round = 0
        self.usage: dict[str, int] = {}

    def warm(self) -> None:
        """Open the session in the background, so the first request does not wait for the agent's start."""
        if (
            self.attached
            and not self.host.closed
            and self.session is None
            and (self._opening is None or self._opening.done())
        ):
            self._opening = asyncio.ensure_future(self._open())
            self._opening.add_done_callback(_retrieved)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Run one native decision turn and submit its control call to Core.

        A superseded placement cancels its native turn. A turn left running after a hang-up is drained before the
        next prompt. A failed turn retires the session.
        """
        generation = self._attachment_generation
        try:
            await self._turn(given, place)
        except Exception as exc:
            logger.warning(f"harness.conductor_failed {type(exc).__name__}: {exc}")
            if generation == self._attachment_generation:
                await self._retire(cancel=True)
            raise

    async def _turn(self, given: ConductorInput, place: Place) -> None:
        generation = self._attachment_generation
        turn = None
        try:
            session = await self._free()
            if not self.attached or generation != self._attachment_generation:
                raise ConductorRoundFailed("The conductor attachment changed before its prompt")
            mailbox = self.mailbox

            def reject(error: DecisionRejected) -> str:
                # Core owns repair after a rejection proved to precede admission.
                turn.rejection = error
                mailbox.put_nowait(Wake())
                return json.dumps({"status": "rejected", "error": str(error)})

            async def placed(decision: Decision | DecisionRejected) -> str:
                if turn.superseded:
                    return json.dumps({"status": "superseded", "executed": False})
                if turn.rejection is not None:
                    return json.dumps({"status": "rejected", "error": str(turn.rejection)})
                if turn.control_started:
                    if decision == turn.decision:
                        return turn.receipt or json.dumps({"status": "pending"})
                    return json.dumps({"status": "refused", "reason": "decision_already_submitted"})
                if self.turn is not turn:
                    return self.host.vocabulary.idle
                if isinstance(decision, DecisionRejected):
                    return reject(decision)
                turn.control_started, turn.decision = True, decision
                try:
                    outcome = await place(decision)
                except DecisionRejected as error:
                    turn.control_started, turn.decision = False, None
                    return reject(error)
                if isinstance(outcome, PlacementExpired):
                    turn.control_started, turn.decision = False, None
                    return json.dumps({"status": "superseded", "executed": False})
                turn.rejection = None
                turn.receipt = outcome
                if (
                    self.turn is turn
                    and turn.ended is None
                    and self.attached
                    and generation == self._attachment_generation
                ):
                    turn.control_completed = True
                return outcome

            self.door.bind(placed)
            text = self.host.renderer.conductor_input(given, fresh=self.fresh)
            if self.fresh and self.blocks:
                text = "\n\n".join([*self.blocks, text])
            loop = asyncio.get_running_loop()
            turn = self.turn = _Turn(
                session.prompt(
                    text,
                    may_dispatch=lambda: (
                        self.attached
                        and not self.host.closed
                        and generation == self._attachment_generation
                        and self.session is session
                    ),
                ),
                self.total,
                loop.time(),
                tuple(row.delegation_id for row in given.arrivals),
                given.conversation_ref,
                round_number=self.round + 1,
                session_id=session.id,
            )
            self.round += 1
            limit = self.settings.conductor_timeout_seconds or self.settings.turn_timeout_seconds
            turn.timer = loop.call_later(limit, self._overran, turn)
            self.fresh = False
            await self._read(turn)
            if turn.failure is not None:
                raise turn.failure
        except asyncio.CancelledError as cancellation:
            if cancellation.args == (PlacementExpired,) and turn is not None and turn is self.turn:
                turn.superseded = True
                if turn.prompt in session.outstanding:
                    cancel_in_background(session, self.settings.stop_timeout_seconds)
                    turn.timer.cancel()
                    turn.timer = asyncio.get_running_loop().call_later(
                        self.settings.stop_timeout_seconds, self._overran, turn
                    )
            try:
                while self.turn is not None:
                    await self._read(self.turn)
            except DecisionRejected:
                await self._retire(cancel=True)
            raise

    async def _free(self) -> HarnessSession:
        """Return the session once its agent is free: a turn still running is read to its end and judged first.

        Such a turn is one a ``decide()`` cancelled twice left, such as at a hang-up; when judging it retires the
        session, a fresh one opens.
        """
        session = await self._session()
        while self.turn is not None:
            # A failed old turn being drained is not the current request's model call.
            try:
                await self._read(self.turn)
            except DecisionRejected:
                await self._retire(cancel=True)
            except ConductorRoundFailed:
                pass
            session = await self._session()
        return session

    async def _read(self, turn: _Turn) -> None:
        """Read one turn through completion, rejection, or retirement."""
        async with self._reading:
            mailbox = self.mailbox
            while turn.ended is None and turn.rejection is None and turn is self.turn:
                (await mailbox.get()).apply(turn)
            if turn is self.turn:
                if turn.rejection is not None and turn.ended is None:
                    raise turn.rejection
                await self._judge(turn)

    async def _judge(self, turn: _Turn) -> None:
        """Require a typed successful native terminal and one completed control envelope."""
        ended, total = turn.ended, usage_total(turn.ended.usage)
        counters = usage.round_counters(ended.usage, self.usage, ended.invocation_usage, agent=self.settings.agent)
        self.usage = usage.snapshot(ended.usage)
        ending = "superseded" if turn.superseded else "error" if ended.error is not None else ended.stop_reason
        self._report_usage(turn, counters, ending=ending)
        turn.timer.cancel()
        self.turn = None
        if total is not None:
            self.total = total
        if turn.superseded:
            if not ended.terminal_confirmed or not self.session.quiescent:
                await self._retire(cancel=False)
            return
        if turn.rejection is not None:
            raise turn.rejection
        completion = self.settings.agent.completion(ended)
        if completion.outcome == "succeeded" and turn.control_completed:
            return
        if completion.outcome == "succeeded" and not turn.control_started:
            # Match the direct conductor's clean terminal without a valid decision. Nothing was admitted, so
            # the Core may use its existing bounded repair; a started control still has an uncertain outcome.
            raise DecisionRejected("Conductor response must contain one complete decision call")
        logger.warning(
            "harness.conductor_turn_ended outcome={} error={} control_completed={}",
            completion.outcome,
            completion.error,
            turn.control_completed,
        )
        connection_lost = not ended.terminal_confirmed and not self.runtime.client.alive
        await self._retire(cancel=False)
        if connection_lost and not turn.control_started:
            raise DecisionRejected("Conductor connection closed before a decision was submitted")
        raise ConductorRoundFailed(f"The harness conductor turn failed: {ended.error or ended.stop_reason}")

    def _report_usage(self, turn: _Turn, counters: dict | None, *, ending: str | None) -> None:
        """Observe a started logical prompt once; an unobserved terminal supplies no counters."""
        if turn.usage_reported:
            return
        turn.usage_reported = True
        try:
            model = self.host.source.endpoint().model
            model_source = "configured_endpoint"
        except Exception:
            # EndpointLookup is application-supplied: telemetry cannot interrupt native cleanup.
            model, model_source = None, "unknown"
        observation = (
            usage.observation(self.settings.agent, turn.ended.usage, turn.ended.invocation_usage) if turn.ended else {}
        )
        suffix = " native_usage_observation={}" if observation else ""
        logger.debug(
            "conductor.round.usage backend=harness role=conductor protocol={} model={} model_source={} round={} "
            "conversation_ref={} delegation_ids={} usage={} reported_counters={} usage_missing={} "
            "prompt={} session={} invocation_id={} ending={} usage_authority=conductor_round_mirror" + suffix,
            self.settings.protocol_name,
            model,
            model_source,
            turn.round,
            turn.conversation_ref,
            turn.delegation_ids,
            usage.responses_usage(counters or {}),
            counters or None,
            not counters,
            turn.prompt,
            turn.session_id,
            turn.invocation_id,
            ending,
            *([observation] if observation else []),
        )

    def _overran(self, turn: _Turn) -> None:
        """Cancel a turn still running at its time limit, once, and retire its session."""
        session = self.session
        if turn is not self.turn or turn.prompt not in session.outstanding:
            return
        seconds = asyncio.get_running_loop().time() - turn.started
        logger.warning(f"harness.conductor_turn_overran seconds={seconds:.3f}")
        turn.failure = turn.rejection or TimeoutError("The harness conductor turn exceeded its deadline")
        if not turn.superseded:
            cancel_in_background(session, self.settings.stop_timeout_seconds)
        self._drop()

    async def _session(self) -> HarnessSession:
        """Return the open session, else the one opening, else a new one.

        A lost session is retired, starting a fresh opening, which is the one awaited so two sessions never race.
        """
        replacing = False
        while True:
            if self.session is not None:
                if self.session.alive and self.runtime.client.alive and (self.turn or not self.door.replay_exhausted):
                    return self.session
                replacing = True
                await self._retire(cancel=False)
            if self._opening is not None:
                opening = self._opening
                try:
                    # Shielded: a cancelled decide() leaves the opening to finish.
                    await asyncio.shield(opening)
                except Exception:
                    if replacing:
                        raise
                if self._opening is opening:
                    self._opening = None
                # Capacity may have evicted the newly opened idle session before this waiter resumed.
                continue
            await self._open()

    async def _open(self) -> None:
        """Open a fresh session in a conductor process; a failed or cancelled opening leaves nothing behind."""
        generation = self._attachment_generation
        if not self.attached:
            raise ConductorRoundFailed("The conductor waits for a listening attachment")
        host, prompt = self.host, host_prompt(self.host)
        runtime = None
        session = door = None
        try:
            runtime = await host.shared(host.conductor_role)
            if not self.attached or generation != self._attachment_generation:
                raise ConductorRoundFailed("The conductor attachment changed while its runtime opened")
            if not runtime.client.alive:
                raise ConductorRoundFailed("The conductor runtime closed while its session opened")
            name = runtime.server_name(host.conductor_role.prefix)
            door = host.own_door(ConductorDoor(name, host.vocabulary))
            server = host.bridge.open(door)
            folder = runtime.session_folder(name)
            blocks = self.settings.agent.prepare(folder, prompt)
            mailbox: Mailbox = asyncio.Queue()
            session = await runtime.client.new_session(folder, server, mailbox, role="conductor")
            try:
                async with asyncio.timeout(self.settings.ready_timeout_seconds):
                    await door.listed.wait()
            except TimeoutError:
                raise RuntimeError("The agent did not list the control tools: its MCP connection failed") from None
            if not self.attached or generation != self._attachment_generation:
                raise ConductorRoundFailed("The conductor attachment changed while its session opened")
            if not session.alive or not runtime.client.alive:
                raise ConductorRoundFailed("The conductor session closed while it opened")
        except BaseException:
            if session is not None:
                session.retire()
            if door is not None:
                host.bridge.close(door, "")
            if runtime is not None:
                give_back(runtime, session)
            raise
        self.runtime, self.session, self.door, self.mailbox = runtime, session, door, mailbox
        self.blocks, self.fresh = blocks, True
        self.turn, self.total = None, 0
        self.usage = dict.fromkeys(usage.COUNTERS, 0)
        logger.debug(f"harness.conductor_opened server={name} session={session.id}")

    async def _retire(self, *, cancel: bool) -> None:
        """Retire the session, cancelling a running prompt if ``cancel`` is set, and warm a fresh one."""
        session = self.session
        self._drop()
        if cancel and session is not None and session.outstanding:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(self.settings.stop_timeout_seconds):
                    await session.cancel()

    def _drop(self) -> None:
        """Retire the session, its turn and the turn's backstop at once, and warm a fresh one; it sends no cancel."""
        session, door, runtime, turn = self.session, self.door, self.runtime, self.turn
        self.session = self.door = self.runtime = self.turn = self.total = None
        if turn is not None:
            self._report_usage(turn, None, ending="timeout" if isinstance(turn.failure, TimeoutError) else "incomplete")
            turn.timer.cancel()
            self.mailbox.put_nowait(Wake())
        if session is not None:
            session.retire()
        if door is not None:
            self.host.bridge.close(door, "")
        if runtime is not None:
            give_back(runtime, session)
        if session is not None and not self.host.closed and self.attached:
            self.warm()

    async def set_attachment(self, attached: bool) -> None:
        """Fence this native control generation and request cancellation, without claiming physical completion."""
        if self.attached == attached:
            return
        self.attached = attached
        self._attachment_generation += 1
        if attached:
            if not self.host.closed:
                self.warm()
            return
        opening, self._opening = self._opening, None
        if opening is not None:
            opening.cancel()
        await self._retire(cancel=True)
        if opening is not None:
            await asyncio.gather(opening, return_exceptions=True)

    def attachment_uncertain_sources(self) -> tuple[str, ...]:
        """Return only sources actually dispatched to an unqualified native control prompt still in flight."""
        if self.turn is None:
            return ()
        return self.turn.delegation_ids

    async def close(self) -> None:
        """Stop any opening and retire the session."""
        if self._opening is not None:
            self._opening.cancel()
            await asyncio.gather(self._opening, return_exceptions=True)
            self._opening = None
        await self._retire(cancel=True)


def host_prompt(host: HarnessAdapter) -> str:
    """Return the conductor's instructions: ``texts.conductor_prompt`` for ``max_running`` tasks."""
    return host.vocabulary.instructions(host.max_running)


def _retrieved(task: asyncio.Future) -> None:
    """Log a background opening's error, so it is never left unretrieved."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(f"harness.conductor_open_failed {type(task.exception()).__name__}: {task.exception()}")
