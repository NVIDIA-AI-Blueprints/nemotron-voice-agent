# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancel a native worker invocation before sending its pending correction."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bridge import WorkerDoor
    from .protocol import HarnessSession
    from .worker import WorkerRun, _Turn


class InputRecovery:
    """Own one cancellation and recovery boundary for the exact native invocation."""

    def __init__(self, run: WorkerRun, turn: _Turn, door: WorkerDoor, session: HarnessSession):
        """Capture the invocation the reader selected before awaiting any settlement."""
        self.run, self.turn, self.door, self.session = run, turn, door, session
        self.foreground = run.foreground
        self.runtime = run.runtime

    @classmethod
    def when_ready(cls, run: WorkerRun) -> InputRecovery | None:
        """Wait for native dispatch and active calls before interrupting the old invocation."""
        session, turn, door = run.session, run.foreground.turn, run.door
        if (
            session is None
            or door is None
            or turn is None
            or turn.cancelled_by_us
            or not run.can_recover_input()
            or not run.task.can_dispatch()
            or run.foreground.running
            or run.foreground.cards_open
            or run.foreground.forwarded
            or door.calls_pending_for(turn)
            or not session.prompt_state(turn.main).dispatched
        ):
            return None
        run.task.trace("harness.input_recovery_requested", prompt=turn.main, invocation_id=turn.invocation_id)
        return cls(run, turn, door, session)

    async def finish(self) -> AsyncIterator[dict]:
        """Confirm the old turn and real tool effects before sending the correction.

        Reuse requires evidence that a previous cancellation cannot hit the new prompt. Otherwise the
        logical task opens with its retained history in a fresh native session.
        """
        run, turn, door, session = self.run, self.turn, self.door, self.session
        if (
            self.foreground is not run.foreground
            or turn is not run.foreground.turn
            or door is not run.door
            or session is not run.session
            or not run.can_recover_input()
            or not run.task.can_dispatch()
        ):
            return
        terminal = session.prompt_state(turn.main).terminal
        if terminal is not None and terminal.terminal_confirmed and session.quiescent:
            run.drain_recovery_events()
            if not (
                door.pending_calls
                or run.foreground.running
                or run.foreground.runtime_calls
                or run.foreground.forwarded
                or run.foreground.cards_open
            ):
                async for event in run.finish_turn():
                    yield event
                return

        pause = run.pause_admissions()
        failure = None
        try:
            async with asyncio.timeout(run.settings.stop_timeout_seconds):
                if pause is None:
                    raise RuntimeError("the old invocation's admission cannot be paused")
                cancel = run.cancel_turn()
                if cancel is not None:
                    await asyncio.shield(cancel)
                await session.ended(turn.main)
                await session.idle()
                await door.drain_admissions()
                run.drain_recovery_events()
                while door.pending_calls:
                    await asyncio.gather(*(asyncio.shield(call) for call in tuple(door.pending_calls)))
                run.drain_recovery_events()
                while run.foreground.running:
                    run.apply_event(await run.mailbox.get())
                run.drain_recovery_events()
                terminal = session.prompt_state(turn.main).terminal
                if terminal is None or not terminal.terminal_confirmed or not session.quiescent:
                    raise RuntimeError("the native invocation's terminal is unconfirmed")
                completion = run.agent.completion(terminal)
                if completion.outcome not in ("succeeded", "cancelled"):
                    raise RuntimeError(completion.error or f"native turn {completion.outcome}")
                if run.foreground.runtime_calls or run.foreground.forwarded:
                    raise RuntimeError("the old invocation still has unsettled tool activity")
        except (TimeoutError, RuntimeError) as error:
            detail = str(error) or "native cancellation, terminal or tool settlement exceeded the stop bound"
            failure = f"owned-input recovery failed: {type(error).__name__}: {detail}"
        if run.stopped or run.foreground.done or run.session is not session or run.runtime is not self.runtime:
            return
        if failure is not None:
            run.return_next_prompts()
            async for event in run.fail_round(failure, usage=run.turn_usage(turn)):
                yield event
            return
        if not (run.task.can_dispatch() and run.task.pending_input_ready()):
            run.steering_ready()
            async for event in run.fail_round(
                "owned-input recovery no longer has eligible input", usage=run.turn_usage(turn)
            ):
                yield event
            return
        async for event in run.close_round(None, run.turn_usage(turn)):
            yield event
        run.between_rounds()
        if (
            run.stopped
            or run.foreground.done
            or run.released
            or not run.task.can_dispatch()
            or self.foreground is not run.foreground
            or session is not run.session
            or door is not run.door
        ):
            return
        reuse = turn.reusable_completion(completion) and run.resume_admissions(pause)
        run.task.refresh_history(unconfirmed_calls=tuple(turn.call_ids) if reuse else ())
        run.task.trace(
            "harness.input_recovery_ready",
            prompt=turn.main,
            invocation_id=turn.invocation_id,
            same_session=reuse,
        )
        if reuse:
            await run.prompt_next_now()
            return
        why = "a settled old invocation could not safely continue its native session"
        await run.reopen(why, instructions=run.instructions)
