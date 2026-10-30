# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The work ledger's record tables and their allowed and refused moves."""

import unittest

from delegation_core.ledger.records import (
    MAX_KEPT_CLOSED,
    Action,
    Delegation,
    Response,
    Result,
    Session,
    Steering,
    Stop,
    Task,
    TransitionError,
    Work,
    WorkLedger,
)

ENDS_WITH_ITS_WORK = {"answered": set(), "failed": set(), "cancelled": set()}
# Each record's lifecycle: every state, and the states it may move to.
TABLES = {
    Session: {"active": {"ended"}, "ended": set()},
    Delegation: {
        "queued": {"running", "answered", "handed_over", "cancelled"},
        "running": {"waiting_client", "answered", "handed_over", "failed", "cancelled"},
        "waiting_client": {"running", "handed_over", "failed", "cancelled"},  # handed over at the client-tool deadline
        "handed_over": set(),
        **ENDS_WITH_ITS_WORK,
    },
    Response: {
        "queued": {"in_progress", "incomplete"},
        "in_progress": {"completed", "failed", "incomplete"},
        "completed": set(),
        "failed": set(),
        "incomplete": set(),
    },
    Work: {
        "queued": {"running", "cancelled"},
        "running": {"waiting_client", "answered", "failed", "cancelled"},
        "waiting_client": {"running", "failed", "cancelled"},
        **ENDS_WITH_ITS_WORK,
    },
    Task: {
        "queued": {"running", "cancelled"},
        "running": {"waiting_client", "answered", "failed", "cancelled"},
        "waiting_client": {"running", "failed", "cancelled"},
        **ENDS_WITH_ITS_WORK,
    },
    Steering: {
        "waiting": {"taken", "follow_up"},
        "taken": {"applied", "not_applied", "follow_up"},
        "not_applied": {"taken", "follow_up"},
        "applied": set(),
        "follow_up": set(),
    },
    Stop: {
        "none": {"stop_requested"},
        "stop_requested": {"stopped", "unconfirmed"},
        "stopped": set(),
        "unconfirmed": set(),
    },
    Result: {"none": {"final"}, "final": {"pushed"}, "pushed": {"expired"}, "expired": set()},
    Action: {
        "started": {"done", "failed", "unclear", "not_run"},
        "done": set(),
        "failed": set(),
        "unclear": set(),
        "not_run": set(),
    },
}
STARTS = {
    Session: "active",
    Delegation: "queued",
    Response: "queued",
    Work: "queued",
    Task: "queued",
    Steering: "waiting",
    Stop: "none",
    Result: "none",
    Action: "started",
}


def record_in(cls, state):
    """A fresh record of ``cls``, put in ``state`` directly (the table is what the test checks)."""
    record = cls("r1")
    record.state = state
    return record


def user(text):
    return {"type": "message", "role": "user", "content": text}


class TransitionTableTests(unittest.TestCase):
    def test_each_record_starts_in_its_first_state(self):
        for cls, start in STARTS.items():
            with self.subTest(record=cls.__name__):
                self.assertEqual(cls("r1").state, start)

    def test_every_allowed_move_is_made(self):
        for cls, table in TABLES.items():
            for state, targets in table.items():
                for target in targets:
                    with self.subTest(record=cls.__name__, move=f"{state} -> {target}"):
                        record = record_in(cls, state)
                        record.move(target)
                        self.assertEqual(record.state, target)

    def test_every_other_move_is_refused_and_changes_nothing(self):
        for cls, table in TABLES.items():
            for state, targets in table.items():
                for target in set(table) - targets:
                    with self.subTest(record=cls.__name__, move=f"{state} -> {target}"):
                        record = record_in(cls, state)
                        with self.assertRaisesRegex(TransitionError, f"may not move from {state} to {target}"):
                            record.move(target)
                        self.assertEqual(record.state, state)

    def test_a_record_has_ended_in_a_state_that_nothing_leaves(self):
        for cls, table in TABLES.items():
            for state, targets in table.items():
                with self.subTest(record=cls.__name__, state=state):
                    self.assertEqual(record_in(cls, state).ended, not targets)


class WorkLedgerTests(unittest.TestCase):
    def test_a_delegation_holds_its_request_as_sent(self):
        ledger = WorkLedger("conv_1")
        items, settings = [user("Find teas.")], {"model": "org/a"}
        delegation = ledger.new_delegation("item_1", input_items=items, settings=settings)
        items.append(user("changed"))
        settings["model"] = "changed"
        self.assertEqual((delegation.input, delegation.settings), ([user("Find teas.")], {"model": "org/a"}))
        self.assertEqual((delegation.state, delegation.work, delegation.handed_to), ("queued", "", None))
        self.assertEqual(ledger.new_response("resp_1", "item_1").state, "queued")  # opened at acceptance

    def test_works_and_their_tasks_are_numbered_and_the_newest_request_carries_the_work(self):
        ledger = WorkLedger("conv_1")
        first = ledger.new_work(["item_a"], {"model": "org/a"})
        second = ledger.new_work(["item_b", "item_c"], {}, briefs=("menu", "hours"))
        self.assertEqual((first.id, second.id), ("W1", "W2"))
        self.assertEqual([(task.id, task.brief) for task in first.tasks], [("W1.1", None)])  # the requests as they came
        self.assertEqual([(task.id, task.brief) for task in second.tasks], [("W2.1", "menu"), ("W2.2", "hours")])
        self.assertEqual((second.carrier, second.requests, second.state), ("item_c", ["item_b", "item_c"], "queued"))
        self.assertEqual({task.state for task in second.tasks}, {"queued"})
        self.assertEqual(first.settings, {"model": "org/a"})
        self.assertIs(ledger.tasks["W2.2"], second.tasks[1])
        self.assertEqual(ledger.session.state, "active")

    def test_steering_is_numbered_per_session_and_names_its_task_and_requests(self):
        ledger = WorkLedger("conv_1")
        ledger.new_work(["item_a"], {})
        first = ledger.new_steering("W1.1", ["item_b"], [user("Make it two.")])
        second = ledger.new_steering("W1.1", ["item_c", "item_d"], [user("Green."), user("Hot.")])
        self.assertEqual((first.id, second.id), ("S1", "S2"))
        self.assertEqual((first.state, first.task, first.delegations), ("waiting", "W1.1", ["item_b"]))
        self.assertEqual(second.items, [user("Green."), user("Hot.")])

    def test_only_the_newest_closed_works_are_kept_and_the_forgotten_ones_are_returned(self):
        # The ledger moves no record itself: the IM expires the forgotten Works' results.
        ledger = WorkLedger("conv_1")
        works = []
        for number in range(MAX_KEPT_CLOSED + 5):
            work = ledger.new_work([f"item_{number}"], {})
            task = work.tasks[0]
            work.move("running")
            task.move("running")
            task.result.move("final")
            task.result.move("pushed")
            task.move("answered")
            work.move("answered")
            works.append(work)
        still_open = ledger.new_work(["item_open"], {})
        self.assertEqual(ledger.prune(), works[:5])
        self.assertEqual(len(ledger.works), MAX_KEPT_CLOSED + 1)
        self.assertIn(still_open.id, ledger.works)
        self.assertNotIn(works[4].id, ledger.works)
        self.assertNotIn("W5.1", ledger.tasks)  # a forgotten Work's tasks go with it
        self.assertIn("W6.1", ledger.tasks)
        self.assertEqual({work.tasks[0].result.state for work in works}, {"pushed"})
        self.assertEqual(ledger.prune(), [])

    def test_a_kept_work_keeps_the_records_it_needs_and_a_forgotten_one_takes_its_own_along(self):
        ledger = WorkLedger("conv_1")
        ledger.new_delegation("item_old", input_items=[user("Find teas.")]).move("handed_over")  # its Work goes on
        handover = ledger.new_response("resp_old", "item_old")
        handover.move("in_progress")  # a handover passes in_progress, then ends with its line
        handover.move("completed")
        ledger.delegations["item_old"].responses.append("resp_old")
        work = ledger.new_work(["item_old", "item_new"], {})
        ledger.new_action("call_kept", "W1.1", "get_menu", "app").move("done")
        ledger.new_steering("W1.1", ["item_new"], [user("Green.")]).move("taken")
        ledger.steerings["S1"].move("applied")
        for number in range(MAX_KEPT_CLOSED + 3):
            # A delegation a conductor answered (no Work), with its response; and a closed Work with its records.
            ledger.new_delegation(f"item_{number}", input_items=[]).move("answered")
            ledger.new_response(f"resp_{number}", f"item_{number}").move("incomplete")
            closed = ledger.new_work([f"work_item_{number}"], {})
            closed.move("cancelled")
            ledger.new_action(f"call_{number}", closed.tasks[0].id, "get_menu", "backend").move("unclear")
            ledger.new_steering(closed.tasks[0].id, [], []).move("follow_up")
        forgotten = ledger.prune()
        self.assertEqual([work.id for work in forgotten], ["W2", "W3", "W4"])
        for kind, key in (
            ("delegations", "item_old"),
            ("responses", "resp_old"),
            ("actions", "call_kept"),
            ("steerings", "S1"),
        ):
            with self.subTest(kept=key):
                self.assertIn(key, getattr(ledger, kind))  # ended, but the open Work W1 needs it
        for kind, key in (("tasks", "W2.1"), ("actions", "call_0"), ("steerings", "S2")):
            with self.subTest(gone=key):
                self.assertNotIn(key, getattr(ledger, kind))  # it went with its forgotten Work
        self.assertIn(work.id, ledger.works)
        for kind in ("works", "tasks", "actions", "steerings", "delegations", "responses"):
            with self.subTest(count=kind):  # what W1 needs is kept and not counted
                self.assertEqual(len(getattr(ledger, kind)), MAX_KEPT_CLOSED + 1)

    def test_open_records_are_never_dropped(self):
        ledger = WorkLedger("conv_1")
        ledger.new_work(["item_x"], {})
        for number in range(MAX_KEPT_CLOSED + 3):
            ledger.new_delegation(f"item_{number}", input_items=[]).move("cancelled")
            ledger.new_action(f"call_{number}", "W1.1", "get_menu", "backend").move("done")
        ledger.works["W1"].move("cancelled")
        ledger.new_delegation("item_open", input_items=[])
        ledger.new_work(["item_open"], {})
        ledger.new_action("call_open", "W2.1", "get_menu", "app")
        ledger.prune()
        self.assertEqual(len(ledger.delegations), MAX_KEPT_CLOSED + 1)
        self.assertEqual(
            (ledger.delegations["item_open"].state, ledger.actions["call_open"].state), ("queued", "started")
        )


if __name__ == "__main__":
    unittest.main()
