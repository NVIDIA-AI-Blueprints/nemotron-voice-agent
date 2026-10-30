# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The emulator on the backend adapter contract."""

import unittest

from tests.unit.contracts.backend import BackendAdapterContract
from tests.unit.delegation_core.emulator import emulator_source

# Each scenario of the adapter contract, as the emulator's script. The emulator never runs an attempt again, so the
# contract's "tail_error" (a failure after a completed attempt) has nothing to script.
SCENARIOS = {
    "answer": "",
    "own_call": "[own:lookup]",
    "tool_fails": "[own-fail:lookup]",
    "app_call": "[say:Let me look.] [client:get_menu {}]",
    "slow": "[slow:30]",
    "tail_error": "",
}


class EmulatorAdapterContractTests(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    def make_adapter(self, scenario):
        self.source = emulator_source({"script": SCENARIOS[scenario]})
        return self.source.new()

    def release(self):
        self.source.release()

    def sent_instructions(self):
        return [
            request["instructions"]
            for adapter in self.source.adapters
            for run in adapter.runs
            for request in run.requests
        ]


if __name__ == "__main__":
    unittest.main()
