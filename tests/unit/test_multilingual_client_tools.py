# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json

from pipecat.adapters.schemas.tools_schema import AdapterType

from examples.multilingual.tools import build_client_tools, perceivable_client_tools


def _tool(name, **extra):
    return {"name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": {}}, **extra}


def test_perceivable_flag_is_validated_and_kept_out_of_the_llm_schema():
    declared = [
        _tool("move_head", perceivable=True),
        _tool("get_time"),  # absent means false
        _tool("search", perceivable=False),
        _tool("dance", perceivable="yes"),  # not a boolean: the tool is kept, the flag is not
        _tool("bad name!", perceivable=True),  # rejected declarations never contribute a flag
        _tool("broken", perceivable=True, parameters="nope"),
        _tool("move_head", perceivable=False),  # duplicate: the first declaration counts
        "not an object",
    ]
    schema, names = build_client_tools(declared)
    assert names == ("move_head", "get_time", "search", "dance")
    assert perceivable_client_tools(declared, names) == ("move_head",)
    assert "perceivable" not in json.dumps(schema.custom_tools[AdapterType.OPENAI])


def test_perceivable_flag_ignores_malformed_payloads():
    assert perceivable_client_tools(None, ()) == ()
    assert perceivable_client_tools({"name": "x", "perceivable": True}, ("x",)) == ()
    # A name the server did not accept (for example past the tool limit) is never perceivable.
    assert perceivable_client_tools([_tool("ghost", perceivable=True)], ("other",)) == ()
