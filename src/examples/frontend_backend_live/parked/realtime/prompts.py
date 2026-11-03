# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text and tool schema for a remote realtime frontend.

A remote realtime model has no routing step of ours in front of it, so what the cascade's routing prompt and
guards do is asked of the model here: call ``delegate`` for work that needs the backend, say a short lead-in, and
speak a backend result only once it arrives.
"""

DELEGATE_TOOL = {
    "type": "function",
    "name": "delegate",
    "description": (
        "Hand work to the backend assistant, which has the tools and verified data you lack. Call it for anything "
        "that depends on that data or changes it. Returns at once; the verified result arrives later."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "A complete, self-contained description of what the caller needs done or looked up.",
            }
        },
        "required": ["request"],
    },
}

DELEGATE_INSTRUCTIONS = """
## Delegating work
To involve the backend, call the delegate function with a complete description of the request, then tell the
caller in one short sentence that you are checking. Do not ask a question or state an outcome in that sentence.
The verified result arrives afterwards as a system message. Never state a price, a total, an order number or any
other backend fact before it does, and speak it only as the system message gives it.
""".strip()

DELEGATION_STARTED = (
    '{"status": "working", "delegation_id": "%s", "note": "The verified result will arrive as a system message. '
    'Do not state any result until then."}'
)
RESULT_ITEM = "Verified backend result for delegation {delegation_id}:\n{text}"
TOLD_NOTE = 'You already told the caller: "{told}". Do not repeat that; report only the result.'
SPEAK_RESULT = (
    "Tell the caller this result now, in one or two short spoken sentences, using only what the result says. {note}"
)
NEWER_NOTE = (
    "The caller has since asked something else, still in progress: {asked}. If it changes this result, say what was "
    "found, briefly, and that you are updating it; otherwise report this result normally."
)
COMMENTARY_ITEM = "Update from the application:\n{text}"
SPEAK_UPDATE = "Say this update to the caller in one or two short spoken sentences, using only what it says. {note}"
THINKING_ITEM = "Background context (do not mention it unless it becomes relevant):\n{text}"
INSTRUCTION_ITEM = "New instruction from the application (it replaces any conflicting earlier direction):\n{text}"
