# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Schema-driven normalization of spoken identifiers for the Frontend/Backend Agent.

* :mod:`.classify` splits caller tools into reads and writes from their own
  names and descriptions; a write's arguments are never rewritten.
* :mod:`.schema_rules` derives identifier rules, complete-spelling patterns,
  phone arguments and spoken labels from the session's tool schemas only.
* :mod:`.arguments` screens a read tool's call, putting dictated identifiers
  into the shape their schema example shows and flagging unfinished ones.
* :mod:`.transcript` writes spelled runs in a user transcript as identifiers
  and detects a spelling a pause split across turns.
* :mod:`.phone` gives the other common form of a ten-digit phone number.

Pure standard-library code with no I/O.
"""

from examples.frontend_backend_agent.src.normalization.arguments import (
    ScreenResult,
    ToolCall,
    canonical_json,
    screen_call,
)
from examples.frontend_backend_agent.src.normalization.classify import (
    MUTATING_NAME_TOKENS,
    READ_NAME_PREFIXES,
    tool_kind,
)
from examples.frontend_backend_agent.src.normalization.phone import other_phone_form, ten_digits
from examples.frontend_backend_agent.src.normalization.rules import IdentifierRule, canonicalize, is_incomplete
from examples.frontend_backend_agent.src.normalization.schema_rules import (
    EMPTY_SCHEMA_RULES,
    SchemaRules,
    build_schema_rules,
    tools_sha256,
)
from examples.frontend_backend_agent.src.normalization.transcript import (
    ends_mid_spelling,
    join_across_turns,
    matches_complete_pattern,
    normalize_transcript,
    spelled_runs,
)

__all__ = [
    "EMPTY_SCHEMA_RULES",
    "MUTATING_NAME_TOKENS",
    "READ_NAME_PREFIXES",
    "IdentifierRule",
    "SchemaRules",
    "ScreenResult",
    "ToolCall",
    "build_schema_rules",
    "canonical_json",
    "canonicalize",
    "ends_mid_spelling",
    "is_incomplete",
    "join_across_turns",
    "matches_complete_pattern",
    "normalize_transcript",
    "other_phone_form",
    "screen_call",
    "spelled_runs",
    "ten_digits",
    "tool_kind",
    "tools_sha256",
]
