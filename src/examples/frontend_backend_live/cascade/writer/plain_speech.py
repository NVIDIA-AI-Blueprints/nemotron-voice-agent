# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend's words as plain spoken sentences, for a result said as written."""

import re


def plain_speech(text: str) -> str:
    """The answer's words without markup: lines become sentences, bullets and emphasis go; numbers and ids stay.

    A list marker goes only when a space follows it, so "12.50 is your total" and "-5 degrees" keep their numbers, and
    stacked markers ("## 1.", "- -") go together. Underscores stay, so an id such as Q91_T is said as written.
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:(?:[-*\u2022]+|\d+[.)]|#+)\s+)+", "", line)
        line = re.sub(r"[*`]+", "", line).strip()
        if line:
            lines.append(line if line[-1] in ".!?:" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines).replace(":.", ":")).strip()
