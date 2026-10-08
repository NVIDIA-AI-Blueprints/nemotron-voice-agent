# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Word-level helpers for recognizing short acknowledgments ("uh huh", "okay")."""

import re


def words(text: str) -> list[str]:
    """Return the lowercase words of ``text``."""
    return re.findall(r"[a-z']+", text.lower())


def continuer_vocabulary(continuers) -> set[str]:
    """Return every word that appears in a configured continuer phrase."""
    return {word for phrase in continuers for word in words(phrase)}
