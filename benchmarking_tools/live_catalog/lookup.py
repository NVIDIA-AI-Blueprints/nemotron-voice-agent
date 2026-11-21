# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A client-owned, read-only catalog function for Live applications."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

LOOKUP_TOOL = {
    "type": "function",
    "name": "lookup_named_catalog_entry",
    "description": (
        "Read the catalog entry explicitly named in the caller request bound by the application. "
        "Use this before quoting availability or price. If the lookup is ambiguous, ask which entry the caller means."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def lookup_named_entry(
    request_text: str | None,
    catalog: Mapping[str, str],
    read_product: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Read the unique ID explicitly named in an application-bound request.

    The caller supplies the binding and a read-only catalog API. The function matches lexical names.
    The worker still interprets the request and applies the requested attributes to the returned record.
    """
    request = _normalize(request_text or "")
    if not request:
        return {"status": "missing_request"}

    matches: dict[str, str] = {}
    for name, product_id in catalog.items():
        label = _normalize(name)
        if label and re.search(rf"(?<!\w){re.escape(label)}(?!\w)", request):
            matches.setdefault(product_id, name)

    if not matches:
        return {"status": "no_match"}
    if len(matches) > 1:
        return {"status": "ambiguous", "matches": [{"name": name, "product_id": key} for key, name in matches.items()]}
    ((product_id, name),) = matches.items()
    return {"status": "matched", "name": name, "product_id": product_id, "product": read_product(product_id)}
