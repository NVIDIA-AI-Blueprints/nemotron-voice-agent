# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that an application reads only the uniquely named catalog entry."""

from unittest.mock import Mock

import pytest

from benchmarking_tools.live_catalog.lookup import lookup_named_entry

CATALOG = {"Headphones": "h1", "Wireless Earbuds": "e1", "Lamp (USB)": "l1"}


@pytest.mark.parametrize(
    ("request_text", "product_id"),
    [
        ("How much are the in-stock black wireless in-ear HEADPHONES?", "h1"),
        ("How much are the Wireless   Earbuds?", "e1"),
        ("Is the lamp (USB) available?", "l1"),
    ],
)
def test_reads_explicit_catalog_label(request_text, product_id):
    """A caller's exact label selects the record even if another category shares its attributes."""
    read = Mock(return_value={"price": 344.55})
    result = lookup_named_entry(request_text, CATALOG, read)
    assert result["status"] == "matched"
    assert result["product_id"] == product_id
    assert result["product"] == {"price": 344.55}
    read.assert_called_once_with(product_id)


@pytest.mark.parametrize(
    ("request_text", "status"),
    [
        (None, "missing_request"),
        ("Do you sell HeadphonesXL?", "no_match"),
        ("Compare Headphones and Wireless Earbuds.", "ambiguous"),
    ],
)
def test_unresolved_requests_do_not_read_product_details(request_text, status):
    """Retain unresolved selection when the input is absent, unmatched, or names multiple products."""
    read = Mock()
    result = lookup_named_entry(request_text, CATALOG, read)
    assert result["status"] == status
    if status == "ambiguous":
        assert {item["product_id"] for item in result["matches"]} == {"h1", "e1"}
    read.assert_not_called()
