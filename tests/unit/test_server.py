# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for FastAPI server request handling."""

import threading
import unittest
from unittest.mock import patch

import server


class ServiceCatalogEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Verify service catalog request behavior."""

    async def test_service_catalog_build_runs_off_event_loop(self) -> None:
        """Build the service catalog outside the event-loop thread."""
        event_loop_thread = threading.get_ident()
        app = server.create_app()
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/services")

        with patch.object(server, "build_services_api_response", side_effect=lambda: threading.get_ident()):
            worker_thread = await endpoint(pipeline_mode="generic-assistant")

        self.assertNotEqual(worker_thread, event_loop_thread)


if __name__ == "__main__":
    unittest.main()
