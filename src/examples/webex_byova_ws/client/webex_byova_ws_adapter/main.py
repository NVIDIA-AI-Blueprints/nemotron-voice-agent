"""Adapter process entrypoint."""

from __future__ import annotations

import logging

from aiohttp import web

from webex_byova_ws_adapter.config import AdapterConfig
from webex_byova_ws_adapter.server import create_app, listener_ssl


def main() -> None:
    """Validate configuration and run the TLS WebSocket listener."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = AdapterConfig()
    config.validate()
    web.run_app(
        create_app(config),
        host=config.host,
        port=config.port,
        ssl_context=listener_ssl(config),
        access_log=logging.getLogger("webex_byova_ws_adapter.access"),
    )


if __name__ == "__main__":
    main()
