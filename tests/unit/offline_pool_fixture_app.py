# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-free child application for real process/lease/opaque-wire checks."""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect


def create_app():
    """Echo opaque frames and hold handler cleanup behind a test-only file."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/deployment")
    async def deployment():
        return {"fixture": "opaque-wire", "pid": os.getpid()}

    @app.websocket("/api/ws/offline")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"pid": os.getpid()})
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await websocket.send_bytes(message["bytes"])
                else:
                    await websocket.send_text(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            root = os.getenv("OFFLINE_POOL_FIXTURE_RELEASE_DIR")
            if root:
                while not (Path(root) / str(os.getpid())).exists():
                    await asyncio.sleep(0.01)

    return app
