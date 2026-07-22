# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat Eval runner entrypoint for the registered examples.

Launch with Pipecat's runner, for example:

    python src/eval_bot.py -t eval --port 7860
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from pipecat.runner.types import RunnerArguments

import examples_registry
from attachment_store import store_attachment
from utils import filter_session_config, set_service_context

_EVAL_ATTACHMENT_KEY = "eval_attachment"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVAL_ATTACHMENT_FIXTURE_ROOT = _PROJECT_ROOT / "src" / "examples" / "omni_assistant" / "images"
_MAX_EVAL_ATTACHMENT_BYTES = 5 * 1024 * 1024


def _select_example(body: dict) -> dict:
    example_key = str(body.get("pipeline_mode") or body.get("example") or body.get("example_key") or "").strip()
    return examples_registry.find(example_key, ignore_lock=True)


def _bind_example_context(example: dict) -> None:
    module_file = examples_registry.example_module_file(example)
    set_service_context(Path(module_file).resolve().parent, example.get("slots") or None)


def _session_id_for_eval(body: dict, runner_args: RunnerArguments) -> str:
    return str(body.get("session_id") or getattr(runner_args, "session_id", "") or "").strip()


def _prepare_body(body: dict, example: dict, runner_args: RunnerArguments) -> dict:
    config = dict(body)
    config["pipeline_mode"] = example["key"]
    if not config.get("prompt_key") and not config.get("prompt_content"):
        prompt_key = examples_registry.prompt_default_key(example["key"], ignore_lock=True)
        if prompt_key:
            config["prompt_key"] = prompt_key

    sanitized = filter_session_config(config)
    prepared = {**sanitized, "pipeline_mode": example["key"]}
    if session_id := _session_id_for_eval(body, runner_args):
        prepared["session_id"] = session_id
    return prepared


def _preload_eval_attachment(body: dict, session_id: str) -> None:
    """Store one eval-supplied attachment so upload-backed examples can see it."""
    attachment = body.get(_EVAL_ATTACHMENT_KEY)
    if not attachment:
        return
    if not isinstance(attachment, dict):
        raise ValueError(f"{_EVAL_ATTACHMENT_KEY} must be a JSON object")
    if not session_id:
        raise ValueError(f"{_EVAL_ATTACHMENT_KEY} requires a session_id")

    raw_path = str(attachment.get("path") or "").strip()
    if not raw_path:
        raise ValueError(f"{_EVAL_ATTACHMENT_KEY}.path is required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path = path.resolve()
    fixture_root = _EVAL_ATTACHMENT_FIXTURE_ROOT.resolve()
    try:
        path.relative_to(fixture_root)
    except ValueError as exc:
        raise ValueError(f"{_EVAL_ATTACHMENT_KEY}.path must be under {fixture_root}") from exc
    if not path.is_file():
        raise ValueError(f"{_EVAL_ATTACHMENT_KEY}.path must reference an existing regular file")
    size = path.stat().st_size
    if size > _MAX_EVAL_ATTACHMENT_BYTES:
        raise ValueError(f"{_EVAL_ATTACHMENT_KEY}.path exceeds {_MAX_EVAL_ATTACHMENT_BYTES} bytes")

    kind = str(attachment.get("kind") or "image").strip().lower()
    name = str(attachment.get("name") or path.name).strip()
    content_type = str(
        attachment.get("content_type") or attachment.get("mime_type") or mimetypes.guess_type(name)[0] or ""
    )
    store_attachment(
        session_id=session_id,
        kind=kind,
        name=name,
        content_type=content_type,
        data=path.read_bytes(),
    )


async def bot(runner_args: RunnerArguments) -> None:
    """Route an eval runner session to the selected example bot."""
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    example = _select_example(body)
    _bind_example_context(example)
    runner_args.body = _prepare_body(body, example, runner_args)
    _preload_eval_attachment(body, str(runner_args.body.get("session_id") or ""))
    bot_fn = examples_registry.resolve_bot(example)
    await bot_fn(runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
