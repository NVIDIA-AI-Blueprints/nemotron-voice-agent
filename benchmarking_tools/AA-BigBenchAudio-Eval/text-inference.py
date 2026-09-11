#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-based inference for BigBench samples.

Reads question.txt from each sample and writes response.txt.
Supports parallel requests via --batch_size.
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import aiohttp

# Configure via environment (see README). Example for NVIDIA API:
#   TEXT_INFERENCE_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
#   TEXT_INFERENCE_API_KEY=your_key
#   TEXT_INFERENCE_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
API_URL = (os.environ.get("TEXT_INFERENCE_API_URL") or "").strip().rstrip("/")
if API_URL and not API_URL.endswith("/chat/completions"):
    API_URL = f"{API_URL}/chat/completions"
API_KEY = (os.environ.get("TEXT_INFERENCE_API_KEY") or "").strip()
MODEL = (os.environ.get("TEXT_INFERENCE_MODEL") or "nvidia/llama-3.3-nemotron-super-49b-v1.5").strip()
SYSTEM_PROMPT = (os.environ.get("TEXT_INFERENCE_SYSTEM_PROMPT") or "").strip() or None


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if not v or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int_optional(key: str) -> int | None:
    v = os.environ.get(key)
    if not v or not v.strip():
        return None
    try:
        return int(v)
    except ValueError:
        return None


TEMPERATURE = _env_float("TEXT_INFERENCE_TEMPERATURE", 0.0)
TOP_P = _env_float("TEXT_INFERENCE_TOP_P", 1.0)
MAX_TOKENS = _env_int_optional("TEXT_INFERENCE_MAX_TOKENS")


def list_sample_ids(root: Path, start: int | None, end: int | None) -> list[int]:
    """List numeric sample IDs in root directory, optionally filtered by range."""
    ids: list[int] = []
    for name in os.listdir(root):
        full = root / name
        if full.is_dir():
            try:
                idx = int(name)
            except ValueError:
                continue
            ids.append(idx)
    ids.sort()
    if start is not None and end is not None:
        ids = [i for i in ids if start <= i <= end]
    elif start is not None:
        ids = [i for i in ids if i >= start]
    elif end is not None:
        ids = [i for i in ids if i <= end]
    return ids


def load_text(path: Path) -> str | None:
    """Load text from file, return None if missing or error."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def extract_response_text(resp_json: dict) -> str:
    """Extract the assistant's response text from the API response."""
    # OpenAI/Azure format: { choices: [ { message: { content: "..." } } ] }
    try:
        choices = resp_json.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content:
                return content.strip()
    except Exception:
        pass

    # Fallback for alternative schemas
    for key in ("output_text", "text", "output", "content"):
        val = resp_json.get(key)
        if isinstance(val, str) and val:
            return val.strip()

    return ""


async def post_to_llm_async(
    session: aiohttp.ClientSession,
    messages: list[dict],
) -> dict:
    """Send a chat completion request (async)."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {"model": MODEL, "messages": messages}
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    if TOP_P is not None:
        payload["top_p"] = TOP_P
    if MAX_TOKENS is not None:
        payload["max_tokens"] = MAX_TOKENS
    # payload["chat_template_kwargs"] = {"enable_thinking": False}
    # payload["nvext"] = {"max_thinking_tokens": 500}

    # === ADD THIS LOGGING ===
    # import json
    # print(f"Text Client LLM Request Payload: {json.dumps(payload, indent=2)}")
    # print(f"Text Client Headers (excluding auth): {{'Content-Type': '{headers.get('Content-Type')}'}}")

    if not API_URL:
        raise RuntimeError("Set TEXT_INFERENCE_API_URL. See README.")
    async with session.post(API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        resp.raise_for_status()
        return await resp.json()


async def run_inference_async(
    session: aiohttp.ClientSession,
    question: str,
) -> str:
    """Run inference on a single question and return the response text (async)."""
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})
    resp_json = await post_to_llm_async(session, messages)
    return extract_response_text(resp_json)


def parse_retry_samples(raw_list: list[str] | None) -> list[int] | None:
    """Parse retry_samples list (supports comma-separated tokens)."""
    if not raw_list:
        return None
    ids: list[int] = []
    for token in raw_list:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                print(f"Warning: ignoring non-integer retry sample '{part}'")
    if not ids:
        return None
    return sorted(set(ids))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for text inference."""
    p = argparse.ArgumentParser(description="Run text inference on BigBench samples using Llama-3.3-Nemotron-Super-49B")
    p.add_argument(
        "--input_dir",
        required=True,
        help="Root directory containing per-index subfolders (e.g., bigbench_audio_dataset)",
    )
    p.add_argument(
        "--start",
        type=int,
        help="Start index (inclusive)",
    )
    p.add_argument(
        "--end",
        type=int,
        help="End index (inclusive)",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have response.txt",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of parallel requests (default: 1)",
    )
    p.add_argument(
        "--retry_samples",
        nargs="+",
        help="Specific sample IDs to (re)process; supports comma-separated lists",
    )
    return p.parse_args()


async def process_sample(
    session: aiohttp.ClientSession,
    idx: int,
    root: Path,
    skip_existing: bool,
    semaphore: asyncio.Semaphore,
) -> str:
    """Process a single sample with semaphore-controlled concurrency. Returns 'success', 'skip', or 'error'."""
    async with semaphore:
        sample_dir = root / str(idx)
        question_path = sample_dir / "question.txt"
        response_path = sample_dir / "response.txt"

        # Skip if already processed
        if skip_existing and response_path.exists():
            print(f"[{idx}] Skipping (response.txt exists)")
            return "skip"

        # Load question
        question = load_text(question_path)
        if not question:
            print(f"[{idx}] Skipping (no question.txt)")
            return "skip"

        try:
            start_ts = time.time()
            response = await run_inference_async(session, question)
            latency_ms = int((time.time() - start_ts) * 1000)

            # Write response
            response_path.write_text(response + "\n", encoding="utf-8")
            print(f"[{idx}] OK ({latency_ms}ms)")
            return "success"

        except aiohttp.ClientResponseError as e:
            print(f"[{idx}] HTTP Error: {e.status} {e.message}")
            return "error"
        except TimeoutError:
            print(f"[{idx}] Timeout Error")
            return "error"
        except Exception as e:
            print(f"[{idx}] Error: {e}")
            return "error"


async def main_async() -> None:
    """Load samples, run LLM inference per question.txt, write response.txt."""
    args = parse_args()
    root = Path(args.input_dir).expanduser()

    retry_ids = parse_retry_samples(args.retry_samples)
    if retry_ids is not None:
        sample_ids = retry_ids
    else:
        sample_ids = list_sample_ids(root, args.start, args.end)
    print(f"Found {len(sample_ids)} samples to process with batch_size={args.batch_size}")
    print(f"Using model: {MODEL}")
    print(f"System prompt: {SYSTEM_PROMPT}, Temperature: {TEMPERATURE}, Top P: {TOP_P}")

    semaphore = asyncio.Semaphore(max(1, args.batch_size))

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(process_sample(session, idx, root, args.skip_existing, semaphore)) for idx in sample_ids
        ]
        results = await asyncio.gather(*tasks) if tasks else []

    success_count = sum(1 for r in results if r == "success")
    skip_count = sum(1 for r in results if r == "skip")
    error_count = sum(1 for r in results if r == "error")
    print(f"\nDone! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")


def main() -> None:
    """Entry point: run main_async()."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
