# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcribe WAVs with Riva ASR to question.txt/response.txt. See README for prerequisites."""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import grpc
import riva.client

# Throttle and timeout (edit here)
SLEEP_SECONDS = 1.0
TIMEOUT_SECONDS = 10.0


def list_sample_ids(root: Path, ids: list[int] | None, start: int | None, end: int | None) -> list[int]:
    """Return sorted numeric subdir names in root, optionally filtered by ids or start/end range."""
    if ids:
        return [int(i) for i in ids]
    candidates: list[int] = []
    for name in os.listdir(root):
        full = root / name
        if full.is_dir():
            try:
                candidates.append(int(name))
            except ValueError:
                continue
    candidates.sort()
    if start is not None and end is not None:
        candidates = [i for i in candidates if start <= i <= end]
    elif start is not None:
        candidates = [i for i in candidates if i >= start]
    elif end is not None:
        candidates = [i for i in candidates if i <= end]
    return candidates


def build_asr_service(host: str, port: int) -> riva.client.ASRService:
    """Create Riva ASR service client for host:port."""
    server = f"{host}:{port}"
    auth = riva.client.Auth(None, False, server, None)
    return riva.client.ASRService(auth)


def build_config(model_name: str | None) -> riva.client.RecognitionConfig:
    """Build Riva recognition config for en-US offline ASR."""
    return riva.client.RecognitionConfig(
        language_code="en-US",
        enable_automatic_punctuation=True,
        verbatim_transcripts=True,
        max_alternatives=1,
        profanity_filter=False,
        model=model_name,
    )


def transcribe_file(
    asr: riva.client.ASRService, cfg: riva.client.RecognitionConfig, wav_path: Path, timeout_seconds: float
) -> str | None:
    """Run offline ASR on wav_path; return transcript or None on timeout/error."""
    result: list[str | None] = [None]
    error_msg: list[str | None] = [None]

    def worker() -> None:
        try:
            with wav_path.open("rb") as fh:
                data = fh.read()
            response = asr.offline_recognize(data, cfg)
            final_parts: list[str] = []
            for result_item in getattr(response, "results", []):
                alternatives = getattr(result_item, "alternatives", None)
                if alternatives:
                    text = getattr(alternatives[0], "transcript", "")
                    if text:
                        final_parts.append(text)
            result[0] = " ".join(final_parts).strip()
        except grpc.RpcError as e:
            try:
                details = e.details()
            except Exception:
                details = str(e)
            error_msg[0] = f"ASR error for {wav_path}: {details}"
        except Exception as e:
            error_msg[0] = f"Failed to transcribe {wav_path}: {e}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(f"Transcription timed out after {timeout_seconds}s for {wav_path}")
        return None

    if error_msg[0]:
        print(error_msg[0])
        return None

    return result[0]


def process_directory(
    root: Path,
    asr: riva.client.ASRService,
    cfg: riva.client.RecognitionConfig,
    start: int | None,
    end: int | None,
    ids: list[int] | None = None,
    skip_existing: bool = True,
    sleep_seconds: float = 0.0,
    timeout_seconds: float = 10.0,
) -> None:
    """Transcribe input.wav→question.txt and output.wav→response.txt for each sample under root."""
    sample_ids = list_sample_ids(root, ids, start, end)
    for sample_id in sample_ids:
        sample_dir = root / str(sample_id)
        input_wav = sample_dir / "input.wav"
        output_wav = sample_dir / "output.wav"
        question_txt = sample_dir / "question.txt"
        response_txt = sample_dir / "response.txt"
        did_inference = False
        did_skip = False

        # input.wav -> question.txt
        if input_wav.exists():
            if skip_existing and question_txt.exists():
                did_skip = True
            else:
                text = transcribe_file(asr, cfg, input_wav, timeout_seconds)
                did_inference = True
                if text is not None:
                    question_txt.write_text(text.strip() + "\n", encoding="utf-8")

        # output.wav -> response.txt
        if output_wav.exists():
            if skip_existing and response_txt.exists():
                did_skip = True
            else:
                text = transcribe_file(asr, cfg, output_wav, timeout_seconds)
                did_inference = True
                if text is not None:
                    response_txt.write_text(text.strip() + "\n", encoding="utf-8")

        if did_inference:
            print(f"successfully processed idx: {sample_id}")
        elif did_skip:
            print(f"skipped idx: {sample_id} (transcripts already exist)")
        if did_inference and sleep_seconds > 0:
            time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transcribe."""
    p = argparse.ArgumentParser(description="Batch transcribe BigBench audio with NVIDIA Riva ASR (offline)")
    p.add_argument("--input_dir", required=True, help="Root directory containing per-index subfolders")
    p.add_argument("--host", default="localhost", help="Riva ASR server host")
    p.add_argument("--port", type=int, default=50051, help="Riva server port")
    p.add_argument("--model", default="parakeet-1.1b-en-US-asr-offline", help="ASR model name")
    p.add_argument("--start", type=int, help="Start index (inclusive)")
    p.add_argument("--end", type=int, help="End index (inclusive)")
    p.add_argument(
        "--retry_samples", type=str, help="Comma-separated list of specific sample IDs to process (overrides start/end)"
    )
    return p.parse_args()


def main() -> None:
    """Run batch transcription on sample dirs under input_dir."""
    args = parse_args()
    root = Path(args.input_dir).expanduser()
    ids_list = None
    if args.retry_samples:
        ids_list = []
        for token in args.retry_samples.split(","):
            s = token.strip()
            if not s:
                continue
            try:
                ids_list.append(int(s))
            except ValueError:
                print(
                    f"Error: --retry_samples must be comma-separated integers; "
                    f"invalid token {s!r} in {args.retry_samples!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
    asr = build_asr_service(args.host, args.port)
    cfg = build_config(args.model)
    process_directory(
        root,
        asr,
        cfg,
        args.start,
        args.end,
        ids=ids_list,
        skip_existing=True,
        sleep_seconds=SLEEP_SECONDS,
        timeout_seconds=TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    main()
