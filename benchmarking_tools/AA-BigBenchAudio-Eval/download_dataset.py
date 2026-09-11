#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download Big Bench Audio from Hugging Face. See README for prerequisites and usage."""

import json
import os
import shutil

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def download_dataset(
    prepared_root: str,
    split: str = "train",
    token: str | None = None,
) -> None:
    """Download Big Bench Audio into prepared_root. Each sample: <id>/input.mp3, <id>/meta.json."""
    os.makedirs(prepared_root, exist_ok=True)
    # token: use HF_TOKEN env or --token for rate limits / gated repos
    ds = load_dataset("ArtificialAnalysis/big_bench_audio", split=split, token=token)

    count = 0
    for example in ds:
        sample_id = int(example.get("id", count))
        sample_dir = os.path.join(prepared_root, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)

        file_name = example.get("file_name") or f"data/question_{sample_id}.mp3"
        meta = {
            "id": sample_id,
            "category": example.get("category"),
            "official_answer": example.get("official_answer"),
            "file_name": file_name,
        }
        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        try:
            hub_path = hf_hub_download(
                repo_id="ArtificialAnalysis/big_bench_audio",
                filename=file_name,
                repo_type="dataset",
                token=token,
            )
            if os.path.exists(hub_path):
                shutil.copyfile(hub_path, os.path.join(sample_dir, "input.mp3"))
        except Exception as e:
            print(f"Warning: failed to fetch MP3 for id={sample_id}: {e}")

        count += 1

    print(f"Downloaded {count} samples (split={split}) into {prepared_root}")


def main() -> None:
    """Parse CLI and download Big Bench Audio dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Big Bench Audio dataset. Creates per-sample dirs with input.mp3 and meta.json."
    )
    parser.add_argument("--input_dir", required=True, help="Output directory for the dataset")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (optional). Or set HF_TOKEN. Needed for gated repos or rate limits.",
    )
    args = parser.parse_args()
    download_dataset(args.input_dir, split=args.split, token=args.token)


if __name__ == "__main__":
    main()
