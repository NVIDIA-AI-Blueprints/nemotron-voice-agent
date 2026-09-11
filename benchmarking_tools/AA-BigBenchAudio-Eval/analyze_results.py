#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze BigBench result.txt files and report CORRECT/INCORRECT accuracy."""

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for analyze_results."""
    p = argparse.ArgumentParser(description="Analyze BigBench result.txt files and report accuracy.")
    p.add_argument("--input_dir", required=True, help="Directory containing per-sample subfolders with result.txt")
    return p.parse_args()


def main() -> None:
    """Run accuracy analysis on result.txt files under input_dir."""
    args = parse_args()
    root = Path(args.input_dir).expanduser()

    correct = 0
    incorrect = 0
    other = 0
    total = 0
    other_ids = []

    # Only consider numeric subdirectories (0, 1, 2, ...).
    for name in os.listdir(root):
        sample_dir = root / name
        if not sample_dir.is_dir():
            continue
        try:
            int(name)
        except ValueError:
            continue

        result_path = sample_dir / "result.txt"
        if not result_path.exists():
            continue

        try:
            label = result_path.read_text(encoding="utf-8").strip()
        except Exception:
            label = ""

        total += 1
        if label == "CORRECT":
            correct += 1
        elif label == "INCORRECT":
            incorrect += 1
        else:
            other += 1
            other_ids.append(int(name))

    accuracy = (correct / total * 100.0) if total else 0.0

    print(f"input_dir: {root}")
    print(f"total result.txt: {total}")
    print(f"CORRECT: {correct}")
    print(f"INCORRECT: {incorrect}")
    print(f"OTHER: {other}")
    print(f"accuracy: {accuracy:.2f}%")

    if other_ids:
        other_ids.sort()
        print("other_ids:")
        for i in other_ids:
            print(i)


if __name__ == "__main__":
    main()
