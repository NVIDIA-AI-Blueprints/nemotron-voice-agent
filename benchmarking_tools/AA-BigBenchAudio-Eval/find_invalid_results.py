#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find all result.txt files with invalid tags in a given input directory."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Find invalid result.txt labels")
    p.add_argument(
        "--input_dir",
        required=True,
        help="Directory to scan (e.g., bigbench_audio_dataset)",
    )
    return p.parse_args()


def main() -> None:
    """Scan input_dir for result.txt with labels other than CORRECT/INCORRECT and print retry command."""
    args = parse_args()
    root = Path(args.input_dir).expanduser()
    allowed = {"CORRECT", "INCORRECT"}

    print(f"Searching for invalid result.txt files under: {root}\n")

    invalid = []
    for res_file in root.rglob("result.txt"):
        # Expect structure: <root>/<sample>/result.txt
        parts = res_file.relative_to(root).parts
        if len(parts) < 2:
            continue
        sample = parts[0]
        if not sample.isdigit():
            continue

        content = res_file.read_text().strip()
        if content not in allowed:
            invalid.append((int(sample), content[:30] if content else "<empty>"))

    if not invalid:
        print("No invalid result.txt files found.\nDone!")
        return

    invalid.sort(key=lambda x: x[0])
    print(f"Found {len(invalid)} invalid result.txt files:")
    for idx, content in invalid:
        print(f"  {idx}: [{content}]")
    ids = ",".join(str(x[0]) for x in invalid)
    print(f"\nRetry command: python eval.py --input_dir {root} --retry_samples {ids}\n")
    print("Done!")


if __name__ == "__main__":
    main()
