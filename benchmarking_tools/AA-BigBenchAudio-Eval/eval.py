# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge evaluation for Big Bench Audio (CORRECT/INCORRECT). See README for judge model and configuration."""

import argparse
import json
import os
import time
from pathlib import Path

import requests

# LLM judge endpoint (set EVAL_API_URL and EVAL_API_KEY; see README).
# Example: NVIDIA API for Claude — EVAL_API_URL=https://prod.api.nvidia.com/llm/v1/.../invoke
API_URL = os.environ.get("EVAL_API_URL", "").strip()
API_KEY = (os.environ.get("EVAL_API_KEY", "") or "").strip()


def list_sample_ids(root: Path, start: int | None, end: int | None, samples: list[int] | None = None) -> list[int]:
    """Return sorted numeric subdir names in root, optionally filtered by start/end or samples list."""
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
    # Filter by specific samples list if provided
    if samples is not None:
        ids = [i for i in ids if i in samples]
    # Otherwise filter by start/end range if specified
    elif start is not None and end is not None:
        ids = [i for i in ids if start <= i <= end]
    return ids


def load_text(path: Path) -> str | None:
    """Read and return trimmed file content, or None on missing/error."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def load_meta(path: Path) -> dict:
    """Load JSON from path; return {} on missing or parse error."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def build_eval_prompt(question: str, official_answer: str, candidate_answer: str) -> str:
    """Build the judge prompt. Caller must ensure question, official_answer, candidate_answer are non-empty."""
    return (
        "Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.\n"
        "For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.\n"
        "If the CANDIDATE ANSWER contradicts itself, assess the first proposed answer.\n"
        "If the CANDIDATE ANSWER provides a final answer and working, assess the final answer only.\n"
        "If the CANDIDATE ANSWER includes irrelevant information, assess only the relevant information.\n"
        "If the CANDIDATE ANSWER includes a numeric value it is ok if it is spelled e.g. 7 or seven\n"
        "It is ok if the CANDIDATE ANSWER involves a misspelling of a person's name "
        "e.g. Leda or Lida, Autry or Audrie.\n"
        "  \n"
        f"The question, for reference only: START QUESTION {question} \n\nEND QUESTION\n"
        "\n"
        f"The OFFICIAL ANSWER:{official_answer}\n"
        "\n"
        "BEGIN CANDIDATE ANSWER TO ASSESS\n"
        "\n"
        f"{candidate_answer}\n"
        "\n"
        "END CANDIDATE ANSWER TO ASSESS\n"
        "\n"
        "Reply only with CORRECT or INCORRECT."
    )


def extract_label(text: str) -> str:
    """Parse LLM response to CORRECT or INCORRECT; default INCORRECT if ambiguous."""
    cleaned = (text or "").strip().upper()
    if "CORRECT" in cleaned and "INCORRECT" in cleaned:
        # Choose the leading token
        first = cleaned.split()[0]
        return first if first in {"CORRECT", "INCORRECT"} else "INCORRECT"
    if "CORRECT" in cleaned:
        return "CORRECT"
    if "INCORRECT" in cleaned:
        return "INCORRECT"
    return cleaned.split()[0] if cleaned else "INCORRECT"


def _invoke_llm_judge(
    messages: list[dict],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 8,
) -> dict:
    """Call the configured LLM judge API (EVAL_API_URL). Uses Anthropic/Bedrock-style payload.

    top_p: Nucleus sampling — only consider tokens in the top p probability mass (0–1).
    Use 1.0 for deterministic judge output (CORRECT/INCORRECT); lower values add sampling.
    """
    if not API_URL or not API_KEY:
        raise RuntimeError(
            "Set EVAL_API_URL and EVAL_API_KEY (see README). "
            "Example: EVAL_API_URL=https://.../invoke EVAL_API_KEY=your_key python eval.py ..."
        )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


def _extract_text_from_response(resp_json: dict) -> str:
    # Common Anthropic-like Bedrock schema: { content: [ {type:"text", text:"..."}, ... ] }
    try:
        content = resp_json.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if isinstance(txt, str):
                        texts.append(txt)
            if texts:
                return "".join(texts)
    except Exception:
        pass

    # Fallbacks for alternative schemas
    for key in ("output_text", "text", "output"):
        val = resp_json.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


MISSING_INFO_LABEL = "INCORRECT - missing information"


def _validate_eval_inputs(question: str | None, official: str | None, candidate: str | None) -> None:
    """Raise ValueError if any required field is missing (None or blank)."""
    if not (question or "").strip():
        raise ValueError("question is missing or empty")
    if not (official or "").strip():
        raise ValueError("official_answer is missing or empty")
    if not (candidate or "").strip():
        raise ValueError("candidate_answer is missing or empty")


def evaluate_sample(_unused_client: object, question: str | None, official: str | None, candidate: str | None) -> str:
    """Run LLM judge on one sample; return CORRECT, INCORRECT, or MISSING_INFO_LABEL if inputs missing."""
    try:
        _validate_eval_inputs(question, official, candidate)
    except ValueError:
        return MISSING_INFO_LABEL

    q = (question or "").strip()
    o = (official or "").strip()
    c = (candidate or "").strip()
    prompt = build_eval_prompt(q, o, c)

    start_ts = time.time()
    resp_json = _invoke_llm_judge(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=8,
    )
    _latency_ms = int((time.time() - start_ts) * 1000)
    text = _extract_text_from_response(resp_json)
    label = extract_label(text)
    return label


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for eval."""
    p = argparse.ArgumentParser(description="Evaluate BigBench responses using LLM judge")
    p.add_argument("--input_dir", required=True, help="Root directory containing per-index subfolders")
    p.add_argument("--start", type=int, help="Start index (inclusive)")
    p.add_argument("--end", type=int, help="End index (inclusive)")
    p.add_argument("--retry_samples", type=str, help="Comma-separated list of specific sample IDs to retry")
    p.add_argument(
        "--compare_only",
        action="store_true",
        help="Do not write result.txt. Print only indices where new label differs from existing (or missing).",
    )
    return p.parse_args()


def main() -> None:
    """Run LLM judge evaluation on samples under input_dir and write result.txt per sample."""
    args = parse_args()
    root = Path(args.input_dir).expanduser()

    # Direct HTTP, no SDK client needed
    client = None

    # Parse retry_samples list if provided
    samples_list = None
    if args.retry_samples:
        samples_list = []
        for token in args.retry_samples.split(","):
            s = token.strip()
            if not s:
                continue
            try:
                samples_list.append(int(s))
            except ValueError:
                print(f"warning: invalid retry_samples token (skipping): {s!r}")

    sample_ids = list_sample_ids(root, args.start, args.end, samples=samples_list)
    for idx in sample_ids:
        sample_dir = root / str(idx)
        result_path = sample_dir / "result.txt"

        try:
            question_path = sample_dir / "question.txt"
            response_path = sample_dir / "response.txt"

            if not question_path.exists():
                raise FileNotFoundError(f"question.txt missing for idx={idx} at {question_path}")
            if not response_path.exists():
                raise FileNotFoundError(f"response.txt missing for idx={idx} at {response_path}")

            question = load_text(question_path)
            candidate = load_text(response_path)
            meta = load_meta(sample_dir / "meta.json")
            official = meta.get("official_answer")

            label = evaluate_sample(client, question, official, candidate)
            normalized_label = (label or "").strip()

            if args.compare_only:
                existing = load_text(result_path)
                existing_norm = (existing or "").strip()
                if not existing_norm:
                    print(f"DIFF idx={idx}: existing=<missing> new={normalized_label}")
                elif existing_norm != normalized_label:
                    print(f"DIFF idx={idx}: existing={existing_norm} new={normalized_label}")
                # else: match, print nothing
            else:
                result_path.write_text(normalized_label + "\n", encoding="utf-8")
                print(f"evaluated idx={idx}: {normalized_label}")
        except Exception as e:
            print(f"evaluation failed for idx={idx}: {e}")


if __name__ == "__main__":
    main()
