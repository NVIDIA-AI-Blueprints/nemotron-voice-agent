#!/usr/bin/env python3
"""Run the official VoiceBench evaluator against direct ASR and LLM endpoints.

This is intentionally a component-level benchmark: VoiceBench audio is sent to
ASR, the resulting transcript is sent to the configured chat-completions LLM,
and the response is scored by VoiceBench.  It does not call the Blueprint's
audio-in/audio-out API, so TTS and output re-transcription are excluded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

STANDARD_DATASETS = ("commoneval", "openbookqa", "ifeval", "advbench", "wildvoice", "bbh", "alpacaeval_full")
MULTI_SPLITS = {
    "mmsu": (
        "law",
        "engineering",
        "other",
        "biology",
        "business",
        "economics",
        "health",
        "philosophy",
        "psychology",
        "history",
        "chemistry",
        "physics",
    ),
    "sd-qa": ("aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf"),
}
ALL_DATASETS = STANDARD_DATASETS + tuple(MULTI_SPLITS)
JUDGE_EVALUATORS = {"commoneval": "open", "wildvoice": "open", "alpacaeval_full": "open", "sd-qa": "qa"}
EVALUATORS = {
    **JUDGE_EVALUATORS,
    "openbookqa": "mcq",
    "mmsu": "mcq",
    "ifeval": "ifeval",
    "advbench": "harm",
    "bbh": "bbh",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user's query has been transcribed from speech "
    "and may contain minor transcription errors."
)


class EndpointFailure(RuntimeError):
    """An endpoint failure that permits the cloud-concurrency fallback."""


def resolve_commit(repo: Path, requested: str) -> str:
    """Resolve and validate the exact target revision recorded for a run."""
    if not (repo / ".git").exists():
        raise SystemExit(f"Target repository is not a Git worktree: {repo}")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", requested):
        raise SystemExit("--target-commit must be an exact 7-40 character Git SHA")
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{requested}^{{commit}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"Commit {requested!r} does not resolve in {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def is_cloud_endpoint(args: argparse.Namespace) -> bool:
    """Return whether either configured endpoint is NVIDIA-hosted."""
    return "nvcf.nvidia.com" in args.asr_server.lower() or "nvidia.com" in args.llm_base_url.lower()


def load_voicebench(voicebench_root: Path):
    """Import official VoiceBench dataset and scoring helpers from its checkout."""
    if not (voicebench_root / "api_judge.py").is_file() or not (voicebench_root / "src" / "evaluator").is_dir():
        raise SystemExit("--voicebench-root must be a VoiceBench checkout containing api_judge.py and src/evaluator")
    sys.path.insert(0, str(voicebench_root))
    from datasets import Audio, load_dataset
    from src.evaluator import evaluator_mapping

    return Audio, evaluator_mapping, load_dataset


def endpoint_metadata(args: argparse.Namespace, resolved_commit: str) -> dict[str, Any]:
    """Build reproducibility metadata before any inference request is sent."""
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_type": "component-level ASR+LLM; excludes Blueprint TTS and output ASR",
        "target_repository": str(args.target_repo.resolve()),
        "requested_commit": args.target_commit,
        "resolved_commit": resolved_commit,
        "asr": {"server": args.asr_server, "model": args.asr_model, "function_id": args.asr_function_id or None},
        "llm": {"base_url": args.llm_base_url, "model": args.llm_model},
        "system_prompt": args.system_prompt,
        "requested_concurrency": args.concurrency,
        "repeats": args.repeats,
        "datasets": args.datasets,
        "cloud_endpoint": is_cloud_endpoint(args),
    }


def make_predictor(args: argparse.Namespace):
    """Create a thread-local direct Riva ASR + OpenAI-compatible LLM caller."""
    import grpc
    import numpy as np
    import riva.client
    from openai import OpenAI

    local = threading.local()
    cloud_asr = "nvcf.nvidia.com" in args.asr_server.lower()
    llm_is_nvidia = "nvidia.com" in args.llm_base_url.lower()

    def client():
        if hasattr(local, "asr"):
            return local.asr, local.llm
        metadata = []
        if cloud_asr:
            if not args.nvidia_api_key or not args.asr_function_id:
                raise EndpointFailure("NVCF ASR requires --nvidia-api-key and --asr-function-id")
            metadata = [["function-id", args.asr_function_id], ["authorization", f"Bearer {args.nvidia_api_key}"]]
        auth = riva.client.Auth(None, use_ssl=cloud_asr, uri=args.asr_server, metadata_args=metadata)
        local.asr = riva.client.ASRService(auth)
        llm_key = args.nvidia_api_key if llm_is_nvidia else args.llm_api_key
        local.llm = OpenAI(base_url=args.llm_base_url, api_key=llm_key or "not-needed")
        return local.asr, local.llm

    def predict(item: dict[str, Any]) -> dict[str, Any]:
        asr, llm = client()
        pcm = (np.clip(item["audio"]["array"], -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=16_000,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                model=args.asr_model,
            ),
            interim_results=False,
        )
        try:
            responses = asr.streaming_response_generator(audio_chunks=iter([pcm]), streaming_config=config)
            transcript = " ".join(
                result.alternatives[0].transcript
                for response in responses
                for result in response.results
                if result.is_final and result.alternatives
            ).strip()
            if not transcript:
                raise EndpointFailure("ASR returned an empty transcript")
            response = llm.chat.completions.create(
                model=args.llm_model,
                messages=[{"role": "system", "content": args.system_prompt}, {"role": "user", "content": transcript}],
                frequency_penalty=0,
                presence_penalty=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.05},
            )
            answer = response.choices[0].message.content if response.choices else None
            if not answer or not answer.strip():
                raise EndpointFailure("LLM returned an empty completion")
        except (grpc.RpcError, TimeoutError) as error:
            raise EndpointFailure(str(error)) from error
        return {key: value for key, value in item.items() if key != "audio"} | {"response": answer.strip()}

    return predict


def output_count(path: Path) -> int:
    """Count completed JSONL records so an interrupted run can resume."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def infer_dataset(
    args: argparse.Namespace, load_dataset: Any, audio_type: Any, dataset: str, output: Path
) -> dict[str, Any]:
    """Run or resume one dataset, lowering cloud concurrency after a failure."""
    splits = ("test",) if dataset in STANDARD_DATASETS else MULTI_SPLITS[dataset]
    parts = [
        load_dataset("hlt-lab/voicebench", dataset, split=split).cast_column("audio", audio_type(sampling_rate=16_000))
        for split in splits
    ]
    total = sum(len(part) for part in parts)
    completed = output_count(output)
    if completed > total:
        raise RuntimeError(f"{output} contains {completed} rows for a {total}-sample dataset")
    items: Iterable[dict[str, Any]] = (item for part in parts for item in part.select(range(len(part))))
    for _ in range(completed):
        next(items)
    predict = make_predictor(args)
    active = args.concurrency
    fallback = False
    with output.open("a") as file:
        while True:
            batch = []
            for _ in range(active):
                try:
                    batch.append(next(items))
                except StopIteration:
                    break
            if not batch:
                break
            with concurrent.futures.ThreadPoolExecutor(max_workers=active) as executor:
                futures = [executor.submit(predict, item) for item in batch]
                for item, future in zip(batch, futures, strict=True):
                    try:
                        record = future.result()
                    except EndpointFailure:
                        if is_cloud_endpoint(args) and active != 1:
                            active, fallback = 1, True
                            record = predict(item)
                        else:
                            raise
                    file.write(json.dumps(record) + "\n")
                    file.flush()
    return {"samples": total, "completed": total, "effective_concurrency": active, "cloud_fallback": fallback}


def judge_and_score(
    voicebench_root: Path, evaluator_mapping: Any, dataset: str, output: Path, workers: int
) -> dict[str, float]:
    """Use VoiceBench's official judge and evaluator for one inference file."""
    source = output
    if dataset in JUDGE_EVALUATORS:
        subprocess.run(
            [sys.executable, "api_judge.py", "--src_file", str(output), "--workers", str(workers)],
            cwd=voicebench_root,
            check=True,
        )
        source = output.with_name("result-" + output.name)
    data = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    return {key: float(value) for key, value in evaluator_mapping[EVALUATORS[dataset]]().evaluate(data).items()}


def aggregate(repeats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Calculate mean and standard deviation for every reported metric."""
    metrics: dict[str, list[float]] = defaultdict(list)
    for report in repeats:
        for dataset, scores in report["scores"].items():
            for name, score in scores.items():
                metrics[f"{dataset}.{name}"].append(score)
    return {
        name: {"mean": mean(values), "stddev": stdev(values) if len(values) > 1 else 0.0}
        for name, values in metrics.items()
    }


def main() -> None:
    """Parse configuration, run requested repeats, and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voicebench-root", type=Path, required=True, help="Checkout of https://github.com/HLT-Media/VoiceBench"
    )
    parser.add_argument(
        "--target-repo", type=Path, required=True, help="Git worktree of the Blueprint revision being recorded"
    )
    parser.add_argument("--target-commit", required=True, help="Exact target Git SHA")
    parser.add_argument("--asr-server", default=os.getenv("ASR_SERVER_URL", "grpc.nvcf.nvidia.com:443"))
    parser.add_argument("--asr-model", default=os.getenv("ASR_MODEL_NAME", "nemotron-asr-streaming"))
    parser.add_argument("--asr-function-id", default=os.getenv("ASR_CLOUD_FUNCTION_ID"))
    parser.add_argument("--llm-base-url", default=os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1"))
    parser.add_argument("--llm-model", default=os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"))
    parser.add_argument("--nvidia-api-key", default=os.getenv("NVIDIA_API_KEY"))
    parser.add_argument("--llm-api-key", default=os.getenv("LOCAL_LLM_API_KEY"))
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--run-name", default=datetime.now().strftime("voicebench-%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrency < 1:
        raise SystemExit("--repeats and --concurrency must be positive")
    resolved_commit = resolve_commit(args.target_repo, args.target_commit)
    metadata = endpoint_metadata(args, resolved_commit)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    audio_type, evaluator_mapping, load_dataset = load_voicebench(args.voicebench_root)
    reports = []
    for number in range(1, args.repeats + 1):
        repeat_dir = run_dir / f"repeat-{number:02d}"
        repeat_dir.mkdir(exist_ok=True)
        report: dict[str, Any] = {"repeat": number, "inference": {}, "scores": {}}
        for dataset in args.datasets:
            output = repeat_dir / f"{dataset}.jsonl"
            report["inference"][dataset] = infer_dataset(args, load_dataset, audio_type, dataset, output)
            if not args.skip_judge:
                report["scores"][dataset] = judge_and_score(
                    args.voicebench_root, evaluator_mapping, dataset, output, args.concurrency
                )
            (repeat_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    summary = {"metadata": metadata, "repeats": reports, "aggregate": aggregate(reports)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
