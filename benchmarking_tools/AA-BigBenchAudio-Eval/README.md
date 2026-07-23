# BigBench Audio Eval

Tools for **[Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio)**: download data, speech or text inference, ASR transcription, LLM-judge accuracy.

## Setup

- **Python 3.12+** recommended, **ffmpeg** (MP3→WAV in `speech-inference.py`).

This tool reuses the repo's root environment — no separate venv required.
Dependencies live in the `benchmark` group of the root `pyproject.toml`.
From the **repository root**:

```bash
uv sync --group benchmark
```

Run scripts with `uv run python …` from this directory (deps come from the `benchmark` group).

## Dataset Layout

Per sample ID folder: `input.mp3` / `input.wav`, `meta.json`, optional `question.txt`, `output.wav`, `response.txt`, `result.txt`.

## Download

```bash
uv run python download_dataset.py --input_dir ./datasets/bigbench_audio --split train
```

Use `HF_TOKEN` or `huggingface-cli login` if needed.

## Experiment 1: Speech pipeline

Complete [Download](#download) first so `./datasets/bigbench_audio` (or your chosen `--input_dir`) exists.

Configure the voice agent ([`.env`](../../.env.example) and the selected example's `services.cloud.yaml` / `services.local.yaml`), then start the API from the **nemotron-voice-agent repo root**:

```bash
PIPELINE_TLS=false uv run python src/server.py
```

Listens on `http://localhost:7860`. Set `PIPELINE_TLS=true` or unset it to use HTTPS on the same port.

**Preprocess** MP3 → 16 kHz mono WAV:

```bash
uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --preprocess
```

**Inference** — `POST /api/session-config` + WebSocket `/api/ws` per sample (same protocol as the Nemotron WebSocket client). Use **`http://` or `https://`** in `--server-url`. Port defaults to **7860** if omitted.

```bash
uv run python speech-inference.py \
  --input_dir ./datasets/bigbench_audio \
  --inference \
  --server-url http://127.0.0.1:7860
```

HTTPS uses normal certificate verification by default. For local self-signed certs, add `--insecure-skip-verify`:

```bash
uv run python speech-inference.py \
  --input_dir ./datasets/bigbench_audio \
  --inference \
  --server-url https://127.0.0.1:7860 \
  --insecure-skip-verify
```

**Transcribe** with ASR (separate from the voice agent: gRPC to Riva, default **`localhost:50051`**). Override with `transcribe.py --host` / `--port`. See [Parakeet deploy](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/deploy) for a local ASR NIM.

```bash
uv run python transcribe.py --input_dir ./datasets/bigbench_audio
```

**Evaluate** with `EVAL_API_URL` and `EVAL_API_KEY`:

```bash
EVAL_API_URL=https://.../invoke EVAL_API_KEY=your_key uv run python eval.py --input_dir ./datasets/bigbench_audio
uv run python find_invalid_results.py --input_dir ./datasets/bigbench_audio
uv run python analyze_results.py --input_dir ./datasets/bigbench_audio
```

## Experiment 2: Text-only pipeline

Transcribe inputs (`uv run python transcribe.py …`), then:

```bash
uv run python text-inference.py --input_dir ./datasets/bigbench_audio
```

Then run `eval.py` / `find_invalid_results.py` / `analyze_results.py` as above.

## Reference results

| Model / API | Reasoning | Text-only (%) | In voice pipeline (%) |
|-------------|-----------|---------------|------------------------|
| Llama Nemotron Super 49B v1.5 | ON | 91.90 | 81.30 |
| Llama Nemotron Super 49B v1.5 | OFF | 82.70 | 60.30 |
| Nemotron 3 Nano 30B | ON | 78.76 | 75.60 |
| Nemotron 3 Nano 30B | OFF | 56.50 | 50.40 |
