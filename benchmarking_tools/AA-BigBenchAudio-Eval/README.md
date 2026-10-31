# BigBench Audio Eval

Tools for **[Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio)**: download data, speech or text inference, ASR transcription, LLM-judge accuracy.

To run BBA on the Frontend Delegation Agent, follow the [Frontend Delegation Agent Runbook](RUNBOOK-FDA.md).

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

**Inference** — with the default `--transport ws`, `POST /api/session-config` + WebSocket `/api/ws` per sample (same protocol as the Nemotron WebSocket client). Use **`http://` or `https://`** in `--server-url`. Port defaults to **7860** if omitted. For agents that serve only the OpenAI Realtime API, use `--transport realtime`, as [Realtime API Transport](#realtime-api-transport) describes.

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

**Evaluate** with an LLM judge. `eval.py` reads the judge endpoint from `EVAL_API_URL` and sends `EVAL_API_KEY` as a Bearer token. `--judge_api` selects the request format:

- `bedrock` (default): posts the Anthropic Bedrock request to `EVAL_API_URL` itself. The request names no model, so the URL selects it. Do not pass `--judge_model`.
- `openai`: posts an OpenAI chat completion to `<EVAL_API_URL>/chat/completions` for the model in `--judge_model`, which is required.

Both formats send the same prompt, temperature 0.0, `top_p` 1.0, and at most 8 tokens, and both parse the reply as `CORRECT` or `INCORRECT`. The following example uses a Bedrock invoke URL:

```bash
EVAL_API_URL=https://.../invoke EVAL_API_KEY=<judge-api-key> uv run python eval.py --input_dir ./datasets/bigbench_audio
uv run python find_invalid_results.py --input_dir ./datasets/bigbench_audio
uv run python analyze_results.py --input_dir ./datasets/bigbench_audio
```

The following example uses Claude Sonnet 5 on the NVIDIA Inference Hub with an Inference Hub key:

```bash
EVAL_API_URL=https://inference-api.nvidia.com/v1 EVAL_API_KEY="$NVIDIA_API_KEY" \
  uv run python eval.py --input_dir ./datasets/bigbench_audio \
  --judge_api openai --judge_model aws/anthropic/bedrock-claude-sonnet-5
```

The retry command that `find_invalid_results.py` prints does not include `--judge_api` or `--judge_model`. Add the options and the environment variables that you used for the first `eval.py` run.

## Experiment 2: Text-only pipeline

Transcribe inputs (`uv run python transcribe.py …`), then:

```bash
uv run python text-inference.py --input_dir ./datasets/bigbench_audio
```

Then run `eval.py` / `find_invalid_results.py` / `analyze_results.py` as above.

## Realtime API Transport

With `--transport realtime`, `speech-inference.py` streams each sample over an OpenAI Realtime-compatible WebSocket in place of `/api/ws`. Use it for agents that serve only the Realtime API, such as the Frontend Delegation Agent in eval mode. For the full procedure on that agent, follow the [Frontend Delegation Agent Runbook](RUNBOOK-FDA.md).

The transport changes only how audio reaches the server and comes back. The client's pacing, silence tail, timeouts, stop rules, and `output.wav` assembly are the same for both transports, so the scores stay comparable.

The following options select and configure the transport:

| Option | Default | Description |
| --- | --- | --- |
| `--transport` | `ws` | `ws` uses `POST /api/session-config` and `/api/ws`. `realtime` uses `/v1/realtime`. |
| `--model` | `nvidia/nemotron-realtime-frontend-delegation` | The Realtime model in the connection URL. It is valid only with `--transport realtime`. With `--transport ws`, it is an argument error. |
| `--server-url` | None (required with `--inference`) | The same `http://` or `https://` base URL as for `ws`. The client connects to `ws://<host>:<port>/v1/realtime?model=<model>`, or `wss://` for an `https://` URL. `--insecure-skip-verify` works as for `ws`. |

For each sample, the client opens a new connection and completes the following handshake before it starts streaming:

1. It sends `Authorization: Bearer <key>` when `REALTIME_API_KEY` is set. The client reads the key from the environment, and otherwise from the repository-root `.env`. An empty or whitespace-only value counts as unset, and then the client sends no header. The client never prints the key.
2. It waits for `session.created`.
3. It sends one `session.update` that selects 16 kHz PCM audio for input and output. The update sets no instructions and no tools.
4. It waits for `session.updated`.

An internal 30-second limit bounds the whole handshake. The limit has no command-line option, and the handshake finishes before the client's scoring clock starts.

After the handshake, the client sends each audio chunk as an `input_audio_buffer.append` event and records the `response.output_audio.delta` audio in `output.wav`. It never sends `input_audio_buffer.commit` or `response.create`. The server's voice activity detection (VAD) ends the turn.

A sample fails when the handshake fails, for example in the following cases:

- The server rejects the credential with HTTP 403 during the WebSocket opening handshake.
- An `error` event arrives before `session.updated`, such as `server_busy` with close code 1013 when the server has no free session.
- The connection closes before `session.updated`.
- The 30-second limit passes.

The client logs a failed sample as `unsuccessful idx: <id> — <reason>` and does not retry it automatically. Rerun failed samples with `--retry_samples`.

The following command runs inference over the Realtime API:

```bash
uv run python speech-inference.py \
  --input_dir ./datasets/bigbench_audio \
  --inference \
  --transport realtime \
  --server-url http://127.0.0.1:7860
```

The Frontend Delegation Agent serves at most 4 Realtime sessions and frees a session when its client disconnects. Use `--batch_size 4` for that agent, as [RUNBOOK-FDA.md](RUNBOOK-FDA.md) describes. The default is 1.

## Reference results

> **Note:** The Nemotron 3 Nano rows are historical benchmark results and do not represent the current default model.

| Model / API | Reasoning | Text-only (%) | In voice pipeline (%) |
|-------------|-----------|---------------|------------------------|
| Llama Nemotron Super 49B v1.5 | ON | 91.90 | 81.30 |
| Llama Nemotron Super 49B v1.5 | OFF | 82.70 | 60.30 |
| Nemotron 3 Nano 30B | ON | 78.76 | 75.60 |
| Nemotron 3 Nano 30B | OFF | 56.50 | 50.40 |
