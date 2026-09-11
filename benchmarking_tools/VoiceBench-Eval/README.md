# VoiceBench ASR+LLM Eval

This runner evaluates the ASR and LLM components directly with the official
[VoiceBench](https://github.com/HLT-Media/VoiceBench) datasets and scoring code.
It is deliberately separate from the Blueprint runtime: the benchmark is
audio → ASR → LLM → text response. TTS, output ASR, WebSocket behavior, and
frontend/backend latency are not measured.

Every run stores the exact target repository SHA, endpoint URLs, ASR model, LLM
model, prompt, repeat count, requested concurrency, per-repeat outputs, and
mean/stddev scores. The target SHA is provenance only: this tool does not
modify or invoke the target repository's pipeline.

## Setup

Clone and install the official evaluator separately. Run the command below with
the VoiceBench environment so its official judge and evaluators are available:

```bash
git clone https://github.com/HLT-Media/VoiceBench.git ../VoiceBench
cd ../VoiceBench
uv sync
```

For local Riva ASR, install this repository's benchmark dependencies as well:

```bash
cd /path/to/nemotron-voice-agent
uv sync --group benchmark
```

## Run

The defaults are three repeats and concurrency six. Set models/endpoints
explicitly so they are both visible in the command and recorded in
`metadata.json`.

```bash
cd /path/to/VoiceBench
NVIDIA_API_KEY=... uv run python \
  /path/to/nemotron-voice-agent/benchmarking_tools/VoiceBench-Eval/run_voicebench.py \
  --voicebench-root /path/to/VoiceBench \
  --target-repo /path/to/nemotron-voice-agent \
  --target-commit <exact-sha> \
  --asr-server localhost:50152 \
  --asr-model cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer \
  --llm-base-url http://10.47.20.203:18000/v1 \
  --llm-model nvidia/nemotron-3.5-lightning-30b-a3b
```

The runner automatically switches from the requested concurrency to one if a
cloud ASR or LLM endpoint fails. It checkpoints JSONL responses after every
sample, so rerunning the same command resumes incomplete dataset outputs.

Use `--dry-run` to validate the target commit and write reproducibility
metadata without contacting endpoints. Use `--datasets advbench` for a focused
run, `--repeats N` to override repeats, `--concurrency N` to override request
parallelism, and `--skip-judge` only when inference outputs are all that is
needed.

The judge-backed VoiceBench subsets require the judge credentials expected by
the official VoiceBench checkout (for example its `.env` configuration).
