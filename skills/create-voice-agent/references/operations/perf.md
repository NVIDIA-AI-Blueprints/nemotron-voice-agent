# Perf | scaling-perf only

Trigger: perf gate|post-scaffold perf|latency/scaling/concurrency/benchmark. Suite: scripts/scaling-perf/ E2E latency concurrency RTVI. Local workstation DGX Jetson. Cloud if /api/ws reachable.

Gate: scaling/latency/concurrency/E2E/perf/benchmarks→scaling-perf after healthy | none/later/skip→skip | cancel→skip
TPL: Benchmarks after agent up: None/Later/Scaling. Vague perf→scaling only suite.
After pick: copy scripts/scaling-perf→perf/scaling-perf; `chmod +x perf/scaling-perf/simulate_concurrency.sh`; uv sync --group benchmark agent+NIMs healthy run perf/results/summary.md. FORBID perf before healthy.

Deploy: copy scripts add pyproject `[dependency-groups] benchmark = [websockets,numpy,resampy,soundfile,aiohttp,requests]` uv sync --group benchmark. Add 16 kHz mono WAV files to dataset/ (see scaling-perf/README.md).

Agent prereq scaling drives wss host/api/ws RTVI nemotron-voice-agent protocol.

| | workstation/DGX | Jetson |
| --- | --- | --- |
| ASR | :50152 | :50052 |
| TTS | :50151 | :50051 |
| LLM | :8002 or :18000 | :18000 |
| agent | :7860 /api/ws | same |

Pipecat WebRTC bot.py -t webrtc NO /api/ws. Options: nemotron-voice-agent src/server.py local NIM+perf_prompts.yaml OR scaffold WS+RTVI agent.

Commands: baseline `./simulate_concurrency.sh --clients 1 --test-duration 150` | sweep `--clients "1 2 4 8 16" --test-duration 150 --cooldown 10` Jetson start "1 2 4" knee p95>4000ms or errors. TCO optional assets/tco.placeholder.csv.

summary.md: environment|protocol WS path|baseline E2E p50/p95|scaling table|safe concurrency|RTVI if asked|TCO|artifact paths

Platform: DGX ASR RestartCount=0 between levels cooldown single GPU | Jetson HF_TOKEN smaller concurrency no coturn localhost | cloud longer duration region note.

Quick: smoke `uv run python3 benchmark.py` | full flags scripts/scaling-perf/README.md
