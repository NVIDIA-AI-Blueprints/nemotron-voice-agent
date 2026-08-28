# Output Contract

Read after approval, container readiness, exact local profile pinning, and any required
row reconfirmation. The generated project must be runnable and repeatable without
re-reading this skill.

Write in two passes, because some values only exist once the services answer. First write
everything needed to start them: dependencies, secret placeholders, Compose, and cache
paths. Then start the services, resolve the runtime model id and TTS voice, and finalize
the agent file, `scripts/smoke.sh`, and the README with those real values. No placeholder
model, voice, endpoint, or command may survive the second pass.

## Always Write

| File | Purpose |
| --- | --- |
| `bot.py` or `agent.py` | selected framework pipeline |
| `pyproject.toml` | dependencies resolved from current framework docs |
| `.env.example` | secret placeholders only |
| `.gitignore` | excludes `.env`, credential variants, caches, and generated model state while preserving `.env.example` |
| `scripts/smoke.sh` | proves the stack before any client connects |
| `README.md` | setup, start, verify, stop, logs, and troubleshooting |

Also write:

- `compose.yaml` when any model service is self-hosted or the project owns coturn
- `speech_glossary.json` when speech customization is approved
- `signaling_server.py` when custom TURN is required for Pipecat WebRTC
- `nvidia_omni_multimodal_service.py` and `audio_only_smart_turn_strategy.py` from the
  current upstream blueprint when the pipeline is Omni

## Generated Compose

The skill ships no static Compose file. Generate `compose.yaml` from current authoritative
deployment instructions:

| Platform | Source |
| --- | --- |
| Workstation / DGX Spark NIM | each locked model's build.nvidia.com self-hosted page |
| Workstation / DGX Omni | locked model's current build.nvidia.com or Hugging Face run docs |
| Jetson Thor LLM | locked Hugging Face model card and the current [vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/) |
| Jetson Thor Speech | current Riva ARM64 Quick Start |
| Project-owned coturn | current coturn deployment documentation |

Translate each current launch command into one Compose service. Preserve its image,
command, environment, mounts, shared-memory or IPC settings, ports, and GPU requirements
except for the approved profile/tags, host ports, cache paths, and GPU placement this skill
explicitly locks. Use current health checks from the same source.

For a shared single-GPU NIM layout, Compose must pin the exact LLM profile, approved
context length, exactly one documented runtime memory-control path, and explicit TTS batch
profile. These are required entries, not optional tuning. Do not rely on auto-selected LLM
precision, default vLLM memory utilization, or the TTS container default. The memory-control
value comes from the budget in `preflight.md` §Deployment fit, so do not reuse a number from
another build. The same requirement applies to a shared-GPU raw vLLM layout such as Jetson
Thor, where `--gpu-memory-utilization` and `--max-model-len` are the equivalent required
entries. If free VRAM measured at startup differs from the budget, update Compose before
starting rather than proceeding with a stale value. Document `platforms/deployment.md`
§Shared-GPU memory gate in the README.

Preserve the documented model cache or export mount for every speech service. First boot
downloads models and can build engines, and that work must survive container replacement.

| Platform / pipeline | Service names |
| --- | --- |
| Workstation / DGX Spark NIM cascaded | `llm`, `tts`, `asr` |
| Workstation / DGX Omni | `omni`, `tts` |
| Jetson Thor cascaded | `llm`, `riva` (ASR + TTS enabled) |
| Jetson Thor Omni | `omni`, `riva` (TTS enabled, ASR disabled) |

For a hybrid deployment, generate only the services assigned locally. Cloud slots remain
agent configuration and do not get placeholder Compose services.

For a multi-GPU workstation, follow `platforms/deployment.md` §Scaling across GPUs. DGX
Spark and Jetson Thor use their platform guides. In Compose, use `device_ids` or `count`,
never both. A multi-GPU LLM service must expose the exact topology required by its
compatible profile.

Compose may read secret values such as `NVIDIA_API_KEY` and `HF_TOKEN` from the user
environment or `.env`. Write only placeholders to `.env.example`; the user supplies real
values after file generation is approved. Keep model ids, endpoints, profiles, tags, ports, and GPU placement
in `compose.yaml` or agent code, not `.env`.

For Jetson Thor, mount the external Riva `model_repository` produced by the one-time
Quick Start initialization. Do not rebuild it during normal `docker compose up`.

Validate before handover:

```bash
docker compose config --quiet
```

Then start one service at a time and wait for readiness. Do not rely on Compose startup
order alone.

## README

Write the locked choices and exact commands for this project:

1. framework, pipeline, platform, transport, language route, per-slot local/cloud
   placement, container image, pinned profile, runtime model id, and discovered TTS voice
2. prerequisites and required credentials
3. one-time setup, including Riva initialization when applicable
4. dependency installation
5. service startup in the required order
6. health and model-id checks, then `scripts/smoke.sh`
7. agent start command and client connection path
8. spoken-exchange acceptance test
9. log commands, stop commands, and links to troubleshooting

Item 3 must warn that a speech service's first boot downloads models and can build an
engine, that this commonly takes far longer than later starts, which log line shows
progress, and which cache path keeps the result.

Do not leave placeholders such as “start the services” or “use the deploy page.” Resolve
those instructions while building and persist the resulting commands in the README.

## Smoke Before Client

The check list below is fixed by this contract. Plan it in the first pass and do not reduce
it later. The second pass only fills in resolved values: take the LLM model string from that
placement's `/v1/models` and the voice from the TTS voice query in `models/tts.md`, then
write both into `bot.py` or `agent.py` in place of anything carried over from the proposal
table.

Generate `scripts/smoke.sh` from the same resolved endpoints. Include only the checks that
apply to the approved pipeline and deployment: cloud slots probe their configured cloud
endpoints instead of local service health. It must exit non-zero on the first failure so the
user never debugs this from a browser:

1. `/v1/models` returns the exact served id the agent uses
2. Cascaded: one **streaming** chat completion with reasoning off, asserting non-empty
   `delta.content` and empty or absent `delta.reasoning_content`. A non-streaming answer
   still passes while a reasoning parser routes every token away from TTS
   (`models/llm.md` §Reasoning parser)
3. Omni: endpoint readiness and one minimal audio request, and no LLM or ASR checks
4. TTS voice query, then one sentence synthesized in the approved locale using a voice that
   query actually returned
5. Cascaded: ASR configuration from the running service, asserting the loaded model and
   language rather than HTTP readiness alone
6. Cascaded: feed the synthesized sentence back through streaming ASR, which is the only
   check that proves both speech directions before a microphone is involved
7. in-process construction of the same services the agent builds, using the same classes
   and settings, which is what catches a missing API-key argument or a misplaced reasoning
   payload

Use the agent's own settings, including per-slot cloud endpoints on a hybrid layout. Print
no secrets. Run it after the services are healthy and before the runner starts. A passing
run is still not a working agent, so the spoken exchange in `operations/run.md` remains the
bar.

## Verify the Generated Project

Re-read this section after generation.

- No real secret is written to disk.
- `compose.yaml`, when generated, parses and exposes only the intended GPUs.
- Every local service reaches readiness.
- `scripts/smoke.sh` passes.
- The documented agent command starts successfully.
- A new terminal can follow the README to complete a spoken exchange.
