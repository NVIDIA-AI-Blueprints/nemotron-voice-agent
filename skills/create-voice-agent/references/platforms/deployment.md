# Deployment

Read after the table is approved. Turns the Deployment row into running endpoints the
agent code can call. Model ids stay in code. `.env` is secrets only.

Choose the platform path first:

| Host class | Deployment source |
| --- | --- |
| `workstation` | current build.nvidia.com instructions for every self-hosted slot |
| `dgx_spark` | NIM path in `platforms/dgx-spark.md`, after GB10 support verification |
| `jetson_thor` | vLLM + Riva L4T path in `platforms/jetson-thor.md` (not NIM) |
| `unsupported_edge` | cloud. Local NIM/vLLM path is unsupported |
| `no_gpu` | cloud |

For workstation and DGX Spark NIM slots, take the image, launch command, ports, and
environment from the locked model's self-hosted page, then apply the verified profile or
tags from `models/llm.md`, `models/asr.md`, or `models/tts.md`. Persist the result as the
generated `compose.yaml` defined by `output-contract.md`.

## Source of truth per slot

| Slot | Cloud | Self-hosted deploy copy |
| --- | --- | --- |
| Cascaded LLM | model page + `/v1/models` | `build.nvidia.com/nvidia/<slug>?nim=self-hosted` |
| ASR | model page (function id) | `build.nvidia.com/nvidia/<slug>/deploy` + ASR matrix tags |
| TTS | model page (function id) | `build.nvidia.com/nvidia/<slug>/deploy` + TTS matrix tags |
| Omni | omni model page | follow `frameworks/omni.md` and that model’s build / HF run docs |

Examples: [Lightning self-hosted](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b?nim=self-hosted) ·
[ASR deploy](https://build.nvidia.com/nvidia/nemotron-asr-streaming/deploy) ·
[TTS deploy](https://build.nvidia.com/nvidia/magpie-tts-multilingual/deploy).

Never invent a `docker run` or image tag from memory. Apply the Speech source precedence
in `models/catalog.md`.

## Cloud

No containers. Wire:

- LLM: `https://integrate.api.nvidia.com/v1` + the id `/v1/models` returns for the locked
  model
- ASR / TTS: `grpc.nvcf.nvidia.com:443` + function ids from each model’s build page
- `NVIDIA_API_KEY` in `.env`

## Hybrid

Lock and wire each slot independently:

| Slot placement | `compose.yaml` | Agent configuration |
| --- | --- | --- |
| Local | include only that service | mapped host endpoint. Self-hosted speech function id is empty |
| Cloud | omit that service | cloud endpoint and locked model or function id |

Start and health-check only the local slots, preserving their relative startup order.
Probe cloud authentication and the locked model or speech configuration before starting
the agent. The generated README must label every slot as local or cloud and include only
the local Compose commands.

For LiveKit, configure STT, LLM, and TTS independently through the current plugin APIs.
Do not assume that selecting one cloud slot moves the full pipeline to cloud.

## Workstation / DGX Spark NIM

1. Confirm the host clears `preflight.md` §Deployment fit (or move overflowing slots to
   cloud).
2. Pass `platforms/readiness.md` before pulling images or weights.
3. `docker login nvcr.io` with username `$oauthtoken` and password `NVIDIA_API_KEY`.
4. For each self-hosted slot, open that slug’s build page (table above) and take the
   current image, run command, ports, and env.
5. Translate each command into the matching service in `compose.yaml` through
   `output-contract.md`.
6. Overlay locked selection from the slot file:
   - LLM: pinned Compatible `NIM_MODEL_PROFILE`, max length, exactly one documented
     runtime memory-control path, and reasoning passthrough from `models/llm.md`
   - ASR / TTS: streaming `NIM_TAGS_SELECTOR` / `CONTAINER_ID` from the matrix and an
     explicit TTS batch profile
7. Clear §Shared-GPU memory gate, then start **one Compose service at a time**. Wait
   until that service is ready and the gate passes again before the next. Do not start
   LLM + ASR + TTS in one operation.
8. Cascaded startup order: LLM → TTS → ASR. Omni startup order: Omni → TTS. Skip any
   cloud slot.
9. Record the host ports and base URLs the build page actually published. Do not assume
   18000 / 50151 / 50152 unless that page says so.
10. Resolve the runtime model id and TTS voice, write them and the recorded endpoints into
    the agent, then run `scripts/smoke.sh` (`output-contract.md` §Smoke before client).
11. Start the agent only after every local slot is ready and smoke passes
    (`operations/run.md`).

Jetson Thor skips the numbered path above and its shared-GPU memory gate, because
`platforms/jetson-thor.md` defines the equivalent unified-memory check. §Service health
checks, §First boot takes much longer, and §Per-slot reuse still apply to it.

### Shared-GPU memory gate

Do not treat LLM readiness as permission to start speech. This gate covers workstation and
DGX Spark layouts. Jetson Thor uses the unified-memory check in
`platforms/jetson-thor.md` §Start and verify instead.

Before the first service starts, re-measure free VRAM and confirm the arithmetic in
`preflight.md` §Deployment fit still passes, with the LLM memory control already present in
`compose.yaml`. Then after each service becomes ready:

1. confirm logs show the locked profile, precision, context, and speech tags
2. record per-process GPU use and free VRAM with `nvidia-smi`
3. compare free VRAM with every remaining slot's matrix reserve plus startup headroom
4. start the next service only when that measured gate passes

If the LLM consumes the speech reserve, stop it and reduce its documented runtime memory
budget or context before starting TTS. If TTS leaves too little for ASR, use a smaller
approved TTS batch profile or move a slot. Do not leave an unhealthy service in a restart
loop while bringing up the next slot.

After every assigned service is healthy and the measured gate passes, update the README
deployment status from `provisional co-location` to `self-hosted, co-located`. The agent
is still not complete until `operations/run.md` passes a spoken exchange.

### Service health checks

Use the health path from the build / NIM docs for that service. Common patterns:

| Slot | Often |
| --- | --- |
| LLM | `GET …/v1/health/ready` and `GET …/v1/models` (record the served id) |
| ASR / TTS | HTTP readiness, then the documented configuration or voice query |

HTTP readiness proves the process is up. It does not prove the expected speech profile is
loaded. Query the running service with the current NVIDIA client or deploy-page
procedure. ASR uses the speech-recognition configuration API. TTS uses the synthesis
configuration API or, on Speech NIM, the documented voice-list endpoint in its current TTS
API reference. The current gRPC RPC names are `GetRivaSpeechRecognitionConfig` and
`GetRivaSynthesisConfig`. Confirm both against the running platform's docs, then confirm
the locked model, language, and TTS voice before reuse or handover. Record the endpoint you
actually used in the README and smoke script.

For the LLM, compare the served model against the approved model. A different id string
can still be the right model, but agent code must carry the exact served id.

### First boot takes much longer

A speech NIM's first start downloads models and can then build a TensorRT engine for this
GPU, which commonly runs 15 to 30 minutes or more with health reporting `starting` or a
temporary `unhealthy`. That is not a failure. Watch the logs for download or engine-build
progress and let it finish.

Declare failure only when the process exits, the logs show a fatal error such as CUDA OOM,
or progress stops well past the documented startup window. Do not change profiles, shrink
batch size, or restart the service just because first boot is slow, because a restart
throws away partial work. Keep the documented cache mount so later starts reuse the
downloaded models and the built engine and come up quickly.

### Per-slot reuse

Before starting anything: `docker ps`, hit health on ports already in use, and run the
speech gRPC configuration query when applicable. Keep a container that already matches
the locked image + profile/tags + model id. Replace only a missing, unhealthy, or wrong
slot. Stop/remove **that** container, update its generated Compose service from the
current source when needed, then start that service alone.
Tell the user which slots will be reused or replaced.
Never tear down healthy slots to fix one mismatch.

### Why this order

Intake decides language, framework, and pipeline. After that the deploy is the same ordered
run every time, and each step above exists because skipping it produced a failure that
looked like something else:

| Step | Guards against |
| --- | --- |
| Resolve image, profile, and served id separately | pull denied, wrong model id |
| Write Compose with the memory control already set | the LLM taking the GPU and starving speech |
| Start one slot at a time through the memory gate | an OOM restart loop hiding the real budget |
| Allow a long first boot | killing a healthy engine build |
| Resolve the voice, finalize settings, run smoke | silent connect failures in the browser |
| Start the agent and speak | a running process that is deaf or mute |

## Scaling across GPUs

This section covers slot placement on a multi-GPU workstation or DGX system. It does not
apply to single-GPU DGX Spark or Jetson Thor, and it does not define replica autoscaling.

List GPU UUIDs and current free memory with `nvidia-smi`. Prefer UUIDs for stable
assignments. Plan from each GPU's free memory, not the sum:

| Available GPUs | Default placement |
| --- | --- |
| One GPU that fits | LLM + ASR + TTS, with tight-fit knobs when required |
| Two GPUs | LLM on one, ASR + TTS on the other |
| Three GPUs | one slot per GPU |

Any GPU hosting more than one service needs its own runtime budget and the Shared-GPU
memory gate above. When ASR and TTS share a device, reserve both selected matrix
footprints plus startup and engine-build headroom before approving that placement.

Translate each slot's current build.nvidia.com command through `output-contract.md`, then
set its Compose GPU reservation to the selected device ids. Follow current
[Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/).
`capabilities: [gpu]` is required. `device_ids` and `count` are mutually exclusive.

The locked `NIM_MODEL_PROFILE` decides tensor parallelism. A `tp2` profile must see both
required GPUs. Do not split one LLM across GPUs unless the support matrix and
`list-model-profiles` return that topology as Compatible.

Start and verify one slot at a time. Confirm from inside each container that it sees only
the intended GPU set, then check host `nvidia-smi` to confirm memory landed on the planned
device. Keep agent endpoint constants aligned with the resulting host ports.

## Agent configuration

- Cloud: use the URLs and function ids in §Cloud.
- Workstation / DGX: use the host ports mapped from each live deploy page.
- Jetson Thor: use the vLLM / Riva mappings in `platforms/jetson-thor.md`.
- Self-hosted speech has empty function ids. Omni has no ASR.

## Anti-patterns

- Shipping a remembered image tag, `docker run`, or Compose recipe instead of generating it
  from the locked model's current build.nvidia.com instructions.
- Applying workstation / DGX NIM instructions to Jetson Thor.
- Starting all local NIMs together or assuming fixed ports.
- Applying the deprecated `NIM_TAGS_SELECTOR` to a generated LLM deployment or
  `NIM_MODEL_PROFILE` to ASR / TTS.
- Killing every container because one slot was wrong.
