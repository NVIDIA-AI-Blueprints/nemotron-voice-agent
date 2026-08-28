# DGX Spark

Read when `preflight.md` classifies the target as `dgx_spark`. DGX Spark uses the
self-hosted NIM workflow in `platforms/deployment.md`, subject to the support matrix for
the locked model and profile.

## Before Proposing

1. Confirm the machine is DGX Spark / GB10 from DMI and `nvidia-smi`.
2. Use unified memory reported as available by the probe, not the advertised 128 GB.
3. Verify every local slot:
   - LLM: DGX Spark appears in the matching model × precision row in the LLM support
     matrix. Use that row as the proposal candidate. Run `list-model-profiles` after
     approval and container readiness to pin the exact Compatible profile.
   - ASR / TTS: the chosen model and profile support DGX Spark in the Speech matrices.
4. Build the runtime budget from available unified memory. Reserve the selected speech
   profiles, OS, framework, CUDA, startup, and engine overhead before assigning the
   remainder to LLM weights and KV cache. Use exactly one memory-control path from
   `models/llm.md` §Tight fit.
5. Mark provisional co-location until `platforms/deployment.md` §Shared-GPU memory gate
   passes.
6. If any slot is unsupported or the full layout does not fit, move that slot to NVIDIA
   cloud and show the hybrid layout in the proposal. Do not substitute a different model
   silently.

## Deploy

When a slot's locked source is a Hugging Face card rather than a NIM page, check that card for
a DGX Spark container before using its generic install line. Cards in this family have named a
separate container for DGX Spark and Jetson Thor. Harvest the rest of that card through
`platforms/jetson-thor.md` §What to take from the card, which applies to any raw vLLM slot.

Follow `platforms/deployment.md` for each approved local slot, including its shared
GPU memory gate.

If startup reports low memory, follow the compatible profile's `--max-model-len` guidance
or move a speech slot to cloud.

## Verify

- LLM readiness succeeds, `/v1/models` serves the approved model, and its exact served id
  is what the agent uses.
- ASR configuration matches the lock and accepts a streaming transcription request.
- TTS configuration matches the lock and returns audio for the approved voice.
- `scripts/smoke.sh` passes before the agent starts.
- A full spoken exchange succeeds before handover (`operations/run.md`).

## Anti-Patterns

- Assuming every NIM supports GB10 because another model does.
- Reusing a workstation image or profile without matrix verification.
- Budgeting against advertised unified memory instead of current available memory.
- Starting all model services concurrently on first load.
