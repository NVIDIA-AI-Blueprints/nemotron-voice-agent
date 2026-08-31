# Model Catalog

Exact ids for the proposal table. Never write them from memory. Open the file for the
slot you are resolving.

| Slot | Read | Discover first | Confirm on |
| --- | --- | --- | --- |
| Language / locale | `models/language-routing.md` | ASR + TTS language tables | selected framework docs MCP |
| Cascaded LLM | `models/llm.md` | LLM support matrix + `/v1/models` | `build.nvidia.com/nvidia/<model>` |
| Omni | `models/llm.md` | same, exclude from cascaded LLM | same |
| ASR | `models/asr.md` | [Speech docs](https://docs.nvidia.com/nim/speech/latest/) → [ASR matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) | `<slug>` and `<slug>/deploy` |
| TTS | `models/tts.md` | [TTS matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html) | same pattern |

Family browse: [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron#section-nvidia-nemotron-models).

## Shared rules

- Proposal table gets exact catalog ids. Family names alone are wrong. Those ids describe
  the choice, and agent code uses the runtime ids resolved after startup.
- Lock language routing before selecting ASR and TTS.
- Model ids, voice ids, and function ids go in generated code or config. `.env` is secrets only.
- LLM tools (`list-model-profiles`, `NIM_MODEL_PROFILE`, LLM support matrix) never apply to ASR or TTS.
- Speech profile tags (`NIM_TAGS_SELECTOR`, Speech matrices, `/deploy`) do not apply to
  generated LLM deployments. Pin an LLM with `NIM_MODEL_PROFILE`.
- For self-hosted Speech, the matrix owns profile fit and tags. The `/deploy` page owns
  image URI and launch shape. Stop if they still conflict.
- Shared-GPU placement uses the runtime budget in `preflight.md`, LLM controls in
  `models/llm.md`, and the explicit TTS batch profile in `models/tts.md`. It remains
  provisional until `platforms/deployment.md` §Shared-GPU memory gate passes.
- Jetson Thor bypasses Speech NIM selection and follows `platforms/jetson-thor.md`.
- If the user names a model outside these families, say so and wait. Do not substitute.

## Where they live

| Identifier | Who | Source |
| --- | --- | --- |
| Catalog slug | Proposal table | `build.nvidia.com/nvidia/<slug>`. Intent only, never agent code |
| Container image | Self-hosted LLM | the self-hosted build page or NGC command that actually pulls |
| `NIM_MODEL_PROFILE` | Self-hosted LLM | `list-model-profiles` on the assigned GPU |
| Runtime model id | LLM (and listing checks) | `GET /v1/models` · cloud `https://integrate.api.nvidia.com/v1/models` |
| Function id | Cloud ASR / TTS | build.nvidia.com model page. gRPC `grpc.nvcf.nvidia.com:443`. Empty when self-hosted |
| `CONTAINER_ID` + `NIM_TAGS_SELECTOR` | Self-hosted ASR / TTS | Speech matrix, then `/deploy` |
| Voice id | TTS | running service's documented voice-discovery API |

Slug, image, and runtime model id are three separate strings (`models/llm.md` §Resolve
three names). Agent code uses the runtime model id only.
