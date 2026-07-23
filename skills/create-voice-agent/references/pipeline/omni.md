# Omni pipeline | Pipecat only

Trigger: PIPELINE_MODE omni|Nemotron Omni|audio-in LLM|ASR+LLM fused.
FORBID: LiveKit omni; NvidiaSTTService+NvidiaLLMService omni layout; ASR NIM sidecar local omni.

Omni=audio-in model replaces STT+text LLM; TTS separate. Not S2S. Not NvidiaLLMService drop-in.
| mode | slots | order |
| --- | --- | --- |
| cascaded | ASR+LLM+TTS | transport→STT→user_agg→LLM→TTS→transport→assistant_agg |
| omni | Omni+TTS | transport→user_agg→Omni LLMService→TTS→transport→assistant_agg |

PIPELINE_MODE=omni in bot.py. LiveKit→redirect Pipecat.

## Gate
Skip ASR row. Rows: pipeline omni|omni model| TTS. Speech: ASR boost N/A; TTS custom_dictionary OK; FORBID Parakeet ASR boost wizard.
Delegated rec: REQ inline 2-row table (Omni model id + TTS model+voice+function_id) per model-selection.md §Rec display before confirm WAIT.

## Model
| deploy | source | endpoint |
| --- | --- | --- |
| cloud | GET /v1/models filter omni | https://integrate.api.nvidia.com/v1 |
| local | HF/vLLM | http://127.0.0.1:8002/v1 |

Defaults(verify): cloud `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | local HF `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
Reasoning extra_body→llm-reasoning.md. Local: docker-compose.nim.omni.yml vLLM sidecar NOT NIM_MODEL_PROFILE LLM compose.

## VRAM
| platform | local omni |
| --- | --- |
| workstation | ≥~80GB 30B NVFP4 + TTS |
| DGX/Jetson 128GB | sidecar OK |
| Orin | unsupported |
| cloud | NVCF |

Budget Omni+TTS only. Probe nvidia-smi.

## Scaffold
```
gate omni+transport+models → model-selection omni if delegated
→ COPY assets/omni_service.py + assets/audio_only_smart_turn.py (+ turn_support if WebRTC remote) to project root
→ bot.omni-runner.md + bot.omni-service-spec.md
→ bot.py per runner (pipeline order, AudioOnlySmartTurnStopStrategy, on_client_connected greeting)
→ pyproject pipecat-ai[nvidia,runner,webrtc,silero] openai loguru
→ local: docker-compose.nim.omni.yml + HF_TOKEN | run.md verify
```

Custom modules (COPY from assets — do not reimplement):
| file | source |
| --- | --- |
| `omni_service.py` | `assets/omni_service.py` |
| `audio_only_smart_turn.py` | `assets/audio_only_smart_turn.py` |
| `bot.py` | `assets/bot.omni-runner.md` |

FORBID: agent-written omni_service from spec alone; stock Smart Turn stop without AudioOnlySmartTurn; ASR in pipeline.

## Greeting (connect)

REQ: **`TTSSpeakFrame`** + context `assistant` message + **`MuteUntilFirstBotCompleteUserMuteStrategy`**. FORBID `LLMRunFrame` / fake user “introduce yourself” for omni connect greeting — text-only omni is unreliable; mic opens on `client-ready` and can race an audio omni turn (TTS speaks garbage like `"27 30"`). See `bot.omni-runner.md` §Greeting; troubleshoot §omni-greeting.

Env: OMNI_EMIT_TRANSCRIPTIONS=true | OMNI_MIN_USER_AUDIO_SECS=0.3 | OMNI_MAX_TOKENS=8192 | OMNI_TEMPERATURE=0.6 | OMNI_TOP_P=0.95
Local compose env: NVIDIA_API_KEY, HF_TOKEN. Health→run.md omni slice.

## Iterate
model swap→model-selection omni | turn issues→troubleshoot turn+AudioOnlySmartTurn | OOM→lower gpu-mem/max-model-len | cascaded→full gate

Out of scope: subagents/webcam/LiveKit omni/tools unless explicit ask.

Anti: ASR in pipeline|ASR sidecar|NvidiaSTTService before omni|nim-llm for omni vLLM|LiveKit agent omni
