# Cloud catalog | GET /v1/models discovery

FORBID hardcode names/function_id. Cloud primary. Local NIM images function_id empty.
Hierarchy: GET /v1/models+this file first build.nvidia.com backup if unclear. User choice overrides.

Key NVIDIA_API_KEY build.nvidia.com | LLM https://integrate.api.nvidia.com/v1 | ASR/TTS grpc.nvcf.nvidia.com:443
```bash
curl -s https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $NVIDIA_API_KEY" -H Accept:application/json
```
Returns data[].id no function_id. build.nvidia.com/{org}/{model}→function_id voices snippets.

GET limits: filter id keywords asr,tts,parakeet,magpie,nemotron exclude embed,guard,vlm. No function_id in JSON.

## Cascaded LLM whitelist
After GET keep ids matching: nemotron-3-nano NOT omni | nemotron-3-super | nemotron-3-ultra
FORBID Llama legacy embed guard vlm omni for cascaded LLM. Pick model-selection §LLM routing. Precision NVCF-managed. Omni→pipeline/omni.md not whitelist.

Backup build pages when unclear: Nano build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b | Super .../nemotron-3-super-120b-a12b | Ultra .../nemotron-3-ultra-550b-a55b. Fetch one card when needed not upfront.

Workflow: cascaded vs S2S default cascaded → GET → filter slots → rec within whitelist language-routing if non-EN → delegated STOP model-selection → named→build card function_id → wire pipecat.md

Constants py:
```python
LLM_MODEL_ID="nvidia/..."; ASR_MODEL="..."; ASR_FUNCTION_ID="..."; TTS_VOICE_ID="..."; TTS_FUNCTION_ID="..."
```
No services.yaml no models in .env.

Do not ask per-slot MCQ if user named all three confirmed. Open slots→model-slot-mcq.md. Other/non-NVIDIA→warn WAIT.
