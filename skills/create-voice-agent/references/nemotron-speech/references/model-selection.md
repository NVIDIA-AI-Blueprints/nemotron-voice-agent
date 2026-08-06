# Speech model selection | ASR/TTS/NMT defaults

Riva tasks entry. Procedure before asr/tts/nmt. Skill defaults first matrix URLs only if selector/VRAM unclear.

Defaults (override wins):
| slot | default |
| ASR EN | Nemotron ASR Streaming |
| ASR auto-detect multilingual | Parakeet RNNT 1.1B Multilingual |
| ASR single locked lang | Nemotron ASR Streaming Multilingual |
| TTS | Magpie TTS Multilingual |
| NMT | Riva Translate 1.6b spoken translation only |

Docs backup: support-matrix asr/tts/nmt customization pages api.nvcf.nvidia.com/v2/nvcf/functions ACTIVE

Procedure:
- **Deployment mode explicit** (gate row 2 cloud|local|jetson) or user states cloud vs local — **do not infer cloud from NVIDIA_API_KEY presence** (key is also required for local nvcr.io pull per setup.md)
- cloud chosen→SERVER=grpc.nvcf.nvidia.com:443 discover FID apply defaults unless user specifies
- local chosen→reuse docker ps riva|nim gRPC Get*Config per slot: `ASR_SERVER=127.0.0.1:50152` `TTS_SERVER=127.0.0.1:50151` `NMT_SERVER=127.0.0.1:50051` (discover each service independently; container gRPC ASR :50052)
- local fresh feasibility GPU VRAM toolkit disk 10-30GB NGC fail→cloud
Handoff cloud→modality ref FID | local running→per-slot SERVER map above | local fresh→CONTAINER_ID+NIM_TAGS_SELECTOR defaults then matrix. ASR audio mono WAV 16bit or Opus.

ASR families: Nemotron ASR Streaming EN default | Nemotron ASR Streaming Multilingual locked lang | Parakeet RNNT Multilingual auto-detect multi | RNNT EN explicit word-boost | CTC word boost timestamps explicit | TDT offline timestamps | Canary multi+bidi translate | Whisper broad langs translate EN offline | Conformer legacy custom asr-custom

TTS Magpie Multilingual voices --list-voices GET list_voices case-sensitive
NMT bidirectional --list-models pairs

Probes local: nvidia-smi docker nvidia-smi NVCF functions curl docker ps riva|nim df cache. Then gRPC Get*Config HTTP health alone insufficient.

Next: asr tts nmt deployment-readiness-checks
