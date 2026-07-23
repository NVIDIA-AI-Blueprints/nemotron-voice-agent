# TTS NIM

Cloud or self-host. Step N/4. Fetch per-release docs.
Docs: support-matrix/tts, customization, protos, realtime-tts, prerequisites. FID via NVCF API.

Prereq self-host setup.md | cloud pip nvidia-riva-client NVIDIA_API_KEY | model model-selection.md

Cloud: FID magpie|tts filter → talk.py grpc.nvcf.nvidia.com:443 --use-ssl metadata function-id+authorization --list-voices omit text no hardcode FID

Self-host: CONTAINER_ID+NIM_TAGS_SELECTOR matrix docker run ports **container** 9000/50051 map to **host** 9000/50151 (gRPC clients use `127.0.0.1:50151`; HTTP health from host: `curl http://127.0.0.1:9000/v1/health/ready` — use `:9000` inside container only) NGC cache chown 1000:1000 batch_size selector | readiness host curl http://127.0.0.1:9000/v1/health/ready OR gRPC GetRivaSynthesisConfig on 127.0.0.1:50151 empty models=unhealthy | list voices talk.py or GET list_voices case-sensitive | synthesize talk.py HTTP synthesize/synthesize_online stream realtime_tts_client WS host :9000

Ports: container 9000 HTTP/WS/health, 50051 gRPC → host-mapped 9000/50151. Generated bot clients: `TTS_SERVER=127.0.0.1:50151`.
Trouble: voice not recognized list-voices | FID rejected refresh NVCF | HTTP stream not WAV sox wrap
