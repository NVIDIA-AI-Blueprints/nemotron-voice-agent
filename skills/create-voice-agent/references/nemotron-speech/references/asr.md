# ASR NIM | cloud+self-host

Announce Step N/M. Fetch per-release from docs not file.
NIM=https://docs.nvidia.com/nim/speech/latest routing: support-matrix/asr, customization, protos, realtime-asr, pipeline-configuration, prerequisites, performance

Naming slugs differ matrix/CONTAINER_ID/NVCF/build—resolve FID NVCF API never hardcode.

Protocol: cloud grpc.nvcf.nvidia.com:443 --use-ssl first HTTP fallback per card | self-host gRPC :50052 HTTP :9001 (host-mapped gRPC often :50152)

Cloud: pip nvidia-riva-client | NVCF functions filter parakeet|canary|whisper|nemotron-asr ACTIVE | transcribe_file.py --server grpc.nvcf.nvidia.com:443 --use-ssl --metadata function-id FID --metadata "authorization" "Bearer $NVIDIA_API_KEY" --language-code en-US --input-file audio.wav (--input-file not --audio-file)

Self-host Step1 vars CONTAINER_ID NIM_TAGS_SELECTOR from matrix | docker run nvcr.io/nim/nvidia/$CONTAINER_ID ports 9001/50052 NGC_API_KEY cache mount | readiness curl http://127.0.0.1:9001/v1/health/ready or gRPC GetRivaSpeechConfig | transcribe via python-clients or gRPC

Features fetch customization: word boost ITN diarization force_eou timestamps. Custom acoustic→asr-custom pipeline-configuration local deploy-time.

Health: curl -fsS http://127.0.0.1:9001/v1/health/ready
