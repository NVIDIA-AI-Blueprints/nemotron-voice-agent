# NMT NIM | text translation

Bidirectional translation. Step N/M. Fetch models/pairs/features docs.
Docs: support-matrix/nmt, nmt/customization, protos, prerequisites

Prereq self-host setup+NGC | cloud nvidia-riva-client+NVIDIA_API_KEY

Workflow: deploy self-host → readiness → --list-models running server language codes → translate

Self-host: CONTAINER_ID matrix docker run 9000/50051 cache mount
Readiness: curl -fsS http://127.0.0.1:9000/v1/health/ready gRPC probe if pre-existing
Cloud: NVCF FID megatron|riva-translate filter translate python-clients grpc.nvcf.nvidia.com:443 --use-ssl

Features fetch customization before dnt tags custom dict max-length batch. Always --list-models on running NIM for exact language codes.
