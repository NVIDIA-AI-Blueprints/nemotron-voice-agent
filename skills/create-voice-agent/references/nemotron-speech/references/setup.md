# Riva NIM setup | host install canonical

Install drivers Docker Container Toolkit NGC login. Readiness checks→deployment-readiness-checks.md (REF here install only). Announce Step N/7. Fetch per-release driver OS WSL2 glibc from docs not infer.

NIM=https://docs.nvidia.com/nim/speech/latest Docs: prerequisites,support-matrix asr/tts/nmt,container-toolkit,docker install,CUDA install guide,NGC API keys

Invariants: x86_64 NVAIE self-host | host driver only CUDA in container | steps 1-3 root 4-7 user once per machine

```
1 nvidia-smi (fetch prerequisites min driver)
2 docker install docs.docker.com; if root via sudo, run `usermod -aG docker <original-login-user>` as that user (not `$USER` after elevation); re-login as operator
3 apt nvidia-container-toolkit; nvidia-ctk runtime configure docker; restart docker
4 verify docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
5 NGC_API_KEY export org.ngc.nvidia.com setup api-keys enable NGC Catalog — **local nvcr.io auth only**; same value as NVIDIA_API_KEY for docker login; **does not select cloud routing** (deployment mode is explicit per gate)
6 echo "${NGC_API_KEY:?Set NGC_API_KEY}"|docker login nvcr.io --username '$oauthtoken' --password-stdin
7 pip install nvidia-riva-client; optional clone python-clients cpp-clients websocket-bridge
```

Trouble: username literal $oauthtoken | no host CUDA toolkit | docker group re-login | ldd --version vs glibc | WSL2 Podman subset fetch prerequisites
Next: model-selection deployment-readiness asr tts nmt
