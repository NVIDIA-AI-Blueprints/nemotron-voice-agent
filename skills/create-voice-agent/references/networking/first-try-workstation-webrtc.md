# First-try WebRTC | canonical REQ+verify

Canonical WebRTC/TURN/Pipecat1.3 REQ+pre-handoff verify. Other refs REF here.
Scope: workstation,dgx_spark,remote browser,SSH+Mac.

## REQ
| # | req | fail |
| --- | --- | --- |
| 1 | pyproject pipecat-ai[nvidia,runner,webrtc,silero]>=1.3.0 | POST /start 400 |
| 2 | turn_support.py+docker-compose.turn.yml+coturn | no UDP audio |
| 3 | apply_turn_patches() before main() | no TURN iceConfig |
| 4 | .env TURN_URL=turn:VM_IP:3478 USER PASS | ICE fail |
| 5 | compose GPU pin multi-GPU | ASR OOM |
| 6 | --host 127.0.0.1 tunnel or VM_IP direct | 0.0.0.0 breaks ICE |
| 7 | WorkstationNvidiaLLMService supports_developer_role=False | vLLM 400 |
| 8 | LLM id curl /v1/models | wrong id |
| 9 | TTS language=Language.EN_US not language_code= | Connect crash |
| 10 | import TransportParams | NameError |
| 11 | turn_support Pipecat1.3 imports per assets/turn_support.py | import error |

Mac handoff: ssh.md §Mac remote connect TPL every remote WebRTC reply. Tunnel REQ --host 127.0.0.1.

Verify:
```bash
curl -sf http://127.0.0.1:18000/v1/health/ready; curl -sf http://127.0.0.1:9001/v1/health/ready; curl -sf http://127.0.0.1:9000/v1/health/ready
curl -s -X POST http://127.0.0.1:7860/start -H 'Content-Type: application/json' -d '{"enableDefaultIceServers":true,"transport":"webrtc"}'|python3 -m json.tool
docker inspect <asr> --format 'RestartCount={{.RestartCount}}'; nvidia-smi
```
Expect iceServers STUN+turn:VM:3478+creds. Multi-GPU: LLM_GPU=0 ASR_GPU=1 TTS_GPU=2.

Failures: POST /start 400→webrtc extra | spins→coturn+/start | language_code→Language.EN_US | ASR RestartCount→GPU pin | ssh tunnel stuck→use `ssh -v -o ExitOnForwardFailure=yes` (`-N` is expected for port-forward-only) | TTS disk→docker image prune
