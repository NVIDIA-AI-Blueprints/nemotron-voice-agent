# WebRTC+coturn | TURN layer only

CTX: Pipecat GPU VM browser laptop often SSH. REQ checklist→first-try-workstation-webrtc.md canonical. Mac handoff→ssh.md §Mac remote connect.

Layers: NIMs 127.0.0.1 | bot VM | :7860 signaling | coturn UDP 3478+49160-49200 | browser tunnel or VM_IP. SSH -L=TCP only audio UDP via TURN VM_IP.

bot entry:
```python
if __name__=="__main__":
    from turn_support import apply_turn_patches; apply_turn_patches()
    from pipecat.runner.run import main; main()
```
turn_support: patch SmallWebRTCRequestHandler /start iceConfig TURN+STUN. FORBID from __future__ annotations(422). Copy assets/turn_support.py Pipecat1.3+.

.env: NVIDIA_API_KEY TURN_USERNAME TURN_PASSWORD TURN_URL=turn:VM_IP:3478
coturn: x86 default `COTURN_IMAGE=instrumentisto/coturn:4` (compose default) | Jetson aarch64 **must** override before compose up:
```bash
export VM_IP=$(hostname -I | awk '{print $1}')
export COTURN_IMAGE=coturn/coturn:4.6.2-r8
export TURN_USERNAME='your-turn-user' TURN_PASSWORD='your-strong-secret'
docker compose -f docker-compose.turn.yml up -d
```
network_mode host external-ip=VM_IP ports 3478 49160-49200 lt-cred-mech — FORBID instrumentisto on Jetson aarch64

Workflow: sidecars run.md → free :7860 → turn compose up → bot --host 127.0.0.1 tunnel or VM_IP direct → firewall TCP7860 UDP3478 49160-49200
WS fallback UDP blocked→pipecat.md §WebSocket bot_ssh_server.py

TURN-specific: /start 422→no future annotations | STUN only→apply_turn_patches restart | tunnel 404→bind 127.0.0.1:7860
Other failures→first-try
