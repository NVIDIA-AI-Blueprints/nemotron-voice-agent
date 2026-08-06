# SSH remote GPU

CTX: GPU via SSH browser on laptop. Build bot on VM open :7860/client. REF: dgx-spark,jetson-thor,pipecat,first-try,webrtc-turn,run.

Default: local GPU/SSH→Pipecat WebRTC /client. use already deployed→per-slot probe (deployment.md §selective); skip recreate for matching LLM/ASR/TTS. Gate if framework missing.

## Mac remote connect (canonical handoff TPL — other refs REF here)

Tunnel: bot `--host 127.0.0.1` FORBID 0.0.0.0

**Terminal 1 (on VM):** start the bot and leave it running.

```bash
uv run python bot.py --host 127.0.0.1 --port 7860 -t webrtc
```

**Terminal 2 (on Mac):** open the SSH tunnel only (do not start the bot here).

```bash
ssh -N -L 7860:127.0.0.1:7860 user@VM_IP
```

Browser: `http://localhost:7860/client`. Port busy on Mac: `-L 17860:127.0.0.1:7860` → open `http://localhost:17860/client`
Direct LAN: bot `--host VM_IP` and browser `http://VM_IP:7860/client`
Audio: ssh `-L` is signaling TCP only; voice UDP uses TURN→webrtc-turn.md

Handoff TPL: `Mac: ssh -N -L 7860:127.0.0.1:7860 user@VM_IP. Open localhost:7860/client. Audio TURN to VM_IP — POST /start turn creds.`

Workflow: docker ps map ports → health run.md → bot DEPLOYMENT_PLATFORM=workstation → start per Mac connect → handoff TPL if Mac
Skeleton bot.workstation-runner.md. WS last resort pipecat.md §WebSocket.

SSH-specific: Mac can't reach VM→--host VM_IP or tunnel | port in use→7861 or pkill | tunnel debug→`ssh -v -o ExitOnForwardFailure=yes` (`-N` expected for tunnel-only)
WebRTC failures→first-try
