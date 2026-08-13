#!/usr/bin/env bash
# Install the Magpie WordTiming-enabled nvidia-riva-client wheel.
# Stock PyPI 2.26.0 cannot decode meta.words / enable_word_time_offsets.
# Prefer this after `uv sync` / image rebuilds (those restore the lock pin).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHL="$ROOT/third_party/wheels/nvidia_riva_client-2.26.0+wordtiming-py3-none-any.whl"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONTAINER="${CONTAINER:-nemotron-voice-agent-generic-assistant-1}"

if [[ ! -f "$WHL" ]]; then
  echo "Missing wheel: $WHL" >&2
  exit 1
fi

echo "==> Host: $PYTHON"
uv pip install --python "$PYTHON" --force-reinstall --no-deps "$WHL"
"$PYTHON" -c "
import riva.client.proto.riva_tts_pb2 as t
assert hasattr(t, 'WordTiming'), 'WordTiming missing'
assert hasattr(t.SynthesizeSpeechRequest(), 'enable_word_time_offsets')
assert 'words' in [f.name for f in t.SynthesizeSpeechResponseMetadata.DESCRIPTOR.fields]
print('host WordTiming OK')
"

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "==> Container: $CONTAINER"
  remote="/tmp/$(basename "$WHL")"
  docker cp "$WHL" "$CONTAINER:$remote"
  docker exec "$CONTAINER" uv pip install --python /app/.venv/bin/python --force-reinstall --no-deps "$remote"
  docker exec "$CONTAINER" /app/.venv/bin/python -c "
import riva.client.proto.riva_tts_pb2 as t
assert hasattr(t, 'WordTiming')
assert hasattr(t.SynthesizeSpeechRequest(), 'enable_word_time_offsets')
assert 'words' in [f.name for f in t.SynthesizeSpeechResponseMetadata.DESCRIPTOR.fields]
print('container WordTiming OK')
"
  # ``uv run`` (container PID1) re-syncs the lock and restores stock PyPI
  # nvidia-riva-client on child respawn. Restart the app under --no-sync so the
  # WordTiming wheel stays loaded for Magpie meta.words.
  echo "==> Reloading $CONTAINER server with uv --no-sync (keeps WordTiming)"
  docker exec -d "$CONTAINER" bash -lc "
    pkill -9 -f 'src/server.py' || true
    sleep 1
    cd /app && exec env UV_NO_SYNC=1 uv run --no-sync python src/server.py --host 0.0.0.0
  "
  # Give the new process a moment; verify proto still WordTiming-capable.
  sleep 3
  docker exec "$CONTAINER" /app/.venv/bin/python -c "
import riva.client.proto.riva_tts_pb2 as t
assert hasattr(t, 'WordTiming'), 'WordTiming lost after reload'
print('container WordTiming still OK after reload')
"
else
  echo "Container $CONTAINER not running; skipped"
fi
