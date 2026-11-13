# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src/examples/generic_assistant_livekit/ui/app.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required")

# Runs the real page script in Node with a stubbed browser. A function that is called but never defined, or a handler
# that is never assigned, shows up here. `node --check` only validates syntax.
HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const scenario = process.argv[3];
const code = fs.readFileSync(process.argv[2], 'utf8');
const elements = {};
const makeEl = () => ({
  style: {}, classList: { add() {}, remove() {}, toggle() {}, replace() {} }, textContent: '',
  value: '', disabled: false, open: true, title: '', options: [], _html: '',
  get innerHTML() { return this._html; },
  set innerHTML(v) { this._html = v; this.options.length = 0; },
  add(option) { this.options.push(option); }, append() {}, appendChild() {}, remove() {},
  querySelector: () => makeEl(), scrollTop: 0, scrollHeight: 0,
});
const document = { getElementById: (id) => (elements[id] ||= makeEl()), createElement: () => makeEl() };
const log = [];
let releaseToken = null;
class Room {
  constructor() {
    this.localParticipant = {
      setMicrophoneEnabled: async () => {},
      publishData: async (d, o) => log.push(['publish', o.topic]),
    };
  }
  registerTextStreamHandler() {}
  on() {}
  async connect() { if (scenario === 'connect-fails') throw new Error('no route'); }
  async startAudio() {}
  async disconnect() { log.push(['disconnect']); }
}
const responses = {
  '/voices': { voices: [{ id: 'A', name: 'A' }, { id: 'B', name: 'B' }], default: 'A' },
  '/deployment': { recipe: 'cloud', label: 'Cloud', models: [{ kind: 'ASR', name: 'x', runs: 'y' }] },
  '/scenarios': { scenarios: [{ id: 'generic', name: 'Generic', description: 'd', examples: ['e'] }] },
};
const context = {
  document, location: { hostname: 'localhost' }, console, URLSearchParams, TextEncoder, TextDecoder, Promise, JSON, Map,
  setTimeout: () => 0,
  Option: function (text, value) { return { text, value }; },
  LivekitClient: { Room, RoomEvent: {}, Track: { Kind: { Audio: 'audio' } }, ConnectionState: {} },
  fetch: async (url) => {
    log.push(['fetch', url.split('?')[0]]);
    if (scenario === 'token-pending' && url.startsWith('/token')) {
      await new Promise((resolve) => { releaseToken = resolve; });
    }
    if (scenario === 'token-fails' && url.startsWith('/token')) {
      return { ok: false, status: 500, json: async () => ({}) };
    }
    const path = url.split('?')[0];
    return { ok: true, status: 200, json: async () => responses[path] || { token: 't', room: 'r' } };
  },
};
vm.createContext(context);
const done = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  vm.runInContext(code, context);  // throws a ReferenceError here if a function is called but not defined
  for (let i = 0; i < 10; i++) await done();
  const out = {
    handlers: ['connect', 'mute', 'voice', 'scenario'].map((id) => {
      const event = id === 'connect' || id === 'mute' ? 'onclick' : 'onchange';
      return [id, typeof (elements[id] || {})[event]];
    }),
    voiceOptions: elements.voice.innerHTML,
    badge: elements['mode-badge'].textContent,
  };
  if (scenario !== 'load') {
    elements.connect.onclick();
    for (let i = 0; i < 10; i++) await done();
    out.voiceDisabledWhilePending = elements.voice.disabled;
    out.scenarioDisabledWhilePending = elements.scenario.disabled;
    if (releaseToken) { releaseToken(); for (let i = 0; i < 10; i++) await done(); }
    out.connectDisabled = elements.connect.disabled;
    out.connectLabel = elements.connect.textContent;
    out.status = elements.status.textContent;
    out.scenarioDisabled = elements.scenario.disabled;
    out.voiceDisabled = elements.voice.disabled;
    out.tokenRequest = log.filter((e) => e[0] === 'fetch' && e[1] === '/token').length;
  }
  out.log = log;
  console.log(JSON.stringify(out));
})().catch((e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(1); });
"""


def run_page(scenario, tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS)
    result = subprocess.run(
        ["node", str(harness), str(APP_JS), scenario], capture_output=True, text=True, timeout=60, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_page_script_loads_and_wires_every_control(tmp_path):
    out = run_page("load", tmp_path)
    assert "error" not in out
    assert dict(out["handlers"]) == {
        "connect": "function",
        "mute": "function",
        "voice": "function",
        "scenario": "function",
    }
    assert ["fetch", "/voices"] in out["log"]
    assert ["fetch", "/deployment"] in out["log"]
    assert ["fetch", "/scenarios"] in out["log"]
    assert out["badge"] == "Cloud"
    assert "Loading voices" not in out["voiceOptions"]  # the list was loaded and replaced the placeholder


def test_connecting_locks_the_use_case_and_requests_a_token(tmp_path):
    out = run_page("connect-ok", tmp_path)
    assert out["tokenRequest"] == 1
    assert out["scenarioDisabled"] is True
    assert out["connectLabel"] == "Disconnect"
    assert out["connectDisabled"] is False


def test_the_voice_and_use_case_are_locked_while_the_connection_is_pending(tmp_path):
    out = run_page("token-pending", tmp_path)
    assert out["voiceDisabledWhilePending"] is True  # a change now would be dropped: no room exists yet
    assert out["scenarioDisabledWhilePending"] is True
    assert out["voiceDisabled"] is False  # once connected, the voice can change during the session
    assert out["scenarioDisabled"] is True  # the use case stays fixed for the session
    assert out["connectLabel"] == "Disconnect"


@pytest.mark.parametrize("scenario", ["token-fails", "connect-fails"])
def test_a_failed_connect_leaves_the_page_ready_for_another_try(tmp_path, scenario):
    out = run_page(scenario, tmp_path)
    assert out["connectDisabled"] is False
    assert out["connectLabel"] == "Connect"
    assert out["scenarioDisabled"] is False  # unlocked again after the failure
    assert out["voiceDisabled"] is False
    assert out["status"].startswith("Error:")
    if scenario == "connect-fails":
        assert ["disconnect"] in out["log"]  # the room is left, so the agent does not wait in it
