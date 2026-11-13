// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from aws-samples/sample-voice-agent-on-aws (MIT-0).

const { Room, RoomEvent, Track, ConnectionState } = LivekitClient;

const $ = (id) => document.getElementById(id);
const wsUrl = `ws://${location.hostname}:7880`;  // signaling; override with ?ws=ws://host:7880
const WS_URL = new URLSearchParams(location.search).get('ws') || wsUrl;
$('server-badge').textContent = WS_URL;

// Media flows directly to LiveKit over UDP and TCP. Remote browsers need a TURN server (not included).

let room = null;
const segments = new Map();  // segment id -> message element

function setStatus(t) { $('status').textContent = t; }

function messageEl(id, role) {
  if (segments.has(id)) return segments.get(id);
  $('empty')?.remove();
  const el = document.createElement('div');
  el.className = `message message-${role}`;
  el.innerHTML = `<span class="message-icon">${role === 'user' ? '🎤' : '🤖'}</span>` +
    `<span class="message-text"></span><span class="message-time"></span>`;
  $('messages').appendChild(el);
  segments.set(id, el);
  return el;
}

function render(id, role, text, final) {
  const el = messageEl(id, role);
  const t = el.querySelector('.message-text');
  t.textContent = text;
  t.classList.toggle('interim', !final);
  if (final) el.querySelector('.message-time').textContent = new Date().toLocaleTimeString();
  const box = $('messages');
  box.scrollTop = box.scrollHeight;
}

async function onTranscription(reader, participant) {
  const role = participant?.identity?.startsWith('agent') ? 'assistant' : 'user';
  const id = reader.info.attributes?.['lk.segment_id'] || reader.info.id;
  let text = '';
  for await (const chunk of reader) {
    text += chunk;
    render(id, role, text, false);
  }
  const final = reader.info.attributes?.['lk.transcription_final'] === 'true';
  render(id, role, text, final);
}

// Agent events arrive on a data-channel topic: tool calls/results, barge-in, session info.
const EVENTS_TOPIC = 'agent.events';
const CONTROL_TOPIC = 'agent.control';  // commands from this page to the agent
const short = (v) => { const t = typeof v === 'string' ? v : JSON.stringify(v); return t.length > 400 ? `${t.slice(0, 400)}…` : t; };

function addEvent(icon, text, cls = '') {
  $('empty')?.remove();
  const el = document.createElement('div');
  el.className = `message message-event ${cls}`;
  el.innerHTML = `<span class="message-icon">${icon}</span><span class="message-text"></span><span class="message-time"></span>`;
  el.querySelector('.message-text').textContent = text;
  el.querySelector('.message-time').textContent = new Date().toLocaleTimeString();
  const box = $('messages');
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

function onAgentEvent(ev) {
  switch (ev.type) {
    case 'session': addEvent('🔊', `Session started · ${ev.scenario} · ${ev.recipe}`); showModels(ev.models); break;
    case 'tool_call': addEvent('🔧', `Tool call: ${ev.name}(${short(ev.args)})`); break;
    case 'tool_result':
      addEvent(ev.ok ? '✅' : '❌', `Tool result: ${ev.name} → ${short(ev.result)} (${ev.ms} ms)`, ev.ok ? 'ok' : 'error');
      break;
    case 'barge_in': addEvent('✋', 'Barge-in: you interrupted the agent mid-reply', 'interrupt'); break;
    case 'false_interruption':
      addEvent('↩️', `False interruption${ev.resumed ? ': agent resumed speaking' : ''}`, 'interrupt');
      break;
    case 'voices': showVoices(ev.voices, ev.current); break;
    case 'voice_changed': addEvent('🗣️', `Voice changed to ${ev.name}`); break;
    case 'error': {
      const limited = /429|too many requests|rate.?limit/i.test(ev.message);
      const hint = limited ? ' The hosted model is rate limited. Wait a minute, or choose another model.' : '';
      addEvent('⚠️', `${ev.source} error: ${ev.message}${hint}`, 'error');
      break;
    }
    case 'agent_state': if (ev.state === 'thinking') setStatus('Thinking…'); break;
  }
}

// The models in use, reported by the agent when the session starts, so the page never shows stale names.
const STAGES = { ASR: 'Speech to text', LLM: 'Language model', TTS: 'Text to speech' };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function showModels(models) {
  const strip = $('model-strip');
  strip.innerHTML = '';
  models.forEach((model, i) => {
    if (i > 0) strip.append(el('span', 'model-arrow', '→'));
    const card = el('div', `model-card model-${model.kind.toLowerCase()}`);
    const top = el('div', 'model-top');
    const kind = el('span', 'model-kind', model.kind);
    kind.title = STAGES[model.kind] || '';
    const name = el('span', 'model-name', model.name);
    if (model.id) name.title = model.id;
    top.append(kind, name);
    const meta = el('div', 'model-meta');
    meta.append(el('span', 'pill pill-runtime', model.runs));
    card.append(top, meta);
    strip.append(card);
  });
}

async function connect() {
  $('connect').disabled = true;
  setStatus('Requesting token…');
  const params = new URLSearchParams({ scenario: $('scenario').value });
  if (voicesLoaded) params.set('voice', $('voice').value);
  const { token } = await (await fetch(`/token?${params}`)).json();
  room = new Room({ adaptiveStream: true, dynacast: true });

  room.registerTextStreamHandler('lk.transcription', onTranscription);
  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) $('audio-sink').appendChild(track.attach());
  });
  room.on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((e) => e.remove()));
  room.on(RoomEvent.ParticipantConnected, (p) => setStatus(`${p.identity} joined`));
  room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
    const local = speakers.find((s) => s.isLocal);
    $('meter-fill').style.width = `${Math.round((local?.audioLevel || 0) * 100)}%`;
    const agent = speakers.find((s) => !s.isLocal);
    if (agent) setStatus('Agent speaking…');
    else if (local) setStatus('Listening…');
  });
  room.on(RoomEvent.DataReceived, (payload, _p, _k, topic) => {
    if (topic !== EVENTS_TOPIC) return;
    try { onAgentEvent(JSON.parse(new TextDecoder().decode(payload))); } catch (e) { console.warn('bad event', e); }
  });
  room.on(RoomEvent.Disconnected, reset);

  try {
    await room.connect(WS_URL, token);
    await room.startAudio();
    await room.localParticipant.setMicrophoneEnabled(true);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    $('connect').disabled = false;
    return;
  }
  setStatus('Connected — waiting for agent');
  $('scenario').disabled = true;  // the use case is fixed for the session
  $('connect').textContent = 'Disconnect';
  $('connect').classList.replace('primary', 'danger');
  $('connect').disabled = false;
  $('mute').classList.remove('hidden');
  $('meter').classList.remove('hidden');
}

// The voices the TTS server offers. The agent worker publishes them when it starts, so the list is available before you
// connect. Choosing a voice before connecting sets the voice for the whole session. Choosing one during a session sends
// a command to the agent, which uses the new voice from its next reply and says a short sample.
let voicesLoaded = false;

function showVoices(voices, current) {
  const select = $('voice');
  select.innerHTML = '';
  for (const voice of voices) select.add(new Option(voice.name, voice.id, false, voice.id === current));
  select.disabled = voices.length < 2;
  select.title = voices.length < 2 ? 'The speech server offers one voice' : '';
  voicesLoaded = voices.length > 0;
}

async function loadVoices() {
  $('voice').innerHTML = '<option>Loading voices…</option>';
  $('voice').disabled = true;
  for (let attempt = 0; attempt < 45; attempt++) {  // the speech server can still be loading models
    try {
      const { voices, default: current } = await (await fetch('/voices')).json();
      if (voices.length) { showVoices(voices, current); return; }
    } catch (e) { /* retry */ }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  $('voice').innerHTML = '<option>Default voice (list unavailable)</option>';
}

$('voice').onchange = async () => {
  if (!room) return;  // before connecting, the choice is sent with the join token
  const message = JSON.stringify({ type: 'set_voice', voice: $('voice').value });
  await room.localParticipant.publishData(new TextEncoder().encode(message), { reliable: true, topic: CONTROL_TOPIC });
};

function reset() {
  room = null;
  $('scenario').disabled = false;
  $('connect').textContent = 'Connect';
  $('connect').classList.replace('danger', 'primary');
  $('connect').disabled = false;
  $('mute').classList.add('hidden');
  $('meter').classList.add('hidden');
  $('mute').textContent = 'Mute mic';
  setStatus('Disconnected');
}

// The deployment mode and the models in use, published by the agent worker, so they show before you connect.
async function loadDeployment() {
  for (let attempt = 0; attempt < 45; attempt++) {  // the agent worker may still be starting
    try {
      const deployment = await (await fetch('/deployment')).json();
      if (deployment.models) {
        $('mode-badge').textContent = deployment.label;
        $('mode-badge').classList.toggle('badge-local', deployment.recipe !== 'cloud');
        showModels(deployment.models);
        return;
      }
    } catch (e) { /* retry */ }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  $('mode-badge').textContent = 'Deployment unknown';
}

// The use cases (prompt and tools) that the agent offers. The choice is sent with the join token and is fixed for the session.
let scenarios = [];
function showScenario() {
  const scenario = scenarios.find((item) => item.id === $('scenario').value);
  if (!scenario) return;
  $('scenario-desc').textContent = scenario.description;
  const script = $('script');
  script.innerHTML = '<div class="script-title">🎙️ Say this</div>';
  for (const example of scenario.examples) script.append(el('div', 'script-line', `“${example}”`));
  const empty = $('empty-examples');
  if (empty) {
    empty.innerHTML = '';
    for (const example of scenario.examples) empty.append(el('div', '', `“${example}”`));
  }
}

async function loadScenarios() {
  try {
    scenarios = (await (await fetch('/scenarios')).json()).scenarios;
    for (const scenario of scenarios) $('scenario').add(new Option(scenario.name, scenario.id));
    showScenario();
  } catch (e) { $('scenario-desc').textContent = 'Use cases are unavailable; the default is used.'; }
}
$('scenario').onchange = showScenario;

loadScenarios();
loadDeployment();
loadVoices();
$('connect').onclick = () => (room ? room.disconnect() : connect());
$('mute').onclick = async () => {
  const on = room.localParticipant.isMicrophoneEnabled;
  await room.localParticipant.setMicrophoneEnabled(!on);
  $('mute').textContent = on ? 'Unmute mic' : 'Mute mic';
};
