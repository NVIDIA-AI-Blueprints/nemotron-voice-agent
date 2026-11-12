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
    case 'session': addEvent('🔊', `Session started · ${ev.recipe}`); showModels(ev.models); break;
    case 'tool_call': addEvent('🔧', `Tool call: ${ev.name}(${short(ev.args)})`); break;
    case 'tool_result':
      addEvent(ev.ok ? '✅' : '❌', `Tool result: ${ev.name} → ${short(ev.result)} (${ev.ms} ms)`, ev.ok ? 'ok' : 'error');
      break;
    case 'barge_in': addEvent('✋', 'Barge-in: you interrupted the agent mid-reply', 'interrupt'); break;
    case 'false_interruption':
      addEvent('↩️', `False interruption${ev.resumed ? ': agent resumed speaking' : ''}`, 'interrupt');
      break;
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
    top.append(kind, el('span', 'model-name', model.name));
    const meta = el('div', 'model-meta');
    meta.append(el('span', 'pill pill-runtime', model.runs));
    card.append(top, meta);
    strip.append(card);
  });
}

async function connect() {
  $('connect').disabled = true;
  setStatus('Requesting token…');
  const { token } = await (await fetch('/token')).json();
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
  $('connect').textContent = 'Disconnect';
  $('connect').classList.replace('primary', 'danger');
  $('connect').disabled = false;
  $('mute').classList.remove('hidden');
  $('meter').classList.remove('hidden');
}

function reset() {
  room = null;
  $('connect').textContent = 'Connect';
  $('connect').classList.replace('danger', 'primary');
  $('connect').disabled = false;
  $('mute').classList.add('hidden');
  $('meter').classList.add('hidden');
  $('mute').textContent = 'Mute mic';
  setStatus('Disconnected');
}

$('connect').onclick = () => (room ? room.disconnect() : connect());
$('mute').onclick = async () => {
  const on = room.localParticipant.isMicrophoneEnabled;
  await room.localParticipant.setMicrophoneEnabled(!on);
  $('mute').textContent = on ? 'Unmute mic' : 'Mute mic';
};
