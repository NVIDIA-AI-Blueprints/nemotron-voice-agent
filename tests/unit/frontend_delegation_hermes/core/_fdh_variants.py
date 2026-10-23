# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

"""Test-only configuration variants over the shipped FDH configs.

The example ships only the demo and eval configs. Tests that need a feature switched on
or off (a "control" with every fix off, or one fix over that control) build it here: each variant is
the settings it changes over a base, which is a shipped config or another variant. :func:`variant`
writes it to a temporary file whose ``extends:`` (and ``voice_profile:``) name the base by absolute path,
so the shipped loader resolves and merges it exactly as it would a file in ``config/``.

Variant names keep the old file names (``profiles/...``, ``voice/...``, ``gateway...``), so a test
reads as before.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import textwrap
from functools import cache
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[4] / "src" / "examples" / "frontend_delegation_hermes" / "config"

_OFF_9 = """\
prompt_features:
  format_correction: false
  write_consent_v2: false
  policy_check: false
  request_scope: false
  transfer_check: false
  capability_check: false
  situational_facts: false
  identity_from_identifier: false
  domain_notes: false
"""
_GUARDS_OFF = """\
delegation:
  guards: {liveness_reply: false, question_never_silent: false, ping_as_status: false}
  replay_unheard_answer: {enabled: false, ttl_s: 30, triggers: [status_intent], whole_if_heard_below: 0}
"""

#: name -> (base, voice_profile or None, settings). A base or voice profile is a shipped config path
#: relative to ``config/`` or another variant's name.
VARIANTS: dict[str, tuple[str, str | None, str]] = {
    # The τ³ voice profile and the stub profiles, once shipped (removed with the τ³ adapter, shared-stack-plan.md
    # step 1): tau3-bench voice settings (client tools, no greeting, the tau2 greeting seeded, one tool-argument
    # rule selected by tool name), and the no-GPU, no-LLM stub runs (rule-based frontend, in-process fake gateway).
    "voice/tau3.yaml": (
        "voice/base.yaml",
        None,
        """\
protocol:
  greeting: {enabled: false, text: ""}
  seed_history_with_client_greeting: "Hi! How can I help you today?"

tools:
  source: client

normalization:
  tool_arguments:
    enabled: true
    rules:
      - tool: get_user_details
        argument: user_id
        label: user ID
        format_hint: firstname_lastname_1234
        spoken_form: true
        strip: " .-"
        collapse_separators: "_"
        case: lower
        pattern: '^[a-z]+_[a-z]+_\\d{4}$'
        on_invalid: answer_locally
    retry_guard:
      enabled: true
      scope: rules
      permanent_failure_pattern: '^Error: .*\\bnot found\\b'
""",
    ),
    "profiles/stub.yaml": (
        "delegation_agent.yaml",
        "voice/tau3.yaml",
        """\
delegation:
  speak_when_delegating: true
  spelling_hold:
    complete_patterns: ['^[A-Za-z0-9]{6}$']
tools:
  executor: wire
frontend:
  decider: scripted
backend:
  link: in_process_fake
  simulated_delay: {seconds: 0, where: per_delegation}
  status_verbalizer: {mode: template}
""",
    ),
    "profiles/stub_gate.yaml": (
        "profiles/stub.yaml",
        None,
        f"backend:\n  gateway_config: {CONFIG / 'gateway.stub_gate.yaml'}\n",
    ),
    # Gateway: backend prompt features.
    "gateway.eval_control.yaml": ("gateway.eval.yaml", None, _OFF_9),
    "gateway.eval_124acf7.yaml": (
        "gateway.eval.yaml",
        None,
        "prompt_features: {situational_facts: false, identity_from_identifier: false, domain_notes: false}\n",
    ),
    "gateway.eval_situational.yaml": (
        "gateway.eval_124acf7.yaml",
        None,
        "prompt_features: {situational_facts: true}\n",
    ),
    "gateway.eval_no_note.yaml": ("gateway.eval.yaml", None, "prompt_features: {domain_notes: false}\n"),
    "gateway.baseline.yaml": (
        "gateway.yaml",
        None,
        _OFF_9 + "  spelling_v2: false\n  spoken_output: false\n  write_consent: false\n  phone_format: false\n",
    ),
    "gateway.identity_control.yaml": ("gateway.yaml", None, _OFF_9 + "  phone_format: false\n"),
    # Voice: normalization over the shipped tau3 and realtime_eval voice profiles.
    "voice/tau3_spelling.yaml": (
        "voice/tau3.yaml",
        None,
        """\
normalization:
  transcript:
    spelled_runs: {enabled: true, min_tokens: 3, case: keep}
  tool_arguments:
    invalid_message_key: tool_argument_invalid_readback
    escalate_invalid_message_key: tool_argument_invalid_spell_all
""",
    ),
    "voice/tau3_recovery_hint.yaml": (
        "voice/tau3_spelling.yaml",
        None,
        """\
normalization:
  tool_arguments:
    result_hints:
      enabled: true
      tools: [find_user_id_by_name_zip, find_user_id_by_email, get_user_details]
      failure_pattern: '^Error: .*\\bnot found\\b'
      count_local_invalid: true
      max_hints: 3
      message_key: lookup_not_found_hint
      escalate_message_key: ""
""",
    ),
    "voice/tau3_fixes.yaml": (
        "voice/tau3_recovery_hint.yaml",
        None,
        "normalization: {tool_arguments: {result_hints: {escalate_message_key: lookup_not_found_hint_words}}}\n",
    ),
    "voice/realtime_eval_control.yaml": (
        "voice/realtime_eval.yaml",
        None,
        """\
normalization:
  tool_arguments:
    format_retry: {enabled: false}
    result_hints:
      message_key: lookup_not_found_hint
      field_provenance: false
      ledger: false
      max_hints: 3
""",
    ),
    "voice/realtime_eval_124acf7.yaml": (
        "voice/realtime_eval.yaml",
        None,
        "normalization: {tool_arguments: {result_hints: {ledger: false, field_provenance: true, max_hints: 3}}}\n",
    ),
    "voice/realtime_eval_identity.yaml": (
        "voice/realtime_eval_124acf7.yaml",
        None,
        """\
normalization:
  tool_arguments:
    result_hints: {ledger: true, identification_tools: auto, field_provenance: false, max_hints: 6}
""",
    ),
    "voice/realtime_eval_a1_format_retry.yaml": (
        "voice/realtime_eval_control.yaml",
        None,
        "normalization: {tool_arguments: {format_retry: {enabled: true, mode: call, fields: phone}}}\n",
    ),
    "voice/realtime_eval_a3_field_provenance.yaml": (
        "voice/realtime_eval_control.yaml",
        None,
        "normalization: {tool_arguments: {result_hints: {field_provenance: true}}}\n",
    ),
    # Profiles: the tau3 control and its single-fix variants.
    "profiles/tau3_eval.yaml": (
        "delegation_agent.yaml",
        "voice/tau3_fixes.yaml",
        """\
delegation:
  speak_when_delegating: true
  spelling_hold:
    complete_patterns: ['^[A-Za-z0-9]{6}$']
backend:
  link: websocket
  simulated_delay: {seconds: 0, where: per_delegation}
tools:
  executor: wire
""",
    ),
    "profiles/tau3_eval_baseline.yaml": (
        "profiles/tau3_eval.yaml",
        "voice/tau3.yaml",
        """\
delegation:
  spelling_hold: {enabled: false, complete_patterns: []}
  replay_unheard_answer: {enabled: false}
  filler_dedupe: {enabled: false}
output:
  proactive_status: {enabled: false}
  clean_answers: false
prompt_features:
  replay_intent: false
""",
    ),
    "profiles/tau3_identity_control.yaml": ("profiles/tau3_eval.yaml", "voice/tau3_spelling.yaml", ""),
    "profiles/tau3_arm_i3_recovery_hint.yaml": (
        "profiles/tau3_identity_control.yaml",
        "voice/tau3_recovery_hint.yaml",
        "",
    ),
    "profiles/tau3_arm_m3_spelling.yaml": (
        "profiles/tau3_eval_baseline.yaml",
        "voice/tau3_spelling.yaml",
        """\
delegation:
  spelling_hold:
    enabled: true
    hold_ms: 1500
    complete_patterns: ['^[A-Za-z0-9]{6}$']
""",
    ),
    # Profiles: the realtime_eval controls and their variants.
    "profiles/realtime_eval_124acf7.yaml": (
        "profiles/realtime_eval.yaml",
        "voice/realtime_eval_124acf7.yaml",
        _GUARDS_OFF,
    ),
    "profiles/realtime_eval_control.yaml": (
        "profiles/realtime_eval.yaml",
        "voice/realtime_eval_control.yaml",
        _GUARDS_OFF,
    ),
    "profiles/realtime_eval_identity.yaml": (
        "profiles/realtime_eval_124acf7.yaml",
        "voice/realtime_eval_identity.yaml",
        "",
    ),
    "profiles/realtime_eval_arm_a1_format_retry.yaml": (
        "profiles/realtime_eval_control.yaml",
        "voice/realtime_eval_a1_format_retry.yaml",
        "",
    ),
    "profiles/realtime_eval_arm_a3_field_provenance.yaml": (
        "profiles/realtime_eval_control.yaml",
        "voice/realtime_eval_a3_field_provenance.yaml",
        "",
    ),
}

_DIR = Path(tempfile.mkdtemp(prefix="fdh-variants-"))
atexit.register(shutil.rmtree, _DIR, ignore_errors=True)


def _path(name: str) -> Path:
    return variant(name) if name in VARIANTS else CONFIG / name


@cache
def variant(name: str) -> Path:
    """The variant ``name`` as a config file (absolute path), written once per test process."""
    base, voice, settings = VARIANTS[name]
    text = f"extends: {_path(base)}\n"
    if voice is not None:
        text += f"voice_profile: {_path(voice)}\n"
    path = _DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + textwrap.dedent(settings))
    return path
