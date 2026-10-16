# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, E501

"""The files that decide FBA's eval (OpenAI Realtime/Tau) behavior are frozen at ``84affa7``.

Plan ``misc/frontend-backend-verdict/fba-removal-plan.md``, section 2.2.1: all of ``core/``, the
Pipecat adapter files no FBA-removal change edits, the test fixtures and the shared test helpers.
A digest may change only with the user's written approval in the PR description.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FBA = ROOT / "src" / "examples" / "frontend_backend_agent"

#: SHA-256 of each frozen file, recorded at commit 84affa7. Re-recorded for the rename from
#: Frontend/Backend Verdict (``frontend_backend_verdict``) to Frontend/Backend Agent, name-only
#: (approved by the user, 2026-10-16).
FROZEN = {
    "src/examples/frontend_backend_agent/adapters/pipecat/client_events.py": "0874deaafc994960858d3950e1b202326be2fddf74ad89cbe1bc954c734e831a",
    "src/examples/frontend_backend_agent/adapters/pipecat/frames.py": "78b3459f24396918ed9529743681ceb7a697f5fb277dd9d79e68a987754dbaf0",
    "src/examples/frontend_backend_agent/adapters/pipecat/serializer.py": "4bf3b7b9f6bdd48209ac04035bdb33d910e92bf41a73f77454fd8fc7592c40bc",
    "src/examples/frontend_backend_agent/adapters/pipecat/services.py": "88802b89f3e3956d3ac5cf7167ebdca4a4d5c6ce057342eabb040ce3e178c4a0",
    "src/examples/frontend_backend_agent/adapters/pipecat/session_log.py": "7f67a0c1021f70083b97aceebc78f164796fd97aecfb856a9b29e66903763223",
    "src/examples/frontend_backend_agent/adapters/pipecat/settings.py": "9c2d590764bdb0d1a21d3b663ac3e90d3cac58c27eb1da87a85c4bbe05f25421",
    "src/examples/frontend_backend_agent/adapters/pipecat/settings/eval.yaml": "664410566684f4f4885a415f56c575376af57ce21fb661e6a2090c12c405fe25",
    "src/examples/frontend_backend_agent/adapters/pipecat/speech_output.py": "f34eec1d223321fba440aaaff0784f2e0d12f6c2c8667580ad121048d6cdae60",
    "src/examples/frontend_backend_agent/adapters/pipecat/timing.py": "5330136d25a470bb4e5d793d8df55a3cbd09b859132cc169ce37d13bf3f25d58",
    "src/examples/frontend_backend_agent/adapters/pipecat/turn_bridge.py": "3c06bb0f37de0989e72a5a2d37249c2a2caa063c4b49b414e0106304a2eca008",
    "src/examples/frontend_backend_agent/adapters/pipecat/warmup.py": "9034f16f7a228d8cd2150309a2fc9dc8ec59e27f78eee153bce8bb17b1c0c49a",
    "src/examples/frontend_backend_agent/core/__init__.py": "0da6107ab5a843e29fcab7de0232c412b0893b845494074247d80803bfb25508",
    "src/examples/frontend_backend_agent/core/agent/__init__.py": "3deba38492691ee589be2f05f9960ec48054f38d1d229fd4b8ba8ceda5726ac2",
    "src/examples/frontend_backend_agent/core/agent/agent.py": "ccb58a5be167ef578e39b44e6725e38748f0f66dedac1ef5b89b2e60396e9538",
    "src/examples/frontend_backend_agent/core/agent/errors.py": "95d085a9e63112a9a67c609602b1c0fd1175833e311f02ef874f1d61da371a2d",
    "src/examples/frontend_backend_agent/core/agent/events.py": "6794ec85a517abf98a160cc159bca36ba2687d3fff9680fd642a530236e77a8f",
    "src/examples/frontend_backend_agent/core/agent/factory.py": "6f653e584e228105c3520b21d773528693db4821233341b1118a6b1df3cda0b6",
    "src/examples/frontend_backend_agent/core/agent/instructions.py": "a49dee62f7e72014d70838673ddc2e1c9eafe3d36589dc810ae600059360fc94",
    "src/examples/frontend_backend_agent/core/agent/llm.py": "19ed784f082f00e6fef9f7f197d1a0181a43c1802c14b7ccf9d568b613cceca4",
    "src/examples/frontend_backend_agent/core/agent/messages.py": "952f02ad2e723e61d3a7d5673ceda516105d895a7f267084e4265dce53e3bef7",
    "src/examples/frontend_backend_agent/core/agent/prompts.py": "bf98ff2d9e076369b72ceeffb57ce98d9767cc9432263d7d01ee0d3735a4455c",
    "src/examples/frontend_backend_agent/core/agent/session.py": "17fd59cfecf08c6057b3eb6b01433229ef1cd971398395bdc70ff6a76f716037",
    "src/examples/frontend_backend_agent/core/audio/__init__.py": "d93a864646578790ac6bb3377d26567fa5aa9658f8c1e4075bade3d2b06e4376",
    "src/examples/frontend_backend_agent/core/audio/formats.py": "1eb9a0fb935f4a63aa802e271376f3d5f97f118b241ddfc321bac36ace753232",
    "src/examples/frontend_backend_agent/core/audio/g711.py": "44e745e6777483f744bc716086a3951aa614dfbd511202ef90fceb1cbdd2b449",
    "src/examples/frontend_backend_agent/core/audio/pcm.py": "5c6f0128e1f8813f690b41dd0adce695d6b18e6a35533cddcd7ee74ca43893ea",
    "src/examples/frontend_backend_agent/core/audio/resample.py": "23ea19b35913cb74da3f44c85ba385304eb37eb9fb73dcc8f9c1f15fb5cd4ed3",
    "src/examples/frontend_backend_agent/core/backend/__init__.py": "2f3289db7b01b22d90b1a5adf85298c27cce1c15349d8dde107497f6069d0cd0",
    "src/examples/frontend_backend_agent/core/backend/backend.py": "63ac9d669bb7a14647770044fee4a84d2266567f96c7b03529d1f6d3b2baa018",
    "src/examples/frontend_backend_agent/core/backend/context.py": "b3d70dbeecb86ac5d69c2979f431c295d8079bf6523218b33967cb3187e9a9fa",
    "src/examples/frontend_backend_agent/core/backend/history.py": "3308dc347906511d077ce3e5ff5c98d163d92abe41acf8da2d234f2c24557147",
    "src/examples/frontend_backend_agent/core/backend/tool_protocol.py": "513755f3a2de71189b1ab62ae67de82aaa1afd6b3bb862c8dc1d1d801c6ecd83",
    "src/examples/frontend_backend_agent/core/clock.py": "3e33c0d9a65fe3d9e182293185f2ab2624e8042fe7341adda05607678f0779c8",
    "src/examples/frontend_backend_agent/core/config/__init__.py": "656c42905ea88a61ad0411bd0713ffdac15d4e298f9aa418388d689c208ef281",
    "src/examples/frontend_backend_agent/core/config/agent.py": "cc65a33863b27563301723fa99d3d6e93c2b121e6b1a973f95babd9f521654a2",
    "src/examples/frontend_backend_agent/core/config/agent.yaml": "455391aaef2c299b1a8aadf423327540f47e5ebe0e8eb2905761f266b47024ca",
    "src/examples/frontend_backend_agent/core/config/catalog.py": "57dfed8031e992086dbcdb9481c28aad86dc53de2d304963f17481e6568f5ff1",
    "src/examples/frontend_backend_agent/core/config/profiles.py": "78d66e10ebef505f9ca20e260fc2f21ea57d85d89a34b984de42ea9ed08cb44c",
    "src/examples/frontend_backend_agent/core/config/profiles/default.yaml": "d2d0504ae158965918a253f02575e4014afb83c45b6d81e1474abc8c65a7ff4d",
    "src/examples/frontend_backend_agent/core/config/profiles/eval.yaml": "791193325dfb35ced9bd26f4e803978570d082ee49297f6332256f14e95ae8d2",
    "src/examples/frontend_backend_agent/core/config/prompts/generic.yaml": "a29ee978525777ad9811935c0f3efa1784d948889676ee13642745d4023a4a0b",
    "src/examples/frontend_backend_agent/core/config/prompts/tau.yaml": "a29ee978525777ad9811935c0f3efa1784d948889676ee13642745d4023a4a0b",
    "src/examples/frontend_backend_agent/core/config/voice.py": "2a485b965e27f7c317db0dedf8f65b1753477df55a14aba341e67dd7716957d0",
    "src/examples/frontend_backend_agent/core/config/voice_agent.yaml": "b9030af98edfac1cc7de755b73dfea34fbe0a5525ce0ddd5309f72bb39aa716e",
    "src/examples/frontend_backend_agent/core/engine/__init__.py": "ae8952a114b299755380ebe3be0a8bf8e9d9178226b429818672a6bf933a7af2",
    "src/examples/frontend_backend_agent/core/engine/event_log.py": "c50c965a350b1276e48ece4af6e8fedc0a0187e8edf8200eea7e5b7b245feac1",
    "src/examples/frontend_backend_agent/core/engine/filler.py": "b04670f0b25ed31aaf132024b58e69dcd664c6fc8ccad58bedc8b39f6b191b33",
    "src/examples/frontend_backend_agent/core/engine/history_repair.py": "1246bb617a6cfcb7bd80d37eaee849de043712b644cfdc358b96848971226f4e",
    "src/examples/frontend_backend_agent/core/engine/interfaces.py": "4706afb9add150944f15ea961f4f71b8d8374260845f4f5a3fdc50637ba7a833",
    "src/examples/frontend_backend_agent/core/engine/playback.py": "93867b54f005e1bea0ca6a9996ed3eb21223b198ed09d64c232a40f71d1078c4",
    "src/examples/frontend_backend_agent/core/engine/port.py": "c80776010935d13b50c6ca645ae18e7618a95bb13da33c92cd42d87edef36df8",
    "src/examples/frontend_backend_agent/core/engine/response_writer.py": "df7c0d430d2acd4659f51ba2bd0e3fe5fb198de6fb7e0309cf531f3b06c8ec74",
    "src/examples/frontend_backend_agent/core/engine/runner.py": "266f296d001f0b6e16393ae094f9c0315b77b8b9aa19646327cfc6aa8e5c97ab",
    "src/examples/frontend_backend_agent/core/engine/scripted.py": "d8873b10e3ed7f9f586cc269448f6fb850265676dab05d062c3edccc2faa1dae",
    "src/examples/frontend_backend_agent/core/engine/server_events.py": "9c70cafdc69f5657dae7ff2f7d28aa059bfa5493927c4852ae0fea1555611101",
    "src/examples/frontend_backend_agent/core/engine/session_settings.py": "ead540e9697d132f3ed9bab55138d718dcee5ff3f183a07625b6fb6762739d8a",
    "src/examples/frontend_backend_agent/core/engine/turn_api.py": "81f824831685fc36dbad10d13b450ccf1fea527ea6e463acb5a5cb5b4bc3e105",
    "src/examples/frontend_backend_agent/core/engine/turn_manager.py": "23c152aa4dd942d5b9d2204c89e9bed9497e7b52b35b45630a723e966f1d5fc7",
    "src/examples/frontend_backend_agent/core/engine/write_gate.py": "11905a98047fe24364ae7ea9194a69887d27bffa9e06ed6021b4cc4819f97f43",
    "src/examples/frontend_backend_agent/core/errors.py": "ad390a490eb734922266f0dd62c3232f74e10170aa607ff0d3268edd47002922",
    "src/examples/frontend_backend_agent/core/frontend/__init__.py": "ec2bd03aefff961e86ac9e6ae194ab3050a98d1e20329a54dbf8bdba1cc64cdf",
    "src/examples/frontend_backend_agent/core/frontend/delegation.py": "69555bd6f75dee8980a64f7d9a95c936a573a6ac395a4b03e046cf0185fa0991",
    "src/examples/frontend_backend_agent/core/frontend/frontend.py": "7100b3a197eab30d29c27fc7a635ec0242af938c06931b01507345d3e1cb6344",
    "src/examples/frontend_backend_agent/core/normalization/__init__.py": "b0031d8d7d874a644b67a70cb60bf563e9d682cb51d773f44526320122f16cff",
    "src/examples/frontend_backend_agent/core/normalization/arguments.py": "c5686dee5de601c66f33c63c64e9070e29df73a07081065376c895e4f1685546",
    "src/examples/frontend_backend_agent/core/normalization/prompts.py": "8ea27d7f1b26b0b56b25418248f08260e5a5ba3c96673d849aed4e07bed7508e",
    "src/examples/frontend_backend_agent/core/normalization/rules.py": "1ecb87663a157101ead95441a78345db60811828129d71b3984ceb1926356279",
    "src/examples/frontend_backend_agent/core/normalization/schema_rules.py": "e9873db96aa00a5a9b512c4b449332d018535305f41c259f1dc9c5e5e934f44b",
    "src/examples/frontend_backend_agent/core/normalization/transcript.py": "8648dc429e9cb2f757afc3349c9d3ae1ff50955dbcedd680e2b577978065b3ec",
    "src/examples/frontend_backend_agent/core/tools/__init__.py": "5b23f1d24b916fb610fac5e3e3e516a511f20cc552698619b041825c6b7ce95e",
    "src/examples/frontend_backend_agent/core/tools/realtime_tools.py": "df388e7b0e75f70469be99d123eacbc3e176e7fd363ebdba613a5be220190af7",
    "src/examples/frontend_backend_agent/core/tools/tools.py": "1ccaf6797a448b93e8dacb9ea271569c2d96e6383f84bbe5317a6443e98f93b7",
    "tests/unit/frontend_backend_agent/_fba_core_fakes.py": "fabe86d646d1b1a8d6c214b0fce740145015d9e132b889439d7829b00842dbb4",
    "tests/unit/frontend_backend_agent/_fba_demo_tools.py": "32e6492c69b435a372d51e25d89d1163f7da1e844b92e5abc06694770fba3b34",
    "tests/unit/frontend_backend_agent/_fba_fakes.py": "bd94db1d2a9df9dcbc853b1b12b48e52f6e435fdcf159eeaac6548adff2d8d3c",
    "tests/unit/frontend_backend_agent/_fba_golden.py": "8ae0c496385616924782ec772c1136680778db6402522ef1e0f2874d75fb3dd5",
    "tests/unit/frontend_backend_agent/_fba_normalization_replay.py": "c5af1ab70aa4e96c42f53e1ea245b1389151d1b28d96bcc19b0b030d72981a43",
    "tests/unit/frontend_backend_agent/_fba_parity.py": "fe916007c1716371f0899492ad294e26093583ae7960a7f9177c1d85e16f87f3",
    "tests/unit/frontend_backend_agent/_fba_pipecat_fakes.py": "82aaf5f832edb43c440376ee3df9cdd28ded8e607f7fd3f242c00676051d230e",
    "tests/unit/frontend_backend_agent/fixtures/profiles/backend_only.yaml": "4efcc506fc010a10c4ad932053f80fc3b0b4755d7da956a6471de4d7fa570324",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend+frontend__backend.json": "b47e2ef64da328452d15ebca860e9fb4d728cf80d662209a1c5d1e8cbc3b1de7",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend+frontend__frontend.json": "1b7469977a619e805151bafe591884c98084f04ac5e1bd1a0365fb85e0305eea",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend__backend.json": "b47e2ef64da328452d15ebca860e9fb4d728cf80d662209a1c5d1e8cbc3b1de7",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend__frontend.json": "386a9f7f7a31397a1dcbf5b4917336de3e5fe05c033230c2a14567b9a86d4920",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend+frontend__backend.json": "40c96c6e0272797c990ca0bd0d52ec42bc1ef9ca4b0d325b5eb576035e82437e",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend+frontend__frontend.json": "de3f3e19ca5c835f1c8875a22878d7388d22d32783e2e2621d0189524380e054",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend__backend.json": "40c96c6e0272797c990ca0bd0d52ec42bc1ef9ca4b0d325b5eb576035e82437e",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend__frontend.json": "5d2c352624974b8660c6a56b16f10bb3c3f8a68172aa5d5e429d6bdb8e4ed803",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend+frontend__backend.json": "36ee73abc56ddccfada236a0b595ad4e46d1857d837a7c88615ef37bcb892977",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend+frontend__frontend.json": "0e4aa5864a806b31a9699bd09ef71a3a1ce6f6a57121984351be6d2b58a00eaf",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend__backend.json": "36ee73abc56ddccfada236a0b595ad4e46d1857d837a7c88615ef37bcb892977",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend__frontend.json": "c27a0f71acf1fa5260cf335ff320125c1b941e592c9118583dc4a33fc7dca8d8",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend+frontend__backend.json": "c3fc0f16b326717c9eeea7a7197846f130c31e3eb30ca0ad60d709ecca68dd8d",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend+frontend__frontend.json": "43a87d7648470938e2f30a1741873857d77bbd0c0c2593a38054ba29e5c19646",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend__backend.json": "c3fc0f16b326717c9eeea7a7197846f130c31e3eb30ca0ad60d709ecca68dd8d",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend__frontend.json": "1f91e1f02a78e1fa5f3871bbd13688409d8ca81c1003ef33a126861ef4d4a90b",
    "tests/unit/frontend_backend_agent/fixtures/tau2_session_update/PROVENANCE.json": "19b4dba5b1d9c8abbb1091871a5a5925dee7ace00b79773b01466b2950d25a3d",
    "tests/unit/frontend_backend_agent/fixtures/tau2_session_update/airline.json": "596cfb21cc7654df018d700d2c29cb06334c111988c5eeddbb227b0a13a28325",
    "tests/unit/frontend_backend_agent/fixtures/tau2_session_update/banking_knowledge.json": "9a6c000f72f56d1559dba08d137a9cf2df4b2f8e9be45277a8f8c7296daac772",
    "tests/unit/frontend_backend_agent/fixtures/tau2_session_update/mock.json": "6a767150f5888881f0d2b59d8eb29e9e27a2d1e61db3cee7cd4a635c49e85ce0",
    "tests/unit/frontend_backend_agent/fixtures/tau2_session_update/retail.json": "920185012c9dd2fda8643ed142a3ccf76bf8027a5d974666e3ee50c0f6a5ea83",
    "tests/unit/frontend_backend_agent/fixtures/tau2_session_update/telecom.json": "44c92e3a050d01c8f744fb43565227dcaec00e9b99b1678dbb6e321f8aff5e49",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/configure_and_prompts.json": "2b433cc4b16e31c6ceefe87eeab9d23f9ccc75e71073c7fe16bcd4b822cf801a",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/merged_profile_configs.json": "527c67578cb19c3907a1c60640fc7e2440265ecbcb686935941b58092b94c287",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/pcm_decode.json": "2e1e904bc13bb66092cc92956173636dfe62fa418a1ea31490110895264c880b",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/pipecat_host.json": "d878d077ca839e5d687e19f32f164f8842274ed81cd257e9a473e992c82d4c8a",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_agent_failure.json": "b39a2af7925007a2f2f197c8e100dd1d299ba9f0c0a8cb38749392f3edeb11e9",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_barge_in_after_answer.json": "b9c9bde48c82bb3356c99d9d559c87da07cc5b68c6e55150f89d510040e6abd5",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_barge_in_during_answer.json": "3354bf491a8a79e46a458fa6e6beb250dfa549bccde46c8c56cb53af1b71a948",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_barge_in_filler.json": "d4caef68a240f04517f4ca02428d620b9d9a4e2e5f5335615b8cfd18e8e85b0b",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_filler_then_answer.json": "28b4a663d1394bab9a19ceb115c899bb09c51f9f4e64b87c886f159040918032",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_parallel_tools.json": "c52ecc592ecd0b3c9055b2b35d652c810130b4bd7b47923fbdd9047eaf100175",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_response_cancel.json": "d4ce40618ea9ccce784df94dfe0b037250fcdb2ebaa136ef3abcff8f486764d1",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_review_continue.json": "d832c5b8591a25dbe0145fb608f7ccb7d6d258633523ea7dc70f418f948509e8",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_review_new.json": "8a3ff2c63e694fc9b02f109eaa006048ef1947fa64e2940b031cb230930ac0f4",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_tool_round.json": "bded4bd32776b37f4414178be2b6e792fd875f17e75ec3ea983acbb65aa1f2bd",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_truncate_known_and_unknown.json": "36a079bd0bc6d83d191a1db0c5648154986f4dd45bf4f6de8dfd59d081664e2d",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_tts_failure.json": "64858e70ee7de38c699391571c71620dd6fd07e42c8f6c446f49ebbc72279b35",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_voice_turn.json": "5205646c623950f9470c42d9d028b6acf16c7f3e00d3682154d358b3d4422eee",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_write_gate_cut.json": "62356c76abee05833edc04b8ca72b1f1d2cdbe1cd8bb43b997761e451cbcfcd8",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/scenario_write_gate_heard.json": "438f57bf6e13562a17fcfa780acf05e00d51d31a540d84f2fa7a693fd148ba5d",
    "tests/unit/frontend_backend_agent/fixtures/verdict_cases.jsonl": "37de0a917b05fd34166de475200567a25aa6544e6d5a98a73dae9dfe23704d09",
}

#: Directories whose file list is frozen too: a new file there could change eval behavior.
FROZEN_TREES = (
    "src/examples/frontend_backend_agent/core",
    "tests/unit/frontend_backend_agent/fixtures",
)


def _files(tree: str) -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / tree).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


class FrozenManifestTests(unittest.TestCase):
    def test_every_frozen_file_is_unchanged(self) -> None:
        changed = [
            path for path, digest in FROZEN.items() if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest
        ]
        self.assertEqual(changed, [], "frozen files changed (plan section 2.2.1); this needs the user's approval")

    def test_no_file_was_added_or_removed_in_a_frozen_tree(self) -> None:
        for tree in FROZEN_TREES:
            with self.subTest(tree=tree):
                expected = {path for path in FROZEN if path.startswith(tree + "/")}
                self.assertEqual(_files(tree), expected)


if __name__ == "__main__":
    unittest.main()
