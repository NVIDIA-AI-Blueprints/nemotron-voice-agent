# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, E501

"""The files that decide FBA's eval (OpenAI Realtime/benchmark) behavior are frozen at ``84affa7``.

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
#: (approved by the user, 2026-10-16). Re-recorded on 2026-10-21 for the customer-service prompt
#: rules and domain notes (``domain.name``, ``instructions.domain_from_tools``), approved by the user
#: in the description of PR #171. Re-recorded on 2026-10-26 for the name-only removal of benchmark
#: names from the test helpers and fixtures (comments, helper names, the fixture directory and a v1
#: golden label) and from the comments, docstrings and one log message of the FBA sources, approved
#: by the user in the description of PR #171.
FROZEN = {
    "src/examples/frontend_backend_agent/adapters/pipecat/client_events.py": "0874deaafc994960858d3950e1b202326be2fddf74ad89cbe1bc954c734e831a",
    "src/examples/frontend_backend_agent/adapters/pipecat/frames.py": "78b3459f24396918ed9529743681ceb7a697f5fb277dd9d79e68a987754dbaf0",
    "src/examples/frontend_backend_agent/adapters/pipecat/serializer.py": "4bf3b7b9f6bdd48209ac04035bdb33d910e92bf41a73f77454fd8fc7592c40bc",
    "src/examples/frontend_backend_agent/adapters/pipecat/services.py": "88802b89f3e3956d3ac5cf7167ebdca4a4d5c6ce057342eabb040ce3e178c4a0",
    "src/examples/frontend_backend_agent/adapters/pipecat/session_log.py": "7f67a0c1021f70083b97aceebc78f164796fd97aecfb856a9b29e66903763223",
    "src/examples/frontend_backend_agent/adapters/pipecat/settings.py": "c78a543d80204ca05d217f081365b49baa6c010fefefc3f7bbbd739bbf4618d7",
    "src/examples/frontend_backend_agent/adapters/pipecat/settings/eval.yaml": "c15ce8c5a288cdc7584b5c9deb1ee0eaf3c155f1872b139691f050486513da76",
    "src/examples/frontend_backend_agent/adapters/pipecat/speech_output.py": "f34eec1d223321fba440aaaff0784f2e0d12f6c2c8667580ad121048d6cdae60",
    "src/examples/frontend_backend_agent/adapters/pipecat/timing.py": "14ccb453997701fcf9b2fa79d95bad56308783014e98f54f9c9545fe2a8b66d8",
    "src/examples/frontend_backend_agent/adapters/pipecat/turn_bridge.py": "3c06bb0f37de0989e72a5a2d37249c2a2caa063c4b49b414e0106304a2eca008",
    "src/examples/frontend_backend_agent/adapters/pipecat/warmup.py": "9034f16f7a228d8cd2150309a2fc9dc8ec59e27f78eee153bce8bb17b1c0c49a",
    "src/examples/frontend_backend_agent/core/__init__.py": "0da6107ab5a843e29fcab7de0232c412b0893b845494074247d80803bfb25508",
    "src/examples/frontend_backend_agent/core/agent/__init__.py": "3deba38492691ee589be2f05f9960ec48054f38d1d229fd4b8ba8ceda5726ac2",
    "src/examples/frontend_backend_agent/core/agent/agent.py": "8e9a99429956677491fe157aa6f39ceb865551530aa36bd12bb0ec14fbe2a1bf",
    "src/examples/frontend_backend_agent/core/agent/errors.py": "95d085a9e63112a9a67c609602b1c0fd1175833e311f02ef874f1d61da371a2d",
    "src/examples/frontend_backend_agent/core/agent/events.py": "6794ec85a517abf98a160cc159bca36ba2687d3fff9680fd642a530236e77a8f",
    "src/examples/frontend_backend_agent/core/agent/factory.py": "6f653e584e228105c3520b21d773528693db4821233341b1118a6b1df3cda0b6",
    "src/examples/frontend_backend_agent/core/agent/instructions.py": "6b36116b9aad77311cc5a469b2a442292be1211e7a1c6c537f4d11bdafc5b4df",
    "src/examples/frontend_backend_agent/core/agent/llm.py": "baaf2cbef6d5cb59cc3b834bdc5f0e5d95751a3a5b7ee3b8686135ea68c32924",
    "src/examples/frontend_backend_agent/core/agent/messages.py": "f7588d92aa5f7268022fe127d81c4214b9b3e6fcb6cdc81e23802b16762ccf48",
    "src/examples/frontend_backend_agent/core/agent/prompts.py": "1701373cd57bcfb679e78505fb4d67240cdf85e87adc9294e7abd4da5bc8a482",
    "src/examples/frontend_backend_agent/core/agent/session.py": "380ea344beea80dfa41e1d5912b27993440537571a411e6ace7c826bf1ff2073",
    "src/examples/frontend_backend_agent/core/audio/__init__.py": "d93a864646578790ac6bb3377d26567fa5aa9658f8c1e4075bade3d2b06e4376",
    "src/examples/frontend_backend_agent/core/audio/formats.py": "1eb9a0fb935f4a63aa802e271376f3d5f97f118b241ddfc321bac36ace753232",
    "src/examples/frontend_backend_agent/core/audio/g711.py": "bf14d4861712a54e8ef5c4010255d5512b4084f6fef508f388d2c6243ef08a88",
    "src/examples/frontend_backend_agent/core/audio/pcm.py": "5c6f0128e1f8813f690b41dd0adce695d6b18e6a35533cddcd7ee74ca43893ea",
    "src/examples/frontend_backend_agent/core/audio/resample.py": "3ac3f57b1c1ca567aabf2ff3dd5866f356c4c6b478a7a2922631acfd07cc5453",
    "src/examples/frontend_backend_agent/core/backend/__init__.py": "2f3289db7b01b22d90b1a5adf85298c27cce1c15349d8dde107497f6069d0cd0",
    "src/examples/frontend_backend_agent/core/backend/backend.py": "63ac9d669bb7a14647770044fee4a84d2266567f96c7b03529d1f6d3b2baa018",
    "src/examples/frontend_backend_agent/core/backend/context.py": "b3d70dbeecb86ac5d69c2979f431c295d8079bf6523218b33967cb3187e9a9fa",
    "src/examples/frontend_backend_agent/core/backend/history.py": "3308dc347906511d077ce3e5ff5c98d163d92abe41acf8da2d234f2c24557147",
    "src/examples/frontend_backend_agent/core/backend/tool_protocol.py": "4237a24ea151f1e726dce558ce1b3e6748881418a2863f31282fe7c4962fd4d1",
    "src/examples/frontend_backend_agent/core/clock.py": "33f6403cafa6c8bc4bdab45b293515ff6b4bee76653d579915887dc88cc38448",
    "src/examples/frontend_backend_agent/core/config/__init__.py": "656c42905ea88a61ad0411bd0713ffdac15d4e298f9aa418388d689c208ef281",
    "src/examples/frontend_backend_agent/core/config/agent.py": "e3755bd573bd6d42a456a9feae5d5a1b9d28bc9ff9fb811b8aaba14e46433e02",
    "src/examples/frontend_backend_agent/core/config/agent.yaml": "383d28926f834eb4a23a96203415b249ec367b5031def4bde6619c62e8c71d7a",
    "src/examples/frontend_backend_agent/core/config/catalog.py": "57dfed8031e992086dbcdb9481c28aad86dc53de2d304963f17481e6568f5ff1",
    "src/examples/frontend_backend_agent/core/config/profiles.py": "65adfd6f8b8db54e9366d00ce35d39ec1293ddca921bcca897f844b1873557c5",
    "src/examples/frontend_backend_agent/core/config/profiles/default.yaml": "37c22b91f9b8f88069fa7f72479eeaa70bdcc0b938e2be75806db4c47d1ad84b",
    "src/examples/frontend_backend_agent/core/config/profiles/eval.yaml": "e9a5b848b61735b7cf0d1de4fd9a4eaadb3ea5530788b8b34b139a1c79708044",
    "src/examples/frontend_backend_agent/core/config/prompts/general_qa.yaml": "87c20d71056e3713a67a37d6ea54148201f4469de1a5cf43ec6c3578fdb9de20",
    "src/examples/frontend_backend_agent/core/config/prompts/generic.yaml": "8dfaed453c0d4091e74447c9e2790de55d2a9fc09caf08b425b9d6a609ab41c7",
    "src/examples/frontend_backend_agent/core/config/prompts/customer_service.yaml": "ad0481c985601b55f0eca37a6dc52f28fcd5c38d9892d93eb26529d8e94fd38c",
    "src/examples/frontend_backend_agent/core/config/voice.py": "fe8298935082a0afdd0bd1ba6ab1bb726127148d079a192c4cd8d7a1ab6a6a1a",
    "src/examples/frontend_backend_agent/core/config/voice_agent.yaml": "999d7e23f938f9c68a2740e3cc1f0ce09cc939ad940afda86cd0235a71839b55",
    "src/examples/frontend_backend_agent/core/engine/__init__.py": "ae8952a114b299755380ebe3be0a8bf8e9d9178226b429818672a6bf933a7af2",
    "src/examples/frontend_backend_agent/core/engine/event_log.py": "c50c965a350b1276e48ece4af6e8fedc0a0187e8edf8200eea7e5b7b245feac1",
    "src/examples/frontend_backend_agent/core/engine/filler.py": "b35fc8303adf0e961e25eac68576ea0e7fc89bda64473fe04ca802f278725ba0",
    "src/examples/frontend_backend_agent/core/engine/history_repair.py": "1246bb617a6cfcb7bd80d37eaee849de043712b644cfdc358b96848971226f4e",
    "src/examples/frontend_backend_agent/core/engine/interfaces.py": "4706afb9add150944f15ea961f4f71b8d8374260845f4f5a3fdc50637ba7a833",
    "src/examples/frontend_backend_agent/core/engine/playback.py": "e26a564352119358679180d17681f948eddcdf0941b23f58b75bdc7673d8ac30",
    "src/examples/frontend_backend_agent/core/engine/port.py": "a60a37e7aff3875471b0b26e1e0b70411265924e2b054ba7e22cc0acdf52c257",
    "src/examples/frontend_backend_agent/core/engine/response_writer.py": "df7c0d430d2acd4659f51ba2bd0e3fe5fb198de6fb7e0309cf531f3b06c8ec74",
    "src/examples/frontend_backend_agent/core/engine/runner.py": "b0979c73b7f594cc8cfcef20c33dcec170bd57a47aebfd5c4e3e0b40f3cc1162",
    "src/examples/frontend_backend_agent/core/engine/scripted.py": "a73eeb682c5a07faf91388280f0712099d881a8f084199b9595ea6af504a6f8a",
    "src/examples/frontend_backend_agent/core/engine/server_events.py": "36f05cc5219bca5f84944d90f049b2bc058faffe0bf8c946fd1837b5aa7b0952",
    "src/examples/frontend_backend_agent/core/engine/session_settings.py": "ead540e9697d132f3ed9bab55138d718dcee5ff3f183a07625b6fb6762739d8a",
    "src/examples/frontend_backend_agent/core/engine/turn_api.py": "81f824831685fc36dbad10d13b450ccf1fea527ea6e463acb5a5cb5b4bc3e105",
    "src/examples/frontend_backend_agent/core/engine/turn_manager.py": "4b6c65ac0d901361b3c814c0f332243ffb53ac7c0dd1b2941ea33fa567ebecf3",
    "src/examples/frontend_backend_agent/core/engine/write_gate.py": "11905a98047fe24364ae7ea9194a69887d27bffa9e06ed6021b4cc4819f97f43",
    "src/examples/frontend_backend_agent/core/errors.py": "ad390a490eb734922266f0dd62c3232f74e10170aa607ff0d3268edd47002922",
    "src/examples/frontend_backend_agent/core/frontend/__init__.py": "ec2bd03aefff961e86ac9e6ae194ab3050a98d1e20329a54dbf8bdba1cc64cdf",
    "src/examples/frontend_backend_agent/core/frontend/delegation.py": "69555bd6f75dee8980a64f7d9a95c936a573a6ac395a4b03e046cf0185fa0991",
    "src/examples/frontend_backend_agent/core/frontend/frontend.py": "7100b3a197eab30d29c27fc7a635ec0242af938c06931b01507345d3e1cb6344",
    "src/examples/frontend_backend_agent/core/normalization/__init__.py": "b0031d8d7d874a644b67a70cb60bf563e9d682cb51d773f44526320122f16cff",
    "src/examples/frontend_backend_agent/core/normalization/arguments.py": "c5686dee5de601c66f33c63c64e9070e29df73a07081065376c895e4f1685546",
    "src/examples/frontend_backend_agent/core/normalization/prompts.py": "8ea27d7f1b26b0b56b25418248f08260e5a5ba3c96673d849aed4e07bed7508e",
    "src/examples/frontend_backend_agent/core/normalization/rules.py": "1ecb87663a157101ead95441a78345db60811828129d71b3984ceb1926356279",
    "src/examples/frontend_backend_agent/core/normalization/schema_rules.py": "7357b5174e216ff6c890f6239802d7b7850c34be9dff599e9491df6d15b770d1",
    "src/examples/frontend_backend_agent/core/normalization/transcript.py": "8648dc429e9cb2f757afc3349c9d3ae1ff50955dbcedd680e2b577978065b3ec",
    "src/examples/frontend_backend_agent/core/tools/__init__.py": "5b23f1d24b916fb610fac5e3e3e516a511f20cc552698619b041825c6b7ce95e",
    "src/examples/frontend_backend_agent/core/tools/realtime_tools.py": "f7c15553577d0161a4eb00d861a261995dd78b2993ba7ef2d3e308ecf474a1d2",
    "src/examples/frontend_backend_agent/core/tools/tools.py": "44a9a5fe6102d5de121c04fb126e6341aa3d98c3dc42a4e0c2baaeb1d552d1de",
    "tests/unit/frontend_backend_agent/_fba_core_fakes.py": "91a0db0e0330332288e1b818d5a88a3f82bc785cea21bdc89b232ebc6b44b9be",
    "tests/unit/frontend_backend_agent/_fba_demo_tools.py": "32e6492c69b435a372d51e25d89d1163f7da1e844b92e5abc06694770fba3b34",
    "tests/unit/frontend_backend_agent/_fba_fakes.py": "bd94db1d2a9df9dcbc853b1b12b48e52f6e435fdcf159eeaac6548adff2d8d3c",
    "tests/unit/frontend_backend_agent/_fba_golden.py": "801e8e78ca401d9a44cf3bc42f05b945fe5084256c11e831a368b0acc470a609",
    "tests/unit/frontend_backend_agent/_fba_normalization_replay.py": "c5af1ab70aa4e96c42f53e1ea245b1389151d1b28d96bcc19b0b030d72981a43",
    "tests/unit/frontend_backend_agent/_fba_parity.py": "b9ceb5d4444b1f4c7811fd2034f32962dc5736ccebaf69fec310b34b496202df",
    "tests/unit/frontend_backend_agent/_fba_pipecat_fakes.py": "3aca88266722ed56af49fbbc9ec812989de25ad3ba3f419b8a6a3498771a82a8",
    "tests/unit/frontend_backend_agent/fixtures/profiles/backend_only.yaml": "8b4ccdc5f1e76b0f90debbea3547eb427a06727294cab67be5808b7a30076bcd",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend+frontend__backend.json": "468b899598a49f41c911d51af0f05e770d8e3879d612a5c52180c537b77c5120",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend+frontend__frontend.json": "578eb3deaa8f01cc5e7560ccf1cc59ce6facf1524daa5f11e631ac3fcfe8c703",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend__backend.json": "468b899598a49f41c911d51af0f05e770d8e3879d612a5c52180c537b77c5120",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/airline__backend__frontend.json": "122dade13a306e560b9767961d13e727656bd9f6fcbf586596f89b2bb8210113",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend+frontend__backend.json": "0982760a79a73cc1e83abd00769ada18ce1185456b2a81a97066632326ea27bc",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend+frontend__frontend.json": "83ae5e22d8a1970717ad5436970d9990d1e3b47892df5cafab7e689728e3ac4c",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend__backend.json": "0982760a79a73cc1e83abd00769ada18ce1185456b2a81a97066632326ea27bc",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/mock__backend__frontend.json": "f0a42a6b00ddff45ea2cb2844f45a22afb2165b665c420a1cc87b06dfa8db120",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend+frontend__backend.json": "6407fb69ca117cbfebb5476645b66e3cf8fdc840a3b926524066e37fe323763a",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend+frontend__frontend.json": "3030d607ff9b3d3c12ab0be656e1772439b5b84e3dcd509687ab8b1709915d56",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend__backend.json": "6407fb69ca117cbfebb5476645b66e3cf8fdc840a3b926524066e37fe323763a",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/retail__backend__frontend.json": "a38d3420914916bf81c8c67db4dc4c017d83da66dd5997ef24974b02b247bc8a",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend+frontend__backend.json": "0dfa77ae0c767de0ca32a506750f3e33952eb999fbb0e15d82c4269712b8f02b",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend+frontend__frontend.json": "679635063857d01b1e12c3c7bc833e0f8870535c25c049e4ebd6dbe522a47f81",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend__backend.json": "0dfa77ae0c767de0ca32a506750f3e33952eb999fbb0e15d82c4269712b8f02b",
    "tests/unit/frontend_backend_agent/fixtures/prompt_snapshots/telecom__backend__frontend.json": "fa3c0b44950ae33fef47242996afa713039bd5bb17300e12fa8cee4984719098",
    "tests/unit/frontend_backend_agent/fixtures/benchmark_session_update/PROVENANCE.json": "19b4dba5b1d9c8abbb1091871a5a5925dee7ace00b79773b01466b2950d25a3d",
    "tests/unit/frontend_backend_agent/fixtures/benchmark_session_update/airline.json": "596cfb21cc7654df018d700d2c29cb06334c111988c5eeddbb227b0a13a28325",
    "tests/unit/frontend_backend_agent/fixtures/benchmark_session_update/banking_knowledge.json": "9a6c000f72f56d1559dba08d137a9cf2df4b2f8e9be45277a8f8c7296daac772",
    "tests/unit/frontend_backend_agent/fixtures/benchmark_session_update/mock.json": "6a767150f5888881f0d2b59d8eb29e9e27a2d1e61db3cee7cd4a635c49e85ce0",
    "tests/unit/frontend_backend_agent/fixtures/benchmark_session_update/retail.json": "920185012c9dd2fda8643ed142a3ccf76bf8027a5d974666e3ee50c0f6a5ea83",
    "tests/unit/frontend_backend_agent/fixtures/benchmark_session_update/telecom.json": "44c92e3a050d01c8f744fb43565227dcaec00e9b99b1678dbb6e321f8aff5e49",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/configure_and_prompts.json": "452c30aed502411689ef81b118af6d4f83bd4c13063430066a4b157b72a56f3c",
    "tests/unit/frontend_backend_agent/fixtures/v1_golden/merged_profile_configs.json": "de325558ad8cebbdcecee7b24db6000047e97d2329753adad4913799e4e86847",
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
